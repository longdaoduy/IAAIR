"""
Retrieval Handler for unified search operations.

This module provides functionality to execute vector and graph searches,
with intelligent query routing and AI response generation.
"""

from typing import Dict, List, Optional
from models.entities.retrieval.QueryType import QueryType
import logging
import os
import re
import json
from clients.vector.MilvusClient import MilvusClient
from pipelines.retrievals.GraphQueryHandler import GraphQueryHandler
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.huggingface.DeepseekClient import DeepseekClient
from clients.huggingface.CLIPClient import CLIPClient
from pymilvus import (Collection)

logger = logging.getLogger(__name__)


class HybridRetrievalHandler:
    """Unified handler for vector and graph-based retrieval operations."""

    def __init__(self, vector_db: Optional[MilvusClient], graph_db: GraphQueryHandler,
                 ai_agent: Optional[DeepseekClient], embedder: SciBERTClient,
                 cache_manager=None, performance_monitor=None,
                 clip_client: Optional[CLIPClient] = None):
        self.milvus_client = vector_db
        self.embedding_client = embedder
        self.graph_handler = graph_db
        self.ai_agent = ai_agent
        self.cache_manager = cache_manager
        self.performance_monitor = performance_monitor
        self.clip_client = clip_client

    async def execute_vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Execute vector search with performance tracking."""
        if self.performance_monitor:
            with self.performance_monitor.track_operation('vector_search'):
                return await self._execute_vector_search_internal(query, top_k)
        else:
            return await self._execute_vector_search_internal(query, top_k)

    async def _execute_vector_search_internal(self, query: str, top_k: int) -> List[Dict]:
        """Internal vector search implementation."""
        try:
            # Check cache first
            if self.cache_manager:
                cached_results = self.cache_manager.get_search_results(
                    query, top_k, use_hybrid=True, routing_strategy="vector"
                )
                if cached_results is not None:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit('search', True)
                        self.performance_monitor.record_result_count('vector', len(cached_results))
                    logger.debug(f"Vector search cache hit for: {query[:50]}...")
                    return cached_results

            if self.performance_monitor:
                self.performance_monitor.record_cache_hit('search', False)

            results = self.search_similar_papers(
                query_text=query,
                top_k=top_k,
                use_hybrid=True
            )

            # Cache results
            if self.cache_manager and results:
                self.cache_manager.cache_search_results(
                    query, results, top_k, use_hybrid=True, routing_strategy="vector"
                )

            if self.performance_monitor:
                self.performance_monitor.record_result_count('vector', len(results or []))

            return results or []
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def search_similar_papers(self, query_text: str, top_k: int = 10, use_hybrid: bool = True) -> List[Dict]:
        """Search for similar papers using hybrid search (dense + sparse) or dense-only search.

        Args:
            query_text: Text query to search for similar papers
            top_k: Number of top results to return
            use_hybrid: Whether to use hybrid search or dense-only search

        Returns:
            List of similar papers with scores and IDs
        """
        try:
            if not self.milvus_client.collection:
                self.milvus_client.collection = Collection(self.milvus_client.config.collection_name)
                self.milvus_client.collection.load()

            # Check embedding cache first
            query_embedding = None
            if self.cache_manager:
                query_embedding = self.cache_manager.get_embedding(query_text)
                if query_embedding is not None and self.performance_monitor:
                    self.performance_monitor.record_cache_hit('embedding', True)

            # Generate embedding if not cached
            if query_embedding is None:
                if self.performance_monitor:
                    self.performance_monitor.record_cache_hit('embedding', False)
                    with self.performance_monitor.track_operation('embedding'):
                        query_embedding = self.embedding_client.generate_text_embedding(query_text)
                else:
                    query_embedding = self.embedding_client.generate_text_embedding(query_text)

                # Cache the embedding
                if self.cache_manager and query_embedding is not None:
                    self.cache_manager.cache_embedding(query_text, query_embedding)

            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return []

            # Execute search with optimized parameters
            return self.milvus_client._hybrid_search(query_text, query_embedding, top_k)

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    async def execute_graph_search(self, query: str, top_k: int) -> List[Dict]:
        """Execute graph search using Cypher query with intelligent query parsing and caching."""
        if self.performance_monitor:
            with self.performance_monitor.track_operation('graph_search'):
                return await self._execute_graph_search_internal(query, top_k)
        else:
            return await self._execute_graph_search_internal(query, top_k)

    async def execute_graph_search_with_template(self, query: str, template_cypher: str, top_k: int) -> List[Dict]:
        """Execute graph search using a user-selected Cypher template with AI-extracted conditions.

        The AI agent analyzes the user's natural language query and the template to produce
        a refined Cypher query with proper WHERE clauses, ORDER BY, and LIMIT injected.

        Args:
            query: The user's natural language search query
            template_cypher: The Cypher template selected by the user
            top_k: Maximum number of results to return
            paper_ids: Optional list of paper IDs to filter results (added as WHERE clause)

        Returns:
            List of matching records from Neo4j
        """
        if self.performance_monitor:
            with self.performance_monitor.track_operation('graph_search_template'):
                return await self._execute_graph_search_with_template_internal(query, template_cypher, top_k)
        else:
            return await self._execute_graph_search_with_template_internal(query, template_cypher, top_k)

    async def _execute_graph_search_with_template_internal(
            self, query: str, template_cypher: str, top_k: int) -> List[Dict]:
        """Internal implementation of template-based graph search.

        If the query is about a keyword or topic (detected automatically),
        applies vector-first strategy: run vector search to find semantically
        similar papers, then inject their IDs as a WHERE condition into
        whatever template was selected — scoping graph results to the most
        relevant papers.
        """
        await self._build_intelligent_cypher_query(query, top_k, template_cypher)

    def _refine_template_with_conditions(self, query: str, template_cypher: str, top_k: int,
                                         paper_ids: Optional[List[str]] = None) -> tuple:
        """Refine Cypher template by programmatically injecting conditions from the query.

        Extracts entities (paper IDs, author names, years, keywords) from the user's
        natural language query and injects appropriate WHERE clauses into the template.

        Args:
            query: The user's natural language search query
            template_cypher: The Cypher template
            top_k: Maximum results
            paper_ids: Optional paper IDs to scope results to specific papers

        Returns:
            (refined_cypher, parameters) tuple
        """
        conditions = []
        parameters = {}

        # 1. Paper IDs filter (from explicit input)
        if paper_ids:
            conditions.append('p.id IN $paper_ids')
            parameters['paper_ids'] = paper_ids

        # 2. Extract inline paper IDs from query text (e.g. "W12345")
        inline_ids = re.findall(r'\b(W\d+)\b', query)
        if inline_ids and not paper_ids:
            conditions.append('p.id IN $paper_ids')
            parameters['paper_ids'] = inline_ids

        # 3. Extract author names from query
        author_names = self._extract_author_names(query)
        if author_names:
            # Check if the template already filters by author name in its WHERE clause
            # (e.g. coauthor_network uses a1.name, search_by_author uses a.name)
            template_already_filters_author = bool(
                re.search(r'toLower\(\s*\w+\.name\s*\)\s*CONTAINS', template_cypher)
                and '$author_names' in template_cypher
            )

            if not template_already_filters_author:
                # Detect the Author variable alias used in this template
                # e.g. "a:Author", "a1:Author", "auth:Author"
                author_var_match = re.search(r'\((\w+):Author\)', template_cypher)
                if author_var_match:
                    author_var = author_var_match.group(1)
                    conditions.append(
                        f'any(name IN $author_names WHERE toLower({author_var}.name) CONTAINS toLower(name))'
                    )
                    parameters['author_names'] = author_names
                elif ':Author' not in template_cypher:
                    # Template has no Author node at all — skip author condition
                    # (injecting a.name without a MATCH on Author would also fail)
                    logger.info("Template has no Author node, skipping author name condition")
                else:
                    # Fallback: use 'a' as the default alias
                    conditions.append(
                        'any(name IN $author_names WHERE toLower(a.name) CONTAINS toLower(name))'
                    )
                    parameters['author_names'] = author_names
            else:
                # Template already has author filtering — just ensure the parameter is set
                parameters['author_names'] = author_names

        # 4. Extract year from query (e.g. "2023", "since 2020", "before 2024")
        year_match = re.search(r'\b(since|after|from)\s+(\d{4})\b', query, re.IGNORECASE)
        if year_match:
            year = year_match.group(2)
            conditions.append(f"p.publication_date >= '{year}'")
        else:
            year_match = re.search(r'\b(before|until|up\s+to)\s+(\d{4})\b', query, re.IGNORECASE)
            if year_match:
                year = year_match.group(2)
                conditions.append(f"p.publication_date < '{year}'")
            else:
                year_match = re.search(r'\b(in|year)\s+(\d{4})\b', query, re.IGNORECASE)
                if year_match:
                    year = year_match.group(2)
                    conditions.append(f"p.publication_date STARTS WITH '{year}'")
                else:
                    # Standalone year at word boundary
                    standalone_year = re.search(r'\b(20\d{2})\b', query)
                    if standalone_year:
                        year = standalone_year.group(1)
                        conditions.append(f"p.publication_date STARTS WITH '{year}'")

        # 5. Extract keywords for title/abstract search (skip if we already have specific filters)
        if not paper_ids and not inline_ids and not author_names:
            keywords = self._extract_keywords(query)
            if keywords:
                keyword_conditions = []
                for i, kw in enumerate(keywords[:5]):  # Limit to 5 keywords
                    param_name = f'kw_{i}'
                    keyword_conditions.append(
                        f"(toLower(p.title) CONTAINS toLower(${param_name}) OR "
                        f"toLower(p.abstract) CONTAINS toLower(${param_name}))"
                    )
                    parameters[param_name] = kw
                if keyword_conditions:
                    # Any keyword matches (OR logic)
                    conditions.append('(' + ' OR '.join(keyword_conditions) + ')')

        # 6. Extract venue name if mentioned
        venue_match = re.search(
            r'(?:in|from|published\s+in|venue|journal|conference)\s+["\']?([A-Z][^"\',]{2,40})["\']?',
            query, re.IGNORECASE
        )
        if venue_match and ':Venue' in template_cypher:
            venue_name = venue_match.group(1).strip()
            conditions.append('toLower(v.name) CONTAINS toLower($venue_name)')
            parameters['venue_name'] = venue_name

        # Build the refined Cypher
        refined_cypher = self._inject_where_conditions(template_cypher, conditions)
        refined_cypher = self._inject_limit(refined_cypher, top_k)

        # Ensure all $param references in the template have values in the parameters dict.
        # Templates like coauthor_network have $author_names baked into their Cypher, but
        # _refine_template_with_conditions may not have extracted them from the query
        # (e.g. when the query is purely keyword/topic-based and vector-first injected paper_ids).
        default_params = self._build_default_params(refined_cypher, top_k)
        for param, default_val in default_params.items():
            if param not in parameters:
                parameters[param] = default_val
                logger.info(f"Auto-filled missing template parameter ${param} with default: {default_val}")

        logger.info(f"Extracted conditions from query: {conditions}")
        return refined_cypher, parameters

    def _inject_where_conditions(self, cypher: str, conditions: List[str]) -> str:
        """Inject WHERE conditions into a Cypher query.

        Handles:
        - Template already has WHERE → append conditions with AND
        - Template has no WHERE → insert WHERE right after the first MATCH
          (works even when OPTIONAL MATCH follows immediately)
        """
        if not conditions:
            return cypher

        condition_str = ' AND '.join(conditions)
        lines = cypher.strip().split('\n')
        new_lines = []
        injected = False
        found_first_match = False

        for i, line in enumerate(lines):
            stripped = line.strip().upper()

            # Case 1: Existing WHERE — append with AND
            if not injected and stripped.startswith('WHERE'):
                new_lines.append(line.rstrip() + ' AND ' + condition_str)
                injected = True
                continue

            # Case 2: First non-OPTIONAL MATCH
            if not injected and not found_first_match and stripped.startswith('MATCH') and not stripped.startswith(
                    'OPTIONAL'):
                found_first_match = True
                new_lines.append(line)
                # Look ahead: if next non-empty line is WHERE, let that handle it
                next_idx = i + 1
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1
                if next_idx < len(lines) and lines[next_idx].strip().upper().startswith('WHERE'):
                    continue  # WHERE follows, will be handled next iteration
                # Otherwise inject WHERE right after this MATCH line
                new_lines.append('WHERE ' + condition_str)
                injected = True
                continue

            new_lines.append(line)

        # Fallback: add before the first RETURN/ORDER BY/LIMIT if still not injected
        if not injected:
            final_lines = []
            for line in new_lines:
                s = line.strip().upper()
                if not injected and (s.startswith('RETURN') or s.startswith('ORDER') or s.startswith('LIMIT')):
                    final_lines.append('WHERE ' + condition_str)
                    injected = True
                final_lines.append(line)
            new_lines = final_lines

        return '\n'.join(new_lines)

    def _inject_limit(self, cypher: str, top_k: int) -> str:
        """Inject or replace LIMIT clause in a Cypher query."""
        import re as _re
        # Replace existing LIMIT with numeric value (e.g. LIMIT 20)
        if _re.search(r'LIMIT\s+\d+', cypher, _re.IGNORECASE):
            return _re.sub(r'LIMIT\s+\d+', f'LIMIT {top_k}', cypher, flags=_re.IGNORECASE)
        # Replace existing LIMIT with parameter (e.g. LIMIT $limit)
        if _re.search(r'LIMIT\s+\$\w+', cypher, _re.IGNORECASE):
            return _re.sub(r'LIMIT\s+\$\w+', f'LIMIT {top_k}', cypher, flags=_re.IGNORECASE)
        # No LIMIT found — add at the end
        return cypher.rstrip().rstrip(';') + f"\nLIMIT {top_k}"

    def _build_default_params(self, cypher: str, top_k: int) -> Dict:
        """Scan a Cypher query for $param references and return a dict with safe defaults.

        This prevents Neo4j ParameterMissing errors when executing templates that
        reference parameters not extracted from the user's query.
        """
        param_defaults = {
            'author_names': [],
            'paper_ids': [],
            'keywords': [],
            'venue_name': '',
            'institution_name': '',
            'year': '',
            'year_from': '1900',
            'year_to': '2099',
            'limit': top_k,
        }
        params = {}
        referenced = re.findall(r'\$(\w+)', cypher)
        for param in referenced:
            if param in param_defaults:
                params[param] = param_defaults[param]
        return params

    async def _execute_graph_search_internal(self, query: str, top_k: int) -> List[Dict]:
        """Internal graph search implementation with caching."""
        try:
            # Check cache first
            if self.cache_manager:
                cached_results = self.cache_manager.get_search_results(
                    query, top_k, use_hybrid=False, routing_strategy="graph"
                )
                if cached_results is not None:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit('search', True)
                        self.performance_monitor.record_result_count('graph', len(cached_results))
                    logger.debug(f"Graph search cache hit for: {query[:50]}...")
                    return cached_results

            if self.performance_monitor:
                self.performance_monitor.record_cache_hit('search', False)

            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return []

            # Parse the query intelligently based on patterns
            cypher_query, parameters = await self._build_intelligent_cypher_query(query, top_k)

            # Debug logging
            logger.info(f"Graph search query: {query}")
            logger.info(f"Generated Cypher: {cypher_query}")
            logger.info(f"Parameters: {parameters}")

            results = self.graph_handler.execute_query(cypher_query, parameters)
            logger.info(f"Graph search returned {len(results)} results")

            # Add basic relevance scores based on query type
            for result in results:
                result['relevance_score'] = 0.8 if self._is_structured_query(query) else 0.5

            # Cache results
            if self.cache_manager and results:
                self.cache_manager.cache_search_results(
                    query, results, top_k, use_hybrid=False, routing_strategy="graph"
                )

            if self.performance_monitor:
                self.performance_monitor.record_result_count('graph', len(results))

            return results
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []

    def _is_structured_query(self, query: str) -> bool:
        """Check if this is a structured query (ID-based, author lookup, etc.)"""
        paper_id_pattern = re.compile(r'\b(W\d+|DOI:|doi:|pmid:|arxiv:)', re.IGNORECASE)
        author_patterns = [
            re.compile(r'who\s+is\s+the\s+author', re.IGNORECASE),
            re.compile(r'authors?\s+of\s+paper', re.IGNORECASE),
            re.compile(r'who\s+wrote', re.IGNORECASE),
            re.compile(r'who\s+authored', re.IGNORECASE)
        ]

        return (paper_id_pattern.search(query) or
                any(pattern.search(query) for pattern in author_patterns))

    def _is_keyword_or_topic_query(self, query: str) -> bool:
        """Detect if the query is asking about a keyword, topic, or subject area.

        Uses the AI agent with few-shot examples to classify the query.
        Falls back to regex-based heuristics if the AI agent is unavailable.

        Returns True when the query is about a research topic, keyword, or subject
        area — meaning vector-first paper ID scoping should be applied.
        """
        # Fast exit: explicit paper IDs are never keyword/topic queries
        if re.search(r'\b(W\d+)\b', query):
            return False

        # Try AI-based detection first
        if self.ai_agent:
            try:
                result = self._detect_keyword_topic_with_ai(query)
                if result is not None:
                    return result
            except Exception as e:
                logger.warning(f"AI keyword/topic detection failed: {e}")

        # Fallback: regex-based heuristics
        return self._detect_keyword_topic_by_rules(query)

    def _detect_keyword_topic_with_ai(self, query: str) -> Optional[bool]:
        """Use AI agent with few-shot examples to classify whether a query is keyword/topic-based.

        Returns:
            True if keyword/topic query, False if not, None if AI response is unparseable.
        """
        prompt = """You are a query classifier for an academic paper search system.

Classify whether the user's query is about a KEYWORD or TOPIC (a research subject, field, or concept) 
versus a STRUCTURED query (asking about a specific author, paper ID, venue, citation, or co-author).

If the query is about a keyword/topic, we will use vector search to find semantically relevant papers 
first, then enrich them with graph metadata. If it's structured, we skip vector search.

Here are examples:

Query: "papers about machine learning"
Answer: YES

Query: "deep learning for medical imaging"
Answer: YES

Query: "natural language processing since 2020"
Answer: YES

Query: "transformer models in NLP"
Answer: YES

Query: "what research exists on graph neural networks"
Answer: YES

Query: "recent advances in protein folding"
Answer: YES

Query: "top cited papers on attention mechanisms"
Answer: YES

Query: "knowledge graph embedding methods"
Answer: YES

Query: "papers by John Smith"
Answer: NO

Query: "paper W12345"
Answer: NO

Query: "who authored this paper"
Answer: NO

Query: "co-authors of Alice Johnson"
Answer: NO

Query: "papers published in Nature"
Answer: NO

Query: "citations of W98765"
Answer: NO

Query: "papers from Stanford University"
Answer: NO

Query: "which journals does Bob Lee publish in"
Answer: NO

Now classify this query:
Query: "{query}"
Answer:""".format(query=query)

        response = self.ai_agent.generate_content(
            prompt=prompt,
            system_prompt="You are a query classifier. Respond with only YES or NO.",
            purpose='keyword_topic_detection'
        )

        if response:
            answer = response.strip().upper().strip('"\'., ')
            if answer in ('YES', 'Y', 'TRUE'):
                logger.info(f"AI classified query as keyword/topic: '{query[:60]}'")
                return True
            elif answer in ('NO', 'N', 'FALSE'):
                logger.info(f"AI classified query as structured (not keyword/topic): '{query[:60]}'")
                return False
            else:
                logger.warning(f"AI returned unparseable answer for keyword/topic detection: '{response}'")

        return None

    def _detect_keyword_topic_by_rules(self, query: str) -> bool:
        """Regex-based fallback for keyword/topic detection when AI is unavailable."""
        query_lower = query.lower()

        # If query mentions specific authors, it's not purely keyword-based
        author_names = self._extract_author_names(query)
        if author_names:
            return False

        # Check for topic/keyword indicator phrases
        topic_indicators = [
            r'\b(?:about|on|regarding|concerning|related\s+to|topic|field|area|domain)\b',
            r'\b(?:papers?|research|studies|work)\s+(?:on|about|in|regarding)\b',
            r'\b(?:find|search|look\s+for|show)\s+(?:papers?|research|studies|articles?)\b',
        ]

        for pattern in topic_indicators:
            if re.search(pattern, query_lower):
                return True

        # If no structured identifiers are found and keywords can be extracted,
        # treat it as a keyword/topic query
        keywords = self._extract_keywords(query)
        if keywords and len(keywords) >= 1:
            return True

        return False

    def _extract_author_names(self, query: str) -> List[str]:
        """Fallback regex-based author name extraction."""
        author_names = []

        # Pattern 1: "papers by John Smith"
        author_patterns = [
            r'papers?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'authored?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'written\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'authors?\s+(["\']?)([^"\']+?)\1(?:\s|and\s+)',
            r'co-?authored?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)'
        ]

        for pattern in author_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                name = match.group(2).strip()
                if len(name.split()) >= 2:  # At least first and last name
                    author_names.append(name)

        # Pattern 2: "John Smith and Mary Johnson"
        and_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+))*'
        and_matches = re.findall(and_pattern, query)
        for match_group in and_matches:
            for name in match_group:
                if name and len(name.split()) >= 2:
                    author_names.append(name.strip())

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in author_names:
            name_clean = name.lower().strip()
            if name_clean not in seen:
                seen.add(name_clean)
                unique_names.append(name)

        return unique_names

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query text."""
        # Remove common stop words and query patterns
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'about', 'papers', 'paper', 'research', 'study', 'studies', 'find', 'search',
            'what', 'where', 'when', 'who', 'how', 'why', 'which', 'is', 'are', 'was', 'were',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'related', 'concerning', 'regarding', 'involving'
        }

        # Clean and tokenize
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = clean_query.split()

        # Filter meaningful keywords
        keywords = []
        for word in words:
            if (len(word) > 2 and
                    word not in stop_words and
                    not word.isdigit() and
                    not re.match(r'^w\d+$', word)):  # Not paper IDs
                keywords.append(word)

        # Add the original query as a keyword for exact matches
        if query.strip():
            keywords.append(query.strip())

        return list(set(keywords))  # Remove duplicates

    # =========================================================================
    # GRAPH TEMPLATE LIBRARY — Loaded from data/graph_templates.json
    # =========================================================================

    _graph_templates_cache: Optional[Dict] = None

    @classmethod
    def _load_graph_templates(cls) -> Dict:
        """Load graph query templates from data/graph_templates.json.

        Templates are cached after first load for performance.
        Returns the template dictionary.
        """
        if cls._graph_templates_cache is not None:
            return cls._graph_templates_cache

        # Resolve path relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        templates_path = os.path.join(project_root, 'data', 'graph_templates.json')

        try:
            with open(templates_path, 'r', encoding='utf-8') as f:
                cls._graph_templates_cache = json.load(f)
            logger.info(f"Loaded {len(cls._graph_templates_cache)} graph templates from {templates_path}")
        except FileNotFoundError:
            logger.error(f"Graph templates file not found: {templates_path}")
            cls._graph_templates_cache = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in graph templates file: {e}")
            cls._graph_templates_cache = {}

        return cls._graph_templates_cache

    @classmethod
    def reload_graph_templates(cls):
        """Force reload templates from disk (e.g. after editing the JSON file)."""
        cls._graph_templates_cache = None
        return cls._load_graph_templates()

    @property
    def GRAPH_TEMPLATES(self) -> Dict:
        """Access graph templates (loaded from data/graph_templates.json)."""
        return self._load_graph_templates()

    # =========================================================================
    # TEMPLATE SELECTION — AI agent picks the best template
    # =========================================================================

    def _select_template_with_ai(self, query: str, extracted: Dict) -> str:
        """Use AI agent to select the best graph template for the user's query.

        Args:
            query: User's natural language query
            extracted: Pre-extracted entities {paper_ids, author_names, keywords, year, ...}

        Returns:
            Template key name from GRAPH_TEMPLATES
        """
        if not self.ai_agent:
            return self._select_template_by_rules(extracted)

        # Build template catalog for the AI
        template_list = []
        for key, tpl in self.GRAPH_TEMPLATES.items():
            template_list.append(f"- {key}: {tpl['description']}")
        templates_str = "\n".join(template_list)

        # Build extracted entities summary
        entities_str = json.dumps({k: v for k, v in extracted.items() if v}, ensure_ascii=False)

        prompt = f"""You are a query router for a Neo4j academic paper database.

Given the user's query and the extracted entities, select the BEST template to answer their question.

User query: "{query}"
Extracted entities: {entities_str}

Available templates:
{templates_str}

Rules:
1. If query mentions specific paper IDs (W12345), use "search_by_paper_ids"
2. If query mentions both an author AND a topic, use "search_by_author_by_keywords"
3. If query mentions an author name only, use "search_by_author"
4. If query asks about co-authors or collaboration, use "coauthor_network"
5. If query asks about which journals/venues an author publishes in, use "author_venue_stats"
6. If query asks about citations or "cited by", use "search_citations"
7. If query mentions a venue/journal/conference name, use "search_by_venue"
8. If query mentions an institution/university, use "search_by_institution"
9. If query mentions "most cited" or "top papers", use "top_cited_papers"
10. If query mentions a year range (since/after/before), use "search_by_year_range"
11. If query mentions a specific year, use "search_by_year"
12. Otherwise, use "search_by_keywords"

Respond with ONLY the template name, nothing else. Example: search_by_author"""

        try:
            response = self.ai_agent.generate_content(
                prompt=prompt,
                system_prompt="You are a query router. Respond with only the template name.",
                purpose='template_selection'
            )

            if response:
                template_key = response.strip().strip('"\'').strip()
                # Validate the response is a known template
                if template_key in self.GRAPH_TEMPLATES:
                    logger.info(f"AI selected template: {template_key}")
                    return template_key
                else:
                    logger.warning(f"AI returned unknown template '{template_key}', falling back to rules")

        except Exception as e:
            logger.warning(f"AI template selection failed: {e}")

        return self._select_template_by_rules(extracted)

    def _select_template_by_rules(self, extracted: Dict) -> str:
        """Rule-based fallback for template selection when AI is unavailable.

        Args:
            extracted: Pre-extracted entities from the query

        Returns:
            Template key name
        """
        has_ids = bool(extracted.get('paper_ids'))
        has_authors = bool(extracted.get('author_names'))
        has_keywords = bool(extracted.get('keywords'))
        has_venue = bool(extracted.get('venue'))
        has_institution = bool(extracted.get('institution'))
        has_year = bool(extracted.get('year'))
        has_year_range = bool(extracted.get('year_from'))
        has_citations = extracted.get('wants_citations', False)
        has_coauthor = extracted.get('wants_coauthors', False)
        has_top_cited = extracted.get('wants_top_cited', False)

        # Priority-based selection
        if has_ids:
            if has_citations:
                return 'search_citations'
            return 'search_by_paper_ids'
        if has_coauthor and has_authors:
            return 'coauthor_network'
        if has_authors and has_keywords:
            return 'search_author_by_keywords'
        if has_authors:
            return 'search_by_author'
        if has_citations:
            return 'search_citations'
        if has_top_cited:
            return 'top_cited_papers'
        if has_venue:
            return 'search_by_venue'
        if has_institution:
            return 'search_by_institution'
        if has_year_range:
            return 'search_by_year_range'
        if has_year:
            return 'search_by_year'
        return 'search_by_keywords'

    # =========================================================================
    # ENTITY EXTRACTION — Pull structured data from natural language query
    # =========================================================================

    def _extract_all_entities(self, query: str) -> Dict:
        """Extract all entities from a natural language query.

        Returns a dict with:
            paper_ids, author_names, keywords, year, year_from, year_to,
            venue, institution, wants_citations, wants_coauthors, wants_top_cited
        """
        extracted = {}
        query_lower = query.lower()

        # 1. Paper IDs
        paper_ids = re.findall(r'\b(W\d+)\b', query)
        if paper_ids:
            extracted['paper_ids'] = paper_ids

        # 2. Author names
        author_names = self._extract_author_names(query)
        if author_names:
            extracted['author_names'] = author_names

        # 3. Year / year range
        range_match = re.search(r'\b(since|after|from)\s+(\d{4})\b', query, re.IGNORECASE)
        if range_match:
            extracted['year_from'] = range_match.group(2)
            extracted['year_to'] = '2099'  # open-ended
        else:
            before_match = re.search(r'\b(before|until|up\s+to)\s+(\d{4})\b', query, re.IGNORECASE)
            if before_match:
                extracted['year_from'] = '1900'
                extracted['year_to'] = before_match.group(2)
            else:
                between_match = re.search(r'\b(\d{4})\s*[-–to]+\s*(\d{4})\b', query)
                if between_match:
                    extracted['year_from'] = between_match.group(1)
                    extracted['year_to'] = between_match.group(2)
                else:
                    year_match = re.search(r'\b(in|year)\s+(\d{4})\b', query, re.IGNORECASE)
                    if year_match:
                        extracted['year'] = year_match.group(2)
                    else:
                        standalone = re.search(r'\b(20\d{2})\b', query)
                        if standalone:
                            extracted['year'] = standalone.group(1)

        # 4. Venue
        venue_match = re.search(
            r'(?:in|from|published\s+in|venue|journal|conference)\s+["\']?([A-Z][^"\',]{2,40})["\']?',
            query, re.IGNORECASE
        )
        if venue_match:
            extracted['venue'] = venue_match.group(1).strip()

        # 5. Institution
        inst_match = re.search(
            r'(?:from|at|institution|university|org(?:anization)?)\s+["\']?([A-Z][^"\',]{2,50})["\']?',
            query, re.IGNORECASE
        )
        if inst_match and not extracted.get('venue'):  # avoid conflict with venue
            extracted['institution'] = inst_match.group(1).strip()

        # 6. Intent flags
        if re.search(r'\bcit(?:e|ed|ation|ing)\b', query_lower):
            extracted['wants_citations'] = True
        if re.search(r'\bco-?author|collaborat', query_lower):
            extracted['wants_coauthors'] = True
        if re.search(r'\b(?:most|top|highest)\s+cited\b|\btop\s+papers?\b|\bmost\s+influential\b', query_lower):
            extracted['wants_top_cited'] = True

        # 7. Keywords (extract last — exclude entities already captured)
        keywords = self._extract_keywords(query)
        if keywords:
            extracted['keywords'] = keywords

        return extracted

    # =========================================================================
    # PARAMETER BUILDERS — Fill template parameters from extracted entities
    # =========================================================================

    def _params_paper_ids(self, extracted: Dict, top_k: int) -> Dict:
        return {"paper_ids": extracted.get('paper_ids', []), "limit": top_k}

    def _params_author(self, extracted: Dict, top_k: int) -> Dict:
        return {"author_names": extracted.get('author_names', []), "limit": top_k}

    def _params_keywords(self, extracted: Dict, top_k: int) -> Dict:
        return {"keywords": extracted.get('keywords', []), "limit": top_k}

    def _params_venue(self, extracted: Dict, top_k: int) -> Dict:
        return {"venue_name": extracted.get('venue', ''), "limit": top_k}

    def _params_institution(self, extracted: Dict, top_k: int) -> Dict:
        return {"institution_name": extracted.get('institution', ''), "limit": top_k}

    def _params_year(self, extracted: Dict, top_k: int) -> Dict:
        return {"year": extracted.get('year', ''), "limit": top_k}

    def _params_year_range(self, extracted: Dict, top_k: int) -> Dict:
        return {"year_from": extracted.get('year_from', '1900'), "year_to": extracted.get('year_to', '2099'),
                "limit": top_k}

    def _params_citations(self, extracted: Dict, top_k: int) -> Dict:
        return {"paper_ids": extracted.get('paper_ids', []), "limit": top_k}

    def _params_author_keywords(self, extracted: Dict, top_k: int) -> Dict:
        return {"author_names": extracted.get('author_names', []), "keywords": extracted.get('keywords', []),
                "limit": top_k}

    def _params_top_cited(self, extracted: Dict, top_k: int) -> Dict:
        return {"limit": top_k}

    def _params_coauthor(self, extracted: Dict, top_k: int) -> Dict:
        return {"author_names": extracted.get('author_names', []), "limit": top_k}

    # =========================================================================
    # BUILD INTELLIGENT CYPHER — Main entry point for graph search
    # =========================================================================

    async def _build_intelligent_cypher_query(self, query: str, top_k: int, template_cypher: str = None) -> tuple:
        """Build Cypher query by selecting the best template and filling parameters.

        Flow:
        1. Check Cypher cache
        2. Extract entities from query (paper IDs, authors, keywords, years, etc.)
        3. AI agent selects the best template (fallback: rule-based selection)
        4. If template is keyword-only → vector-first: run vector search to get
           relevant paper IDs, then use search_by_paper_ids for rich graph metadata
        5. Fill template parameters from extracted entities
        6. Cache and return

        Returns:
            (cypher_query, parameters) tuple
        """
        try:
            # Check Cypher cache first
            if self.cache_manager:
                cached = self.cache_manager.get_cypher(query, top_k)
                if cached is not None:
                    cypher_query, parameters = cached
                    logger.info(f"Cypher cache HIT for: {query[:50]}...")
                    return cypher_query, parameters

            # Step 1: Extract entities from the query
            extracted = self._extract_all_entities(query)
            logger.info(f"Extracted entities: {extracted}")

            # Step 2: AI agent selects the best template
            if template_cypher:
                template_key = template_cypher
            else:
                template_key = self._select_template_with_ai(query, extracted)
            logger.info(f"AI selected template: {template_key}")

            # Step 3: Vector-first for keyword queries
            # When the query is keyword-based (no specific IDs, authors, etc.),
            # use vector search to find semantically relevant papers first,
            # then fetch their full graph metadata via search_by_paper_ids.
            # Use .get() with an empty list default to simplify the loop
            keywords = extracted.get('keywords', [])

            all_results = {}

            for keyword in keywords:
                results = await self._execute_vector_search_internal(keyword, top_k)

                for paper in results:
                    p_id = paper.get('paper_id')
                    dist = paper.get('distance', 0.0)
                    if p_id not in all_results or dist < all_results[p_id]:
                        all_results[p_id] = dist

            # Sort by distance (ascending) and take the top_k
            sorted_ids = sorted(all_results, key=all_results.get)[:top_k]

            # Extend the extracted list
            extracted.setdefault('paper_ids', []).extend(sorted_ids)

            template = self.GRAPH_TEMPLATES[template_key]
            logger.info(f"Final template: {template_key} — {template['description']}")

            # Step 4: Build parameters using the template's param builder
            param_builder = getattr(self, template['param_builder'])
            parameters = param_builder(extracted, top_k)

            cypher_query = template['cypher'].strip()

            # Step 5: Cache the result
            if self.cache_manager:
                self.cache_manager.cache_cypher(query, top_k, cypher_query, parameters)

            return cypher_query, parameters

        except Exception as e:
            logger.error(f"Error in intelligent Cypher generation: {e}")
            # Ultimate fallback — vector-first then paper IDs lookup
            paper_ids = self._vector_first_paper_ids(query, top_k) if self.milvus_client else None
            if paper_ids:
                return self.GRAPH_TEMPLATES['search_by_paper_ids']['cypher'].strip(), \
                    {"paper_ids": paper_ids, "limit": top_k}
            keywords = self._extract_keywords(query)
            return self.GRAPH_TEMPLATES['search_by_keywords']['cypher'].strip(), \
                {"keywords": keywords or [query], "limit": top_k}

    def _vector_first_paper_ids(self, query: str, top_k: int) -> List[str]:
        """Run vector search to extract relevant paper IDs for graph enrichment.

        This is the core of the vector-first strategy: use semantic similarity
        to find relevant papers, then pass their IDs to a graph template for
        rich metadata (authors, venues, citations, etc.).

        Args:
            query: Natural language search query
            top_k: Number of papers to retrieve

        Returns:
            List of paper IDs (e.g. ['W12345', 'W67890'])
        """
        try:
            vector_results = self.search_similar_papers(
                query_text=query,
                top_k=top_k,
                use_hybrid=True
            )
            if not vector_results:
                logger.warning("Vector-first: no results from vector search")
                return []

            paper_ids = [r.get('paper_id') for r in vector_results]

            logger.info(f"Vector-first: extracted {len(paper_ids)} paper IDs from vector search")
            return paper_ids

        except Exception as e:
            logger.error(f"Vector-first search failed: {e}")
            return []

    async def _execute_graph_refinement(self, paper_ids: List[str], top_k: int) -> List[Dict]:
        """Refine vector results using graph relationships."""
        try:
            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return []

            # Find related papers through citations and collaborations
            cypher_query = """
                MATCH (p:Paper)
                WHERE p.id IN $paper_ids
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                       p.doi as doi, p.publication_date as publication_date,
                       p.cited_by_count as cited_by_count,
                       collect(DISTINCT a.name) as authors,
                       v.name as venue
                ORDER BY p.id
            """

            logger.info(paper_ids)

            results = self.graph_handler.execute_query(cypher_query, {
                "paper_ids": paper_ids,
                "top_k": top_k
            })

            # # Add relevance scores based on graph distance
            # for result in results:
            #     result['relevance_score'] = 0.7  # Higher score for graph-refined results

            return results
        except Exception as e:
            logger.error(f"Graph refinement error: {e}")
            return []

    async def generate_ai_response(
            self,
            query: str,
            search_results: List,
            query_type: QueryType
    ) -> Optional[str]:
        """Generate AI response from vector and graph searches."""
        try:
            if not self.ai_agent:
                logger.info("AI Agent is not available for response generation")
                return None

            # Prepare context from search results
            context_papers = []
            for result in search_results[:5]:  # Use top 5 results
                # Handle both dict and object types
                if hasattr(result, '__dict__'):
                    # It's an object, use getattr
                    context_papers.append({
                        "title": getattr(result, "title", "") or "",
                        "authors": getattr(result, "authors", []) or [],
                        "abstract": (getattr(result, "abstract", "") or "")[:300],
                        "relevance_score": getattr(result, "relevance_score", 0),
                        "venue": getattr(result, "venue", "") or "",
                        "publication_date": getattr(result, "publication_date", "") or "",
                        "paper_id": getattr(result, "paper_id", "") or getattr(result, "id", ""),
                        "doi": getattr(result, "doi", "") or ""
                    })
                else:
                    # It's a dict, use .get()
                    context_papers.append({
                        "title": result.get("title", "") or "",
                        "authors": result.get("authors", []) or [],
                        "abstract": (result.get("abstract", "") or "")[:300],
                        "relevance_score": result.get("relevance_score", 0),
                        "venue": result.get("venue", "") or "",
                        "publication_date": result.get("publication_date", "") or "",
                        "paper_id": result.get("paper_id", "") or result.get("id", ""),
                        "doi": result.get("doi", "") or ""
                    })

            # Create specialized prompt and system prompt based on query type
            if query_type == QueryType.STRUCTURAL:
                system_prompt = "You are a helpful research assistant. Be direct, factual, and concise."
                prompt = f"""Answer this question directly: "{query}"

Search Results:
{self._format_papers_for_prompt(context_papers[:3])}

Instructions:
- Answer in 2-4 sentences maximum
- Use specific information (authors, dates, titles)
- Be factual and precise

Answer:"""
            elif query_type == QueryType.SEMANTIC:
                system_prompt = "You are a research assistant. Synthesize findings concisely."
                prompt = f"""Answer this question based on the search results: "{query}"

Search Results:
{self._format_papers_for_prompt(context_papers[:4])}

Instructions:
- Synthesize key findings in 3-5 sentences
- Cite specific papers and authors
- Connect related findings

Answer:"""
            else:  # FACTUAL, HYBRID, or other types
                system_prompt = "You are a research assistant. Be accurate and concise."
                prompt = f"""Answer this question based on the search results: "{query}"

Search Results:
{self._format_papers_for_prompt(context_papers[:4])}

Instructions:
- Answer directly in 3-5 sentences
- Cite specific papers and authors when relevant
- Be accurate — only use information from the provided results

Answer:"""

            # Generate response using DeepseekClient with both prompt and system_prompt
            ai_answer = self.ai_agent.generate_content(
                prompt=prompt,
                system_prompt=system_prompt,
                purpose='answer_synthesis'
            )

            if not ai_answer:
                logger.error("Empty response from Llama for AI generation")
                return None

            logger.info(f"Generated AI response of {len(ai_answer)} characters")
            return ai_answer

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None

    async def verify_claims_scifact(self, ai_answer: str, context_papers: List) -> List[Dict]:
        """
        Implements SciFact-style verification by breaking the AI response into
        atomic claims and checking them against the retrieved substrate.
        """
        # Step 1: Claim Extraction
        # Use a prompt to break the response into individual facts
        claims = self.extract_atomic_claims(ai_answer)

        verification_results = []

        for claim in claims:
            # Step 2: Evidence Alignment
            # Compare the claim against the 'top k' abstracts and graph metadata
            logic_prompt = f"""
            Claim: {claim}
            Evidence: {self._format_papers_for_prompt(context_papers)}

            Label this claim as:
            - SUPPORTED: If the evidence explicitly confirms the claim.
            - CONTRADICTED: If the evidence explicitly refutes the claim.
            - NO_EVIDENCE: If the evidence is relevant but doesn't prove/disprove it.
            """

            label = self.ai_agent.generate_content(logic_prompt, purpose='scifact_verification')
            verification_results.append({"claim": claim, "label": label})

        return verification_results

    def extract_atomic_claims(self, ai_answer: str) -> List[str]:
        """
        Decomposes a complex AI response into atomic, verifiable scientific claims.
        This supports the 'Verification' research goal of reducing hallucinations[cite: 25].
        """
        try:
            if not ai_answer or len(ai_answer.strip()) == 0:
                return []

            # This prompt instructs the LLM to act as a claim extractor,
            # focusing on 'scientific discovery' and 'factual groundedness'[cite: 7, 47].
            extraction_prompt = f"""
            You are a scientific fact-checker. Break the following AI-generated response into a list of atomic, independent claims.

            Rules:
            1. Each claim must be a single sentence.
            2. Each claim must be self-contained (replace pronouns like 'it' or 'they' with the specific entity names).
            3. Remove introductory phrases (e.g., "The paper says", "According to the results").
            4. Focus on scientific facts, authors, dates, and relationships between entities.

            Response to decompose:
            "{ai_answer}"

            Return the claims as a simple bulleted list.
            """

            # Using your existing AI agent infrastructure
            raw_claims = self.ai_agent.generate_content(prompt=extraction_prompt, purpose='claim_extraction')

            if not raw_claims:
                return []

            # Clean the output into a Python list
            processed_claims = [
                claim.strip().lstrip('- ').lstrip('* ').strip()
                for claim in raw_claims.split('\n')
                if len(claim.strip()) > 10  # Filter out noise or empty lines
            ]

            logger.info(f"Extracted {len(processed_claims)} atomic claims for verification [cite: 25]")
            return processed_claims

        except Exception as e:
            logger.error(f"Error in claim extraction: {e}")
            return []

    def _format_papers_for_prompt(self, papers: List[Dict]) -> str:
        """Format papers for inclusion in response generation."""
        formatted_papers = []

        for i, paper in enumerate(papers, 1):
            authors = paper.get("authors", [])
            authors_str = ", ".join(authors[:3]) if authors else "N/A"
            if len(authors) > 3:
                authors_str += " et al."

            paper_text = f"""{i}. Title: {paper.get('title', 'N/A')}
   Authors: {authors_str}
   Venue: {paper.get('venue', 'N/A')}
   Date: {paper.get('publication_date', 'N/A')}
   Relevance: {paper.get('relevance_score', 0):.2f}
   Abstract: {paper.get('abstract', 'N/A')}"""

            formatted_papers.append(paper_text)

        return "\n\n".join(formatted_papers)

    # =========================================================================
    # IMAGE SEARCH METHODS
    # =========================================================================

    async def search_by_image(self, image_embedding: List[float], top_k: int = 10,
                              search_figures: bool = True, search_tables: bool = True,
                              text_query: Optional[str] = None) -> Dict:
        """Search figures and tables collections by image embedding.

        Args:
            image_embedding: CLIP image embedding vector
            top_k: Number of results per collection
            search_figures: Whether to search figures collection
            search_tables: Whether to search tables collection
            text_query: Optional text query for hybrid image+text search

        Returns:
            Dict with figure_results, table_results, and related_papers
        """
        figure_results = []
        table_results = []

        try:
            # Search figures collection
            if search_figures:
                figure_results = self.milvus_client.search_figures_by_image(
                    image_embedding, top_k
                )

            # Search tables collection
            if search_tables:
                table_results = self.milvus_client.search_tables_by_image(
                    image_embedding, top_k
                )

            # If text query provided, also search by description and merge results
            if text_query and self.embedding_client:
                text_embedding = self.embedding_client.generate_text_embedding(text_query)
                if text_embedding:
                    if search_figures:
                        text_fig_results = self.milvus_client.search_figures_by_description(
                            text_embedding, top_k
                        )
                        figure_results = self._merge_visual_results(figure_results, text_fig_results)

                    if search_tables:
                        text_tab_results = self.milvus_client.search_tables_by_description(
                            text_embedding, top_k
                        )
                        table_results = self._merge_visual_results(table_results, text_tab_results)

            # Collect related paper IDs from visual results
            paper_ids = set()
            for r in figure_results + table_results:
                if r.get("paper_id"):
                    paper_ids.add(r["paper_id"])

            # Fetch related paper details — prefer Neo4j for full metadata
            related_papers = []
            paper_ids_list = list(paper_ids)[:20]

            if paper_ids_list and self.graph_handler:
                try:
                    related_papers = await self._execute_graph_refinement(paper_ids_list, len(paper_ids_list))
                    logger.info(f"Fetched {len(related_papers)} related papers from Neo4j")
                except Exception as e:
                    logger.warning(f"Neo4j paper enrichment failed, falling back to Milvus: {e}")
                    related_papers = []

            # Fallback to Milvus if Neo4j returned nothing
            if not related_papers and paper_ids_list and self.milvus_client.collection:
                for pid in paper_ids_list:
                    try:
                        paper_results = self.milvus_client.collection.query(
                            expr=f'id == "{pid}"',
                            output_fields=["id", "title", "abstract"]
                        )
                        if paper_results:
                            p = paper_results[0]
                            related_papers.append({
                                "paper_id": p.get("id", ""),
                                "title": p.get("title", ""),
                                "abstract": p.get("abstract", ""),
                                "authors": [],
                                "venue": None,
                                "doi": None,
                                "publication_date": None,
                                "cited_by_count": 0
                            })
                    except Exception as e:
                        logger.debug(f"Could not fetch paper {pid}: {e}")

            # Attach visual match counts per paper
            paper_visual_counts = {}
            for r in figure_results + table_results:
                pid = r.get("paper_id")
                if pid:
                    paper_visual_counts.setdefault(pid, {"figures": 0, "tables": 0})
                    if r.get("collection") == "figures" or r.get("search_type", "").startswith("figure"):
                        paper_visual_counts[pid]["figures"] += 1
                    else:
                        paper_visual_counts[pid]["tables"] += 1

            for paper in related_papers:
                pid = paper.get("paper_id", paper.get("id", ""))
                counts = paper_visual_counts.get(pid, {"figures": 0, "tables": 0})
                paper["matched_figures"] = counts["figures"]
                paper["matched_tables"] = counts["tables"]
                # Add a relevance score based on visual match density
                paper.setdefault("relevance_score", 0.0)
                if paper["relevance_score"] == 0:
                    paper["relevance_score"] = min(1.0, (counts["figures"] + counts["tables"]) * 0.15)

            # Sort papers by number of visual matches
            related_papers.sort(
                key=lambda p: p.get("matched_figures", 0) + p.get("matched_tables", 0),
                reverse=True
            )
            logger.info(related_papers)
            return {
                "figure_results": figure_results[:top_k],
                "table_results": table_results[:top_k],
                "related_papers": related_papers
            }

        except Exception as e:
            logger.error(f"Image search error: {e}")
            return {
                "figure_results": figure_results,
                "table_results": table_results,
                "related_papers": []
            }

    def _merge_visual_results(self, image_results: List[Dict], text_results: List[Dict],
                              image_weight: float = 0.6, text_weight: float = 0.4) -> List[Dict]:
        """Merge image-based and text-based visual search results using weighted scores.

        Args:
            image_results: Results from image embedding search
            text_results: Results from text description search
            image_weight: Weight for image similarity scores
            text_weight: Weight for text similarity scores

        Returns:
            Merged and re-ranked results
        """
        merged = {}

        for r in image_results:
            key = r.get("id", "")
            merged[key] = {
                **r,
                "image_score": r.get("similarity_score", 0),
                "text_score": 0,
                "combined_score": r.get("similarity_score", 0) * image_weight,
                "search_type": "image"
            }

        for r in text_results:
            key = r.get("id", "")
            if key in merged:
                merged[key]["text_score"] = r.get("similarity_score", 0)
                merged[key]["combined_score"] = (
                        merged[key]["image_score"] * image_weight +
                        r.get("similarity_score", 0) * text_weight
                )
                merged[key]["search_type"] = "hybrid_visual"
            else:
                merged[key] = {
                    **r,
                    "image_score": 0,
                    "text_score": r.get("similarity_score", 0),
                    "combined_score": r.get("similarity_score", 0) * text_weight,
                    "search_type": "text"
                }

        # Sort by combined score
        sorted_results = sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)

        # Update similarity_score to combined_score
        for r in sorted_results:
            r["similarity_score"] = r["combined_score"]

        return sorted_results
