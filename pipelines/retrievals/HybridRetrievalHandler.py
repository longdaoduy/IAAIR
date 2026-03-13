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
from clients.huggingface.LLM_Client import LLMClient
from clients.huggingface.CLIPClient import CLIPClient
from pymilvus import (Collection)
from utils.async_utils import run_blocking

logger = logging.getLogger(__name__)


class HybridRetrievalHandler:
    """Unified handler for vector and graph-based retrieval operations."""

    def __init__(self, vector_db: Optional[MilvusClient], graph_db: GraphQueryHandler,
                 ai_agent: Optional[LLMClient], embedder: SciBERTClient,
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

            results = await self.search_similar_papers(
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

    async def search_similar_papers(self, query_text: str, top_k: int = 10, use_hybrid: bool = True) -> List[Dict]:
        """Search for similar papers using hybrid search (dense + sparse) or dense-only search.

        Blocking operations (embedding generation, Milvus search) are offloaded
        to a thread pool so they don't block the asyncio event loop.

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

            # Generate embedding if not cached — offload to thread pool
            if query_embedding is None:
                if self.performance_monitor:
                    self.performance_monitor.record_cache_hit('embedding', False)
                    with self.performance_monitor.track_operation('embedding'):
                        query_embedding = await run_blocking(
                            self.embedding_client.generate_text_embedding, query_text
                        )
                else:
                    query_embedding = await run_blocking(
                        self.embedding_client.generate_text_embedding, query_text
                    )

                # Cache the embedding
                if self.cache_manager and query_embedding is not None:
                    self.cache_manager.cache_embedding(query_text, query_embedding)

            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return []

            # Execute Milvus search — offload to thread pool
            return await run_blocking(
                self.milvus_client._hybrid_search, query_text, query_embedding, top_k
            )

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    async def execute_hybrid_search(self, query: str, top_k: int, template_cypher: str = None) -> tuple:
        """Execute graph search using Cypher query with intelligent query parsing and caching.

        Returns:
            (results, template_info) tuple where template_info is a dict with
            'template_key' and 'description' of the graph template used.
        """
        if self.performance_monitor:
            with self.performance_monitor.track_operation('graph_search'):
                return await self._execute_hybrid_search_internal(query, top_k, template_cypher)
        else:
            return await self._execute_hybrid_search_internal(query, top_k, template_cypher)

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

    async def _execute_hybrid_search_internal(self, query: str, top_k: int, template_cypher: str = None) -> tuple:
        """Internal graph search implementation with caching.

        Returns:
            (results, template_info) tuple where template_info is a dict with
            'template_key' and 'description' of the graph template used.
        """
        template_info = {"template_key": None, "description": None}
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
                    return cached_results, template_info

            if self.performance_monitor:
                self.performance_monitor.record_cache_hit('search', False)

            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return [], template_info

            # Parse the query intelligently based on patterns
            cypher_query, parameters, template_key = await self._build_intelligent_cypher_query(query, top_k, template_cypher)

            # Build template info
            template_desc = self.GRAPH_TEMPLATES.get(template_key, {}).get('description', template_key)
            template_info = {"template_key": template_key, "description": template_desc}

            # Debug logging
            logger.info(f"Graph search query: {query}")
            logger.info(f"Generated Cypher: {cypher_query}")
            logger.info(f"Parameters: {parameters}")
            logger.info(f"Template used: {template_key} — {template_desc}")

            results = await run_blocking(self.graph_handler.execute_query, cypher_query, parameters)
            logger.info(f"Graph search returned {len(results)} results")

            # Cache results
            if self.cache_manager and results:
                self.cache_manager.cache_search_results(
                    query, results, top_k, use_hybrid=False, routing_strategy="graph"
                )

            if self.performance_monitor:
                self.performance_monitor.record_result_count('graph', len(results))

            return results, template_info
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return [], template_info

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

    async def _select_template_with_ai(self, query: str, extracted: Dict) -> str:
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
Given the user's query and the extracted entities, select the BEST template.

User query: "{query}"
Extracted entities: {entities_str}

Available templates:
{templates_str}

Here are example queries and their correct template:

Query: "Who are the authors of paper W1775749144?"
Template: search_by_paper_ids

Query: "What papers has Kaiming He authored?"
Template: search_by_author

Query: "Which papers cite W2128635872?"
Template: search_citations

Query: "How many citations does paper W2100837269 have?"
Template: search_by_paper_ids

Query: "What venue published paper W3038568908?"
Template: search_by_paper_ids

Query: "Which papers were published in Nature?"
Template: search_by_venue

Query: "What papers were published in Analytical Biochemistry?"
Template: search_by_venue

Query: "Which papers were co-authored by Georg Kresse and J. Furthmüller?"
Template: coauthor_network

Query: "What papers are co-authored by Kaiming He and Jian Sun?"
Template: coauthor_network

Query: "Which paper has the highest citation count?"
Template: top_cited_papers

Query: "What is the DOI of paper W1979290264?"
Template: search_by_paper_ids

Query: "What is the publication year of paper W2107277218?"
Template: search_by_paper_ids

Query: "papers about protein quantification methods"
Template: search_by_keywords

Query: "Research on deep learning architectures for computer vision"
Template: search_by_keywords

Query: "papers by Kaiming He about deep learning"
Template: search_author_by_keywords

Query: "What papers discuss bioinformatics algorithms?"
Template: search_by_keywords

Query: "papers from Stanford University"
Template: search_by_institution

Query: "papers published since 2020"
Template: search_by_year_range

Query: "papers published in 2023"
Template: search_by_year

Query: "which journals does Stephen F. Altschul publish in?"
Template: author_venue_stats

Rules (use if no example above matches):
1. Paper ID mentioned (W12345) + asking about citations → search_citations
2. Paper ID mentioned (W12345) for any other question → search_by_paper_ids
3. Two or more author names + "co-author" or "together" or "collaboration" → coauthor_network
4. Author name + topic/keywords → search_author_by_keywords
5. Author name only → search_by_author
6. "most cited" or "top papers" or "highest citation" → top_cited_papers
7. "which journals/venues" + author name → author_venue_stats
8. Venue/journal/conference name mentioned → search_by_venue
9. Institution/university mentioned → search_by_institution
10. Year range (since/after/before/between) → search_by_year_range
11. Specific year → search_by_year
12. Topic, keyword, or concept search → search_by_keywords

Respond with ONLY the template name, nothing else."""

        try:
            response = await run_blocking(
                self.ai_agent.generate_content,
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
            (cypher_query, parameters, template_key) tuple
        """
        try:
            # Check Cypher cache first
            if self.cache_manager:
                cached = self.cache_manager.get_cypher(query, top_k)
                if cached is not None:
                    cypher_query, parameters = cached
                    logger.info(f"Cypher cache HIT for: {query[:50]}...")
                    return cypher_query, parameters, "cached"

            # Step 1: Extract entities from the query
            extracted = self._extract_all_entities(query)
            logger.info(f"Extracted entities: {extracted}")

            # Step 2: AI agent selects the best template
            # template_cypher can be a template KEY name (e.g. "search_by_institution")
            # or raw Cypher text passed from the API.
            if template_cypher:
                if template_cypher in self.GRAPH_TEMPLATES:
                    # It's a known template key name
                    template_key = template_cypher
                else:
                    # It's raw Cypher text — reverse-lookup the template key
                    template_key = None
                    for key, tpl in self.GRAPH_TEMPLATES.items():
                        if tpl['cypher'].strip() == template_cypher.strip():
                            template_key = key
                            break
                    if not template_key:
                        # No matching template found — use raw Cypher directly
                        logger.info("Using raw Cypher template (no matching template key found)")
                        refined_cypher, parameters = self._refine_template_with_conditions(
                            query, template_cypher, top_k
                        )
                        if self.cache_manager:
                            self.cache_manager.cache_cypher(query, top_k, refined_cypher, parameters)
                        return refined_cypher, parameters, "raw_cypher"
            else:
                template_key = await self._select_template_with_ai(query, extracted)
            logger.info(f"Selected template: {template_key}")

            # Step 3: Vector-first for keyword queries
            # Only run keyword → vector search when the selected template
            # requires $paper_ids but none were extracted from the query.
            # Templates with their own structured filters (author, venue, year,
            # keywords, etc.) don't need this expensive round-trip.
            template_cypher_text = self.GRAPH_TEMPLATES.get(template_key, {}).get('cypher', '')
            needs_paper_ids = '$paper_ids' in template_cypher_text
            has_paper_ids = bool(extracted.get('paper_ids'))

            if needs_paper_ids and not has_paper_ids:
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
                logger.info(f"Vector-first: injected {len(sorted_ids)} paper IDs for template '{template_key}'")
            else:
                if has_paper_ids:
                    logger.info(f"Skipping keyword vector search — paper IDs already extracted from query")
                else:
                    logger.info(
                        f"Skipping keyword vector search — template '{template_key}' uses its own filters (no $paper_ids needed)")

            template = self.GRAPH_TEMPLATES[template_key]
            logger.info(f"Final template: {template_key} — {template['description']}")

            # Step 4: Build parameters using the template's param builder
            param_builder = getattr(self, template['param_builder'])
            parameters = param_builder(extracted, top_k)

            cypher_query = template['cypher'].strip()

            # Step 5: Cache the result
            if self.cache_manager:
                self.cache_manager.cache_cypher(query, top_k, cypher_query, parameters)

            return cypher_query, parameters, template_key

        except Exception as e:
            logger.error(f"Error in intelligent Cypher generation: {e}")
            # Ultimate fallback — vector-first then paper IDs lookup
            paper_ids = await self._vector_first_paper_ids(query, top_k) if self.milvus_client else None
            if paper_ids:
                return self.GRAPH_TEMPLATES['search_by_paper_ids']['cypher'].strip(), \
                    {"paper_ids": paper_ids, "limit": top_k}, "search_by_paper_ids"
            keywords = self._extract_keywords(query)
            return self.GRAPH_TEMPLATES['search_by_keywords']['cypher'].strip(), \
                {"keywords": keywords or [query], "limit": top_k}, "search_by_keywords"

    async def _vector_first_paper_ids(self, query: str, top_k: int) -> List[str]:
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
            vector_results = await self.search_similar_papers(
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

            results = await run_blocking(self.graph_handler.execute_query, cypher_query, {
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
            template_info: Optional[Dict] = None,
    ) -> Optional[str]:
        """Generate AI response from vector and graph searches.

        Args:
            query: The user's natural language query
            search_results: List of search result objects or dicts
            template_info: Optional dict with 'template_key' and 'description'
                           of the graph template used for retrieval
        """
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

            # Build template-aware context for the prompt
            template_context = ""
            if template_info and template_info.get("template_key"):
                tpl_key = template_info["template_key"]
                tpl_desc = template_info.get("description", tpl_key)
                template_context = f"\nRetrieval Strategy: {tpl_key} — {tpl_desc}"

            # Build template-specific instructions
            template_instructions = self._get_template_specific_instructions(template_info)

            system_prompt = "You are a research assistant specializing in academic literature. Be accurate, concise, and tailor your response to the type of query."
            prompt = f"""Answer this question based on the search results: "{query}"
{template_context}

Search Results:
{self._format_papers_for_prompt(context_papers[:4])}

Instructions:
{template_instructions}
- Be accurate — only use information from the provided results

Answer:"""

            # Generate response using LLMClient — offload to thread pool
            ai_answer = await run_blocking(
                self.ai_agent.generate_content,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose='answer_synthesis'
            )

            if not ai_answer:
                logger.error("Empty response from Llama for AI generation")
                return None

            logger.info(f"Generated AI response of {len(ai_answer)} characters (template: {template_info})")
            return ai_answer

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None

    def _get_template_specific_instructions(self, template_info: Optional[Dict] = None) -> str:
        """Return tailored generation instructions based on the graph template used."""
        if not template_info or not template_info.get("template_key"):
            return (
                "- Answer directly in 3-5 sentences\n"
                "- Cite specific papers and authors when relevant"
            )

        tpl_key = template_info["template_key"]

        instructions_map = {
            "search_by_paper_ids": (
                "- Provide details about the requested paper(s): title, authors, venue, year, and DOI\n"
                "- Include citation count if available\n"
                "- Summarize the paper's main contribution from its abstract"
            ),
            "search_by_author": (
                "- List the papers found for the requested author(s)\n"
                "- Highlight the author's most cited or notable works\n"
                "- Mention venues and publication years"
            ),
            "search_by_keywords": (
                "- Summarize the key findings across the retrieved papers\n"
                "- Group results by sub-topic if applicable\n"
                "- Cite specific papers and authors when relevant"
            ),
            "search_citations": (
                "- List the citation relationships found\n"
                "- Identify which papers cite which, and describe the connections\n"
                "- Highlight any influential papers in the citation chain"
            ),
            "coauthor_network": (
                "- Describe the collaboration between the mentioned authors\n"
                "- List co-authored papers with titles and years\n"
                "- Note the venues where their joint work was published"
            ),
            "search_by_venue": (
                "- List papers found in the requested venue/journal/conference\n"
                "- Highlight the most cited papers from that venue\n"
                "- Mention key authors and topics"
            ),
            "search_by_institution": (
                "- List papers associated with the requested institution\n"
                "- Highlight notable authors and research areas\n"
                "- Mention publication venues and years"
            ),
            "search_by_year": (
                "- List papers from the requested year/period\n"
                "- Highlight the most impactful publications\n"
                "- Note trends or common topics in that timeframe"
            ),
            "search_by_year_range": (
                "- List papers from the requested time range\n"
                "- Highlight the most impactful publications\n"
                "- Note trends or common topics across the period"
            ),
            "search_author_by_keywords": (
                "- List the author's papers matching the requested topic\n"
                "- Summarize how the author's work relates to the topic\n"
                "- Cite specific papers with titles and years"
            ),
            "top_cited_papers": (
                "- List the most cited papers with their citation counts\n"
                "- Briefly describe each paper's contribution\n"
                "- Mention authors and venues"
            ),
            "author_venue_stats": (
                "- List the venues/journals where the author has published\n"
                "- Include paper counts per venue if available\n"
                "- Highlight the author's primary publication outlets"
            ),
        }

        return instructions_map.get(tpl_key, (
            "- Answer directly in 3-5 sentences\n"
            "- Cite specific papers and authors when relevant"
        ))


    def _format_papers_for_prompt(papers: List[Dict]) -> str:
        """Format papers for inclusion in Gemini prompt."""
        formatted_papers = []

        for i, paper in enumerate(papers, 1):
            authors = paper.get("authors", [])
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " et al."

            paper_text = f"""{i}. Title: {paper.get('title', 'N/A')}
           Authors: {authors_str or 'N/A'}
           Venue: {paper.get('venue', 'N/A')}
           Date: {paper.get('publication_date', 'N/A')}
           Relevance: {paper.get('relevance_score', 0):.2f}
           Abstract: {paper.get('abstract', 'N/A')}"""

            formatted_papers.append(paper_text)

        return "\n\n".join(formatted_papers)
    async def verify_claims_scifact(self, ai_answer: str, context_papers: List) -> List[Dict]:
        """
        Implements SciFact-style verification by breaking the AI response into
        atomic claims and checking them against the retrieved substrate.
        """
        # Step 1: Claim Extraction
        # Use a prompt to break the response into individual facts
        claims = await self.extract_atomic_claims(ai_answer)

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

            label = await run_blocking(
                self.ai_agent.generate_content, logic_prompt,
                purpose='scifact_verification'
            )
            verification_results.append({"claim": claim, "label": label})

        return verification_results

    async def extract_atomic_claims(self, ai_answer: str) -> List[str]:
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
            raw_claims = await run_blocking(
                self.ai_agent.generate_content,
                prompt=extraction_prompt, purpose='claim_extraction'
            )

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
            # Search figures collection — offload to thread pool
            if search_figures:
                figure_results = await run_blocking(
                    self.milvus_client.search_figures_by_image,
                    image_embedding, top_k
                )

            # Search tables collection — offload to thread pool
            if search_tables:
                table_results = await run_blocking(
                    self.milvus_client.search_tables_by_image,
                    image_embedding, top_k
                )

            # If text query provided, also search by description and merge results
            if text_query and self.embedding_client:
                text_embedding = await run_blocking(
                    self.embedding_client.generate_text_embedding, text_query
                )
                if text_embedding:
                    if search_figures:
                        text_fig_results = await run_blocking(
                            self.milvus_client.search_figures_by_description,
                            text_embedding, top_k
                        )
                        figure_results = self._merge_visual_results(figure_results, text_fig_results)

                    if search_tables:
                        text_tab_results = await run_blocking(
                            self.milvus_client.search_tables_by_description,
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
                        paper_results = await run_blocking(
                            self.milvus_client.collection.query,
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
