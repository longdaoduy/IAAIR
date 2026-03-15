"""
Retrieval Handler for unified search operations.

This module provides functionality to execute milvus and neo4j searches,
with intelligent query routing and AI response generation.
"""

from typing import Dict, List, Optional
import logging
import os
import re
import json
from clients.milvus.MilvusClient import MilvusClient
from clients.neo4j.Neo4jClient import Neo4jClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.huggingface.LLM_Client import LLMClient
from clients.huggingface.CLIPClient import CLIPClient
from pymilvus import (Collection)
from utils.async_utils import run_blocking

logger = logging.getLogger(__name__)


class HybridRetrievalHandler:
    """Unified handler for milvus and neo4j-based retrievals operations."""

    def __init__(self, vector_db: Optional[MilvusClient], graph_db: Neo4jClient,
                 ai_agent: Optional[LLMClient], embedder: SciBERTClient,
                 cache_manager=None, performance_monitor=None,
                 clip_client: Optional[CLIPClient] = None):
        self.milvus_client = vector_db
        self.embedding_client = embedder
        self.graph_handler = graph_db
        self.ai_agent = ai_agent
        self.answer_agent = None
        self.cache_manager = cache_manager
        self.performance_monitor = performance_monitor
        self.clip_client = clip_client

    async def execute_vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Internal milvus search implementation."""
        try:
            # Check cache first
            if self.cache_manager:
                cached_results = self.cache_manager.get_search_results(
                    query, top_k, use_hybrid=True, routing_strategy="milvus"
                )
                if cached_results is not None:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit('search', True)
                        self.performance_monitor.record_result_count('milvus', len(cached_results))
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
                    query, results, top_k, use_hybrid=True, routing_strategy="milvus"
                )

            if self.performance_monitor:
                self.performance_monitor.record_result_count('milvus', len(results or []))

            return results or []
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    async def execute_multimodal_vector_search(self, query: str, keywords: List[str],
                                                top_k: int) -> tuple:
        """Multi-modal vector search: SciBERT + CLIP cross-modal + re-ranking.

        Combines text-based vector search with cross-modal visual search to produce
        a single, re-ranked list of paper IDs. This improves ranking quality by
        boosting papers that have both textual and visual relevance.

        Flow:
        1. Run SciBERT vector search per keyword → collect paper scores
        2. Run CLIP cross-modal visual search scoped to discovered papers
        3. Re-rank paper IDs by combining vector similarity + visual scores
        4. Return re-ranked paper IDs and visual evidence metadata

        Args:
            query: Original natural language search query
            keywords: Extracted keywords for vector search
            top_k: Number of top results to return

        Returns:
            (sorted_paper_ids, visual_data) tuple where:
                sorted_paper_ids: Re-ranked list of paper IDs (best first)
                visual_data: Dict with figure_results, table_results, paper_visual_scores
        """
        empty_visual = {"figure_results": [], "table_results": [], "paper_visual_scores": {}}

        try:
            # ── Step 1: Vector search (SciBERT) per keyword ──
            all_vector_scores = {}   # paper_id → best distance (lower = better)
            all_vector_results = []

            for keyword in keywords:
                results = await self.execute_vector_search(keyword, top_k)
                all_vector_results.extend(results)

                for paper in results:
                    p_id = paper.get('paper_id')
                    dist = paper.get('distance', 0.0)
                    if p_id not in all_vector_scores or dist < all_vector_scores[p_id]:
                        all_vector_scores[p_id] = dist

            if not all_vector_scores:
                logger.warning("Multi-modal vector search: no results from SciBERT")
                return [], empty_visual

            # ── Step 2: Cross-modal visual search scoped to vector search papers ──
            vector_paper_ids = list(all_vector_scores.keys())
            visual_data = await self.search_visual_by_text(query, top_k, paper_ids=vector_paper_ids)
            paper_visual_scores = visual_data.get('paper_visual_scores', {})
            logger.info(
                f"Visual search (scoped to {len(vector_paper_ids)} papers): "
                f"{len(paper_visual_scores)} with visual evidence, "
                f"{len(visual_data.get('figure_results', []))} figures, "
                f"{len(visual_data.get('table_results', []))} tables"
            )

            # ── Step 3: Re-rank by combining vector + visual scores ──
            # Normalize vector distances to [0, 1] similarity (1 = best)
            max_dist = max(all_vector_scores.values()) if all_vector_scores else 1.0
            min_dist = min(all_vector_scores.values()) if all_vector_scores else 0.0
            dist_range = max_dist - min_dist if max_dist > min_dist else 1.0

            combined_scores = {}  # paper_id → combined score (higher = better)
            vector_weight = 0.7
            visual_rerank_weight = 0.3

            for pid, dist in all_vector_scores.items():
                vec_sim = 1.0 - ((dist - min_dist) / dist_range)  # normalize to similarity
                vis_score = paper_visual_scores.get(pid, 0.0)
                combined_scores[pid] = vector_weight * vec_sim + visual_rerank_weight * vis_score

            # Sort by combined score (descending) and take top_k
            sorted_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:top_k]

            # Store multimodal scores in visual_data so downstream can use them
            visual_data['multimodal_scores'] = combined_scores

            # Store deduplicated vector results for fallback when graph returns 0
            seen_ids = set()
            deduped_vector_results = []
            for r in all_vector_results:
                pid = r.get('paper_id')
                if pid and pid not in seen_ids and pid in set(sorted_ids):
                    seen_ids.add(pid)
                    deduped_vector_results.append(r)
            # Sort fallback results by combined_scores to preserve re-ranked order
            deduped_vector_results.sort(
                key=lambda r: combined_scores.get(r.get('paper_id'), 0), reverse=True
            )
            visual_data['vector_results'] = deduped_vector_results

            logger.info(
                f"Re-ranked paper IDs: {len(sorted_ids)} "
                f"(vector: {len(all_vector_scores)}, visual boost on: {len(paper_visual_scores)})"
            )

            return sorted_ids, visual_data

        except Exception as e:
            logger.error(f"Multi-modal vector search error: {e}")
            return [], empty_visual

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
        """Execute neo4j search using Cypher query with intelligent query parsing and caching.

        Returns:
            (results, template_info, visual_data) tuple where template_info
            is a dict with 'template_key' and 'description' of the neo4j template used,
            and visual_data is a dict with figure_results, table_results,
            and paper_visual_scores from cross-modal visual search (used for display only,
            ranking is already applied during the vector search step).
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

        # 6. Extract venue name if mentioned (skip vague/generic descriptors)
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
        # (e.g. when the query is purely keyword/topic-based and milvus-first injected paper_ids).
        default_params = self._build_default_params(refined_cypher, top_k)
        for param, default_val in default_params.items():
            if param not in parameters:
                parameters[param] = default_val
                logger.info(f"Auto-filled missing template parameter ${param} with default: {default_val}")

        logger.info(f"Extracted conditions from query: {conditions}")
        return refined_cypher, parameters

    @staticmethod
    def _inject_where_conditions(cypher: str, conditions: List[str]) -> str:
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

    @staticmethod
    def _inject_limit(cypher: str, top_k: int) -> str:
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

    @staticmethod
    def _build_default_params(cypher: str, top_k: int) -> Dict:
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
        """Internal neo4j search implementation with caching.

        Returns:
            (results, template_info, visual_data) tuple where template_info
            is a dict with 'template_key' and 'description' of the neo4j template used,
            and visual_data is a dict with figure_results, table_results,
            and paper_visual_scores (for display/evidence only — ranking already applied).
        """
        template_info = {"template_key": None, "description": None}
        empty_visual = {"figure_results": [], "table_results": [], "paper_visual_scores": {}}
        try:
            # Check cache first
            if self.cache_manager:
                cached_results = self.cache_manager.get_search_results(
                    query, top_k, use_hybrid=False, routing_strategy="neo4j"
                )
                if cached_results is not None:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit('search', True)
                        self.performance_monitor.record_result_count('neo4j', len(cached_results))
                    logger.debug(f"Graph search cache hit for: {query[:50]}...")
                    return cached_results, template_info, empty_visual

            if self.performance_monitor:
                self.performance_monitor.record_cache_hit('search', False)

            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return [], template_info, empty_visual

            # Parse the query intelligently based on patterns
            cypher_query, parameters, template_key, visual_data, keywords = await self._build_intelligent_cypher_query(query, top_k, template_cypher)

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

            # Fallback: if graph returned 0 results but multimodal vector search
            # found papers, use the vector results directly so the user still
            # gets results (with basic metadata from Milvus instead of rich
            # Neo4j metadata like authors, venue, citations).
            if not results and visual_data.get('vector_results'):
                if not visual_data.get('vector_results'):
                    sorted_ids, visual_data = await self.execute_multimodal_vector_search(
                        query, keywords, top_k
                    )
                logger.warning(
                    f"Graph returned 0 results — falling back to "
                    f"{len(visual_data['vector_results'])} vector search results"
                )
                results = visual_data['vector_results']

            # Cache results
            if self.cache_manager and results:
                self.cache_manager.cache_search_results(
                    query, results, top_k, use_hybrid=False, routing_strategy="neo4j"
                )

            if self.performance_monitor:
                self.performance_monitor.record_result_count('neo4j', len(results))

            return results, template_info, visual_data
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return [], template_info, empty_visual

    @staticmethod
    def _extract_author_names(query: str) -> List[str]:
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

    @staticmethod
    def _extract_keywords(query: str) -> List[str]:
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
        """Load neo4j query templates from data/graph_templates.json.

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
            logger.info(f"Loaded {len(cls._graph_templates_cache)} neo4j templates from {templates_path}")
        except FileNotFoundError:
            logger.error(f"Graph templates file not found: {templates_path}")
            cls._graph_templates_cache = {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in neo4j templates file: {e}")
            cls._graph_templates_cache = {}

        return cls._graph_templates_cache

    @classmethod
    def reload_graph_templates(cls):
        """Force reload templates from disk (e.g. after editing the JSON file)."""
        cls._graph_templates_cache = None
        return cls._load_graph_templates()

    @property
    def GRAPH_TEMPLATES(self) -> Dict:
        """Access neo4j templates (loaded from data/graph_templates.json)."""
        return self._load_graph_templates()

    # =========================================================================
    # TEMPLATE SELECTION — AI agent picks the best template
    # =========================================================================

    async def _paraphrase_query_as_description(self, query: str) -> str:
        """Paraphrase the user's query into a clean description format.

        Removes special characters (?, !, ...) and rephrases the query
        so it reads like a template description while preserving meaning.
        If the AI agent is unavailable, falls back to simple regex cleanup.

        Args:
            query: The user's original natural language query

        Returns:
            A cleaned, description-style version of the query
        """
        if not self.ai_agent:
            # Fallback: strip special characters only
            return re.sub(r'[?!.;:]+', '', query).strip()

        paraphrase_prompt = f"""Rephrase the following user query into a short, clean description.
Remove all special characters such as ? ! ... ; : and question marks.
Keep the same meaning. Do NOT answer the query — only rephrase it.
Output ONLY the rephrased description, nothing else.

User query: "{query}"
Rephrased description:"""

        try:
            response = await run_blocking(
                self.ai_agent.generate_content,
                prompt=paraphrase_prompt,
                system_prompt="You rephrase queries into clean descriptions. Output only the rephrased text.",
                purpose='query_paraphrase'
            )
            if response:
                paraphrased = response.strip().strip('"\'')
                # Extra safety: remove any leftover special chars
                paraphrased = re.sub(r'[?!]+', '', paraphrased).strip()
                if paraphrased:
                    logger.info(f"Paraphrased query: '{query}' → '{paraphrased}'")
                    return paraphrased
        except Exception as e:
            logger.warning(f"Query paraphrase failed: {e}")

        # Fallback: simple regex cleanup
        return re.sub(r'[?!.;:]+', '', query).strip()

    async def _select_template_with_ai(self, query: str, extracted: Dict) -> str:
        """Use AI agent to select the best neo4j template for the user's query.

        The query is first paraphrased into a clean description format (no special
        characters) to match the style of template descriptions and few-shot examples.

        Args:
            query: User's natural language query
            extracted: Pre-extracted entities {paper_ids, author_names, keywords, year, ...}

        Returns:
            Template key name from GRAPH_TEMPLATES
        """
        if not self.ai_agent:
            return self._select_template_by_rules(extracted)

        # Paraphrase the query into a description-like format

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

Here are example queries (described as clean descriptions) and their correct template:

Description: Retrieve the authors of paper W1775749144
Template: search_by_paper_ids

Description: Find papers authored by Kaiming He
Template: search_by_author

Description: Find papers that cite W2128635872
Template: search_citations

Description: Get the citation count for paper W2100837269
Template: search_by_paper_ids

Description: Find the venue that published paper W3038568908
Template: search_by_paper_ids

Description: Retrieve papers published in Nature
Template: search_by_venue

Description: Retrieve papers published in Analytical Biochemistry
Template: search_by_venue

Description: Find papers co-authored by Georg Kresse and J Furthmuller
Template: coauthor_network

Description: Find papers co-authored by Kaiming He and Jian Sun
Template: coauthor_network

Description: Find the paper with the highest citation count
Template: top_cited_papers

Description: Get the DOI of paper W1979290264
Template: search_by_paper_ids

Description: Get the publication year of paper W2107277218
Template: search_by_paper_ids

Description: Papers about protein quantification methods
Template: search_by_keywords

Description: Research on deep learning architectures for computer vision
Template: search_by_keywords

Description: Papers by Kaiming He about deep learning
Template: search_author_by_keywords

Description: Papers discussing bioinformatics algorithms
Template: search_by_keywords

Description: Papers from Stanford University
Template: search_by_institution

Description: Papers published since 2020
Template: search_by_year_range

Description: Papers published in 2023
Template: search_by_year

Description: Journals where Stephen F Altschul publishes
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

    @staticmethod
    def _select_template_by_rules(extracted: Dict) -> str:
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

    async def _extract_all_entities(self, query: str) -> Dict:
        """Extract all entities from a natural language query.

        Uses the AI agent as the primary extractor for accurate entity recognition
        (author names, venues, institutions, intents, etc.). Falls back to the
        rule-based regex extractor when the AI agent is unavailable or fails.

        Paper IDs (W12345 format) are always extracted via regex for reliability.

        Returns a dict with:
            paper_ids, author_names, keywords, year, year_from, year_to,
            venue, institution, wants_citations, wants_coauthors, wants_top_cited
        """
        # Always extract paper IDs via regex — they have a deterministic format
        extracted = {}
        paper_ids = re.findall(r'\b(W\d+)\b', query)
        if paper_ids:
            extracted['paper_ids'] = paper_ids

        # Try AI extraction first, fall back to rules
        if self.ai_agent:
            ai_extracted = await self._extract_all_entities_with_ai(query)
            if ai_extracted:
                # Merge: AI results take priority, but keep regex paper_ids
                ai_extracted.setdefault('paper_ids', [])
                if paper_ids:
                    # Combine and deduplicate paper IDs from AI + regex
                    combined_ids = list(dict.fromkeys(
                        extracted.get('paper_ids', []) + ai_extracted.get('paper_ids', [])
                    ))
                    ai_extracted['paper_ids'] = combined_ids
                logger.info(f"AI entity extraction succeeded: {ai_extracted}")
                return ai_extracted

        # Fallback to rule-based extraction
        logger.info("Falling back to rule-based entity extraction")
        return self._extract_all_entities_by_rules(query, extracted)

    async def _extract_all_entities_with_ai(self, query: str) -> Optional[Dict]:
        """Use AI agent to extract structured entities from a natural language query.

        The AI agent is better at understanding context, disambiguating author names
        from keywords, recognizing venues/institutions, and detecting user intent.

        Args:
            query: The user's natural language query

        Returns:
            Dict with extracted entities, or None if AI extraction fails
        """
        prompt = f"""You are an entity extractor for an academic paper search system.
Extract structured entities from the user's query. Return a valid JSON object.

User query: "{query}"

Extract the following fields (use null or empty list [] if not found):
{{
  "paper_ids": ["W12345", ...],       // OpenAlex paper IDs starting with W followed by digits
  "author_names": ["First Last", ...], // Full author names (first + last name minimum)
  "keywords": ["keyword1", ...],       // Research topics, concepts, methods (NOT author names, venues, or years)
  "year": "2023",                       // Specific year if mentioned (e.g. "papers in 2023")
  "year_from": "2020",                  // Start year for ranges (e.g. "since 2020", "after 2020")
  "year_to": "2024",                    // End year for ranges (e.g. "before 2024", "until 2024")
  "venue": "Nature",                    // Journal, conference, or venue name
  "institution": "Stanford University", // University or research institution
  "wants_citations": false,             // true if asking about citations or citing relationships
  "wants_coauthors": false,             // true if asking about co-authorship or collaboration
  "wants_top_cited": false              // true if asking for most cited / top / most influential papers
}}

Rules:
1. "paper_ids" are always in the format W followed by digits (e.g. W1775749144)
2. "author_names" must have at least first and last name (e.g. "Kaiming He", not just "He")
3. "keywords" are research topics/concepts — do NOT include author names, venue names, years, or common words
4. For year ranges: "since 2020" → year_from="2020", year_to="2099"; "before 2024" → year_from="1900", year_to="2024"
5. For a single year: "in 2023" → year="2023" (leave year_from and year_to as null)
6. "venue" MUST be a SPECIFIC, real journal/conference name (e.g. "Nature", "NeurIPS", "Analytical Biochemistry", "The Lancet").
   NEVER use vague/generic descriptors as venue — these are NOT valid venues:
   "top journals", "high-impact journals", "medical journals", "leading conferences",
   "prestigious venues", "top medical journals", "major conferences", "good journals".
   If the query mentions a category of venues (e.g. "medicine journals", "machine learning venues") without naming a specific one,
   set venue to null and extract ONLY the subject/topic words into keywords — strip out generic category words
   (journals, journal, conferences, conference, venues, venue, publications, outlets) and vague qualifiers
   (top, leading, major, prestigious, high-impact, best, good, notable, prominent, renowned).
   Examples: "medicine journals" → keyword "medicine"; "machine learning conferences" → keyword "machine learning";
   "top medical journals" → keyword "medical"; "high-impact AI venues" → keyword "AI".
7. "institution" MUST be a SPECIFIC, real institution name (e.g. "Stanford", "Google DeepMind", "MIT").
   NEVER use vague/generic descriptors as institution — these are NOT valid institutions:
   "top universities", "leading research labs", "major institutions".
   If no specific institution is named, set institution to null and extract ONLY the subject/topic words into keywords —
   strip out generic category words (universities, university, institutions, institution, labs, laboratory, organizations)
   and vague qualifiers (top, leading, major, prestigious, best, renowned).
   Examples: "biomedical research labs" → keyword "biomedical"; "top engineering universities" → keyword "engineering".
8. Set intent flags (wants_citations, wants_coauthors, wants_top_cited) based on the query's intent

Examples:
Query: "Find papers by Kaiming He about deep learning since 2020"
{{"paper_ids": [], "author_names": ["Kaiming He"], "keywords": ["deep learning"], "year": null, "year_from": "2020", "year_to": "2099", "venue": null, "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": false}}

Query: "What are the most cited papers about transformer architectures?"
{{"paper_ids": [], "author_names": [], "keywords": ["transformer architectures"], "year": null, "year_from": null, "year_to": null, "venue": null, "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": true}}

Query: "Papers co-authored by Georg Kresse and J Furthmuller"
{{"paper_ids": [], "author_names": ["Georg Kresse", "J Furthmuller"], "keywords": [], "year": null, "year_from": null, "year_to": null, "venue": null, "institution": null, "wants_citations": false, "wants_coauthors": true, "wants_top_cited": false}}

Query: "Find papers that cite W2128635872"
{{"paper_ids": ["W2128635872"], "author_names": [], "keywords": [], "year": null, "year_from": null, "year_to": null, "venue": null, "institution": null, "wants_citations": true, "wants_coauthors": false, "wants_top_cited": false}}

Query: "Papers published in Nature about protein folding"
{{"paper_ids": [], "author_names": [], "keywords": ["protein folding"], "year": null, "year_from": null, "year_to": null, "venue": "Nature", "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": false}}

Query: "Cancer papers published in high-impact medical journals"
{{"paper_ids": [], "author_names": [], "keywords": ["cancer", "medical"], "year": null, "year_from": null, "year_to": null, "venue": null, "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": false}}

Query: "Papers published in medicine journals"
{{"paper_ids": [], "author_names": [], "keywords": ["medicine"], "year": null, "year_from": null, "year_to": null, "venue": null, "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": false}}

Query: "Machine learning papers from top AI conferences"
{{"paper_ids": [], "author_names": [], "keywords": ["machine learning", "AI"], "year": null, "year_from": null, "year_to": null, "venue": null, "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": false}}

Query: "Deep learning research from top engineering universities"
{{"paper_ids": [], "author_names": [], "keywords": ["deep learning", "engineering"], "year": null, "year_from": null, "year_to": null, "venue": null, "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": false}}

Query: "Papers about CRISPR published in Cell"
{{"paper_ids": [], "author_names": [], "keywords": ["CRISPR"], "year": null, "year_from": null, "year_to": null, "venue": "Cell", "institution": null, "wants_citations": false, "wants_coauthors": false, "wants_top_cited": false}}

Return ONLY the JSON object, nothing else."""

        try:
            response = await run_blocking(
                self.ai_agent.generate_content,
                prompt=prompt,
                system_prompt="You are a precise entity extractor. Return only valid JSON.",
                purpose='entity_extraction'
            )

            if not response:
                return None

            # Clean response — strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith('```'):
                # Remove ```json ... ``` wrapping
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            cleaned = cleaned.strip()

            ai_result = json.loads(cleaned)

            # Validate and normalize the AI output
            extracted = {}

            # Paper IDs
            pids = ai_result.get('paper_ids', [])
            if pids and isinstance(pids, list):
                # Validate format: must match W followed by digits
                valid_pids = [p for p in pids if isinstance(p, str) and re.match(r'^W\d+$', p)]
                if valid_pids:
                    extracted['paper_ids'] = valid_pids

            # Author names
            authors = ai_result.get('author_names', [])
            if authors and isinstance(authors, list):
                valid_authors = [a for a in authors if isinstance(a, str) and len(a.split()) >= 2]
                if valid_authors:
                    extracted['author_names'] = valid_authors

            # Keywords
            kws = ai_result.get('keywords', [])
            if kws and isinstance(kws, list):
                valid_kws = [k for k in kws if isinstance(k, str) and len(k.strip()) > 1]
                if valid_kws:
                    extracted['keywords'] = valid_kws

            # Year (single)
            year = ai_result.get('year')
            if year and isinstance(year, str) and re.match(r'^\d{4}$', year):
                extracted['year'] = year

            # Year range
            year_from = ai_result.get('year_from')
            year_to = ai_result.get('year_to')
            if year_from and isinstance(year_from, str) and re.match(r'^\d{4}$', year_from):
                extracted['year_from'] = year_from
                extracted['year_to'] = year_to if (year_to and isinstance(year_to, str) and re.match(r'^\d{4}$', year_to)) else '2099'

            # Venue — reject vague/generic descriptors
            venue = ai_result.get('venue')
            if venue and isinstance(venue, str) and len(venue.strip()) > 1:
                venue_clean = venue.strip()
                extracted['venue'] = venue_clean

            # Institution — reject vague/generic descriptors
            institution = ai_result.get('institution')
            if institution and isinstance(institution, str) and len(institution.strip()) > 1:
                inst_clean = institution.strip()
                extracted['institution'] = inst_clean

            # Intent flags
            if ai_result.get('wants_citations') is True:
                extracted['wants_citations'] = True
            if ai_result.get('wants_coauthors') is True:
                extracted['wants_coauthors'] = True
            if ai_result.get('wants_top_cited') is True:
                extracted['wants_top_cited'] = True

            return extracted

        except json.JSONDecodeError as e:
            logger.warning(f"AI entity extraction returned invalid JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"AI entity extraction failed: {e}")
            return None

    def _extract_all_entities_by_rules(self, query: str, extracted: Optional[Dict] = None) -> Dict:
        """Rule-based fallback for entity extraction when AI is unavailable.

        Uses regex patterns to extract entities from the query. This is the
        original extraction logic, now serving as a reliable backup.

        Args:
            query: The user's natural language query
            extracted: Optional pre-populated dict (e.g. with paper_ids already set)

        Returns a dict with:
            paper_ids, author_names, keywords, year, year_from, year_to,
            venue, institution, wants_citations, wants_coauthors, wants_top_cited
        """
        if extracted is None:
            extracted = {}
        query_lower = query.lower()

        # 1. Paper IDs (if not already set)
        if not extracted.get('paper_ids'):
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

        # 4. Venue — reject vague/generic descriptors
        venue_match = re.search(
            r'(?:in|from|published\s+in|venue|journal|conference)\s+["\']?([A-Z][^"\',]{2,40})["\']?',
            query, re.IGNORECASE
        )
        if venue_match:
            venue_name = venue_match.group(1).strip()
            extracted['venue'] = venue_name

        # 5. Institution — reject vague/generic descriptors
        inst_match = re.search(
            r'(?:from|at|institution|university|org(?:anization)?)\s+["\']?([A-Z][^"\',]{2,50})["\']?',
            query, re.IGNORECASE
        )
        if inst_match and not extracted.get('venue'):  # avoid conflict with venue
            inst_name = inst_match.group(1).strip()
            extracted['institution'] = inst_name

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
    # BUILD INTELLIGENT CYPHER — Main entry point for neo4j search
    # =========================================================================

    async def _build_intelligent_cypher_query(self, query: str, top_k: int, template_cypher: str = None) -> tuple:
        """Build Cypher query by selecting the best template and filling parameters.

        Flow:
        1. Check Cypher cache
        2. Extract entities from query (paper IDs, authors, keywords, years, etc.)
        3. AI agent selects the best template (fallback: rule-based selection)
        4. If template needs paper IDs → run multi-modal vector search
           (SciBERT + CLIP cross-modal + re-ranking) to produce ranked paper IDs
        5. Fill template parameters from extracted entities
        6. Cache and return

        Returns:
            (cypher_query, parameters, template_key, visual_data) tuple
            where visual_data is a dict with figure_results, table_results,
            paper_visual_scores (for display/evidence — ranking already applied)
        """
        empty_visual = {"figure_results": [], "table_results": [], "paper_visual_scores": {}}
        try:
            # Check Cypher cache first
            if self.cache_manager:
                cached = self.cache_manager.get_cypher(query, top_k)
                if cached is not None:
                    cypher_query, parameters = cached
                    logger.info(f"Cypher cache HIT for: {query[:50]}...")
                    return cypher_query, parameters, "cached", empty_visual, []

            paraphrased_query = await self._paraphrase_query_as_description(query)

            # Step 1: Extract entities from the query (AI-first, rule-based fallback)
            extracted = await self._extract_all_entities(paraphrased_query)
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
                            paraphrased_query, template_cypher, top_k
                        )
                        if self.cache_manager:
                            self.cache_manager.cache_cypher(query, top_k, refined_cypher, parameters)
                        return refined_cypher, parameters, "raw_cypher", empty_visual, []
            else:
                template_key = await self._select_template_with_ai(paraphrased_query, extracted)
            if template_key == 'search_by_keywords':
                template_key = 'search_by_paper_ids'
            logger.info(f"Selected template: {template_key}")

            # Step 3: Multi-modal vector search for keyword queries
            # Only run when the selected template requires $paper_ids but none
            # were extracted from the query. The multi-modal search combines
            # SciBERT vector search + CLIP cross-modal search + re-ranking
            # to produce the best-ranked paper IDs.
            template_cypher_text = self.GRAPH_TEMPLATES.get(template_key, {}).get('cypher', '')
            needs_paper_ids = '$paper_ids' in template_cypher_text
            has_paper_ids = bool(extracted.get('paper_ids'))

            visual_data = empty_visual
            keywords = extracted.get('keywords', [])

            if (needs_paper_ids and not has_paper_ids) or template_key == 'search_by_keywords':
                if template_key == 'search_by_keywords':
                    template_key = 'search_by_paper_ids'

                # Run multi-modal vector search (SciBERT + CLIP + re-ranking)
                sorted_ids, visual_data = await self.execute_multimodal_vector_search(
                    query, keywords, top_k
                )

                # Inject re-ranked paper IDs into extracted entities for the Cypher query
                extracted.setdefault('paper_ids', []).extend(sorted_ids)
                logger.info(f"Multi-modal search: injected {len(sorted_ids)} re-ranked paper IDs for template '{template_key}'")
            else:
                # Still run visual search scoped to existing paper IDs for evidence enrichment
                existing_pids = extracted.get('paper_ids', [])
                visual_data = await self.search_visual_by_text(query, top_k, paper_ids=existing_pids)

                if has_paper_ids:
                    logger.info(f"Skipping keyword milvus search — paper IDs already extracted from query")
                else:
                    logger.info(
                        f"Skipping keyword milvus search — template '{template_key}' uses its own filters (no $paper_ids needed)")

            template = self.GRAPH_TEMPLATES[template_key]
            logger.info(f"Final template: {template_key} — {template['description']}")

            # Step 4: Build parameters using the template's param builder
            param_builder = getattr(self, template['param_builder'])
            parameters = param_builder(extracted, top_k)

            cypher_query = template['cypher'].strip()

            # Step 5: Cache the result
            if self.cache_manager:
                self.cache_manager.cache_cypher(paraphrased_query, top_k, cypher_query, parameters)

            return cypher_query, parameters, template_key, visual_data, keywords

        except Exception as e:
            logger.error(f"Error in intelligent Cypher generation: {e}")
            paraphrased_query = await self._paraphrase_query_as_description(query)

            # Ultimate fallback — milvus-first then paper IDs lookup
            paper_ids = await self._vector_first_paper_ids(paraphrased_query, top_k) if self.milvus_client else None
            if paper_ids:
                return self.GRAPH_TEMPLATES['search_by_paper_ids']['cypher'].strip(), \
                    {"paper_ids": paper_ids, "limit": top_k}, "search_by_paper_ids", empty_visual, []
            keywords = self._extract_keywords(paraphrased_query)
            return self.GRAPH_TEMPLATES['search_by_keywords']['cypher'].strip(), \
                {"keywords": keywords or [paraphrased_query], "limit": top_k}, "search_by_keywords", empty_visual, keywords

    async def _vector_first_paper_ids(self, query: str, top_k: int) -> List[str]:
        """Run milvus search to extract relevant paper IDs for neo4j enrichment.

        This is the core of the milvus-first strategy: use semantic similarity
        to find relevant papers, then pass their IDs to a neo4j template for
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
                logger.warning("Vector-first: no results from milvus search")
                return []

            paper_ids = [r.get('paper_id') for r in vector_results]

            logger.info(f"Vector-first: extracted {len(paper_ids)} paper IDs from milvus search")
            return paper_ids

        except Exception as e:
            logger.error(f"Vector-first search failed: {e}")
            return []

    # async def _execute_graph_refinement(self, paper_ids: List[str], top_k: int) -> List[Dict]:
    #     """Refine milvus results using neo4j relationships."""
    #     try:
    #         if not self.graph_handler:
    #             logger.error("Graph handler not initialized")
    #             return []
    #
    #         # Find related papers through citations and collaborations
    #         cypher_query = """
    #             MATCH (p:Paper)
    #             WHERE p.id IN $paper_ids
    #             OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
    #             OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
    #             RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
    #                    p.doi as doi, p.publication_date as publication_date,
    #                    p.cited_by_count as cited_by_count,
    #                    collect(DISTINCT a.name) as authors,
    #                    v.name as venue
    #             ORDER BY p.id
    #         """
    #
    #         logger.info(paper_ids)
    #
    #         results = await run_blocking(self.graph_handler.execute_query, cypher_query, {
    #             "paper_ids": paper_ids,
    #             "top_k": top_k
    #         })
    #
    #         return results
    #     except Exception as e:
    #         logger.error(f"Graph refinement error: {e}")
    #         return []

    async def generate_ai_response(
            self,
            query: str,
            search_results: List,
            template_info: Optional[Dict] = None,
            visual_evidence: Optional[Dict] = None,
    ) -> Optional[str]:
        """Generate AI response from milvus, neo4j, and cross-modal visual searches.

        Args:
            query: The user's natural language query
            search_results: List of search result objects or dicts
            template_info: Optional dict with 'template_key' and 'description'
                           of the neo4j template used for retrievals
            visual_evidence: Optional dict with 'figure_results', 'table_results',
                             and 'paper_visual_scores' from cross-modal visual search
        """
        try:
            if not self.answer_agent:
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
                        "doi": getattr(result, "doi", "") or "",
                        "matched_figures": getattr(result, "matched_figures", 0),
                        "matched_tables": getattr(result, "matched_tables", 0),
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
                        "doi": result.get("doi", "") or "",
                        "matched_figures": result.get("matched_figures", 0),
                        "matched_tables": result.get("matched_tables", 0),
                    })

            # Build template-aware context for the prompt
            template_context = ""
            if template_info and template_info.get("template_key"):
                tpl_key = template_info["template_key"]
                tpl_desc = template_info.get("description", tpl_key)
                template_context = f"\nRetrieval Strategy: {tpl_key} — {tpl_desc}"

            # Build visual evidence context for the prompt
            visual_context = ""
            if visual_evidence:
                figs = visual_evidence.get('figure_results', [])
                tabs = visual_evidence.get('table_results', [])
                if figs or tabs:
                    visual_context = "\n\nVisual Evidence Retrieved:"
                    for i, fig in enumerate(figs[:5], 1):
                        desc = (fig.get('description', '') or '')[:150]
                        pid = fig.get('paper_id', '')
                        score = fig.get('similarity_score', 0)
                        visual_context += f"\n  Figure {i} (paper {pid}, relevance {score:.2f}): {desc}"
                    for i, tab in enumerate(tabs[:5], 1):
                        desc = (tab.get('description', '') or '')[:150]
                        pid = tab.get('paper_id', '')
                        score = tab.get('similarity_score', 0)
                        visual_context += f"\n  Table {i} (paper {pid}, relevance {score:.2f}): {desc}"

            # Build template-specific instructions
            template_instructions = self._get_template_specific_instructions(template_info)

            # Add visual evidence instructions if visual results are present
            if visual_context:
                template_instructions += (
                    "\n- Reference relevant figures and tables when they support the answer"
                    "\n- Mention which paper each visual element comes from"
                    "\n- If the query is about experimental results, graphs, or data, emphasize the visual evidence"
                )

            system_prompt = "You are a research assistant specializing in academic literature. Be accurate, concise, and tailor your response to the type of query."
            prompt = f"""Answer this question based on the search results: "{query}"
{template_context}

Search Results:
{self._format_papers_for_prompt(context_papers[:4])}
{visual_context}

Instructions:
{template_instructions}
- Be accurate — only use information from the provided results

Answer:"""

            # Generate response using LLMClient — offload to thread pool
            ai_answer = await run_blocking(
                self.answer_agent.generate_content,
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
        """Return tailored generation instructions based on the neo4j template used."""
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
            # Compare the claim against the 'top k' abstracts and neo4j metadata
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

    @staticmethod
    def _format_papers_for_prompt(papers: List[Dict]) -> str:
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
    # CROSS-MODAL VISUAL SEARCH (text query → figure/table retrieval)
    # =========================================================================

    async def search_visual_by_text(self, query: str, top_k: int = 5,
                                      paper_ids: Optional[List[str]] = None) -> Dict:
        """Cross-modal search: find figures and tables for papers found by vector search.

        Visual search is scoped to the paper IDs already discovered by vector search.
        This ensures visual evidence is only retrieved for papers the system has
        already identified as relevant, avoiding noise from unrelated papers.

        Flow:
        1. Generates a CLIP text embedding from the query
        2. Searches figures_collection and tables_collection by image_embedding similarity
        3. Also searches by description_embedding (SciBERT) for description-level matches
        4. Filters results to keep only figures/tables belonging to the given paper_ids
        5. Returns visual results + a per-paper visual score boost

        Args:
            query: Natural language search query
            top_k: Max figures + tables to return per collection
            paper_ids: List of paper IDs from vector search to scope visual results to.
                       If None or empty, returns empty results.

        Returns:
            Dict with:
                figure_results: list of matched figures (scoped to paper_ids)
                table_results:  list of matched tables (scoped to paper_ids)
                paper_visual_scores: {paper_id: float} mapping of visual relevance boosts
        """
        empty_result = {"figure_results": [], "table_results": [], "paper_visual_scores": {}}
        figure_results = []
        table_results = []
        paper_visual_scores: Dict[str, float] = {}

        try:
            if not paper_ids:
                logger.debug("No paper_ids provided — skipping visual search")
                return empty_result

            if not self.clip_client and not self.embedding_client:
                logger.debug("No CLIP or SciBERT client — skipping cross-modal visual search")
                return empty_result

            # Build the set of allowed paper IDs for filtering
            allowed_pids = set(paper_ids)
            # Search with a larger pool to have enough results after filtering
            search_k = top_k * 3

            # 1. CLIP text → image embedding search (cross-modal)
            clip_text_embedding = None
            if self.clip_client:
                clip_text_embedding = await run_blocking(
                    self.clip_client.generate_text_embedding, query
                )

            if clip_text_embedding:
                # Search figures by CLIP text embedding against image_embedding field
                try:
                    clip_fig = await run_blocking(
                        self.milvus_client.search_figures_by_image,
                        clip_text_embedding, search_k
                    )
                    figure_results.extend(clip_fig)
                except Exception as e:
                    logger.debug(f"CLIP figure search failed: {e}")

                # Search tables by CLIP text embedding against image_embedding field
                try:
                    clip_tab = await run_blocking(
                        self.milvus_client.search_tables_by_image,
                        clip_text_embedding, search_k
                    )
                    table_results.extend(clip_tab)
                except Exception as e:
                    logger.debug(f"CLIP table search failed: {e}")

            # 2. SciBERT description embedding search (text-to-text within visual collections)
            if self.embedding_client:
                desc_embedding = None
                if self.cache_manager:
                    desc_embedding = self.cache_manager.get_embedding(query)
                if desc_embedding is None:
                    desc_embedding = await run_blocking(
                        self.embedding_client.generate_text_embedding, query
                    )
                    if self.cache_manager and desc_embedding:
                        self.cache_manager.cache_embedding(query, desc_embedding)

                if desc_embedding:
                    try:
                        desc_fig = await run_blocking(
                            self.milvus_client.search_figures_by_description,
                            desc_embedding, search_k
                        )
                        # Merge CLIP + description results
                        figure_results = self._merge_visual_results(figure_results, desc_fig)
                    except Exception as e:
                        logger.debug(f"Description figure search failed: {e}")

                    try:
                        desc_tab = await run_blocking(
                            self.milvus_client.search_tables_by_description,
                            desc_embedding, search_k
                        )
                        table_results = self._merge_visual_results(table_results, desc_tab)
                    except Exception as e:
                        logger.debug(f"Description table search failed: {e}")

            # 3. Filter results to only keep figures/tables belonging to vector search papers
            # Debug: log paper_ids from visual results vs allowed paper_ids
            visual_fig_pids = set(r.get('paper_id') for r in figure_results if r.get('paper_id'))
            visual_tab_pids = set(r.get('paper_id') for r in table_results if r.get('paper_id'))
            logger.info(
                f"Visual filter debug — "
                f"allowed_pids ({len(allowed_pids)}): {list(allowed_pids)[:5]}, "
                f"figure paper_ids ({len(visual_fig_pids)}): {list(visual_fig_pids)[:5]}, "
                f"table paper_ids ({len(visual_tab_pids)}): {list(visual_tab_pids)[:5]}, "
                f"intersection figs: {visual_fig_pids & allowed_pids}, "
                f"intersection tabs: {visual_tab_pids & allowed_pids}"
            )
            figure_results = [r for r in figure_results if r.get('paper_id') in allowed_pids]
            table_results = [r for r in table_results if r.get('paper_id') in allowed_pids]

            # 4. Build per-paper visual scores from matched visual evidence
            for r in figure_results + table_results:
                pid = r.get("paper_id")
                score = r.get("similarity_score", 0.0)
                if pid:
                    # Aggregate: keep the max visual score per paper
                    paper_visual_scores[pid] = max(paper_visual_scores.get(pid, 0.0), score)

            logger.info(
                f"Cross-modal visual search (scoped to {len(allowed_pids)} papers): "
                f"{len(figure_results)} figures, {len(table_results)} tables, "
                f"{len(paper_visual_scores)} papers with visual evidence"
            )

            return {
                "figure_results": figure_results[:top_k],
                "table_results": table_results[:top_k],
                "paper_visual_scores": paper_visual_scores
            }

        except Exception as e:
            logger.error(f"Cross-modal visual search error: {e}")
            return empty_result
    #
    # # =========================================================================
    # # IMAGE SEARCH METHODS
    # # =========================================================================
    #
    # async def search_by_image(self, image_embedding: List[float], top_k: int = 10,
    #                           search_figures: bool = True, search_tables: bool = True,
    #                           text_query: Optional[str] = None) -> Dict:
    #     """Search figures and tables collections by image embedding.
    #
    #     Args:
    #         image_embedding: CLIP image embedding milvus
    #         top_k: Number of results per collection
    #         search_figures: Whether to search figures collection
    #         search_tables: Whether to search tables collection
    #         text_query: Optional text query for hybrid image+text search
    #
    #     Returns:
    #         Dict with figure_results, table_results, and related_papers
    #     """
    #     figure_results = []
    #     table_results = []
    #
    #     try:
    #         # Search figures collection — offload to thread pool
    #         if search_figures:
    #             figure_results = await run_blocking(
    #                 self.milvus_client.search_figures_by_image,
    #                 image_embedding, top_k
    #             )
    #
    #         # Search tables collection — offload to thread pool
    #         if search_tables:
    #             table_results = await run_blocking(
    #                 self.milvus_client.search_tables_by_image,
    #                 image_embedding, top_k
    #             )
    #
    #         # If text query provided, also search by description and merge results
    #         if text_query and self.embedding_client:
    #             text_embedding = await run_blocking(
    #                 self.embedding_client.generate_text_embedding, text_query
    #             )
    #             if text_embedding:
    #                 if search_figures:
    #                     text_fig_results = await run_blocking(
    #                         self.milvus_client.search_figures_by_description,
    #                         text_embedding, top_k
    #                     )
    #                     figure_results = self._merge_visual_results(figure_results, text_fig_results)
    #
    #                 if search_tables:
    #                     text_tab_results = await run_blocking(
    #                         self.milvus_client.search_tables_by_description,
    #                         text_embedding, top_k
    #                     )
    #                     table_results = self._merge_visual_results(table_results, text_tab_results)
    #
    #         # Collect related paper IDs from visual results
    #         paper_ids = set()
    #         for r in figure_results + table_results:
    #             if r.get("paper_id"):
    #                 paper_ids.add(r["paper_id"])
    #
    #         # Fetch related paper details — prefer Neo4j for full metadata
    #         related_papers = []
    #         paper_ids_list = list(paper_ids)[:20]
    #
    #         # if paper_ids_list and self.graph_handler:
    #         #     try:
    #         #         related_papers = await self._execute_graph_refinement(paper_ids_list, len(paper_ids_list))
    #         #         logger.info(f"Fetched {len(related_papers)} related papers from Neo4j")
    #         #     except Exception as e:
    #         #         logger.warning(f"Neo4j paper enrichment failed, falling back to Milvus: {e}")
    #         #         related_papers = []
    #
    #         # Fallback to Milvus if Neo4j returned nothing
    #         if not related_papers and paper_ids_list and self.milvus_client.collection:
    #             for pid in paper_ids_list:
    #                 try:
    #                     paper_results = await run_blocking(
    #                         self.milvus_client.collection.query,
    #                         expr=f'id == "{pid}"',
    #                         output_fields=["id", "title", "abstract"]
    #                     )
    #                     if paper_results:
    #                         p = paper_results[0]
    #                         related_papers.append({
    #                             "paper_id": p.get("id", ""),
    #                             "title": p.get("title", ""),
    #                             "abstract": p.get("abstract", ""),
    #                             "authors": [],
    #                             "venue": None,
    #                             "doi": None,
    #                             "publication_date": None,
    #                             "cited_by_count": 0
    #                         })
    #                 except Exception as e:
    #                     logger.debug(f"Could not fetch paper {pid}: {e}")
    #
    #         # Attach visual match counts per paper
    #         paper_visual_counts = {}
    #         for r in figure_results + table_results:
    #             pid = r.get("paper_id")
    #             if pid:
    #                 paper_visual_counts.setdefault(pid, {"figures": 0, "tables": 0})
    #                 if r.get("collection") == "figures" or r.get("search_type", "").startswith("figure"):
    #                     paper_visual_counts[pid]["figures"] += 1
    #                 else:
    #                     paper_visual_counts[pid]["tables"] += 1
    #
    #         for paper in related_papers:
    #             pid = paper.get("paper_id", paper.get("id", ""))
    #             counts = paper_visual_counts.get(pid, {"figures": 0, "tables": 0})
    #             paper["matched_figures"] = counts["figures"]
    #             paper["matched_tables"] = counts["tables"]
    #             # Add a relevance score based on visual match density
    #             paper.setdefault("relevance_score", 0.0)
    #             if paper["relevance_score"] == 0:
    #                 paper["relevance_score"] = min(1.0, (counts["figures"] + counts["tables"]) * 0.15)
    #
    #         # Sort papers by number of visual matches
    #         related_papers.sort(
    #             key=lambda p: p.get("matched_figures", 0) + p.get("matched_tables", 0),
    #             reverse=True
    #         )
    #         logger.info(related_papers)
    #         return {
    #             "figure_results": figure_results[:top_k],
    #             "table_results": table_results[:top_k],
    #             "related_papers": related_papers
    #         }
    #
    #     except Exception as e:
    #         logger.error(f"Image search error: {e}")
    #         return {
    #             "figure_results": figure_results,
    #             "table_results": table_results,
    #             "related_papers": []
    #         }

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
