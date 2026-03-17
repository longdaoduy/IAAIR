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
        self.answer_agent = LLMClient()
        self.template_agent = LLMClient()
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
            all_vector_scores = {}  # paper_id → best distance (lower = better)
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
                    query, top_k, use_hybrid=False, routing_strategy="graph"
                )
                if cached_results is not None:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit('search', True)
                        self.performance_monitor.record_result_count('neo4j', len(cached_results))
                    logger.debug(f"Graph search cache hit for: {query[:50]}...")
                    # Still run visual search on cached paper IDs so figures/tables show up
                    cached_pids = [r.get('paper_id') for r in cached_results if r.get('paper_id')]
                    visual_data = empty_visual
                    if cached_pids:
                        visual_data = await self.search_visual_by_text(query, top_k, paper_ids=cached_pids)
                    return cached_results, template_info, visual_data

            if self.performance_monitor:
                self.performance_monitor.record_cache_hit('search', False)

            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return [], template_info, empty_visual

            # Check Cypher cache first
            if self.cache_manager:
                cached = self.cache_manager.get_cypher(query, top_k)
                if cached is not None:
                    cypher_query, parameters = cached
                    logger.info(f"Cypher cache HIT for: {query[:50]}...")
                    # Execute the cached Cypher and return proper (results, template_info, visual_data)
                    results = await run_blocking(self.graph_handler.execute_query, cypher_query, parameters)
                    cached_pids = [r.get('paper_id') for r in results if r.get('paper_id')]
                    visual_data = empty_visual
                    if cached_pids:
                        visual_data = await self.search_visual_by_text(query, top_k, paper_ids=cached_pids)
                    return results, template_info, visual_data

            # Step 1: Extract entities — AI first, regex fallback
            # AI is much better at understanding natural language structure
            # (e.g. distinguishing author names from institutions).
            # Regex is kept as a fallback and to catch paper_ids / intent flags
            # that the AI might miss.
            extracted = {}
            if self.template_agent:
                try:
                    extracted = await self._extract_entities_with_ai(query)
                except Exception as e:
                    logger.warning(f"AI entity extraction failed: {e}")

            # Always run regex to catch paper IDs and intent flags reliably
            regex_extracted = self._extract_entities_regex(query)

            # Merge: AI wins for semantic fields, regex fills gaps
            for key, value in regex_extracted.items():
                if key not in extracted or not extracted[key]:
                    extracted[key] = value

            # Ensure keywords are always present (from regex at minimum)
            if 'keywords' not in extracted or not extracted['keywords']:
                extracted['keywords'] = regex_extracted.get('keywords', self._extract_keywords(query))

            extracted = self._normalize_extracted(extracted)
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
                        return refined_cypher, parameters, "raw_cypher", []
            else:
                template_key = await self._select_template(query, extracted)

            logger.info(f"Selected template: {template_key}")

            # Step 3: Vector-first for keyword queries
            # Only run keyword → vector search when the selected template
            # requires $paper_ids but none were extracted from the query.
            # Templates with their own structured filters (author, venue, year,
            # keywords, etc.) don't need this expensive round-trip.
            template_cypher_text = self.GRAPH_TEMPLATES.get(template_key, {}).get('cypher', '')
            needs_paper_ids = '$paper_ids' in template_cypher_text
            has_paper_ids = bool(extracted.get('paper_ids'))
            keywords = extracted.get('keywords', [])
            template = self.GRAPH_TEMPLATES[template_key]
            logger.info(f"Final template: {template_key} — {template['description']}")

            # Step 4: Build parameters using the template's param builder
            param_builder = getattr(self, template['param_builder'])
            parameters = param_builder(extracted, top_k)

            # Build template info
            template_desc = self.GRAPH_TEMPLATES.get(template_key, {}).get('description', template_key)
            template_info = {"template_key": template_key, "description": template_desc}

            cypher_query = template['cypher'].strip()

            if needs_paper_ids and not has_paper_ids:
                # Vector-first: use multimodal search to get re-ranked paper IDs + visual data
                sorted_ids, visual_data = await self.execute_multimodal_vector_search(query, keywords, top_k)

                # Inject discovered paper IDs into extracted entities and rebuild parameters
                extracted.setdefault('paper_ids', []).extend(sorted_ids)
                parameters = param_builder(extracted, top_k)

                logger.info(f"Vector-first: injected {len(sorted_ids)} paper IDs for template '{template_key}'")

                # Execute graph query with fallback to vector results on failure
                try:
                    results = await run_blocking(self.graph_handler.execute_query, cypher_query, parameters)
                    logger.info(f"Graph search returned {len(results)} results")
                except Exception as e:
                    logger.warning(f"Graph query failed ({e}), falling back to vector results")
                    results = []

                # If graph returned 0 results, fall back to vector results
                if not results and visual_data.get('vector_results'):
                    results = visual_data['vector_results'][:top_k]
                    logger.info(f"Graph returned 0 results, falling back to {len(results)} vector results")

                # Update visual data with final graph paper IDs
                graph_paper_ids = [r.get('paper_id') for r in results if r.get('paper_id')]
                if graph_paper_ids:
                    visual_data = await self.search_visual_by_text(query, top_k, paper_ids=graph_paper_ids)

            elif template_key != 'search_by_keywords':
                # Graph-first: run graph query, then use vector search to re-rank
                try:
                    results = await run_blocking(self.graph_handler.execute_query, cypher_query, parameters)
                    logger.info(f"Graph search returned {len(results)} results")
                except Exception as e:
                    logger.warning(f"Graph query failed ({e}), falling back to vector search")
                    results = []

                paper_ids = [r.get('paper_id') for r in results if r.get('paper_id')]

                # If graph failed or returned 0, fall back to keyword vector search
                if not results:
                    # Search each keyword independently and merge by best distance
                    vector_score_map = {}  # paper_id → best distance
                    all_vector_results = []
                    for keyword in keywords:
                        kw_results = await self._execute_vector_search_internal(keyword, top_k)
                        all_vector_results.extend(kw_results)
                        for paper in kw_results:
                            p_id = paper.get('paper_id')
                            dist = paper.get('distance', 0.0)
                            if p_id not in vector_score_map or dist < vector_score_map[p_id]:
                                vector_score_map[p_id] = dist

                    # Deduplicate and sort by distance (ascending = most similar)
                    seen = set()
                    deduped = []
                    for r in all_vector_results:
                        pid = r.get('paper_id')
                        if pid and pid not in seen:
                            seen.add(pid)
                            deduped.append(r)
                    deduped.sort(key=lambda r: vector_score_map.get(r.get('paper_id'), float('inf')))
                    paper_ids = [r.get('paper_id') for r in deduped[:top_k] if r.get('paper_id')]

                    # Enrich with graph metadata via search_by_paper_ids
                    if paper_ids:
                        try:
                            paper_ids_template = self.GRAPH_TEMPLATES.get('search_by_paper_ids', {})
                            if paper_ids_template:
                                enrich_cypher = paper_ids_template['cypher'].strip()
                                enrich_params = self._params_paper_ids({'paper_ids': paper_ids}, len(paper_ids))
                                results = await run_blocking(
                                    self.graph_handler.execute_query, enrich_cypher, enrich_params
                                )
                                # Re-sort enriched results by vector distance
                                results.sort(
                                    key=lambda r: vector_score_map.get(r.get('paper_id'), float('inf'))
                                )
                                logger.info(f"Vector fallback: enriched {len(results)} papers via search_by_paper_ids")
                        except Exception as e:
                            logger.warning(f"Graph enrichment failed ({e}), using raw vector results")
                            results = deduped[:top_k]

                    if not results:
                        results = deduped[:top_k]

                    logger.info(f"Keyword vector fallback returned {len(results)} results")
                elif paper_ids:
                    # Re-rank graph results using vector similarity scores
                    vector_scores = {}
                    for pid in paper_ids:
                        vector_results = await self._execute_vector_search_internal(pid, top_k)
                        for paper in vector_results:
                            p_id = paper.get('paper_id')
                            dist = paper.get('distance', 0.0)
                            if p_id not in vector_scores or dist < vector_scores[p_id]:
                                vector_scores[p_id] = dist

                    # Re-sort results by vector distance (ascending = more similar)
                    if vector_scores:
                        results.sort(
                            key=lambda r: vector_scores.get(r.get('paper_id'), float('inf'))
                        )
                        results = results[:top_k]

                visual_data = await self.search_visual_by_text(query, top_k, paper_ids=paper_ids)
            else:
                # search_by_keywords: run both graph and vector, merge and re-rank
                try:
                    graph_results = await run_blocking(self.graph_handler.execute_query, cypher_query, parameters)
                except Exception as e:
                    logger.warning(f"Graph query failed ({e}), continuing with vector-only results")
                    graph_results = []

                graph_paper_ids = [r.get('paper_id') for r in graph_results if r.get('paper_id')]
                logger.info(f"Graph search returned {len(graph_results)} results")

                # Vector search on the original query
                vector_results = await self._execute_vector_search_internal(query=query, top_k=top_k)
                vector_paper_ids = [r.get('paper_id') for r in vector_results if r.get('paper_id')]

                # Enrich vector results with graph metadata via search_by_paper_ids template
                # so they have the same schema as graph results (authors, venue, doi, etc.)
                vector_enriched = []
                vector_only_ids = [pid for pid in vector_paper_ids if pid not in set(graph_paper_ids)]
                if vector_only_ids:
                    try:
                        paper_ids_template = self.GRAPH_TEMPLATES.get('search_by_paper_ids', {})
                        if paper_ids_template:
                            enrich_cypher = paper_ids_template['cypher'].strip()
                            enrich_params = self._params_paper_ids({'paper_ids': vector_only_ids}, len(vector_only_ids))
                            vector_enriched = await run_blocking(
                                self.graph_handler.execute_query, enrich_cypher, enrich_params
                            )
                            logger.info(f"Enriched {len(vector_enriched)} vector-only papers via search_by_paper_ids")
                    except Exception as e:
                        logger.warning(f"Graph enrichment failed ({e}), using raw vector results")
                        # Fall back to raw vector results (without graph metadata)
                        vector_enriched = [r for r in vector_results if r.get('paper_id') not in set(graph_paper_ids)]

                # Build a score map from vector results for re-ranking
                vector_score_map = {}
                for paper in vector_results:
                    p_id = paper.get('paper_id')
                    dist = paper.get('distance', 0.0)
                    if p_id not in vector_score_map or dist < vector_score_map[p_id]:
                        vector_score_map[p_id] = dist

                # Merge results: graph results + enriched vector-only results
                graph_pid_set = set(graph_paper_ids)
                merged_results = list(graph_results)
                for vr in vector_enriched:
                    if vr.get('paper_id') not in graph_pid_set:
                        merged_results.append(vr)

                # Re-rank merged results by vector similarity (papers without vector score go last)
                merged_results.sort(
                    key=lambda r: vector_score_map.get(r.get('paper_id'), float('inf'))
                )
                results = merged_results[:top_k]

                # Visual search on the merged paper IDs
                final_paper_ids = [r.get('paper_id') for r in results if r.get('paper_id')]
                visual_data = await self.search_visual_by_text(query, top_k, paper_ids=final_paper_ids)

            # Step 5: Cache the result
            if self.cache_manager:
                self.cache_manager.cache_cypher(query, top_k, cypher_query, parameters)

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
    @staticmethod
    def _normalize_extracted(extracted: Dict) -> Dict:
        """Normalize extracted entities for cleaner template selection & parameter filling.

        Deterministic cleanup — no LLM call needed:
        - Trim whitespace from all string values
        - Normalize author names (title-case, strip punctuation)
        - Deduplicate and lowercase keywords
        - Remove empty / whitespace-only entries
        - Ensure list values contain no None or empty strings

        Args:
            extracted: Raw extracted entities dict from _extract_entities_with_ai or _extract_entities_regex

        Returns:
            Cleaned copy of the extracted dict
        """
        cleaned: Dict = {}

        for key, value in extracted.items():
            if value is None:
                continue

            if isinstance(value, list):
                # Clean each item in the list
                normed = []
                for item in value:
                    if isinstance(item, str):
                        item = item.strip()
                        if not item:
                            continue
                        normed.append(item)
                    elif item is not None:
                        normed.append(item)
                if not normed:
                    continue
                value = normed

            elif isinstance(value, str):
                value = value.strip()
                if not value:
                    continue

            elif isinstance(value, bool):
                if not value:
                    continue

            cleaned[key] = value

        # Normalize author names: title-case, strip trailing punctuation
        if 'author_names' in cleaned:
            cleaned['author_names'] = [
                re.sub(r'[.,:;!?]+$', '', name).strip().title()
                for name in cleaned['author_names']
                if name.strip()
            ]
            # Remove duplicates (case-insensitive) while preserving order
            seen = set()
            unique = []
            for name in cleaned['author_names']:
                key_lower = name.lower()
                if key_lower not in seen:
                    seen.add(key_lower)
                    unique.append(name)
            cleaned['author_names'] = unique
            if not cleaned['author_names']:
                del cleaned['author_names']

        # Normalize keywords: lowercase, deduplicate, remove very short ones
        if 'keywords' in cleaned:
            seen_kw = set()
            unique_kw = []
            for kw in cleaned['keywords']:
                kw_lower = kw.lower().strip()
                if len(kw_lower) > 2 and kw_lower not in seen_kw:
                    seen_kw.add(kw_lower)
                    unique_kw.append(kw_lower)
            cleaned['keywords'] = unique_kw
            if not cleaned['keywords']:
                del cleaned['keywords']

        # Normalize venue / institution: title-case, strip punctuation
        for field in ('venue', 'institution'):
            if field in cleaned and isinstance(cleaned[field], str):
                cleaned[field] = re.sub(r'[.,:;!?]+$', '', cleaned[field]).strip()
                if not cleaned[field]:
                    del cleaned[field]

        logger.debug(f"Normalized extracted entities: {cleaned}")
        return cleaned

    # Mapping from template trigger names to the extracted entity keys they require.
    # A template is "viable" only when ALL its required entity keys are present
    # in the extracted dict (non-empty).
    TRIGGER_TO_ENTITY_KEYS: Dict[str, List[str]] = {
        'paper_ids': ['paper_ids'],
        'author_names': ['author_names'],
        'keywords': ['keywords'],
        'venue': ['venue'],
        'institution': ['institution'],
        'year': ['year'],
        'year_range': ['year_from'],
        'citations': ['paper_ids', 'wants_citations'],
        'coauthor': ['author_names', 'wants_coauthors'],
        'top_cited': ['wants_top_cited'],
        'author_venues': ['author_names'],
    }

    def _get_viable_templates(self, extracted: Dict) -> List[str]:
        """Return template keys whose required parameters are present in extracted entities.

        A template is viable when every entity key implied by its triggers
        exists and is truthy in the extracted dict.  Templates that need no
        specific entity (e.g. top_cited_papers when the intent flag is set)
        are also included.

        search_by_keywords is always viable as an ultimate fallback because
        keywords are extracted from virtually every query.
        """
        viable = []
        for tpl_key, tpl in self.GRAPH_TEMPLATES.items():
            triggers = tpl.get('triggers', [])
            if not triggers:
                viable.append(tpl_key)
                continue

            # All trigger groups must be satisfiable
            all_satisfied = True
            for trigger in triggers:
                required_keys = self.TRIGGER_TO_ENTITY_KEYS.get(trigger, [])
                if not all(bool(extracted.get(k)) for k in required_keys):
                    all_satisfied = False
                    break

            if all_satisfied:
                viable.append(tpl_key)

        # search_by_keywords is always a valid fallback
        if 'search_by_keywords' not in viable:
            viable.append('search_by_keywords')

        return viable

    # Priority order for rule-based ranking (most specific → most generic).
    # Templates listed earlier are preferred when multiple are viable.
    TEMPLATE_PRIORITY: List[str] = [
        'search_by_paper_ids',
        'search_citations',
        'coauthor_network',
        'search_author_by_keywords',
        'search_by_author',
        'author_venue_stats',
        'top_cited_papers',
        'search_by_venue',
        'search_by_institution',
        'search_by_year_range',
        'search_by_year',
        'search_by_keywords',  # ultimate fallback — always viable
    ]

    def _rank_viable_templates(self, extracted: Dict) -> List[str]:
        """Return viable template keys sorted by priority (best first).

        A template is viable when every entity key implied by its triggers
        is present and truthy in *extracted*.  The result is sorted according
        to TEMPLATE_PRIORITY so the top entry is the deterministic best pick.

        search_by_keywords is always appended as a fallback.
        """
        viable = self._get_viable_templates(extracted)

        # Sort by priority order; unknown templates go to the end
        priority_index = {k: i for i, k in enumerate(self.TEMPLATE_PRIORITY)}
        viable.sort(key=lambda k: priority_index.get(k, len(self.TEMPLATE_PRIORITY)))
        return viable

    async def _select_template(self, query: str, extracted: Dict) -> str:
        """Select the best graph template for the user's query.

        Flow:
        1. Build a ranked shortlist of viable templates (filtered by
           extracted entities, sorted by priority).
        2. If only one candidate → return it immediately.
        3. If multiple candidates AND an AI agent is available → let the
           AI choose from the shortlist (constrained prompt — the model
           can only output one of the listed keys).
        4. Otherwise → return the top-priority candidate.

        Args:
            query: User's natural language query
            extracted: Normalized extracted entities dict

        Returns:
            Template key name from GRAPH_TEMPLATES
        """
        candidates = self._rank_viable_templates(extracted)
        logger.info(f"Viable templates (ranked): {candidates}")

        # Fast path: single candidate or no AI agent
        if len(candidates) <= 1 or not self.ai_agent:
            selected = candidates[0] if candidates else 'search_by_keywords'
            logger.info(f"Template selected (deterministic): {selected}")
            return selected

        # ── Let AI choose from the shortlist ──
        try:
            options = "\n".join(
                f"  {key}: {self.GRAPH_TEMPLATES[key]['description']}"
                for key in candidates
                if key in self.GRAPH_TEMPLATES
            )

            prompt = (
                f'User query: "{query}"\n'
                f'Extracted entities: {extracted}\n\n'
                f'Pick the single best template from this list:\n'
                f'{options}\n\n'
                f'Reply with ONLY the template key (e.g. search_by_author). '
                f'No explanation.'
            )

            raw = await run_blocking(
                self.ai_agent.generate_content,
                prompt=prompt,
                system_prompt=(
                    'You are a query router. Given a user query and its '
                    'extracted entities, pick the single most appropriate '
                    'Neo4j template key from the provided list. '
                    'Reply with ONLY the key.'
                ),
                purpose='template_selection',
                max_tokens=32,
            )

            if raw:
                choice = raw.strip().strip('"\' ').lower()
                if choice in candidates:
                    logger.info(f"Template selected (AI): {choice}")
                    return choice
                logger.warning(f"AI returned '{choice}' which is not in candidates {candidates}")

        except Exception as e:
            logger.warning(f"AI template selection failed ({e}), using top-priority candidate")

        # Fallback: top-priority candidate
        selected = candidates[0]
        logger.info(f"Template selected (priority fallback): {selected}")
        return selected

    # =========================================================================
    # ENTITY EXTRACTION — Pull structured data from natural language query
    # =========================================================================

    async def _extract_entities_with_ai(self, query: str) -> Dict:
        """Use the AI agent to extract entities from a natural language query.

        The LLM is much better than regex at understanding natural language
        structure (e.g. "What papers has Stephen F. Altschul authored?" →
        author_names: ["Stephen F. Altschul"]).

        Returns a dict with the same keys as _extract_entities_regex:
            paper_ids, author_names, keywords, year, year_from, year_to,
            venue, institution, wants_citations, wants_coauthors, wants_top_cited
        """
        prompt = (
            f'Query: "{query}"\n'
            f'Extract as JSON: {{"paper_ids":[],"author_names":[],"keywords":[],'
            f'"year":"","year_from":"","year_to":"","venue":"","institution":"",'
            f'"wants_citations":false,"wants_coauthors":false,"wants_top_cited":false}}\n'
            f'Omit empty keys. JSON:'
        )

        raw = await run_blocking(
            self.template_agent.generate_content,
            prompt=prompt,
            system_prompt='Extract entities from academic queries. Return ONLY valid JSON.',
            purpose='entity_extraction',
        )

        if not raw:
            return {}

        # Parse the JSON from the LLM response
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.split('\n', 1)[-1]
            if cleaned.endswith('```'):
                cleaned = cleaned.rsplit('```', 1)[0]
            cleaned = cleaned.strip()

            # Find the first { ... } block
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            if start == -1 or end == -1:
                logger.warning(f"AI entity extraction returned no JSON object: {raw[:200]}")
                return {}

            extracted = json.loads(cleaned[start:end + 1])

            # Validate types — only keep expected keys with correct types
            valid: Dict = {}
            list_keys = ('paper_ids', 'author_names', 'keywords')
            str_keys = ('year', 'year_from', 'year_to', 'venue', 'institution')
            bool_keys = ('wants_citations', 'wants_coauthors', 'wants_top_cited')

            for k in list_keys:
                v = extracted.get(k)
                if isinstance(v, list) and v:
                    valid[k] = [str(item).strip() for item in v if item]
                elif isinstance(v, str) and v.strip():
                    valid[k] = [v.strip()]

            for k in str_keys:
                v = extracted.get(k)
                if isinstance(v, str) and v.strip():
                    valid[k] = v.strip()
                elif isinstance(v, (int, float)):
                    valid[k] = str(int(v))

            for k in bool_keys:
                v = extracted.get(k)
                if v is True:
                    valid[k] = True

            logger.info(f"AI entity extraction: {valid}")
            return valid

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"AI entity extraction JSON parse failed: {e} — raw: {raw[:200]}")
            return {}

    def _extract_entities_regex(self, query: str) -> Dict:
        """Regex-based entity extraction (fast fallback).

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

            system_prompt = (
                "You are a precise research assistant. Strict rules:\n"
                "- Write ONLY a single plain-text paragraph of 5-6 complete sentences.\n"
                "- Do NOT use bullet points, numbered lists, headers, or markdown formatting.\n"
                "- Do NOT repeat or rephrase the question.\n"
                "- Do NOT start with phrases like 'Based on the search results' or 'According to'.\n"
                "- ONLY state facts from the provided evidence. NEVER invent authors, dates, or findings.\n"
                "- Cite papers using their EXACT title and authors from the evidence.\n"
                "- If evidence is insufficient, say so in one sentence."
            )
            prompt = f"""Question: "{query}"
{template_context}

Evidence:
{self._format_papers_for_prompt(context_papers[:4])}

{template_instructions}

Write a single paragraph of 5-6 sentences answering the question. No bullet points. No lists. Start directly with the answer:"""

            # Generate response using LLMClient — offload to thread pool
            ai_answer = await run_blocking(
                self.ai_agent.generate_content,
                prompt=prompt,
                system_prompt=system_prompt,
                purpose='answer_synthesis',
                max_tokens=350,
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
                "Respond in 5-6 sentences. Summarize key findings from the evidence.\n"
                "Cite papers by exact title and authors as listed. Do not add information not in the evidence."
            )

        tpl_key = template_info["template_key"]

        instructions_map = {
            "search_by_paper_ids": (
                "State the paper's exact title, authors, venue, year, and DOI from the evidence.\n"
                "Include citation count if listed. Summarize the abstract in 1-2 sentences."
            ),
            "search_by_author": (
                "List the author's papers found, citing exact titles, venues, and years from the evidence.\n"
                "Mention the most cited work if citation counts are available."
            ),
            "search_by_keywords": (
                "Identify which paper(s) best answer the question, citing exact titles and authors.\n"
                "Summarize the most relevant finding in 2-3 sentences using only the abstracts provided."
            ),
            "search_citations": (
                "Describe the citation relationships using only the papers listed in the evidence.\n"
                "State which paper cites which, with exact titles."
            ),
            "coauthor_network": (
                "List the co-authored papers with exact titles, venues, and years from the evidence.\n"
                "Describe the collaboration scope briefly."
            ),
            "search_by_venue": (
                "List the papers from the venue, citing exact titles and authors from the evidence.\n"
                "Mention citation counts if available."
            ),
            "search_by_institution": (
                "List papers from the institution with exact titles and authors from the evidence.\n"
                "Note the research areas covered."
            ),
            "search_by_year": (
                "List the papers from the requested period with exact titles and authors.\n"
                "Highlight the most cited if counts are available."
            ),
            "search_by_year_range": (
                "List papers from the time range with exact titles and authors.\n"
                "Note any trends visible from the evidence."
            ),
            "search_author_by_keywords": (
                "List the author's papers matching the topic, citing exact titles and years.\n"
                "Summarize how the work relates to the topic in 1-2 sentences."
            ),
            "top_cited_papers": (
                "List the most cited papers with their exact citation counts, titles, and authors.\n"
                "Briefly describe each paper's contribution in one sentence."
            ),
            "author_venue_stats": (
                "List the venues where the author published, with paper counts if available.\n"
                "Highlight the primary publication outlets."
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
        """Format papers compactly for LLM prompt (minimize input tokens)."""
        lines = []
        for i, p in enumerate(papers, 1):
            authors = p.get('authors', [])
            auth = ', '.join(authors[:2]) + (' et al.' if len(authors) > 2 else '') if authors else ''
            abstract = (p.get('abstract', '') or '')[:300]
            lines.append(
                f"{i}. {p.get('title', 'Untitled')} | {auth} | "
                f"{p.get('venue', '') or ''} {p.get('publication_date', '') or ''}\n"
                f"   {abstract}"
            )
        return '\n'.join(lines)

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

            # 3. Try to filter results to figures/tables belonging to search papers,
            #    but fall back to unscoped results when there's no overlap.
            #    The figures/tables collections may contain papers not in the
            #    text search results, so strict filtering can eliminate everything.
            visual_fig_pids = set(r.get('paper_id') for r in figure_results if r.get('paper_id'))
            visual_tab_pids = set(r.get('paper_id') for r in table_results if r.get('paper_id'))
            fig_intersection = visual_fig_pids & allowed_pids
            tab_intersection = visual_tab_pids & allowed_pids
            logger.info(
                f"Visual filter debug — "
                f"allowed_pids ({len(allowed_pids)}): {list(allowed_pids)[:5]}, "
                f"figure paper_ids ({len(visual_fig_pids)}): {list(visual_fig_pids)[:5]}, "
                f"table paper_ids ({len(visual_tab_pids)}): {list(visual_tab_pids)[:5]}, "
                f"intersection figs: {fig_intersection}, "
                f"intersection tabs: {tab_intersection}"
            )

            scoped_figs = [r for r in figure_results if r.get('paper_id') in allowed_pids]
            scoped_tabs = [r for r in table_results if r.get('paper_id') in allowed_pids]

            if scoped_figs or scoped_tabs:
                # Use scoped results when there is overlap
                figure_results = scoped_figs
                table_results = scoped_tabs
                logger.info("Visual filter: using scoped results (overlap found)")
            else:
                # Fallback: keep the top unscoped results ranked by similarity
                # so the UI still shows relevant visual evidence
                figure_results = sorted(
                    figure_results, key=lambda r: r.get('similarity_score', 0), reverse=True
                )[:top_k]
                table_results = sorted(
                    table_results, key=lambda r: r.get('similarity_score', 0), reverse=True
                )[:top_k]
                logger.info(
                    f"Visual filter: no overlap with search papers — "
                    f"falling back to top {len(figure_results)} figures, "
                    f"{len(table_results)} tables (unscoped)"
                )

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
