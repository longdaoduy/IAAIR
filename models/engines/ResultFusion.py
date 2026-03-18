import math
from models.entities.retrievals.SearchResult import SearchResult
from typing import Dict, List, Optional


class ResultFusion:
    """Convert hybrid search results into SearchResult objects with confidence scores.

    The hybrid results combine graph and vector search (multi-modal vector search
    with cross-modal re-ranking finds paper IDs → graph query enriches with metadata).

    Confidence scoring:
        - multimodal_confidence: from the vector+visual re-ranking score (SciBERT + CLIP)
        - graph_confidence: computed from Neo4j metadata completeness and citation impact
        - hybrid_confidence: weighted combination of multimodal + graph confidence
    """

    # Weights for combining confidence scores into hybrid_confidence
    MULTIMODAL_WEIGHT = 0.6
    GRAPH_WEIGHT = 0.4

    @staticmethod
    def _compute_graph_confidence(result: Dict) -> float:
        """Compute graph confidence from Neo4j metadata quality.

        Measures how rich and complete the graph data is for this paper.
        A paper with authors, venue, DOI, abstract, and high citations
        gets a higher graph confidence than one with sparse metadata.

        Returns:
            float in [0, 1]
        """
        score = 0.0

        # Has title (basic completeness)
        if result.get('title'):
            score += 0.15

        # Has abstract
        if result.get('abstract'):
            score += 0.15

        # Has authors (more authors = richer graph connections)
        authors = result.get('authors', []) or []
        if authors:
            score += min(0.20, 0.05 * len(authors))  # up to 0.20 for 4+ authors

        # Has venue (published in a known venue)
        if result.get('venue'):
            score += 0.15

        # Has DOI (verified publication)
        if result.get('doi'):
            score += 0.10

        # Has publication date
        if result.get('publication_date'):
            score += 0.05

        # Citation impact (log-scaled, up to 0.20)
        cited_by = result.get('cited_by_count', 0) or 0
        if cited_by > 0:
            # log10(1) = 0, log10(10) = 1, log10(100) = 2, log10(1000) = 3
            citation_score = min(0.20, 0.05 * math.log10(1 + cited_by))
            score += citation_score

        return min(score, 1.0)

    def fuse_results(self, hybrid_results: List[Dict],
                     visual_data: Optional[Dict] = None) -> List[SearchResult]:
        """Convert hybrid search results to SearchResult objects with confidence scoring.

        Computes three confidence dimensions:
        - multimodal_confidence: re-ranking score from vector + visual search
        - graph_confidence: metadata completeness from Neo4j
        - hybrid_confidence: weighted sum of multimodal + graph

        Args:
            hybrid_results: Results from the hybrid search pipeline (already
                            ranked by multi-modal vector search + graph enrichment)
            visual_data:    Optional cross-modal visual search data containing:
                            - multimodal_scores: {paper_id: float} combined vector+visual scores
                            - paper_visual_scores: {paper_id: float} visual-only scores
                            - figure_results: list of matched figures
                            - table_results:  list of matched tables
        """
        # Extract visual/multimodal data
        multimodal_scores = {}
        paper_visual_scores = {}
        figure_results = []
        table_results = []
        if visual_data:
            multimodal_scores = visual_data.get('multimodal_scores', {})
            paper_visual_scores = visual_data.get('paper_visual_scores', {})
            figure_results = visual_data.get('figure_results', [])
            table_results = visual_data.get('table_results', [])

        # Build per-paper visual evidence lookup
        paper_visual_evidence: Dict[str, list] = {}
        paper_fig_count: Dict[str, int] = {}
        paper_tab_count: Dict[str, int] = {}
        for r in figure_results:
            pid = r.get('paper_id')
            if pid:
                paper_visual_evidence.setdefault(pid, []).append(r)
                paper_fig_count[pid] = paper_fig_count.get(pid, 0) + 1
        for r in table_results:
            pid = r.get('paper_id')
            if pid:
                paper_visual_evidence.setdefault(pid, []).append(r)
                paper_tab_count[pid] = paper_tab_count.get(pid, 0) + 1

        # Convert hybrid results to SearchResult objects (preserving existing order)
        fused_results = []
        for result in hybrid_results:
            paper_id = result.get('paper_id') or result.get('id')
            if not paper_id:
                continue

            # ── Confidence scores ──
            # Multimodal confidence: from vector+visual re-ranking (0-1 range)
            multimodal_conf = multimodal_scores.get(paper_id, 0.0)

            # Graph confidence: from Neo4j metadata quality (0-1 range)
            graph_conf = self._compute_graph_confidence(result)

            # Visual-only score (for display)
            v_score = paper_visual_scores.get(paper_id, 0.0)

            # Hybrid confidence: weighted combination
            hybrid_conf = (
                self.MULTIMODAL_WEIGHT * multimodal_conf +
                self.GRAPH_WEIGHT * graph_conf
            )

            # Use hybrid_confidence as the relevance_score for sorting/display
            relevance_score = min(hybrid_conf, 1.0)

            # Determine source path
            source_path = ['hybrid_search']
            if multimodal_conf > 0:
                source_path.append('multimodal_search')
            if v_score > 0:
                source_path.append('visual_search')

            # Trim visual_evidence to essential fields for response size
            trimmed_evidence = []
            for ve in paper_visual_evidence.get(paper_id, [])[:6]:
                trimmed_evidence.append({
                    'id': ve.get('id', ''),
                    'description': (ve.get('description', '') or '')[:200],
                    'similarity_score': ve.get('similarity_score', 0.0),
                    'collection': ve.get('collection', ''),
                    'search_type': ve.get('search_type', ''),
                })

            search_result = SearchResult(
                paper_id=paper_id,
                title=result.get('title') or 'Unknown Title',
                abstract=result.get('abstract'),
                authors=result.get('authors', []) or [],
                venue=result.get('venue'),
                publication_date=result.get('publication_date'),
                doi=result.get('doi'),
                cited_by_count=result.get('cited_by_count', 0) or 0,
                relevance_score=relevance_score,
                vector_score=multimodal_conf,  # multimodal (vector+visual) score
                graph_score=graph_conf,         # graph metadata quality score
                visual_score=v_score,
                matched_figures=paper_fig_count.get(paper_id, 0),
                matched_tables=paper_tab_count.get(paper_id, 0),
                visual_evidence=trimmed_evidence,
                source_path=source_path,
                attributions=[],
                confidence_scores={
                    'hybrid_confidence': round(relevance_score, 4),
                    'multimodal_confidence': round(multimodal_conf, 4),
                    'graph_confidence': round(graph_conf, 4),
                    'visual_confidence': round(v_score, 4),
                }
            )
            fused_results.append(search_result)

        # Boost relevance_score for explicitly requested paper IDs so they
        # always appear at the top of results regardless of multimodal score.
        requested_pids = set()
        if visual_data:
            requested_pids = set(visual_data.get('requested_paper_ids', []))
        if requested_pids:
            best_score = max((r.relevance_score for r in fused_results), default=1.0)
            boost = best_score + 1.0  # guarantee above all other scores
            for r in fused_results:
                if r.paper_id in requested_pids:
                    r.relevance_score = r.relevance_score + boost

        # Re-sort by hybrid_confidence (descending) since graph_confidence
        # may change the relative ordering from the original pipeline
        fused_results.sort(key=lambda r: r.relevance_score, reverse=True)

        return fused_results
