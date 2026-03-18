from models.entities.retrievals.SearchResult import SearchResult
from typing import Dict, List, Optional


class ResultFusion:
    """Convert hybrid search result dicts into SearchResult objects.

    Confidence scores (_relevance_score, _graph_confidence, etc.) are
    pre-computed by HybridRetrievalHandler._compute_final_scores() and
    attached to each result dict.  This class simply reads those values
    and builds typed SearchResult objects — it no longer computes scores.
    """

    def fuse_results(self, hybrid_results: List[Dict],
                     visual_data: Optional[Dict] = None) -> List[SearchResult]:
        """Convert pre-scored hybrid result dicts into SearchResult objects.

        The result dicts are expected to carry keys prefixed with ``_``
        (set by ``HybridRetrievalHandler._compute_final_scores``):
            _relevance_score, _multimodal_confidence, _graph_confidence,
            _visual_confidence, _hybrid_confidence, _matched_figures,
            _matched_tables, _visual_evidence, _source_path,
            _confidence_scores.

        Args:
            hybrid_results: Results from the hybrid search pipeline,
                            already scored and sorted.
            visual_data:    Optional — no longer used for scoring but
                            kept for API compatibility.

        Returns:
            List of SearchResult objects preserving the input order.
        """
        fused_results: List[SearchResult] = []

        for result in hybrid_results:
            paper_id = result.get('paper_id') or result.get('id')
            if not paper_id:
                continue

            search_result = SearchResult(
                paper_id=paper_id,
                title=result.get('title') or 'Unknown Title',
                abstract=result.get('abstract'),
                authors=result.get('authors', []) or [],
                venue=result.get('venue'),
                publication_date=result.get('publication_date'),
                doi=result.get('doi'),
                cited_by_count=result.get('cited_by_count', 0) or 0,
                relevance_score=result.get('_relevance_score', 0.0),
                vector_score=result.get('_multimodal_confidence', 0.0),
                graph_score=result.get('_graph_confidence', 0.0),
                visual_score=result.get('_visual_confidence', 0.0),
                matched_figures=result.get('_matched_figures', 0),
                matched_tables=result.get('_matched_tables', 0),
                visual_evidence=result.get('_visual_evidence', []),
                source_path=result.get('_source_path', ['hybrid_search']),
                attributions=[],
                confidence_scores=result.get('_confidence_scores', {}),
            )
            fused_results.append(search_result)

        return fused_results
