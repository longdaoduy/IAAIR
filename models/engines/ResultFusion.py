from models.entities.retrievals.SearchResult import SearchResult
from typing import Dict, List, Optional


class ResultFusion:
    """Fuse results from hybrid search (combined graph+vector) with visual evidence."""

    def __init__(self):
        self.default_weights = {
            'hybrid_score': 0.7,
            'visual_score': 0.15
        }

    def fuse_results(self, hybrid_results: List[Dict],
                     fusion_weights: Optional[Dict[str, float]] = None,
                     visual_data: Optional[Dict] = None) -> List[SearchResult]:
        """Fuse hybrid search results with cross-modal visual evidence.

        The hybrid_results already combine graph and vector search (vector search
        finds paper IDs which are then enriched by the graph query), so this method
        only needs to merge visual evidence and compute the final relevance score.

        Args:
            hybrid_results: Results from the hybrid search pipeline (already
                            combines vector and graph search)
            fusion_weights: Optional custom fusion weights
            visual_data:    Optional cross-modal visual search data containing:
                            - paper_visual_scores: {paper_id: float} visual relevance boosts
                            - figure_results: list of matched figures
                            - table_results:  list of matched tables
        """
        weights = fusion_weights or self.default_weights
        visual_weight = weights.get('visual_score', 0.15)

        # Extract visual data
        paper_visual_scores = {}
        figure_results = []
        table_results = []
        if visual_data:
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

        # Create result index by paper_id
        all_results = {}

        # 1. Process hybrid results (already combined graph + vector search)
        for result in hybrid_results:
            paper_id = result.get('paper_id') or result.get('id')
            if not paper_id:
                continue

            all_results[paper_id] = {
                'paper_id': paper_id,
                'title': result.get('title') or 'Unknown Title',
                'abstract': result.get('abstract'),
                'authors': result.get('authors', []),
                'venue': result.get('venue'),
                'publication_date': result.get('publication_date'),
                'doi': result.get('doi'),
                'cited_by_count': result.get('cited_by_count', 0) or 0,
                'hybrid_score': min(result.get('relevance_score', 0.5), 1.0),
                'visual_score': paper_visual_scores.get(paper_id, 0.0),
                'matched_figures': paper_fig_count.get(paper_id, 0),
                'matched_tables': paper_tab_count.get(paper_id, 0),
                'visual_evidence': paper_visual_evidence.get(paper_id, []),
                'source_path': ['hybrid_search']
            }

        # 2. Add papers discovered ONLY through visual evidence (not in hybrid results)
        for pid, v_score in paper_visual_scores.items():
            if pid not in all_results:
                all_results[pid] = {
                    'paper_id': pid,
                    'title': 'Unknown Title',
                    'abstract': None,
                    'authors': [],
                    'venue': None,
                    'publication_date': None,
                    'doi': None,
                    'cited_by_count': 0,
                    'hybrid_score': 0.0,
                    'visual_score': v_score,
                    'matched_figures': paper_fig_count.get(pid, 0),
                    'matched_tables': paper_tab_count.get(pid, 0),
                    'visual_evidence': paper_visual_evidence.get(pid, []),
                    'source_path': ['visual_search']
                }

        # Calculate fusion scores
        fused_results = []
        for result_data in all_results.values():
            v_score = result_data.get('visual_score', 0.0)
            h_score = result_data.get('hybrid_score', 0.0)

            # Calculate fusion score: hybrid relevance + visual boost
            raw_relevance_score = h_score

            # Add visual boost — cross-modal evidence increases paper relevance
            if v_score > 0:
                raw_relevance_score += visual_weight * v_score
                if 'visual_search' not in result_data['source_path']:
                    result_data['source_path'].append('visual_search')

            # Normalize the final score to ensure it's reasonable (cap at 1.0)
            relevance_score = min(raw_relevance_score, 1.0)

            # Trim visual_evidence to essential fields for response size
            trimmed_evidence = []
            for ve in result_data.get('visual_evidence', [])[:6]:
                trimmed_evidence.append({
                    'id': ve.get('id', ''),
                    'description': (ve.get('description', '') or '')[:200],
                    'similarity_score': ve.get('similarity_score', 0.0),
                    'collection': ve.get('collection', ''),
                    'search_type': ve.get('search_type', ''),
                })

            search_result = SearchResult(
                paper_id=result_data['paper_id'],
                title=result_data['title'] or 'Unknown Title',
                abstract=result_data['abstract'],
                authors=result_data['authors'] or [],
                venue=result_data['venue'],
                publication_date=result_data['publication_date'],
                doi=result_data['doi'],
                cited_by_count=result_data.get('cited_by_count', 0) or 0,
                relevance_score=relevance_score,
                vector_score=None,
                graph_score=None,
                visual_score=v_score,
                matched_figures=result_data.get('matched_figures', 0),
                matched_tables=result_data.get('matched_tables', 0),
                visual_evidence=trimmed_evidence,
                source_path=result_data['source_path'],
                attributions=[],
                confidence_scores={
                    'hybrid_confidence': h_score,
                    'visual_confidence': v_score,
                    'raw_fusion_score': raw_relevance_score
                }
            )
            fused_results.append(search_result)

        # Sort by relevance score
        fused_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return fused_results
