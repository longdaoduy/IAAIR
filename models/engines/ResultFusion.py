from models.entities.retrieval.SearchResult import SearchResult
from typing import Dict, List, Optional


class ResultFusion:
    """Fuse results from different retrievals strategies."""

    def __init__(self):
        self.default_weights = {
            'vector_score': 0.4,
            'graph_score': 0.3,
            'rerank_score': 0.3
        }

    def fuse_results(self, vector_results: List[Dict], graph_results: List[Dict],
                     fusion_weights: Optional[Dict[str, float]] = None) -> List[SearchResult]:
        """Fuse results from vector and graph search."""
        weights = fusion_weights or self.default_weights

        # Create result index by paper_id
        all_results = {}

        # 1. Process Vector Results
        for result in vector_results:
            # Handle both potential key names
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
                'vector_score': min(result.get('similarity_score', 0.0), 1.0),
                'graph_score': 0.0,
                'source_path': ['vector_search']
            }

        # 2. Process Graph Results and Merge
        for result in graph_results:
            paper_id = result.get('paper_id') or result.get('id')
            if not paper_id:
                continue

            if paper_id in all_results:
                # ENHANCEMENT: Fill in missing info from Graph (metadata enrichment)
                existing = all_results[paper_id]
                # If vector search was missing metadata, use the graph's richer data
                existing['authors'] = existing.get('authors') or result.get('authors', [])
                existing['venue'] = existing.get('venue') or result.get('venue')
                existing['doi'] = existing.get('doi') or result.get('doi')
                existing['publication_date'] = existing.get('publication_date') or result.get('publication_date')

                # Update scores and path
                existing['graph_score'] = min(result.get('relevance_score', 0.5), 1.0)
                existing['source_path'].append('graph_search')  # Removed duplicate append
            else:
                # 3. Handle papers found ONLY in Graph
                all_results[paper_id] = {
                    'paper_id': paper_id,
                    'title': result.get('title') or 'Unknown Title',
                    'abstract': result.get('abstract'),
                    'authors': result.get('authors', []),
                    'venue': result.get('venue'),
                    'publication_date': result.get('publication_date'),
                    'doi': result.get('doi'),
                    'vector_score': 0.0,
                    'graph_score': min(result.get('relevance_score', 0.5), 1.0),
                    'source_path': ['graph_search']
                }

        # Calculate fusion scores
        fused_results = []
        for result_data in all_results.values():
            # Calculate weighted fusion score
            raw_relevance_score = (
                    weights['vector_score'] * result_data['vector_score'] +
                    weights['graph_score'] * result_data['graph_score']
            )

            # Normalize the final score to ensure it's reasonable (optional cap at 1.0)
            relevance_score = min(raw_relevance_score, 1.0)

            search_result = SearchResult(
                paper_id=result_data['paper_id'],
                title=result_data['title'] or 'Unknown Title',
                abstract=result_data['abstract'],
                authors=result_data['authors'] or [],
                venue=result_data['venue'],
                publication_date=result_data['publication_date'],
                doi=result_data['doi'],
                relevance_score=relevance_score,
                vector_score=result_data['vector_score'],
                graph_score=result_data['graph_score'],
                source_path=result_data['source_path'],
                attributions=[],
                confidence_scores={
                    'vector_confidence': result_data['vector_score'],
                    'graph_confidence': result_data['graph_score'],
                    'raw_fusion_score': raw_relevance_score  # Keep track of original score
                }
            )
            fused_results.append(search_result)

        # Sort by relevance score
        fused_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return fused_results
