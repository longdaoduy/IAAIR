from models.entities.retrieval.SearchResult import SearchResult
from models.entities.retrieval.AttributionSpan import AttributionSpan
from typing import List

class AttributionTracker:
    """Track source attribution for retrieved content."""

    def __init__(self):
        self.confidence_threshold = 0.7

    def track_attributions(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Add attribution tracking to search results."""
        for result in results:
            # Create basic attribution spans
            attributions = self._create_attribution_spans(result, query)
            result.attributions = attributions

        return results

    def _create_attribution_spans(self, result: SearchResult, query: str) -> List[AttributionSpan]:
        """Create attribution spans for a result."""
        attributions = []

        # Title attribution
        if result.title:
            attributions.append(AttributionSpan(
                text=result.title,
                source_id=result.paper_id,
                source_type='paper',
                confidence=0.9,
                char_start=0,
                char_end=len(result.title),
                supporting_passages=[result.title]
            ))

        # Abstract attribution (if available)
        if result.abstract:
            # Simple span creation - would be more sophisticated in practice
            abstract_words = result.abstract.split()
            query_words = set(query.lower().split())

            for i, word in enumerate(abstract_words):
                if word.lower() in query_words:
                    # Create span around matching word (simplified)
                    start_pos = sum(len(w) + 1 for w in abstract_words[:i])
                    end_pos = start_pos + len(word)

                    attributions.append(AttributionSpan(
                        text=word,
                        source_id=result.paper_id,
                        source_type='abstract',
                        confidence=0.8,
                        char_start=start_pos,
                        char_end=end_pos,
                        supporting_passages=[result.abstract[:100] + '...']
                    ))

        return attributions