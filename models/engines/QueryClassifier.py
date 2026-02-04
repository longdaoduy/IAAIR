from models.entities.retrieval.QueryType import QueryType
from typing import Tuple

class QueryClassifier:
    """Classify queries to determine optimal routing strategy."""

    def __init__(self):
        # Keywords indicating different query types
        self.semantic_keywords = {'similar', 'related', 'about', 'concerning', 'regarding'}
        self.structural_keywords = {'cited', 'authored', 'collaborated', 'published', 'references'}
        self.factual_keywords = {'who', 'what', 'when', 'where', 'which', 'how many'}

    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """Classify query and return confidence score."""
        query_lower = query.lower()

        # Count keyword matches
        semantic_score = sum(1 for kw in self.semantic_keywords if kw in query_lower)
        structural_score = sum(1 for kw in self.structural_keywords if kw in query_lower)
        factual_score = sum(1 for kw in self.factual_keywords if kw in query_lower)

        # Simple heuristic-based classification
        scores = {
            QueryType.SEMANTIC: semantic_score + (0.5 if len(query.split()) > 5 else 0),
            QueryType.STRUCTURAL: structural_score + (
                0.3 if any(op in query_lower for op in ['and', 'or', 'not']) else 0),
            QueryType.FACTUAL: factual_score + (0.4 if query_lower.startswith(tuple(self.factual_keywords)) else 0)
        }

        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        # If no clear winner, classify as hybrid
        if max_score < 1.0 or sum(scores.values()) > 1.5:
            return QueryType.HYBRID, 0.6

        confidence = min(max_score / 2.0, 1.0)
        return max_type, confidence