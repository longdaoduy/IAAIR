from models.entities.retrieval.QueryType import QueryType
from typing import Tuple
import re

class QueryClassifier:
    """Classify queries to determine optimal routing strategy."""

    def __init__(self):
        # Keywords indicating different query types
        self.semantic_keywords = {'similar', 'related', 'about', 'concerning', 'regarding'}
        self.structural_keywords = {'cited', 'authored', 'collaborated', 'published', 'references', 'author', 'co-author', 'collaborator'}
        self.factual_keywords = {'who', 'what', 'when', 'where', 'which', 'how many'}
        
        # Patterns for specific query types
        self.paper_id_pattern = re.compile(r'\b(W\d+|DOI:|doi:|pmid:|arxiv:)', re.IGNORECASE)
        self.author_query_patterns = [
            re.compile(r'who\s+is\s+the\s+author', re.IGNORECASE),
            re.compile(r'authors?\s+of\s+paper', re.IGNORECASE),
            re.compile(r'who\s+wrote', re.IGNORECASE),
            re.compile(r'who\s+authored', re.IGNORECASE)
        ]

    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """Classify query and return confidence score."""
        query_lower = query.lower()

        # First check for specific structural patterns
        # Paper ID + author query pattern
        if self.paper_id_pattern.search(query) and any(pattern.search(query) for pattern in self.author_query_patterns):
            return QueryType.STRUCTURAL, 0.95
        
        # General author query patterns
        if any(pattern.search(query) for pattern in self.author_query_patterns):
            return QueryType.STRUCTURAL, 0.85
        
        # Paper ID queries (likely structural)
        if self.paper_id_pattern.search(query):
            return QueryType.STRUCTURAL, 0.80

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