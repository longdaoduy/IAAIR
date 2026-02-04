from enum import Enum

class QueryType(str, Enum):
    """Query classification types."""
    SEMANTIC = "semantic"          # Concept-based queries
    STRUCTURAL = "structural"      # Relationship-based queries
    HYBRID = "hybrid"             # Mixed semantic and structural
    FACTUAL = "factual"           # Specific fact retrieval