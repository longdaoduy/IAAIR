"""
Configuration settings for the hybrid fusion system.
"""

from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class HybridFusionConfig:
    """Configuration for hybrid fusion system."""
    
    # Routing Configuration
    ADAPTIVE_ROUTING_ENABLED: bool = True
    DEFAULT_ROUTING_STRATEGY: str = "adaptive"
    QUERY_CLASSIFICATION_THRESHOLD: float = 0.7
    
    # Fusion Weights
    DEFAULT_VECTOR_WEIGHT: float = 0.4
    DEFAULT_GRAPH_WEIGHT: float = 0.3
    DEFAULT_RERANK_WEIGHT: float = 0.3
    
    # Reranking Configuration
    RERANKING_ENABLED: bool = True
    CITATION_WEIGHT: float = 0.3
    RECENCY_WEIGHT: float = 0.2
    VENUE_WEIGHT: float = 0.2
    AUTHOR_WEIGHT: float = 0.15
    SEMANTIC_WEIGHT: float = 0.15
    
    # Attribution Configuration
    ATTRIBUTION_ENABLED: bool = True
    ATTRIBUTION_CONFIDENCE_THRESHOLD: float = 0.7
    MAX_ATTRIBUTION_SPANS: int = 10
    MIN_SPAN_LENGTH: int = 5
    
    # Performance Configuration
    MAX_VECTOR_RESULTS: int = 100
    MAX_GRAPH_RESULTS: int = 100
    PARALLEL_TIMEOUT_SECONDS: int = 30
    PERFORMANCE_HISTORY_SIZE: int = 100
    
    # Scientific Domain Configuration
    HIGH_IMPACT_VENUES: list = None
    FIELD_SPECIFIC_WEIGHTS: Dict[str, float] = None
    
    def __post_init__(self):
        if self.HIGH_IMPACT_VENUES is None:
            self.HIGH_IMPACT_VENUES = [
                "Nature", "Science", "Cell", "Nature Medicine",
                "Nature Biotechnology", "NEJM", "Lancet", "PNAS",
                "Nature Neuroscience", "Nature Genetics"
            ]
        
        if self.FIELD_SPECIFIC_WEIGHTS is None:
            self.FIELD_SPECIFIC_WEIGHTS = {
                "computer_science": {"citation_weight": 0.25, "recency_weight": 0.35},
                "medicine": {"citation_weight": 0.35, "recency_weight": 0.15},
                "biology": {"citation_weight": 0.3, "recency_weight": 0.2},
                "physics": {"citation_weight": 0.4, "recency_weight": 0.1}
            }

# Global configuration instance
CONFIG = HybridFusionConfig()