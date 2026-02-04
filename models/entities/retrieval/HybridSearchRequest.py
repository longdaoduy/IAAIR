from typing import Dict, Optional
from pydantic import BaseModel, Field
from models.entities.retrieval.RoutingStrategy import RoutingStrategy


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search with fusion."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, gt=0, le=100, description="Number of results to return")
    routing_strategy: RoutingStrategy = Field(RoutingStrategy.ADAPTIVE, description="Routing strategy")
    enable_reranking: bool = Field(True, description="Enable neural reranking")
    enable_attribution: bool = Field(True, description="Track source attribution")
    fusion_weights: Optional[Dict[str, float]] = Field(None, description="Custom fusion weights")
    include_provenance: bool = Field(False, description="Include detailed provenance")