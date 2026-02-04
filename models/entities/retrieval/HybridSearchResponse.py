from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from models.entities.retrieval.RoutingStrategy import RoutingStrategy
from models.entities.retrieval.QueryType import QueryType
from models.entities.retrieval.SearchResult import SearchResult

class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    success: bool
    message: str
    query: str
    query_type: QueryType
    routing_used: RoutingStrategy
    results_found: int
    search_time_seconds: float
    fusion_time_seconds: Optional[float] = None
    reranking_time_seconds: Optional[float] = None
    results: List[SearchResult]
    fusion_stats: Dict[str, Any] = {}
    attribution_stats: Dict[str, Any] = {}
