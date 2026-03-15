from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from models.entities.retrievals.SearchResult import SearchResult

class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    success: bool
    message: Optional[str] = None
    query: str
    query_type: Optional[str] = None
    results_found: int
    search_time_seconds: float
    fusion_time_seconds: Optional[float] = None
    reranking_time_seconds: Optional[float] = None
    response_generation_time_seconds: Optional[float] = None
    results: List[SearchResult]
    ai_response: Optional[str] = None
    graph_template_used: Optional[str] = None
    fusion_stats: Dict[str, Any] = {}
    # Visual evidence from cross-modal search
    visual_results: List[Dict[str, Any]] = []  # Matched figures/tables
    visual_stats: Dict[str, Any] = {}  # Visual search statistics
    # attribution_stats: Dict[str, Any] = {}
