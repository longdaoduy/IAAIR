from typing import Dict, Any
from pydantic import BaseModel
from models.entities.retrieval.RoutingStrategy import RoutingStrategy
from models.entities.retrieval.QueryType import QueryType

class QueryAnalysisResponse(BaseModel):
    """Response for query analysis."""
    success: bool
    query: str
    query_type: QueryType
    confidence: float
    suggested_routing: RoutingStrategy
    analysis_details: Dict[str, Any]