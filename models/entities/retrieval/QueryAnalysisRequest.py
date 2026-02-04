from pydantic import BaseModel, Field


class QueryAnalysisRequest(BaseModel):
    """Request for query classification analysis."""
    query: str = Field(..., min_length=1, description="Query to analyze")
    include_routing_suggestion: bool = Field(True, description="Include routing strategy suggestion")
