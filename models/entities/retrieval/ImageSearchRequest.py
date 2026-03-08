from typing import Optional
from pydantic import BaseModel, Field


class ImageSearchRequest(BaseModel):
    """Request model for image-based search across figures and tables."""
    text_query: Optional[str] = Field(None, description="Optional text query for hybrid image+text search")
    top_k: int = Field(10, gt=0, le=50, description="Number of results to return per collection")
    search_figures: bool = Field(True, description="Search figures collection")
    search_tables: bool = Field(True, description="Search tables collection")
    include_related_papers: bool = Field(True, description="Include related paper details")
