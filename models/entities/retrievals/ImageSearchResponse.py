from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class VisualResult(BaseModel):
    """A single figure or table search result."""
    id: str
    paper_id: Optional[str] = None
    description: Optional[str] = None
    similarity_score: float = 0.0
    image_score: Optional[float] = None
    text_score: Optional[float] = None
    collection: str = ""  # "figures" or "tables"
    search_type: str = ""  # "image_to_figure", "hybrid_visual", etc.


class RelatedPaper(BaseModel):
    """A paper related to visual search results."""
    id: str
    title: Optional[str] = None
    abstract: Optional[str] = None


class ImageSearchResponse(BaseModel):
    """Response model for image-based search."""
    success: bool
    message: str
    search_time_seconds: float
    text_query: Optional[str] = None
    figure_results: List[VisualResult] = []
    table_results: List[VisualResult] = []
    related_papers: List[RelatedPaper] = []
    total_figures_found: int = 0
    total_tables_found: int = 0
