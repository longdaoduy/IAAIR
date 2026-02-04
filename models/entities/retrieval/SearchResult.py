from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from models.entities.retrieval.AttributionSpan import AttributionSpan


class SearchResult(BaseModel):
    """Enhanced search result with attribution and provenance."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    venue: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    # Scoring and ranking
    relevance_score: float = Field(..., ge=0.0, description="Composite relevance score")
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    rerank_score: Optional[float] = None
    # Attribution and provenance
    attributions: List[AttributionSpan] = []
    source_path: List[str] = []  # Retrieval path for provenance
    confidence_scores: Dict[str, float] = {}

