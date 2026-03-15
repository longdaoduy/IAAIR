from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from models.entities.retrievals.AttributionSpan import AttributionSpan


class SearchResult(BaseModel):
    """Enhanced search result with attribution and provenance."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    venue: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    cited_by_count: int = 0
    # Scoring and ranking
    relevance_score: float = Field(..., ge=0.0, description="Composite relevance score")
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    rerank_score: Optional[float] = None
    visual_score: Optional[float] = None  # Cross-modal visual evidence score
    # Visual evidence linked to this paper
    matched_figures: int = 0
    matched_tables: int = 0
    visual_evidence: List[Dict] = []  # List of matched figures/tables with descriptions
    # Attribution and provenance
    attributions: List[AttributionSpan] = []
    source_path: List[str] = []  # Retrieval path for provenance
    confidence_scores: Dict[str, float] = {}

