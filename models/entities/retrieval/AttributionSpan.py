from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator

class AttributionSpan(BaseModel):
    """Attribution span with source tracking."""
    text: str
    source_id: str
    source_type: str  # 'paper', 'abstract', 'citation'
    confidence: float = Field(..., ge=0.0, le=1.0)
    char_start: int
    char_end: int
    supporting_passages: List[str] = []
