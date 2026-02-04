from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

class SearchResponse(BaseModel):
    """Response model for semantic search."""
    success: bool
    message: str
    query: str
    results_found: int
    search_time_seconds: float
    results: List[Dict[str, Any]]