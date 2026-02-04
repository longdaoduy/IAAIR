from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

class GraphQueryResponse(BaseModel):
    """Response model for graph queries."""
    success: bool
    message: str
    query_time_seconds: float
    results_count: int
    results: List[Dict[str, Any]]
    query_info: Optional[Dict[str, Any]] = None
