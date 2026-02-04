from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str = Field(..., min_length=1, description="Text query to search for similar papers")
    top_k: int = Field(10, gt=0, le=50, description="Number of top results to return (1-50)")
    include_details: bool = Field(True, description="Whether to include detailed paper information from Neo4j")
    use_hybrid: bool = Field(True, description="Whether to use hybrid search (dense + sparse) or dense-only search")
