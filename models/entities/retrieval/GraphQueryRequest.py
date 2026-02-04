from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

class GraphQueryRequest(BaseModel):
    """Request model for custom Cypher queries."""
    query: str = Field(..., description="Cypher query to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum results to return")
