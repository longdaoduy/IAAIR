from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator

class PaperRequest(BaseModel):
    """Request model for paper ingestion."""
    num_papers: int = Field(..., gt=0, le=1000, description="Number of papers to pull (1-1000)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for OpenAlex API")
    include_neo4j: bool = Field(False, description="Whether to upload to Neo4j")
    include_zilliz: bool = Field(False, description="Whether to upload to Zilliz")