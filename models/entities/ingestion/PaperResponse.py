from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator

class PaperResponse(BaseModel):
    """Response model for paper ingestion."""
    success: bool
    message: str
    papers_processed: int
    neo4j_uploaded: bool
    zilliz_uploaded: bool
    json_filename: str
    timestamp: str
    summary: Dict[str, Any]