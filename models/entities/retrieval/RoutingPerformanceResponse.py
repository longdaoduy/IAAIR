from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator

class RoutingPerformanceResponse(BaseModel):
    """Response for routing performance metrics."""
    success: bool
    timestamp: str
    performance_metrics: Dict[str, Dict[str, float]]
    recommendations: List[str]