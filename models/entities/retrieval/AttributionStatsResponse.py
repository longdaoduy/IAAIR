from typing import Dict
from pydantic import BaseModel


class AttributionStatsResponse(BaseModel):
    """Response for attribution statistics."""
    success: bool
    timestamp: str
    total_queries_tracked: int
    attribution_accuracy: float
    high_confidence_rate: float
    source_type_distribution: Dict[str, int]
    average_attributions_per_result: float