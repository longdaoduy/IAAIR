from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SciMMIRBenchmarkResult:
    """Results from SciMMIR benchmark evaluations."""
    model_name: str
    benchmark_name: str
    total_samples: int

    # Text-to-Image retrievals metrics
    text2img_mrr: float
    text2img_recall_at_1: float
    text2img_recall_at_5: float
    text2img_recall_at_10: float

    # Image-to-Text retrievals metrics
    img2text_mrr: float
    img2text_recall_at_1: float
    img2text_recall_at_5: float
    img2text_recall_at_10: float

    timestamp: datetime
    evaluation_details: Dict[str, Any]

    # Subset-specific results (like CLIP-BERT evaluations) - must be after non-default fields
    subset_results: Optional[Dict[str, Dict[str, float]]] = None  # e.g., "figure_result": {"text2img_mrr": 0.12, ...}

