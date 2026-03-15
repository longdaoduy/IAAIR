from typing import Dict
from dataclasses import dataclass

@dataclass
class MockEvaluationSummary:
    """Summary of mock evaluations results."""
    total_questions: int
    successful_questions: int
    failed_questions: int
    avg_response_time: float
    overall_precision: float
    overall_recall: float
    overall_f1: float
    avg_ai_response_similarity: float
    avg_ai_generation_time: float
    ai_response_success_rate: float
    avg_dcg_at_5: float
    avg_dcg_at_10: float
    avg_ndcg_at_5: float
    avg_ndcg_at_10: float
    graph_performance: Dict[str, float]
    semantic_performance: Dict[str, float]
    category_breakdown: Dict[str, Dict[str, float]]