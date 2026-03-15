from typing import  List, Optional
from dataclasses import dataclass

@dataclass
class MockEvaluationResult:
    """Results from mock data evaluations."""
    question_id: str
    question: str
    question_type: str  # 'neo4j' or 'semantic'
    category: str
    success: bool
    response_time: float
    retrieved_papers: List[str]
    expected_papers: List[str]
    precision: float
    recall: float
    f1_score: float
    ai_response: Optional[str] = None
    expected_ai_response: Optional[str] = None
    ai_response_similarity: Optional[float] = None
    ai_generation_time: Optional[float] = None
    error_message: Optional[str] = None
    similarity_scores: Optional[List[float]] = None
    dcg_at_5: Optional[float] = None
    dcg_at_10: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    verification_labels: Optional[List[str]] = None