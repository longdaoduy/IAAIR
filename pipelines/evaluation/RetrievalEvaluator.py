"""
nDCG@k and MRR evaluation framework for scientific retrieval.

This module implements comprehensive retrieval quality metrics
with scientific domain-specific considerations.
"""

import math
import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime
from dataclasses import dataclass

from models.entities.retrieval.SearchResult import SearchResult

logger = logging.getLogger(__name__)

@dataclass
class RetrievalBenchmark:
    """Single benchmark query with ground truth relevance."""
    query_id: str
    query_text: str
    relevant_paper_ids: List[str]  # Ground truth relevant papers
    relevance_scores: Dict[str, float]  # paper_id -> relevance score (0-3 scale)
    query_type: str  # 'semantic', 'structural', 'factual', 'hybrid'
    domain: Optional[str] = None  # 'biomedical', 'cs', 'physics', etc.

@dataclass 
class EvaluationResult:
    """Results from evaluation run."""
    benchmark_name: str
    total_queries: int
    avg_ndcg_at_10: float
    avg_ndcg_at_5: float
    avg_mrr: float
    avg_precision_at_10: float
    avg_recall_at_10: float
    by_query_type: Dict[str, Dict[str, float]]
    by_domain: Dict[str, Dict[str, float]]
    timestamp: datetime
    details: List[Dict[str, Any]]


class RetrievalEvaluator:
    """Evaluator for scientific retrieval systems using standard IR metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def ndcg_at_k(self, retrieved_results: List[SearchResult], 
                  ground_truth: Dict[str, float], k: int = 10) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            retrieved_results: List of search results in ranked order
            ground_truth: Dict mapping paper_id to relevance score (0-3)
            k: Cutoff for evaluation
            
        Returns:
            nDCG@k score (0.0 to 1.0)
        """
        if not retrieved_results or not ground_truth:
            return 0.0
            
        # Calculate DCG@k
        dcg = 0.0
        for i, result in enumerate(retrieved_results[:k]):
            relevance = ground_truth.get(result.paper_id, 0.0)
            if relevance > 0:
                # DCG formula: rel / log2(rank + 1)
                dcg += relevance / math.log2(i + 2)
        
        # Calculate IDCG@k (ideal DCG)
        sorted_relevance = sorted(ground_truth.values(), reverse=True)[:k]
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevance):
            if relevance > 0:
                idcg += relevance / math.log2(i + 2)
        
        # Return normalized DCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def mrr(self, retrieved_results: List[SearchResult], 
            relevant_paper_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            retrieved_results: List of search results in ranked order
            relevant_paper_ids: List of relevant paper IDs
            
        Returns:
            MRR score (0.0 to 1.0)
        """
        if not retrieved_results or not relevant_paper_ids:
            return 0.0
            
        for i, result in enumerate(retrieved_results):
            if result.paper_id in relevant_paper_ids:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def precision_at_k(self, retrieved_results: List[SearchResult],
                       relevant_paper_ids: List[str], k: int = 10) -> float:
        """Calculate Precision@k."""
        if not retrieved_results or not relevant_paper_ids:
            return 0.0
            
        retrieved_k = retrieved_results[:k]
        relevant_retrieved = sum(1 for r in retrieved_k if r.paper_id in relevant_paper_ids)
        
        return relevant_retrieved / min(len(retrieved_k), k)
    
    def recall_at_k(self, retrieved_results: List[SearchResult],
                    relevant_paper_ids: List[str], k: int = 10) -> float:
        """Calculate Recall@k."""
        if not retrieved_results or not relevant_paper_ids:
            return 0.0
            
        retrieved_k = retrieved_results[:k]
        relevant_retrieved = sum(1 for r in retrieved_k if r.paper_id in relevant_paper_ids)
        
        return relevant_retrieved / len(relevant_paper_ids)
    
    def evaluate_single_query(self, retrieved_results: List[SearchResult],
                              benchmark: RetrievalBenchmark) -> Dict[str, float]:
        """Evaluate a single query against its benchmark."""
        metrics = {
            'ndcg_at_5': self.ndcg_at_k(retrieved_results, benchmark.relevance_scores, 5),
            'ndcg_at_10': self.ndcg_at_k(retrieved_results, benchmark.relevance_scores, 10),
            'mrr': self.mrr(retrieved_results, benchmark.relevant_paper_ids),
            'precision_at_10': self.precision_at_k(retrieved_results, benchmark.relevant_paper_ids, 10),
            'recall_at_10': self.recall_at_k(retrieved_results, benchmark.relevant_paper_ids, 10),
        }
        
        return metrics
    
    def evaluate_benchmark_suite(self, benchmarks: List[RetrievalBenchmark],
                                 retrieval_function) -> EvaluationResult:
        """
        Evaluate retrieval system on a suite of benchmarks.
        
        Args:
            benchmarks: List of benchmark queries
            retrieval_function: Function that takes query and returns List[SearchResult]
            
        Returns:
            Comprehensive evaluation results
        """
        all_metrics = []
        query_type_metrics = {}
        domain_metrics = {}
        
        self.logger.info(f"Evaluating {len(benchmarks)} benchmark queries...")
        
        for benchmark in benchmarks:
            try:
                # Run retrieval
                retrieved_results = retrieval_function(benchmark.query_text)
                
                # Calculate metrics
                metrics = self.evaluate_single_query(retrieved_results, benchmark)
                metrics['query_id'] = benchmark.query_id
                metrics['query_type'] = benchmark.query_type
                metrics['domain'] = benchmark.domain or 'unknown'
                
                all_metrics.append(metrics)
                
                # Group by query type
                if benchmark.query_type not in query_type_metrics:
                    query_type_metrics[benchmark.query_type] = []
                query_type_metrics[benchmark.query_type].append(metrics)
                
                # Group by domain
                domain = benchmark.domain or 'unknown'
                if domain not in domain_metrics:
                    domain_metrics[domain] = []
                domain_metrics[domain].append(metrics)
                
            except Exception as e:
                self.logger.error(f"Error evaluating query {benchmark.query_id}: {e}")
                continue
        
        # Calculate aggregate metrics
        def avg_metrics(metrics_list):
            if not metrics_list:
                return {}
            return {
                'ndcg_at_5': np.mean([m['ndcg_at_5'] for m in metrics_list]),
                'ndcg_at_10': np.mean([m['ndcg_at_10'] for m in metrics_list]),
                'mrr': np.mean([m['mrr'] for m in metrics_list]),
                'precision_at_10': np.mean([m['precision_at_10'] for m in metrics_list]),
                'recall_at_10': np.mean([m['recall_at_10'] for m in metrics_list]),
                'count': len(metrics_list)
            }
        
        # Aggregate by query type
        by_query_type = {}
        for query_type, metrics_list in query_type_metrics.items():
            by_query_type[query_type] = avg_metrics(metrics_list)
        
        # Aggregate by domain
        by_domain = {}
        for domain, metrics_list in domain_metrics.items():
            by_domain[domain] = avg_metrics(metrics_list)
        
        return EvaluationResult(
            benchmark_name="Scientific Retrieval Benchmark",
            total_queries=len(all_metrics),
            avg_ndcg_at_10=np.mean([m['ndcg_at_10'] for m in all_metrics]) if all_metrics else 0.0,
            avg_ndcg_at_5=np.mean([m['ndcg_at_5'] for m in all_metrics]) if all_metrics else 0.0,
            avg_mrr=np.mean([m['mrr'] for m in all_metrics]) if all_metrics else 0.0,
            avg_precision_at_10=np.mean([m['precision_at_10'] for m in all_metrics]) if all_metrics else 0.0,
            avg_recall_at_10=np.mean([m['recall_at_10'] for m in all_metrics]) if all_metrics else 0.0,
            by_query_type=by_query_type,
            by_domain=by_domain,
            timestamp=datetime.now(),
            details=all_metrics
        )


class ScientificBenchmarkLoader:
    """Load and manage scientific retrieval benchmarks."""
    
    def __init__(self):
        self.benchmarks = []
        
    def load_default_benchmarks(self) -> List[RetrievalBenchmark]:
        """Load default scientific benchmarks for evaluation."""
        
        # Scientific query benchmarks with ground truth
        benchmarks = [
            RetrievalBenchmark(
                query_id="sci_001",
                query_text="CRISPR gene editing applications in cancer treatment",
                relevant_paper_ids=["W2963920772", "W2951235503", "W2748394829"],
                relevance_scores={
                    "W2963920772": 3.0,  # Highly relevant
                    "W2951235503": 2.0,  # Relevant  
                    "W2748394829": 2.0,  # Relevant
                    "W2123456789": 1.0   # Somewhat relevant
                },
                query_type="semantic",
                domain="biomedical"
            ),
            RetrievalBenchmark(
                query_id="sci_002", 
                query_text="papers authored by Geoffrey Hinton on deep learning",
                relevant_paper_ids=["W2950635152", "W2123456789"],
                relevance_scores={
                    "W2950635152": 3.0,
                    "W2123456789": 2.0
                },
                query_type="structural",
                domain="cs"
            ),
            RetrievalBenchmark(
                query_id="sci_003",
                query_text="transformer architecture attention mechanism neural networks",
                relevant_paper_ids=["W2964069955", "W2950635152", "W2963920772"],
                relevance_scores={
                    "W2964069955": 3.0,  # "Attention is All You Need"
                    "W2950635152": 2.0,
                    "W2963920772": 1.0
                },
                query_type="semantic", 
                domain="cs"
            ),
            RetrievalBenchmark(
                query_id="sci_004",
                query_text="what is the impact factor of Nature journal",
                relevant_paper_ids=["W1234567890"],  # Hypothetical paper about Nature
                relevance_scores={
                    "W1234567890": 3.0
                },
                query_type="factual",
                domain="general"
            ),
            RetrievalBenchmark(
                query_id="sci_005",
                query_text="machine learning applications in drug discovery and molecular design",
                relevant_paper_ids=["W2963920772", "W2748394829", "W2951235503"],
                relevance_scores={
                    "W2963920772": 3.0,
                    "W2748394829": 2.0,
                    "W2951235503": 2.0
                },
                query_type="hybrid",
                domain="biomedical"
            )
        ]
        
        self.benchmarks = benchmarks
        return benchmarks
    
    def create_custom_benchmark(self, query_text: str, relevant_papers: List[str],
                                query_type: str = "semantic", domain: str = "general") -> RetrievalBenchmark:
        """Create a custom benchmark for evaluation."""
        query_id = f"custom_{len(self.benchmarks) + 1:03d}"
        
        # Default relevance scores (all relevant papers get score 2.0)
        relevance_scores = {paper_id: 2.0 for paper_id in relevant_papers}
        
        benchmark = RetrievalBenchmark(
            query_id=query_id,
            query_text=query_text,
            relevant_paper_ids=relevant_papers,
            relevance_scores=relevance_scores,
            query_type=query_type,
            domain=domain
        )
        
        self.benchmarks.append(benchmark)
        return benchmark