"""
Comprehensive evaluation coordinator for scientific RAG system.

This module orchestrates all evaluation components and provides
unified evaluation APIs for the scientific RAG system.
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from pipelines.evaluation.RetrievalEvaluator import (
    RetrievalEvaluator, 
    ScientificBenchmarkLoader,
    EvaluationResult
)
from pipelines.evaluation.AttributionFidelityEvaluator import (
    AttributionFidelityEvaluator,
    AttributionBenchmarkLoader,
    AttributionMetrics
)
from pipelines.evaluation.SciFractVerificationPipeline import (
    SciFractVerificationPipeline,
    VerificationEvaluator,
    create_verification_benchmarks
)
from pipelines.evaluation.PerformanceRegressionTester import (
    PerformanceRegressionTester,
    RegressionTestResult
)

logger = logging.getLogger(__name__)


class ComprehensiveEvaluationSuite:
    """Comprehensive evaluation suite for scientific RAG systems."""
    
    def __init__(self, system_under_test: Any):
        """
        Initialize evaluation suite.
        
        Args:
            system_under_test: The RAG system to evaluate
        """
        self.system = system_under_test
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluators
        self.retrieval_evaluator = RetrievalEvaluator()
        self.attribution_evaluator = AttributionFidelityEvaluator()
        self.verification_pipeline = SciFractVerificationPipeline(
            retrieval_client=getattr(system_under_test, 'retrieval_handler', None),
            llm_client=getattr(system_under_test, 'deepseek_client', None)
        )
        self.verification_evaluator = VerificationEvaluator()
        self.regression_tester = PerformanceRegressionTester()
        
        # Initialize benchmark loaders
        self.benchmark_loader = ScientificBenchmarkLoader()
        self.attribution_benchmark_loader = AttributionBenchmarkLoader()
        
    def run_full_evaluation(self, version: str = "current") -> Dict[str, Any]:
        """
        Run comprehensive evaluation across all dimensions.
        
        Args:
            version: Version identifier for this evaluation
            
        Returns:
            Complete evaluation results
        """
        self.logger.info(f"Starting comprehensive evaluation for version {version}")
        
        results = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'retrieval_quality': None,
            'attribution_fidelity': None,
            'verification_accuracy': None,
            'performance_regression': None,
            'overall_score': None
        }
        
        try:
            # 1. Retrieval Quality Evaluation
            self.logger.info("Running retrieval quality evaluation...")
            retrieval_results = self._evaluate_retrieval_quality()
            results['retrieval_quality'] = retrieval_results
            
            # 2. Attribution Fidelity Evaluation
            self.logger.info("Running attribution fidelity evaluation...")
            attribution_results = self._evaluate_attribution_fidelity()
            results['attribution_fidelity'] = attribution_results
            
            # 3. Verification Accuracy Evaluation
            self.logger.info("Running verification accuracy evaluation...")
            verification_results = self._evaluate_verification_accuracy()
            results['verification_accuracy'] = verification_results
            
            # 4. Performance Regression Testing
            self.logger.info("Running performance regression testing...")
            regression_results = self._run_performance_regression_test(version)
            results['performance_regression'] = regression_results
            
            # 5. Calculate Overall Score
            overall_score = self._calculate_overall_score(results)
            results['overall_score'] = overall_score
            
            self.logger.info(f"Comprehensive evaluation completed. Overall score: {overall_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive evaluation: {e}")
            results['error'] = str(e)
        
        return results
    
    def _evaluate_retrieval_quality(self) -> Dict[str, Any]:
        """Evaluate retrieval quality using nDCG@k and other metrics."""
        
        # Load benchmarks
        benchmarks = self.benchmark_loader.load_default_benchmarks()
        
        # Define retrieval function
        def retrieval_function(query_text: str):
            if hasattr(self.system, 'retrieval_handler'):
                return self.system.retrieval_handler.search_similar_papers(
                    query_text=query_text, 
                    top_k=20,
                    use_hybrid=True
                )
            else:
                # Fallback for simpler systems
                return []
        
        # Run evaluation
        evaluation_result = self.retrieval_evaluator.evaluate_benchmark_suite(
            benchmarks, retrieval_function
        )
        
        return {
            'avg_ndcg_at_10': evaluation_result.avg_ndcg_at_10,
            'avg_ndcg_at_5': evaluation_result.avg_ndcg_at_5, 
            'avg_mrr': evaluation_result.avg_mrr,
            'avg_precision_at_10': evaluation_result.avg_precision_at_10,
            'avg_recall_at_10': evaluation_result.avg_recall_at_10,
            'by_query_type': evaluation_result.by_query_type,
            'by_domain': evaluation_result.by_domain,
            'total_queries': evaluation_result.total_queries,
            'details': evaluation_result.details
        }
    
    def _evaluate_attribution_fidelity(self) -> Dict[str, Any]:
        """Evaluate attribution fidelity and source tracking."""
        
        # Load attribution benchmarks
        benchmarks = self.attribution_benchmark_loader.load_default_attribution_benchmarks()
        
        # Generate search results with attributions
        search_results = []
        for benchmark in benchmarks:
            if hasattr(self.system, 'retrieval_handler'):
                results = self.system.retrieval_handler.search_similar_papers(
                    query_text=benchmark.query_text,
                    top_k=10,
                    use_hybrid=True
                )
                
                # Add attribution tracking if available
                if hasattr(self.system, 'attribution_tracker') and results:
                    results = self.system.attribution_tracker.track_attributions(
                        results, benchmark.query_text
                    )
                
                search_results.extend(results)
        
        # Evaluate attribution quality
        attribution_metrics = self.attribution_evaluator.evaluate_attribution_quality(
            search_results, benchmarks
        )
        
        return {
            'exact_span_match_rate': attribution_metrics.exact_span_match_rate,
            'partial_span_match_rate': attribution_metrics.partial_span_match_rate,
            'citation_coverage': attribution_metrics.citation_coverage,
            'wrong_source_rate': attribution_metrics.wrong_source_rate,
            'attribution_precision': attribution_metrics.attribution_precision,
            'attribution_recall': attribution_metrics.attribution_recall,
            'average_confidence': attribution_metrics.average_confidence,
            'high_confidence_rate': attribution_metrics.high_confidence_rate
        }
    
    def _evaluate_verification_accuracy(self) -> Dict[str, Any]:
        """Evaluate scientific claim verification accuracy."""
        
        # Load verification benchmarks
        benchmarks = create_verification_benchmarks()
        
        # Run verification on each benchmark
        verification_results = []
        for benchmark in benchmarks:
            try:
                result = self.verification_pipeline.verify_claim(benchmark.claim)
                verification_results.append(result)
            except Exception as e:
                self.logger.warning(f"Error verifying claim {benchmark.claim.claim_id}: {e}")
                continue
        
        # Evaluate verification accuracy
        metrics = self.verification_evaluator.evaluate_verification(
            verification_results, benchmarks
        )
        
        return {
            'accuracy': metrics.get('accuracy', 0.0),
            'supports_precision': metrics.get('supports_precision', 0.0),
            'supports_recall': metrics.get('supports_recall', 0.0),
            'supports_f1': metrics.get('supports_f1', 0.0),
            'refutes_precision': metrics.get('refutes_precision', 0.0),
            'refutes_recall': metrics.get('refutes_recall', 0.0),
            'refutes_f1': metrics.get('refutes_f1', 0.0),
            'total_verified_claims': len(verification_results)
        }
    
    def _run_performance_regression_test(self, version: str) -> Dict[str, Any]:
        """Run performance regression testing."""
        
        try:
            # Load retrieval benchmarks for performance testing
            benchmarks = self.benchmark_loader.load_default_benchmarks()
            
            # Try to run regression test against existing baseline
            try:
                result = self.regression_tester.run_regression_test(
                    self.system, benchmarks, "latest"
                )
                return {
                    'test_passed': result.passed,
                    'regressions': result.regressions,
                    'improvements': result.improvements,
                    'current_metrics': result.current_metrics,
                    'baseline_metrics': result.baseline_metrics
                }
            except ValueError:
                # No baseline exists, create one
                self.logger.info("No baseline found, creating new baseline")
                baseline = self.regression_tester.capture_performance_baseline(
                    self.system, benchmarks, version
                )
                return {
                    'test_passed': True,
                    'baseline_created': True,
                    'baseline_metrics': baseline.__dict__,
                    'regressions': [],
                    'improvements': []
                }
                
        except Exception as e:
            self.logger.error(f"Error in performance regression testing: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall evaluation score."""
        
        scores = []
        weights = []
        
        # Retrieval quality (40% weight)
        if results.get('retrieval_quality'):
            retrieval_score = (
                results['retrieval_quality'].get('avg_ndcg_at_10', 0) * 0.4 +
                results['retrieval_quality'].get('avg_mrr', 0) * 0.3 +
                results['retrieval_quality'].get('avg_precision_at_10', 0) * 0.3
            )
            scores.append(retrieval_score)
            weights.append(0.4)
        
        # Attribution fidelity (30% weight)
        if results.get('attribution_fidelity'):
            attribution_score = (
                results['attribution_fidelity'].get('exact_span_match_rate', 0) * 0.3 +
                results['attribution_fidelity'].get('citation_coverage', 0) * 0.3 +
                results['attribution_fidelity'].get('attribution_precision', 0) * 0.2 +
                (1 - results['attribution_fidelity'].get('wrong_source_rate', 1)) * 0.2
            )
            scores.append(attribution_score)
            weights.append(0.3)
        
        # Verification accuracy (20% weight)
        if results.get('verification_accuracy'):
            verification_score = results['verification_accuracy'].get('accuracy', 0)
            scores.append(verification_score)
            weights.append(0.2)
        
        # Performance regression (10% weight - binary pass/fail)
        if results.get('performance_regression'):
            performance_score = 1.0 if results['performance_regression'].get('test_passed', False) else 0.5
            scores.append(performance_score)
            weights.append(0.1)
        
        # Calculate weighted average
        if scores and weights:
            overall_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return overall_score
        else:
            return 0.0
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report."""
        
        report = f"""
Scientific RAG System Evaluation Report
======================================
Version: {results.get('version', 'Unknown')}
Timestamp: {results.get('timestamp', 'Unknown')}
Overall Score: {results.get('overall_score', 0):.2f}/1.0

"""

        # Overall assessment
        overall_score = results.get('overall_score', 0)
        if overall_score >= 0.85:
            status = "🟢 EXCELLENT - Production Ready"
        elif overall_score >= 0.70:
            status = "🟡 GOOD - Minor Improvements Needed"
        elif overall_score >= 0.50:
            status = "🟠 ACCEPTABLE - Significant Improvements Needed"
        else:
            status = "🔴 POOR - Major Issues Detected"
        
        report += f"Status: {status}\n\n"
        
        # Retrieval Quality
        if results.get('retrieval_quality'):
            rq = results['retrieval_quality']
            report += f"""📊 RETRIEVAL QUALITY:
  • nDCG@10: {rq.get('avg_ndcg_at_10', 0):.3f}
  • nDCG@5:  {rq.get('avg_ndcg_at_5', 0):.3f}
  • MRR:     {rq.get('avg_mrr', 0):.3f}
  • Precision@10: {rq.get('avg_precision_at_10', 0):.3f}
  • Recall@10:    {rq.get('avg_recall_at_10', 0):.3f}
  • Queries Evaluated: {rq.get('total_queries', 0)}

"""
        
        # Attribution Fidelity
        if results.get('attribution_fidelity'):
            af = results['attribution_fidelity']
            report += f"""🎯 ATTRIBUTION FIDELITY:
  • Exact Span Match Rate: {af.get('exact_span_match_rate', 0):.1%}
  • Partial Span Match Rate: {af.get('partial_span_match_rate', 0):.1%}
  • Citation Coverage: {af.get('citation_coverage', 0):.1%}
  • Wrong Source Rate: {af.get('wrong_source_rate', 0):.1%}
  • Attribution Precision: {af.get('attribution_precision', 0):.1%}
  • Average Confidence: {af.get('average_confidence', 0):.2f}

"""
        
        # Verification Accuracy  
        if results.get('verification_accuracy'):
            va = results['verification_accuracy']
            report += f"""✅ VERIFICATION ACCURACY:
  • Overall Accuracy: {va.get('accuracy', 0):.1%}
  • Supports F1: {va.get('supports_f1', 0):.1%}
  • Refutes F1: {va.get('refutes_f1', 0):.1%}
  • Claims Verified: {va.get('total_verified_claims', 0)}

"""
        
        # Performance Regression
        if results.get('performance_regression'):
            pr = results['performance_regression']
            status = "✅ PASSED" if pr.get('test_passed', False) else "❌ FAILED"
            report += f"""⚡ PERFORMANCE REGRESSION: {status}
  • Regressions Detected: {len(pr.get('regressions', []))}
  • Performance Improvements: {len(pr.get('improvements', []))}

"""
            
            if pr.get('regressions'):
                report += "  Regressions:\n"
                for reg in pr['regressions']:
                    report += f"    • {reg}\n"
                report += "\n"
        
        # Recommendations
        report += "📋 RECOMMENDATIONS:\n"
        
        if overall_score >= 0.85:
            report += "  • System performance is excellent, ready for production\n"
            report += "  • Consider monitoring for regressions with regular evaluations\n"
        elif overall_score >= 0.70:
            report += "  • Good performance with room for improvement\n"
            report += "  • Focus on areas with lower scores\n"
        else:
            report += "  • Significant improvements needed before production deployment\n"
            report += "  • Review system architecture and model performance\n"
            
        if results.get('retrieval_quality', {}).get('avg_ndcg_at_10', 0) < 0.6:
            report += "  • Improve retrieval quality through better embeddings or ranking\n"
            
        if results.get('attribution_fidelity', {}).get('exact_span_match_rate', 0) < 0.7:
            report += "  • Enhance attribution accuracy and span detection\n"
            
        if results.get('performance_regression', {}).get('regressions'):
            report += "  • Address performance regressions before deployment\n"
        
        return report