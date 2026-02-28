"""
Performance regression testing framework for scientific RAG system.

This module provides automated testing to detect performance regressions
across different system configurations and data updates.
"""

import time
import logging
import json
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path

from models.entities.retrieval.SearchResult import SearchResult
from pipelines.evaluation.RetrievalEvaluator import RetrievalEvaluator, RetrievalBenchmark
from pipelines.evaluation.AttributionFidelityEvaluator import AttributionFidelityEvaluator, AttributionMetrics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceBaseline:
    """Baseline performance metrics for comparison."""
    benchmark_name: str
    version: str
    timestamp: datetime
    
    # Retrieval metrics
    avg_ndcg_at_10: float
    avg_ndcg_at_5: float
    avg_mrr: float
    avg_precision_at_10: float
    
    # Attribution metrics
    exact_span_match_rate: float
    citation_coverage: float
    attribution_precision: float
    
    # Performance metrics
    avg_latency_ms: float
    p95_latency_ms: float
    queries_per_second: float
    memory_usage_mb: float
    
    # System configuration
    config_hash: str
    model_versions: Dict[str, str]

@dataclass
class RegressionTestResult:
    """Result of regression test comparing current vs baseline."""
    test_name: str
    passed: bool
    current_metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    regressions: List[str]  # List of metrics that regressed
    improvements: List[str]  # List of metrics that improved
    timestamp: datetime
    
@dataclass
class PerformanceThresholds:
    """Acceptable thresholds for performance changes."""
    # Quality metrics (negative change is regression)
    ndcg_regression_threshold: float = -0.05  # 5% drop is regression
    attribution_regression_threshold: float = -0.10  # 10% drop is regression
    
    # Performance metrics (positive change is regression for latency)
    latency_regression_threshold: float = 0.20  # 20% increase is regression
    throughput_regression_threshold: float = -0.15  # 15% decrease is regression
    memory_regression_threshold: float = 0.25  # 25% increase is regression


class PerformanceRegressionTester:
    """Test for performance regressions in scientific RAG system."""
    
    def __init__(self, baseline_dir: str = "./evaluation_baselines"):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        self.thresholds = PerformanceThresholds()
        self.logger = logging.getLogger(__name__)
        
    def capture_performance_baseline(self, system_under_test: Any,
                                   benchmarks: List[RetrievalBenchmark],
                                   version: str = "1.0.0") -> PerformanceBaseline:
        """
        Capture comprehensive performance baseline.
        
        Args:
            system_under_test: System to test (should have search methods)
            benchmarks: Test queries for evaluation
            version: Version identifier
            
        Returns:
            Performance baseline measurements
        """
        self.logger.info(f"Capturing performance baseline for version {version}")
        
        # Initialize evaluators
        retrieval_evaluator = RetrievalEvaluator()
        attribution_evaluator = AttributionFidelityEvaluator()
        
        # Measure retrieval quality
        def retrieval_function(query):
            return system_under_test.search_similar_papers(query, top_k=20)
        
        eval_result = retrieval_evaluator.evaluate_benchmark_suite(benchmarks, retrieval_function)
        
        # Measure latency and throughput
        latencies = []
        memory_measurements = []
        
        # Run performance tests
        for benchmark in benchmarks[:10]:  # Use subset for performance testing
            start_time = time.time()
            
            # Memory usage before
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute query
            results = system_under_test.search_similar_papers(benchmark.query_text, top_k=10)
            
            # Memory usage after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_measurements.append(memory_after)
            
            # Record latency
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        qps = 1000 / avg_latency if avg_latency > 0 else 0  # Queries per second
        avg_memory = np.mean(memory_measurements)
        
        # Create baseline
        baseline = PerformanceBaseline(
            benchmark_name="Scientific RAG Regression Test",
            version=version,
            timestamp=datetime.now(),
            avg_ndcg_at_10=eval_result.avg_ndcg_at_10,
            avg_ndcg_at_5=eval_result.avg_ndcg_at_5,
            avg_mrr=eval_result.avg_mrr,
            avg_precision_at_10=eval_result.avg_precision_at_10,
            exact_span_match_rate=0.0,  # Would measure with attribution evaluator
            citation_coverage=0.0,
            attribution_precision=0.0,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            queries_per_second=qps,
            memory_usage_mb=avg_memory,
            config_hash=self._generate_config_hash(system_under_test),
            model_versions=self._get_model_versions(system_under_test)
        )
        
        # Save baseline
        self._save_baseline(baseline)
        
        return baseline
    
    def run_regression_test(self, system_under_test: Any,
                          benchmarks: List[RetrievalBenchmark],
                          baseline_version: str = "latest") -> RegressionTestResult:
        """
        Run regression test against saved baseline.
        
        Args:
            system_under_test: Current system to test
            benchmarks: Test queries
            baseline_version: Version to compare against
            
        Returns:
            Regression test results
        """
        self.logger.info(f"Running regression test against baseline {baseline_version}")
        
        # Load baseline
        baseline = self._load_baseline(baseline_version)
        if not baseline:
            self.logger.error(f"No baseline found for version {baseline_version}")
            raise ValueError(f"Baseline {baseline_version} not found")
        
        # Capture current performance
        current_baseline = self.capture_performance_baseline(
            system_under_test, benchmarks, "current"
        )
        
        # Compare metrics
        regressions = []
        improvements = []
        
        # Quality metrics (lower is worse)
        quality_comparisons = [
            ("nDCG@10", current_baseline.avg_ndcg_at_10, baseline.avg_ndcg_at_10, 
             self.thresholds.ndcg_regression_threshold),
            ("nDCG@5", current_baseline.avg_ndcg_at_5, baseline.avg_ndcg_at_5,
             self.thresholds.ndcg_regression_threshold),
            ("MRR", current_baseline.avg_mrr, baseline.avg_mrr,
             self.thresholds.ndcg_regression_threshold),
            ("Precision@10", current_baseline.avg_precision_at_10, baseline.avg_precision_at_10,
             self.thresholds.ndcg_regression_threshold),
            ("Attribution Precision", current_baseline.attribution_precision, baseline.attribution_precision,
             self.thresholds.attribution_regression_threshold)
        ]
        
        for metric_name, current, baseline_val, threshold in quality_comparisons:
            if baseline_val > 0:  # Avoid division by zero
                change = (current - baseline_val) / baseline_val
                if change < threshold:
                    regressions.append(f"{metric_name}: {change:.1%} drop (threshold: {threshold:.1%})")
                elif change > abs(threshold):  # Improvement
                    improvements.append(f"{metric_name}: {change:.1%} improvement")
        
        # Performance metrics (higher is worse for latency, lower is worse for throughput)
        performance_comparisons = [
            ("Average Latency", current_baseline.avg_latency_ms, baseline.avg_latency_ms,
             self.thresholds.latency_regression_threshold, True),  # True = higher is worse
            ("P95 Latency", current_baseline.p95_latency_ms, baseline.p95_latency_ms,
             self.thresholds.latency_regression_threshold, True),
            ("Queries/Second", current_baseline.queries_per_second, baseline.queries_per_second,
             self.thresholds.throughput_regression_threshold, False),  # False = lower is worse
            ("Memory Usage", current_baseline.memory_usage_mb, baseline.memory_usage_mb,
             self.thresholds.memory_regression_threshold, True)
        ]
        
        for metric_name, current, baseline_val, threshold, higher_is_worse in performance_comparisons:
            if baseline_val > 0:
                change = (current - baseline_val) / baseline_val
                
                if higher_is_worse:
                    if change > threshold:  # Regression for latency/memory
                        regressions.append(f"{metric_name}: {change:.1%} increase (threshold: {threshold:.1%})")
                    elif change < -abs(threshold):  # Improvement
                        improvements.append(f"{metric_name}: {abs(change):.1%} decrease")
                else:
                    if change < threshold:  # Regression for throughput
                        regressions.append(f"{metric_name}: {change:.1%} decrease (threshold: {threshold:.1%})")
                    elif change > abs(threshold):  # Improvement
                        improvements.append(f"{metric_name}: {change:.1%} increase")
        
        # Determine if test passed
        test_passed = len(regressions) == 0
        
        result = RegressionTestResult(
            test_name="Scientific RAG Regression Test",
            passed=test_passed,
            current_metrics=asdict(current_baseline),
            baseline_metrics=asdict(baseline),
            regressions=regressions,
            improvements=improvements,
            timestamp=datetime.now()
        )
        
        # Log results
        if test_passed:
            self.logger.info("✅ Regression test PASSED - No performance regressions detected")
        else:
            self.logger.warning(f"❌ Regression test FAILED - {len(regressions)} regressions detected")
            for regression in regressions:
                self.logger.warning(f"  • {regression}")
        
        if improvements:
            self.logger.info(f"🚀 Performance improvements detected:")
            for improvement in improvements:
                self.logger.info(f"  • {improvement}")
        
        return result
    
    def _generate_config_hash(self, system_under_test: Any) -> str:
        """Generate hash of system configuration."""
        # Simplified - in practice would hash actual config
        import hashlib
        config_str = str(type(system_under_test).__name__)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_model_versions(self, system_under_test: Any) -> Dict[str, str]:
        """Extract model versions from system."""
        # Simplified - in practice would extract actual model versions
        return {
            "scibert": "1.0.0",
            "clip": "1.0.0",
            "reranker": "1.0.0"
        }
    
    def _save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to disk."""
        baseline_file = self.baseline_dir / f"baseline_{baseline.version}.json"
        
        # Convert datetime to string for JSON serialization
        baseline_dict = asdict(baseline)
        baseline_dict['timestamp'] = baseline.timestamp.isoformat()
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_dict, f, indent=2)
        
        # Also save as 'latest' for easy reference
        latest_file = self.baseline_dir / "baseline_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(baseline_dict, f, indent=2)
        
        self.logger.info(f"Saved performance baseline to {baseline_file}")
    
    def _load_baseline(self, version: str) -> Optional[PerformanceBaseline]:
        """Load baseline from disk."""
        baseline_file = self.baseline_dir / f"baseline_{version}.json"
        
        if not baseline_file.exists():
            return None
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_dict = json.load(f)
            
            # Convert timestamp back from string
            baseline_dict['timestamp'] = datetime.fromisoformat(baseline_dict['timestamp'])
            
            return PerformanceBaseline(**baseline_dict)
        except Exception as e:
            self.logger.error(f"Error loading baseline {version}: {e}")
            return None
    
    def generate_regression_report(self, result: RegressionTestResult) -> str:
        """Generate human-readable regression test report."""
        
        status_emoji = "✅ PASSED" if result.passed else "❌ FAILED"
        
        report = f"""
Performance Regression Test Report
=================================
Test: {result.test_name}
Status: {status_emoji}
Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

"""

        if result.regressions:
            report += "🔴 REGRESSIONS DETECTED:\n"
            for regression in result.regressions:
                report += f"  • {regression}\n"
            report += "\n"
        
        if result.improvements:
            report += "🟢 IMPROVEMENTS DETECTED:\n"
            for improvement in result.improvements:
                report += f"  • {improvement}\n"
            report += "\n"
        
        # Key metrics comparison
        report += "📊 KEY METRICS COMPARISON:\n"
        report += f"{'Metric':<20} {'Current':<12} {'Baseline':<12} {'Change':<10}\n"
        report += "-" * 54 + "\n"
        
        key_metrics = [
            ("nDCG@10", "avg_ndcg_at_10"),
            ("MRR", "avg_mrr"),
            ("Latency (ms)", "avg_latency_ms"),
            ("QPS", "queries_per_second"),
            ("Memory (MB)", "memory_usage_mb")
        ]
        
        for name, key in key_metrics:
            current = result.current_metrics.get(key, 0)
            baseline = result.baseline_metrics.get(key, 0)
            change = ((current - baseline) / baseline * 100) if baseline > 0 else 0
            
            report += f"{name:<20} {current:<12.3f} {baseline:<12.3f} {change:+7.1f}%\n"
        
        report += "\n"
        
        # Recommendations
        if result.regressions:
            report += "📋 RECOMMENDATIONS:\n"
            report += "  • Review recent code changes that may impact performance\n"
            report += "  • Check if model updates or data changes caused regressions\n"
            report += "  • Consider reverting to previous version if critical regressions\n"
            report += "  • Run additional profiling to identify bottlenecks\n"
        else:
            report += "🎉 All metrics within acceptable thresholds!\n"
        
        return report