"""
Performance Monitor for IAAIR System

This module provides detailed performance tracking and latency breakdown
for identifying bottlenecks in the hybrid search pipeline, with Prometheus integration.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Detailed latency breakdown for a search operation."""
    query: str
    total_time: float
    embedding_time: float = 0.0
    vector_search_time: float = 0.0
    graph_search_time: float = 0.0
    fusion_time: float = 0.0
    reranking_time: float = 0.0
    ai_response_time: float = 0.0
    cache_hits: Dict[str, bool] = field(default_factory=dict)
    result_counts: Dict[str, int] = field(default_factory=dict)
    routing_strategy: str = "unknown"
    query_type: str = "unknown"
    endpoint: str = "hybrid-search"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_queries: int = 0
    avg_total_time: float = 0.0
    avg_embedding_time: float = 0.0
    avg_vector_search_time: float = 0.0
    avg_graph_search_time: float = 0.0
    avg_fusion_time: float = 0.0
    avg_reranking_time: float = 0.0
    avg_ai_response_time: float = 0.0
    cache_hit_rates: Dict[str, float] = field(default_factory=dict)
    slow_queries: List[LatencyBreakdown] = field(default_factory=list)  # Queries > 5s
    routing_breakdown: Dict[str, int] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor and track performance metrics across the system with Prometheus integration."""

    def __init__(self, slow_query_threshold: float = 5.0, enable_prometheus: bool = True):
        self.slow_query_threshold = slow_query_threshold
        self.current_breakdown: Optional[LatencyBreakdown] = None
        self.breakdowns: List[LatencyBreakdown] = []
        self.max_history = 1000  # Keep last 1000 queries

        # Initialize Prometheus integration
        self.prometheus_integration = None
        from models.engines.PrometheusMonitor import get_prometheus_integration, initialize_prometheus
        self.prometheus_integration = get_prometheus_integration()
        if self.prometheus_integration is None:
            self.prometheus_integration = initialize_prometheus()
        logger.info("Prometheus monitoring enabled for PerformanceMonitor")

    def start_query_tracking(self, query: str, endpoint: str = "hybrid-search") -> LatencyBreakdown:
        """Start tracking a new query."""
        self.current_breakdown = LatencyBreakdown(
            query=query,
            endpoint=endpoint,
            total_time=time.time()  # Will be converted to duration later
        )

        # Track active request in Prometheus
        if self.prometheus_integration:
            self.prometheus_integration.metrics.increment_active_requests()

        return self.current_breakdown

    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager to track individual operations with Prometheus integration."""
        if not self.current_breakdown:
            yield
            return

        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time

            # Map operation names to breakdown fields
            operation_mapping = {
                'embedding': 'embedding_time',
                'vector_search': 'vector_search_time',
                'graph_search': 'graph_search_time',
                'fusion': 'fusion_time',
                'reranking': 'reranking_time',
                'ai_response': 'ai_response_time'
            }

            if operation_name in operation_mapping:
                field_name = operation_mapping[operation_name]
                setattr(self.current_breakdown, field_name, elapsed)
                logger.debug(f"{operation_name} took {elapsed:.3f}s")

                # Record in Prometheus
                if self.prometheus_integration:
                    self.prometheus_integration.metrics.record_component_latency(operation_name, elapsed)

    def record_cache_hit(self, cache_type: str, hit: bool):
        """Record cache hit/miss with Prometheus integration."""
        if self.current_breakdown:
            self.current_breakdown.cache_hits[cache_type] = hit

            # Record in Prometheus
            if self.prometheus_integration:
                if hit:
                    self.prometheus_integration.metrics.record_cache_hit(cache_type)
                else:
                    self.prometheus_integration.metrics.record_cache_miss(cache_type)

    def record_result_count(self, source: str, count: int):
        """Record number of results from a source with Prometheus integration."""
        if self.current_breakdown:
            self.current_breakdown.result_counts[source] = count

            # Record in Prometheus
            if self.prometheus_integration:
                self.prometheus_integration.metrics.record_results(source, count)

    def record_routing_strategy(self, strategy: str):
        """Record the routing strategy used."""
        if self.current_breakdown:
            self.current_breakdown.routing_strategy = strategy

    def record_query_type(self, query_type: str):
        """Record the query type."""
        if self.current_breakdown:
            self.current_breakdown.query_type = query_type

    def record_error(self, error_type: str, component: str):
        """Record an error with Prometheus integration."""
        if self.prometheus_integration:
            self.prometheus_integration.metrics.record_error(error_type, component)

    def finish_query_tracking(self) -> Optional[LatencyBreakdown]:
        """Finish tracking current query and store results with Prometheus integration."""
        if not self.current_breakdown:
            return None

        # Convert start time to total duration
        self.current_breakdown.total_time = time.time() - self.current_breakdown.total_time

        # Record complete request in Prometheus
        if self.prometheus_integration:
            component_times = {
                'embedding': self.current_breakdown.embedding_time,
                'vector_search': self.current_breakdown.vector_search_time,
                'graph_search': self.current_breakdown.graph_search_time,
                'fusion': self.current_breakdown.fusion_time,
                'reranking': self.current_breakdown.reranking_time,
                'ai_response': self.current_breakdown.ai_response_time
            }

            self.prometheus_integration.record_search_request(
                endpoint=self.current_breakdown.endpoint,
                routing_strategy=self.current_breakdown.routing_strategy,
                query_type=self.current_breakdown.query_type,
                duration=self.current_breakdown.total_time,
                component_times=component_times,
                cache_hits=self.current_breakdown.cache_hits,
                result_counts=self.current_breakdown.result_counts
            )

            # Decrement active requests
            self.prometheus_integration.metrics.decrement_active_requests()

        # Store breakdown
        self.breakdowns.append(self.current_breakdown)

        # Keep only recent history
        if len(self.breakdowns) > self.max_history:
            self.breakdowns = self.breakdowns[-self.max_history:]

        # Log if slow query
        if self.current_breakdown.total_time > self.slow_query_threshold:
            logger.warning(
                f"Slow query detected ({self.current_breakdown.total_time:.2f}s): "
                f"{self.current_breakdown.query[:100]}..."
            )
            self._log_detailed_breakdown(self.current_breakdown)

        result = self.current_breakdown
        self.current_breakdown = None
        return result

    def update_cache_metrics(self, cache_stats: Dict[str, Dict[str, Any]]):
        """Update cache metrics in Prometheus."""
        if self.prometheus_integration:
            self.prometheus_integration.update_cache_metrics(cache_stats)

    def _log_detailed_breakdown(self, breakdown: LatencyBreakdown):
        """Log detailed breakdown for slow queries."""
        logger.warning("=== SLOW QUERY BREAKDOWN ===")
        logger.warning(f"Query: {breakdown.query}")
        logger.warning(f"Total time: {breakdown.total_time:.3f}s")
        logger.warning(
            f"  Embedding: {breakdown.embedding_time:.3f}s ({breakdown.embedding_time / breakdown.total_time * 100:.1f}%)")
        logger.warning(
            f"  Vector search: {breakdown.vector_search_time:.3f}s ({breakdown.vector_search_time / breakdown.total_time * 100:.1f}%)")
        logger.warning(
            f"  Graph search: {breakdown.graph_search_time:.3f}s ({breakdown.graph_search_time / breakdown.total_time * 100:.1f}%)")
        logger.warning(
            f"  Fusion: {breakdown.fusion_time:.3f}s ({breakdown.fusion_time / breakdown.total_time * 100:.1f}%)")
        logger.warning(
            f"  Reranking: {breakdown.reranking_time:.3f}s ({breakdown.reranking_time / breakdown.total_time * 100:.1f}%)")
        logger.warning(
            f"  AI response: {breakdown.ai_response_time:.3f}s ({breakdown.ai_response_time / breakdown.total_time * 100:.1f}%)")
        logger.warning(f"  Routing strategy: {breakdown.routing_strategy}")
        logger.warning(f"  Cache hits: {breakdown.cache_hits}")
        logger.warning(f"  Result counts: {breakdown.result_counts}")
        logger.warning("===========================")

    def get_performance_metrics(self, recent_queries: int = 100) -> PerformanceMetrics:
        """Get aggregated performance metrics."""
        if not self.breakdowns:
            return PerformanceMetrics()

        # Get recent breakdowns
        recent = self.breakdowns[-recent_queries:] if recent_queries else self.breakdowns

        if not recent:
            return PerformanceMetrics()

        # Calculate averages
        total_queries = len(recent)
        avg_total_time = sum(b.total_time for b in recent) / total_queries
        avg_embedding_time = sum(b.embedding_time for b in recent) / total_queries
        avg_vector_search_time = sum(b.vector_search_time for b in recent) / total_queries
        avg_graph_search_time = sum(b.graph_search_time for b in recent) / total_queries
        avg_fusion_time = sum(b.fusion_time for b in recent) / total_queries
        avg_reranking_time = sum(b.reranking_time for b in recent) / total_queries
        avg_ai_response_time = sum(b.ai_response_time for b in recent) / total_queries

        # Calculate cache hit rates
        cache_hit_rates = {}
        for cache_type in ['embedding', 'search', 'ai_response']:
            hits = sum(1 for b in recent if b.cache_hits.get(cache_type, False))
            total = sum(1 for b in recent if cache_type in b.cache_hits)
            cache_hit_rates[cache_type] = (hits / max(1, total)) * 100

        # Find slow queries
        slow_queries = [b for b in recent if b.total_time > self.slow_query_threshold]

        # Routing strategy breakdown
        routing_breakdown = {}
        for breakdown in recent:
            strategy = breakdown.routing_strategy
            routing_breakdown[strategy] = routing_breakdown.get(strategy, 0) + 1

        return PerformanceMetrics(
            total_queries=total_queries,
            avg_total_time=avg_total_time,
            avg_embedding_time=avg_embedding_time,
            avg_vector_search_time=avg_vector_search_time,
            avg_graph_search_time=avg_graph_search_time,
            avg_fusion_time=avg_fusion_time,
            avg_reranking_time=avg_reranking_time,
            avg_ai_response_time=avg_ai_response_time,
            cache_hit_rates=cache_hit_rates,
            slow_queries=slow_queries,
            routing_breakdown=routing_breakdown
        )

    def get_bottleneck_analysis(self, recent_queries: int = 100) -> Dict[str, Any]:
        """Analyze bottlenecks in the system."""
        metrics = self.get_performance_metrics(recent_queries)

        if metrics.total_queries == 0:
            return {"message": "No queries to analyze"}

        # Calculate relative time percentages
        total_avg = metrics.avg_total_time

        breakdown_percentages = {
            'embedding': (metrics.avg_embedding_time / total_avg) * 100 if total_avg > 0 else 0,
            'vector_search': (metrics.avg_vector_search_time / total_avg) * 100 if total_avg > 0 else 0,
            'graph_search': (metrics.avg_graph_search_time / total_avg) * 100 if total_avg > 0 else 0,
            'fusion': (metrics.avg_fusion_time / total_avg) * 100 if total_avg > 0 else 0,
            'reranking': (metrics.avg_reranking_time / total_avg) * 100 if total_avg > 0 else 0,
            'ai_response': (metrics.avg_ai_response_time / total_avg) * 100 if total_avg > 0 else 0
        }

        # Identify primary bottleneck
        primary_bottleneck = max(breakdown_percentages, key=breakdown_percentages.get)

        # Generate recommendations
        recommendations = []

        if breakdown_percentages['embedding'] > 20:
            recommendations.append("High embedding time - implement embedding caching")

        if breakdown_percentages['vector_search'] > 30:
            recommendations.append("High vector search time - optimize Milvus parameters (reduce nprobe, ef)")

        if breakdown_percentages['reranking'] > 25:
            recommendations.append("High reranking time - reduce candidates or implement selective reranking")

        if breakdown_percentages['ai_response'] > 40:
            recommendations.append("High AI response time - implement response caching or use faster model")

        if metrics.cache_hit_rates.get('embedding', 0) < 50:
            recommendations.append("Low embedding cache hit rate - increase cache size or TTL")

        if metrics.cache_hit_rates.get('search', 0) < 30:
            recommendations.append("Low search cache hit rate - analyze query patterns")

        return {
            'avg_total_time': round(metrics.avg_total_time, 3),
            'breakdown_percentages': {k: round(v, 1) for k, v in breakdown_percentages.items()},
            'primary_bottleneck': primary_bottleneck,
            'slow_queries_count': len(metrics.slow_queries),
            'cache_hit_rates': {k: round(v, 1) for k, v in metrics.cache_hit_rates.items()},
            'routing_breakdown': metrics.routing_breakdown,
            'recommendations': recommendations
        }

    def export_performance_report(self) -> str:
        """Export detailed performance report."""
        metrics = self.get_performance_metrics()
        analysis = self.get_bottleneck_analysis()

        report = "# IAAIR Performance Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## Summary\n"
        report += f"- Total queries analyzed: {metrics.total_queries}\n"
        report += f"- Average response time: {metrics.avg_total_time:.3f}s\n"
        report += f"- Slow queries (>{self.slow_query_threshold}s): {len(metrics.slow_queries)}\n\n"

        report += "## Time Breakdown\n"
        for operation, percentage in analysis['breakdown_percentages'].items():
            time_val = getattr(metrics, f'avg_{operation}_time', 0)
            report += f"- {operation.replace('_', ' ').title()}: {time_val:.3f}s ({percentage:.1f}%)\n"

        report += f"\n**Primary bottleneck:** {analysis['primary_bottleneck'].replace('_', ' ').title()}\n\n"

        report += "## Cache Performance\n"
        for cache_type, hit_rate in metrics.cache_hit_rates.items():
            report += f"- {cache_type.replace('_', ' ').title()}: {hit_rate:.1f}% hit rate\n"

        report += "\n## Routing Strategy Usage\n"
        for strategy, count in metrics.routing_breakdown.items():
            percentage = (count / metrics.total_queries) * 100
            report += f"- {strategy}: {count} queries ({percentage:.1f}%)\n"

        report += "\n## Recommendations\n"
        for i, rec in enumerate(analysis['recommendations'], 1):
            report += f"{i}. {rec}\n"

        if metrics.slow_queries:
            report += "\n## Slow Queries Sample\n"
            for i, sq in enumerate(metrics.slow_queries[:5], 1):
                report += f"\n### Query {i} ({sq.total_time:.2f}s)\n"
                report += f"**Query:** {sq.query[:100]}...\n"
                report += f"**Strategy:** {sq.routing_strategy}\n"
                report += f"**Breakdown:** "
                report += f"embed:{sq.embedding_time:.2f}s, "
                report += f"vector:{sq.vector_search_time:.2f}s, "
                report += f"graph:{sq.graph_search_time:.2f}s, "
                report += f"ai:{sq.ai_response_time:.2f}s\n"

        return report
