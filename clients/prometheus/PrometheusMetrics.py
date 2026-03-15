
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry,
    generate_latest,
)
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """Prometheus metrics collector for IAAIR system."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics."""
        self.registry = registry or CollectorRegistry()

        # Request metrics
        self.request_count = Counter(
            'iaair_requests_total',
            'Total number of search requests',
            ['endpoint', 'method', 'routing_strategy', 'query_type'],
            registry=self.registry
        )

        # Detailed endpoint metrics
        self.endpoint_requests = Counter(
            'iaair_endpoint_requests_total',
            'Total requests per endpoint',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )

        # Request duration with more labels
        self.request_duration = Histogram(
            'iaair_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint', 'routing_strategy'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )

        # Component latency metrics
        self.embedding_duration = Histogram(
            'iaair_embedding_duration_seconds',
            'Embedding generation duration in seconds',
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            registry=self.registry
        )

        self.vector_search_duration = Histogram(
            'iaair_vector_search_duration_seconds',
            'Vector search duration in seconds',
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        self.graph_search_duration = Histogram(
            'iaair_graph_search_duration_seconds',
            'Graph search duration in seconds',
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        self.reranking_duration = Histogram(
            'iaair_reranking_duration_seconds',
            'Reranking duration in seconds',
            buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )

        self.ai_response_duration = Histogram(
            'iaair_ai_response_duration_seconds',
            'AI response generation duration in seconds',
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
            registry=self.registry
        )

        # LLM call metrics
        self.llm_calls = Counter(
            'iaair_llm_calls_total',
            'Total LLM generation calls',
            ['purpose'],
            registry=self.registry
        )

        self.llm_call_duration = Histogram(
            'iaair_llm_call_duration_seconds',
            'LLM call duration in seconds by purpose',
            ['purpose'],
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0],
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            'iaair_cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )

        self.cache_misses = Counter(
            'iaair_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )

        self.cache_size = Gauge(
            'iaair_cache_size',
            'Current cache size',
            ['cache_type'],
            registry=self.registry
        )

        # Result metrics
        self.results_count = Histogram(
            'iaair_results_count',
            'Number of results returned',
            ['search_type'],
            buckets=[0, 1, 5, 10, 20, 50, 100],
            registry=self.registry
        )

        # Error metrics
        self.errors = Counter(
            'iaair_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )

        # System metrics
        self.active_requests = Gauge(
            'iaair_active_requests',
            'Number of active requests',
            registry=self.registry
        )

        # Routing strategy effectiveness
        self.routing_strategy_performance = Histogram(
            'iaair_routing_strategy_performance_seconds',
            'Performance by routing strategy',
            ['strategy', 'query_type'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            registry=self.registry
        )

        # System info
        self.system_info = Info(
            'iaair_system_info',
            'System information',
            registry=self.registry
        )

        # Set system info
        self.system_info.info({
            'version': '2.0.0',
            'component': 'iaair-hybrid-search',
            'features': 'caching,smart_routing,selective_reranking'
        })

        self._lock = threading.Lock()

    def record_request(self, endpoint: str, routing_strategy: str, query_type: str, duration: float,
                       method: str = "POST"):
        """Record a search request."""
        with self._lock:
            self.request_count.labels(
                endpoint=endpoint,
                method=method,
                routing_strategy=routing_strategy,
                query_type=query_type
            ).inc()

            self.request_duration.labels(
                endpoint=endpoint,
                routing_strategy=routing_strategy
            ).observe(duration)

            self.routing_strategy_performance.labels(
                strategy=routing_strategy,
                query_type=query_type
            ).observe(duration)

    def record_component_latency(self, component: str, duration: float):
        """Record component latency."""
        with self._lock:
            if component == 'embedding':
                self.embedding_duration.observe(duration)
            elif component == 'vector_search':
                self.vector_search_duration.observe(duration)
            elif component == 'graph_search':
                self.graph_search_duration.observe(duration)
            elif component == 'reranking':
                self.reranking_duration.observe(duration)
            elif component == 'ai_response':
                self.ai_response_duration.observe(duration)

    def record_llm_call(self, purpose: str, duration: float):
        """Record an LLM generation call."""
        with self._lock:
            self.llm_calls.labels(purpose=purpose).inc()
            self.llm_call_duration.labels(purpose=purpose).observe(duration)

    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        with self._lock:
            self.cache_hits.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        with self._lock:
            self.cache_misses.labels(cache_type=cache_type).inc()

    def update_cache_size(self, cache_type: str, size: int):
        """Update cache size gauge."""
        with self._lock:
            self.cache_size.labels(cache_type=cache_type).set(size)

    def record_results(self, search_type: str, count: int):
        """Record number of results."""
        with self._lock:
            self.results_count.labels(search_type=search_type).observe(count)

    def record_error(self, error_type: str, component: str):
        """Record an error."""
        with self._lock:
            self.errors.labels(error_type=error_type, component=component).inc()

    def increment_active_requests(self):
        """Increment active requests counter."""
        with self._lock:
            self.active_requests.inc()

    def decrement_active_requests(self):
        """Decrement active requests counter."""
        with self._lock:
            self.active_requests.dec()

    @contextmanager
    def track_active_request(self):
        """Context manager to track active requests."""
        self.increment_active_requests()
        try:
            yield
        finally:
            self.decrement_active_requests()

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')

    def track_endpoint_request(self, endpoint: str, method: str, status: str):
        """Track a request to a specific endpoint."""
        with self._lock:
            self.endpoint_requests.labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).inc()

    def get_endpoint_statistics(self) -> Dict[str, Any]:
        """Get detailed endpoint statistics from Prometheus metrics."""
        try:
            endpoint_stats = {}

            # Collect endpoint request metrics
            for metric_family in self.registry.collect():
                if metric_family.name == 'iaair_endpoint_requests_total':
                    for sample in metric_family.samples:
                        endpoint = sample.labels.get('endpoint', 'unknown')
                        method = sample.labels.get('method', 'unknown')
                        status = sample.labels.get('status', 'unknown')

                        key = f"{method} {endpoint}"
                        if key not in endpoint_stats:
                            endpoint_stats[key] = {
                                'endpoint': endpoint,
                                'method': method,
                                'success_count': 0,
                                'error_count': 0,
                                'total_count': 0
                            }

                        if status == 'success':
                            endpoint_stats[key]['success_count'] = int(sample.value)
                        elif status == 'error':
                            endpoint_stats[key]['error_count'] = int(sample.value)

            # Calculate totals and rates
            for stats in endpoint_stats.values():
                stats['total_count'] = stats['success_count'] + stats['error_count']
                if stats['total_count'] > 0:
                    stats['success_rate'] = stats['success_count'] / stats['total_count']
                    stats['error_rate'] = stats['error_count'] / stats['total_count']
                else:
                    stats['success_rate'] = 0.0
                    stats['error_rate'] = 0.0

            return endpoint_stats

        except Exception as e:
            logger.error(f"Error getting endpoint statistics: {e}")
            return {}