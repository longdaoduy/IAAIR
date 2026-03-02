"""
Prometheus Metrics Exporter for IAAIR System

This module integrates with Prometheus to export performance metrics
for monitoring with Grafana dashboards and alerting.
"""

import time
import logging
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Info, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST, start_http_server
)
from contextlib import contextmanager
from dataclasses import dataclass
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
    
    def record_request(self, endpoint: str, routing_strategy: str, query_type: str, duration: float, method: str = "POST"):
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


class PrometheusIntegration:
    """Integration layer for Prometheus monitoring."""
    
    def __init__(self, port: int = 8001, enable_server: bool = True):
        """Initialize Prometheus integration."""
        self.metrics = PrometheusMetrics()
        self.port = port
        self.enable_server = enable_server
        self._server_started = False
        
        if enable_server:
            self.start_metrics_server()
    
    def start_metrics_server(self):
        """Start Prometheus metrics server."""
        if not self._server_started:
            try:
                start_http_server(self.port, registry=self.metrics.registry)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on port {self.port}")
                logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    def record_search_request(self, endpoint: str, routing_strategy: str, 
                            query_type: str, duration: float, component_times: Dict[str, float],
                            cache_hits: Dict[str, bool], result_counts: Dict[str, int]):
        """Record a complete search request with all metrics."""
        
        # Record main request metrics
        method = "POST" if endpoint in ["/hybrid-search", "/search"] else "GET"
        self.metrics.record_request(endpoint, routing_strategy, query_type, duration, method)
        
        # Record component latencies
        for component, time_val in component_times.items():
            if time_val > 0:
                self.metrics.record_component_latency(component, time_val)
        
        # Record cache performance
        for cache_type, hit in cache_hits.items():
            if hit:
                self.metrics.record_cache_hit(cache_type)
            else:
                self.metrics.record_cache_miss(cache_type)
        
        # Record result counts
        for search_type, count in result_counts.items():
            if count > 0:
                self.metrics.record_results(search_type, count)
    
    def update_cache_metrics(self, cache_stats: Dict[str, Dict[str, Any]]):
        """Update cache size metrics."""
        for cache_type, stats in cache_stats.items():
            if 'cache_size' in stats:
                self.metrics.update_cache_size(cache_type, stats['cache_size'])


# Global Prometheus integration instance
prometheus_integration: Optional[PrometheusIntegration] = None


def initialize_prometheus(port: int = 8001, enable_server: bool = True) -> PrometheusIntegration:
    """Initialize global Prometheus integration."""
    global prometheus_integration
    
    if prometheus_integration is None:
        prometheus_integration = PrometheusIntegration(port=port, enable_server=enable_server)
        logger.info("Prometheus monitoring initialized")
    
    return prometheus_integration


def get_prometheus_integration() -> Optional[PrometheusIntegration]:
    """Get the global Prometheus integration instance."""
    return prometheus_integration