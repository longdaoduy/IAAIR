"""
Prometheus Metrics Exporter for IAAIR System

This module integrates with Prometheus to export performance metrics
for monitoring with Grafana dashboards and alerting.
"""

import logging
from typing import Dict, Any, Optional
from prometheus_client import (start_http_server,
    disable_created_metrics
)
from clients.prometheus.PrometheusMetrics import PrometheusMetrics

logger = logging.getLogger(__name__)

# Disable _created metrics that cause "out of bounds" errors in newer Prometheus
disable_created_metrics()

class PrometheusMonitoring:
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
prometheus_integration: Optional[PrometheusMonitoring] = None


def initialize_prometheus(port: int = 8001, enable_server: bool = True) -> PrometheusMonitoring:
    """Initialize global Prometheus integration."""
    global prometheus_integration
    
    if prometheus_integration is None:
        prometheus_integration = PrometheusMonitoring(port=port, enable_server=enable_server)
        logger.info("Prometheus monitoring initialized")
    
    return prometheus_integration


def get_prometheus_integration() -> Optional[PrometheusMonitoring]:
    """Get the global Prometheus integration instance."""
    return prometheus_integration