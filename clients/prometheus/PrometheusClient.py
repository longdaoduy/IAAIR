"""
Prometheus Metrics Exporter for IAAIR System

Slim integration layer — exposes only essential metrics:
  cache size, hybrid search count/latency, AI agent calls/tokens,
  cache hits, template usage, result counts.
"""

import logging
from typing import Dict, Any, Optional

from prometheus_client import start_http_server, disable_created_metrics
from clients.prometheus.PrometheusMetrics import PrometheusMetrics

logger = logging.getLogger(__name__)

# Disable _created metrics that cause "out of bounds" errors in newer Prometheus
disable_created_metrics()


class PrometheusMonitoring:
    """Integration layer for Prometheus monitoring."""

    def __init__(self, port: int = 8001, enable_server: bool = True):
        self.metrics = PrometheusMetrics()
        self.port = port
        self._server_started = False

        if enable_server:
            self.start_metrics_server()

    def start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        if not self._server_started:
            try:
                start_http_server(self.port, registry=self.metrics.registry)
                self._server_started = True
                logger.info(f"Prometheus metrics server started on port {self.port}")
                logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")

    # -- Convenience methods (delegate to PrometheusMetrics) --

    def record_search(self, duration: float):
        """Record a hybrid search request with its duration."""
        self.metrics.record_search(duration)

    def record_ai_call(self, purpose: str, duration: float, tokens: int = 0):
        """Record an AI agent call (purpose, latency, tokens generated)."""
        self.metrics.record_ai_call(purpose, duration, tokens)

    def record_cache_hit(self, cache_type: str):
        self.metrics.record_cache_hit(cache_type)

    def record_cache_miss(self, cache_type: str):
        self.metrics.record_cache_miss(cache_type)

    def update_cache_size(self, cache_type: str, size: int):
        self.metrics.update_cache_size(cache_type, size)

    def record_template_used(self, template_key: str):
        """Record which Neo4j template was selected."""
        self.metrics.record_template_used(template_key)

    def record_search_strategy(self, strategy: str):
        """Record which search strategy was used (graph_only / vector_first)."""
        self.metrics.record_search_strategy(strategy)

    def record_results(self, count: int):
        """Record number of results returned."""
        self.metrics.record_results(count)

    def record_verification_label(self, label: str):
        """Record a SciFact verification label."""
        self.metrics.record_verification_label(label)

    def record_verification_duration(self, duration: float):
        """Record SciFact verification latency."""
        self.metrics.record_verification_duration(duration)

    def update_cache_metrics(self, cache_stats: Dict[str, Dict[str, Any]]):
        """Bulk-update cache size gauges from cache_manager stats."""
        for cache_type, stats in cache_stats.items():
            if 'cache_size' in stats:
                self.metrics.update_cache_size(cache_type, stats['cache_size'])


# -- Global singleton --

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
