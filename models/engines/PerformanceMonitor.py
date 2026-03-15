"""
Performance Monitor for IAAIR System

Provides performance tracking with Prometheus integration.
Tracks only: search latency, cache hits, result counts, AI calls, template usage.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track performance metrics with Prometheus integration."""

    def __init__(self, slow_query_threshold: float = 5.0):
        self.slow_query_threshold = slow_query_threshold
        self._search_start: Optional[float] = None

        # Initialize Prometheus integration
        self.prometheus_integration = None
        try:
            from clients.prometheus.PrometheusClient import get_prometheus_integration, initialize_prometheus
            self.prometheus_integration = get_prometheus_integration()
            if self.prometheus_integration is None:
                self.prometheus_integration = initialize_prometheus()
            logger.info("Prometheus monitoring enabled for PerformanceMonitor")
        except Exception as e:
            logger.warning(f"Prometheus integration unavailable: {e}")

    # -- Search tracking --

    def start_search_tracking(self):
        """Mark the start of a hybrid search request."""
        self._search_start = time.time()

    def finish_search_tracking(self, result_count: int = 0, template_key: str = None):
        """Mark the end of a hybrid search request and push metrics.

        Args:
            result_count: Number of results returned to the user.
            template_key: The Neo4j template that was selected.
        """
        if self._search_start is None:
            return

        duration = time.time() - self._search_start
        self._search_start = None

        if duration > self.slow_query_threshold:
            logger.warning(f"Slow search detected: {duration:.2f}s")

        if self.prometheus_integration:
            self.prometheus_integration.record_search(duration)
            if result_count > 0:
                self.prometheus_integration.record_results(result_count)
            if template_key:
                self.prometheus_integration.record_template_used(template_key)

    @contextmanager
    def track_operation(self, operation_name: str):
        """Context manager — currently only used to time graph_search
        (the duration is folded into the overall search duration recorded
        by start/finish_search_tracking)."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            logger.debug(f"{operation_name} took {elapsed:.3f}s")

    # -- Cache tracking --

    def record_cache_hit(self, cache_type: str, hit: bool):
        """Record a cache hit or miss."""
        if self.prometheus_integration:
            if hit:
                self.prometheus_integration.record_cache_hit(cache_type)
            else:
                self.prometheus_integration.record_cache_miss(cache_type)

    def record_result_count(self, source: str, count: int):
        """Record number of results from a source (for logging only)."""
        logger.debug(f"Results from {source}: {count}")

    def update_cache_metrics(self, cache_stats: Dict[str, Dict[str, Any]]):
        """Push current cache sizes to Prometheus."""
        if self.prometheus_integration:
            self.prometheus_integration.update_cache_metrics(cache_stats)

    # -- AI call tracking (called from LLM_Client) --

    def record_ai_call(self, purpose: str, duration: float, tokens: int = 0):
        """Record an AI agent call with purpose, duration, and token count."""
        if self.prometheus_integration:
            self.prometheus_integration.record_ai_call(purpose, duration, tokens)
