from models.entities.retrieval.RoutingStrategy import RoutingStrategy
from models.entities.retrieval.QueryType import QueryType
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.engines.QueryClassifier import QueryClassifier

# ===============================================================================
# HYBRID FUSION SYSTEM
# ===============================================================================

class RoutingDecisionEngine:
    """Decide optimal routing strategy based on query and system state."""

    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.performance_history = {}  # Track routing performance

    def decide_routing(self, query: str, request: HybridSearchRequest) -> RoutingStrategy:
        """Decide routing strategy based on query analysis."""
        if request.routing_strategy != RoutingStrategy.ADAPTIVE:
            return request.routing_strategy

        query_type, confidence = self.query_classifier.classify_query(query)

        # Routing decision logic
        if query_type == QueryType.SEMANTIC and confidence > 0.7:
            return RoutingStrategy.VECTOR_FIRST
        elif query_type == QueryType.STRUCTURAL and confidence > 0.7:
            return RoutingStrategy.GRAPH_FIRST
        else:
            return RoutingStrategy.PARALLEL

    def update_performance(self, strategy: RoutingStrategy, query_type: QueryType,
                           latency: float, relevance_score: float):
        """Update performance tracking for adaptive routing."""
        key = f"{strategy}_{query_type}"
        if key not in self.performance_history:
            self.performance_history[key] = {'latencies': [], 'relevance_scores': []}

        self.performance_history[key]['latencies'].append(latency)
        self.performance_history[key]['relevance_scores'].append(relevance_score)

        # Keep only recent history (last 100 queries)
        for metric_list in self.performance_history[key].values():
            if len(metric_list) > 100:
                metric_list.pop(0)