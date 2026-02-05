from enum import Enum

# Hybrid Fusion Models
class RoutingStrategy(str, Enum):
    """Routing strategies for hybrid fusion."""
    VECTOR_FIRST = "vector_first"  # Vector search -> Graph refinement
    GRAPH_FIRST = "graph_first"    # Graph search -> Vector similarity
    PARALLEL = "parallel"          # Both in parallel with fusion
    ADAPTIVE = "adaptive"          # Auto-select based on query analysis