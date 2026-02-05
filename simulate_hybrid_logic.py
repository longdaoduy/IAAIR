#!/usr/bin/env python3

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from models.engines.RoutingDecisionEngine import RoutingDecisionEngine
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.entities.retrieval.RoutingStrategy import RoutingStrategy
import re

def _is_paper_id_query(query: str) -> bool:
    """Check if the query is asking for a specific paper by ID."""
    paper_id_pattern = re.compile(r'\b(W\d+|DOI:|doi:|pmid:|arxiv:)', re.IGNORECASE)
    return bool(paper_id_pattern.search(query))

async def simulate_hybrid_search_logic():
    """Simulate the hybrid search logic step by step."""
    
    # Test query
    query = "who is the author of paper have id = W2036113194"
    print(f"Query: {query}")
    
    # Step 1: Initialize components
    routing_engine = RoutingDecisionEngine()
    
    # Step 2: Create request
    request = HybridSearchRequest(
        query=query,
        top_k=10,
        routing_strategy=RoutingStrategy.ADAPTIVE
    )
    
    # Step 3: Get routing strategy
    routing_strategy = routing_engine.decide_routing(request.query, request)
    query_type, confidence = routing_engine.query_classifier.classify_query(query)
    print(f"Routing strategy: {routing_strategy}")
    print(f"Query classification: {query_type}, confidence: {confidence}")
    
    # Step 4: Simulate routing logic
    print("\n=== SIMULATING ROUTING LOGIC ===")
    
    if routing_strategy == RoutingStrategy.VECTOR_FIRST:
        print("VECTOR_FIRST strategy")
        # Vector search first, then graph refinement
        vector_results = ["mock_vector_result"]  # Simulate vector results
        graph_results = ["mock_graph_result"]   # Simulate graph results
        
    elif routing_strategy == RoutingStrategy.GRAPH_FIRST:
        print("GRAPH_FIRST strategy")
        # Graph search first, then vector similarity
        graph_results = ["mock_graph_result"]   # Simulate graph results
        
        # Check if this is a specific paper ID query
        if _is_paper_id_query(request.query):
            print("✅ Paper ID query detected - setting vector_results = []")
            vector_results = []
            if graph_results:
                graph_results = graph_results[:1]  # Limit to single exact match
                print(f"   Limited graph_results to: {len(graph_results)} result(s)")
            else:
                print("   WARNING: Graph search returned no results")
        elif graph_results and not _is_paper_id_query(request.query):
            print("Regular query - executing vector refinement")
            vector_results = ["mock_vector_refinement"]
        else:
            print("No graph results - setting vector_results = []")
            vector_results = []
            
    elif routing_strategy == RoutingStrategy.PARALLEL:
        print("PARALLEL strategy")
        # Execute both searches in parallel
        vector_results = ["mock_vector_result"]
        graph_results = ["mock_graph_result"]
    
    # Step 5: Show final results before fusion
    print(f"\n=== BEFORE FUSION ===")
    print(f"vector_results: {len(vector_results or [])} items")
    print(f"graph_results: {len(graph_results or [])} items")
    
    # Step 6: Simulate fusion stats
    fusion_stats = {
        'vector_results_count': len(vector_results or []),
        'graph_results_count': len(graph_results or []),
        'fusion_method': routing_strategy.value,
    }
    print(f"\n=== FUSION STATS ===")
    print(f"Fusion stats: {fusion_stats}")
    
    return vector_results, graph_results, fusion_stats

if __name__ == "__main__":
    import asyncio
    asyncio.run(simulate_hybrid_search_logic())