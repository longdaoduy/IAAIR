#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from models.engines.RoutingDecisionEngine import RoutingDecisionEngine
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.entities.retrieval.RoutingStrategy import RoutingStrategy

async def test_gemini_routing():
    """Test Gemini-powered routing decisions."""
    
    # Test queries with different characteristics
    test_queries = [
        "who is the author of paper have id = W2036113194",  # Should be GRAPH_FIRST
        "find papers similar to machine learning in healthcare",  # Should be VECTOR_FIRST
        "papers about deep learning written by Geoffrey Hinton and their citations",  # Should be PARALLEL
        "what are the recent trends in artificial intelligence research",  # Should be VECTOR_FIRST
        "list all papers authored by John Smith published in 2023"  # Should be GRAPH_FIRST
    ]
    
    # Initialize routing engine
    routing_engine = RoutingDecisionEngine()
    
    print(f"Gemini integration status: {routing_engine.use_gemini}")
    print(f"Gemini model available: {routing_engine.gemini_model is not None}")
    print("-" * 80)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Create request
        request = HybridSearchRequest(
            query=query,
            top_k=10,
            routing_strategy=RoutingStrategy.ADAPTIVE
        )
        
        # Test routing decision
        strategy = routing_engine.decide_routing(request.query, request)
        print(f"Selected strategy: {strategy.value}")
        
        # Also show fallback classification for comparison
        query_type, confidence = routing_engine.query_classifier.classify_query(query)
        print(f"Fallback classification: {query_type}, confidence: {confidence}")

if __name__ == "__main__":
    asyncio.run(test_gemini_routing())