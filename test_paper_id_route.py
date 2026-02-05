#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from models.engines.RoutingDecisionEngine import RoutingDecisionEngine
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.entities.retrieval.RoutingStrategy import RoutingStrategy

async def test_paper_id_routing():
    """Test the routing logic for paper ID queries."""
    
    # Test query
    query = "who is the author of paper have id = W2036113194"
    
    # Initialize routing engine
    routing_engine = RoutingDecisionEngine()
    
    # Create request
    request = HybridSearchRequest(
        query=query,
        top_k=10,
        routing_strategy=RoutingStrategy.ADAPTIVE
    )
    
    # Test routing decision
    routing_strategy = routing_engine.decide_routing(request.query, request)
    print(f"Query: {query}")
    print(f"Routing strategy: {routing_strategy}")
    
    # Test query classification
    query_type, confidence = routing_engine.query_classifier.classify_query(query)
    print(f"Query classification: {query_type}, confidence: {confidence}")
    
    # Test paper ID detection 
    # We need to import the _is_paper_id_query function from main.py
    # But let's recreate it here to avoid import issues
    import re
    def _is_paper_id_query(query: str) -> bool:
        """Check if the query is asking for a specific paper by ID."""
        paper_id_pattern = re.compile(r'\b(W\d+|DOI:|doi:|pmid:|arxiv:)', re.IGNORECASE)
        return bool(paper_id_pattern.search(query))
    
    is_paper_id = _is_paper_id_query(query)
    print(f"Is paper ID query: {is_paper_id}")
    
    # Test what would happen in the routing logic
    if routing_strategy.name == "GRAPH_FIRST" and is_paper_id:
        print("✅ Paper ID query should use GRAPH_FIRST with vector_results = []")
    else:
        print("❌ Something is wrong with the routing logic")

if __name__ == "__main__":
    asyncio.run(test_paper_id_routing())