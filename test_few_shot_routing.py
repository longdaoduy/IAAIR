#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from models.engines.RoutingDecisionEngine import RoutingDecisionEngine
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.entities.retrieval.RoutingStrategy import RoutingStrategy

async def test_few_shot_learning():
    """Test few-shot learning capabilities for query routing."""
    
    # Test queries to demonstrate few-shot learning
    test_queries = [
        # Structural queries (should use GRAPH_FIRST)
        "who is the author of paper have id = W2036113194",
        "papers authored by Geoffrey Hinton",
        "find all co-authors of John Smith",
        "papers with DOI:10.1038/nature14539",
        "collaboration network between universities",
        
        # Semantic queries (should use VECTOR_FIRST)  
        "papers about machine learning in healthcare",
        "research related to climate change and sustainability", 
        "find papers similar to deep learning applications",
        "studies concerning artificial intelligence ethics",
        "research regarding quantum computing algorithms",
        
        # Hybrid/Complex queries (should use PARALLEL)
        "recent trends and applications of neural networks in NLP",
        "what are the most cited papers about artificial intelligence",
        "collaborative research between MIT and Stanford on robotics",
        "evolution of machine learning techniques in medical diagnosis",
        
        # Edge cases to test learning
        "papers published in Nature journal by researchers at Google",
        "how many papers cite the transformer architecture paper",
        "interdisciplinary research combining biology and computer science"
    ]
    
    # Initialize routing engine
    routing_engine = RoutingDecisionEngine()
    
    print("=== Few-Shot Learning Routing Decision Test ===")
    print(f"Gemini integration status: {routing_engine.use_gemini}")
    print(f"Few-shot examples loaded: {len(routing_engine.few_shot_examples)}")
    print("-" * 80)
    
    if routing_engine.use_gemini:
        print("🎯 Using Few-Shot Learning with Gemini")
    else:
        print("⚠️  Fallback to Rule-Based Classification (No Gemini API Key)")
    
    print("-" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        # Create request
        request = HybridSearchRequest(
            query=query,
            top_k=10,
            routing_strategy=RoutingStrategy.ADAPTIVE
        )
        
        # Test routing decision
        strategy = routing_engine.decide_routing(request.query, request)
        
        # Get query classification 
        query_type, confidence = routing_engine.get_query_classification(query)
        
        print(f"   📊 Classification: {query_type.value}, confidence: {confidence:.2f}")
        print(f"   🎯 Selected strategy: {strategy.value}")
        
        # Show expected vs actual for validation
        if "id =" in query.lower() or "doi:" in query.lower() or "author" in query.lower():
            expected = "GRAPH_FIRST"
        elif "similar" in query.lower() or "about" in query.lower() or "related" in query.lower():
            expected = "VECTOR_FIRST"
        elif any(word in query.lower() for word in ["trends", "most cited", "evolution", "how many"]):
            expected = "PARALLEL"
        else:
            expected = "VARIABLE"
            
        if expected != "VARIABLE":
            status = "✅" if strategy.value == expected else "⚠️"
            print(f"   {status} Expected: {expected}")

def show_few_shot_examples():
    """Display the few-shot learning examples loaded from file."""
    routing_engine = RoutingDecisionEngine()
    
    print("\n" + "="*80)
    print("Few-Shot Learning Examples (loaded from data/few_shot_examples.json):")
    print("="*80)
    
    if not routing_engine.few_shot_examples:
        print("❌ No examples loaded!")
        return
    
    for i, example in enumerate(routing_engine.few_shot_examples, 1):
        print(f"\n{i:2d}. Query: \"{example['query']}\"")
        print(f"    Type: {example['query_type']:<12} Strategy: {example['routing']:<12} Confidence: {example['confidence']}")
        print(f"    Reasoning: {example['reasoning']}")
    
    print(f"\n📊 Total examples loaded: {len(routing_engine.few_shot_examples)}")
    
    # Show distribution
    from collections import Counter
    query_types = Counter(ex['query_type'] for ex in routing_engine.few_shot_examples)
    routing_strategies = Counter(ex['routing'] for ex in routing_engine.few_shot_examples)
    
    print(f"Query Types: {dict(query_types)}")
    print(f"Routing Strategies: {dict(routing_strategies)}")

async def compare_approaches():
    """Compare few-shot learning vs rule-based classification."""
    routing_engine = RoutingDecisionEngine()
    
    comparison_queries = [
        "who authored the paper about attention mechanisms",
        "papers similar to BERT model applications", 
        "collaboration patterns in machine learning research",
        "recent developments in computer vision"
    ]
    
    print("\n" + "="*80)
    print("Comparison: Few-Shot Learning vs Rule-Based")
    print("="*80)
    
    for query in comparison_queries:
        print(f"\nQuery: {query}")
        
        # Few-shot learning result
        if routing_engine.use_gemini:
            request = HybridSearchRequest(query=query, routing_strategy=RoutingStrategy.ADAPTIVE)
            few_shot_result = routing_engine._few_shot_route_decision(query, request)
            if few_shot_result:
                strategy, query_type, confidence = few_shot_result
                print(f"🎯 Few-Shot: {strategy.value} ({query_type.value}, {confidence:.2f})")
            else:
                print("🎯 Few-Shot: Failed")
        else:
            print("🎯 Few-Shot: Not available (no API key)")
        
        # Rule-based result  
        rule_query_type, rule_confidence = routing_engine.query_classifier.classify_query(query)
        if rule_query_type.value == "SEMANTIC" and rule_confidence > 0.7:
            rule_strategy = "VECTOR_FIRST"
        elif rule_query_type.value == "STRUCTURAL" and rule_confidence > 0.7:
            rule_strategy = "GRAPH_FIRST" 
        else:
            rule_strategy = "PARALLEL"
            
        print(f"📏 Rule-Based: {rule_strategy} ({rule_query_type.value}, {rule_confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(test_few_shot_learning())
    show_few_shot_examples()
    asyncio.run(compare_approaches())