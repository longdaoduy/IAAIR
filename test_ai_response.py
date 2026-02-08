#!/usr/bin/env python3

import asyncio
import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

async def test_ai_response_generation():
    """Test the AI response generation functionality."""
    
    # Test queries that would benefit from AI responses
    test_cases = [
        {
            "query": "who is the author of paper have id = W2036113194",
            "expected_response_type": "Factual answer with specific author names"
        },
        {
            "query": "recent trends in machine learning for healthcare",
            "expected_response_type": "Comprehensive analysis of research trends"
        },
        {
            "query": "papers about deep learning written by Geoffrey Hinton",
            "expected_response_type": "Structured list with paper details"
        }
    ]
    
    print("=== AI Response Generation Test ===")
    print("This test demonstrates the new Gemini-powered response generation.")
    print("Note: Requires GEMINI_API_KEY to be set for full functionality.\\n")
    
    # Show example request structure
    print("Example Hybrid Search Request with AI Response:")
    example_request = {
        "query": "who is the author of paper have id = W2036113194",
        "top_k": 10,
        "routing_strategy": "adaptive",
        "enable_reranking": True,
        "enable_attribution": True,
        "enable_ai_response": True,  # New parameter
        "fusion_weights": None,
        "include_provenance": False
    }
    print(json.dumps(example_request, indent=2))
    
    print("\\n" + "="*60)
    print("Expected Response Structure:")
    
    example_response = {
        "success": True,
        "message": "Hybrid search completed using graph_first strategy",
        "query": "who is the author of paper have id = W2036113194",
        "query_type": "structural",
        "routing_used": "graph_first",
        "results_found": 1,
        "search_time_seconds": 0.45,
        "fusion_time_seconds": 0.02,
        "reranking_time_seconds": None,
        "response_generation_time_seconds": 1.23,  # New timing field
        "results": [
            {
                "paper_id": "W2036113194",
                "title": "Special points for Brillouin-zone integrations",
                "authors": ["Hendrik J. Monkhorst", "J.D. Pack"],
                "relevance_score": 0.95
            }
        ],
        "ai_response": "Based on the search results, the paper with ID W2036113194 titled 'Special points for Brillouin-zone integrations' was authored by Hendrik J. Monkhorst and J.D. Pack. This influential paper was published in Physical Review B and has been widely cited in computational physics research.",  # New AI-generated response
        "fusion_stats": {
            "vector_results_count": 0,
            "graph_results_count": 1,
            "fusion_method": "graph_first"
        },
        "attribution_stats": {
            "total_attributions": 2,
            "high_confidence_attributions": 2,
            "attribution_enabled": True
        }
    }
    
    print(json.dumps(example_response, indent=2))
    
    print("\\n" + "="*60)
    print("Key Features of AI Response Generation:")
    print("• Query-aware prompts: Different prompt styles for structural vs semantic queries")
    print("• Context integration: Uses top search results as context for response")
    print("• Academic focus: Tailored for research paper analysis and citation")
    print("• Fallback support: Gracefully handles cases where Gemini is unavailable")
    print("• Performance tracking: Measures response generation time separately")
    
    print("\\n" + "="*60)
    print("API Usage Examples:")
    print()
    print("# Curl example with AI response enabled")
    print('curl -X POST "http://localhost:8000/hybrid-search" \\\\')
    print('  -H "Content-Type: application/json" \\\\')
    print('  -d \'{')
    print('    "query": "who is the author of paper have id = W2036113194",')
    print('    "top_k": 10,')
    print('    "enable_ai_response": true')
    print('  }\'')
    
    print("\\n# Python requests example")
    print('import requests')
    print()
    print('response = requests.post("http://localhost:8000/hybrid-search", json={')
    print('    "query": "recent trends in machine learning for healthcare",')
    print('    "top_k": 10,')
    print('    "enable_ai_response": True,')
    print('    "enable_attribution": True')
    print('})')
    print()
    print('result = response.json()')
    print('print("AI Response:", result["ai_response"])')

if __name__ == "__main__":
    asyncio.run(test_ai_response_generation())