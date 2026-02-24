#!/usr/bin/env python3
"""
Test script for mock data evaluation

This script tests the mock data evaluation API endpoint locally.
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipelines.evaluation.MockDataEvaluator import MockDataEvaluator
from models.engines.ServiceFactory import ServiceFactory


async def test_mock_evaluation():
    """Test mock data evaluation functionality."""
    print("🧪 Testing Mock Data Evaluation")
    print("=" * 50)
    
    try:
        # Test 1: Load mock data
        print("Test 1: Loading mock data...")
        evaluator = MockDataEvaluator(None)  # No service factory needed for loading
        questions = evaluator.load_mock_data()
        
        if not questions:
            print("❌ Failed to load mock data")
            return False
        
        print(f"✅ Loaded {len(questions)} questions")
        
        # Show breakdown
        graph_questions = [q for q in questions if q['type'] == 'graph']
        semantic_questions = [q for q in questions if q['type'] == 'semantic']
        
        print(f"   - Graph questions: {len(graph_questions)}")
        print(f"   - Semantic questions: {len(semantic_questions)}")
        
        # Show categories
        categories = {}
        for q in questions:
            category = q['category']
            categories[category] = categories.get(category, 0) + 1
        
        print("   - Categories:")
        for category, count in categories.items():
            print(f"     * {category}: {count}")
        
        print()
        
        # Test 2: Show sample questions
        print("Test 2: Sample questions")
        print("Graph question sample:")
        if graph_questions:
            sample = graph_questions[0]
            print(f"   ID: {sample['id']}")
            print(f"   Question: {sample['question']}")
            print(f"   Category: {sample['category']}")
            print(f"   Expected papers: {sample['expected_evidence'].get('paper_ids', [])}")
        
        print("\nSemantic question sample:")
        if semantic_questions:
            sample = semantic_questions[0]
            print(f"   ID: {sample['id']}")
            print(f"   Question: {sample['question']}")
            print(f"   Category: {sample['category']}")
            print(f"   Expected papers: {sample['expected_evidence'].get('paper_ids', [])}")
        
        print()
        
        # Test 3: Test metric calculation
        print("Test 3: Testing metric calculation...")
        retrieved = ['W1775749144', 'W2194775991', 'W3038568908']
        expected = ['W1775749144', 'W2100837269']
        
        precision, recall, f1 = evaluator._calculate_metrics(retrieved, expected)
        print(f"   Retrieved: {retrieved}")
        print(f"   Expected: {expected}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1: {f1:.3f}")
        
        print()
        print("✅ Mock data evaluation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_api_locally():
    """Test the API endpoint locally (if server is running)."""
    try:
        import requests
        
        print("🌐 Testing API endpoints...")
        base_url = "http://localhost:8000"
        
        # Test preview endpoint
        print("Testing preview endpoint...")
        response = requests.get(f"{base_url}/evaluation/mock-data/preview")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Preview endpoint works - {data['total_questions']} questions loaded")
        else:
            print(f"❌ Preview endpoint failed: {response.status_code}")
            
    except ImportError:
        print("⚠️  requests library not available, skipping API test")
    except Exception as e:
        print(f"⚠️  API test failed (server might not be running): {e}")


if __name__ == "__main__":
    print("Mock Data Evaluation Test Suite")
    print("=" * 60)
    
    # Run async test
    success = asyncio.run(test_mock_evaluation())
    
    if success:
        print("\n🎉 All tests passed!")
        
        # Try API test
        test_api_locally()
        
        print("\nTo test the full evaluation API:")
        print("1. Start the API server: python main.py")
        print("2. Test preview: curl http://localhost:8000/evaluation/mock-data/preview")
        print("3. Test evaluation: curl -X POST http://localhost:8000/evaluation/mock-data")
        
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)