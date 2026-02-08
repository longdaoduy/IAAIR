#!/usr/bin/env python3
"""
Few-Shot Examples Management Utility

This script helps manage the few-shot learning examples for the routing decision engine.
It provides functionality to:
- View existing examples
- Add new examples
- Validate example format
- Update example metadata
"""

import json
import os
import sys
from datetime import datetime
from collections import Counter

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def get_examples_path():
    """Get the path to the few-shot examples file."""
    return os.path.join(os.path.dirname(__file__), 'data', 'few_shot_examples.json')

def load_examples():
    """Load examples from JSON file."""
    try:
        with open(get_examples_path(), 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ Examples file not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None

def save_examples(data):
    """Save examples to JSON file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(get_examples_path()), exist_ok=True)
        
        with open(get_examples_path(), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("✅ Examples saved successfully!")
        return True
    except Exception as e:
        print(f"❌ Error saving examples: {e}")
        return False

def view_examples():
    """Display all examples with statistics."""
    data = load_examples()
    if not data:
        return
    
    examples = data.get('few_shot_examples', [])
    metadata = data.get('metadata', {})
    
    print("=" * 80)
    print(f"📚 Few-Shot Learning Examples ({len(examples)} total)")
    print("=" * 80)
    
    # Show metadata
    if metadata:
        print(f"Version: {metadata.get('version', 'N/A')}")
        print(f"Created: {metadata.get('created', 'N/A')}")
        print(f"Description: {metadata.get('description', 'N/A')}")
        print()
    
    # Show examples
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. Query: \"{example['query']}\"")
        print(f"    Type: {example['query_type']:<12} Strategy: {example['routing']:<12} Confidence: {example['confidence']}")
        print(f"    Reasoning: {example['reasoning']}")
        print()
    
    # Show statistics
    query_types = Counter(ex['query_type'] for ex in examples)
    routing_strategies = Counter(ex['routing'] for ex in examples)
    
    print("📊 Statistics:")
    print(f"Query Types: {dict(query_types)}")
    print(f"Routing Strategies: {dict(routing_strategies)}")

def add_example():
    """Interactive function to add a new example."""
    print("➕ Adding New Few-Shot Example")
    print("-" * 40)
    
    # Get input
    query = input("Query: ").strip()
    if not query:
        print("❌ Query cannot be empty!")
        return
    
    print("\\nQuery Types: STRUCTURAL, SEMANTIC, FACTUAL, HYBRID")
    query_type = input("Query Type: ").strip().upper()
    if query_type not in ['STRUCTURAL', 'SEMANTIC', 'FACTUAL', 'HYBRID']:
        print("❌ Invalid query type!")
        return
    
    print("\\nRouting Strategies: VECTOR_FIRST, GRAPH_FIRST, PARALLEL")
    routing = input("Routing Strategy: ").strip().upper()
    if routing not in ['VECTOR_FIRST', 'GRAPH_FIRST', 'PARALLEL']:
        print("❌ Invalid routing strategy!")
        return
    
    try:
        confidence = float(input("Confidence (0.0-1.0): ").strip())
        if not 0.0 <= confidence <= 1.0:
            print("❌ Confidence must be between 0.0 and 1.0!")
            return
    except ValueError:
        print("❌ Invalid confidence value!")
        return
    
    reasoning = input("Reasoning: ").strip()
    if not reasoning:
        print("❌ Reasoning cannot be empty!")
        return
    
    # Load existing data
    data = load_examples()
    if not data:
        data = {"few_shot_examples": [], "metadata": {}}
    
    # Add new example
    new_example = {
        "query": query,
        "query_type": query_type,
        "routing": routing,
        "confidence": confidence,
        "reasoning": reasoning
    }
    
    data["few_shot_examples"].append(new_example)
    
    # Update metadata
    examples = data["few_shot_examples"]
    query_types = Counter(ex['query_type'] for ex in examples)
    routing_strategies = Counter(ex['routing'] for ex in examples)
    
    data["metadata"] = {
        "version": "1.0",
        "created": data.get("metadata", {}).get("created", datetime.now().strftime("%Y-%m-%d")),
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "description": "Few-shot learning examples for query routing classification",
        "total_examples": len(examples),
        "query_types": dict(query_types),
        "routing_strategies": dict(routing_strategies)
    }
    
    # Save
    if save_examples(data):
        print(f"\\n✅ Added new example! Total examples: {len(examples)}")

def validate_examples():
    """Validate the format and consistency of examples."""
    data = load_examples()
    if not data:
        return
    
    examples = data.get('few_shot_examples', [])
    errors = []
    warnings = []
    
    print("🔍 Validating Few-Shot Examples...")
    print("-" * 40)
    
    valid_query_types = {'STRUCTURAL', 'SEMANTIC', 'FACTUAL', 'HYBRID'}
    valid_routing = {'VECTOR_FIRST', 'GRAPH_FIRST', 'PARALLEL'}
    
    for i, example in enumerate(examples, 1):
        # Check required fields
        required_fields = ['query', 'query_type', 'routing', 'confidence', 'reasoning']
        for field in required_fields:
            if field not in example:
                errors.append(f"Example {i}: Missing field '{field}'")
            elif not example[field]:
                errors.append(f"Example {i}: Empty field '{field}'")
        
        # Validate query_type
        if example.get('query_type') not in valid_query_types:
            errors.append(f"Example {i}: Invalid query_type '{example.get('query_type')}'")
        
        # Validate routing
        if example.get('routing') not in valid_routing:
            errors.append(f"Example {i}: Invalid routing '{example.get('routing')}'")
        
        # Validate confidence
        try:
            conf = float(example.get('confidence', 0))
            if not 0.0 <= conf <= 1.0:
                errors.append(f"Example {i}: Confidence {conf} not in range [0.0, 1.0]")
        except (ValueError, TypeError):
            errors.append(f"Example {i}: Invalid confidence value")
        
        # Check for very similar queries (potential duplicates)
        for j, other in enumerate(examples[i:], i+1):
            if example['query'].lower() == other['query'].lower():
                warnings.append(f"Examples {i} and {j}: Identical queries")
            elif len(set(example['query'].lower().split()) & set(other['query'].lower().split())) >= 3:
                warnings.append(f"Examples {i} and {j}: Very similar queries")
    
    # Report results
    if errors:
        print(f"❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"   {error}")
    else:
        print("✅ No validation errors found!")
    
    if warnings:
        print(f"\\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"   {warning}")
    else:
        print("✅ No warnings found!")
    
    # Statistics
    query_types = Counter(ex.get('query_type') for ex in examples if ex.get('query_type'))
    routing_strategies = Counter(ex.get('routing') for ex in examples if ex.get('routing'))
    
    print(f"\\n📊 Current distribution:")
    print(f"   Query Types: {dict(query_types)}")
    print(f"   Routing Strategies: {dict(routing_strategies)}")
    
    return len(errors) == 0

def main():
    """Main CLI interface."""
    if len(sys.argv) < 2:
        print("Few-Shot Examples Management Utility")
        print("=" * 40)
        print("Usage:")
        print("  python few_shot_manager.py view      - View all examples")
        print("  python few_shot_manager.py add       - Add new example")
        print("  python few_shot_manager.py validate  - Validate examples")
        print("  python few_shot_manager.py stats     - Show statistics only")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'view':
        view_examples()
    elif command == 'add':
        add_example()
    elif command == 'validate':
        validate_examples()
    elif command == 'stats':
        data = load_examples()
        if data:
            examples = data.get('few_shot_examples', [])
            query_types = Counter(ex['query_type'] for ex in examples)
            routing_strategies = Counter(ex['routing'] for ex in examples)
            
            print(f"📊 Few-Shot Examples Statistics")
            print(f"Total examples: {len(examples)}")
            print(f"Query Types: {dict(query_types)}")
            print(f"Routing Strategies: {dict(routing_strategies)}")
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()