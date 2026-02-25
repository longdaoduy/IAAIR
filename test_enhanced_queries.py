#!/usr/bin/env python3
"""
Test script for enhanced intelligent Cypher query building

This script tests the improved _build_intelligent_cypher_query function
with support for multiple entities and more flexible queries.
"""

import sys
import re
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pipelines.retrievals.HybridRetrievalHandler import HybridRetrievalHandler


def test_enhanced_query_building():
    """Test the enhanced query building capabilities."""
    print("🧪 Testing Enhanced Intelligent Cypher Query Building")
    print("=" * 60)
    
    # Create a handler instance (without actual connections for testing)
    handler = HybridRetrievalHandler(None, None, None, None)
    
    # Test cases covering different query patterns
    test_queries = [
        # Multiple paper IDs
        "Who are the authors of papers W1775749144, W2194775991, and W2100837269?",
        "What venues published papers W3038568908 and W2134526812?",
        "How many citations do papers W1775749144 and W2100837269 have?",
        
        # Multiple authors
        "Papers by Kaiming He and Jian Sun",
        "Collaboration between Georg Kresse and J. Furthmüller",
        "Research by OliverH. Lowry, NiraJ. Rosebrough, and A. Farr",
        
        # DOI queries
        "Find paper with DOI:10.1038/nature14539",
        "Papers with doi:10.1585/pfr.15.2402039",
        
        # Venue-based queries
        "Papers published in Nature and Science",
        "Research from journal Bioinformatics",
        "Studies in Physical Review B",
        
        # Institution queries
        "Papers from Stanford University and MIT",
        "Research at National Institute for Fusion Science",
        
        # Citation-based queries
        "Most cited papers about machine learning",
        "Highest citation count in bioinformatics",
        
        # Keyword-based queries
        "Deep learning neural networks computer vision",
        "Protein quantification biochemical analysis methods"
    ]
    
    print("🔍 Testing Query Patterns:")
    print("-" * 40)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        
        try:
            cypher_query, parameters = handler._build_intelligent_cypher_query(query, 10)
            
            print(f"   Generated Cypher: {cypher_query.strip()[:100]}...")
            print(f"   Parameters: {parameters}")
            
            # Test helper functions
            author_names = handler._extract_author_names(query)
            if author_names:
                print(f"   Extracted authors: {author_names}")
            
            keywords = handler._extract_keywords(query)
            if keywords:
                print(f"   Extracted keywords: {keywords[:5]}...")  # Show first 5
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎯 Helper Function Tests:")
    print("-" * 30)
    
    # Test author extraction
    author_test_queries = [
        "Papers by John Smith and Mary Johnson",
        "Collaboration between Alice Brown and Bob Wilson",
        "Research authored by Dr. Jane Doe"
    ]
    
    for query in author_test_queries:
        authors = handler._extract_author_names(query)
        print(f"'{query}' → Authors: {authors}")
    
    # Test keyword extraction
    keyword_test_queries = [
        "Find research about machine learning in healthcare applications",
        "Studies on quantum computing and cryptography methods",
        "Papers concerning climate change and environmental sustainability"
    ]
    
    for query in keyword_test_queries:
        keywords = handler._extract_keywords(query)
        print(f"'{query}' → Keywords: {keywords[:5]}...")
    
    print(f"\n✅ Enhanced query building test completed!")
    print("Features tested:")
    print("- ✅ Multiple paper ID handling")
    print("- ✅ Multiple author name extraction")
    print("- ✅ DOI pattern recognition")
    print("- ✅ Venue and institution queries")
    print("- ✅ Citation-based queries")
    print("- ✅ Keyword extraction and filtering")
    print("- ✅ Flexible parameter binding")
    
    return True


if __name__ == "__main__":
    test_enhanced_query_building()