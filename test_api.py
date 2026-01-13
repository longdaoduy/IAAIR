"""
Test client for IAAIR Paper Ingestion API.

This script demonstrates how to use the FastAPI endpoints.
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        
        health_data = response.json()
        print(f"✅ Health check successful:")
        print(f"   Status: {health_data['status']}")
        print(f"   Services: {health_data['services']}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def pull_papers(num_papers: int = 5, include_neo4j: bool = True, include_zilliz: bool = True) -> Dict[str, Any]:
    """Test the pull papers endpoint."""
    print(f"🚀 Testing paper ingestion endpoint with {num_papers} papers...")
    print(f"   Neo4j upload: {include_neo4j}")
    print(f"   Zilliz upload: {include_zilliz}")
    
    payload = {
        "num_papers": num_papers,
        "include_neo4j": include_neo4j,
        "include_zilliz": include_zilliz,
        "filters": {
            "from_publication_date": "2023-01-01",
            "to_publication_date": "2024-12-31"
        }
    }
    
    try:
        print("📡 Sending request to API...")
        response = requests.post(f"{API_BASE_URL}/pull-papers", json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("✅ Paper ingestion completed successfully!")
        print(f"   Papers processed: {result['papers_processed']}")
        print(f"   Neo4j uploaded: {result['neo4j_uploaded']}")
        print(f"   Zilliz uploaded: {result['zilliz_uploaded']}")
        print(f"   JSON filename: {result['json_filename']}")
        print(f"   Processing time: {result['summary']['processing_time_seconds']:.2f} seconds")
        
        # Print detailed summary
        summary = result['summary']
        print("\\n📊 Processing Summary:")
        print(f"   Papers fetched: {summary['papers_fetched']}")
        print(f"   Authors extracted: {summary['authors_extracted']}")
        print(f"   Citations extracted: {summary['citations_extracted']}")
        print(f"   Avg citations per paper: {summary['avg_citations_per_paper']:.1f}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Paper ingestion failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error detail: {error_detail}")
            except:
                print(f"   Response text: {e.response.text}")
        return {}

def download_file(filename: str):
    """Test downloading a generated file."""
    print(f"📥 Testing file download for: {filename}")
    
    try:
        response = requests.get(f"{API_BASE_URL}/download/{filename}")
        response.raise_for_status()
        
        # Save the file
        local_filename = f"downloaded_{filename}"
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ File downloaded successfully as: {local_filename}")
        
        # Show file size
        import os
        file_size = os.path.getsize(local_filename)
        print(f"   File size: {file_size:,} bytes")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ File download failed: {e}")
        return False

def test_semantic_search(query: str = "machine learning in healthcare", top_k: int = 5, include_details: bool = True):
    """Test the semantic search endpoint."""
    print(f"🔍 Testing semantic search endpoint...")
    print(f"   Query: '{query}'")
    print(f"   Top K: {top_k}")
    print(f"   Include details: {include_details}")
    
    payload = {
        "query": query,
        "top_k": top_k,
        "include_details": include_details
    }
    
    try:
        print("📡 Sending search request to API...")
        response = requests.post(f"{API_BASE_URL}/search", json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        print("✅ Semantic search completed successfully!")
        print(f"   Results found: {result['results_found']}")
        print(f"   Search time: {result['search_time_seconds']:.2f} seconds")
        
        # Print top results
        if result['results']:
            print("\\n🔬 Top Results:")
            for i, paper in enumerate(result['results'][:3], 1):  # Show top 3
                print(f"\\n   {i}. Score: {paper.get('similarity_score', 'N/A'):.3f}")
                print(f"      Title: {paper.get('title', 'No title')[:100]}{'...' if len(paper.get('title', '')) > 100 else ''}")
                print(f"      DOI: {paper.get('doi', 'N/A')}")
                print(f"      Authors: {len(paper.get('authors', []))} author(s)")
                print(f"      Cited by: {paper.get('cited_by_count', 0)} papers")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Semantic search failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"   Error detail: {error_detail}")
            except:
                print(f"   Response text: {e.response.text}")
        return {}

def main():
    """Run the test client."""
    print("🧪 IAAIR Paper Ingestion API Test Client")
    print("=" * 50)
    
    # Test health check first
    if not test_health_check():
        print("❌ API is not healthy. Please check if the server is running.")
        return
    
    print("\\n" + "=" * 50)
    
    # Test paper ingestion
    result = pull_papers(
        num_papers=3,  # Small number for testing
        include_neo4j=True,
        include_zilliz=True
    )
    
    if result and result.get('json_filename'):
        print("\\n" + "=" * 50)
        
        # Test file download
        download_file(result['json_filename'])
    
    print("\\n" + "=" * 50)
    
    # Test semantic search (only if we have data in the system)
    test_semantic_search(
        query="machine learning algorithms for medical diagnosis",
        top_k=5,
        include_details=True
    )
    
    print("\\n🎉 Test completed!")

if __name__ == "__main__":
    main()