#!/usr/bin/env python3
"""
Test script for Semantic Scholar API with the correct endpoint format.
"""

import requests
import json
from urllib.parse import quote

def test_semantic_scholar_api():
    """Test the Semantic Scholar API with the format you provided."""
    
    # API configuration
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    api_key = "PkEo5fAuu37zVik3hcGuG8se7wlMgD1D2UfWTm2V"
    
    # Test query
    query = "Radiation Resistant Camera System for Monitoring Deuterium Plasma Discharges in the Large Helical Device"
    
    # Parameters
    params = {
        "query": query,
        "limit": 1,
        "fields": "title,abstract,year,citationCount"
    }
    
    # Headers
    headers = {
        "x-api-key": api_key,
        "Accept": "application/json",
        "User-Agent": "KnowledgeFabric/1.0"
    }
    
    print("Testing Semantic Scholar API...")
    print(f"URL: {base_url}")
    print(f"Query: {query}")
    print(f"Headers: {headers}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse Data:")
            print(json.dumps(data, indent=2))
            
            if "data" in data and len(data["data"]) > 0:
                paper = data["data"][0]
                print(f"\n=== First Paper ===")
                print(f"Title: {paper.get('title', 'N/A')}")
                print(f"Year: {paper.get('year', 'N/A')}")
                print(f"Citation Count: {paper.get('citationCount', 'N/A')}")
                print(f"Abstract: {paper.get('abstract', 'N/A')[:200]}...")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_semantic_scholar_api()