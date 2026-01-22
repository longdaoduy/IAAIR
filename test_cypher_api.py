"""
Test script for the Cypher Subgraph API.

This script demonstrates the functionality of the graph query endpoints
and validates that the API can successfully query the Neo4j knowledge graph.
"""

import requests
import json
import time
from typing import Dict, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class CypherAPITester:
    """Test class for Cypher Subgraph API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.graph_url = f"{base_url}/graph"
        
    def test_database_stats(self):
        """Test database statistics endpoint."""
        print("🔍 Testing database statistics...")
        
        try:
            response = requests.get(f"{self.graph_url}/stats")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Database stats retrieved successfully!")
                print(f"📊 Statistics:")
                
                stats = data.get('stats', {})
                for key, value in stats.items():
                    formatted_key = key.replace('_', ' ').title()
                    print(f"   • {formatted_key}: {value:,}")
                
                return True
            else:
                print(f"❌ Database stats failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Database stats error: {e}")
            return False
    
    def test_custom_query(self):
        """Test custom Cypher query execution."""
        print("\n🔍 Testing custom Cypher query...")
        
        # Simple query to get paper count by year
        query_data = {
            "query": """
            MATCH (p:Paper)
            WHERE p.publication_date IS NOT NULL
            WITH toInteger(split(p.publication_date, '-')[0]) as year, count(p) as paper_count
            RETURN year, paper_count
            ORDER BY year DESC
            LIMIT 5
            """,
            "parameters": {},
            "limit": 5
        }
        
        try:
            response = requests.post(
                f"{self.graph_url}/query",
                json=query_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Custom query executed successfully!")
                print(f"📊 Query took {data['query_time_seconds']:.3f} seconds")
                print(f"📋 Found {data['results_count']} results:")
                
                for result in data['results'][:3]:
                    print(f"   • {result.get('year', 'N/A')}: {result.get('paper_count', 0)} papers")
                
                return True
            else:
                print(f"❌ Custom query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Custom query error: {e}")
            return False
    
    def test_papers_by_author(self, author_name: str = "Smith"):
        """Test finding papers by author."""
        print(f"\n🔍 Testing papers by author: '{author_name}'...")
        
        request_data = {
            "author_name": author_name,
            "limit": 5
        }
        
        try:
            response = requests.post(
                f"{self.graph_url}/papers/by-author",
                json=request_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Author papers query successful!")
                print(f"📊 Query took {data['query_time_seconds']:.3f} seconds")
                print(f"📋 Found {data['results_count']} papers:")
                
                for result in data['results'][:3]:
                    paper_title = result.get('paper_title', 'N/A')[:60] + '...'
                    citations = result.get('cited_by_count', 0)
                    print(f"   • {paper_title} ({citations} citations)")
                
                return True
            else:
                print(f"❌ Author papers query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Author papers query error: {e}")
            return False
    
    def test_most_cited_papers(self):
        """Test finding most cited papers."""
        print("\n🔍 Testing most cited papers...")
        
        try:
            response = requests.get(
                f"{self.graph_url}/papers/most-cited",
                params={"limit": 5, "min_citations": 1}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Most cited papers query successful!")
                print(f"📊 Query took {data['query_time_seconds']:.3f} seconds")
                print(f"📋 Found {data['results_count']} papers:")
                
                for result in data['results'][:3]:
                    paper_title = result.get('paper_title', 'N/A')[:60] + '...'
                    citations = result.get('cited_by_count', 0)
                    print(f"   • {paper_title} ({citations} citations)")
                
                return True
            else:
                print(f"❌ Most cited papers query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Most cited papers query error: {e}")
            return False
    
    def test_paper_citations(self):
        """Test finding paper citations (requires existing paper ID)."""
        print("\n🔍 Testing paper citations...")
        
        # First, try to get a paper ID from most cited papers
        try:
            response = requests.get(
                f"{self.graph_url}/papers/most-cited",
                params={"limit": 1, "min_citations": 1}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    paper_id = data['results'][0].get('paper_id')
                    
                    if paper_id:
                        # Now test finding papers that cite this paper
                        request_data = {
                            "paper_id": paper_id,
                            "direction": "citing",
                            "limit": 3
                        }
                        
                        response = requests.post(
                            f"{self.graph_url}/papers/citations",
                            json=request_data,
                            headers={'Content-Type': 'application/json'}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            print("✅ Paper citations query successful!")
                            print(f"📊 Query took {data['query_time_seconds']:.3f} seconds")
                            print(f"📋 Found {data['results_count']} citing papers:")
                            
                            for result in data['results'][:2]:
                                citing_title = result.get('citing_paper_title', 'N/A')[:50] + '...'
                                print(f"   • {citing_title}")
                            
                            return True
                        else:
                            print(f"❌ Paper citations query failed: {response.status_code}")
                            return False
                    else:
                        print("⚠️  No paper ID found for citation test")
                        return False
                else:
                    print("⚠️  No papers found for citation test")
                    return False
            else:
                print("⚠️  Could not get paper for citation test")
                return False
                
        except Exception as e:
            print(f"❌ Paper citations query error: {e}")
            return False
    
    def test_coauthors(self, author_name: str = "Johnson"):
        """Test finding coauthors."""
        print(f"\n🔍 Testing coauthors for: '{author_name}'...")
        
        request_data = {
            "author_name": author_name,
            "limit": 5
        }
        
        try:
            response = requests.post(
                f"{self.graph_url}/authors/coauthors",
                json=request_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Coauthors query successful!")
                print(f"📊 Query took {data['query_time_seconds']:.3f} seconds")
                print(f"📋 Found {data['results_count']} coauthors:")
                
                for result in data['results'][:3]:
                    coauthor_name = result.get('coauthor_name', 'N/A')
                    collaboration_count = result.get('collaboration_count', 0)
                    print(f"   • {coauthor_name} ({collaboration_count} collaborations)")
                
                return True
            else:
                print(f"❌ Coauthors query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Coauthors query error: {e}")
            return False
    
    def test_venue_papers(self, venue_name: str = "Nature"):
        """Test finding papers in venue."""
        print(f"\n🔍 Testing papers in venue: '{venue_name}'...")
        
        request_data = {
            "venue_name": venue_name,
            "limit": 5
        }
        
        try:
            response = requests.post(
                f"{self.graph_url}/venues/papers",
                json=request_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Venue papers query successful!")
                print(f"📊 Query took {data['query_time_seconds']:.3f} seconds")
                print(f"📋 Found {data['results_count']} papers:")
                
                for result in data['results'][:3]:
                    paper_title = result.get('paper_title', 'N/A')[:50] + '...'
                    venue_name = result.get('venue_name', 'N/A')
                    citations = result.get('cited_by_count', 0)
                    print(f"   • {paper_title} in {venue_name} ({citations} citations)")
                
                return True
            else:
                print(f"❌ Venue papers query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Venue papers query error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all API tests."""
        print("🧪 Testing Cypher Subgraph API")
        print("=" * 50)
        
        tests = [
            ("Database Statistics", self.test_database_stats),
            ("Custom Query", self.test_custom_query),
            ("Papers by Author", self.test_papers_by_author),
            ("Most Cited Papers", self.test_most_cited_papers),
            ("Paper Citations", self.test_paper_citations),
            ("Coauthors", self.test_coauthors),
            ("Venue Papers", self.test_venue_papers)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                results[test_name] = success
                time.sleep(0.5)  # Brief pause between tests
            except Exception as e:
                print(f"❌ {test_name} failed with error: {e}")
                results[test_name] = False
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 30)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:.<20} {status}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Cypher Subgraph API is working correctly!")
        else:
            print(f"⚠️  {total - passed} tests failed. Check API server and Neo4j connection.")
        
        return passed == total

def main():
    """Main test function."""
    print("🚀 Starting Cypher Subgraph API Tests")
    print("Make sure the API server is running on http://localhost:8000")
    print("And that Neo4j contains academic paper data")
    print()
    
    tester = CypherAPITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. The Cypher Subgraph API is ready for hybrid fusion")
        print("2. You can now build Graph ↔ Vector routing logic")
        print("3. Try the interactive API docs at http://localhost:8000/docs")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Ensure the API server is running (python CypherSubgraphAPI.py)")
        print("2. Check Neo4j connection and data")
        print("3. Verify graph database contains papers, authors, and citations")

if __name__ == "__main__":
    main()