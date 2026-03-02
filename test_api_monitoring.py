#!/usr/bin/env python3
"""
Test script to verify API endpoint monitoring functionality
"""
import requests
import time
import json
from datetime import datetime


def test_api_monitoring():
    """Test the API endpoint monitoring system."""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing IAAIR API Endpoint Monitoring")
    print("=" * 50)
    
    # Test endpoints to hit
    test_endpoints = [
        ("GET", "/"),
        ("GET", "/health"),
        ("GET", "/api/stats"),
        ("GET", "/cache/stats"),
        ("GET", "/performance/stats"),
        ("POST", "/search", {"query": "machine learning", "k": 5}),
        ("POST", "/hybrid-search", {"query": "artificial intelligence", "k": 10, "routing_strategy": "ADAPTIVE"}),
        ("GET", "/metrics")
    ]
    
    print(f"📊 Making test requests to {len(test_endpoints)} different endpoints...")
    print()
    
    # Make requests to different endpoints
    for i, (method, endpoint, *data) in enumerate(test_endpoints, 1):
        try:
            print(f"{i}. {method} {endpoint}")
            
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=30)
            elif method == "POST":
                payload = data[0] if data else {}
                response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=30)
            
            print(f"   Status: {response.status_code}")
            
            # Make multiple requests to some endpoints to generate statistics
            if endpoint in ["/", "/health", "/api/stats"]:
                for _ in range(2):  # Make 2 additional requests
                    if method == "GET":
                        requests.get(f"{base_url}{endpoint}", timeout=10)
                    time.sleep(0.1)
                    
        except requests.exceptions.RequestException as e:
            print(f"   Error: {e}")
        except Exception as e:
            print(f"   Unexpected error: {e}")
        
        time.sleep(0.5)  # Small delay between requests
    
    print()
    print("📈 Fetching API Statistics...")
    print("=" * 50)
    
    # Get API statistics
    try:
        response = requests.get(f"{base_url}/api/stats", timeout=30)
        if response.status_code == 200:
            stats = response.json()
            
            print(f"✅ Monitoring Status: {stats.get('monitoring', {}).get('status', 'unknown')}")
            print(f"📊 Total Endpoints Tracked: {stats.get('monitoring', {}).get('total_endpoints_tracked', 0)}")
            print()
            
            # Summary statistics
            summary = stats.get('summary', {})
            print("📋 Summary Statistics:")
            print(f"   Total Requests: {summary.get('total_requests', 0)}")
            print(f"   Total Successes: {summary.get('total_successes', 0)}")
            print(f"   Total Errors: {summary.get('total_errors', 0)}")
            print(f"   Success Rate: {summary.get('overall_success_rate', 0):.2%}")
            print(f"   Error Rate: {summary.get('overall_error_rate', 0):.2%}")
            print()
            
            # Top endpoints
            top_endpoints = stats.get('top_endpoints', {}).get('top_5', [])
            if top_endpoints:
                print("🔥 Top 5 Most Used Endpoints:")
                for i, endpoint in enumerate(top_endpoints, 1):
                    print(f"   {i}. {endpoint['method']} {endpoint['endpoint']} - {endpoint['total_count']} requests")
                print()
            
            # Insights
            insights = stats.get('insights', {})
            print("💡 Insights:")
            print(f"   Busiest Endpoint: {insights.get('busiest_endpoint', 'None')}")
            print(f"   Endpoints with Errors: {insights.get('endpoints_with_errors', 0)}")
            print(f"   Perfect Endpoints: {insights.get('perfect_endpoints', 0)}")
            print()
            
            # Detailed endpoint stats
            endpoint_details = stats.get('endpoint_details', [])
            if endpoint_details:
                print("📊 Detailed Endpoint Statistics:")
                print("-" * 80)
                print(f"{'Method':<8} {'Endpoint':<25} {'Total':<8} {'Success':<9} {'Error':<8} {'Success Rate':<12}")
                print("-" * 80)
                
                for endpoint in endpoint_details[:10]:  # Show top 10
                    print(f"{endpoint['method']:<8} {endpoint['endpoint']:<25} "
                          f"{endpoint['total_count']:<8} {endpoint['success_count']:<9} "
                          f"{endpoint['error_count']:<8} {endpoint['success_rate']:<12.2%}")
            
        else:
            print(f"❌ Failed to get API statistics: HTTP {response.status_code}")
            print(f"Response: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error getting API statistics: {e}")
        print("💡 Make sure the IAAIR API server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    print()
    print("🔗 Monitoring URLs:")
    print(f"   API Statistics: {base_url}/api/stats")
    print(f"   Prometheus Metrics: {base_url}/metrics")
    print(f"   Grafana Dashboard: http://localhost:3000")
    print()
    print("✅ Test completed!")


if __name__ == "__main__":
    test_api_monitoring()