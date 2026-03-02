#!/usr/bin/env python3
"""
Real-time API Monitoring Dashboard
Displays live statistics of IAAIR API endpoint usage
"""
import requests
import time
import os
import sys
from datetime import datetime


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_number(num):
    """Format numbers with appropriate suffixes."""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)


def display_dashboard(api_url="http://localhost:8000"):
    """Display real-time API monitoring dashboard."""
    
    print("🚀 IAAIR API Real-time Monitoring Dashboard")
    print("=" * 60)
    print(f"📡 Monitoring: {api_url}")
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    refresh_count = 0
    
    try:
        while True:
            refresh_count += 1
            
            try:
                # Fetch API statistics
                response = requests.get(f"{api_url}/api/stats", timeout=10)
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    # Clear screen for updates (after first display)
                    if refresh_count > 1:
                        clear_screen()
                        print("🚀 IAAIR API Real-time Monitoring Dashboard")
                        print("=" * 60)
                        print(f"📡 Monitoring: {api_url}")
                        print(f"🔄 Refresh #{refresh_count} - {datetime.now().strftime('%H:%M:%S')}")
                        print("Press Ctrl+C to stop")
                        print("=" * 60)
                        print()
                    
                    # Display monitoring status
                    monitoring = stats.get('monitoring', {})
                    print(f"📊 Status: {monitoring.get('status', 'unknown').upper()}")
                    print(f"🎯 Endpoints Tracked: {monitoring.get('total_endpoints_tracked', 0)}")
                    print()
                    
                    # Display summary statistics
                    summary = stats.get('summary', {})
                    total_requests = summary.get('total_requests', 0)
                    success_rate = summary.get('overall_success_rate', 0)
                    error_rate = summary.get('overall_error_rate', 0)
                    
                    print("📈 SUMMARY STATISTICS")
                    print("-" * 30)
                    print(f"Total Requests:    {format_number(total_requests)}")
                    print(f"Successful:        {format_number(summary.get('total_successes', 0))}")
                    print(f"Errors:            {format_number(summary.get('total_errors', 0))}")
                    print(f"Success Rate:      {success_rate:.1%}")
                    print(f"Error Rate:        {error_rate:.1%}")
                    
                    # Color coding for success rate
                    if success_rate >= 0.95:
                        status_icon = "🟢"
                    elif success_rate >= 0.90:
                        status_icon = "🟡"
                    else:
                        status_icon = "🔴"
                    
                    print(f"Health Status:     {status_icon}")
                    print()
                    
                    # Display top endpoints
                    top_endpoints = stats.get('top_endpoints', {}).get('top_5', [])
                    if top_endpoints:
                        print("🔥 TOP ENDPOINTS (Last 5)")
                        print("-" * 50)
                        print(f"{'Method':<8} {'Endpoint':<20} {'Requests':<10} {'Success Rate'}")
                        print("-" * 50)
                        
                        for endpoint in top_endpoints:
                            method = endpoint['method']
                            path = endpoint['endpoint'][:18] + ".." if len(endpoint['endpoint']) > 18 else endpoint['endpoint']
                            count = format_number(endpoint['total_count'])
                            rate = endpoint['success_rate']
                            rate_icon = "🟢" if rate >= 0.95 else "🟡" if rate >= 0.90 else "🔴"
                            
                            print(f"{method:<8} {path:<20} {count:<10} {rate:.1%} {rate_icon}")
                    
                    print()
                    
                    # Display insights
                    insights = stats.get('insights', {})
                    print("💡 INSIGHTS")
                    print("-" * 20)
                    print(f"Busiest:           {insights.get('busiest_endpoint', 'None')}")
                    print(f"With Errors:       {insights.get('endpoints_with_errors', 0)}")
                    print(f"Perfect (No Err):  {insights.get('perfect_endpoints', 0)}")
                    
                    print()
                    print("🔗 MONITORING LINKS")
                    print("-" * 25)
                    print(f"API Stats:         {api_url}/api/stats")
                    print(f"Prometheus:        {api_url}/metrics")
                    print(f"Grafana:           http://localhost:3000")
                    
                else:
                    print(f"❌ API Error: HTTP {response.status_code}")
                    print(f"Response: {response.text[:200]}")
                
            except requests.exceptions.ConnectionError:
                print(f"❌ Connection failed to {api_url}")
                print("💡 Make sure IAAIR API server is running")
            except requests.exceptions.Timeout:
                print("⏰ Request timeout - API may be slow")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print()
            print(f"Next update in 10 seconds... (Refresh #{refresh_count})")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="IAAIR API Real-time Monitoring Dashboard")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="IAAIR API base URL (default: http://localhost:8000)")
    
    args = parser.parse_args()
    
    display_dashboard(args.url)