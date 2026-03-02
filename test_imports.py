#!/usr/bin/env python3
"""
Quick import test to verify all components load correctly
"""
import sys
import traceback

def test_imports():
    """Test that all main components can be imported."""
    print("🧪 Testing IAAIR Component Imports...")
    print("=" * 40)
    
    tests = [
        ("ServiceFactory", "from models.engines.ServiceFactory import ServiceFactory"),
        ("PerformanceMonitor", "from models.engines.PerformanceMonitor import PerformanceMonitor"),
        ("PrometheusMonitor", "from models.engines.PrometheusMonitor import PrometheusMetrics, initialize_prometheus"),
        ("FastAPI imports", "from fastapi import FastAPI; from starlette.middleware.base import BaseHTTPMiddleware"),
        ("Main app", "import main")
    ]
    
    results = []
    
    for test_name, import_statement in tests:
        try:
            print(f"Testing {test_name}...", end=" ")
            exec(import_statement)
            print("✅ OK")
            results.append((test_name, True, None))
        except Exception as e:
            print(f"❌ FAILED: {e}")
            results.append((test_name, False, str(e)))
    
    print()
    print("📊 Test Results:")
    print("-" * 40)
    
    passed = 0
    failed = 0
    
    for test_name, success, error in results:
        if success:
            print(f"✅ {test_name}")
            passed += 1
        else:
            print(f"❌ {test_name}: {error}")
            failed += 1
    
    print()
    print(f"📈 Summary: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\n🔧 Issues found! Check the errors above.")
        return False
    else:
        print("\n🎉 All imports successful! Server should start correctly.")
        return True

if __name__ == "__main__":
    try:
        success = test_imports()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)