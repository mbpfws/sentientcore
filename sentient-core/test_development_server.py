#!/usr/bin/env python3
"""
Development Server Test Script
Tests the development API server endpoints

Run this AFTER starting development_api_server.py in a separate terminal
"""

import requests
import json
import time
from typing import Dict, Any

def test_endpoint(url: str, method: str = "GET", data: Dict[Any, Any] = None, name: str = "") -> bool:
    """Test a single endpoint"""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=5)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            print(f"❌ {name}: Unsupported method {method}")
            return False
        
        if response.status_code == 200:
            print(f"✅ {name}: SUCCESS (HTTP {response.status_code})")
            try:
                result = response.json()
                if isinstance(result, dict) and len(result) <= 5:  # Show small responses
                    print(f"   Response: {json.dumps(result, indent=2)[:200]}...")
                else:
                    print(f"   Response: {type(result).__name__} with {len(result) if hasattr(result, '__len__') else 'N/A'} items")
            except:
                print(f"   Response: {response.text[:100]}...")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: CONNECTION FAILED - Server not running or not accessible")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ {name}: TIMEOUT - Server took too long to respond")
        return False
    except Exception as e:
        print(f"❌ {name}: ERROR - {e}")
        return False

def main():
    """Test all development server endpoints"""
    print("=" * 60)
    print("🧪 DEVELOPMENT SERVER TEST SUITE")
    print("=" * 60)
    print()
    
    base_url = "http://127.0.0.1:8007"
    
    print(f"Testing server at: {base_url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test cases
    tests = [
        {
            "url": f"{base_url}/",
            "method": "GET",
            "name": "Root Endpoint"
        },
        {
            "url": f"{base_url}/health",
            "method": "GET",
            "name": "Health Check"
        },
        {
            "url": f"{base_url}/api/chat/message/json",
            "method": "GET",
            "name": "Chat GET"
        },
        {
            "url": f"{base_url}/api/chat/message/json",
            "method": "POST",
            "data": {
                "message": "Hello, this is a test message!",
                "model": "test-model",
                "temperature": 0.7
            },
            "name": "Chat POST"
        },
        {
            "url": f"{base_url}/api/status",
            "method": "GET",
            "name": "API Status"
        },
        {
            "url": f"{base_url}/api/test",
            "method": "GET",
            "name": "Test Endpoint"
        }
    ]
    
    # Run tests
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        print(f"[{i}/{total}] Testing {test['name']}...")
        success = test_endpoint(
            url=test["url"],
            method=test["method"],
            data=test.get("data"),
            name=test["name"]
        )
        if success:
            passed += 1
        print()
    
    # Results
    print("=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total} tests")
    print(f"❌ Failed: {total - passed}/{total} tests")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Development server is working correctly.")
        print("\n🔗 You can now access:")
        print(f"   • Server: {base_url}")
        print(f"   • API Docs: {base_url}/docs")
        print(f"   • ReDoc: {base_url}/redoc")
    elif passed > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {passed} out of {total} tests passed.")
        print("   Some endpoints are working, but there may be issues.")
    else:
        print("\n💥 ALL TESTS FAILED!")
        print("   The development server is not running or not accessible.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure development_api_server.py is running")
        print("   2. Check that it's running on port 8007")
        print("   3. Verify no firewall is blocking the connection")
        print("   4. Try running the server directly in your IDE or terminal")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()