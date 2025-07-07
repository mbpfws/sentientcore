#!/usr/bin/env python3
"""
Test script for ultra minimal server on port 8004
"""

import requests
import time
import sys

def test_endpoint(url, endpoint_name):
    """Test a single endpoint"""
    try:
        print(f"Testing {endpoint_name}: {url}")
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ {endpoint_name} - SUCCESS (Status: {response.status_code})")
            try:
                data = response.json()
                print(f"   Response: {data}")
            except:
                print(f"   Response: {response.text[:100]}...")
            return True
        else:
            print(f"❌ {endpoint_name} - FAILED (Status: {response.status_code})")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ {endpoint_name} - CONNECTION ERROR: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ {endpoint_name} - TIMEOUT: {e}")
        return False
    except Exception as e:
        print(f"❌ {endpoint_name} - ERROR: {e}")
        return False

def main():
    """Main test function"""
    print("=== Ultra Minimal Server Test ===")
    print("Waiting 2 seconds for server to start...")
    time.sleep(2)
    
    base_url = "http://127.0.0.1:8004"
    
    # Test endpoints
    endpoints = [
        ("/", "Root"),
        ("/health", "Health"),
        ("/api/chat/message/json", "Chat"),
        ("/test", "Test")
    ]
    
    results = []
    
    for endpoint, name in endpoints:
        url = f"{base_url}{endpoint}"
        success = test_endpoint(url, name)
        results.append((name, success))
        print()  # Empty line for readability
    
    # Summary
    print("=== Test Summary ===")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Server is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Server might not be running or accessible.")
        sys.exit(1)

if __name__ == "__main__":
    main()