#!/usr/bin/env python3
"""
Test the debug endpoint
"""

import requests
import json

def test_debug_endpoint():
    """Test the debug endpoint"""
    try:
        print("Testing debug endpoint...")
        response = requests.get("http://localhost:8001/debug/memory-service")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_store_endpoint():
    """Test the store endpoint"""
    try:
        print("\nTesting store endpoint...")
        response = requests.get("http://localhost:8001/debug/test-store")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response:")
            print(json.dumps(data, indent=2))
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_debug_endpoint()
    test_store_endpoint()