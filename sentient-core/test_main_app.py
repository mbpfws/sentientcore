#!/usr/bin/env python3
"""
Test the actual main FastAPI app to reproduce the exact error
"""

import sys
import os
import asyncio
import requests
import json
from typing import Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Testing main FastAPI app...")

# Test the actual endpoint
def test_memory_store_endpoint():
    """Test the /api/core-services/memory/store endpoint"""
    url = "http://localhost:8000/api/core-services/memory/store"
    
    payload = {
        "content": "Test memory content",
        "layer": "knowledge_synthesis",
        "memory_type": "research_finding",
        "metadata": {"test": "true"}
    }
    
    try:
        print(f"Making request to: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"Success! Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
            
        return response
        
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        print("Make sure the FastAPI server is running on localhost:8000")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def test_memory_stats_endpoint():
    """Test the /api/core-services/memory/stats endpoint"""
    url = "http://localhost:8000/api/core-services/memory/stats"
    
    try:
        print(f"\nTesting stats endpoint: {url}")
        response = requests.get(url, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"Success! Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
            
        return response
        
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

if __name__ == "__main__":
    print("=== Testing Main FastAPI App Endpoints ===")
    
    # Test stats first (simpler endpoint)
    stats_response = test_memory_stats_endpoint()
    
    # Test store endpoint
    store_response = test_memory_store_endpoint()
    
    if store_response is None and stats_response is None:
        print("\nNo server running. Starting a test to check if we can import the main app...")
        
        try:
            print("\nTrying to import main app...")
            from app.api.app import app
            print(f"Main app imported successfully: {app}")
            
            # Check if we can access the router
            print("\nChecking routers...")
            for route in app.routes:
                if hasattr(route, 'path') and 'memory' in route.path:
                    print(f"Found memory route: {route.path} - {route.methods if hasattr(route, 'methods') else 'N/A'}")
                    
        except Exception as e:
            print(f"Failed to import main app: {e}")
            import traceback
            traceback.print_exc()