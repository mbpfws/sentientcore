#!/usr/bin/env python3
"""
Direct test of memory endpoint to debug the store_memory issue
"""

import requests
import json

def test_memory_endpoint():
    """Test the memory storage endpoint"""
    url = "http://localhost:8000/api/core-services/memory/store"
    
    payload = {
        "layer": "knowledge_synthesis",
        "memory_type": "research_finding",
        "content": "Test memory content for debugging",
        "metadata": {
            "source": "test_script",
            "topic": "debugging"
        },
        "tags": ["test", "debug"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Testing endpoint: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_memory_endpoint()