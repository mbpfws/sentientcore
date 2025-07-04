#!/usr/bin/env python3

import sys
import os
import asyncio
import uvicorn
from fastapi.testclient import TestClient

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the FastAPI app
from app.api.app import app

def test_memory_endpoint_with_testclient():
    """Test memory endpoint using FastAPI TestClient"""
    print("=== Testing Memory Endpoint with TestClient ===")
    
    client = TestClient(app)
    
    # Test payload
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
    
    print(f"Testing endpoint: /api/core-services/memory/store")
    print(f"Payload: {payload}")
    
    try:
        # Make the request
        response = client.post("/api/core-services/memory/store", json=payload)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"Success Response: {response.json()}")
        else:
            print(f"Error Response: {response.json()}")
            
    except Exception as e:
        print(f"Exception during request: {e}")
        import traceback
        traceback.print_exc()

def test_direct_import():
    """Test direct import of MemoryService in the same environment"""
    print("\n=== Testing Direct Import in Same Environment ===")
    
    try:
        from core.services.memory_service import MemoryService
        service = MemoryService()
        print(f"Direct import successful: {service}")
        print(f"Has store_memory: {hasattr(service, 'store_memory')}")
        if hasattr(service, 'store_memory'):
            print(f"store_memory method: {getattr(service, 'store_memory')}")
    except Exception as e:
        print(f"Direct import failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_import()
    test_memory_endpoint_with_testclient()