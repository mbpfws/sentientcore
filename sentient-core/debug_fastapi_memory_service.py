#!/usr/bin/env python3
"""
Debug script to inspect MemoryService within FastAPI context
"""

import requests
import json
import sys
import os
import inspect

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the same way as the router
from core.services.memory_service import MemoryService, MemoryLayer, MemoryType

def inspect_memory_service_class():
    """Inspect the MemoryService class in detail"""
    print("=== Inspecting MemoryService Class ===")
    
    print(f"MemoryService class: {MemoryService}")
    print(f"MemoryService module: {MemoryService.__module__}")
    print(f"MemoryService file: {inspect.getfile(MemoryService)}")
    
    # Get all methods and their signatures
    methods = inspect.getmembers(MemoryService, predicate=inspect.isfunction)
    print(f"\nClass methods ({len(methods)}):")
    for name, method in methods:
        if not name.startswith('_'):
            try:
                sig = inspect.signature(method)
                print(f"  {name}{sig}")
            except Exception as e:
                print(f"  {name}: {e}")
    
    # Check specifically for store_memory
    if hasattr(MemoryService, 'store_memory'):
        store_memory = getattr(MemoryService, 'store_memory')
        print(f"\nstore_memory found:")
        print(f"  Type: {type(store_memory)}")
        print(f"  Is coroutine function: {inspect.iscoroutinefunction(store_memory)}")
        try:
            sig = inspect.signature(store_memory)
            print(f"  Signature: {sig}")
        except Exception as e:
            print(f"  Signature error: {e}")
    else:
        print("\nstore_memory NOT found in class!")

def test_memory_service_instance():
    """Test MemoryService instance"""
    print("\n=== Testing MemoryService Instance ===")
    
    try:
        # Create instance the same way as router
        memory_service = MemoryService()
        
        print(f"Instance class: {memory_service.__class__}")
        print(f"Instance type: {type(memory_service)}")
        
        # Check all attributes
        all_attrs = dir(memory_service)
        public_methods = [attr for attr in all_attrs if not attr.startswith('_')]
        print(f"\nPublic methods/attributes ({len(public_methods)}): {public_methods}")
        
        # Check specifically for store_memory
        print(f"\nHas store_memory attribute: {hasattr(memory_service, 'store_memory')}")
        
        if hasattr(memory_service, 'store_memory'):
            store_memory_method = getattr(memory_service, 'store_memory')
            print(f"store_memory type: {type(store_memory_method)}")
            print(f"store_memory callable: {callable(store_memory_method)}")
            print(f"store_memory is coroutine function: {inspect.iscoroutinefunction(store_memory_method)}")
            
            # Try to get signature
            try:
                sig = inspect.signature(store_memory_method)
                print(f"store_memory signature: {sig}")
            except Exception as e:
                print(f"Signature error: {e}")
        else:
            print("store_memory NOT found on instance!")
            
            # Let's see what methods are actually available
            methods = [attr for attr in dir(memory_service) if callable(getattr(memory_service, attr)) and not attr.startswith('_')]
            print(f"Available callable methods: {methods}")
        
        return memory_service
        
    except Exception as e:
        print(f"Error creating MemoryService instance: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_fastapi_endpoint():
    """Test the FastAPI endpoint"""
    print("\n=== Testing FastAPI endpoint ===")
    
    url = "http://localhost:8000/api/core-services/memory/store"
    payload = {
        "layer": "knowledge_synthesis",
        "memory_type": "research_finding",
        "content": "Debug test content",
        "metadata": {
            "source": "debug_script",
            "topic": "debugging"
        },
        "tags": ["debug", "test"]
    }
    
    print(f"Testing endpoint: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print(f"Success Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Starting detailed MemoryService debugging...")
    
    # Inspect the class
    inspect_memory_service_class()
    
    # Test instance
    memory_service = test_memory_service_instance()
    
    # Test FastAPI endpoint
    test_fastapi_endpoint()
    
    print("\nDebugging complete.")