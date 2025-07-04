#!/usr/bin/env python3
"""
Test script to debug the exact import path issue in FastAPI
"""

import sys
import os
import asyncio
from typing import Dict, Any, List, Optional

# Add project root to path exactly like main app
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

# Test 1: Direct import like our working tests
print("\n=== Test 1: Direct import (working) ===")
try:
    from core.services.memory_service import MemoryService as DirectMemoryService
    print(f"Direct import successful: {DirectMemoryService}")
    print(f"Direct module: {DirectMemoryService.__module__}")
    print(f"Direct has store_memory: {hasattr(DirectMemoryService, 'store_memory')}")
except Exception as e:
    print(f"Direct import failed: {e}")

# Test 2: Import through app.api.routers path like main FastAPI app
print("\n=== Test 2: Import through app.api.routers (FastAPI way) ===")
try:
    # This mimics how the main app imports: from app.api.routers import core_services
    from app.api.routers import core_services
    print(f"Router import successful: {core_services}")
    
    # Check what MemoryService the router module sees
    router_memory_service = core_services.MemoryService
    print(f"Router MemoryService: {router_memory_service}")
    print(f"Router module: {router_memory_service.__module__}")
    print(f"Router has store_memory: {hasattr(router_memory_service, 'store_memory')}")
    
    # Compare the two classes
    print(f"\nClass comparison:")
    print(f"Direct == Router: {DirectMemoryService == router_memory_service}")
    print(f"Direct is Router: {DirectMemoryService is router_memory_service}")
    
except Exception as e:
    print(f"Router import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Create instances and compare
print("\n=== Test 3: Instance comparison ===")
try:
    direct_instance = DirectMemoryService()
    print(f"Direct instance: {direct_instance}")
    print(f"Direct instance class: {direct_instance.__class__}")
    print(f"Direct instance has store_memory: {hasattr(direct_instance, 'store_memory')}")
    
    if 'router_memory_service' in locals():
        router_instance = router_memory_service()
        print(f"Router instance: {router_instance}")
        print(f"Router instance class: {router_instance.__class__}")
        print(f"Router instance has store_memory: {hasattr(router_instance, 'store_memory')}")
        
        print(f"\nInstance class comparison:")
        print(f"Direct class == Router class: {direct_instance.__class__ == router_instance.__class__}")
        print(f"Direct class is Router class: {direct_instance.__class__ is router_instance.__class__}")
        
except Exception as e:
    print(f"Instance creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check module loading
print("\n=== Test 4: Module loading analysis ===")
print("Loaded modules containing 'memory_service':")
for name, module in sys.modules.items():
    if 'memory_service' in name:
        print(f"  {name}: {module}")

# Test 5: Test the actual router endpoint logic
print("\n=== Test 5: Router endpoint simulation ===")
async def test_router_endpoint():
    try:
        # Simulate the global variable pattern from the router
        memory_service = None
        
        if memory_service is None:
            # Use the same import as the router
            if 'router_memory_service' in locals():
                memory_service = router_memory_service()
            else:
                memory_service = DirectMemoryService()
            
            print(f"Created memory_service: {memory_service}")
            print(f"Type: {type(memory_service)}")
            print(f"Has store_memory: {hasattr(memory_service, 'store_memory')}")
            
            if hasattr(memory_service, 'store_memory'):
                print(f"store_memory type: {type(memory_service.store_memory)}")
                print("Starting memory service...")
                await memory_service.start()
                print("Memory service started successfully")
            else:
                print("ERROR: store_memory method not found!")
                available_methods = [attr for attr in dir(memory_service) if not attr.startswith('_')]
                print(f"Available methods: {available_methods}")
                
    except Exception as e:
        print(f"Router endpoint simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_router_endpoint())