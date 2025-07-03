#!/usr/bin/env python3
"""
Test script to exactly mimic the FastAPI endpoint flow
"""

import os
import sys
import asyncio

# Mimic the exact same path setup as in the FastAPI router
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "app", "api", "routers", "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.services.memory_service import MemoryService, MemoryLayer, MemoryType

# Mimic the global variable exactly as in FastAPI
memory_service = None

async def test_store_memory_flow():
    """Test the exact flow used in the FastAPI endpoint"""
    global memory_service
    
    print("Testing exact FastAPI flow...")
    
    try:
        # Mimic the exact initialization logic from FastAPI
        if memory_service is None:
            print("Initializing MemoryService...")
            memory_service = MemoryService()
            print(f"MemoryService instance: {memory_service}")
            print(f"MemoryService class: {memory_service.__class__}")
            print(f"MemoryService module: {memory_service.__class__.__module__}")
            
            # Check if store_memory exists before calling start()
            if hasattr(memory_service, 'store_memory'):
                print(f"✅ store_memory method exists BEFORE start()")
            else:
                print(f"❌ store_memory method does NOT exist BEFORE start()")
                print(f"Available methods: {[method for method in dir(memory_service) if not method.startswith('_')]}")
            
            await memory_service.start()
            print("MemoryService started")
            
            # Check if store_memory exists after calling start()
            if hasattr(memory_service, 'store_memory'):
                print(f"✅ store_memory method exists AFTER start()")
            else:
                print(f"❌ store_memory method does NOT exist AFTER start()")
                print(f"Available methods: {[method for method in dir(memory_service) if not method.startswith('_')]}")
        
        # Test the actual method call
        print("\nTesting store_memory call...")
        memory_id = await memory_service.store_memory(
            layer=MemoryLayer.KNOWLEDGE_SYNTHESIS,
            memory_type=MemoryType.RESEARCH_FINDING,
            content="Test memory content",
            metadata={"test": True},
            tags=["test"]
        )
        
        print(f"✅ Successfully stored memory with ID: {memory_id}")
        
    except Exception as e:
        print(f"❌ Error in store_memory flow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_store_memory_flow())