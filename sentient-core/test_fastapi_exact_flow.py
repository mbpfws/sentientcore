#!/usr/bin/env python3

import sys
import os
import asyncio

# Add project root to path (exactly like FastAPI router)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.services.memory_service import MemoryService, MemoryLayer, MemoryType

async def test_exact_fastapi_flow():
    """Test the exact flow that FastAPI uses"""
    print("=== Testing Exact FastAPI Flow ===")
    
    # Mimic the global variable pattern
    memory_service = None
    
    try:
        # Mimic the FastAPI endpoint logic
        if memory_service is None:
            print("Creating MemoryService instance...")
            memory_service = MemoryService()
            print(f"Instance created: {memory_service}")
            print(f"Instance type: {type(memory_service)}")
            
            # Check if store_memory exists before calling start
            print(f"Has store_memory before start: {hasattr(memory_service, 'store_memory')}")
            if hasattr(memory_service, 'store_memory'):
                print(f"store_memory method: {getattr(memory_service, 'store_memory')}")
            
            print("Calling start()...")
            await memory_service.start()
            print("Start completed")
            
            # Check if store_memory exists after calling start
            print(f"Has store_memory after start: {hasattr(memory_service, 'store_memory')}")
            if hasattr(memory_service, 'store_memory'):
                print(f"store_memory method: {getattr(memory_service, 'store_memory')}")
            else:
                print("ERROR: store_memory method disappeared after start()!")
                # List all available methods
                print("Available methods:")
                for attr in dir(memory_service):
                    if not attr.startswith('_'):
                        print(f"  {attr}: {type(getattr(memory_service, attr))}")
        
        # Test the actual store_memory call
        print("\n=== Testing store_memory call ===")
        layer = MemoryLayer.KNOWLEDGE_SYNTHESIS
        memory_type = MemoryType.RESEARCH_FINDING
        content = "Test memory content for debugging"
        metadata = {"source": "test_script", "topic": "debugging"}
        tags = ["test", "debug"]
        
        print(f"Calling store_memory with:")
        print(f"  layer: {layer}")
        print(f"  memory_type: {memory_type}")
        print(f"  content: {content}")
        print(f"  metadata: {metadata}")
        print(f"  tags: {tags}")
        
        memory_id = await memory_service.store_memory(
            layer=layer,
            memory_type=memory_type,
            content=content,
            metadata=metadata,
            tags=tags
        )
        
        print(f"Success! Memory ID: {memory_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Additional debugging
        print(f"\nMemory service object: {memory_service}")
        print(f"Memory service type: {type(memory_service)}")
        print(f"Memory service dict: {memory_service.__dict__ if memory_service else 'None'}")

if __name__ == "__main__":
    asyncio.run(test_exact_fastapi_flow())