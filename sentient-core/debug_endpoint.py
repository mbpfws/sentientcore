#!/usr/bin/env python3
"""
Debug endpoint to inspect MemoryService within FastAPI context
"""

from fastapi import FastAPI, HTTPException
import inspect
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.services.memory_service import MemoryService, MemoryLayer, MemoryType

app = FastAPI()

@app.get("/debug/memory-service")
async def debug_memory_service():
    """Debug endpoint to inspect MemoryService"""
    try:
        # Inspect the class
        class_info = {
            "class": str(MemoryService),
            "module": MemoryService.__module__,
            "file": inspect.getfile(MemoryService),
            "has_store_memory_class": hasattr(MemoryService, 'store_memory')
        }
        
        # Create instance
        memory_service = MemoryService()
        
        instance_info = {
            "instance_class": str(memory_service.__class__),
            "instance_type": str(type(memory_service)),
            "has_store_memory_instance": hasattr(memory_service, 'store_memory'),
            "available_methods": [attr for attr in dir(memory_service) if not attr.startswith('_')]
        }
        
        if hasattr(memory_service, 'store_memory'):
            store_memory_info = {
                "type": str(type(memory_service.store_memory)),
                "callable": callable(memory_service.store_memory),
                "is_coroutine": inspect.iscoroutinefunction(memory_service.store_memory)
            }
        else:
            store_memory_info = {"error": "store_memory not found"}
        
        return {
            "success": True,
            "class_info": class_info,
            "instance_info": instance_info,
            "store_memory_info": store_memory_info,
            "python_path": sys.path[:5],  # First 5 entries
            "working_directory": os.getcwd()
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/test-store")
async def debug_test_store():
    """Test storing memory within FastAPI context"""
    try:
        memory_service = MemoryService()
        await memory_service.start()
        
        # Try to call store_memory
        memory_id = await memory_service.store_memory(
            layer=MemoryLayer.KNOWLEDGE_SYNTHESIS,
            memory_type=MemoryType.RESEARCH_FINDING,
            content="Debug test from FastAPI",
            metadata={"source": "fastapi_debug"},
            tags=["debug"]
        )
        
        return {
            "success": True,
            "memory_id": memory_id
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)