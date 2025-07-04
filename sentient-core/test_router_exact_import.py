#!/usr/bin/env python3
"""
Test script that exactly mimics the FastAPI router's import and instantiation
"""

import sys
import os
import asyncio
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import exactly like the router does
from core.services.memory_service import MemoryService, MemoryLayer, MemoryType
from fastapi import HTTPException
from pydantic import BaseModel

class MemoryStoreRequest(BaseModel):
    """Request model for storing memory"""
    layer: str  # knowledge_synthesis, conversation_history, codebase_knowledge, stack_dependencies
    memory_type: str  # conversation, code_snippet, documentation, etc.
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

async def store_memory_endpoint(request: MemoryStoreRequest):
    """Exact copy of the FastAPI endpoint logic"""
    try:
        # Global variable simulation
        memory_service = None
        
        if memory_service is None:
            print("Creating new MemoryService instance...")
            memory_service = MemoryService()
            print(f"MemoryService created: {memory_service}")
            print(f"MemoryService type: {type(memory_service)}")
            print(f"Has store_memory: {hasattr(memory_service, 'store_memory')}")
            
            if hasattr(memory_service, 'store_memory'):
                print(f"store_memory type: {type(memory_service.store_memory)}")
            else:
                print("ERROR: store_memory not found!")
                available_methods = [attr for attr in dir(memory_service) if not attr.startswith('_')]
                print(f"Available methods: {available_methods}")
                return {"error": "store_memory not found"}
            
            print("Starting MemoryService...")
            await memory_service.start()
            print("MemoryService started")
        
        # Convert string layer to enum
        layer_map = {
            "knowledge_synthesis": MemoryLayer.KNOWLEDGE_SYNTHESIS,
            "conversation_history": MemoryLayer.CONVERSATION_HISTORY,
            "codebase_knowledge": MemoryLayer.CODEBASE_KNOWLEDGE,
            "stack_dependencies": MemoryLayer.STACK_DEPENDENCIES
        }
        
        layer = layer_map.get(request.layer)
        if not layer:
            raise HTTPException(status_code=400, detail=f"Invalid layer: {request.layer}")
        
        # Convert string memory type to enum
        memory_type_map = {
            "conversation": MemoryType.CONVERSATION,
            "code_snippet": MemoryType.CODE_SNIPPET,
            "documentation": MemoryType.DOCUMENTATION,
            "research_finding": MemoryType.RESEARCH_FINDING,
            "architectural_decision": MemoryType.ARCHITECTURAL_DECISION,
            "dependency_info": MemoryType.DEPENDENCY_INFO,
            "best_practice": MemoryType.BEST_PRACTICE,
            "error_solution": MemoryType.ERROR_SOLUTION
        }
        
        memory_type = memory_type_map.get(request.memory_type)
        if not memory_type:
            raise HTTPException(status_code=400, detail=f"Invalid memory type: {request.memory_type}")
        
        print(f"About to call store_memory with:")
        print(f"  layer: {layer}")
        print(f"  memory_type: {memory_type}")
        print(f"  content: {request.content[:50]}...")
        print(f"  metadata: {request.metadata}")
        print(f"  tags: {request.tags}")
        
        # Store the memory
        print("Calling store_memory...")
        memory_id = await memory_service.store_memory(
            layer=layer,
            memory_type=memory_type,
            content=request.content,
            metadata=request.metadata or {},
            tags=request.tags or []
        )
        print(f"store_memory returned: {memory_id}")
        
        return {
            "success": True,
            "memory_id": memory_id,
            "layer": request.layer,
            "memory_type": request.memory_type
        }
        
    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error storing memory: {str(e)}")

async def main():
    """Test the endpoint logic"""
    print("Testing exact FastAPI router logic...")
    
    # Create test request
    request = MemoryStoreRequest(
        layer="knowledge_synthesis",
        memory_type="research_finding",
        content="Test content for debugging",
        metadata={"source": "test", "topic": "debugging"},
        tags=["test", "debug"]
    )
    
    print(f"Test request: {request}")
    
    try:
        result = await store_memory_endpoint(request)
        print(f"Success: {result}")
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())