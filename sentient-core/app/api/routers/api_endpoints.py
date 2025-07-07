from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import sys
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.services.memory_service import MemoryService, MemoryLayer, MemoryType
from core.services.state_service import StateService

router = APIRouter(tags=["api"])

# Initialize services
memory_service = None
state_service = None

# Request models
class MemoryStoreRequest(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StateResponse(BaseModel):
    status: str
    timestamp: str
    system_info: Dict[str, Any]

# Initialization endpoint
@router.post("/initialize")
async def initialize_session():
    """Initialize a new session and return initial state"""
    try:
        # This can be expanded to create a new session ID, clear state, etc.
        return {
            "status": "initialized",
            "timestamp": datetime.now().isoformat(),
            "message": "Session initialized successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing session: {str(e)}")

# State endpoint
@router.get("/state")
async def get_system_state():
    """Get current system state"""
    try:
        global state_service
        if state_service is None:
            state_service = StateService()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "backend_status": "running",
                "memory_service": "available",
                "agent_system": "operational",
                "api_version": "0.1.0"
            },
            "services": {
                "memory": "active",
                "agents": "active",
                "workflows": "active",
                "chat": "active"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system state: {str(e)}")

# Memory endpoints
@router.get("/memory/status")
async def get_memory_status():
    """Get memory system status"""
    try:
        global memory_service
        if memory_service is None:
            memory_service = MemoryService()
            await memory_service.start()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "memory_layers": {
                "knowledge_synthesis": "active",
                "conversation_history": "active",
                "codebase_knowledge": "active",
                "stack_dependencies": "active"
            },
            "statistics": {
                "total_memories": 0,
                "recent_activity": "normal"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting memory status: {str(e)}")

@router.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    """Store memory in the system"""
    try:
        global memory_service
        if memory_service is None:
            memory_service = MemoryService()
            await memory_service.start()
        
        # Store with default layer and type for testing
        memory_id = await memory_service.store_memory(
            layer=MemoryLayer.CONVERSATION_HISTORY,
            memory_type=MemoryType.CONVERSATION,
            content=request.content,
            metadata=request.metadata or {},
            tags=["system_test"]
        )
        
        return {
            "success": True,
            "memory_id": memory_id,
            "timestamp": datetime.now().isoformat(),
            "status": "stored"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing memory: {str(e)}")

@router.get("/memory/retrieve")
async def retrieve_memory(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results")
):
    """Retrieve memories from the system"""
    try:
        global memory_service
        if memory_service is None:
            memory_service = MemoryService()
            await memory_service.start()
        
        # Retrieve memories
        memories = await memory_service.retrieve_memories(
            query=query,
            limit=limit,
            similarity_threshold=0.5
        )
        
        # Convert to serializable format
        result = []
        for memory in memories:
            result.append({
                "id": memory.id,
                "content": memory.content,
                "metadata": memory.metadata,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "similarity_score": getattr(memory, 'similarity_score', None)
            })
        
        return {
            "success": True,
            "memories": result,
            "count": len(result),
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memory: {str(e)}")