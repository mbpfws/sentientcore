from fastapi import APIRouter, HTTPException, Query, Body, Depends, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import sys
from datetime import datetime

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.services.service_factory import get_service_factory, ServiceFactory
from core.services.memory_service import MemoryService, MemoryLayer, MemoryType
from core.services.state_service import StateService, AgentStatus, WorkflowStatus
from core.services.search_service import SearchService, SearchType, SearchQuery

router = APIRouter(tags=["unified"])

# Pydantic models for request/response
class MemoryStoreRequest(BaseModel):
    layer: Optional[str] = None
    memory_type: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class MemoryRetrieveRequest(BaseModel):
    query: str
    layer: Optional[str] = None
    memory_type: Optional[str] = None
    limit: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.7

class SearchRequest(BaseModel):
    query: str
    search_type: str = "knowledge"
    providers: Optional[List[str]] = None
    max_results: Optional[int] = 10
    include_metadata: Optional[bool] = True

class StateUpdateRequest(BaseModel):
    entity_id: str
    updates: Dict[str, Any]

# Dependency to get services
def get_memory_service(request: Request) -> MemoryService:
    return request.app.state.service_factory.memory_service

def get_state_service(request: Request) -> StateService:
    return request.app.state.service_factory.state_service

def get_search_service(request: Request) -> SearchService:
    return request.app.state.service_factory.search_service

# Endpoints
@router.post("/initialize")
async def initialize_session():
    return {
        "status": "initialized",
        "timestamp": datetime.now().isoformat(),
        "message": "Session initialized successfully."
    }

@router.get("/state")
async def get_system_state(state_service: StateService = Depends(get_state_service)):
    # This can be expanded to return actual state from the state_service
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "backend_status": "running",
            "memory_service": "available",
            "agent_system": "operational",
            "api_version": "0.1.0"
        }
    }

@router.post("/memory/store")
async def store_memory(request: MemoryStoreRequest, memory_service: MemoryService = Depends(get_memory_service)):
    try:
        layer = MemoryLayer(request.layer) if request.layer else MemoryLayer.CONVERSATION_HISTORY
        memory_type = MemoryType(request.memory_type) if request.memory_type else MemoryType.CONVERSATION

        memory_id = await memory_service.store_memory(
            layer=layer,
            memory_type=memory_type,
            content=request.content,
            metadata=request.metadata or {},
            tags=request.tags or []
        )
        return {
            "success": True,
            "memory_id": memory_id,
            "layer": layer.value,
            "memory_type": memory_type.value
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid layer or memory_type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing memory: {str(e)}")

@router.post("/memory/retrieve")
async def retrieve_memory(request: MemoryRetrieveRequest, memory_service: MemoryService = Depends(get_memory_service)):
    try:
        layer = MemoryLayer(request.layer) if request.layer else None
        memory_type = MemoryType(request.memory_type) if request.memory_type else None

        memories = await memory_service.retrieve_memories(
            query=request.query,
            layer=layer,
            memory_type=memory_type,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        result = [
            {
                "id": mem.id,
                "content": mem.content,
                "layer": mem.layer.value,
                "memory_type": mem.memory_type.value,
                "metadata": mem.metadata,
                "tags": mem.tags,
                "created_at": mem.created_at.isoformat(),
                "similarity_score": getattr(mem, 'similarity_score', None)
            }
            for mem in memories
        ]
        return {
            "success": True,
            "memories": result,
            "count": len(result)
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid layer or memory_type: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memory: {str(e)}")