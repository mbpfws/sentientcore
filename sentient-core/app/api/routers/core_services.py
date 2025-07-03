from fastapi import APIRouter, HTTPException, Depends, Query, Body
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
from core.services.state_service import StateService, AgentStatus, WorkflowStatus
from core.services.search_service import SearchService, SearchType, SearchQuery
from core.services.vector_service import EnhancedVectorService as VectorService

router = APIRouter(prefix="/core-services", tags=["core-services"])

# Initialize services (these would typically be dependency injected)
memory_service = None
state_service = None
search_service = None
vector_service = None

# Pydantic models for request/response
class MemoryStoreRequest(BaseModel):
    """Request model for storing memory"""
    layer: str  # knowledge_synthesis, conversation_history, codebase_knowledge, stack_dependencies
    memory_type: str  # conversation, code_snippet, documentation, etc.
    content: str
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class MemoryRetrieveRequest(BaseModel):
    """Request model for retrieving memory"""
    query: str
    layer: Optional[str] = None
    memory_type: Optional[str] = None
    limit: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.7

class SearchRequest(BaseModel):
    """Request model for search operations"""
    query: str
    search_type: str = "knowledge"  # knowledge, code, documentation
    providers: Optional[List[str]] = None  # tavily, exa, duckduckgo
    max_results: Optional[int] = 10
    include_metadata: Optional[bool] = True

class StateUpdateRequest(BaseModel):
    """Request model for state updates"""
    entity_id: str
    updates: Dict[str, Any]

# Memory Management Endpoints
@router.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    """Store information in the hierarchical memory system"""
    try:
        global memory_service
        if memory_service is None:
            memory_service = MemoryService()
            await memory_service.start()
        
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
        
        # Store the memory
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
            "layer": request.layer,
            "memory_type": request.memory_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing memory: {str(e)}")

@router.post("/memory/retrieve")
async def retrieve_memory(request: MemoryRetrieveRequest):
    """Retrieve information from the hierarchical memory system"""
    try:
        global memory_service
        if memory_service is None:
            memory_service = MemoryService()
            await memory_service.start()
        
        # Convert layer if specified
        layer = None
        if request.layer:
            layer_map = {
                "knowledge_synthesis": MemoryLayer.KNOWLEDGE_SYNTHESIS,
                "conversation_history": MemoryLayer.CONVERSATION_HISTORY,
                "codebase_knowledge": MemoryLayer.CODEBASE_KNOWLEDGE,
                "stack_dependencies": MemoryLayer.STACK_DEPENDENCIES
            }
            layer = layer_map.get(request.layer)
        
        # Convert memory type if specified
        memory_type = None
        if request.memory_type:
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
        
        # Retrieve memories
        memories = await memory_service.retrieve_memories(
            query=request.query,
            layer=layer,
            memory_type=memory_type,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        # Convert to serializable format
        result = []
        for memory in memories:
            result.append({
                "id": memory.id,
                "content": memory.content,
                "layer": memory.layer.value if memory.layer else None,
                "memory_type": memory.memory_type.value if memory.memory_type else None,
                "metadata": memory.metadata,
                "tags": memory.tags,
                "created_at": memory.created_at.isoformat() if memory.created_at else None,
                "similarity_score": getattr(memory, 'similarity_score', None)
            })
        
        return {
            "success": True,
            "memories": result,
            "count": len(result)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memory: {str(e)}")

@router.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    try:
        global memory_service
        if memory_service is None:
            memory_service = MemoryService()
            await memory_service.start()
        
        stats = await memory_service.get_memory_stats()
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting memory stats: {str(e)}")

# State Management Endpoints
@router.get("/state/agents")
async def get_agent_states():
    """Get current state of all agents"""
    try:
        global state_service
        if state_service is None:
            state_service = StateService()
            await state_service.start()
        
        agent_states = await state_service.get_all_agent_states()
        
        # Convert to serializable format
        result = {}
        for agent_id, state in agent_states.items():
            result[agent_id] = {
                "agent_id": state.agent_id,
                "status": state.status.value,
                "current_task": state.current_task,
                "last_activity": state.last_activity.isoformat() if state.last_activity else None,
                "metadata": state.metadata,
                "performance_metrics": state.performance_metrics,
                "error_info": state.error_info
            }
        
        return {
            "success": True,
            "agent_states": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent states: {str(e)}")

@router.get("/state/workflow/{workflow_id}")
async def get_workflow_state(workflow_id: str):
    """Get state of a specific workflow"""
    try:
        global state_service
        if state_service is None:
            state_service = StateService()
            await state_service.start()
        
        workflow_state = await state_service.get_workflow_state(workflow_id)
        
        if not workflow_state:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return {
            "success": True,
            "workflow_state": {
                "workflow_id": workflow_state.workflow_id,
                "status": workflow_state.status.value,
                "current_step": workflow_state.current_step,
                "progress": workflow_state.progress,
                "started_at": workflow_state.started_at.isoformat() if workflow_state.started_at else None,
                "completed_at": workflow_state.completed_at.isoformat() if workflow_state.completed_at else None,
                "steps_completed": workflow_state.steps_completed,
                "steps_remaining": workflow_state.steps_remaining,
                "metadata": workflow_state.metadata,
                "error_info": workflow_state.error_info
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting workflow state: {str(e)}")

@router.post("/state/agents/{agent_id}/update")
async def update_agent_state(agent_id: str, request: StateUpdateRequest):
    """Update state of a specific agent"""
    try:
        global state_service
        if state_service is None:
            state_service = StateService()
            await state_service.start()
        
        # Convert status string to enum if provided
        if 'status' in request.updates and isinstance(request.updates['status'], str):
            status_map = {
                "idle": AgentStatus.IDLE,
                "active": AgentStatus.ACTIVE,
                "busy": AgentStatus.BUSY,
                "error": AgentStatus.ERROR,
                "offline": AgentStatus.OFFLINE
            }
            status = status_map.get(request.updates['status'])
            if status:
                request.updates['status'] = status
        
        updated_state = await state_service.update_agent_state(agent_id, **request.updates)
        
        return {
            "success": True,
            "agent_state": {
                "agent_id": updated_state.agent_id,
                "status": updated_state.status.value,
                "current_task": updated_state.current_task,
                "last_activity": updated_state.last_activity.isoformat() if updated_state.last_activity else None,
                "metadata": updated_state.metadata,
                "performance_metrics": updated_state.performance_metrics,
                "error_info": updated_state.error_info
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating agent state: {str(e)}")

@router.post("/state/workflow/{workflow_id}/update")
async def update_workflow_state(workflow_id: str, request: StateUpdateRequest):
    """Update state of a specific workflow"""
    try:
        global state_service
        if state_service is None:
            state_service = StateService()
            await state_service.start()
        
        # Convert status string to enum if provided
        if 'status' in request.updates and isinstance(request.updates['status'], str):
            status_map = {
                "pending": WorkflowStatus.PENDING,
                "running": WorkflowStatus.RUNNING,
                "paused": WorkflowStatus.PAUSED,
                "completed": WorkflowStatus.COMPLETED,
                "failed": WorkflowStatus.FAILED,
                "cancelled": WorkflowStatus.CANCELLED
            }
            status = status_map.get(request.updates['status'])
            if status:
                request.updates['status'] = status
        
        updated_state = await state_service.update_workflow_state(workflow_id, **request.updates)
        
        return {
            "success": True,
            "workflow_state": {
                "workflow_id": updated_state.workflow_id,
                "status": updated_state.status.value,
                "current_step": updated_state.current_step,
                "progress": updated_state.progress,
                "started_at": updated_state.started_at.isoformat() if updated_state.started_at else None,
                "completed_at": updated_state.completed_at.isoformat() if updated_state.completed_at else None,
                "steps_completed": updated_state.steps_completed,
                "steps_remaining": updated_state.steps_remaining,
                "metadata": updated_state.metadata,
                "error_info": updated_state.error_info
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating workflow state: {str(e)}")

# Search Service Endpoints
@router.post("/search/knowledge")
async def search_knowledge(request: SearchRequest):
    """Search for knowledge using multiple providers"""
    try:
        global search_service
        if search_service is None:
            search_service = SearchService()
        
        # Convert search type
        search_type_map = {
            "knowledge": SearchType.KNOWLEDGE,
            "code": SearchType.CODE,
            "documentation": SearchType.DOCUMENTATION
        }
        
        search_type = search_type_map.get(request.search_type, SearchType.KNOWLEDGE)
        
        # Create search query
        search_query = SearchQuery(
            query=request.query,
            search_type=search_type,
            max_results=request.max_results or 10,
            providers=request.providers,
            include_metadata=request.include_metadata or True
        )
        
        # Perform search
        results = await search_service.search(search_query)
        
        # Convert to serializable format
        serialized_results = []
        for result in results:
            serialized_results.append({
                "title": result.title,
                "content": result.content,
                "url": result.url,
                "source": result.source,
                "relevance_score": result.relevance_score,
                "metadata": result.metadata,
                "created_at": result.created_at.isoformat() if result.created_at else None
            })
        
        return {
            "success": True,
            "results": serialized_results,
            "count": len(serialized_results),
            "query": request.query,
            "search_type": request.search_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

@router.get("/search/providers")
async def get_search_providers():
    """Get available search providers and their status"""
    try:
        global search_service
        if search_service is None:
            search_service = SearchService()
        
        providers = search_service.get_available_providers()
        
        return {
            "success": True,
            "providers": providers
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting search providers: {str(e)}")

# Service Health and Status
@router.get("/health")
async def get_services_health():
    """Get health status of all core services"""
    try:
        health_status = {
            "memory_service": memory_service is not None,
            "state_service": state_service is not None,
            "search_service": search_service is not None,
            "vector_service": vector_service is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Get performance stats if services are available
        if state_service:
            health_status["state_performance"] = await state_service.get_performance_stats()
        
        if memory_service:
            health_status["memory_stats"] = await memory_service.get_memory_stats()
        
        return {
            "success": True,
            "health": health_status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting service health: {str(e)}")

# Service lifecycle management
@router.post("/initialize")
async def initialize_services():
    """Initialize all core services"""
    try:
        global memory_service, state_service, search_service, vector_service
        
        # Initialize services
        if memory_service is None:
            memory_service = MemoryService()
        
        if state_service is None:
            state_service = StateService()
            await state_service.start()
        
        if search_service is None:
            search_service = SearchService()
        
        if vector_service is None:
            from core.services.vector_service import SentenceTransformerProvider, ChromaVectorStore
            embedding_provider = SentenceTransformerProvider()
            vector_store = ChromaVectorStore(
                collection_name="sentient_vectors",
                persist_directory="./vector_db"
            )
            vector_service = VectorService(embedding_provider=embedding_provider, vector_store=vector_store)
        
        return {
            "success": True,
            "message": "All core services initialized successfully",
            "services": {
                "memory_service": True,
                "state_service": True,
                "search_service": True,
                "vector_service": True
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing services: {str(e)}")