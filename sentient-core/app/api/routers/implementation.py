from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import json
import asyncio
import time
from pathlib import Path

# Import implementation agent
from core.agents.feature_implementation_agent import get_feature_implementation_agent
from core.services.enhanced_llm_service_main import get_enhanced_llm_service
from core.services.memory_service import get_memory_service, MemoryLayer

# Import models
from pydantic import BaseModel
from enum import Enum

class ImplementationMode(str, Enum):
    VALIDATION = "validation"
    PHASE2 = "phase2"
    PHASE3 = "phase3"
    PHASE4 = "phase4"
    FULL = "full"

class ImplementationPhase(str, Enum):
    PLAN_INGESTION = "plan_ingestion"
    CODE_GENERATION = "code_generation"
    VALIDATION_TESTING = "validation_testing"
    REPORTING_OUTPUT = "reporting_output"
    COMPLETED = "completed"
    ERROR = "error"

class ImplementationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

class BuildStatus(str, Enum):
    PENDING = "pending"
    BUILDING = "building"
    SUCCESS = "success"
    FAILED = "failed"

# Request/Response Models
class ImplementationRequest(BaseModel):
    feature_build_plan: str
    synthesized_knowledge_docs: List[str]
    implementation_mode: ImplementationMode = ImplementationMode.FULL
    project_root: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ImplementationArtifact(BaseModel):
    name: str
    path: str
    type: str  # file, directory, build_output, test_report
    size: Optional[int] = None
    created_at: datetime
    content_preview: Optional[str] = None

class TestResult(BaseModel):
    test_name: str
    status: TestStatus
    duration: Optional[float] = None
    error_message: Optional[str] = None
    output: Optional[str] = None
    timestamp: datetime

class ImplementationProgress(BaseModel):
    implementation_id: str
    status: ImplementationStatus
    current_phase: ImplementationPhase
    progress_percentage: float
    phases_completed: List[ImplementationPhase]
    current_task: Optional[str] = None
    artifacts: List[ImplementationArtifact]
    test_results: List[TestResult]
    build_status: BuildStatus
    error_message: Optional[str] = None
    started_at: datetime
    updated_at: datetime
    estimated_completion: Optional[datetime] = None

class ImplementationResult(BaseModel):
    implementation_id: str
    status: ImplementationStatus
    final_phase: ImplementationPhase
    total_duration: float
    artifacts: List[ImplementationArtifact]
    test_results: List[TestResult]
    build_status: BuildStatus
    success_rate: float
    error_message: Optional[str] = None
    completed_at: datetime
    summary: str

class ImplementationUpdate(BaseModel):
    implementation_id: str
    phase: ImplementationPhase
    status: ImplementationStatus
    progress_percentage: float
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None

router = APIRouter(prefix="/implementation", tags=["implementation"])

# In-memory storage for implementation tracking (in production, use a database)
implementation_store: Dict[str, Dict[str, Any]] = {}
active_implementations: Dict[str, asyncio.Task] = {}

def get_implementation_agent():
    """Get feature implementation agent instance"""
    llm_service = get_enhanced_llm_service()
    memory_service = get_memory_service()
    return get_feature_implementation_agent(llm_service, memory_service)

@router.post("/start", response_model=Dict[str, str])
async def start_implementation(
    request: ImplementationRequest,
    background_tasks: BackgroundTasks,
    agent = Depends(get_implementation_agent)
) -> Dict[str, str]:
    """Start a new implementation workflow"""
    try:
        implementation_id = str(uuid.uuid4())
        
        # Initialize implementation tracking
        implementation_store[implementation_id] = {
            "id": implementation_id,
            "status": ImplementationStatus.PENDING,
            "current_phase": ImplementationPhase.PLAN_INGESTION,
            "progress_percentage": 0.0,
            "phases_completed": [],
            "artifacts": [],
            "test_results": [],
            "build_status": BuildStatus.PENDING,
            "started_at": datetime.now(),
            "updated_at": datetime.now(),
            "request": request.dict()
        }
        
        # Start implementation in background
        task = asyncio.create_task(
            run_implementation(implementation_id, request, agent)
        )
        active_implementations[implementation_id] = task
        
        return {
            "implementation_id": implementation_id,
            "status": "started",
            "message": "Implementation workflow started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start implementation: {str(e)}")

async def run_implementation(
    implementation_id: str,
    request: ImplementationRequest,
    agent
):
    """Run the implementation workflow in background"""
    try:
        # Update status to running
        implementation_store[implementation_id]["status"] = ImplementationStatus.RUNNING
        implementation_store[implementation_id]["updated_at"] = datetime.now()
        
        # Prepare agent parameters
        agent_params = {
            "plan_content": request.feature_build_plan,
            "knowledge_documents": request.synthesized_knowledge_docs or [],
            "mode": request.implementation_mode.value
        }
        
        # Execute implementation
        result = await agent.invoke(agent_params)
        
        # Update with success
        implementation_store[implementation_id].update({
            "status": ImplementationStatus.COMPLETED,
            "current_phase": ImplementationPhase.COMPLETED,
            "progress_percentage": 100.0,
            "updated_at": datetime.now(),
            "result": result
        })
        
    except Exception as e:
        # Update with error
        implementation_store[implementation_id].update({
            "status": ImplementationStatus.FAILED,
            "current_phase": ImplementationPhase.ERROR,
            "error_message": str(e),
            "updated_at": datetime.now()
        })
    
    finally:
        # Clean up active task
        if implementation_id in active_implementations:
            del active_implementations[implementation_id]

@router.get("/progress/{implementation_id}", response_model=ImplementationProgress)
async def get_implementation_progress(
    implementation_id: str
) -> ImplementationProgress:
    """Get current progress of an implementation"""
    if implementation_id not in implementation_store:
        raise HTTPException(status_code=404, detail="Implementation not found")
    
    data = implementation_store[implementation_id]
    
    return ImplementationProgress(
        implementation_id=implementation_id,
        status=data["status"],
        current_phase=data["current_phase"],
        progress_percentage=data["progress_percentage"],
        phases_completed=data["phases_completed"],
        current_task=data.get("current_task"),
        artifacts=data["artifacts"],
        test_results=data["test_results"],
        build_status=data["build_status"],
        error_message=data.get("error_message"),
        started_at=data["started_at"],
        updated_at=data["updated_at"],
        estimated_completion=data.get("estimated_completion")
    )

@router.get("/result/{implementation_id}", response_model=ImplementationResult)
async def get_implementation_result(
    implementation_id: str
) -> ImplementationResult:
    """Get final result of a completed implementation"""
    if implementation_id not in implementation_store:
        raise HTTPException(status_code=404, detail="Implementation not found")
    
    data = implementation_store[implementation_id]
    
    if data["status"] not in [ImplementationStatus.COMPLETED, ImplementationStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Implementation not yet completed")
    
    # Calculate metrics
    total_duration = (data["updated_at"] - data["started_at"]).total_seconds()
    success_rate = 100.0 if data["status"] == ImplementationStatus.COMPLETED else 0.0
    
    return ImplementationResult(
        implementation_id=implementation_id,
        status=data["status"],
        final_phase=data["current_phase"],
        total_duration=total_duration,
        artifacts=data["artifacts"],
        test_results=data["test_results"],
        build_status=data["build_status"],
        success_rate=success_rate,
        error_message=data.get("error_message"),
        completed_at=data["updated_at"],
        summary=data.get("result", {}).get("summary", "Implementation completed")
    )

@router.get("/list")
async def list_implementations(
    status: Optional[ImplementationStatus] = Query(None),
    limit: int = Query(50, ge=1, le=100)
) -> Dict[str, Any]:
    """List all implementations with optional filtering"""
    implementations = list(implementation_store.values())
    
    # Filter by status if provided
    if status:
        implementations = [impl for impl in implementations if impl["status"] == status]
    
    # Sort by creation time (newest first) and limit
    implementations.sort(key=lambda x: x["started_at"], reverse=True)
    implementations = implementations[:limit]
    
    return {
        "implementations": implementations,
        "total": len(implementations),
        "active_count": len(active_implementations)
    }

@router.post("/cancel/{implementation_id}")
async def cancel_implementation(
    implementation_id: str
) -> Dict[str, str]:
    """Cancel a running implementation"""
    if implementation_id not in implementation_store:
        raise HTTPException(status_code=404, detail="Implementation not found")
    
    # Cancel the background task if it's running
    if implementation_id in active_implementations:
        task = active_implementations[implementation_id]
        task.cancel()
        del active_implementations[implementation_id]
    
    # Update status
    implementation_store[implementation_id].update({
        "status": ImplementationStatus.CANCELLED,
        "updated_at": datetime.now()
    })
    
    return {
        "implementation_id": implementation_id,
        "status": "cancelled",
        "message": "Implementation cancelled successfully"
    }

@router.get("/artifacts/{implementation_id}")
async def get_implementation_artifacts(
    implementation_id: str
) -> Dict[str, Any]:
    """Get all artifacts generated by an implementation"""
    if implementation_id not in implementation_store:
        raise HTTPException(status_code=404, detail="Implementation not found")
    
    data = implementation_store[implementation_id]
    
    return {
        "implementation_id": implementation_id,
        "artifacts": data["artifacts"],
        "total_artifacts": len(data["artifacts"])
    }

@router.get("/artifacts/{implementation_id}/{artifact_name}/download")
async def download_artifact(
    implementation_id: str,
    artifact_name: str
) -> StreamingResponse:
    """Download a specific artifact"""
    if implementation_id not in implementation_store:
        raise HTTPException(status_code=404, detail="Implementation not found")
    
    data = implementation_store[implementation_id]
    artifact = next((a for a in data["artifacts"] if a["name"] == artifact_name), None)
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    # In a real implementation, you would stream the actual file
    # For now, return a placeholder
    def generate_content():
        yield f"Content of {artifact_name}\n"
        yield f"Implementation ID: {implementation_id}\n"
        yield f"Generated at: {artifact['created_at']}\n"
    
    return StreamingResponse(
        generate_content(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={artifact_name}"}
    )

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check for implementation service"""
    return {
        "status": "healthy",
        "active_implementations": len(active_implementations),
        "total_implementations": len(implementation_store),
        "timestamp": datetime.now()
    }