from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Import the interactive workflow components
from core.state.interactive_workflow_models import (
    InteractiveWorkflow,
    InteractiveWorkflowState,
    WorkflowStep,
    UserInteractionRequest,
    UserInteractionResponse,
    InteractiveWorkflowStatus,
    UserApprovalState,
    StepType,
    InteractionType,
    StepPriority
)
from core.state.interactive_state_manager import InteractiveStateManager
from core.workflow.approval_gates import ApprovalGateManager, ApprovalDecision
from core.workflow.step_by_step_orchestrator import StepByStepOrchestrator, OrchestrationMode
from core.workflow.task_breakdown_engine import InteractiveTaskBreakdownEngine, BreakdownStrategy
from core.models import EnhancedTask, TaskStatus

# Import API models
from app.api.models.interactive_workflow_schemas import (
    CreateWorkflowRequest,
    AddStepRequest,
    SubmitApprovalRequest,
    TaskBreakdownRequest,
    WorkflowControlRequest,
    WorkflowResponse,
    StepResponse,
    ApprovalResponse,
    WorkflowListResponse,
    TaskBreakdownResponse,
    WorkflowMetricsResponse,
    ExecutionReportResponse,
    ErrorResponse,
    SuccessResponse
)

router = APIRouter(prefix="/interactive-workflows", tags=["interactive-workflows"])

# Direct service instantiation for better performance
def get_state_manager() -> InteractiveStateManager:
    """Get interactive state manager instance"""
    return InteractiveStateManager()

def get_approval_gate_manager() -> ApprovalGateManager:
    """Get approval gate manager instance"""
    return ApprovalGateManager()

def get_orchestrator() -> StepByStepOrchestrator:
    """Get step-by-step orchestrator instance"""
    state_manager = get_state_manager()
    approval_manager = get_approval_gate_manager()
    return StepByStepOrchestrator(state_manager, approval_manager)

def get_breakdown_engine() -> InteractiveTaskBreakdownEngine:
    """Get task breakdown engine instance"""
    state_manager = get_state_manager()
    return InteractiveTaskBreakdownEngine(state_manager)

# Additional local models for specific endpoints
from pydantic import BaseModel

class StartWorkflowRequest(BaseModel):
    workflow_id: str

class ApprovalRequest(BaseModel):
    workflow_id: str
    step_id: str
    decision: str  # approve, reject, modify
    feedback: Optional[str] = None
    modifications: Optional[Dict[str, Any]] = None

@router.post("/create", response_model=WorkflowResponse, summary="Create a new interactive workflow")
async def create_workflow(
    request: CreateWorkflowRequest,
    state_manager: InteractiveStateManager = Depends(get_state_manager)
) -> WorkflowResponse:
    """Create a new interactive workflow"""
    try:
        # Convert orchestration mode string to enum
        mode_map = {
            "manual": OrchestrationMode.MANUAL,
            "semi_automatic": OrchestrationMode.SEMI_AUTOMATIC,
            "automatic": OrchestrationMode.AUTOMATIC,
            "guided": OrchestrationMode.GUIDED
        }
        mode = mode_map.get(request.orchestration_mode, OrchestrationMode.SEMI_AUTOMATIC)
        
        # Create workflow
        workflow_id = await state_manager.create_workflow(
            title=request.name,
            description=request.description,
            task_id=getattr(request, 'task_id', None)
        )
        
        # Auto-start if requested
        if request.auto_start:
            await state_manager.start_workflow(workflow_id)
        
        workflow_state = await state_manager.get_workflow_state(workflow_id)
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            name=request.name,
            description=request.description,
            status=workflow_state.status if workflow_state else InteractiveWorkflowStatus.CREATED,
            orchestration_mode=mode,
            created_at=workflow_state.created_at if workflow_state else datetime.now(),
            updated_at=workflow_state.updated_at if workflow_state else datetime.now(),
            current_step_index=workflow_state.current_step_index if workflow_state else 0,
            total_steps=len(workflow_state.steps) if workflow_state else 0,
            progress_percentage=workflow_state.get_progress_percentage() if workflow_state else 0.0,
            metadata=getattr(request, 'metadata', None)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

@router.post("/start")
def start_workflow(
    request: StartWorkflowRequest,
    orchestrator: StepByStepOrchestrator = Depends(get_orchestrator)
) -> JSONResponse:
    """Start an interactive workflow"""
    try:
        result = orchestrator.start_workflow(request.workflow_id)
        
        return JSONResponse({
            "status": "success",
            "execution_result": result.dict() if result else None,
            "message": f"Workflow {request.workflow_id} started successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")

@router.get("/list")
def list_workflows(
    status: Optional[str] = Query(None, description="Filter by workflow status"),
    state_manager: InteractiveStateManager = Depends(get_state_manager)
) -> JSONResponse:
    """List all interactive workflows"""
    try:
        workflows = state_manager.list_workflows()
        
        # Filter by status if provided
        if status:
            status_enum = InteractiveWorkflowStatus(status)
            workflows = [w for w in workflows if w.status == status_enum]
        
        return JSONResponse({
            "status": "success",
            "workflows": [w.dict() for w in workflows],
            "count": len(workflows)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")

@router.get("/{workflow_id}")
def get_workflow(
    workflow_id: str,
    state_manager: InteractiveStateManager = Depends(get_state_manager)
) -> JSONResponse:
    """Get detailed information about a specific workflow"""
    try:
        workflow_state = state_manager.get_workflow_state(workflow_id)
        
        if not workflow_state:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return JSONResponse({
            "status": "success",
            "workflow_state": workflow_state.dict()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")

@router.get("/{workflow_id}/steps")
async def get_workflow_steps(
    workflow_id: str,
    state_manager: InteractiveStateManager = Depends(get_state_manager)
) -> JSONResponse:
    """Get all steps for a specific workflow"""
    try:
        steps = await state_manager.get_workflow_steps(workflow_id)
        
        return JSONResponse({
            "status": "success",
            "workflow_id": workflow_id,
            "steps": [step.dict() for step in steps],
            "count": len(steps)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow steps: {str(e)}")

@router.get("/{workflow_id}/pending-approvals")
async def get_pending_approvals(
    workflow_id: str,
    state_manager: InteractiveStateManager = Depends(get_state_manager)
) -> JSONResponse:
    """Get all pending approval requests for a workflow"""
    try:
        approvals = await state_manager.get_pending_approvals(workflow_id)
        
        return JSONResponse({
            "status": "success",
            "workflow_id": workflow_id,
            "pending_approvals": [approval.dict() for approval in approvals],
            "count": len(approvals)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pending approvals: {str(e)}")

@router.post("/approve")
async def submit_approval(
    request: ApprovalRequest,
    state_manager: InteractiveStateManager = Depends(get_state_manager)
) -> JSONResponse:
    """Submit an approval decision for a workflow step"""
    try:
        # Convert decision string to enum
        decision_map = {
            "approve": ApprovalDecision.APPROVED,
            "reject": ApprovalDecision.REJECTED,
            "modify": ApprovalDecision.NEEDS_MODIFICATION
        }
        decision = decision_map.get(request.decision)
        
        if not decision:
            raise HTTPException(status_code=400, detail=f"Invalid decision: {request.decision}")
        
        # Create approval response
        response = UserInteractionResponse(
            interaction_id=f"{request.workflow_id}_{request.step_id}",
            response_type=InteractionType.APPROVAL,
            decision=decision.value,
            feedback=request.feedback,
            modifications=request.modifications or {},
            timestamp=datetime.now()
        )
        
        # Submit approval
        result = await state_manager.submit_user_approval(
            workflow_id=request.workflow_id,
            step_id=request.step_id,
            response=response
        )
        
        return JSONResponse({
            "status": "success",
            "approval_result": result,
            "message": f"Approval submitted for step {request.step_id}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit approval: {str(e)}")

@router.post("/control")
async def control_workflow(
    request: WorkflowControlRequest,
    orchestrator: StepByStepOrchestrator = Depends(get_orchestrator)
) -> JSONResponse:
    """Control workflow execution (pause, resume, cancel, restart)"""
    try:
        result = None
        
        if request.action == "pause":
            result = await orchestrator.pause_workflow(request.workflow_id)
        elif request.action == "resume":
            result = await orchestrator.resume_workflow(request.workflow_id)
        elif request.action == "cancel":
            result = await orchestrator.cancel_workflow(request.workflow_id)
        elif request.action == "restart":
            result = await orchestrator.restart_workflow(request.workflow_id)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {request.action}")
        
        return JSONResponse({
            "status": "success",
            "action": request.action,
            "workflow_id": request.workflow_id,
            "result": result,
            "message": f"Workflow {request.action} completed successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to {request.action} workflow: {str(e)}")

@router.post("/breakdown-task")
async def breakdown_task(
    request: TaskBreakdownRequest,
    breakdown_engine: InteractiveTaskBreakdownEngine = Depends(get_breakdown_engine)
) -> JSONResponse:
    """Break down a high-level task into interactive workflow steps"""
    try:
        # Convert dict to EnhancedTask
        task = EnhancedTask(**request.task)
        
        # Perform breakdown
        breakdown_result = await breakdown_engine.breakdown_task(
            task=task,
            strategy=request.strategy
        )
        
        return JSONResponse({
            "status": "success",
            "breakdown_result": breakdown_result.dict(),
            "message": f"Task breakdown completed with {len(breakdown_result.steps)} steps"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to breakdown task: {str(e)}")

@router.get("/{workflow_id}/metrics")
async def get_workflow_metrics(
    workflow_id: str,
    orchestrator: StepByStepOrchestrator = Depends(get_orchestrator)
) -> JSONResponse:
    """Get execution metrics for a specific workflow"""
    try:
        metrics = await orchestrator.get_workflow_metrics(workflow_id)
        
        return JSONResponse({
            "status": "success",
            "workflow_id": workflow_id,
            "metrics": metrics
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow metrics: {str(e)}")

@router.get("/{workflow_id}/execution-report")
async def get_execution_report(
    workflow_id: str,
    orchestrator: StepByStepOrchestrator = Depends(get_orchestrator)
) -> JSONResponse:
    """Get detailed execution report for a workflow"""
    try:
        report = await orchestrator.get_execution_report(workflow_id)
        
        return JSONResponse({
            "status": "success",
            "workflow_id": workflow_id,
            "execution_report": report.dict() if report else None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get execution report: {str(e)}")

@router.delete("/{workflow_id}")
def delete_workflow(
    workflow_id: str,
    state_manager: InteractiveStateManager = Depends(get_state_manager)
) -> JSONResponse:
    """Delete a workflow and all its associated data"""
    try:
        success = state_manager.delete_workflow(workflow_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        
        return JSONResponse({
            "status": "success",
            "workflow_id": workflow_id,
            "message": "Workflow deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete workflow: {str(e)}")

@router.get("/health")
async def health_check() -> JSONResponse:
    """Health check for interactive workflows service"""
    try:
        # Check if all components are accessible
        state_manager = get_state_manager()
        approval_manager = get_approval_gate_manager()
        orchestrator = get_orchestrator()
        breakdown_engine = get_breakdown_engine()
        
        return JSONResponse({
            "status": "healthy",
            "components": {
                "state_manager": "operational",
                "approval_manager": "operational",
                "orchestrator": "operational",
                "breakdown_engine": "operational"
            },
            "message": "Interactive workflows service is operational"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Interactive workflows service is not operational"
            }
        )