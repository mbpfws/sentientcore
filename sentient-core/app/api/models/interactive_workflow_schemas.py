from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Import the core models
from core.state.interactive_workflow_models import (
    InteractiveWorkflowStatus,
    UserApprovalState,
    StepType,
    InteractionType,
    StepPriority,
    WorkflowStep,
    InteractiveWorkflow,
    UserInteractionRequest,
    UserInteractionResponse
)
from core.workflow.approval_gates import ApprovalDecision
from core.workflow.step_by_step_orchestrator import OrchestrationMode
from core.workflow.task_breakdown_engine import BreakdownStrategy, ComplexityLevel, TaskCategory

# Request Models
class CreateWorkflowRequest(BaseModel):
    """Request model for creating a new interactive workflow"""
    name: str = Field(..., description="Name of the workflow", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Description of the workflow", max_length=1000)
    orchestration_mode: OrchestrationMode = Field(OrchestrationMode.SEMI_AUTOMATIC, description="Orchestration mode for the workflow")
    auto_start: bool = Field(False, description="Whether to automatically start the workflow after creation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the workflow")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Feature Implementation Workflow",
                "description": "Interactive workflow for implementing a new feature",
                "orchestration_mode": "semi_automatic",
                "auto_start": False,
                "metadata": {
                    "project_id": "proj_123",
                    "priority": "high"
                }
            }
        }

class AddStepRequest(BaseModel):
    """Request model for adding a step to a workflow"""
    step_id: str = Field(..., description="Unique identifier for the step")
    name: str = Field(..., description="Name of the step", min_length=1, max_length=200)
    description: Optional[str] = Field(None, description="Description of the step", max_length=1000)
    step_type: StepType = Field(..., description="Type of the step")
    priority: StepPriority = Field(StepPriority.MEDIUM, description="Priority of the step")
    requires_approval: bool = Field(True, description="Whether the step requires user approval")
    dependencies: List[str] = Field(default_factory=list, description="List of step IDs this step depends on")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated duration in minutes", ge=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the step")
    
    class Config:
        schema_extra = {
            "example": {
                "step_id": "step_001",
                "name": "Generate API Endpoints",
                "description": "Create FastAPI endpoints for the new feature",
                "step_type": "code_generation",
                "priority": "high",
                "requires_approval": True,
                "dependencies": [],
                "estimated_duration_minutes": 30,
                "metadata": {
                    "file_count": 3,
                    "complexity": "medium"
                }
            }
        }

class SubmitApprovalRequest(BaseModel):
    """Request model for submitting an approval decision"""
    decision: ApprovalDecision = Field(..., description="The approval decision")
    comments: Optional[str] = Field(None, description="Comments about the approval decision", max_length=1000)
    requested_changes: Optional[List[str]] = Field(None, description="List of requested changes if rejected")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the approval")
    
    class Config:
        schema_extra = {
            "example": {
                "decision": "approved",
                "comments": "Code looks good, ready to proceed",
                "requested_changes": [],
                "metadata": {
                    "reviewer_id": "user_123",
                    "review_duration_minutes": 15
                }
            }
        }

class TaskBreakdownRequest(BaseModel):
    """Request model for breaking down a task into workflow steps"""
    task_description: str = Field(..., description="Description of the task to break down", min_length=1, max_length=2000)
    strategy: BreakdownStrategy = Field(BreakdownStrategy.CODE_DEVELOPMENT, description="Strategy to use for breakdown")
    complexity_hint: Optional[ComplexityLevel] = Field(None, description="Hint about the expected complexity")
    category_hint: Optional[TaskCategory] = Field(None, description="Hint about the task category")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the breakdown")
    max_steps: Optional[int] = Field(10, description="Maximum number of steps to generate", ge=1, le=50)
    
    class Config:
        schema_extra = {
            "example": {
                "task_description": "Implement user authentication system with JWT tokens",
                "strategy": "code_development",
                "complexity_hint": "high",
                "category_hint": "backend_development",
                "context": {
                    "framework": "FastAPI",
                    "database": "PostgreSQL",
                    "existing_user_model": True
                },
                "max_steps": 8
            }
        }

class WorkflowControlRequest(BaseModel):
    """Request model for workflow control operations (pause, resume, cancel, restart)"""
    reason: Optional[str] = Field(None, description="Reason for the control action", max_length=500)
    force: bool = Field(False, description="Whether to force the action even if workflow is in incompatible state")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the control action")
    
    class Config:
        schema_extra = {
            "example": {
                "reason": "Need to update requirements before proceeding",
                "force": False,
                "metadata": {
                    "requested_by": "user_123",
                    "priority": "normal"
                }
            }
        }

# Response Models
class WorkflowResponse(BaseModel):
    """Response model for workflow operations"""
    workflow_id: str
    name: str
    description: Optional[str]
    status: InteractiveWorkflowStatus
    orchestration_mode: OrchestrationMode
    created_at: datetime
    updated_at: datetime
    current_step_index: int
    total_steps: int
    progress_percentage: float
    metadata: Optional[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "wf_123456",
                "name": "Feature Implementation Workflow",
                "description": "Interactive workflow for implementing a new feature",
                "status": "running",
                "orchestration_mode": "semi_automatic",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T11:45:00Z",
                "current_step_index": 2,
                "total_steps": 8,
                "progress_percentage": 25.0,
                "metadata": {
                    "project_id": "proj_123",
                    "priority": "high"
                }
            }
        }

class StepResponse(BaseModel):
    """Response model for workflow step operations"""
    step_id: str
    name: str
    description: Optional[str]
    step_type: StepType
    priority: StepPriority
    status: str  # From WorkflowStep.status
    requires_approval: bool
    dependencies: List[str]
    estimated_duration_minutes: Optional[int]
    actual_duration_minutes: Optional[int]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Optional[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "step_id": "step_001",
                "name": "Generate API Endpoints",
                "description": "Create FastAPI endpoints for the new feature",
                "step_type": "code_generation",
                "priority": "high",
                "status": "completed",
                "requires_approval": True,
                "dependencies": [],
                "estimated_duration_minutes": 30,
                "actual_duration_minutes": 25,
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:55:00Z",
                "metadata": {
                    "file_count": 3,
                    "complexity": "medium"
                }
            }
        }

class ApprovalResponse(BaseModel):
    """Response model for approval operations"""
    approval_id: str
    workflow_id: str
    step_id: str
    status: UserApprovalState
    requested_at: datetime
    submitted_at: Optional[datetime]
    decision: Optional[ApprovalDecision]
    comments: Optional[str]
    requested_changes: Optional[List[str]]
    metadata: Optional[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "approval_id": "approval_123",
                "workflow_id": "wf_123456",
                "step_id": "step_001",
                "status": "approved",
                "requested_at": "2024-01-15T10:55:00Z",
                "submitted_at": "2024-01-15T11:10:00Z",
                "decision": "approved",
                "comments": "Code looks good, ready to proceed",
                "requested_changes": [],
                "metadata": {
                    "reviewer_id": "user_123",
                    "review_duration_minutes": 15
                }
            }
        }

class WorkflowListResponse(BaseModel):
    """Response model for listing workflows"""
    workflows: List[WorkflowResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool
    
    class Config:
        schema_extra = {
            "example": {
                "workflows": [
                    {
                        "workflow_id": "wf_123456",
                        "name": "Feature Implementation Workflow",
                        "status": "running",
                        "progress_percentage": 25.0
                    }
                ],
                "total_count": 15,
                "page": 1,
                "page_size": 10,
                "has_next": True,
                "has_previous": False
            }
        }

class TaskBreakdownResponse(BaseModel):
    """Response model for task breakdown operations"""
    breakdown_id: str
    task_description: str
    strategy: BreakdownStrategy
    complexity: ComplexityLevel
    category: TaskCategory
    steps: List[StepResponse]
    estimated_total_duration_minutes: int
    dependencies_count: int
    approval_gates_count: int
    created_at: datetime
    metadata: Optional[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "breakdown_id": "breakdown_123",
                "task_description": "Implement user authentication system",
                "strategy": "code_development",
                "complexity": "high",
                "category": "backend_development",
                "steps": [],
                "estimated_total_duration_minutes": 240,
                "dependencies_count": 3,
                "approval_gates_count": 5,
                "created_at": "2024-01-15T10:00:00Z",
                "metadata": {
                    "framework": "FastAPI",
                    "database": "PostgreSQL"
                }
            }
        }

class WorkflowMetricsResponse(BaseModel):
    """Response model for workflow metrics"""
    workflow_id: str
    total_steps: int
    completed_steps: int
    pending_steps: int
    failed_steps: int
    approval_requests: int
    pending_approvals: int
    approved_count: int
    rejected_count: int
    total_estimated_duration_minutes: int
    actual_duration_minutes: int
    efficiency_percentage: float
    created_at: datetime
    last_activity_at: datetime
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "wf_123456",
                "total_steps": 8,
                "completed_steps": 3,
                "pending_steps": 5,
                "failed_steps": 0,
                "approval_requests": 3,
                "pending_approvals": 1,
                "approved_count": 2,
                "rejected_count": 0,
                "total_estimated_duration_minutes": 240,
                "actual_duration_minutes": 75,
                "efficiency_percentage": 95.5,
                "created_at": "2024-01-15T10:00:00Z",
                "last_activity_at": "2024-01-15T11:45:00Z"
            }
        }

class ExecutionReportResponse(BaseModel):
    """Response model for workflow execution reports"""
    workflow_id: str
    report_id: str
    generated_at: datetime
    summary: Dict[str, Any]
    step_details: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    approval_history: List[Dict[str, Any]]
    issues_encountered: List[Dict[str, Any]]
    recommendations: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "workflow_id": "wf_123456",
                "report_id": "report_123",
                "generated_at": "2024-01-15T12:00:00Z",
                "summary": {
                    "status": "completed",
                    "success_rate": 100.0,
                    "total_duration_minutes": 180
                },
                "step_details": [],
                "performance_metrics": {
                    "average_step_duration": 22.5,
                    "approval_response_time": 8.3
                },
                "approval_history": [],
                "issues_encountered": [],
                "recommendations": [
                    "Consider automating step validation",
                    "Add more detailed error handling"
                ]
            }
        }

# Error Response Models
class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid workflow configuration",
                "details": {
                    "field": "orchestration_mode",
                    "issue": "Invalid value provided"
                },
                "timestamp": "2024-01-15T12:00:00Z"
            }
        }

class SuccessResponse(BaseModel):
    """Standard success response model"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {
                    "operation_id": "op_123",
                    "affected_items": 1
                },
                "timestamp": "2024-01-15T12:00:00Z"
            }
        }

# WebSocket Message Models
class WebSocketMessage(BaseModel):
    """Base model for WebSocket messages"""
    type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Optional[Dict[str, Any]] = None

class WorkflowUpdateMessage(WebSocketMessage):
    """WebSocket message for workflow updates"""
    workflow_id: str
    update_type: str  # created, started, paused, resumed, completed, error
    
class StepUpdateMessage(WebSocketMessage):
    """WebSocket message for step updates"""
    workflow_id: str
    step_id: str
    update_type: str  # started, completed, failed, approval_requested
    
class ApprovalUpdateMessage(WebSocketMessage):
    """WebSocket message for approval updates"""
    workflow_id: str
    step_id: str
    approval_id: str
    update_type: str  # requested, submitted, approved, rejected
    requires_action: bool = False