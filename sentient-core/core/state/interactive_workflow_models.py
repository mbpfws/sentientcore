"""Interactive Workflow State Management Models

This module extends the existing state management system with models and enums
specifically designed for interactive step-by-step workflows that require user
approval gates, interaction checkpoints, and step-by-step progression tracking.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
from pydantic import BaseModel, Field
import uuid
from dataclasses import dataclass

# --- Interactive Workflow Enums ---

class InteractiveWorkflowStatus(str, Enum):
    """Status of an interactive workflow."""
    INITIALIZED = "initialized"
    WAITING_USER_INPUT = "waiting_user_input"
    USER_REVIEWING = "user_reviewing"
    STEP_APPROVED = "step_approved"
    STEP_REJECTED = "step_rejected"
    STEP_MODIFIED = "step_modified"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

class UserApprovalState(str, Enum):
    """State of user approval for workflow steps."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_MODIFICATION = "needs_modification"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"

class StepType(str, Enum):
    """Types of workflow steps."""
    INFORMATION_GATHERING = "information_gathering"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VALIDATION = "validation"
    REVIEW = "review"
    DEPLOYMENT = "deployment"
    CLEANUP = "cleanup"
    USER_INPUT = "user_input"
    DECISION_POINT = "decision_point"

class InteractionType(str, Enum):
    """Types of user interactions."""
    APPROVAL_GATE = "approval_gate"
    INPUT_REQUEST = "input_request"
    CHOICE_SELECTION = "choice_selection"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    REVIEW_CHECKPOINT = "review_checkpoint"
    CONFIRMATION = "confirmation"
    FEEDBACK = "feedback"

class StepPriority(str, Enum):
    """Priority levels for workflow steps."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# --- Interactive Workflow Models ---

class UserInteractionRequest(BaseModel):
    """Represents a request for user interaction."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interaction_type: InteractionType
    title: str
    description: str
    prompt_message: str
    
    # Options for user selection
    options: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Default values or suggestions
    default_value: Optional[Any] = None
    suggested_values: List[Any] = Field(default_factory=list)
    
    # Validation rules
    required: bool = True
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    timeout_seconds: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Context
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class UserInteractionResponse(BaseModel):
    """Represents a user's response to an interaction request."""
    request_id: str
    response_value: Any
    approval_state: UserApprovalState
    user_comments: Optional[str] = None
    modifications_requested: List[Dict[str, Any]] = Field(default_factory=list)
    responded_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class WorkflowStep(BaseModel):
    """Represents a single step in an interactive workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    step_type: StepType
    priority: StepPriority = StepPriority.MEDIUM
    
    # Step ordering and dependencies
    sequence_number: int
    dependencies: List[str] = Field(default_factory=list)  # Step IDs
    
    # Execution details
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    
    # User interaction
    requires_approval: bool = False
    interaction_request: Optional[UserInteractionRequest] = None
    user_response: Optional[UserInteractionResponse] = None
    approval_state: UserApprovalState = UserApprovalState.PENDING
    
    # Status and results
    status: InteractiveWorkflowStatus = InteractiveWorkflowStatus.INITIALIZED
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Artifacts and outputs
    artifacts: List[str] = Field(default_factory=list)
    output_files: List[str] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent assignment
    assigned_agent: Optional[str] = None
    agent_type: Optional[str] = None

class InteractiveWorkflow(BaseModel):
    """Represents a complete interactive workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    
    # Workflow structure
    steps: List[WorkflowStep] = Field(default_factory=list)
    current_step_id: Optional[str] = None
    
    # Overall status
    status: InteractiveWorkflowStatus = InteractiveWorkflowStatus.INITIALIZED
    
    # User interaction tracking
    pending_interactions: List[str] = Field(default_factory=list)  # Request IDs
    completed_interactions: List[str] = Field(default_factory=list)
    
    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    approved_steps: int = 0
    rejected_steps: int = 0
    
    # Timing
    estimated_total_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    
    # Session and user info
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_interaction_at: Optional[datetime] = None
    
    # Configuration
    auto_approve_low_risk: bool = False
    require_approval_for_all: bool = False
    timeout_behavior: str = "pause"  # "pause", "auto_approve", "cancel"
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Results and artifacts
    final_result: Optional[Dict[str, Any]] = None
    workflow_artifacts: List[str] = Field(default_factory=list)
    
    def get_current_step(self) -> Optional[WorkflowStep]:
        """Get the current step being executed."""
        if not self.current_step_id:
            return None
        return next((step for step in self.steps if step.id == self.current_step_id), None)
    
    def get_next_step(self) -> Optional[WorkflowStep]:
        """Get the next step to be executed."""
        current_step = self.get_current_step()
        if not current_step:
            # Return first step if no current step
            return self.steps[0] if self.steps else None
        
        # Find next step by sequence number
        next_sequence = current_step.sequence_number + 1
        return next((step for step in self.steps if step.sequence_number == next_sequence), None)
    
    def get_pending_approvals(self) -> List[WorkflowStep]:
        """Get all steps waiting for user approval."""
        return [
            step for step in self.steps 
            if step.requires_approval and step.approval_state == UserApprovalState.PENDING
        ]
    
    def calculate_progress_percentage(self) -> float:
        """Calculate workflow completion percentage."""
        if not self.steps:
            return 0.0
        return (self.completed_steps / len(self.steps)) * 100.0

class InteractiveWorkflowState(BaseModel):
    """Extended state model for interactive workflows."""
    # Active workflows
    active_workflows: Dict[str, InteractiveWorkflow] = Field(default_factory=dict)
    
    # Current workflow being executed
    current_workflow_id: Optional[str] = None
    
    # Pending user interactions across all workflows
    pending_interactions: Dict[str, UserInteractionRequest] = Field(default_factory=dict)
    
    # Interaction history
    interaction_history: List[UserInteractionResponse] = Field(default_factory=list)
    
    # User session info
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # UI state
    ui_mode: str = "step_by_step"  # "step_by_step", "overview", "detailed"
    show_technical_details: bool = False
    auto_scroll_to_current: bool = True
    
    # Notification settings
    notification_preferences: Dict[str, bool] = Field(default_factory=lambda: {
        "step_completion": True,
        "approval_required": True,
        "workflow_completion": True,
        "errors": True
    })
    
    # Performance tracking
    workflow_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    def get_current_workflow(self) -> Optional[InteractiveWorkflow]:
        """Get the currently active workflow."""
        if not self.current_workflow_id:
            return None
        return self.active_workflows.get(self.current_workflow_id)
    
    def get_all_pending_interactions(self) -> List[UserInteractionRequest]:
        """Get all pending interactions across all workflows."""
        return list(self.pending_interactions.values())
    
    def add_workflow(self, workflow: InteractiveWorkflow) -> None:
        """Add a new workflow to the state."""
        self.active_workflows[workflow.id] = workflow
        if not self.current_workflow_id:
            self.current_workflow_id = workflow.id

# --- State Change Events for Interactive Workflows ---

class InteractiveWorkflowEvent(BaseModel):
    """Base class for interactive workflow events."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    workflow_id: str
    event_type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class StepStartedEvent(InteractiveWorkflowEvent):
    """Event fired when a workflow step starts."""
    event_type: str = "step_started"
    step_id: str

class StepCompletedEvent(InteractiveWorkflowEvent):
    """Event fired when a workflow step completes."""
    event_type: str = "step_completed"
    step_id: str
    success: bool

class ApprovalRequestedEvent(InteractiveWorkflowEvent):
    """Event fired when user approval is requested."""
    event_type: str = "approval_requested"
    interaction_request_id: str

class ApprovalReceivedEvent(InteractiveWorkflowEvent):
    """Event fired when user approval is received."""
    event_type: str = "approval_received"
    interaction_response_id: str
    approved: bool

class WorkflowPausedEvent(InteractiveWorkflowEvent):
    """Event fired when a workflow is paused."""
    event_type: str = "workflow_paused"
    reason: str

class WorkflowResumedEvent(InteractiveWorkflowEvent):
    """Event fired when a workflow is resumed."""
    event_type: str = "workflow_resumed"

# --- Utility Functions ---

def create_approval_gate(title: str, description: str, step: WorkflowStep) -> UserInteractionRequest:
    """Create a standard approval gate interaction request."""
    return UserInteractionRequest(
        interaction_type=InteractionType.APPROVAL_GATE,
        title=title,
        description=description,
        prompt_message=f"Please review and approve the following step: {step.title}",
        options=[
            {"value": "approve", "label": "Approve", "description": "Proceed with this step"},
            {"value": "reject", "label": "Reject", "description": "Skip this step"},
            {"value": "modify", "label": "Request Changes", "description": "Request modifications before proceeding"}
        ],
        context={"step_id": step.id, "step_type": step.step_type.value}
    )

def create_input_request(title: str, description: str, input_type: str = "text") -> UserInteractionRequest:
    """Create a user input request."""
    return UserInteractionRequest(
        interaction_type=InteractionType.INPUT_REQUEST,
        title=title,
        description=description,
        prompt_message=f"Please provide input for: {title}",
        validation_rules={"type": input_type},
        required=True
    )

def create_choice_selection(title: str, description: str, choices: List[Dict[str, Any]]) -> UserInteractionRequest:
    """Create a choice selection interaction request."""
    return UserInteractionRequest(
        interaction_type=InteractionType.CHOICE_SELECTION,
        title=title,
        description=description,
        prompt_message=f"Please select an option for: {title}",
        options=choices,
        required=True
    )