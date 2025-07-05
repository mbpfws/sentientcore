"""User Approval Gates

Implements user approval mechanisms for interactive workflows,
including approval requests, response handling, and gate management.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uuid
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from ..state.interactive_workflow_models import (
    UserInteractionRequest, UserInteractionResponse, InteractionType,
    UserApprovalState, StepType, StepPriority
)
from ..state.interactive_state_manager import InteractiveStateManager

class ApprovalGateType(Enum):
    """Types of approval gates."""
    SIMPLE_APPROVAL = "simple_approval"  # Yes/No approval
    DETAILED_REVIEW = "detailed_review"  # Detailed review with comments
    CONDITIONAL_APPROVAL = "conditional_approval"  # Approval with conditions
    MULTI_STEP_APPROVAL = "multi_step_approval"  # Multiple approval steps
    TIMEOUT_APPROVAL = "timeout_approval"  # Auto-approve after timeout
    ESCALATION_APPROVAL = "escalation_approval"  # Escalate if not approved

class ApprovalDecision(Enum):
    """Approval decision types."""
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    ESCALATE = "escalate"
    DEFER = "defer"

@dataclass
class ApprovalCriteria:
    """Criteria for approval decisions."""
    required_approvers: List[str] = field(default_factory=list)
    minimum_approvals: int = 1
    allow_self_approval: bool = True
    require_unanimous: bool = False
    timeout_seconds: Optional[int] = None
    auto_approve_on_timeout: bool = False
    escalation_users: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ApprovalContext:
    """Context information for approval requests."""
    workflow_id: str
    step_id: str
    step_title: str
    step_description: str
    step_type: StepType
    priority: StepPriority
    estimated_impact: str
    risk_level: str
    dependencies: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ApprovalGate(ABC):
    """Abstract base class for approval gates."""
    
    def __init__(self, gate_id: str, gate_type: ApprovalGateType, 
                 criteria: ApprovalCriteria, state_manager: InteractiveStateManager):
        self.gate_id = gate_id
        self.gate_type = gate_type
        self.criteria = criteria
        self.state_manager = state_manager
        self.created_at = datetime.now()
        
        # Tracking
        self.approval_requests: Dict[str, UserInteractionRequest] = {}
        self.approval_responses: Dict[str, List[UserInteractionResponse]] = {}
        self.gate_metrics = {
            "total_requests": 0,
            "approved_requests": 0,
            "rejected_requests": 0,
            "timeout_requests": 0,
            "average_response_time": 0.0
        }
    
    @abstractmethod
    async def request_approval(self, context: ApprovalContext, 
                              user_id: str, session_id: str) -> str:
        """Request approval through this gate."""
        pass
    
    @abstractmethod
    async def process_response(self, response: UserInteractionResponse) -> ApprovalDecision:
        """Process an approval response."""
        pass
    
    async def is_approved(self, request_id: str) -> bool:
        """Check if a request is approved."""
        responses = self.approval_responses.get(request_id, [])
        if not responses:
            return False
        
        approved_responses = [r for r in responses if r.approval_state == UserApprovalState.APPROVED]
        
        if self.criteria.require_unanimous:
            return len(approved_responses) == len(self.criteria.required_approvers)
        else:
            return len(approved_responses) >= self.criteria.minimum_approvals
    
    async def is_rejected(self, request_id: str) -> bool:
        """Check if a request is rejected."""
        responses = self.approval_responses.get(request_id, [])
        return any(r.approval_state == UserApprovalState.REJECTED for r in responses)
    
    def update_metrics(self, response: UserInteractionResponse):
        """Update gate metrics."""
        self.gate_metrics["total_requests"] += 1
        
        if response.approval_state == UserApprovalState.APPROVED:
            self.gate_metrics["approved_requests"] += 1
        elif response.approval_state == UserApprovalState.REJECTED:
            self.gate_metrics["rejected_requests"] += 1
        elif response.approval_state == UserApprovalState.TIMEOUT:
            self.gate_metrics["timeout_requests"] += 1
        
        # Update average response time
        request = self.approval_requests.get(response.request_id)
        if request:
            response_time = (response.responded_at - request.created_at).total_seconds()
            current_avg = self.gate_metrics["average_response_time"]
            total_requests = self.gate_metrics["total_requests"]
            self.gate_metrics["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )

class SimpleApprovalGate(ApprovalGate):
    """Simple yes/no approval gate."""
    
    def __init__(self, gate_id: str, criteria: ApprovalCriteria, 
                 state_manager: InteractiveStateManager):
        super().__init__(gate_id, ApprovalGateType.SIMPLE_APPROVAL, criteria, state_manager)
    
    async def request_approval(self, context: ApprovalContext, 
                              user_id: str, session_id: str) -> str:
        """Request simple approval."""
        request_id = f"approval_{self.gate_id}_{uuid.uuid4().hex[:8]}"
        
        # Create interaction request
        interaction_request = UserInteractionRequest(
            id=request_id,
            interaction_type=InteractionType.APPROVAL_REQUEST,
            title=f"Approval Required: {context.step_title}",
            description=f"Please review and approve the following step:\n\n"
                       f"**Step:** {context.step_title}\n"
                       f"**Description:** {context.step_description}\n"
                       f"**Type:** {context.step_type.value}\n"
                       f"**Priority:** {context.priority.value}\n"
                       f"**Estimated Impact:** {context.estimated_impact}\n"
                       f"**Risk Level:** {context.risk_level}",
            required_response_type="boolean",
            options=["Approve", "Reject"],
            default_value="Approve",
            timeout_seconds=self.criteria.timeout_seconds,
            metadata={
                "gate_id": self.gate_id,
                "gate_type": self.gate_type.value,
                "workflow_id": context.workflow_id,
                "step_id": context.step_id,
                "context": context.__dict__
            },
            user_id=user_id,
            session_id=session_id
        )
        
        # Store request
        self.approval_requests[request_id] = interaction_request
        self.approval_responses[request_id] = []
        
        # Submit to state manager
        await self.state_manager.request_user_approval(
            context.workflow_id, context.step_id, interaction_request
        )
        
        return request_id
    
    async def process_response(self, response: UserInteractionResponse) -> ApprovalDecision:
        """Process approval response."""
        # Store response
        if response.request_id in self.approval_responses:
            self.approval_responses[response.request_id].append(response)
        
        # Update metrics
        self.update_metrics(response)
        
        # Determine decision
        if response.approval_state == UserApprovalState.APPROVED:
            return ApprovalDecision.APPROVE
        elif response.approval_state == UserApprovalState.REJECTED:
            return ApprovalDecision.REJECT
        elif response.approval_state == UserApprovalState.TIMEOUT:
            if self.criteria.auto_approve_on_timeout:
                return ApprovalDecision.APPROVE
            else:
                return ApprovalDecision.REJECT
        else:
            return ApprovalDecision.DEFER

class DetailedReviewGate(ApprovalGate):
    """Detailed review gate with comments and conditions."""
    
    def __init__(self, gate_id: str, criteria: ApprovalCriteria, 
                 state_manager: InteractiveStateManager):
        super().__init__(gate_id, ApprovalGateType.DETAILED_REVIEW, criteria, state_manager)
    
    async def request_approval(self, context: ApprovalContext, 
                              user_id: str, session_id: str) -> str:
        """Request detailed review."""
        request_id = f"review_{self.gate_id}_{uuid.uuid4().hex[:8]}"
        
        # Create detailed interaction request
        interaction_request = UserInteractionRequest(
            id=request_id,
            interaction_type=InteractionType.DETAILED_REVIEW,
            title=f"Detailed Review Required: {context.step_title}",
            description=f"Please provide a detailed review of the following step:\n\n"
                       f"**Step:** {context.step_title}\n"
                       f"**Description:** {context.step_description}\n"
                       f"**Type:** {context.step_type.value}\n"
                       f"**Priority:** {context.priority.value}\n"
                       f"**Estimated Impact:** {context.estimated_impact}\n"
                       f"**Risk Level:** {context.risk_level}\n\n"
                       f"**Dependencies:** {', '.join(context.dependencies) if context.dependencies else 'None'}\n"
                       f"**Artifacts:** {', '.join(context.artifacts) if context.artifacts else 'None'}\n\n"
                       f"Please provide your review comments and decision.",
            required_response_type="object",
            options=["Approve", "Request Changes", "Reject"],
            validation_schema={
                "type": "object",
                "properties": {
                    "decision": {"type": "string", "enum": ["approve", "request_changes", "reject"]},
                    "comments": {"type": "string", "minLength": 10},
                    "conditions": {"type": "array", "items": {"type": "string"}},
                    "risk_assessment": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["decision", "comments"]
            },
            timeout_seconds=self.criteria.timeout_seconds,
            metadata={
                "gate_id": self.gate_id,
                "gate_type": self.gate_type.value,
                "workflow_id": context.workflow_id,
                "step_id": context.step_id,
                "context": context.__dict__
            },
            user_id=user_id,
            session_id=session_id
        )
        
        # Store request
        self.approval_requests[request_id] = interaction_request
        self.approval_responses[request_id] = []
        
        # Submit to state manager
        await self.state_manager.request_user_approval(
            context.workflow_id, context.step_id, interaction_request
        )
        
        return request_id
    
    async def process_response(self, response: UserInteractionResponse) -> ApprovalDecision:
        """Process detailed review response."""
        # Store response
        if response.request_id in self.approval_responses:
            self.approval_responses[response.request_id].append(response)
        
        # Update metrics
        self.update_metrics(response)
        
        # Parse response value
        if isinstance(response.response_value, dict):
            decision = response.response_value.get("decision", "defer")
            
            if decision == "approve":
                return ApprovalDecision.APPROVE
            elif decision == "request_changes":
                return ApprovalDecision.REQUEST_CHANGES
            elif decision == "reject":
                return ApprovalDecision.REJECT
        
        # Fallback to approval state
        if response.approval_state == UserApprovalState.APPROVED:
            return ApprovalDecision.APPROVE
        elif response.approval_state == UserApprovalState.REJECTED:
            return ApprovalDecision.REJECT
        else:
            return ApprovalDecision.DEFER

class MultiStepApprovalGate(ApprovalGate):
    """Multi-step approval gate requiring multiple approvers."""
    
    def __init__(self, gate_id: str, criteria: ApprovalCriteria, 
                 state_manager: InteractiveStateManager):
        super().__init__(gate_id, ApprovalGateType.MULTI_STEP_APPROVAL, criteria, state_manager)
        self.pending_approvers: Dict[str, List[str]] = {}  # request_id -> list of pending approvers
    
    async def request_approval(self, context: ApprovalContext, 
                              user_id: str, session_id: str) -> str:
        """Request multi-step approval."""
        request_id = f"multi_{self.gate_id}_{uuid.uuid4().hex[:8]}"
        
        # Initialize pending approvers
        self.pending_approvers[request_id] = self.criteria.required_approvers.copy()
        
        # Create interaction requests for each approver
        for approver_id in self.criteria.required_approvers:
            approver_request_id = f"{request_id}_{approver_id}"
            
            interaction_request = UserInteractionRequest(
                id=approver_request_id,
                interaction_type=InteractionType.APPROVAL_REQUEST,
                title=f"Multi-Step Approval Required: {context.step_title}",
                description=f"You are one of {len(self.criteria.required_approvers)} required approvers for this step:\n\n"
                           f"**Step:** {context.step_title}\n"
                           f"**Description:** {context.step_description}\n"
                           f"**Type:** {context.step_type.value}\n"
                           f"**Priority:** {context.priority.value}\n"
                           f"**Estimated Impact:** {context.estimated_impact}\n"
                           f"**Risk Level:** {context.risk_level}\n\n"
                           f"**Required Approvals:** {self.criteria.minimum_approvals}/{len(self.criteria.required_approvers)}\n"
                           f"**Unanimous Required:** {'Yes' if self.criteria.require_unanimous else 'No'}",
                required_response_type="boolean",
                options=["Approve", "Reject"],
                timeout_seconds=self.criteria.timeout_seconds,
                metadata={
                    "gate_id": self.gate_id,
                    "gate_type": self.gate_type.value,
                    "workflow_id": context.workflow_id,
                    "step_id": context.step_id,
                    "parent_request_id": request_id,
                    "approver_id": approver_id,
                    "context": context.__dict__
                },
                user_id=approver_id,
                session_id=session_id
            )
            
            # Store individual request
            self.approval_requests[approver_request_id] = interaction_request
            
            # Submit to state manager
            await self.state_manager.request_user_approval(
                context.workflow_id, context.step_id, interaction_request
            )
        
        # Initialize response tracking
        self.approval_responses[request_id] = []
        
        return request_id
    
    async def process_response(self, response: UserInteractionResponse) -> ApprovalDecision:
        """Process multi-step approval response."""
        # Find parent request ID
        parent_request_id = None
        for req_id, req in self.approval_requests.items():
            if req.id == response.request_id:
                parent_request_id = req.metadata.get("parent_request_id")
                break
        
        if not parent_request_id:
            return ApprovalDecision.DEFER
        
        # Store response
        if parent_request_id in self.approval_responses:
            self.approval_responses[parent_request_id].append(response)
        
        # Update metrics
        self.update_metrics(response)
        
        # Remove approver from pending list
        approver_id = None
        for req_id, req in self.approval_requests.items():
            if req.id == response.request_id:
                approver_id = req.metadata.get("approver_id")
                break
        
        if approver_id and parent_request_id in self.pending_approvers:
            if approver_id in self.pending_approvers[parent_request_id]:
                self.pending_approvers[parent_request_id].remove(approver_id)
        
        # Check if we have enough responses
        responses = self.approval_responses[parent_request_id]
        approved_responses = [r for r in responses if r.approval_state == UserApprovalState.APPROVED]
        rejected_responses = [r for r in responses if r.approval_state == UserApprovalState.REJECTED]
        
        # Check for rejection (any rejection fails the approval)
        if rejected_responses:
            return ApprovalDecision.REJECT
        
        # Check for approval
        if self.criteria.require_unanimous:
            if len(approved_responses) == len(self.criteria.required_approvers):
                return ApprovalDecision.APPROVE
        else:
            if len(approved_responses) >= self.criteria.minimum_approvals:
                return ApprovalDecision.APPROVE
        
        # Still waiting for more approvals
        return ApprovalDecision.DEFER

class ApprovalGateManager:
    """Manages multiple approval gates and routing."""
    
    def __init__(self, state_manager: InteractiveStateManager):
        self.state_manager = state_manager
        self.gates: Dict[str, ApprovalGate] = {}
        self.gate_routing: Dict[str, str] = {}  # step_type -> gate_id
        self.default_gate_id: Optional[str] = None
        
        # Metrics
        self.manager_metrics = {
            "total_approvals_requested": 0,
            "total_approvals_processed": 0,
            "average_approval_time": 0.0
        }
    
    def register_gate(self, gate: ApprovalGate, is_default: bool = False):
        """Register an approval gate."""
        self.gates[gate.gate_id] = gate
        
        if is_default:
            self.default_gate_id = gate.gate_id
    
    def set_gate_routing(self, step_type: str, gate_id: str):
        """Set routing for a step type to a specific gate."""
        if gate_id in self.gates:
            self.gate_routing[step_type] = gate_id
    
    def get_gate_for_step(self, step_type: str) -> Optional[ApprovalGate]:
        """Get the appropriate gate for a step type."""
        gate_id = self.gate_routing.get(step_type, self.default_gate_id)
        return self.gates.get(gate_id) if gate_id else None
    
    async def request_approval(self, context: ApprovalContext, 
                              user_id: str, session_id: str,
                              gate_id: Optional[str] = None) -> str:
        """Request approval through appropriate gate."""
        # Get gate
        if gate_id:
            gate = self.gates.get(gate_id)
        else:
            gate = self.get_gate_for_step(context.step_type.value)
        
        if not gate:
            raise ValueError(f"No approval gate found for step type: {context.step_type.value}")
        
        # Update metrics
        self.manager_metrics["total_approvals_requested"] += 1
        
        # Request approval
        return await gate.request_approval(context, user_id, session_id)
    
    async def process_response(self, response: UserInteractionResponse) -> ApprovalDecision:
        """Process approval response through appropriate gate."""
        # Find gate that owns this request
        gate = None
        for g in self.gates.values():
            if response.request_id in g.approval_requests:
                gate = g
                break
        
        if not gate:
            raise ValueError(f"No gate found for request: {response.request_id}")
        
        # Update metrics
        self.manager_metrics["total_approvals_processed"] += 1
        
        # Process response
        decision = await gate.process_response(response)
        
        # Update average approval time
        request = gate.approval_requests.get(response.request_id)
        if request:
            approval_time = (response.responded_at - request.created_at).total_seconds()
            current_avg = self.manager_metrics["average_approval_time"]
            total_processed = self.manager_metrics["total_approvals_processed"]
            self.manager_metrics["average_approval_time"] = (
                (current_avg * (total_processed - 1) + approval_time) / total_processed
            )
        
        return decision
    
    def get_gate_metrics(self, gate_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a specific gate or all gates."""
        if gate_id:
            gate = self.gates.get(gate_id)
            return gate.gate_metrics if gate else {}
        else:
            return {
                "manager_metrics": self.manager_metrics,
                "gate_metrics": {gid: gate.gate_metrics for gid, gate in self.gates.items()}
            }
    
    def list_gates(self) -> List[Dict[str, Any]]:
        """List all registered gates."""
        return [
            {
                "gate_id": gate.gate_id,
                "gate_type": gate.gate_type.value,
                "created_at": gate.created_at.isoformat(),
                "criteria": {
                    "required_approvers": gate.criteria.required_approvers,
                    "minimum_approvals": gate.criteria.minimum_approvals,
                    "require_unanimous": gate.criteria.require_unanimous,
                    "timeout_seconds": gate.criteria.timeout_seconds
                },
                "metrics": gate.gate_metrics
            }
            for gate in self.gates.values()
        ]

# Factory functions
def create_simple_approval_gate(gate_id: str, criteria: ApprovalCriteria, 
                               state_manager: InteractiveStateManager) -> SimpleApprovalGate:
    """Create a simple approval gate."""
    return SimpleApprovalGate(gate_id, criteria, state_manager)

def create_detailed_review_gate(gate_id: str, criteria: ApprovalCriteria, 
                               state_manager: InteractiveStateManager) -> DetailedReviewGate:
    """Create a detailed review gate."""
    return DetailedReviewGate(gate_id, criteria, state_manager)

def create_multi_step_approval_gate(gate_id: str, criteria: ApprovalCriteria, 
                                   state_manager: InteractiveStateManager) -> MultiStepApprovalGate:
    """Create a multi-step approval gate."""
    return MultiStepApprovalGate(gate_id, criteria, state_manager)

def create_default_approval_manager(state_manager: InteractiveStateManager) -> ApprovalGateManager:
    """Create a default approval gate manager with standard gates."""
    manager = ApprovalGateManager(state_manager)
    
    # Create default gates
    simple_criteria = ApprovalCriteria(
        minimum_approvals=1,
        timeout_seconds=300,  # 5 minutes
        auto_approve_on_timeout=False
    )
    simple_gate = create_simple_approval_gate("simple_default", simple_criteria, state_manager)
    manager.register_gate(simple_gate, is_default=True)
    
    # Detailed review gate for high-risk steps
    detailed_criteria = ApprovalCriteria(
        minimum_approvals=1,
        timeout_seconds=600,  # 10 minutes
        auto_approve_on_timeout=False
    )
    detailed_gate = create_detailed_review_gate("detailed_review", detailed_criteria, state_manager)
    manager.register_gate(detailed_gate)
    
    # Multi-step gate for critical operations
    multi_criteria = ApprovalCriteria(
        required_approvers=["admin", "lead"],
        minimum_approvals=2,
        require_unanimous=True,
        timeout_seconds=1800,  # 30 minutes
        auto_approve_on_timeout=False
    )
    multi_gate = create_multi_step_approval_gate("multi_critical", multi_criteria, state_manager)
    manager.register_gate(multi_gate)
    
    # Set routing
    manager.set_gate_routing("code_generation", "simple_default")
    manager.set_gate_routing("file_modification", "simple_default")
    manager.set_gate_routing("system_configuration", "detailed_review")
    manager.set_gate_routing("deployment", "multi_critical")
    manager.set_gate_routing("database_migration", "multi_critical")
    
    return manager