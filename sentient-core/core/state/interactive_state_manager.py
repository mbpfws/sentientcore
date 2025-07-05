"""Interactive State Manager

Extends the existing EnhancedStateManager to support interactive workflow states,
user approval gates, and step-by-step progression tracking.
"""

from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import json
from pathlib import Path

from .enhanced_state_manager import (
    EnhancedStateManager, StateChangeType, StatePersistenceMode, 
    StateValidationLevel, StateChange, StateSnapshot, StateEventBus
)
from .interactive_workflow_models import (
    InteractiveWorkflowState, InteractiveWorkflow, WorkflowStep,
    UserInteractionRequest, UserInteractionResponse, InteractiveWorkflowStatus,
    UserApprovalState, InteractiveWorkflowEvent, StepStartedEvent,
    StepCompletedEvent, ApprovalRequestedEvent, ApprovalReceivedEvent,
    WorkflowPausedEvent, WorkflowResumedEvent
)
from ..models import AppState

class InteractiveStateChangeType:
    """Extended state change types for interactive workflows."""
    # Base state change types (from StateChangeType)
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    ROLLBACK = "rollback"
    
    # Interactive workflow specific types
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_PAUSED = "workflow_paused"
    WORKFLOW_RESUMED = "workflow_resumed"
    WORKFLOW_COMPLETED = "workflow_completed"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RECEIVED = "approval_received"
    USER_INTERACTION = "user_interaction"

class InteractiveWorkflowEventBus(StateEventBus):
    """Enhanced event bus for interactive workflow events."""
    
    def __init__(self):
        super().__init__()
        self.workflow_subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.interaction_subscribers: List[Callable] = []
    
    async def emit_workflow_event(self, event: InteractiveWorkflowEvent):
        """Emit a workflow-specific event."""
        # Emit to general subscribers
        await self.emit(StateChange(
            id=event.id,
            timestamp=event.timestamp,
            change_type=event.event_type,
            path=f"workflows.{event.workflow_id}",
            old_value=None,
            new_value=event.data,
            metadata={"event_type": event.event_type, "workflow_id": event.workflow_id},
            user_id=event.user_id,
            session_id=event.session_id
        ))
        
        # Emit to workflow-specific subscribers
        for callback in self.workflow_subscribers[event.workflow_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                print(f"Error in workflow event callback: {e}")
    
    def subscribe_to_workflow(self, workflow_id: str, callback: Callable):
        """Subscribe to events for a specific workflow."""
        self.workflow_subscribers[workflow_id].append(callback)
    
    def subscribe_to_interactions(self, callback: Callable):
        """Subscribe to all user interaction events."""
        self.interaction_subscribers.append(callback)
    
    async def emit_interaction_event(self, event: InteractiveWorkflowEvent):
        """Emit an interaction-specific event."""
        await self.emit_workflow_event(event)
        
        # Emit to interaction subscribers
        for callback in self.interaction_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                print(f"Error in interaction event callback: {e}")

class InteractiveStateManager(EnhancedStateManager):
    """Enhanced state manager with interactive workflow support."""
    
    def __init__(self, 
                 persistence_mode: StatePersistenceMode = StatePersistenceMode.HYBRID,
                 validation_level: StateValidationLevel = StateValidationLevel.BASIC,
                 storage_path: Optional[Path] = None,
                 max_history_size: int = 1000):
        
        super().__init__(persistence_mode, validation_level, storage_path, max_history_size)
        
        # Interactive workflow state
        self.interactive_state = InteractiveWorkflowState()
        
        # Enhanced event bus
        self.workflow_event_bus = InteractiveWorkflowEventBus()
        
        # Workflow execution tracking
        self.active_workflow_tasks: Dict[str, asyncio.Task] = {}
        self.workflow_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # User interaction tracking
        self.pending_interaction_timeouts: Dict[str, asyncio.Task] = {}
        
        # Metrics
        self.workflow_metrics = defaultdict(list)
        
        print("Interactive State Manager initialized")
    
    async def get_interactive_state(self) -> InteractiveWorkflowState:
        """Get the current interactive workflow state."""
        async with self.state_lock:
            return self.interactive_state.copy(deep=True)
    
    async def create_workflow(self, workflow: InteractiveWorkflow) -> str:
        """Create a new interactive workflow."""
        async with self.state_lock:
            # Add workflow to state
            self.interactive_state.add_workflow(workflow)
            
            # Update total steps
            workflow.total_steps = len(workflow.steps)
            
            # Create state change record
            change = StateChange(
                id=f"workflow_create_{workflow.id}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                change_type=InteractiveStateChangeType.WORKFLOW_CREATED,
                path=f"interactive_state.active_workflows.{workflow.id}",
                old_value=None,
                new_value=workflow.dict(),
                metadata={"workflow_id": workflow.id, "workflow_name": workflow.name},
                user_id=workflow.user_id,
                session_id=workflow.session_id
            )
            
            self.change_history.append(change)
            await self.event_bus.emit(change)
            
            # Emit workflow event
            event = InteractiveWorkflowEvent(
                workflow_id=workflow.id,
                event_type="workflow_created",
                data={"workflow_name": workflow.name, "total_steps": workflow.total_steps},
                user_id=workflow.user_id,
                session_id=workflow.session_id
            )
            await self.workflow_event_bus.emit_workflow_event(event)
            
            return workflow.id
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start executing an interactive workflow."""
        async with self.workflow_locks[workflow_id]:
            workflow = self.interactive_state.active_workflows.get(workflow_id)
            if not workflow:
                return False
            
            # Update workflow status
            workflow.status = InteractiveWorkflowStatus.IN_PROGRESS
            workflow.started_at = datetime.now()
            
            # Set current step to first step
            if workflow.steps:
                first_step = min(workflow.steps, key=lambda s: s.sequence_number)
                workflow.current_step_id = first_step.id
            
            # Record state change
            await self._record_workflow_change(
                workflow_id, InteractiveStateChangeType.WORKFLOW_STARTED,
                {"started_at": workflow.started_at.isoformat()}
            )
            
            # Emit event
            event = InteractiveWorkflowEvent(
                workflow_id=workflow_id,
                event_type="workflow_started",
                data={"started_at": workflow.started_at.isoformat()},
                user_id=workflow.user_id,
                session_id=workflow.session_id
            )
            await self.workflow_event_bus.emit_workflow_event(event)
            
            # Start workflow execution task
            task = asyncio.create_task(self._execute_workflow(workflow_id))
            self.active_workflow_tasks[workflow_id] = task
            
            return True
    
    async def pause_workflow(self, workflow_id: str, reason: str = "User requested") -> bool:
        """Pause a running workflow."""
        async with self.workflow_locks[workflow_id]:
            workflow = self.interactive_state.active_workflows.get(workflow_id)
            if not workflow or workflow.status != InteractiveWorkflowStatus.IN_PROGRESS:
                return False
            
            # Update status
            workflow.status = InteractiveWorkflowStatus.PAUSED
            
            # Cancel execution task
            if workflow_id in self.active_workflow_tasks:
                self.active_workflow_tasks[workflow_id].cancel()
                del self.active_workflow_tasks[workflow_id]
            
            # Record change
            await self._record_workflow_change(
                workflow_id, InteractiveStateChangeType.WORKFLOW_PAUSED,
                {"reason": reason}
            )
            
            # Emit event
            event = WorkflowPausedEvent(
                workflow_id=workflow_id,
                reason=reason,
                user_id=workflow.user_id,
                session_id=workflow.session_id
            )
            await self.workflow_event_bus.emit_workflow_event(event)
            
            return True
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        async with self.workflow_locks[workflow_id]:
            workflow = self.interactive_state.active_workflows.get(workflow_id)
            if not workflow or workflow.status != InteractiveWorkflowStatus.PAUSED:
                return False
            
            # Update status
            workflow.status = InteractiveWorkflowStatus.IN_PROGRESS
            
            # Record change
            await self._record_workflow_change(
                workflow_id, InteractiveStateChangeType.WORKFLOW_RESUMED, {}
            )
            
            # Emit event
            event = WorkflowResumedEvent(
                workflow_id=workflow_id,
                user_id=workflow.user_id,
                session_id=workflow.session_id
            )
            await self.workflow_event_bus.emit_workflow_event(event)
            
            # Restart execution task
            task = asyncio.create_task(self._execute_workflow(workflow_id))
            self.active_workflow_tasks[workflow_id] = task
            
            return True
    
    async def request_user_approval(self, workflow_id: str, step_id: str, 
                                   interaction_request: UserInteractionRequest) -> str:
        """Request user approval for a workflow step."""
        async with self.state_lock:
            # Add to pending interactions
            self.interactive_state.pending_interactions[interaction_request.id] = interaction_request
            
            # Update workflow
            workflow = self.interactive_state.active_workflows.get(workflow_id)
            if workflow:
                workflow.pending_interactions.append(interaction_request.id)
                workflow.status = InteractiveWorkflowStatus.WAITING_USER_INPUT
                workflow.last_interaction_at = datetime.now()
            
            # Set timeout if specified
            if interaction_request.timeout_seconds:
                timeout_task = asyncio.create_task(
                    self._handle_interaction_timeout(interaction_request.id, interaction_request.timeout_seconds)
                )
                self.pending_interaction_timeouts[interaction_request.id] = timeout_task
            
            # Record change
            await self._record_workflow_change(
                workflow_id, InteractiveStateChangeType.APPROVAL_REQUESTED,
                {"interaction_id": interaction_request.id, "step_id": step_id}
            )
            
            # Emit event
            event = ApprovalRequestedEvent(
                workflow_id=workflow_id,
                interaction_request_id=interaction_request.id,
                data={"step_id": step_id, "interaction_type": interaction_request.interaction_type.value},
                user_id=workflow.user_id if workflow else None,
                session_id=workflow.session_id if workflow else None
            )
            await self.workflow_event_bus.emit_interaction_event(event)
            
            return interaction_request.id
    
    async def submit_user_response(self, response: UserInteractionResponse) -> bool:
        """Submit a user response to an interaction request."""
        async with self.state_lock:
            # Get the interaction request
            interaction_request = self.interactive_state.pending_interactions.get(response.request_id)
            if not interaction_request:
                return False
            
            # Find the workflow and step
            workflow_id = None
            step_id = None
            
            for wf_id, workflow in self.interactive_state.active_workflows.items():
                if response.request_id in workflow.pending_interactions:
                    workflow_id = wf_id
                    # Find step with this interaction
                    for step in workflow.steps:
                        if step.interaction_request and step.interaction_request.id == response.request_id:
                            step_id = step.id
                            step.user_response = response
                            step.approval_state = response.approval_state
                            break
                    break
            
            if not workflow_id:
                return False
            
            # Remove from pending interactions
            del self.interactive_state.pending_interactions[response.request_id]
            
            # Cancel timeout if exists
            if response.request_id in self.pending_interaction_timeouts:
                self.pending_interaction_timeouts[response.request_id].cancel()
                del self.pending_interaction_timeouts[response.request_id]
            
            # Update workflow
            workflow = self.interactive_state.active_workflows[workflow_id]
            workflow.pending_interactions.remove(response.request_id)
            workflow.completed_interactions.append(response.request_id)
            workflow.last_interaction_at = datetime.now()
            
            # Update approval counts
            if response.approval_state == UserApprovalState.APPROVED:
                workflow.approved_steps += 1
            elif response.approval_state == UserApprovalState.REJECTED:
                workflow.rejected_steps += 1
            
            # Add to interaction history
            self.interactive_state.interaction_history.append(response)
            
            # Record change
            await self._record_workflow_change(
                workflow_id, InteractiveStateChangeType.APPROVAL_RECEIVED,
                {"interaction_id": response.request_id, "approved": response.approval_state == UserApprovalState.APPROVED}
            )
            
            # Emit event
            event = ApprovalReceivedEvent(
                workflow_id=workflow_id,
                interaction_response_id=response.request_id,
                approved=response.approval_state == UserApprovalState.APPROVED,
                data={"step_id": step_id, "approval_state": response.approval_state.value},
                user_id=response.user_id,
                session_id=response.session_id
            )
            await self.workflow_event_bus.emit_interaction_event(event)
            
            # Resume workflow if it was waiting
            if workflow.status == InteractiveWorkflowStatus.WAITING_USER_INPUT and not workflow.pending_interactions:
                workflow.status = InteractiveWorkflowStatus.IN_PROGRESS
                # Resume execution if not already running
                if workflow_id not in self.active_workflow_tasks:
                    task = asyncio.create_task(self._execute_workflow(workflow_id))
                    self.active_workflow_tasks[workflow_id] = task
            
            return True
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow step by step."""
        try:
            while True:
                workflow = self.interactive_state.active_workflows.get(workflow_id)
                if not workflow or workflow.status != InteractiveWorkflowStatus.IN_PROGRESS:
                    break
                
                current_step = workflow.get_current_step()
                if not current_step:
                    # Workflow completed
                    await self._complete_workflow(workflow_id)
                    break
                
                # Check if step is waiting for approval
                if (current_step.requires_approval and 
                    current_step.approval_state == UserApprovalState.PENDING):
                    # Wait for approval
                    workflow.status = InteractiveWorkflowStatus.WAITING_USER_INPUT
                    break
                
                # Execute the step
                await self._execute_step(workflow_id, current_step.id)
                
                # Move to next step
                next_step = workflow.get_next_step()
                if next_step:
                    workflow.current_step_id = next_step.id
                else:
                    # No more steps
                    await self._complete_workflow(workflow_id)
                    break
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            # Workflow was paused/cancelled
            pass
        except Exception as e:
            # Handle workflow error
            await self._handle_workflow_error(workflow_id, str(e))
    
    async def _execute_step(self, workflow_id: str, step_id: str):
        """Execute a single workflow step."""
        workflow = self.interactive_state.active_workflows.get(workflow_id)
        if not workflow:
            return
        
        step = next((s for s in workflow.steps if s.id == step_id), None)
        if not step:
            return
        
        # Start step
        step.status = InteractiveWorkflowStatus.IN_PROGRESS
        step.started_at = datetime.now()
        
        # Emit step started event
        event = StepStartedEvent(
            workflow_id=workflow_id,
            step_id=step_id,
            data={"step_title": step.title, "step_type": step.step_type.value},
            user_id=workflow.user_id,
            session_id=workflow.session_id
        )
        await self.workflow_event_bus.emit_workflow_event(event)
        
        try:
            # Simulate step execution (in real implementation, this would call the actual agent/service)
            await asyncio.sleep(1)  # Placeholder for actual work
            
            # Mark step as completed
            step.status = InteractiveWorkflowStatus.COMPLETED
            step.completed_at = datetime.now()
            step.actual_duration = step.completed_at - step.started_at
            
            # Update workflow progress
            workflow.completed_steps += 1
            
            # Emit step completed event
            event = StepCompletedEvent(
                workflow_id=workflow_id,
                step_id=step_id,
                success=True,
                data={"duration": step.actual_duration.total_seconds()},
                user_id=workflow.user_id,
                session_id=workflow.session_id
            )
            await self.workflow_event_bus.emit_workflow_event(event)
            
        except Exception as e:
            # Handle step error
            step.status = InteractiveWorkflowStatus.ERROR
            step.error_message = str(e)
            step.completed_at = datetime.now()
            
            # Emit step completed event with error
            event = StepCompletedEvent(
                workflow_id=workflow_id,
                step_id=step_id,
                success=False,
                data={"error": str(e)},
                user_id=workflow.user_id,
                session_id=workflow.session_id
            )
            await self.workflow_event_bus.emit_workflow_event(event)
    
    async def _complete_workflow(self, workflow_id: str):
        """Complete a workflow."""
        workflow = self.interactive_state.active_workflows.get(workflow_id)
        if not workflow:
            return
        
        workflow.status = InteractiveWorkflowStatus.COMPLETED
        workflow.completed_at = datetime.now()
        workflow.actual_duration = workflow.completed_at - workflow.started_at
        
        # Clean up
        if workflow_id in self.active_workflow_tasks:
            del self.active_workflow_tasks[workflow_id]
        
        # Record completion
        await self._record_workflow_change(
            workflow_id, InteractiveStateChangeType.WORKFLOW_COMPLETED,
            {"completed_at": workflow.completed_at.isoformat(), "duration": workflow.actual_duration.total_seconds()}
        )
        
        # Emit completion event
        event = InteractiveWorkflowEvent(
            workflow_id=workflow_id,
            event_type="workflow_completed",
            data={"duration": workflow.actual_duration.total_seconds(), "total_steps": workflow.total_steps},
            user_id=workflow.user_id,
            session_id=workflow.session_id
        )
        await self.workflow_event_bus.emit_workflow_event(event)
    
    async def _handle_workflow_error(self, workflow_id: str, error_message: str):
        """Handle workflow execution error."""
        workflow = self.interactive_state.active_workflows.get(workflow_id)
        if workflow:
            workflow.status = InteractiveWorkflowStatus.ERROR
            
            # Clean up
            if workflow_id in self.active_workflow_tasks:
                del self.active_workflow_tasks[workflow_id]
    
    async def _handle_interaction_timeout(self, interaction_id: str, timeout_seconds: int):
        """Handle interaction timeout."""
        await asyncio.sleep(timeout_seconds)
        
        # Check if interaction is still pending
        if interaction_id in self.interactive_state.pending_interactions:
            # Create timeout response
            response = UserInteractionResponse(
                request_id=interaction_id,
                response_value=None,
                approval_state=UserApprovalState.TIMEOUT,
                user_comments="Interaction timed out",
                responded_at=datetime.now()
            )
            
            await self.submit_user_response(response)
    
    async def _record_workflow_change(self, workflow_id: str, change_type: str, data: Dict[str, Any]):
        """Record a workflow state change."""
        change = StateChange(
            id=f"workflow_{change_type}_{workflow_id}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            change_type=change_type,
            path=f"interactive_state.active_workflows.{workflow_id}",
            old_value=None,
            new_value=data,
            metadata={"workflow_id": workflow_id}
        )
        
        self.change_history.append(change)
        await self.event_bus.emit(change)
    
    async def get_workflow_analytics(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics for workflows."""
        if workflow_id:
            workflow = self.interactive_state.active_workflows.get(workflow_id)
            if not workflow:
                return {}
            
            return {
                "workflow_id": workflow_id,
                "status": workflow.status.value,
                "progress_percentage": workflow.calculate_progress_percentage(),
                "total_steps": workflow.total_steps,
                "completed_steps": workflow.completed_steps,
                "approved_steps": workflow.approved_steps,
                "rejected_steps": workflow.rejected_steps,
                "pending_interactions": len(workflow.pending_interactions),
                "duration": workflow.actual_duration.total_seconds() if workflow.actual_duration else None,
                "created_at": workflow.created_at.isoformat(),
                "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
                "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None
            }
        else:
            # Overall analytics
            total_workflows = len(self.interactive_state.active_workflows)
            completed_workflows = sum(1 for w in self.interactive_state.active_workflows.values() 
                                    if w.status == InteractiveWorkflowStatus.COMPLETED)
            
            return {
                "total_workflows": total_workflows,
                "completed_workflows": completed_workflows,
                "active_workflows": sum(1 for w in self.interactive_state.active_workflows.values() 
                                       if w.status == InteractiveWorkflowStatus.IN_PROGRESS),
                "paused_workflows": sum(1 for w in self.interactive_state.active_workflows.values() 
                                       if w.status == InteractiveWorkflowStatus.PAUSED),
                "total_interactions": len(self.interactive_state.interaction_history),
                "pending_interactions": len(self.interactive_state.pending_interactions)
            }

# Global instance
_interactive_state_manager = None

def get_interactive_state_manager() -> InteractiveStateManager:
    """Get the global interactive state manager instance."""
    global _interactive_state_manager
    if _interactive_state_manager is None:
        _interactive_state_manager = InteractiveStateManager()
    return _interactive_state_manager

def initialize_interactive_state_manager(**kwargs) -> InteractiveStateManager:
    """Initialize and return the interactive state manager."""
    global _interactive_state_manager
    _interactive_state_manager = InteractiveStateManager(**kwargs)
    return _interactive_state_manager