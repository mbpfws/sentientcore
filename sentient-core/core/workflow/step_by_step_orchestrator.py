"""Step-by-Step Workflow Orchestrator

Orchestrates interactive workflows with step-by-step execution,
user approval gates, and progress tracking.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import uuid
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path

from ..state.interactive_workflow_models import (
    InteractiveWorkflow, WorkflowStep, InteractiveWorkflowStatus,
    UserApprovalState, StepType, StepPriority, InteractionType,
    UserInteractionRequest, UserInteractionResponse
)
from ..state.interactive_state_manager import InteractiveStateManager
from .approval_gates import (
    ApprovalGateManager, ApprovalContext, ApprovalDecision,
    create_default_approval_manager
)
from ..models import AppState, EnhancedTask, TaskStatus

class OrchestrationMode(Enum):
    """Workflow orchestration modes."""
    MANUAL = "manual"  # User controls each step
    SEMI_AUTOMATIC = "semi_automatic"  # Auto-execute with approval gates
    AUTOMATIC = "automatic"  # Full automation with optional checkpoints
    GUIDED = "guided"  # Interactive guidance with suggestions

class StepExecutionResult(Enum):
    """Results of step execution."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PENDING_APPROVAL = "pending_approval"
    USER_REJECTED = "user_rejected"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class ExecutionContext:
    """Context for step execution."""
    workflow_id: str
    step_id: str
    user_id: str
    session_id: str
    app_state: AppState
    execution_mode: OrchestrationMode
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepExecutionReport:
    """Report of step execution."""
    step_id: str
    result: StepExecutionResult
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[timedelta] = None
    output: Optional[Any] = None
    error_message: Optional[str] = None
    artifacts_created: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    user_feedback: Optional[str] = None

class StepExecutor(ABC):
    """Abstract base class for step executors."""
    
    @abstractmethod
    async def execute(self, step: WorkflowStep, context: ExecutionContext) -> StepExecutionReport:
        """Execute a workflow step."""
        pass
    
    @abstractmethod
    def can_execute(self, step_type: StepType) -> bool:
        """Check if this executor can handle the step type."""
        pass

class CodeGenerationExecutor(StepExecutor):
    """Executor for code generation steps."""
    
    def can_execute(self, step_type: StepType) -> bool:
        return step_type in [StepType.CODE_GENERATION, StepType.FILE_CREATION]
    
    async def execute(self, step: WorkflowStep, context: ExecutionContext) -> StepExecutionReport:
        """Execute code generation step."""
        started_at = datetime.now()
        
        try:
            # Simulate code generation (in real implementation, this would call the actual agent)
            await asyncio.sleep(2)  # Simulate work
            
            # Create mock output
            output = {
                "files_created": [f"src/{step.title.lower().replace(' ', '_')}.py"],
                "lines_of_code": 150,
                "functions_created": 3,
                "classes_created": 1
            }
            
            completed_at = datetime.now()
            
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.SUCCESS,
                started_at=started_at,
                completed_at=completed_at,
                duration=completed_at - started_at,
                output=output,
                artifacts_created=output["files_created"],
                metrics={
                    "execution_time_seconds": (completed_at - started_at).total_seconds(),
                    "complexity_score": 7.5
                }
            )
            
        except Exception as e:
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.ERROR,
                started_at=started_at,
                completed_at=datetime.now(),
                error_message=str(e)
            )

class FileOperationExecutor(StepExecutor):
    """Executor for file operation steps."""
    
    def can_execute(self, step_type: StepType) -> bool:
        return step_type in [StepType.FILE_MODIFICATION, StepType.FILE_DELETION]
    
    async def execute(self, step: WorkflowStep, context: ExecutionContext) -> StepExecutionReport:
        """Execute file operation step."""
        started_at = datetime.now()
        
        try:
            # Simulate file operations
            await asyncio.sleep(1)  # Simulate work
            
            output = {
                "operation": step.step_type.value,
                "files_affected": [f"src/{step.title.lower().replace(' ', '_')}.py"],
                "backup_created": True
            }
            
            completed_at = datetime.now()
            
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.SUCCESS,
                started_at=started_at,
                completed_at=completed_at,
                duration=completed_at - started_at,
                output=output,
                artifacts_created=[f"backup_{step.id}.bak"],
                metrics={
                    "execution_time_seconds": (completed_at - started_at).total_seconds(),
                    "files_processed": len(output["files_affected"])
                }
            )
            
        except Exception as e:
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.ERROR,
                started_at=started_at,
                completed_at=datetime.now(),
                error_message=str(e)
            )

class TestingExecutor(StepExecutor):
    """Executor for testing steps."""
    
    def can_execute(self, step_type: StepType) -> bool:
        return step_type in [StepType.TESTING, StepType.VALIDATION]
    
    async def execute(self, step: WorkflowStep, context: ExecutionContext) -> StepExecutionReport:
        """Execute testing step."""
        started_at = datetime.now()
        
        try:
            # Simulate testing
            await asyncio.sleep(3)  # Simulate test execution
            
            output = {
                "tests_run": 25,
                "tests_passed": 23,
                "tests_failed": 2,
                "coverage_percentage": 87.5,
                "test_report_path": f"reports/test_report_{step.id}.html"
            }
            
            completed_at = datetime.now()
            
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.SUCCESS if output["tests_failed"] == 0 else StepExecutionResult.FAILED,
                started_at=started_at,
                completed_at=completed_at,
                duration=completed_at - started_at,
                output=output,
                artifacts_created=[output["test_report_path"]],
                metrics={
                    "execution_time_seconds": (completed_at - started_at).total_seconds(),
                    "test_success_rate": output["tests_passed"] / output["tests_run"],
                    "coverage_score": output["coverage_percentage"]
                }
            )
            
        except Exception as e:
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.ERROR,
                started_at=started_at,
                completed_at=datetime.now(),
                error_message=str(e)
            )

class DeploymentExecutor(StepExecutor):
    """Executor for deployment steps."""
    
    def can_execute(self, step_type: StepType) -> bool:
        return step_type in [StepType.DEPLOYMENT, StepType.CONFIGURATION]
    
    async def execute(self, step: WorkflowStep, context: ExecutionContext) -> StepExecutionReport:
        """Execute deployment step."""
        started_at = datetime.now()
        
        try:
            # Simulate deployment
            await asyncio.sleep(5)  # Simulate deployment time
            
            output = {
                "deployment_target": "production",
                "services_deployed": ["api-service", "web-frontend"],
                "deployment_url": "https://app.example.com",
                "health_check_passed": True
            }
            
            completed_at = datetime.now()
            
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.SUCCESS,
                started_at=started_at,
                completed_at=completed_at,
                duration=completed_at - started_at,
                output=output,
                artifacts_created=[f"deployment_log_{step.id}.txt"],
                metrics={
                    "execution_time_seconds": (completed_at - started_at).total_seconds(),
                    "services_count": len(output["services_deployed"])
                }
            )
            
        except Exception as e:
            return StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.ERROR,
                started_at=started_at,
                completed_at=datetime.now(),
                error_message=str(e)
            )

class StepByStepOrchestrator:
    """Orchestrates step-by-step workflow execution."""
    
    def __init__(self, state_manager: InteractiveStateManager, 
                 approval_manager: Optional[ApprovalGateManager] = None):
        self.state_manager = state_manager
        self.approval_manager = approval_manager or create_default_approval_manager(state_manager)
        
        # Step executors
        self.executors: List[StepExecutor] = [
            CodeGenerationExecutor(),
            FileOperationExecutor(),
            TestingExecutor(),
            DeploymentExecutor()
        ]
        
        # Execution tracking
        self.active_executions: Dict[str, asyncio.Task] = {}
        self.execution_reports: Dict[str, List[StepExecutionReport]] = {}
        self.execution_locks: Dict[str, asyncio.Lock] = {}
        
        # Event handlers
        self.step_started_handlers: List[Callable] = []
        self.step_completed_handlers: List[Callable] = []
        self.approval_requested_handlers: List[Callable] = []
        
        # Metrics
        self.orchestrator_metrics = {
            "workflows_orchestrated": 0,
            "steps_executed": 0,
            "approvals_requested": 0,
            "average_step_duration": 0.0,
            "success_rate": 0.0
        }
        
        print("Step-by-Step Orchestrator initialized")
    
    def add_executor(self, executor: StepExecutor):
        """Add a custom step executor."""
        self.executors.append(executor)
    
    def get_executor_for_step(self, step_type: StepType) -> Optional[StepExecutor]:
        """Get the appropriate executor for a step type."""
        for executor in self.executors:
            if executor.can_execute(step_type):
                return executor
        return None
    
    async def orchestrate_workflow(self, workflow_id: str, 
                                  execution_mode: OrchestrationMode = OrchestrationMode.SEMI_AUTOMATIC,
                                  app_state: Optional[AppState] = None) -> bool:
        """Start orchestrating a workflow."""
        # Get workflow
        interactive_state = await self.state_manager.get_interactive_state()
        workflow = interactive_state.active_workflows.get(workflow_id)
        
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        # Initialize execution tracking
        if workflow_id not in self.execution_locks:
            self.execution_locks[workflow_id] = asyncio.Lock()
        
        self.execution_reports[workflow_id] = []
        
        # Start workflow
        success = await self.state_manager.start_workflow(workflow_id)
        if not success:
            return False
        
        # Update metrics
        self.orchestrator_metrics["workflows_orchestrated"] += 1
        
        # Start orchestration task
        task = asyncio.create_task(
            self._orchestrate_workflow_execution(workflow_id, execution_mode, app_state)
        )
        self.active_executions[workflow_id] = task
        
        return True
    
    async def _orchestrate_workflow_execution(self, workflow_id: str, 
                                             execution_mode: OrchestrationMode,
                                             app_state: Optional[AppState]):
        """Execute workflow orchestration."""
        try:
            async with self.execution_locks[workflow_id]:
                while True:
                    # Get current workflow state
                    interactive_state = await self.state_manager.get_interactive_state()
                    workflow = interactive_state.active_workflows.get(workflow_id)
                    
                    if not workflow or workflow.status != InteractiveWorkflowStatus.IN_PROGRESS:
                        break
                    
                    current_step = workflow.get_current_step()
                    if not current_step:
                        # Workflow completed
                        break
                    
                    # Execute current step
                    await self._execute_workflow_step(workflow_id, current_step, execution_mode, app_state)
                    
                    # Check if workflow is still in progress
                    interactive_state = await self.state_manager.get_interactive_state()
                    workflow = interactive_state.active_workflows.get(workflow_id)
                    
                    if not workflow or workflow.status != InteractiveWorkflowStatus.IN_PROGRESS:
                        break
                    
                    # Move to next step
                    next_step = workflow.get_next_step()
                    if next_step:
                        workflow.current_step_id = next_step.id
                    else:
                        # No more steps
                        break
                    
                    # Small delay to prevent tight loop
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            # Orchestration was cancelled
            pass
        except Exception as e:
            print(f"Error in workflow orchestration: {e}")
        finally:
            # Clean up
            if workflow_id in self.active_executions:
                del self.active_executions[workflow_id]
    
    async def _execute_workflow_step(self, workflow_id: str, step: WorkflowStep, 
                                    execution_mode: OrchestrationMode, app_state: Optional[AppState]):
        """Execute a single workflow step."""
        # Create execution context
        interactive_state = await self.state_manager.get_interactive_state()
        workflow = interactive_state.active_workflows.get(workflow_id)
        
        context = ExecutionContext(
            workflow_id=workflow_id,
            step_id=step.id,
            user_id=workflow.user_id if workflow else "system",
            session_id=workflow.session_id if workflow else "default",
            app_state=app_state or AppState(),
            execution_mode=execution_mode,
            metadata=step.metadata
        )
        
        # Check if step requires approval
        if step.requires_approval and step.approval_state == UserApprovalState.PENDING:
            await self._request_step_approval(step, context)
            return
        
        # Check if step was rejected
        if step.approval_state == UserApprovalState.REJECTED:
            report = StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.USER_REJECTED,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                user_feedback="Step rejected by user"
            )
            self.execution_reports[workflow_id].append(report)
            return
        
        # Execute step
        await self._execute_step_with_executor(step, context)
    
    async def _request_step_approval(self, step: WorkflowStep, context: ExecutionContext):
        """Request approval for a step."""
        # Create approval context
        approval_context = ApprovalContext(
            workflow_id=context.workflow_id,
            step_id=step.id,
            step_title=step.title,
            step_description=step.description,
            step_type=step.step_type,
            priority=step.priority,
            estimated_impact=step.estimated_impact or "Medium",
            risk_level=step.risk_level or "Medium",
            dependencies=step.dependencies,
            metadata=step.metadata
        )
        
        # Request approval
        try:
            request_id = await self.approval_manager.request_approval(
                approval_context, context.user_id, context.session_id
            )
            
            # Update step with interaction request
            step.interaction_request = UserInteractionRequest(
                id=request_id,
                interaction_type=InteractionType.APPROVAL_REQUEST,
                title=f"Approval Required: {step.title}",
                description=f"Please approve the execution of: {step.description}",
                user_id=context.user_id,
                session_id=context.session_id
            )
            
            # Update metrics
            self.orchestrator_metrics["approvals_requested"] += 1
            
            # Notify handlers
            for handler in self.approval_requested_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(step, approval_context)
                    else:
                        handler(step, approval_context)
                except Exception as e:
                    print(f"Error in approval requested handler: {e}")
                    
        except Exception as e:
            print(f"Error requesting approval: {e}")
            # Mark step as failed
            step.status = InteractiveWorkflowStatus.ERROR
            step.error_message = f"Failed to request approval: {e}"
    
    async def _execute_step_with_executor(self, step: WorkflowStep, context: ExecutionContext):
        """Execute step using appropriate executor."""
        # Get executor
        executor = self.get_executor_for_step(step.step_type)
        if not executor:
            # No executor found
            report = StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.ERROR,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error_message=f"No executor found for step type: {step.step_type.value}"
            )
            self.execution_reports[context.workflow_id].append(report)
            return
        
        # Notify step started
        for handler in self.step_started_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(step, context)
                else:
                    handler(step, context)
            except Exception as e:
                print(f"Error in step started handler: {e}")
        
        # Execute step
        try:
            report = await executor.execute(step, context)
            
            # Update step status
            if report.result == StepExecutionResult.SUCCESS:
                step.status = InteractiveWorkflowStatus.COMPLETED
                step.completed_at = report.completed_at
                step.actual_duration = report.duration
            else:
                step.status = InteractiveWorkflowStatus.ERROR
                step.error_message = report.error_message
            
            # Store report
            self.execution_reports[context.workflow_id].append(report)
            
            # Update metrics
            self.orchestrator_metrics["steps_executed"] += 1
            if report.duration:
                current_avg = self.orchestrator_metrics["average_step_duration"]
                total_steps = self.orchestrator_metrics["steps_executed"]
                self.orchestrator_metrics["average_step_duration"] = (
                    (current_avg * (total_steps - 1) + report.duration.total_seconds()) / total_steps
                )
            
            # Update success rate
            successful_steps = sum(1 for reports in self.execution_reports.values() 
                                 for r in reports if r.result == StepExecutionResult.SUCCESS)
            total_executed = sum(len(reports) for reports in self.execution_reports.values())
            if total_executed > 0:
                self.orchestrator_metrics["success_rate"] = successful_steps / total_executed
            
            # Notify step completed
            for handler in self.step_completed_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(step, context, report)
                    else:
                        handler(step, context, report)
                except Exception as e:
                    print(f"Error in step completed handler: {e}")
                    
        except Exception as e:
            # Execution failed
            report = StepExecutionReport(
                step_id=step.id,
                result=StepExecutionResult.ERROR,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error_message=str(e)
            )
            
            step.status = InteractiveWorkflowStatus.ERROR
            step.error_message = str(e)
            
            self.execution_reports[context.workflow_id].append(report)
    
    async def pause_workflow_orchestration(self, workflow_id: str) -> bool:
        """Pause workflow orchestration."""
        if workflow_id in self.active_executions:
            self.active_executions[workflow_id].cancel()
            del self.active_executions[workflow_id]
        
        return await self.state_manager.pause_workflow(workflow_id, "Orchestration paused")
    
    async def resume_workflow_orchestration(self, workflow_id: str, 
                                           execution_mode: OrchestrationMode = OrchestrationMode.SEMI_AUTOMATIC,
                                           app_state: Optional[AppState] = None) -> bool:
        """Resume workflow orchestration."""
        success = await self.state_manager.resume_workflow(workflow_id)
        if not success:
            return False
        
        # Restart orchestration task
        task = asyncio.create_task(
            self._orchestrate_workflow_execution(workflow_id, execution_mode, app_state)
        )
        self.active_executions[workflow_id] = task
        
        return True
    
    def add_step_started_handler(self, handler: Callable):
        """Add handler for step started events."""
        self.step_started_handlers.append(handler)
    
    def add_step_completed_handler(self, handler: Callable):
        """Add handler for step completed events."""
        self.step_completed_handlers.append(handler)
    
    def add_approval_requested_handler(self, handler: Callable):
        """Add handler for approval requested events."""
        self.approval_requested_handlers.append(handler)
    
    def get_workflow_execution_report(self, workflow_id: str) -> Dict[str, Any]:
        """Get execution report for a workflow."""
        reports = self.execution_reports.get(workflow_id, [])
        
        if not reports:
            return {"workflow_id": workflow_id, "reports": [], "summary": {}}
        
        # Calculate summary
        total_steps = len(reports)
        successful_steps = sum(1 for r in reports if r.result == StepExecutionResult.SUCCESS)
        failed_steps = sum(1 for r in reports if r.result in [StepExecutionResult.FAILED, StepExecutionResult.ERROR])
        pending_steps = sum(1 for r in reports if r.result == StepExecutionResult.PENDING_APPROVAL)
        
        total_duration = sum((r.duration.total_seconds() for r in reports if r.duration), 0)
        
        summary = {
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "failed_steps": failed_steps,
            "pending_steps": pending_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "total_duration_seconds": total_duration,
            "average_step_duration": total_duration / total_steps if total_steps > 0 else 0
        }
        
        return {
            "workflow_id": workflow_id,
            "reports": [{
                "step_id": r.step_id,
                "result": r.result.value,
                "started_at": r.started_at.isoformat(),
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "duration_seconds": r.duration.total_seconds() if r.duration else None,
                "output": r.output,
                "error_message": r.error_message,
                "artifacts_created": r.artifacts_created,
                "metrics": r.metrics
            } for r in reports],
            "summary": summary
        }
    
    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return self.orchestrator_metrics.copy()
    
    def list_active_orchestrations(self) -> List[str]:
        """List active workflow orchestrations."""
        return list(self.active_executions.keys())

# Factory function
def create_step_by_step_orchestrator(state_manager: InteractiveStateManager, 
                                    approval_manager: Optional[ApprovalGateManager] = None) -> StepByStepOrchestrator:
    """Create a step-by-step orchestrator with default configuration."""
    return StepByStepOrchestrator(state_manager, approval_manager)