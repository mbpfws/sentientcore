"""Advanced Workflow Engine

Provides sophisticated workflow management capabilities for complex multi-agent
development scenarios including dynamic workflow adaptation, parallel task execution,
conditional branching, workflow templates, and real-time workflow monitoring.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
import asyncio
from enum import Enum
from dataclasses import dataclass, field
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_FOR_INPUT = "waiting_for_input"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class ExecutionMode(Enum):
    """Task execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    HUMAN_IN_LOOP = "human_in_loop"


@dataclass
class WorkflowTask:
    """Represents a single task within a workflow."""
    id: str
    name: str
    description: str
    agent_type: str
    task_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    human_validation_required: bool = False
    validation_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowCondition:
    """Represents a conditional branching point in a workflow."""
    condition_id: str
    condition_type: str  # "success", "failure", "custom"
    condition_expression: str
    true_path: List[str]  # Task IDs to execute if condition is true
    false_path: List[str]  # Task IDs to execute if condition is false
    evaluation_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition with tasks and conditions."""
    id: str
    name: str
    description: str
    version: str
    tasks: List[WorkflowTask]
    conditions: List[WorkflowCondition] = field(default_factory=list)
    global_timeout: Optional[int] = None
    max_parallel_tasks: int = 5
    human_intervention_points: List[str] = field(default_factory=list)
    rollback_strategy: str = "none"  # "none", "partial", "full"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Runtime execution state of a workflow."""
    execution_id: str
    workflow_definition: WorkflowDefinition
    status: WorkflowStatus
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    task_results: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    human_inputs: Dict[str, Any] = field(default_factory=dict)
    intervention_history: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedWorkflowEngine:
    """Advanced workflow engine for complex multi-agent orchestration."""
    
    def __init__(self, agent_registry, state_service, memory_service):
        self.agent_registry = agent_registry
        self.state_service = state_service
        self.memory_service = memory_service
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.workflow_templates: Dict[str, WorkflowDefinition] = {}
        self.execution_history: List[WorkflowExecution] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.condition_evaluators: Dict[str, Callable] = {}
        self.human_intervention_callbacks: Dict[str, Callable] = {}
        
    async def initialize(self):
        """Initialize the workflow engine."""
        await self._load_workflow_templates()
        await self._setup_condition_evaluators()
        await self._setup_human_intervention_handlers()
        
    async def create_workflow_from_template(self, template_name: str, parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create workflow from template with parameters."""
        template = self.workflow_templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Clone template and apply parameters
        workflow = self._clone_workflow_definition(template)
        workflow.id = str(uuid.uuid4())
        workflow = await self._apply_parameters_to_workflow(workflow, parameters)
        
        return workflow
    
    async def execute_workflow(self, workflow: WorkflowDefinition, context: Dict[str, Any] = None) -> str:
        """Execute workflow and return execution ID."""
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_definition=workflow,
            status=WorkflowStatus.PENDING,
            execution_context=context or {},
            start_time=datetime.utcnow()
        )
        
        self.active_executions[execution_id] = execution
        
        # Start workflow execution
        asyncio.create_task(self._execute_workflow_async(execution))
        
        return execution_id
    
    async def _execute_workflow_async(self, execution: WorkflowExecution):
        """Asynchronously execute workflow."""
        try:
            execution.status = WorkflowStatus.RUNNING
            await self._update_execution_state(execution)
            
            # Get initial tasks (tasks with no dependencies)
            initial_tasks = self._get_initial_tasks(execution.workflow_definition)
            
            # Execute tasks
            await self._execute_task_graph(execution, initial_tasks)
            
            # Check final status
            if all(task.status == TaskStatus.COMPLETED for task in execution.workflow_definition.tasks):
                execution.status = WorkflowStatus.COMPLETED
            else:
                execution.status = WorkflowStatus.FAILED
            
            execution.end_time = datetime.utcnow()
            execution.progress = 1.0
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.end_time = datetime.utcnow()
            await self._handle_workflow_error(execution, str(e))
        
        finally:
            await self._finalize_execution(execution)
    
    async def _execute_task_graph(self, execution: WorkflowExecution, ready_tasks: List[WorkflowTask]):
        """Execute tasks based on dependency graph."""
        while ready_tasks:
            # Group tasks by execution mode
            parallel_tasks = [t for t in ready_tasks if t.execution_mode == ExecutionMode.PARALLEL]
            sequential_tasks = [t for t in ready_tasks if t.execution_mode == ExecutionMode.SEQUENTIAL]
            human_loop_tasks = [t for t in ready_tasks if t.execution_mode == ExecutionMode.HUMAN_IN_LOOP]
            
            # Execute parallel tasks
            if parallel_tasks:
                await self._execute_parallel_tasks(execution, parallel_tasks)
            
            # Execute sequential tasks
            for task in sequential_tasks:
                await self._execute_single_task(execution, task)
            
            # Handle human-in-the-loop tasks
            for task in human_loop_tasks:
                await self._execute_human_loop_task(execution, task)
            
            # Update progress
            completed_count = len(execution.completed_tasks)
            total_count = len(execution.workflow_definition.tasks)
            execution.progress = completed_count / total_count if total_count > 0 else 0
            
            # Get next ready tasks
            ready_tasks = self._get_ready_tasks(execution)
            
            # Check for conditional branching
            await self._evaluate_conditions(execution)
    
    async def _execute_parallel_tasks(self, execution: WorkflowExecution, tasks: List[WorkflowTask]):
        """Execute multiple tasks in parallel."""
        max_parallel = execution.workflow_definition.max_parallel_tasks
        
        # Split tasks into batches
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i:i + max_parallel]
            
            # Execute batch in parallel
            task_coroutines = [self._execute_single_task(execution, task) for task in batch]
            await asyncio.gather(*task_coroutines, return_exceptions=True)
    
    async def _execute_single_task(self, execution: WorkflowExecution, task: WorkflowTask):
        """Execute a single task."""
        try:
            task.status = TaskStatus.RUNNING
            task.start_time = datetime.utcnow()
            execution.current_tasks.append(task.id)
            
            await self._update_execution_state(execution)
            
            # Get appropriate agent
            agent = await self._get_agent_for_task(task)
            if not agent:
                raise Exception(f"No available agent for task type: {task.agent_type}")
            
            # Execute task with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    agent.process_task(task.task_data),
                    timeout=task.timeout
                )
            else:
                result = await agent.process_task(task.task_data)
            
            # Store result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.utcnow()
            
            execution.completed_tasks.append(task.id)
            execution.task_results[task.id] = result
            
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = "Task timeout"
            execution.failed_tasks.append(task.id)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            execution.failed_tasks.append(task.id)
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                execution.failed_tasks.remove(task.id)
                await asyncio.sleep(2 ** task.retry_count)  # Exponential backoff
                await self._execute_single_task(execution, task)
        
        finally:
            if task.id in execution.current_tasks:
                execution.current_tasks.remove(task.id)
            await self._update_execution_state(execution)
    
    async def _execute_human_loop_task(self, execution: WorkflowExecution, task: WorkflowTask):
        """Execute task requiring human intervention."""
        task.status = TaskStatus.RUNNING
        execution.status = WorkflowStatus.WAITING_FOR_INPUT
        
        # Store intervention request
        intervention_request = {
            "execution_id": execution.execution_id,
            "task_id": task.id,
            "task_name": task.name,
            "description": task.description,
            "validation_criteria": task.validation_criteria,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending"
        }
        
        execution.intervention_history.append(intervention_request)
        await self._update_execution_state(execution)
        
        # Wait for human input
        await self._wait_for_human_input(execution, task)
    
    async def _wait_for_human_input(self, execution: WorkflowExecution, task: WorkflowTask):
        """Wait for human input on a task."""
        # This would typically involve UI interaction
        # For now, we'll simulate with a timeout
        timeout = 3600  # 1 hour timeout for human input
        
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # Check if human input has been provided
            if task.id in execution.human_inputs:
                human_input = execution.human_inputs[task.id]
                
                if human_input.get("approved", False):
                    task.status = TaskStatus.COMPLETED
                    task.result = human_input
                    execution.completed_tasks.append(task.id)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = human_input.get("reason", "Human rejected task")
                    execution.failed_tasks.append(task.id)
                
                execution.status = WorkflowStatus.RUNNING
                break
            
            await asyncio.sleep(10)  # Check every 10 seconds
        else:
            # Timeout reached
            task.status = TaskStatus.FAILED
            task.error = "Human input timeout"
            execution.failed_tasks.append(task.id)
            execution.status = WorkflowStatus.RUNNING
    
    async def provide_human_input(self, execution_id: str, task_id: str, input_data: Dict[str, Any]):
        """Provide human input for a waiting task."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        execution.human_inputs[task_id] = input_data
        
        # Update intervention history
        for intervention in execution.intervention_history:
            if intervention["task_id"] == task_id and intervention["status"] == "pending":
                intervention["status"] = "completed"
                intervention["response"] = input_data
                intervention["response_timestamp"] = datetime.utcnow().isoformat()
                break
        
        await self._update_execution_state(execution)
    
    async def _evaluate_conditions(self, execution: WorkflowExecution):
        """Evaluate workflow conditions and handle branching."""
        for condition in execution.workflow_definition.conditions:
            if await self._should_evaluate_condition(execution, condition):
                result = await self._evaluate_single_condition(execution, condition)
                
                if result:
                    # Execute true path
                    await self._execute_conditional_path(execution, condition.true_path)
                else:
                    # Execute false path
                    await self._execute_conditional_path(execution, condition.false_path)
    
    async def _evaluate_single_condition(self, execution: WorkflowExecution, condition: WorkflowCondition) -> bool:
        """Evaluate a single condition."""
        evaluator = self.condition_evaluators.get(condition.condition_type)
        if not evaluator:
            raise ValueError(f"Unknown condition type: {condition.condition_type}")
        
        context = {
            "execution": execution,
            "task_results": execution.task_results,
            **condition.evaluation_context
        }
        
        return await evaluator(condition.condition_expression, context)
    
    def _get_initial_tasks(self, workflow: WorkflowDefinition) -> List[WorkflowTask]:
        """Get tasks with no dependencies."""
        return [task for task in workflow.tasks if not task.dependencies]
    
    def _get_ready_tasks(self, execution: WorkflowExecution) -> List[WorkflowTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        
        for task in execution.workflow_definition.tasks:
            if (task.status == TaskStatus.PENDING and 
                all(dep_id in execution.completed_tasks for dep_id in task.dependencies)):
                ready_tasks.append(task)
        
        return ready_tasks
    
    async def _get_agent_for_task(self, task: WorkflowTask):
        """Get appropriate agent for task execution."""
        # This would integrate with the agent registry
        # For now, return a mock agent
        return MockAgent(task.agent_type)
    
    async def _update_execution_state(self, execution: WorkflowExecution):
        """Update execution state in state service."""
        if self.state_service:
            await self.state_service.update_workflow_execution(execution)
    
    async def _handle_workflow_error(self, execution: WorkflowExecution, error: str):
        """Handle workflow execution error."""
        logger.error(f"Workflow {execution.execution_id} failed: {error}")
        
        # Store error in execution context
        execution.execution_context["error"] = error
        execution.execution_context["error_timestamp"] = datetime.utcnow().isoformat()
    
    async def _finalize_execution(self, execution: WorkflowExecution):
        """Finalize workflow execution."""
        # Move to history
        self.execution_history.append(execution)
        
        # Remove from active executions
        if execution.execution_id in self.active_executions:
            del self.active_executions[execution.execution_id]
        
        # Update state service
        await self._update_execution_state(execution)
    
    async def _load_workflow_templates(self):
        """Load workflow templates from storage."""
        # This would load templates from files or database
        # For now, create some basic templates
        self.workflow_templates["feature_development"] = self._create_feature_development_template()
        self.workflow_templates["research_workflow"] = self._create_research_workflow_template()
    
    async def _setup_condition_evaluators(self):
        """Setup condition evaluators."""
        self.condition_evaluators["success"] = self._evaluate_success_condition
        self.condition_evaluators["failure"] = self._evaluate_failure_condition
        self.condition_evaluators["custom"] = self._evaluate_custom_condition
    
    async def _setup_human_intervention_handlers(self):
        """Setup human intervention handlers."""
        # This would setup UI callbacks and notification systems
        pass
    
    def _clone_workflow_definition(self, template: WorkflowDefinition) -> WorkflowDefinition:
        """Clone a workflow definition."""
        import copy
        return copy.deepcopy(template)
    
    async def _apply_parameters_to_workflow(self, workflow: WorkflowDefinition, parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Apply parameters to workflow template."""
        # This would substitute parameters in task data and conditions
        # For now, just return the workflow as-is
        return workflow
    
    async def _should_evaluate_condition(self, execution: WorkflowExecution, condition: WorkflowCondition) -> bool:
        """Check if condition should be evaluated."""
        # This would check if the condition's prerequisites are met
        return True
    
    async def _execute_conditional_path(self, execution: WorkflowExecution, task_ids: List[str]):
        """Execute tasks in a conditional path."""
        tasks_to_execute = []
        for task_id in task_ids:
            task = next((t for t in execution.workflow_definition.tasks if t.id == task_id), None)
            if task and task.status == TaskStatus.PENDING:
                tasks_to_execute.append(task)
        
        if tasks_to_execute:
            await self._execute_task_graph(execution, tasks_to_execute)
    
    async def _evaluate_success_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate success condition."""
        # Simple success condition evaluation
        execution = context["execution"]
        return len(execution.failed_tasks) == 0
    
    async def _evaluate_failure_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate failure condition."""
        # Simple failure condition evaluation
        execution = context["execution"]
        return len(execution.failed_tasks) > 0
    
    async def _evaluate_custom_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate custom condition."""
        # This would evaluate custom expressions
        # For now, just return True
        return True
    
    def _create_feature_development_template(self) -> WorkflowDefinition:
        """Create feature development workflow template."""
        return WorkflowDefinition(
            id="feature_development_template",
            name="Feature Development Workflow",
            description="Standard workflow for feature development",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="analyze_requirements",
                    name="Analyze Requirements",
                    description="Analyze and understand feature requirements",
                    agent_type="research_agent",
                    task_data={"type": "requirement_analysis"},
                    execution_mode=ExecutionMode.SEQUENTIAL
                ),
                WorkflowTask(
                    id="design_architecture",
                    name="Design Architecture",
                    description="Design system architecture for the feature",
                    agent_type="architect_agent",
                    task_data={"type": "architecture_design"},
                    dependencies=["analyze_requirements"],
                    execution_mode=ExecutionMode.SEQUENTIAL
                ),
                WorkflowTask(
                    id="implement_backend",
                    name="Implement Backend",
                    description="Implement backend components",
                    agent_type="backend_developer",
                    task_data={"type": "backend_implementation"},
                    dependencies=["design_architecture"],
                    execution_mode=ExecutionMode.PARALLEL
                ),
                WorkflowTask(
                    id="implement_frontend",
                    name="Implement Frontend",
                    description="Implement frontend components",
                    agent_type="frontend_developer",
                    task_data={"type": "frontend_implementation"},
                    dependencies=["design_architecture"],
                    execution_mode=ExecutionMode.PARALLEL
                ),
                WorkflowTask(
                    id="integration_testing",
                    name="Integration Testing",
                    description="Perform integration testing",
                    agent_type="testing_agent",
                    task_data={"type": "integration_testing"},
                    dependencies=["implement_backend", "implement_frontend"],
                    execution_mode=ExecutionMode.SEQUENTIAL
                )
            ]
        )
    
    def _create_research_workflow_template(self) -> WorkflowDefinition:
        """Create research workflow template."""
        return WorkflowDefinition(
            id="research_workflow_template",
            name="Research Workflow",
            description="Standard workflow for research tasks",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="gather_information",
                    name="Gather Information",
                    description="Gather relevant information and sources",
                    agent_type="research_agent",
                    task_data={"type": "information_gathering"},
                    execution_mode=ExecutionMode.SEQUENTIAL
                ),
                WorkflowTask(
                    id="analyze_data",
                    name="Analyze Data",
                    description="Analyze gathered data and information",
                    agent_type="research_agent",
                    task_data={"type": "data_analysis"},
                    dependencies=["gather_information"],
                    execution_mode=ExecutionMode.SEQUENTIAL
                ),
                WorkflowTask(
                    id="synthesize_findings",
                    name="Synthesize Findings",
                    description="Synthesize research findings",
                    agent_type="research_agent",
                    task_data={"type": "synthesis"},
                    dependencies=["analyze_data"],
                    execution_mode=ExecutionMode.SEQUENTIAL
                )
            ]
        )


class MockAgent:
    """Mock agent for testing purposes."""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
    
    async def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return results."""
        # Simulate task processing
        await asyncio.sleep(0.1)
        
        return {
            "status": "completed",
            "result": f"Task processed by {self.agent_type}",
            "task_data": task_data,
            "timestamp": datetime.utcnow().isoformat()
        }