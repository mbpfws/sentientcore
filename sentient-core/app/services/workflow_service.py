from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator
import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import traceback

from .memory_service import MemoryService, MemoryType
from .state_manager import EnhancedStateManager, StateChangeType
from .llm_service import EnhancedLLMService, ChatMessage, MessageRole
from .sse_manager import SSEConnectionManager, EventType


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TaskType(Enum):
    """Types of tasks in workflow"""
    LLM_GENERATION = "llm_generation"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    CUSTOM = "custom"
    AGENT_TASK = "agent_task"
    PARALLEL_GROUP = "parallel_group"
    CONDITIONAL = "conditional"


class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskDependency:
    """Represents a task dependency"""
    task_id: str
    condition: Optional[str] = None  # Optional condition for dependency
    required_status: TaskStatus = TaskStatus.COMPLETED


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowTask:
    """Represents a task in a workflow"""
    id: str
    name: str
    task_type: TaskType
    handler: Union[str, Callable]
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[TaskDependency] = field(default_factory=list)
    priority: Priority = Priority.NORMAL
    timeout: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Defines a workflow structure"""
    id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    max_parallel_tasks: int = 5
    error_handling: str = "stop"  # stop, continue, retry
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance"""
    id: str
    workflow_id: str
    session_id: Optional[str]
    status: WorkflowStatus
    current_task: Optional[str] = None
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator:
    """Advanced workflow orchestration service"""
    
    def __init__(
        self,
        memory_service: MemoryService,
        state_manager: EnhancedStateManager,
        llm_service: EnhancedLLMService,
        sse_manager: SSEConnectionManager
    ):
        self.logger = logging.getLogger(__name__)
        
        # Core services
        self.memory_service = memory_service
        self.state_manager = state_manager
        self.llm_service = llm_service
        self.sse_manager = sse_manager
        
        # Workflow storage
        self._workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self._active_executions: Dict[str, WorkflowExecution] = {}
        self._execution_history: List[WorkflowExecution] = []
        
        # Task handlers registry
        self._task_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # Execution control
        self._executor_pool = asyncio.Semaphore(10)  # Max concurrent executions
        self._task_queues: Dict[str, asyncio.Queue] = {}  # Per-execution task queues
        self._running_tasks: Dict[str, asyncio.Task] = {}  # Track running async tasks
        
        # Performance tracking
        self._stats = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_workflow_time": 0.0,
            "average_task_time": 0.0
        }
    
    def _register_default_handlers(self):
        """Register default task handlers"""
        self._task_handlers.update({
            "llm_generation": self._handle_llm_generation,
            "research": self._handle_research_task,
            "analysis": self._handle_analysis_task,
            "synthesis": self._handle_synthesis_task,
            "validation": self._handle_validation_task,
            "parallel_group": self._handle_parallel_group,
            "conditional": self._handle_conditional_task
        })
    
    async def register_workflow(self, workflow: WorkflowDefinition) -> str:
        """Register a new workflow definition"""
        # Validate workflow
        await self._validate_workflow(workflow)
        
        # Store workflow
        self._workflow_definitions[workflow.id] = workflow
        
        # Store in memory service for persistence
        await self.memory_service.store_memory(
            content=json.dumps({
                "workflow_id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "task_count": len(workflow.tasks),
                "created_at": workflow.created_at.isoformat()
            }),
            memory_type=MemoryType.WORKFLOW,
            metadata={
                "workflow_id": workflow.id,
                "type": "workflow_definition"
            }
        )
        
        self.logger.info(f"Registered workflow: {workflow.name} ({workflow.id})")
        return workflow.id
    
    async def _validate_workflow(self, workflow: WorkflowDefinition):
        """Validate workflow definition"""
        if not workflow.tasks:
            raise ValueError("Workflow must have at least one task")
        
        task_ids = {task.id for task in workflow.tasks}
        
        # Validate task dependencies
        for task in workflow.tasks:
            for dep in task.dependencies:
                if dep.task_id not in task_ids:
                    raise ValueError(f"Task {task.id} depends on non-existent task {dep.task_id}")
        
        # Check for circular dependencies
        await self._check_circular_dependencies(workflow.tasks)
        
        # Validate task handlers
        for task in workflow.tasks:
            if isinstance(task.handler, str) and task.handler not in self._task_handlers:
                raise ValueError(f"Unknown task handler: {task.handler}")
    
    async def _check_circular_dependencies(self, tasks: List[WorkflowTask]):
        """Check for circular dependencies in workflow"""
        task_map = {task.id: task for task in tasks}
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep in task.dependencies:
                    if has_cycle(dep.task_id):
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    raise ValueError(f"Circular dependency detected involving task {task.id}")
    
    async def start_workflow(
        self,
        workflow_id: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start workflow execution"""
        if workflow_id not in self._workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_def = self._workflow_definitions[workflow_id]
        execution_id = str(uuid.uuid4())
        
        # Create execution instance
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            session_id=session_id,
            status=WorkflowStatus.PENDING,
            context=context or {},
            metadata={
                "parameters": parameters or {},
                "workflow_name": workflow_def.name
            }
        )
        
        # Store execution
        self._active_executions[execution_id] = execution
        
        # Create task queue for this execution
        self._task_queues[execution_id] = asyncio.Queue()
        
        # Start execution in background
        task = asyncio.create_task(self._execute_workflow(execution_id))
        self._running_tasks[execution_id] = task
        
        # Notify via SSE
        await self.sse_manager.broadcast_workflow_update(
            execution_id,
            {
                "status": execution.status.value,
                "workflow_name": workflow_def.name,
                "started_at": execution.started_at.isoformat()
            },
            session_id
        )
        
        self.logger.info(f"Started workflow execution: {execution_id}")
        return execution_id
    
    async def _execute_workflow(self, execution_id: str):
        """Execute workflow tasks"""
        async with self._executor_pool:
            execution = self._active_executions[execution_id]
            workflow_def = self._workflow_definitions[execution.workflow_id]
            
            try:
                execution.status = WorkflowStatus.RUNNING
                start_time = datetime.now()
                
                # Build task dependency graph
                task_graph = self._build_task_graph(workflow_def.tasks)
                
                # Execute tasks based on dependencies
                await self._execute_task_graph(execution_id, task_graph)
                
                # Check final status
                if execution.failed_tasks:
                    execution.status = WorkflowStatus.FAILED
                    execution.error = f"Failed tasks: {', '.join(execution.failed_tasks)}"
                else:
                    execution.status = WorkflowStatus.COMPLETED
                
                execution.completed_at = datetime.now()
                
                # Update statistics
                execution_time = (execution.completed_at - start_time).total_seconds()
                self._update_workflow_stats(execution.status == WorkflowStatus.COMPLETED, execution_time)
                
                # Store execution result
                await self._store_execution_result(execution)
                
                # Notify completion
                await self.sse_manager.broadcast_workflow_update(
                    execution_id,
                    {
                        "status": execution.status.value,
                        "completed_at": execution.completed_at.isoformat(),
                        "execution_time": execution_time,
                        "completed_tasks": len(execution.completed_tasks),
                        "failed_tasks": len(execution.failed_tasks)
                    },
                    execution.session_id
                )
                
            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.error = str(e)
                execution.completed_at = datetime.now()
                
                self.logger.error(f"Workflow execution failed: {e}\n{traceback.format_exc()}")
                
                await self.sse_manager.broadcast_error(
                    f"Workflow execution failed: {e}",
                    execution.session_id
                )
            
            finally:
                # Cleanup
                self._active_executions.pop(execution_id, None)
                self._task_queues.pop(execution_id, None)
                self._running_tasks.pop(execution_id, None)
                
                # Move to history
                self._execution_history.append(execution)
                
                # Limit history size
                if len(self._execution_history) > 1000:
                    self._execution_history = self._execution_history[-500:]
    
    def _build_task_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        
        for task in tasks:
            graph[task.id] = [dep.task_id for dep in task.dependencies]
        
        return graph
    
    async def _execute_task_graph(self, execution_id: str, task_graph: Dict[str, List[str]]):
        """Execute tasks based on dependency graph"""
        execution = self._active_executions[execution_id]
        workflow_def = self._workflow_definitions[execution.workflow_id]
        task_map = {task.id: task for task in workflow_def.tasks}
        
        completed = set()
        running = set()
        semaphore = asyncio.Semaphore(workflow_def.max_parallel_tasks)
        
        async def execute_task(task_id: str):
            if task_id in completed or task_id in running:
                return
            
            # Check dependencies
            dependencies = task_graph.get(task_id, [])
            for dep_id in dependencies:
                if dep_id not in completed:
                    # Wait for dependency
                    while dep_id not in completed:
                        await asyncio.sleep(0.1)
                        if execution.status != WorkflowStatus.RUNNING:
                            return
            
            running.add(task_id)
            
            async with semaphore:
                try:
                    task = task_map[task_id]
                    execution.current_task = task_id
                    
                    # Execute task
                    result = await self._execute_single_task(execution_id, task)
                    
                    # Store result
                    execution.task_results[task_id] = result
                    
                    if result.status == TaskStatus.COMPLETED:
                        execution.completed_tasks.append(task_id)
                        completed.add(task_id)
                    else:
                        execution.failed_tasks.append(task_id)
                        
                        # Handle error based on workflow policy
                        if workflow_def.error_handling == "stop":
                            execution.status = WorkflowStatus.FAILED
                            return
                    
                except Exception as e:
                    self.logger.error(f"Task execution error: {e}")
                    execution.failed_tasks.append(task_id)
                    
                    if workflow_def.error_handling == "stop":
                        execution.status = WorkflowStatus.FAILED
                        return
                
                finally:
                    running.discard(task_id)
                    execution.current_task = None
        
        # Start all tasks that have no dependencies
        initial_tasks = [task_id for task_id, deps in task_graph.items() if not deps]
        
        # Execute all tasks
        all_tasks = list(task_graph.keys())
        tasks_to_execute = [execute_task(task_id) for task_id in all_tasks]
        
        await asyncio.gather(*tasks_to_execute, return_exceptions=True)
    
    async def _execute_single_task(self, execution_id: str, task: WorkflowTask) -> TaskResult:
        """Execute a single task"""
        start_time = datetime.now()
        task.started_at = start_time
        task.status = TaskStatus.RUNNING
        
        try:
            # Get handler
            if isinstance(task.handler, str):
                handler = self._task_handlers.get(task.handler)
                if not handler:
                    raise ValueError(f"Unknown task handler: {task.handler}")
            else:
                handler = task.handler
            
            # Execute with timeout
            if task.timeout:
                output = await asyncio.wait_for(
                    handler(execution_id, task),
                    timeout=task.timeout
                )
            else:
                output = await handler(execution_id, task)
            
            # Create successful result
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                output=output,
                execution_time=execution_time,
                metadata={
                    "task_name": task.name,
                    "task_type": task.task_type.value
                }
            )
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Update statistics
            self._update_task_stats(True, execution_time)
            
            # Notify progress
            execution = self._active_executions[execution_id]
            await self.sse_manager.broadcast_workflow_update(
                execution_id,
                {
                    "current_task": task.name,
                    "task_status": "completed",
                    "progress": len(execution.completed_tasks) + 1,
                    "total_tasks": len(self._workflow_definitions[execution.workflow_id].tasks)
                },
                execution.session_id
            )
            
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Task {task.name} timed out after {task.timeout} seconds"
            self.logger.error(error_msg)
            
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            task.status = TaskStatus.FAILED
            task.result = result
            
            return result
            
        except Exception as e:
            error_msg = f"Task {task.name} failed: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                self.logger.info(f"Retrying task {task.name} (attempt {task.retry_count}/{task.max_retries})")
                
                # Wait before retry
                await asyncio.sleep(min(2 ** task.retry_count, 30))  # Exponential backoff
                
                return await self._execute_single_task(execution_id, task)
            
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=error_msg,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            task.status = TaskStatus.FAILED
            task.result = result
            
            # Update statistics
            self._update_task_stats(False, result.execution_time)
            
            return result
    
    # Default task handlers
    async def _handle_llm_generation(self, execution_id: str, task: WorkflowTask) -> Any:
        """Handle LLM generation task"""
        execution = self._active_executions[execution_id]
        
        # Extract parameters
        prompt = task.parameters.get("prompt", "")
        system_message = task.parameters.get("system_message")
        provider = task.parameters.get("provider", "default")
        stream = task.parameters.get("stream", False)
        
        # Build messages
        messages = []
        if system_message:
            messages.append(ChatMessage(MessageRole.SYSTEM, system_message))
        
        # Add context from execution
        context_prompt = task.parameters.get("use_context", False)
        if context_prompt and execution.context:
            context_str = json.dumps(execution.context, indent=2)
            prompt = f"Context:\n{context_str}\n\nTask:\n{prompt}"
        
        messages.append(ChatMessage(MessageRole.USER, prompt))
        
        # Generate response
        response = await self.llm_service.chat_completion(
            messages=messages,
            provider_name=provider,
            stream=stream,
            **task.parameters.get("llm_params", {})
        )
        
        if stream:
            # Collect streaming response
            content = ""
            async for chunk in response:
                content += chunk
            return content
        else:
            return response.content
    
    async def _handle_research_task(self, execution_id: str, task: WorkflowTask) -> Any:
        """Handle research task"""
        # This would integrate with research agents
        query = task.parameters.get("query", "")
        sources = task.parameters.get("sources", [])
        
        # Placeholder for research implementation
        return {
            "query": query,
            "sources": sources,
            "results": "Research results would be here",
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_analysis_task(self, execution_id: str, task: WorkflowTask) -> Any:
        """Handle analysis task"""
        execution = self._active_executions[execution_id]
        
        # Get data to analyze
        data_source = task.parameters.get("data_source")
        analysis_type = task.parameters.get("analysis_type", "general")
        
        if data_source and data_source in execution.task_results:
            data = execution.task_results[data_source].output
        else:
            data = task.parameters.get("data")
        
        # Perform analysis using LLM
        analysis_prompt = f"""
        Perform {analysis_type} analysis on the following data:
        
        {json.dumps(data, indent=2) if data else 'No data provided'}
        
        Provide detailed insights, patterns, and recommendations.
        """
        
        messages = [ChatMessage(MessageRole.USER, analysis_prompt)]
        response = await self.llm_service.chat_completion(messages=messages)
        
        return {
            "analysis_type": analysis_type,
            "insights": response.content,
            "data_summary": str(data)[:500] if data else None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_synthesis_task(self, execution_id: str, task: WorkflowTask) -> Any:
        """Handle synthesis task"""
        execution = self._active_executions[execution_id]
        
        # Get source tasks
        source_tasks = task.parameters.get("source_tasks", [])
        synthesis_type = task.parameters.get("synthesis_type", "summary")
        
        # Collect outputs from source tasks
        source_outputs = []
        for task_id in source_tasks:
            if task_id in execution.task_results:
                source_outputs.append(execution.task_results[task_id].output)
        
        # Synthesize using LLM
        synthesis_prompt = f"""
        Synthesize the following outputs into a coherent {synthesis_type}:
        
        {json.dumps(source_outputs, indent=2)}
        
        Provide a comprehensive synthesis that combines key insights.
        """
        
        messages = [ChatMessage(MessageRole.USER, synthesis_prompt)]
        response = await self.llm_service.chat_completion(messages=messages)
        
        return {
            "synthesis_type": synthesis_type,
            "result": response.content,
            "source_count": len(source_outputs),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_validation_task(self, execution_id: str, task: WorkflowTask) -> Any:
        """Handle validation task"""
        execution = self._active_executions[execution_id]
        
        # Get data to validate
        target_task = task.parameters.get("target_task")
        validation_criteria = task.parameters.get("criteria", [])
        
        if target_task and target_task in execution.task_results:
            data = execution.task_results[target_task].output
        else:
            data = task.parameters.get("data")
        
        # Validate using LLM
        validation_prompt = f"""
        Validate the following data against these criteria:
        Criteria: {json.dumps(validation_criteria, indent=2)}
        
        Data to validate:
        {json.dumps(data, indent=2) if data else 'No data provided'}
        
        Provide validation results with pass/fail status and explanations.
        """
        
        messages = [ChatMessage(MessageRole.USER, validation_prompt)]
        response = await self.llm_service.chat_completion(messages=messages)
        
        return {
            "validation_result": response.content,
            "criteria_count": len(validation_criteria),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _handle_parallel_group(self, execution_id: str, task: WorkflowTask) -> Any:
        """Handle parallel task group"""
        # This would be handled by the task graph execution
        return {"message": "Parallel group executed", "timestamp": datetime.now().isoformat()}
    
    async def _handle_conditional_task(self, execution_id: str, task: WorkflowTask) -> Any:
        """Handle conditional task"""
        execution = self._active_executions[execution_id]
        
        # Evaluate condition
        condition = task.parameters.get("condition")
        condition_task = task.parameters.get("condition_task")
        
        if condition_task and condition_task in execution.task_results:
            condition_data = execution.task_results[condition_task].output
            # Simple condition evaluation (could be enhanced)
            condition_met = bool(condition_data)
        else:
            condition_met = bool(condition)
        
        return {
            "condition_met": condition_met,
            "condition": condition,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _store_execution_result(self, execution: WorkflowExecution):
        """Store workflow execution result"""
        await self.memory_service.store_memory(
            content=json.dumps({
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "completed_tasks": len(execution.completed_tasks),
                "failed_tasks": len(execution.failed_tasks),
                "execution_time": (execution.completed_at - execution.started_at).total_seconds() if execution.completed_at else None,
                "error": execution.error
            }),
            memory_type=MemoryType.WORKFLOW,
            metadata={
                "execution_id": execution.id,
                "workflow_id": execution.workflow_id,
                "type": "workflow_execution",
                "status": execution.status.value
            }
        )
    
    def _update_workflow_stats(self, success: bool, execution_time: float):
        """Update workflow statistics"""
        self._stats["total_workflows"] += 1
        
        if success:
            self._stats["successful_workflows"] += 1
        else:
            self._stats["failed_workflows"] += 1
        
        # Update average execution time
        total_time = self._stats["average_workflow_time"] * (self._stats["total_workflows"] - 1)
        self._stats["average_workflow_time"] = (total_time + execution_time) / self._stats["total_workflows"]
    
    def _update_task_stats(self, success: bool, execution_time: float):
        """Update task statistics"""
        self._stats["total_tasks"] += 1
        
        if success:
            self._stats["successful_tasks"] += 1
        else:
            self._stats["failed_tasks"] += 1
        
        # Update average task time
        total_time = self._stats["average_task_time"] * (self._stats["total_tasks"] - 1)
        self._stats["average_task_time"] = (total_time + execution_time) / self._stats["total_tasks"]
    
    # Public API methods
    async def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution status"""
        return self._active_executions.get(execution_id)
    
    async def pause_workflow(self, execution_id: str) -> bool:
        """Pause workflow execution"""
        if execution_id in self._active_executions:
            self._active_executions[execution_id].status = WorkflowStatus.PAUSED
            return True
        return False
    
    async def resume_workflow(self, execution_id: str) -> bool:
        """Resume paused workflow"""
        if execution_id in self._active_executions:
            execution = self._active_executions[execution_id]
            if execution.status == WorkflowStatus.PAUSED:
                execution.status = WorkflowStatus.RUNNING
                return True
        return False
    
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        if execution_id in self._active_executions:
            self._active_executions[execution_id].status = WorkflowStatus.CANCELLED
            
            # Cancel running task
            if execution_id in self._running_tasks:
                self._running_tasks[execution_id].cancel()
            
            return True
        return False
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows"""
        return [
            {
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "task_count": len(wf.tasks),
                "created_at": wf.created_at.isoformat()
            }
            for wf in self._workflow_definitions.values()
        ]
    
    async def list_active_executions(self) -> List[Dict[str, Any]]:
        """List active workflow executions"""
        return [
            {
                "id": ex.id,
                "workflow_id": ex.workflow_id,
                "status": ex.status.value,
                "current_task": ex.current_task,
                "progress": len(ex.completed_tasks),
                "started_at": ex.started_at.isoformat()
            }
            for ex in self._active_executions.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            **self._stats,
            "registered_workflows": len(self._workflow_definitions),
            "active_executions": len(self._active_executions),
            "execution_history": len(self._execution_history),
            "registered_handlers": len(self._task_handlers)
        }
    
    async def register_task_handler(self, name: str, handler: Callable):
        """Register custom task handler"""
        self._task_handlers[name] = handler
        self.logger.info(f"Registered task handler: {name}")
    
    async def cleanup_old_executions(self, max_age_hours: int = 24):
        """Clean up old execution history"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        original_count = len(self._execution_history)
        self._execution_history = [
            ex for ex in self._execution_history
            if ex.completed_at and ex.completed_at > cutoff_time
        ]
        
        cleaned_count = original_count - len(self._execution_history)
        self.logger.info(f"Cleaned up {cleaned_count} old workflow executions")
        return cleaned_count