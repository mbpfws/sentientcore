"""Execution Manager for Parallel and Sequential Task Execution

This module provides advanced task execution capabilities including:
- Parallel task execution with configurable limits
- Sequential task execution with dependency management
- Mixed execution modes within workflows
- Resource management and throttling
- Task scheduling and prioritization
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
from abc import ABC, abstractmethod

from .advanced_engine import (
    WorkflowTask, 
    TaskStatus, 
    ExecutionMode,
    WorkflowExecution
)


class ExecutionStrategy(Enum):
    """Execution strategies for task management."""
    PARALLEL = "parallel"  # Execute all tasks in parallel
    SEQUENTIAL = "sequential"  # Execute tasks one by one
    MIXED = "mixed"  # Mix of parallel and sequential based on dependencies
    PRIORITY_BASED = "priority_based"  # Execute based on task priority
    RESOURCE_AWARE = "resource_aware"  # Execute based on resource availability
    ADAPTIVE = "adaptive"  # Dynamically adapt execution strategy


class ResourceType(Enum):
    """Types of resources that can be managed."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK_IO = "disk_io"
    CUSTOM = "custom"


@dataclass
class ResourceLimit:
    """Resource limit configuration."""
    resource_type: ResourceType
    max_value: float
    current_usage: float = 0.0
    unit: str = "count"
    
    def is_available(self, required: float) -> bool:
        """Check if resource is available for the required amount."""
        return (self.current_usage + required) <= self.max_value
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource if available."""
        if self.is_available(amount):
            self.current_usage += amount
            return True
        return False
    
    def release(self, amount: float):
        """Release allocated resource."""
        self.current_usage = max(0, self.current_usage - amount)


@dataclass
class TaskExecution:
    """Represents a task execution instance."""
    task: WorkflowTask
    execution_id: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get execution duration."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None
    
    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed (success or failure)."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]


@dataclass
class ExecutionConfig:
    """Configuration for task execution."""
    max_parallel_tasks: int = 5
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    task_timeout: float = 300.0  # seconds
    resource_limits: Dict[ResourceType, ResourceLimit] = field(default_factory=dict)
    execution_strategy: ExecutionStrategy = ExecutionStrategy.MIXED
    priority_weights: Dict[str, float] = field(default_factory=dict)
    enable_task_batching: bool = False
    batch_size: int = 10
    adaptive_scaling: bool = True
    performance_monitoring: bool = True


class TaskExecutor(ABC):
    """Abstract base class for task executors."""
    
    @abstractmethod
    async def execute_task(
        self,
        task_execution: TaskExecution,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a single task."""
        pass
    
    @abstractmethod
    async def can_execute(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> bool:
        """Check if this executor can handle the task."""
        pass


class DefaultTaskExecutor(TaskExecutor):
    """Default task executor implementation."""
    
    async def execute_task(
        self,
        task_execution: TaskExecution,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a task using the task's function."""
        task = task_execution.task
        
        if task.function:
            if asyncio.iscoroutinefunction(task.function):
                return await task.function(**task.parameters, **context)
            else:
                return task.function(**task.parameters, **context)
        else:
            # Mock execution for tasks without functions
            await asyncio.sleep(0.1)  # Simulate work
            return f"Task {task.task_id} completed"
    
    async def can_execute(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> bool:
        """Default executor can handle any task."""
        return True


class ParallelExecutionManager:
    """Manages parallel execution of tasks with resource limits."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.running_tasks: Dict[str, TaskExecution] = {}
        self.completed_tasks: Dict[str, TaskExecution] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.resource_manager = ResourceManager(config.resource_limits)
        self.executors: List[TaskExecutor] = [DefaultTaskExecutor()]
        self.performance_metrics: Dict[str, Any] = {}
        self._execution_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
    async def initialize(self):
        """Initialize the execution manager."""
        await self.resource_manager.initialize()
        
        # Start worker tasks
        self.workers = []
        for i in range(self.config.max_parallel_tasks):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
    
    async def shutdown(self):
        """Shutdown the execution manager."""
        self._shutdown_event.set()
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Cancel running tasks
        for task_execution in self.running_tasks.values():
            if hasattr(task_execution, '_task_handle'):
                task_execution._task_handle.cancel()
    
    async def execute_tasks(
        self,
        tasks: List[WorkflowTask],
        execution_context: Dict[str, Any],
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    ) -> Dict[str, TaskExecution]:
        """Execute a list of tasks according to the specified mode."""
        
        if execution_mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_sequential(tasks, execution_context)
        elif execution_mode == ExecutionMode.PARALLEL:
            return await self._execute_parallel(tasks, execution_context)
        elif execution_mode == ExecutionMode.HUMAN_IN_LOOP:
            return await self._execute_human_in_loop(tasks, execution_context)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")
    
    async def _execute_sequential(
        self,
        tasks: List[WorkflowTask],
        execution_context: Dict[str, Any]
    ) -> Dict[str, TaskExecution]:
        """Execute tasks sequentially."""
        
        results = {}
        
        for task in tasks:
            task_execution = TaskExecution(
                task=task,
                execution_id=f"{task.task_id}_{int(time.time())}",
                execution_context=execution_context.copy()
            )
            
            # Wait for dependencies
            await self._wait_for_dependencies(task, results)
            
            # Execute task
            await self._execute_single_task(task_execution, execution_context)
            results[task.task_id] = task_execution
            
            # Update context with results
            if task_execution.result:
                execution_context[f"task_{task.task_id}_result"] = task_execution.result
        
        return results
    
    async def _execute_parallel(
        self,
        tasks: List[WorkflowTask],
        execution_context: Dict[str, Any]
    ) -> Dict[str, TaskExecution]:
        """Execute tasks in parallel with dependency management."""
        
        # Create task executions
        task_executions = {}
        for task in tasks:
            task_execution = TaskExecution(
                task=task,
                execution_id=f"{task.task_id}_{int(time.time())}",
                execution_context=execution_context.copy()
            )
            task_executions[task.task_id] = task_execution
        
        # Submit tasks to queue
        for task_execution in task_executions.values():
            await self.task_queue.put(task_execution)
        
        # Wait for all tasks to complete
        while task_executions:
            await asyncio.sleep(0.1)
            
            # Check for completed tasks
            completed = []
            for task_id, task_execution in task_executions.items():
                if task_execution.is_completed:
                    completed.append(task_id)
                    
                    # Update context with results
                    if task_execution.result:
                        execution_context[f"task_{task_id}_result"] = task_execution.result
            
            # Remove completed tasks
            for task_id in completed:
                self.completed_tasks[task_id] = task_executions.pop(task_id)
        
        return self.completed_tasks
    
    async def _execute_human_in_loop(
        self,
        tasks: List[WorkflowTask],
        execution_context: Dict[str, Any]
    ) -> Dict[str, TaskExecution]:
        """Execute tasks with human intervention points."""
        
        results = {}
        
        for task in tasks:
            task_execution = TaskExecution(
                task=task,
                execution_id=f"{task.task_id}_{int(time.time())}",
                execution_context=execution_context.copy()
            )
            
            # Check if human intervention is required
            if task.requires_human_input:
                # Wait for human input
                await self._wait_for_human_input(task_execution)
            
            # Execute task
            await self._execute_single_task(task_execution, execution_context)
            results[task.task_id] = task_execution
            
            # Update context
            if task_execution.result:
                execution_context[f"task_{task.task_id}_result"] = task_execution.result
        
        return results
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing tasks."""
        
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task_execution = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Check dependencies
                if not await self._check_dependencies(task_execution.task):
                    # Put back in queue and wait
                    await self.task_queue.put(task_execution)
                    await asyncio.sleep(0.5)
                    continue
                
                # Check resource availability
                if not await self.resource_manager.can_allocate_resources(
                    task_execution.task.resource_requirements or {}
                ):
                    # Put back in queue and wait
                    await self.task_queue.put(task_execution)
                    await asyncio.sleep(0.5)
                    continue
                
                # Execute task
                async with self._execution_lock:
                    self.running_tasks[task_execution.task.task_id] = task_execution
                
                await self._execute_single_task(
                    task_execution,
                    task_execution.execution_context
                )
                
                # Move to completed
                async with self._execution_lock:
                    self.running_tasks.pop(task_execution.task.task_id, None)
                    self.completed_tasks[task_execution.task.task_id] = task_execution
                
                # Mark queue task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_single_task(
        self,
        task_execution: TaskExecution,
        execution_context: Dict[str, Any]
    ):
        """Execute a single task with error handling and retries."""
        
        task = task_execution.task
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Allocate resources
                await self.resource_manager.allocate_resources(
                    task.resource_requirements or {}
                )
                
                # Update status
                task_execution.status = TaskStatus.RUNNING
                task_execution.started_at = datetime.now()
                task_execution.retry_count = attempt
                
                # Find appropriate executor
                executor = await self._find_executor(task, execution_context)
                
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        executor.execute_task(task_execution, execution_context),
                        timeout=self.config.task_timeout
                    )
                    
                    # Success
                    task_execution.result = result
                    task_execution.status = TaskStatus.COMPLETED
                    task_execution.completed_at = datetime.now()
                    
                    # Record performance metrics
                    if self.config.performance_monitoring:
                        await self._record_performance_metrics(task_execution)
                    
                    break
                    
                except asyncio.TimeoutError:
                    raise Exception(f"Task {task.task_id} timed out after {self.config.task_timeout} seconds")
                
            except Exception as e:
                task_execution.error = e
                
                if attempt < max_retries:
                    # Wait before retry
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                else:
                    # Final failure
                    task_execution.status = TaskStatus.FAILED
                    task_execution.completed_at = datetime.now()
                    break
            
            finally:
                # Release resources
                await self.resource_manager.release_resources(
                    task.resource_requirements or {}
                )
    
    async def _find_executor(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> TaskExecutor:
        """Find appropriate executor for the task."""
        
        for executor in self.executors:
            if await executor.can_execute(task, context):
                return executor
        
        # Fallback to default executor
        return self.executors[0]
    
    async def _check_dependencies(self, task: WorkflowTask) -> bool:
        """Check if task dependencies are satisfied."""
        
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
            
            dep_execution = self.completed_tasks[dep_id]
            if dep_execution.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def _wait_for_dependencies(
        self,
        task: WorkflowTask,
        completed_tasks: Dict[str, TaskExecution]
    ):
        """Wait for task dependencies to complete."""
        
        if not task.dependencies:
            return
        
        while True:
            all_satisfied = True
            
            for dep_id in task.dependencies:
                if dep_id not in completed_tasks:
                    all_satisfied = False
                    break
                
                dep_execution = completed_tasks[dep_id]
                if dep_execution.status != TaskStatus.COMPLETED:
                    all_satisfied = False
                    break
            
            if all_satisfied:
                break
            
            await asyncio.sleep(0.1)
    
    async def _wait_for_human_input(self, task_execution: TaskExecution):
        """Wait for human input for a task."""
        
        # This would integrate with the human-in-the-loop system
        # For now, simulate waiting
        print(f"Waiting for human input for task: {task_execution.task.task_id}")
        
        # In a real implementation, this would:
        # 1. Send notification to human operator
        # 2. Wait for input/approval
        # 3. Update task execution context with human input
        
        await asyncio.sleep(1.0)  # Simulate human response time
    
    async def _record_performance_metrics(self, task_execution: TaskExecution):
        """Record performance metrics for the task execution."""
        
        task_id = task_execution.task.task_id
        duration = task_execution.duration
        
        if task_id not in self.performance_metrics:
            self.performance_metrics[task_id] = {
                "executions": 0,
                "total_duration": timedelta(0),
                "avg_duration": timedelta(0),
                "success_rate": 0.0,
                "retry_rate": 0.0
            }
        
        metrics = self.performance_metrics[task_id]
        metrics["executions"] += 1
        
        if duration:
            metrics["total_duration"] += duration
            metrics["avg_duration"] = metrics["total_duration"] / metrics["executions"]
        
        # Update success rate
        if task_execution.status == TaskStatus.COMPLETED:
            success_count = metrics.get("success_count", 0) + 1
            metrics["success_count"] = success_count
            metrics["success_rate"] = success_count / metrics["executions"]
        
        # Update retry rate
        if task_execution.retry_count > 0:
            retry_count = metrics.get("retry_count", 0) + 1
            metrics["retry_count"] = retry_count
            metrics["retry_rate"] = retry_count / metrics["executions"]
    
    def add_executor(self, executor: TaskExecutor):
        """Add a custom task executor."""
        self.executors.append(executor)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all tasks."""
        return self.performance_metrics.copy()
    
    def get_running_tasks(self) -> Dict[str, TaskExecution]:
        """Get currently running tasks."""
        return self.running_tasks.copy()
    
    def get_completed_tasks(self) -> Dict[str, TaskExecution]:
        """Get completed tasks."""
        return self.completed_tasks.copy()


class ResourceManager:
    """Manages resource allocation and limits."""
    
    def __init__(self, resource_limits: Dict[ResourceType, ResourceLimit]):
        self.resource_limits = resource_limits.copy()
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the resource manager."""
        # Set default limits if not provided
        if ResourceType.CPU not in self.resource_limits:
            self.resource_limits[ResourceType.CPU] = ResourceLimit(
                ResourceType.CPU, max_value=4.0, unit="cores"
            )
        
        if ResourceType.MEMORY not in self.resource_limits:
            self.resource_limits[ResourceType.MEMORY] = ResourceLimit(
                ResourceType.MEMORY, max_value=8192.0, unit="MB"
            )
    
    async def can_allocate_resources(
        self,
        requirements: Dict[ResourceType, float]
    ) -> bool:
        """Check if resources can be allocated."""
        
        async with self._lock:
            for resource_type, required_amount in requirements.items():
                if resource_type in self.resource_limits:
                    limit = self.resource_limits[resource_type]
                    if not limit.is_available(required_amount):
                        return False
        
        return True
    
    async def allocate_resources(
        self,
        requirements: Dict[ResourceType, float]
    ) -> bool:
        """Allocate resources if available."""
        
        async with self._lock:
            # Check availability first
            for resource_type, required_amount in requirements.items():
                if resource_type in self.resource_limits:
                    limit = self.resource_limits[resource_type]
                    if not limit.is_available(required_amount):
                        return False
            
            # Allocate all resources
            for resource_type, required_amount in requirements.items():
                if resource_type in self.resource_limits:
                    limit = self.resource_limits[resource_type]
                    limit.allocate(required_amount)
        
        return True
    
    async def release_resources(
        self,
        requirements: Dict[ResourceType, float]
    ):
        """Release allocated resources."""
        
        async with self._lock:
            for resource_type, amount in requirements.items():
                if resource_type in self.resource_limits:
                    limit = self.resource_limits[resource_type]
                    limit.release(amount)
    
    def get_resource_usage(self) -> Dict[ResourceType, Dict[str, float]]:
        """Get current resource usage."""
        
        usage = {}
        for resource_type, limit in self.resource_limits.items():
            usage[resource_type] = {
                "current": limit.current_usage,
                "max": limit.max_value,
                "available": limit.max_value - limit.current_usage,
                "utilization": (limit.current_usage / limit.max_value) * 100
            }
        
        return usage


class SequentialExecutionManager:
    """Manages sequential execution of tasks with dependency handling."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.execution_history: List[TaskExecution] = []
        self.resource_manager = ResourceManager(config.resource_limits)
        self.executors: List[TaskExecutor] = [DefaultTaskExecutor()]
    
    async def initialize(self):
        """Initialize the sequential execution manager."""
        await self.resource_manager.initialize()
    
    async def execute_tasks(
        self,
        tasks: List[WorkflowTask],
        execution_context: Dict[str, Any]
    ) -> Dict[str, TaskExecution]:
        """Execute tasks sequentially with dependency resolution."""
        
        # Sort tasks by dependencies and priority
        sorted_tasks = await self._sort_tasks_by_dependencies(tasks)
        
        results = {}
        
        for task in sorted_tasks:
            task_execution = TaskExecution(
                task=task,
                execution_id=f"{task.task_id}_{int(time.time())}",
                execution_context=execution_context.copy()
            )
            
            # Execute task
            await self._execute_single_task(task_execution, execution_context)
            
            # Store result
            results[task.task_id] = task_execution
            self.execution_history.append(task_execution)
            
            # Update context with results
            if task_execution.result:
                execution_context[f"task_{task.task_id}_result"] = task_execution.result
        
        return results
    
    async def _sort_tasks_by_dependencies(
        self,
        tasks: List[WorkflowTask]
    ) -> List[WorkflowTask]:
        """Sort tasks by dependencies using topological sort."""
        
        # Create dependency graph
        task_map = {task.task_id: task for task in tasks}
        in_degree = {task.task_id: 0 for task in tasks}
        
        # Calculate in-degrees
        for task in tasks:
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id in in_degree:
                        in_degree[task.task_id] += 1
        
        # Topological sort
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []
        
        while queue:
            # Sort by priority if available
            queue.sort(key=lambda tid: task_map[tid].priority or 0, reverse=True)
            
            current_id = queue.pop(0)
            current_task = task_map[current_id]
            sorted_tasks.append(current_task)
            
            # Update in-degrees for dependent tasks
            for task in tasks:
                if task.dependencies and current_id in task.dependencies:
                    in_degree[task.task_id] -= 1
                    if in_degree[task.task_id] == 0:
                        queue.append(task.task_id)
        
        # Check for circular dependencies
        if len(sorted_tasks) != len(tasks):
            remaining = [task.task_id for task in tasks if task not in sorted_tasks]
            raise ValueError(f"Circular dependencies detected in tasks: {remaining}")
        
        return sorted_tasks
    
    async def _execute_single_task(
        self,
        task_execution: TaskExecution,
        execution_context: Dict[str, Any]
    ):
        """Execute a single task with error handling."""
        
        task = task_execution.task
        max_retries = self.config.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                # Allocate resources
                await self.resource_manager.allocate_resources(
                    task.resource_requirements or {}
                )
                
                # Update status
                task_execution.status = TaskStatus.RUNNING
                task_execution.started_at = datetime.now()
                task_execution.retry_count = attempt
                
                # Find executor
                executor = await self._find_executor(task, execution_context)
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    executor.execute_task(task_execution, execution_context),
                    timeout=self.config.task_timeout
                )
                
                # Success
                task_execution.result = result
                task_execution.status = TaskStatus.COMPLETED
                task_execution.completed_at = datetime.now()
                break
                
            except Exception as e:
                task_execution.error = e
                
                if attempt < max_retries:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    continue
                else:
                    task_execution.status = TaskStatus.FAILED
                    task_execution.completed_at = datetime.now()
                    break
            
            finally:
                # Release resources
                await self.resource_manager.release_resources(
                    task.resource_requirements or {}
                )
    
    async def _find_executor(
        self,
        task: WorkflowTask,
        context: Dict[str, Any]
    ) -> TaskExecutor:
        """Find appropriate executor for the task."""
        
        for executor in self.executors:
            if await executor.can_execute(task, context):
                return executor
        
        return self.executors[0]
    
    def add_executor(self, executor: TaskExecutor):
        """Add a custom task executor."""
        self.executors.append(executor)
    
    def get_execution_history(self) -> List[TaskExecution]:
        """Get execution history."""
        return self.execution_history.copy()


class MixedExecutionManager:
    """Manages mixed execution modes combining parallel and sequential execution."""
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.parallel_manager = ParallelExecutionManager(config)
        self.sequential_manager = SequentialExecutionManager(config)
    
    async def initialize(self):
        """Initialize both execution managers."""
        await self.parallel_manager.initialize()
        await self.sequential_manager.initialize()
    
    async def shutdown(self):
        """Shutdown execution managers."""
        await self.parallel_manager.shutdown()
    
    async def execute_tasks(
        self,
        tasks: List[WorkflowTask],
        execution_context: Dict[str, Any]
    ) -> Dict[str, TaskExecution]:
        """Execute tasks using mixed execution strategy."""
        
        # Analyze task dependencies and group them
        task_groups = await self._analyze_and_group_tasks(tasks)
        
        all_results = {}
        
        for group in task_groups:
            if group["execution_mode"] == "parallel":
                results = await self.parallel_manager.execute_tasks(
                    group["tasks"],
                    execution_context,
                    ExecutionMode.PARALLEL
                )
            else:
                results = await self.sequential_manager.execute_tasks(
                    group["tasks"],
                    execution_context
                )
            
            all_results.update(results)
            
            # Update context with group results
            for task_id, task_execution in results.items():
                if task_execution.result:
                    execution_context[f"task_{task_id}_result"] = task_execution.result
        
        return all_results
    
    async def _analyze_and_group_tasks(
        self,
        tasks: List[WorkflowTask]
    ) -> List[Dict[str, Any]]:
        """Analyze tasks and group them for optimal execution."""
        
        # Build dependency graph
        task_map = {task.task_id: task for task in tasks}
        dependency_graph = {}
        
        for task in tasks:
            dependency_graph[task.task_id] = task.dependencies or []
        
        # Find independent task groups
        groups = []
        processed = set()
        
        # Find tasks with no dependencies (can run in parallel)
        independent_tasks = [
            task for task in tasks 
            if not task.dependencies and task.task_id not in processed
        ]
        
        if independent_tasks:
            groups.append({
                "execution_mode": "parallel",
                "tasks": independent_tasks
            })
            processed.update(task.task_id for task in independent_tasks)
        
        # Process remaining tasks in dependency order
        while len(processed) < len(tasks):
            # Find tasks whose dependencies are satisfied
            ready_tasks = []
            for task in tasks:
                if task.task_id not in processed:
                    if all(dep_id in processed for dep_id in (task.dependencies or [])):
                        ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or other issue
                remaining = [task.task_id for task in tasks if task.task_id not in processed]
                raise ValueError(f"Cannot resolve dependencies for tasks: {remaining}")
            
            # Group ready tasks by execution preference
            parallel_tasks = []
            sequential_tasks = []
            
            for task in ready_tasks:
                if getattr(task, 'prefer_parallel', True) and len(ready_tasks) > 1:
                    parallel_tasks.append(task)
                else:
                    sequential_tasks.append(task)
            
            # Add parallel group if any
            if parallel_tasks:
                groups.append({
                    "execution_mode": "parallel",
                    "tasks": parallel_tasks
                })
                processed.update(task.task_id for task in parallel_tasks)
            
            # Add sequential group if any
            if sequential_tasks:
                groups.append({
                    "execution_mode": "sequential",
                    "tasks": sequential_tasks
                })
                processed.update(task.task_id for task in sequential_tasks)
        
        return groups
    
    def add_executor(self, executor: TaskExecutor):
        """Add executor to both managers."""
        self.parallel_manager.add_executor(executor)
        self.sequential_manager.add_executor(executor)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get combined performance metrics."""
        return {
            "parallel": self.parallel_manager.get_performance_metrics(),
            "sequential": self.sequential_manager.get_execution_history()
        }