"""Unit tests for the Execution Manager module.

Tests cover:
- Parallel task execution with resource management
- Sequential task execution with dependency resolution
- Mixed execution modes
- Resource allocation and limits
- Task retry mechanisms
- Performance monitoring
- Error handling and recovery
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List

from core.workflows.execution_manager import (
    ExecutionStrategy,
    ResourceType,
    ResourceLimit,
    TaskExecution,
    ExecutionConfig,
    TaskExecutor,
    DefaultTaskExecutor,
    ParallelExecutionManager,
    SequentialExecutionManager,
    MixedExecutionManager,
    ResourceManager
)
from core.workflows.advanced_engine import (
    WorkflowTask,
    TaskStatus,
    ExecutionMode
)


class TestResourceLimit:
    """Test ResourceLimit functionality."""
    
    def test_resource_limit_creation(self):
        """Test creating a resource limit."""
        limit = ResourceLimit(
            resource_type=ResourceType.CPU,
            max_value=4.0,
            unit="cores"
        )
        
        assert limit.resource_type == ResourceType.CPU
        assert limit.max_value == 4.0
        assert limit.current_usage == 0.0
        assert limit.unit == "cores"
    
    def test_resource_availability_check(self):
        """Test checking resource availability."""
        limit = ResourceLimit(ResourceType.MEMORY, max_value=1000.0)
        
        # Should be available
        assert limit.is_available(500.0) is True
        assert limit.is_available(1000.0) is True
        
        # Should not be available
        assert limit.is_available(1001.0) is False
    
    def test_resource_allocation(self):
        """Test resource allocation and release."""
        limit = ResourceLimit(ResourceType.CPU, max_value=4.0)
        
        # Allocate resources
        assert limit.allocate(2.0) is True
        assert limit.current_usage == 2.0
        
        # Try to allocate more than available
        assert limit.allocate(3.0) is False
        assert limit.current_usage == 2.0
        
        # Allocate remaining
        assert limit.allocate(2.0) is True
        assert limit.current_usage == 4.0
        
        # Release resources
        limit.release(1.0)
        assert limit.current_usage == 3.0
        
        # Release more than allocated (should not go negative)
        limit.release(5.0)
        assert limit.current_usage == 0.0


class TestTaskExecution:
    """Test TaskExecution functionality."""
    
    def test_task_execution_creation(self):
        """Test creating a task execution."""
        task = WorkflowTask(
            task_id="test_task",
            name="Test Task",
            description="A test task"
        )
        
        execution = TaskExecution(
            task=task,
            execution_id="exec_1"
        )
        
        assert execution.task == task
        assert execution.execution_id == "exec_1"
        assert execution.status == TaskStatus.PENDING
        assert execution.retry_count == 0
    
    def test_task_execution_duration(self):
        """Test duration calculation."""
        task = WorkflowTask(task_id="test", name="Test")
        execution = TaskExecution(task=task, execution_id="exec_1")
        
        # No duration when not started
        assert execution.duration is None
        
        # Set start time
        start_time = datetime.now()
        execution.started_at = start_time
        assert execution.duration is None
        
        # Set completion time
        execution.completed_at = start_time + timedelta(seconds=5)
        assert execution.duration == timedelta(seconds=5)
    
    def test_task_execution_status_properties(self):
        """Test status property methods."""
        task = WorkflowTask(task_id="test", name="Test")
        execution = TaskExecution(task=task, execution_id="exec_1")
        
        # Initially pending
        assert not execution.is_running
        assert not execution.is_completed
        
        # Running
        execution.status = TaskStatus.RUNNING
        assert execution.is_running
        assert not execution.is_completed
        
        # Completed
        execution.status = TaskStatus.COMPLETED
        assert not execution.is_running
        assert execution.is_completed
        
        # Failed
        execution.status = TaskStatus.FAILED
        assert not execution.is_running
        assert execution.is_completed


class TestDefaultTaskExecutor:
    """Test DefaultTaskExecutor functionality."""
    
    @pytest.mark.asyncio
    async def test_can_execute(self):
        """Test executor capability check."""
        executor = DefaultTaskExecutor()
        task = WorkflowTask(task_id="test", name="Test")
        
        result = await executor.can_execute(task, {})
        assert result is True
    
    @pytest.mark.asyncio
    async def test_execute_task_with_function(self):
        """Test executing task with function."""
        def test_function(param1, param2):
            return param1 + param2
        
        task = WorkflowTask(
            task_id="test",
            name="Test",
            function=test_function,
            parameters={"param1": 5, "param2": 3}
        )
        
        execution = TaskExecution(task=task, execution_id="exec_1")
        executor = DefaultTaskExecutor()
        
        result = await executor.execute_task(execution, {})
        assert result == 8
    
    @pytest.mark.asyncio
    async def test_execute_task_with_async_function(self):
        """Test executing task with async function."""
        async def async_test_function(param1, param2):
            await asyncio.sleep(0.01)
            return param1 * param2
        
        task = WorkflowTask(
            task_id="test",
            name="Test",
            function=async_test_function,
            parameters={"param1": 4, "param2": 3}
        )
        
        execution = TaskExecution(task=task, execution_id="exec_1")
        executor = DefaultTaskExecutor()
        
        result = await executor.execute_task(execution, {})
        assert result == 12
    
    @pytest.mark.asyncio
    async def test_execute_task_without_function(self):
        """Test executing task without function (mock execution)."""
        task = WorkflowTask(task_id="test", name="Test")
        execution = TaskExecution(task=task, execution_id="exec_1")
        executor = DefaultTaskExecutor()
        
        result = await executor.execute_task(execution, {})
        assert result == "Task test completed"


class TestResourceManager:
    """Test ResourceManager functionality."""
    
    @pytest.mark.asyncio
    async def test_resource_manager_initialization(self):
        """Test resource manager initialization."""
        limits = {
            ResourceType.CPU: ResourceLimit(ResourceType.CPU, 4.0),
            ResourceType.MEMORY: ResourceLimit(ResourceType.MEMORY, 1000.0)
        }
        
        manager = ResourceManager(limits)
        await manager.initialize()
        
        assert ResourceType.CPU in manager.resource_limits
        assert ResourceType.MEMORY in manager.resource_limits
    
    @pytest.mark.asyncio
    async def test_resource_manager_default_limits(self):
        """Test resource manager with default limits."""
        manager = ResourceManager({})
        await manager.initialize()
        
        # Should have default CPU and memory limits
        assert ResourceType.CPU in manager.resource_limits
        assert ResourceType.MEMORY in manager.resource_limits
        assert manager.resource_limits[ResourceType.CPU].max_value == 4.0
        assert manager.resource_limits[ResourceType.MEMORY].max_value == 8192.0
    
    @pytest.mark.asyncio
    async def test_resource_allocation(self):
        """Test resource allocation and release."""
        limits = {
            ResourceType.CPU: ResourceLimit(ResourceType.CPU, 4.0),
            ResourceType.MEMORY: ResourceLimit(ResourceType.MEMORY, 1000.0)
        }
        
        manager = ResourceManager(limits)
        await manager.initialize()
        
        requirements = {
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 500.0
        }
        
        # Check availability
        can_allocate = await manager.can_allocate_resources(requirements)
        assert can_allocate is True
        
        # Allocate resources
        allocated = await manager.allocate_resources(requirements)
        assert allocated is True
        
        # Check usage
        usage = manager.get_resource_usage()
        assert usage[ResourceType.CPU]["current"] == 2.0
        assert usage[ResourceType.MEMORY]["current"] == 500.0
        
        # Try to allocate more than available
        large_requirements = {
            ResourceType.CPU: 3.0,  # Only 2.0 available
            ResourceType.MEMORY: 600.0  # Only 500.0 available
        }
        
        can_allocate = await manager.can_allocate_resources(large_requirements)
        assert can_allocate is False
        
        allocated = await manager.allocate_resources(large_requirements)
        assert allocated is False
        
        # Release resources
        await manager.release_resources(requirements)
        
        usage = manager.get_resource_usage()
        assert usage[ResourceType.CPU]["current"] == 0.0
        assert usage[ResourceType.MEMORY]["current"] == 0.0


class TestSequentialExecutionManager:
    """Test SequentialExecutionManager functionality."""
    
    @pytest.mark.asyncio
    async def test_sequential_execution_simple(self):
        """Test simple sequential execution."""
        config = ExecutionConfig(max_parallel_tasks=1)
        manager = SequentialExecutionManager(config)
        await manager.initialize()
        
        # Create simple tasks
        tasks = [
            WorkflowTask(
                task_id="task1",
                name="Task 1",
                function=lambda: "result1"
            ),
            WorkflowTask(
                task_id="task2",
                name="Task 2",
                function=lambda: "result2"
            )
        ]
        
        results = await manager.execute_tasks(tasks, {})
        
        assert len(results) == 2
        assert "task1" in results
        assert "task2" in results
        assert results["task1"].status == TaskStatus.COMPLETED
        assert results["task2"].status == TaskStatus.COMPLETED
        assert results["task1"].result == "result1"
        assert results["task2"].result == "result2"
    
    @pytest.mark.asyncio
    async def test_sequential_execution_with_dependencies(self):
        """Test sequential execution with dependencies."""
        config = ExecutionConfig()
        manager = SequentialExecutionManager(config)
        await manager.initialize()
        
        execution_order = []
        
        def task1_func():
            execution_order.append("task1")
            return "result1"
        
        def task2_func():
            execution_order.append("task2")
            return "result2"
        
        def task3_func():
            execution_order.append("task3")
            return "result3"
        
        tasks = [
            WorkflowTask(
                task_id="task3",
                name="Task 3",
                function=task3_func,
                dependencies=["task1", "task2"]
            ),
            WorkflowTask(
                task_id="task1",
                name="Task 1",
                function=task1_func
            ),
            WorkflowTask(
                task_id="task2",
                name="Task 2",
                function=task2_func,
                dependencies=["task1"]
            )
        ]
        
        results = await manager.execute_tasks(tasks, {})
        
        # Check execution order
        assert execution_order == ["task1", "task2", "task3"]
        
        # Check results
        assert len(results) == 3
        for task_id in ["task1", "task2", "task3"]:
            assert results[task_id].status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_sequential_execution_circular_dependency(self):
        """Test handling of circular dependencies."""
        config = ExecutionConfig()
        manager = SequentialExecutionManager(config)
        await manager.initialize()
        
        tasks = [
            WorkflowTask(
                task_id="task1",
                name="Task 1",
                dependencies=["task2"]
            ),
            WorkflowTask(
                task_id="task2",
                name="Task 2",
                dependencies=["task1"]
            )
        ]
        
        with pytest.raises(ValueError, match="Circular dependencies"):
            await manager.execute_tasks(tasks, {})


class TestParallelExecutionManager:
    """Test ParallelExecutionManager functionality."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution_simple(self):
        """Test simple parallel execution."""
        config = ExecutionConfig(max_parallel_tasks=3)
        manager = ParallelExecutionManager(config)
        await manager.initialize()
        
        execution_times = {}
        
        async def timed_task(task_id: str, delay: float):
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(delay)
            end = asyncio.get_event_loop().time()
            execution_times[task_id] = (start, end)
            return f"result_{task_id}"
        
        tasks = [
            WorkflowTask(
                task_id="task1",
                name="Task 1",
                function=lambda: timed_task("task1", 0.1)
            ),
            WorkflowTask(
                task_id="task2",
                name="Task 2",
                function=lambda: timed_task("task2", 0.1)
            ),
            WorkflowTask(
                task_id="task3",
                name="Task 3",
                function=lambda: timed_task("task3", 0.1)
            )
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await manager.execute_tasks(tasks, {}, ExecutionMode.PARALLEL)
        end_time = asyncio.get_event_loop().time()
        
        await manager.shutdown()
        
        # Check that tasks ran in parallel (total time should be close to max individual time)
        total_time = end_time - start_time
        assert total_time < 0.3  # Should be much less than 0.3 (3 * 0.1) if parallel
        
        # Check results
        assert len(results) == 3
        for task_id in ["task1", "task2", "task3"]:
            assert results[task_id].status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_parallel_execution_with_resource_limits(self):
        """Test parallel execution with resource limits."""
        resource_limits = {
            ResourceType.CPU: ResourceLimit(ResourceType.CPU, 2.0)
        }
        
        config = ExecutionConfig(
            max_parallel_tasks=5,
            resource_limits=resource_limits
        )
        
        manager = ParallelExecutionManager(config)
        await manager.initialize()
        
        async def cpu_task():
            await asyncio.sleep(0.1)
            return "completed"
        
        tasks = [
            WorkflowTask(
                task_id=f"task{i}",
                name=f"Task {i}",
                function=cpu_task,
                resource_requirements={ResourceType.CPU: 1.0}
            )
            for i in range(3)
        ]
        
        results = await manager.execute_tasks(tasks, {}, ExecutionMode.PARALLEL)
        await manager.shutdown()
        
        # All tasks should complete despite resource limits
        assert len(results) == 3
        for i in range(3):
            assert results[f"task{i}"].status == TaskStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_parallel_execution_task_failure_and_retry(self):
        """Test task failure and retry in parallel execution."""
        config = ExecutionConfig(max_retries=2, retry_delay=0.01)
        manager = ParallelExecutionManager(config)
        await manager.initialize()
        
        call_count = {"task1": 0}
        
        def failing_task():
            call_count["task1"] += 1
            if call_count["task1"] < 3:  # Fail first 2 attempts
                raise Exception("Task failed")
            return "success"
        
        tasks = [
            WorkflowTask(
                task_id="task1",
                name="Failing Task",
                function=failing_task
            )
        ]
        
        results = await manager.execute_tasks(tasks, {}, ExecutionMode.PARALLEL)
        await manager.shutdown()
        
        # Task should succeed after retries
        assert results["task1"].status == TaskStatus.COMPLETED
        assert results["task1"].result == "success"
        assert results["task1"].retry_count == 2
        assert call_count["task1"] == 3


class TestMixedExecutionManager:
    """Test MixedExecutionManager functionality."""
    
    @pytest.mark.asyncio
    async def test_mixed_execution_task_grouping(self):
        """Test task grouping for mixed execution."""
        config = ExecutionConfig(max_parallel_tasks=3)
        manager = MixedExecutionManager(config)
        await manager.initialize()
        
        execution_order = []
        
        def track_execution(task_id: str):
            execution_order.append(task_id)
            return f"result_{task_id}"
        
        tasks = [
            # Independent tasks (should run in parallel)
            WorkflowTask(
                task_id="task1",
                name="Task 1",
                function=lambda: track_execution("task1")
            ),
            WorkflowTask(
                task_id="task2",
                name="Task 2",
                function=lambda: track_execution("task2")
            ),
            # Dependent task (should run after task1)
            WorkflowTask(
                task_id="task3",
                name="Task 3",
                function=lambda: track_execution("task3"),
                dependencies=["task1"]
            ),
            # Another dependent task (should run after task2)
            WorkflowTask(
                task_id="task4",
                name="Task 4",
                function=lambda: track_execution("task4"),
                dependencies=["task2"]
            )
        ]
        
        results = await manager.execute_tasks(tasks, {})
        await manager.shutdown()
        
        # Check that all tasks completed
        assert len(results) == 4
        for task_id in ["task1", "task2", "task3", "task4"]:
            assert results[task_id].status == TaskStatus.COMPLETED
        
        # Check dependency order
        task1_index = execution_order.index("task1")
        task2_index = execution_order.index("task2")
        task3_index = execution_order.index("task3")
        task4_index = execution_order.index("task4")
        
        assert task1_index < task3_index  # task1 before task3
        assert task2_index < task4_index  # task2 before task4
    
    @pytest.mark.asyncio
    async def test_mixed_execution_complex_dependencies(self):
        """Test mixed execution with complex dependency chains."""
        config = ExecutionConfig(max_parallel_tasks=2)
        manager = MixedExecutionManager(config)
        await manager.initialize()
        
        tasks = [
            # Level 0 (independent)
            WorkflowTask(task_id="A", name="Task A", function=lambda: "A"),
            WorkflowTask(task_id="B", name="Task B", function=lambda: "B"),
            
            # Level 1 (depend on level 0)
            WorkflowTask(
                task_id="C",
                name="Task C",
                function=lambda: "C",
                dependencies=["A"]
            ),
            WorkflowTask(
                task_id="D",
                name="Task D",
                function=lambda: "D",
                dependencies=["B"]
            ),
            
            # Level 2 (depend on level 1)
            WorkflowTask(
                task_id="E",
                name="Task E",
                function=lambda: "E",
                dependencies=["C", "D"]
            )
        ]
        
        results = await manager.execute_tasks(tasks, {})
        await manager.shutdown()
        
        # All tasks should complete
        assert len(results) == 5
        for task_id in ["A", "B", "C", "D", "E"]:
            assert results[task_id].status == TaskStatus.COMPLETED
            assert results[task_id].result == task_id


class TestExecutionConfig:
    """Test ExecutionConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()
        
        assert config.max_parallel_tasks == 5
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.task_timeout == 300.0
        assert config.execution_strategy == ExecutionStrategy.MIXED
        assert config.enable_task_batching is False
        assert config.adaptive_scaling is True
        assert config.performance_monitoring is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        resource_limits = {
            ResourceType.CPU: ResourceLimit(ResourceType.CPU, 8.0)
        }
        
        config = ExecutionConfig(
            max_parallel_tasks=10,
            max_retries=5,
            retry_delay=2.0,
            task_timeout=600.0,
            resource_limits=resource_limits,
            execution_strategy=ExecutionStrategy.PARALLEL,
            enable_task_batching=True,
            batch_size=20
        )
        
        assert config.max_parallel_tasks == 10
        assert config.max_retries == 5
        assert config.retry_delay == 2.0
        assert config.task_timeout == 600.0
        assert config.resource_limits == resource_limits
        assert config.execution_strategy == ExecutionStrategy.PARALLEL
        assert config.enable_task_batching is True
        assert config.batch_size == 20


class TestCustomTaskExecutor:
    """Test custom task executor functionality."""
    
    class CustomExecutor(TaskExecutor):
        """Custom executor for testing."""
        
        def __init__(self):
            self.executed_tasks = []
        
        async def execute_task(self, task_execution: TaskExecution, context: Dict[str, Any]) -> Any:
            self.executed_tasks.append(task_execution.task.task_id)
            return f"custom_result_{task_execution.task.task_id}"
        
        async def can_execute(self, task: WorkflowTask, context: Dict[str, Any]) -> bool:
            return task.task_id.startswith("custom_")
    
    @pytest.mark.asyncio
    async def test_custom_executor_integration(self):
        """Test integration of custom executor."""
        config = ExecutionConfig()
        manager = ParallelExecutionManager(config)
        await manager.initialize()
        
        custom_executor = self.CustomExecutor()
        manager.add_executor(custom_executor)
        
        tasks = [
            WorkflowTask(
                task_id="custom_task1",
                name="Custom Task 1"
            ),
            WorkflowTask(
                task_id="regular_task1",
                name="Regular Task 1",
                function=lambda: "regular_result"
            )
        ]
        
        results = await manager.execute_tasks(tasks, {}, ExecutionMode.PARALLEL)
        await manager.shutdown()
        
        # Check that custom executor was used for custom task
        assert "custom_task1" in custom_executor.executed_tasks
        assert results["custom_task1"].result == "custom_result_custom_task1"
        
        # Check that regular task was executed by default executor
        assert results["regular_task1"].result == "regular_result"


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test collection of performance metrics."""
        config = ExecutionConfig(performance_monitoring=True)
        manager = ParallelExecutionManager(config)
        await manager.initialize()
        
        async def slow_task():
            await asyncio.sleep(0.1)
            return "completed"
        
        tasks = [
            WorkflowTask(
                task_id="slow_task",
                name="Slow Task",
                function=slow_task
            )
        ]
        
        # Execute task multiple times
        for i in range(3):
            results = await manager.execute_tasks(tasks, {}, ExecutionMode.PARALLEL)
            assert results["slow_task"].status == TaskStatus.COMPLETED
        
        await manager.shutdown()
        
        # Check performance metrics
        metrics = manager.get_performance_metrics()
        assert "slow_task" in metrics
        
        task_metrics = metrics["slow_task"]
        assert task_metrics["executions"] == 3
        assert task_metrics["success_rate"] == 1.0
        assert task_metrics["avg_duration"].total_seconds() > 0


if __name__ == "__main__":
    pytest.main([__file__])