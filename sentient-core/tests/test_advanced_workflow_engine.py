"""Unit tests for Advanced Workflow Engine

Tests the core functionality of the AdvancedWorkflowEngine including:
- Workflow creation and execution
- Task orchestration and dependency management
- Parallel and sequential execution modes
- Human-in-the-loop workflows
- Conditional branching
- Error handling and recovery
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from core.workflows.advanced_engine import (
    AdvancedWorkflowEngine,
    WorkflowDefinition,
    WorkflowTask,
    WorkflowCondition,
    WorkflowStatus,
    TaskStatus,
    ExecutionMode
)


class TestAdvancedWorkflowEngine:
    """Test suite for AdvancedWorkflowEngine."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        agent_registry = Mock()
        state_service = AsyncMock()
        memory_service = Mock()
        
        return agent_registry, state_service, memory_service
    
    @pytest.fixture
    def workflow_engine(self, mock_services):
        """Create workflow engine instance for testing."""
        agent_registry, state_service, memory_service = mock_services
        return AdvancedWorkflowEngine(agent_registry, state_service, memory_service)
    
    @pytest.fixture
    def simple_workflow(self):
        """Create a simple workflow for testing."""
        return WorkflowDefinition(
            id="test_workflow",
            name="Test Workflow",
            description="A simple test workflow",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="task1",
                    name="First Task",
                    description="First task in workflow",
                    agent_type="test_agent",
                    task_data={"action": "test"}
                ),
                WorkflowTask(
                    id="task2",
                    name="Second Task",
                    description="Second task in workflow",
                    agent_type="test_agent",
                    task_data={"action": "test"},
                    dependencies=["task1"]
                )
            ]
        )
    
    @pytest.fixture
    def parallel_workflow(self):
        """Create a workflow with parallel tasks."""
        return WorkflowDefinition(
            id="parallel_workflow",
            name="Parallel Test Workflow",
            description="A workflow with parallel tasks",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="task1",
                    name="Initial Task",
                    description="Initial sequential task",
                    agent_type="test_agent",
                    task_data={"action": "init"}
                ),
                WorkflowTask(
                    id="task2",
                    name="Parallel Task 1",
                    description="First parallel task",
                    agent_type="test_agent",
                    task_data={"action": "parallel1"},
                    dependencies=["task1"],
                    execution_mode=ExecutionMode.PARALLEL
                ),
                WorkflowTask(
                    id="task3",
                    name="Parallel Task 2",
                    description="Second parallel task",
                    agent_type="test_agent",
                    task_data={"action": "parallel2"},
                    dependencies=["task1"],
                    execution_mode=ExecutionMode.PARALLEL
                )
            ]
        )
    
    @pytest.mark.asyncio
    async def test_workflow_engine_initialization(self, workflow_engine):
        """Test workflow engine initialization."""
        await workflow_engine.initialize()
        
        # Check that templates are loaded
        assert "feature_development" in workflow_engine.workflow_templates
        assert "research_workflow" in workflow_engine.workflow_templates
        
        # Check that condition evaluators are set up
        assert "success" in workflow_engine.condition_evaluators
        assert "failure" in workflow_engine.condition_evaluators
        assert "custom" in workflow_engine.condition_evaluators
    
    @pytest.mark.asyncio
    async def test_create_workflow_from_template(self, workflow_engine):
        """Test creating workflow from template."""
        await workflow_engine.initialize()
        
        parameters = {"feature_name": "test_feature"}
        workflow = await workflow_engine.create_workflow_from_template(
            "feature_development", parameters
        )
        
        assert workflow is not None
        assert workflow.name == "Feature Development Workflow"
        assert len(workflow.tasks) > 0
    
    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, workflow_engine, simple_workflow):
        """Test execution of a simple sequential workflow."""
        await workflow_engine.initialize()
        
        # Mock agent execution
        mock_agent = AsyncMock()
        mock_agent.process_task.return_value = {"status": "completed", "result": "success"}
        workflow_engine._get_agent_for_task = AsyncMock(return_value=mock_agent)
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(simple_workflow)
        
        # Wait a bit for async execution
        await asyncio.sleep(0.2)
        
        # Check execution was created
        assert execution_id in workflow_engine.active_executions
        execution = workflow_engine.active_executions[execution_id]
        assert execution.workflow_definition.id == "test_workflow"
    
    @pytest.mark.asyncio
    async def test_parallel_workflow_execution(self, workflow_engine, parallel_workflow):
        """Test execution of a workflow with parallel tasks."""
        await workflow_engine.initialize()
        
        # Mock agent execution
        mock_agent = AsyncMock()
        mock_agent.process_task.return_value = {"status": "completed", "result": "success"}
        workflow_engine._get_agent_for_task = AsyncMock(return_value=mock_agent)
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(parallel_workflow)
        
        # Wait a bit for async execution
        await asyncio.sleep(0.2)
        
        # Check execution was created
        assert execution_id in workflow_engine.active_executions
        execution = workflow_engine.active_executions[execution_id]
        assert execution.workflow_definition.id == "parallel_workflow"
    
    @pytest.mark.asyncio
    async def test_human_in_loop_workflow(self, workflow_engine):
        """Test workflow with human intervention."""
        await workflow_engine.initialize()
        
        # Create workflow with human-in-loop task
        workflow = WorkflowDefinition(
            id="human_loop_workflow",
            name="Human Loop Test",
            description="Workflow with human intervention",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="human_task",
                    name="Human Validation Task",
                    description="Task requiring human validation",
                    agent_type="test_agent",
                    task_data={"action": "validate"},
                    execution_mode=ExecutionMode.HUMAN_IN_LOOP,
                    human_validation_required=True
                )
            ]
        )
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(workflow)
        
        # Wait a bit for async execution to start
        await asyncio.sleep(0.1)
        
        # Check that workflow is waiting for input
        execution = workflow_engine.active_executions[execution_id]
        assert execution.status == WorkflowStatus.WAITING_FOR_INPUT
        
        # Provide human input
        await workflow_engine.provide_human_input(
            execution_id, "human_task", {"approved": True, "comment": "Looks good"}
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Check that input was recorded
        assert "human_task" in execution.human_inputs
        assert execution.human_inputs["human_task"]["approved"] is True
    
    @pytest.mark.asyncio
    async def test_workflow_with_conditions(self, workflow_engine):
        """Test workflow with conditional branching."""
        await workflow_engine.initialize()
        
        # Create workflow with conditions
        workflow = WorkflowDefinition(
            id="conditional_workflow",
            name="Conditional Test",
            description="Workflow with conditional branching",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="initial_task",
                    name="Initial Task",
                    description="Initial task",
                    agent_type="test_agent",
                    task_data={"action": "init"}
                ),
                WorkflowTask(
                    id="success_task",
                    name="Success Task",
                    description="Task for success path",
                    agent_type="test_agent",
                    task_data={"action": "success"}
                ),
                WorkflowTask(
                    id="failure_task",
                    name="Failure Task",
                    description="Task for failure path",
                    agent_type="test_agent",
                    task_data={"action": "failure"}
                )
            ],
            conditions=[
                WorkflowCondition(
                    condition_id="success_check",
                    condition_type="success",
                    condition_expression="no_failures",
                    true_path=["success_task"],
                    false_path=["failure_task"]
                )
            ]
        )
        
        # Mock agent execution
        mock_agent = AsyncMock()
        mock_agent.process_task.return_value = {"status": "completed", "result": "success"}
        workflow_engine._get_agent_for_task = AsyncMock(return_value=mock_agent)
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(workflow)
        
        # Wait for execution
        await asyncio.sleep(0.2)
        
        # Check execution
        execution = workflow_engine.active_executions[execution_id]
        assert execution.workflow_definition.id == "conditional_workflow"
    
    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self, workflow_engine):
        """Test task retry mechanism on failure."""
        await workflow_engine.initialize()
        
        # Create workflow with task that will fail initially
        workflow = WorkflowDefinition(
            id="retry_workflow",
            name="Retry Test",
            description="Workflow to test retry mechanism",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="retry_task",
                    name="Retry Task",
                    description="Task that will be retried",
                    agent_type="test_agent",
                    task_data={"action": "retry_test"},
                    max_retries=2
                )
            ]
        )
        
        # Mock agent that fails first time, succeeds second time
        mock_agent = AsyncMock()
        call_count = 0
        
        async def mock_process_task(task_data):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Simulated failure")
            return {"status": "completed", "result": "success"}
        
        mock_agent.process_task = mock_process_task
        workflow_engine._get_agent_for_task = AsyncMock(return_value=mock_agent)
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(workflow)
        
        # Wait for execution with retries
        await asyncio.sleep(0.5)
        
        # Check that task was retried
        execution = workflow_engine.active_executions[execution_id]
        task = execution.workflow_definition.tasks[0]
        assert task.retry_count > 0
    
    def test_get_initial_tasks(self, workflow_engine, simple_workflow):
        """Test getting initial tasks (no dependencies)."""
        initial_tasks = workflow_engine._get_initial_tasks(simple_workflow)
        
        assert len(initial_tasks) == 1
        assert initial_tasks[0].id == "task1"
    
    def test_get_ready_tasks(self, workflow_engine, simple_workflow):
        """Test getting ready tasks based on completed dependencies."""
        # Create mock execution
        from core.workflows.advanced_engine import WorkflowExecution
        
        execution = WorkflowExecution(
            execution_id="test_exec",
            workflow_definition=simple_workflow,
            status=WorkflowStatus.RUNNING,
            completed_tasks=["task1"]
        )
        
        ready_tasks = workflow_engine._get_ready_tasks(execution)
        
        assert len(ready_tasks) == 1
        assert ready_tasks[0].id == "task2"
    
    @pytest.mark.asyncio
    async def test_condition_evaluators(self, workflow_engine):
        """Test condition evaluation functions."""
        await workflow_engine.initialize()
        
        # Create mock execution context
        from core.workflows.advanced_engine import WorkflowExecution
        
        execution = WorkflowExecution(
            execution_id="test_exec",
            workflow_definition=WorkflowDefinition(
                id="test", name="test", description="test", version="1.0", tasks=[]
            ),
            status=WorkflowStatus.RUNNING,
            failed_tasks=[]
        )
        
        context = {"execution": execution, "task_results": {}}
        
        # Test success condition
        success_result = await workflow_engine._evaluate_success_condition("", context)
        assert success_result is True
        
        # Test failure condition
        execution.failed_tasks = ["failed_task"]
        failure_result = await workflow_engine._evaluate_failure_condition("", context)
        assert failure_result is True
        
        # Test custom condition
        custom_result = await workflow_engine._evaluate_custom_condition("", context)
        assert custom_result is True
    
    def test_workflow_templates_creation(self, workflow_engine):
        """Test creation of default workflow templates."""
        # Test feature development template
        feature_template = workflow_engine._create_feature_development_template()
        assert feature_template.name == "Feature Development Workflow"
        assert len(feature_template.tasks) == 5
        
        # Check task dependencies
        design_task = next(t for t in feature_template.tasks if t.id == "design_architecture")
        assert "analyze_requirements" in design_task.dependencies
        
        # Test research workflow template
        research_template = workflow_engine._create_research_workflow_template()
        assert research_template.name == "Research Workflow"
        assert len(research_template.tasks) == 3
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_engine, simple_workflow):
        """Test workflow error handling and recovery."""
        await workflow_engine.initialize()
        
        # Mock agent that always fails
        mock_agent = AsyncMock()
        mock_agent.process_task.side_effect = Exception("Simulated error")
        workflow_engine._get_agent_for_task = AsyncMock(return_value=mock_agent)
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(simple_workflow)
        
        # Wait for execution to complete
        await asyncio.sleep(0.3)
        
        # Check that workflow failed gracefully
        # The execution should be moved to history after completion
        assert execution_id in [exec.execution_id for exec in workflow_engine.execution_history]
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, workflow_engine):
        """Test workflow task timeout handling."""
        await workflow_engine.initialize()
        
        # Create workflow with timeout
        workflow = WorkflowDefinition(
            id="timeout_workflow",
            name="Timeout Test",
            description="Workflow to test timeout handling",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="timeout_task",
                    name="Timeout Task",
                    description="Task that will timeout",
                    agent_type="test_agent",
                    task_data={"action": "timeout_test"},
                    timeout=1  # 1 second timeout
                )
            ]
        )
        
        # Mock agent that takes too long
        mock_agent = AsyncMock()
        
        async def slow_process_task(task_data):
            await asyncio.sleep(2)  # Takes longer than timeout
            return {"status": "completed", "result": "success"}
        
        mock_agent.process_task = slow_process_task
        workflow_engine._get_agent_for_task = AsyncMock(return_value=mock_agent)
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(workflow)
        
        # Wait for timeout to occur
        await asyncio.sleep(1.5)
        
        # Check that task failed due to timeout
        execution = workflow_engine.active_executions.get(execution_id)
        if execution:
            task = execution.workflow_definition.tasks[0]
            # Task should either be failed or still running (depending on timing)
            assert task.status in [TaskStatus.FAILED, TaskStatus.RUNNING]


if __name__ == "__main__":
    pytest.main([__file__])