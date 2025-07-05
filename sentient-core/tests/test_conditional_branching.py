"""Unit tests for the conditional branching system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from core.workflows.conditional_branching import (
    ConditionType,
    ComparisonOperator,
    LogicalOperator,
    ConditionExpression,
    ComplexCondition,
    ConditionalBranch,
    BranchingContext,
    SimpleConditionEvaluator,
    TemporalConditionEvaluator,
    ConditionalBranchingEngine
)
from core.workflows.advanced_engine import WorkflowExecution, WorkflowStatus


class TestConditionExpression:
    """Test ConditionExpression functionality."""
    
    def test_condition_expression_creation(self):
        """Test creating condition expressions."""
        condition = ConditionExpression(
            left_operand="$task_result",
            operator=ComparisonOperator.EQUALS,
            right_operand="success"
        )
        
        assert condition.left_operand == "$task_result"
        assert condition.operator == ComparisonOperator.EQUALS
        assert condition.right_operand == "success"
        assert condition.data_type == "auto"


class TestComplexCondition:
    """Test ComplexCondition functionality."""
    
    def test_complex_condition_creation(self):
        """Test creating complex conditions."""
        expr1 = ConditionExpression("$status", ComparisonOperator.EQUALS, "completed")
        expr2 = ConditionExpression("$score", ComparisonOperator.GREATER_THAN, 80)
        
        complex_condition = ComplexCondition(
            expressions=[expr1, expr2],
            logical_operator=LogicalOperator.AND
        )
        
        assert len(complex_condition.expressions) == 2
        assert complex_condition.logical_operator == LogicalOperator.AND
        assert not complex_condition.parentheses


class TestBranchingContext:
    """Test BranchingContext functionality."""
    
    def test_branching_context_creation(self):
        """Test creating branching context."""
        context = BranchingContext(
            workflow_id="wf_123",
            execution_id="exec_456",
            current_task_id="task_789",
            task_results={"task1": "success"},
            workflow_variables={"var1": "value1"}
        )
        
        assert context.workflow_id == "wf_123"
        assert context.execution_id == "exec_456"
        assert context.current_task_id == "task_789"
        assert context.task_results["task1"] == "success"
        assert context.workflow_variables["var1"] == "value1"
        assert isinstance(context.timestamp, datetime)


class TestSimpleConditionEvaluator:
    """Test SimpleConditionEvaluator functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = SimpleConditionEvaluator()
        self.context = BranchingContext(
            workflow_id="test_wf",
            execution_id="test_exec",
            task_results={
                "task1": "completed",
                "task2": {"status": "success", "score": 95}
            },
            workflow_variables={
                "threshold": 80,
                "status": "active"
            },
            system_context={
                "current_time": "14:30"
            }
        )
    
    @pytest.mark.asyncio
    async def test_simple_equals_condition(self):
        """Test simple equals condition."""
        condition = ConditionExpression(
            left_operand="$task1",
            operator=ComparisonOperator.EQUALS,
            right_operand="completed"
        )
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_simple_greater_than_condition(self):
        """Test simple greater than condition."""
        condition = ConditionExpression(
            left_operand="$task2.score",
            operator=ComparisonOperator.GREATER_THAN,
            right_operand=90
        )
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_variable_reference_condition(self):
        """Test condition with variable references."""
        condition = ConditionExpression(
            left_operand="$task2.score",
            operator=ComparisonOperator.GREATER_THAN,
            right_operand="$threshold"
        )
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_string_condition_parsing(self):
        """Test parsing string conditions."""
        condition_str = "$status == 'active'"
        
        result = await self.evaluator.evaluate(condition_str, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_contains_condition(self):
        """Test contains condition."""
        condition_str = "$task1 contains 'comp'"
        
        result = await self.evaluator.evaluate(condition_str, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_complex_and_condition(self):
        """Test complex AND condition."""
        expr1 = ConditionExpression("$task1", ComparisonOperator.EQUALS, "completed")
        expr2 = ConditionExpression("$task2.score", ComparisonOperator.GREATER_THAN, 80)
        
        complex_condition = ComplexCondition(
            expressions=[expr1, expr2],
            logical_operator=LogicalOperator.AND
        )
        
        result = await self.evaluator.evaluate(complex_condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_complex_or_condition(self):
        """Test complex OR condition."""
        expr1 = ConditionExpression("$task1", ComparisonOperator.EQUALS, "failed")
        expr2 = ConditionExpression("$task2.score", ComparisonOperator.GREATER_THAN, 80)
        
        complex_condition = ComplexCondition(
            expressions=[expr1, expr2],
            logical_operator=LogicalOperator.OR
        )
        
        result = await self.evaluator.evaluate(complex_condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_not_condition(self):
        """Test NOT condition."""
        expr1 = ConditionExpression("$task1", ComparisonOperator.EQUALS, "failed")
        
        complex_condition = ComplexCondition(
            expressions=[expr1],
            logical_operator=LogicalOperator.NOT
        )
        
        result = await self.evaluator.evaluate(complex_condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_exists_condition(self):
        """Test exists condition."""
        condition_str = "$task1 exists"
        
        result = await self.evaluator.evaluate(condition_str, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_not_exists_condition(self):
        """Test not exists condition."""
        condition = ConditionExpression(
            left_operand="$nonexistent",
            operator=ComparisonOperator.NOT_EXISTS,
            right_operand=None
        )
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_type_conversion(self):
        """Test automatic type conversion."""
        condition = ConditionExpression(
            left_operand="$task2.score",
            operator=ComparisonOperator.EQUALS,
            right_operand="95",  # String that should be converted to number
            data_type="number"
        )
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True


class TestTemporalConditionEvaluator:
    """Test TemporalConditionEvaluator functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = TemporalConditionEvaluator()
        self.context = BranchingContext(
            workflow_id="test_wf",
            execution_id="test_exec",
            timestamp=datetime(2023, 6, 15, 14, 30)  # Thursday 2:30 PM
        )
    
    @pytest.mark.asyncio
    async def test_after_time_condition(self):
        """Test after time condition."""
        condition = "after 12:00"
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_before_time_condition(self):
        """Test before time condition."""
        condition = "before 16:00"
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_between_time_condition(self):
        """Test between time condition."""
        condition = "between 09:00 and 17:00"
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_weekday_condition(self):
        """Test weekday condition."""
        condition = "weekday"
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is True  # Thursday is a weekday
    
    @pytest.mark.asyncio
    async def test_weekend_condition(self):
        """Test weekend condition."""
        condition = "weekend"
        
        result = await self.evaluator.evaluate(condition, self.context)
        assert result is False  # Thursday is not a weekend


class TestConditionalBranchingEngine:
    """Test ConditionalBranchingEngine functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.engine = ConditionalBranchingEngine()
        self.context = BranchingContext(
            workflow_id="test_wf",
            execution_id="test_exec",
            task_results={"task1": "completed"},
            workflow_variables={"score": 85}
        )
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test engine initialization."""
        await self.engine.initialize()
        
        assert ConditionType.SIMPLE in self.engine.evaluators
        assert ConditionType.COMPLEX in self.engine.evaluators
        assert ConditionType.TEMPORAL in self.engine.evaluators
    
    @pytest.mark.asyncio
    async def test_register_branch(self):
        """Test registering a conditional branch."""
        await self.engine.initialize()
        
        branch = ConditionalBranch(
            branch_id="branch1",
            name="Test Branch",
            condition="$task1 == 'completed'",
            true_path=["task2", "task3"],
            false_path=["task4"]
        )
        
        await self.engine.register_branch(branch)
        
        assert "branch1" in self.engine.branches
        assert self.engine.branches["branch1"].name == "Test Branch"
    
    @pytest.mark.asyncio
    async def test_evaluate_branch(self):
        """Test evaluating a branch condition."""
        await self.engine.initialize()
        
        branch = ConditionalBranch(
            branch_id="branch1",
            name="Test Branch",
            condition="$task1 == 'completed'",
            true_path=["task2", "task3"],
            false_path=["task4"]
        )
        
        await self.engine.register_branch(branch)
        
        result = await self.engine.evaluate_branch("branch1", self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_next_tasks_true_path(self):
        """Test getting next tasks for true condition."""
        await self.engine.initialize()
        
        branch = ConditionalBranch(
            branch_id="branch1",
            name="Test Branch",
            condition="$task1 == 'completed'",
            true_path=["task2", "task3"],
            false_path=["task4"]
        )
        
        await self.engine.register_branch(branch)
        
        next_tasks = await self.engine.get_next_tasks("branch1", self.context)
        assert next_tasks == ["task2", "task3"]
    
    @pytest.mark.asyncio
    async def test_get_next_tasks_false_path(self):
        """Test getting next tasks for false condition."""
        await self.engine.initialize()
        
        branch = ConditionalBranch(
            branch_id="branch1",
            name="Test Branch",
            condition="$task1 == 'failed'",
            true_path=["task2", "task3"],
            false_path=["task4"]
        )
        
        await self.engine.register_branch(branch)
        
        next_tasks = await self.engine.get_next_tasks("branch1", self.context)
        assert next_tasks == ["task4"]
    
    @pytest.mark.asyncio
    async def test_evaluate_all_branches(self):
        """Test evaluating all registered branches."""
        await self.engine.initialize()
        
        branch1 = ConditionalBranch(
            branch_id="branch1",
            name="Branch 1",
            condition="$task1 == 'completed'",
            true_path=["task2"],
            false_path=["task3"]
        )
        
        branch2 = ConditionalBranch(
            branch_id="branch2",
            name="Branch 2",
            condition="$score > 90",
            true_path=["task4"],
            false_path=["task5"]
        )
        
        await self.engine.register_branch(branch1)
        await self.engine.register_branch(branch2)
        
        results = await self.engine.evaluate_all_branches(self.context)
        
        assert results["branch1"] is True
        assert results["branch2"] is False  # score is 85, not > 90
    
    @pytest.mark.asyncio
    async def test_custom_evaluator(self):
        """Test registering and using custom evaluator."""
        await self.engine.initialize()
        
        async def custom_evaluator(condition, context):
            return context.workflow_variables.get("score", 0) > 80
        
        await self.engine.register_custom_evaluator("high_score", custom_evaluator)
        
        branch = ConditionalBranch(
            branch_id="branch1",
            name="Custom Branch",
            condition="custom:high_score()",
            true_path=["task2"],
            false_path=["task3"]
        )
        
        await self.engine.register_branch(branch)
        
        result = await self.engine.evaluate_branch("branch1", self.context)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_create_context_from_execution(self):
        """Test creating context from workflow execution."""
        await self.engine.initialize()
        
        # Mock workflow execution
        execution = Mock(spec=WorkflowExecution)
        execution.workflow_id = "wf_123"
        execution.execution_id = "exec_456"
        execution.current_task_id = "task_789"
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.now()
        execution.variables = {"var1": "value1"}
        
        # Mock task results
        task_result = Mock()
        task_result.result = "success"
        task_result.status = Mock()
        task_result.status.value = "completed"
        execution.task_results = {"task1": task_result}
        
        context = await self.engine.create_context_from_execution(
            execution,
            {"user_var": "user_value"}
        )
        
        assert context.workflow_id == "wf_123"
        assert context.execution_id == "exec_456"
        assert context.current_task_id == "task_789"
        assert context.task_results["task1"] == "success"
        assert context.workflow_variables["var1"] == "value1"
        assert context.user_context["user_var"] == "user_value"
        assert context.system_context["status"] == "running"
        assert context.system_context["progress"] == 1.0


class TestIntegration:
    """Integration tests for conditional branching system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_branching(self):
        """Test end-to-end workflow with conditional branching."""
        engine = ConditionalBranchingEngine()
        await engine.initialize()
        
        # Setup context
        context = BranchingContext(
            workflow_id="integration_test",
            execution_id="exec_001",
            task_results={
                "data_validation": "passed",
                "quality_check": {"score": 92, "issues": []}
            },
            workflow_variables={
                "quality_threshold": 85,
                "environment": "production"
            }
        )
        
        # Register branches
        validation_branch = ConditionalBranch(
            branch_id="validation_check",
            name="Data Validation Check",
            condition="$data_validation == 'passed'",
            true_path=["quality_analysis"],
            false_path=["error_handling", "notification"]
        )
        
        quality_branch = ConditionalBranch(
            branch_id="quality_check",
            name="Quality Score Check",
            condition="$quality_check.score >= $quality_threshold",
            true_path=["deploy_to_production"],
            false_path=["additional_review", "improvement_suggestions"]
        )
        
        environment_branch = ConditionalBranch(
            branch_id="environment_check",
            name="Environment Check",
            condition="$environment == 'production'",
            true_path=["production_deployment", "monitoring_setup"],
            false_path=["staging_deployment"]
        )
        
        await engine.register_branch(validation_branch)
        await engine.register_branch(quality_branch)
        await engine.register_branch(environment_branch)
        
        # Evaluate all branches
        results = await engine.evaluate_all_branches(context)
        
        # Verify results
        assert results["validation_check"] is True
        assert results["quality_check"] is True
        assert results["environment_check"] is True
        
        # Get next tasks for each branch
        validation_tasks = await engine.get_next_tasks("validation_check", context)
        quality_tasks = await engine.get_next_tasks("quality_check", context)
        environment_tasks = await engine.get_next_tasks("environment_check", context)
        
        assert validation_tasks == ["quality_analysis"]
        assert quality_tasks == ["deploy_to_production"]
        assert environment_tasks == ["production_deployment", "monitoring_setup"]


if __name__ == "__main__":
    pytest.main([__file__])