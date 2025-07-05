"""Comprehensive test suite for Enhanced Workflow Orchestrator

This module tests all components of the enhanced workflow orchestration system,
including resource management, scheduling, orchestration strategies, and
integration with the existing multi-agent system.
"""

import pytest
import asyncio
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import uuid
import json
from concurrent.futures import Future

# Import the modules to test
from core.orchestration.enhanced_workflow_orchestrator import (
    EnhancedWorkflowOrchestrator,
    OrchestrationEngine,
    ResourceManager,
    WorkflowScheduler,
    OrchestrationStrategy,
    WorkflowPriority,
    ResourceType,
    ResourceRequirement,
    WorkflowSchedule,
    OrchestrationPlan,
    default_orchestrator
)
from core.models import AppState, EnhancedTask, TaskStatus
from core.services.enhanced_integration_service import EnhancedIntegrationService

class TestResourceManager:
    """Test suite for ResourceManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resource_manager = ResourceManager()
    
    def test_initialization(self):
        """Test ResourceManager initialization."""
        assert self.resource_manager.available_resources[ResourceType.CPU] == 100.0
        assert self.resource_manager.available_resources[ResourceType.MEMORY] == 100.0
        assert len(self.resource_manager.allocated_resources) == 0
        assert len(self.resource_manager.resource_history) == 0
    
    def test_custom_initialization(self):
        """Test ResourceManager with custom initial resources."""
        custom_resources = {
            ResourceType.CPU: 200.0,
            ResourceType.MEMORY: 150.0
        }
        rm = ResourceManager(custom_resources)
        
        assert rm.available_resources[ResourceType.CPU] == 200.0
        assert rm.available_resources[ResourceType.MEMORY] == 150.0
    
    def test_check_resource_availability_sufficient(self):
        """Test resource availability check with sufficient resources."""
        requirements = [
            ResourceRequirement(ResourceType.CPU, 50.0),
            ResourceRequirement(ResourceType.MEMORY, 30.0)
        ]
        
        assert self.resource_manager.check_resource_availability(requirements) is True
    
    def test_check_resource_availability_insufficient(self):
        """Test resource availability check with insufficient resources."""
        requirements = [
            ResourceRequirement(ResourceType.CPU, 150.0),  # More than available
            ResourceRequirement(ResourceType.MEMORY, 30.0)
        ]
        
        assert self.resource_manager.check_resource_availability(requirements) is False
    
    def test_allocate_resources_success(self):
        """Test successful resource allocation."""
        workflow_id = "test_workflow_1"
        requirements = [
            ResourceRequirement(ResourceType.CPU, 50.0),
            ResourceRequirement(ResourceType.MEMORY, 30.0)
        ]
        
        result = self.resource_manager.allocate_resources(workflow_id, requirements)
        
        assert result is True
        assert self.resource_manager.available_resources[ResourceType.CPU] == 50.0
        assert self.resource_manager.available_resources[ResourceType.MEMORY] == 70.0
        assert workflow_id in self.resource_manager.allocated_resources
        assert len(self.resource_manager.resource_history) == 1
    
    def test_allocate_resources_failure(self):
        """Test failed resource allocation due to insufficient resources."""
        workflow_id = "test_workflow_2"
        requirements = [
            ResourceRequirement(ResourceType.CPU, 150.0)  # More than available
        ]
        
        result = self.resource_manager.allocate_resources(workflow_id, requirements)
        
        assert result is False
        assert self.resource_manager.available_resources[ResourceType.CPU] == 100.0
        assert workflow_id not in self.resource_manager.allocated_resources
    
    def test_release_resources_success(self):
        """Test successful resource release."""
        workflow_id = "test_workflow_3"
        requirements = [
            ResourceRequirement(ResourceType.CPU, 40.0),
            ResourceRequirement(ResourceType.MEMORY, 20.0)
        ]
        
        # First allocate
        self.resource_manager.allocate_resources(workflow_id, requirements)
        
        # Then release
        result = self.resource_manager.release_resources(workflow_id)
        
        assert result is True
        assert self.resource_manager.available_resources[ResourceType.CPU] == 100.0
        assert self.resource_manager.available_resources[ResourceType.MEMORY] == 100.0
        assert workflow_id not in self.resource_manager.allocated_resources
        assert len(self.resource_manager.resource_history) == 2  # Allocate + Release
    
    def test_release_resources_nonexistent(self):
        """Test releasing resources for non-existent workflow."""
        result = self.resource_manager.release_resources("nonexistent_workflow")
        assert result is False
    
    def test_get_resource_utilization(self):
        """Test resource utilization calculation."""
        workflow_id = "test_workflow_4"
        requirements = [
            ResourceRequirement(ResourceType.CPU, 25.0),
            ResourceRequirement(ResourceType.MEMORY, 50.0)
        ]
        
        self.resource_manager.allocate_resources(workflow_id, requirements)
        utilization = self.resource_manager.get_resource_utilization()
        
        assert utilization[ResourceType.CPU] == 25.0  # 25/100 * 100
        assert utilization[ResourceType.MEMORY] == 50.0  # 50/100 * 100
    
    def test_concurrent_resource_operations(self):
        """Test thread safety of resource operations."""
        def allocate_and_release(workflow_id: str):
            requirements = [ResourceRequirement(ResourceType.CPU, 10.0)]
            if self.resource_manager.allocate_resources(workflow_id, requirements):
                time.sleep(0.1)  # Simulate work
                self.resource_manager.release_resources(workflow_id)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=allocate_and_release, args=[f"workflow_{i}"])
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All resources should be released
        assert self.resource_manager.available_resources[ResourceType.CPU] == 100.0
        assert len(self.resource_manager.allocated_resources) == 0

class TestWorkflowScheduler:
    """Test suite for WorkflowScheduler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resource_manager = ResourceManager()
        self.scheduler = WorkflowScheduler(self.resource_manager)
    
    def test_initialization(self):
        """Test WorkflowScheduler initialization."""
        assert len(self.scheduler.scheduled_workflows) == 0
        assert len(self.scheduler.running_workflows) == 0
        assert len(self.scheduler.completed_workflows) == 0
        assert len(self.scheduler.failed_workflows) == 0
    
    def test_schedule_workflow(self):
        """Test workflow scheduling."""
        schedule = WorkflowSchedule(
            workflow_id="test_workflow",
            scheduled_time=datetime.now(),
            priority=WorkflowPriority.NORMAL,
            resource_requirements=[ResourceRequirement(ResourceType.CPU, 10.0)]
        )
        
        result = self.scheduler.schedule_workflow(schedule)
        
        assert result is True
        assert len(self.scheduler.scheduled_workflows) == 1
    
    def test_get_next_workflow_available(self):
        """Test getting next workflow when one is available."""
        schedule = WorkflowSchedule(
            workflow_id="test_workflow",
            scheduled_time=datetime.now() - timedelta(seconds=1),  # Past time
            priority=WorkflowPriority.HIGH,
            resource_requirements=[ResourceRequirement(ResourceType.CPU, 10.0)]
        )
        
        self.scheduler.schedule_workflow(schedule)
        next_workflow = self.scheduler.get_next_workflow()
        
        assert next_workflow is not None
        assert next_workflow.workflow_id == "test_workflow"
    
    def test_get_next_workflow_future_time(self):
        """Test getting next workflow when scheduled for future."""
        schedule = WorkflowSchedule(
            workflow_id="future_workflow",
            scheduled_time=datetime.now() + timedelta(hours=1),  # Future time
            priority=WorkflowPriority.HIGH,
            resource_requirements=[ResourceRequirement(ResourceType.CPU, 10.0)]
        )
        
        self.scheduler.schedule_workflow(schedule)
        next_workflow = self.scheduler.get_next_workflow()
        
        assert next_workflow is None
    
    def test_get_next_workflow_insufficient_resources(self):
        """Test getting next workflow with insufficient resources."""
        schedule = WorkflowSchedule(
            workflow_id="resource_heavy_workflow",
            scheduled_time=datetime.now() - timedelta(seconds=1),
            priority=WorkflowPriority.HIGH,
            resource_requirements=[ResourceRequirement(ResourceType.CPU, 200.0)]  # More than available
        )
        
        self.scheduler.schedule_workflow(schedule)
        next_workflow = self.scheduler.get_next_workflow()
        
        assert next_workflow is None
    
    def test_priority_ordering(self):
        """Test that workflows are returned in priority order."""
        # Schedule workflows with different priorities
        schedules = [
            WorkflowSchedule(
                workflow_id="low_priority",
                scheduled_time=datetime.now() - timedelta(seconds=1),
                priority=WorkflowPriority.LOW,
                resource_requirements=[ResourceRequirement(ResourceType.CPU, 5.0)]
            ),
            WorkflowSchedule(
                workflow_id="high_priority",
                scheduled_time=datetime.now() - timedelta(seconds=1),
                priority=WorkflowPriority.HIGH,
                resource_requirements=[ResourceRequirement(ResourceType.CPU, 5.0)]
            ),
            WorkflowSchedule(
                workflow_id="critical_priority",
                scheduled_time=datetime.now() - timedelta(seconds=1),
                priority=WorkflowPriority.CRITICAL,
                resource_requirements=[ResourceRequirement(ResourceType.CPU, 5.0)]
            )
        ]
        
        for schedule in schedules:
            self.scheduler.schedule_workflow(schedule)
        
        # Should get critical first
        next_workflow = self.scheduler.get_next_workflow()
        assert next_workflow.workflow_id == "critical_priority"
        
        # Then high
        next_workflow = self.scheduler.get_next_workflow()
        assert next_workflow.workflow_id == "high_priority"
        
        # Finally low
        next_workflow = self.scheduler.get_next_workflow()
        assert next_workflow.workflow_id == "low_priority"
    
    def test_mark_workflow_running(self):
        """Test marking workflow as running."""
        workflow_id = "running_workflow"
        start_time = datetime.now()
        
        self.scheduler.mark_workflow_running(workflow_id, start_time)
        
        assert workflow_id in self.scheduler.running_workflows
        assert self.scheduler.running_workflows[workflow_id]['start_time'] == start_time
        assert self.scheduler.running_workflows[workflow_id]['status'] == 'running'
    
    def test_mark_workflow_completed_success(self):
        """Test marking workflow as successfully completed."""
        workflow_id = "completed_workflow"
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)
        
        self.scheduler.mark_workflow_running(workflow_id, start_time)
        self.scheduler.mark_workflow_completed(workflow_id, end_time, True, {"result": "success"})
        
        assert workflow_id not in self.scheduler.running_workflows
        assert workflow_id in self.scheduler.completed_workflows
        assert self.scheduler.completed_workflows[workflow_id]['success'] is True
        assert self.scheduler.completed_workflows[workflow_id]['duration'] == 30.0
    
    def test_mark_workflow_completed_failure(self):
        """Test marking workflow as failed."""
        workflow_id = "failed_workflow"
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=15)
        
        self.scheduler.mark_workflow_running(workflow_id, start_time)
        self.scheduler.mark_workflow_completed(workflow_id, end_time, False, {"error": "test error"})
        
        assert workflow_id not in self.scheduler.running_workflows
        assert workflow_id in self.scheduler.failed_workflows
        assert self.scheduler.failed_workflows[workflow_id]['success'] is False
    
    def test_get_scheduler_status(self):
        """Test getting scheduler status."""
        # Add some workflows in different states
        schedule = WorkflowSchedule(
            workflow_id="scheduled_workflow",
            scheduled_time=datetime.now() + timedelta(hours=1),
            priority=WorkflowPriority.NORMAL,
            resource_requirements=[]
        )
        self.scheduler.schedule_workflow(schedule)
        
        self.scheduler.mark_workflow_running("running_workflow", datetime.now())
        self.scheduler.mark_workflow_completed("completed_workflow", datetime.now(), True)
        
        status = self.scheduler.get_scheduler_status()
        
        assert status['scheduled_count'] == 1
        assert status['running_count'] == 1
        assert status['completed_count'] == 1
        assert status['failed_count'] == 0
        assert 'resource_utilization' in status

class TestOrchestrationEngine:
    """Test suite for OrchestrationEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the dependencies
        self.mock_langgraph = Mock()
        self.mock_nlp = Mock()
        self.mock_state_manager = Mock()
        
        self.engine = OrchestrationEngine(
            langgraph_integration=self.mock_langgraph,
            nlp_processor=self.mock_nlp,
            state_manager=self.mock_state_manager
        )
    
    def test_initialization(self):
        """Test OrchestrationEngine initialization."""
        assert self.engine.langgraph_integration == self.mock_langgraph
        assert self.engine.nlp_processor == self.mock_nlp
        assert self.engine.state_manager == self.mock_state_manager
        assert isinstance(self.engine.resource_manager, ResourceManager)
        assert isinstance(self.engine.scheduler, WorkflowScheduler)
        assert len(self.engine.active_plans) == 0
    
    def test_create_orchestration_plan(self):
        """Test creating an orchestration plan."""
        workflow_configs = [
            {
                'workflow_id': 'workflow_1',
                'priority': WorkflowPriority.HIGH.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 20.0}
                ]
            },
            {
                'workflow_id': 'workflow_2',
                'priority': WorkflowPriority.NORMAL.value,
                'dependencies': ['workflow_1']
            }
        ]
        
        plan = self.engine.create_orchestration_plan(
            name="Test Plan",
            description="A test orchestration plan",
            strategy=OrchestrationStrategy.SEQUENTIAL,
            workflow_configs=workflow_configs
        )
        
        assert plan.name == "Test Plan"
        assert plan.strategy == OrchestrationStrategy.SEQUENTIAL
        assert len(plan.workflows) == 2
        assert plan.workflows[0].workflow_id == 'workflow_1'
        assert plan.workflows[1].dependencies == ['workflow_1']
        assert plan.plan_id in self.engine.active_plans
    
    @patch('core.orchestration.enhanced_workflow_orchestrator.OrchestrationEngine._execute_single_workflow')
    def test_execute_sequential_strategy(self, mock_execute):
        """Test sequential execution strategy."""
        mock_execute.return_value = {'success': True, 'result': 'test'}
        
        workflow_configs = [
            {'workflow_id': 'seq_1'},
            {'workflow_id': 'seq_2'}
        ]
        
        plan = self.engine.create_orchestration_plan(
            name="Sequential Test",
            description="Test sequential execution",
            strategy=OrchestrationStrategy.SEQUENTIAL,
            workflow_configs=workflow_configs
        )
        
        result = self.engine.execute_orchestration_plan(plan.plan_id)
        
        assert result['success'] is True
        assert result['strategy'] == 'sequential'
        assert len(result['results']) == 2
        assert mock_execute.call_count == 2
    
    @patch('core.orchestration.enhanced_workflow_orchestrator.OrchestrationEngine._execute_single_workflow')
    def test_execute_parallel_strategy(self, mock_execute):
        """Test parallel execution strategy."""
        mock_execute.return_value = {'success': True, 'result': 'test'}
        
        workflow_configs = [
            {'workflow_id': 'par_1'},
            {'workflow_id': 'par_2'},
            {'workflow_id': 'par_3'}
        ]
        
        plan = self.engine.create_orchestration_plan(
            name="Parallel Test",
            description="Test parallel execution",
            strategy=OrchestrationStrategy.PARALLEL,
            workflow_configs=workflow_configs
        )
        
        result = self.engine.execute_orchestration_plan(plan.plan_id)
        
        assert result['success'] is True
        assert result['strategy'] == 'parallel'
        assert len(result['results']) == 3
    
    def test_execute_nonexistent_plan(self):
        """Test executing a non-existent plan."""
        with pytest.raises(ValueError, match="Plan .* not found"):
            self.engine.execute_orchestration_plan("nonexistent_plan")
    
    def test_get_orchestration_status(self):
        """Test getting orchestration status."""
        status = self.engine.get_orchestration_status()
        
        assert 'active_plans' in status
        assert 'scheduler_status' in status
        assert 'resource_utilization' in status
        assert 'execution_history_count' in status
        assert 'running' in status
        assert status['running'] is False  # Not started yet
    
    def test_orchestration_loop_lifecycle(self):
        """Test starting and stopping the orchestration loop."""
        # Start the loop
        self.engine.start_orchestration_loop()
        assert self.engine.running is True
        assert self.engine.orchestration_thread is not None
        
        # Stop the loop
        self.engine.stop_orchestration_loop()
        assert self.engine.running is False

class TestEnhancedWorkflowOrchestrator:
    """Test suite for EnhancedWorkflowOrchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the integration service
        self.mock_integration_service = Mock(spec=EnhancedIntegrationService)
        self.mock_integration_service.langgraph_integration = Mock()
        self.mock_integration_service.nlp_processor = Mock()
        self.mock_integration_service.state_manager = Mock()
        
        self.orchestrator = EnhancedWorkflowOrchestrator(
            integration_service=self.mock_integration_service
        )
    
    def test_initialization(self):
        """Test EnhancedWorkflowOrchestrator initialization."""
        assert self.orchestrator.integration_service == self.mock_integration_service
        assert isinstance(self.orchestrator.orchestration_engine, OrchestrationEngine)
        assert len(self.orchestrator.workflow_templates) == 0
        assert len(self.orchestrator.orchestration_history) == 0
    
    def test_create_research_orchestration(self):
        """Test creating a research orchestration plan."""
        research_queries = [
            "What is machine learning?",
            "How does neural networks work?",
            "What are the applications of AI?"
        ]
        
        plan_id = self.orchestrator.create_research_orchestration(
            research_queries,
            priority=WorkflowPriority.HIGH
        )
        
        assert plan_id is not None
        assert plan_id in self.orchestrator.orchestration_engine.active_plans
        
        plan = self.orchestrator.orchestration_engine.active_plans[plan_id]
        assert len(plan.workflows) == 3
        assert plan.strategy == OrchestrationStrategy.PARALLEL
        assert "Research Orchestration" in plan.name
    
    def test_create_analysis_pipeline(self):
        """Test creating an analysis pipeline."""
        data_sources = ["source1.csv", "source2.json", "source3.xml"]
        analysis_types = ["statistical", "sentiment", "trend"]
        
        plan_id = self.orchestrator.create_analysis_pipeline(
            data_sources,
            analysis_types
        )
        
        assert plan_id is not None
        assert plan_id in self.orchestrator.orchestration_engine.active_plans
        
        plan = self.orchestrator.orchestration_engine.active_plans[plan_id]
        # Should have 3 collection + 3 analysis workflows
        assert len(plan.workflows) == 6
        assert plan.strategy == OrchestrationStrategy.CONDITIONAL
        
        # Check dependencies
        analysis_workflows = [w for w in plan.workflows if 'analyze_' in w.workflow_id]
        for analysis_wf in analysis_workflows:
            assert len(analysis_wf.dependencies) == 3  # Depends on all collection workflows
    
    @patch('core.orchestration.enhanced_workflow_orchestrator.OrchestrationEngine.execute_orchestration_plan')
    def test_execute_plan(self, mock_execute):
        """Test executing a plan."""
        mock_execute.return_value = {'success': True, 'results': []}
        
        # Create a simple plan first
        plan_id = self.orchestrator.create_research_orchestration(["test query"])
        
        # Execute it
        result = self.orchestrator.execute_plan(plan_id)
        
        assert result['success'] is True
        mock_execute.assert_called_once_with(plan_id)
    
    def test_get_orchestration_insights(self):
        """Test getting orchestration insights."""
        insights = self.orchestrator.get_orchestration_insights()
        
        assert 'active_plans' in insights
        assert 'scheduler_status' in insights
        assert 'resource_utilization' in insights
        assert 'execution_history_count' in insights
        assert 'running' in insights
    
    def test_orchestrator_lifecycle(self):
        """Test starting and stopping the orchestrator."""
        # Start
        self.orchestrator.start()
        assert self.orchestrator.orchestration_engine.running is True
        
        # Stop
        self.orchestrator.stop()
        assert self.orchestrator.orchestration_engine.running is False

class TestIntegrationScenarios:
    """Test suite for integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = EnhancedWorkflowOrchestrator()
    
    def test_resource_contention_scenario(self):
        """Test scenario with resource contention."""
        # Create multiple resource-heavy workflows
        workflow_configs = [
            {
                'workflow_id': f'heavy_workflow_{i}',
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 40.0},
                    {'resource_type': ResourceType.MEMORY.value, 'amount': 30.0}
                ]
            }
            for i in range(5)  # 5 workflows, but only enough resources for ~2
        ]
        
        plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
            name="Resource Contention Test",
            description="Test resource management under contention",
            strategy=OrchestrationStrategy.RESOURCE_AWARE,
            workflow_configs=workflow_configs
        )
        
        # The plan should be created successfully
        assert plan.plan_id in self.orchestrator.orchestration_engine.active_plans
        assert len(plan.workflows) == 5
    
    def test_dependency_chain_scenario(self):
        """Test scenario with complex dependency chains."""
        workflow_configs = [
            {'workflow_id': 'root'},
            {'workflow_id': 'child1', 'dependencies': ['root']},
            {'workflow_id': 'child2', 'dependencies': ['root']},
            {'workflow_id': 'grandchild1', 'dependencies': ['child1', 'child2']},
            {'workflow_id': 'grandchild2', 'dependencies': ['child1']}
        ]
        
        plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
            name="Dependency Chain Test",
            description="Test complex dependency management",
            strategy=OrchestrationStrategy.CONDITIONAL,
            workflow_configs=workflow_configs
        )
        
        assert len(plan.workflows) == 5
        
        # Verify dependency structure
        workflows_by_id = {w.workflow_id: w for w in plan.workflows}
        assert len(workflows_by_id['root'].dependencies) == 0
        assert 'root' in workflows_by_id['child1'].dependencies
        assert 'root' in workflows_by_id['child2'].dependencies
        assert set(workflows_by_id['grandchild1'].dependencies) == {'child1', 'child2'}
        assert 'child1' in workflows_by_id['grandchild2'].dependencies
    
    def test_mixed_priority_scenario(self):
        """Test scenario with mixed priority workflows."""
        workflow_configs = [
            {'workflow_id': 'background_task', 'priority': WorkflowPriority.BACKGROUND.value},
            {'workflow_id': 'normal_task', 'priority': WorkflowPriority.NORMAL.value},
            {'workflow_id': 'urgent_task', 'priority': WorkflowPriority.CRITICAL.value},
            {'workflow_id': 'high_task', 'priority': WorkflowPriority.HIGH.value}
        ]
        
        plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
            name="Mixed Priority Test",
            description="Test priority-based execution",
            strategy=OrchestrationStrategy.PRIORITY_BASED,
            workflow_configs=workflow_configs
        )
        
        # Schedule all workflows
        for workflow_schedule in plan.workflows:
            self.orchestrator.orchestration_engine.scheduler.schedule_workflow(workflow_schedule)
        
        # Get workflows in priority order
        execution_order = []
        while True:
            next_workflow = self.orchestrator.orchestration_engine.scheduler.get_next_workflow()
            if next_workflow is None:
                break
            execution_order.append(next_workflow.workflow_id)
        
        # Should execute in priority order: critical, high, normal, background
        expected_order = ['urgent_task', 'high_task', 'normal_task', 'background_task']
        assert execution_order == expected_order

class TestPerformanceAndStress:
    """Test suite for performance and stress testing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = EnhancedWorkflowOrchestrator()
    
    def test_large_number_of_workflows(self):
        """Test handling a large number of workflows."""
        num_workflows = 100
        
        workflow_configs = [
            {
                'workflow_id': f'workflow_{i}',
                'priority': WorkflowPriority.NORMAL.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 1.0}
                ]
            }
            for i in range(num_workflows)
        ]
        
        start_time = time.time()
        
        plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
            name="Large Scale Test",
            description="Test with many workflows",
            strategy=OrchestrationStrategy.PARALLEL,
            workflow_configs=workflow_configs
        )
        
        creation_time = time.time() - start_time
        
        assert len(plan.workflows) == num_workflows
        assert creation_time < 5.0  # Should create 100 workflows in under 5 seconds
    
    def test_concurrent_plan_creation(self):
        """Test concurrent plan creation."""
        def create_plan(plan_index: int):
            workflow_configs = [
                {
                    'workflow_id': f'plan_{plan_index}_workflow_{i}',
                    'priority': WorkflowPriority.NORMAL.value
                }
                for i in range(5)
            ]
            
            return self.orchestrator.orchestration_engine.create_orchestration_plan(
                name=f"Concurrent Plan {plan_index}",
                description=f"Plan created concurrently #{plan_index}",
                strategy=OrchestrationStrategy.SEQUENTIAL,
                workflow_configs=workflow_configs
            )
        
        # Create multiple plans concurrently
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_plan, i) for i in range(10)]
            plans = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        assert len(plans) == 10
        assert len(self.orchestrator.orchestration_engine.active_plans) == 10
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and execute many small plans
        for i in range(50):
            workflow_configs = [
                {
                    'workflow_id': f'memory_test_{i}_workflow_{j}',
                    'priority': WorkflowPriority.NORMAL.value
                }
                for j in range(3)
            ]
            
            plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
                name=f"Memory Test Plan {i}",
                description="Testing memory usage",
                strategy=OrchestrationStrategy.SEQUENTIAL,
                workflow_configs=workflow_configs
            )
            
            # Clean up immediately
            del self.orchestrator.orchestration_engine.active_plans[plan.plan_id]
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50

class TestDefaultOrchestrator:
    """Test suite for the default orchestrator instance."""
    
    def test_default_orchestrator_exists(self):
        """Test that default orchestrator is available."""
        assert default_orchestrator is not None
        assert isinstance(default_orchestrator, EnhancedWorkflowOrchestrator)
    
    def test_default_orchestrator_functionality(self):
        """Test basic functionality of default orchestrator."""
        # Test creating a simple research orchestration
        plan_id = default_orchestrator.create_research_orchestration(
            ["Test query for default orchestrator"]
        )
        
        assert plan_id is not None
        assert plan_id in default_orchestrator.orchestration_engine.active_plans
    
    def test_default_orchestrator_insights(self):
        """Test getting insights from default orchestrator."""
        insights = default_orchestrator.get_orchestration_insights()
        
        assert isinstance(insights, dict)
        assert 'active_plans' in insights
        assert 'scheduler_status' in insights

if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        
        if test_category == "resource":
            pytest.main(["-v", "TestResourceManager"])
        elif test_category == "scheduler":
            pytest.main(["-v", "TestWorkflowScheduler"])
        elif test_category == "engine":
            pytest.main(["-v", "TestOrchestrationEngine"])
        elif test_category == "orchestrator":
            pytest.main(["-v", "TestEnhancedWorkflowOrchestrator"])
        elif test_category == "integration":
            pytest.main(["-v", "TestIntegrationScenarios"])
        elif test_category == "performance":
            pytest.main(["-v", "TestPerformanceAndStress"])
        else:
            print("Available test categories: resource, scheduler, engine, orchestrator, integration, performance")
    else:
        # Run all tests
        pytest.main(["-v", __file__])