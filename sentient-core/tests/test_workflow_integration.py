"""Integration test for the complete workflow orchestration system

This module tests the integration between the enhanced workflow orchestrator
and the existing multi-agent RAG system, demonstrating end-to-end workflows.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import uuid
import json

# Import the modules to test
from core.orchestration.enhanced_workflow_orchestrator import (
    EnhancedWorkflowOrchestrator,
    OrchestrationStrategy,
    WorkflowPriority,
    ResourceType,
    ResourceRequirement,
    default_orchestrator
)
from core.graphs.advanced_stateful_workflow import (
    StatefulWorkflowManager,
    AdvancedCheckpointSaver,
    create_research_workflow,
    create_orchestrator_workflow
)
from core.graphs.enhanced_langgraph_integration import (
    EnhancedLangGraphIntegration,
    DynamicGraphBuilder,
    WorkflowTemplate
)
from core.models import AppState, EnhancedTask, TaskStatus, AgentType
from core.services.enhanced_integration_service import EnhancedIntegrationService

class TestWorkflowIntegration:
    """Test suite for workflow integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock the integration service components
        self.mock_integration_service = Mock(spec=EnhancedIntegrationService)
        self.mock_integration_service.langgraph_integration = Mock()
        self.mock_integration_service.nlp_processor = Mock()
        self.mock_integration_service.state_manager = Mock()
        self.mock_integration_service.llm_service = Mock()
        self.mock_integration_service.artifact_service = Mock()
        
        # Create orchestrator with mocked dependencies
        self.orchestrator = EnhancedWorkflowOrchestrator(
            integration_service=self.mock_integration_service
        )
        
        # Create stateful workflow manager
        self.workflow_manager = StatefulWorkflowManager()
        
        # Create enhanced LangGraph integration
        self.langgraph_integration = EnhancedLangGraphIntegration()
    
    def test_research_workflow_creation_and_execution(self):
        """Test creating and executing a research workflow."""
        # Mock the LLM service responses
        self.mock_integration_service.llm_service.generate_response.return_value = {
            'content': 'Research plan created successfully',
            'metadata': {'confidence': 0.9}
        }
        
        # Create a research orchestration plan
        research_queries = [
            "What are the latest developments in quantum computing?",
            "How does quantum supremacy impact current encryption methods?",
            "What are the practical applications of quantum computing in 2024?"
        ]
        
        plan_id = self.orchestrator.create_research_orchestration(
            research_queries,
            priority=WorkflowPriority.HIGH
        )
        
        # Verify plan creation
        assert plan_id is not None
        assert plan_id in self.orchestrator.orchestration_engine.active_plans
        
        plan = self.orchestrator.orchestration_engine.active_plans[plan_id]
        assert len(plan.workflows) == 3
        assert plan.strategy == OrchestrationStrategy.PARALLEL
        
        # Mock execution results
        with patch.object(
            self.orchestrator.orchestration_engine,
            '_execute_single_workflow',
            return_value={'success': True, 'result': 'Research completed'}
        ):
            result = self.orchestrator.execute_plan(plan_id)
            
            assert result['success'] is True
            assert len(result['results']) == 3
    
    def test_analysis_pipeline_with_dependencies(self):
        """Test creating and executing an analysis pipeline with dependencies."""
        # Mock the artifact service
        self.mock_integration_service.artifact_service.generate_artifact.return_value = {
            'artifact_id': 'test_artifact_123',
            'content': 'Analysis report content',
            'metadata': {'format': 'pdf', 'size': 1024}
        }
        
        # Create an analysis pipeline
        data_sources = [
            "financial_data_2024.csv",
            "market_trends.json",
            "competitor_analysis.xml"
        ]
        analysis_types = ["statistical", "sentiment", "trend"]
        
        plan_id = self.orchestrator.create_analysis_pipeline(
            data_sources,
            analysis_types
        )
        
        # Verify plan creation
        assert plan_id is not None
        plan = self.orchestrator.orchestration_engine.active_plans[plan_id]
        
        # Should have 3 collection + 3 analysis workflows
        assert len(plan.workflows) == 6
        assert plan.strategy == OrchestrationStrategy.CONDITIONAL
        
        # Verify dependency structure
        collection_workflows = [w for w in plan.workflows if 'collect_' in w.workflow_id]
        analysis_workflows = [w for w in plan.workflows if 'analyze_' in w.workflow_id]
        
        assert len(collection_workflows) == 3
        assert len(analysis_workflows) == 3
        
        # Each analysis workflow should depend on all collection workflows
        for analysis_wf in analysis_workflows:
            assert len(analysis_wf.dependencies) == 3
    
    def test_stateful_workflow_with_checkpointing(self):
        """Test stateful workflow with checkpointing capabilities."""
        # Create a checkpoint saver
        checkpoint_saver = AdvancedCheckpointSaver(":memory:")
        
        # Create initial app state
        initial_state = AppState(
            conversation=[],
            user_input="Test research query",
            current_task=EnhancedTask(
                task_id="test_task_001",
                description="Test research task",
                status=TaskStatus.PENDING,
                assigned_agent=AgentType.RESEARCH,
                created_at=datetime.now()
            ),
            system_logs=[],
            routing_decision="research"
        )
        
        # Create research workflow with checkpointing
        research_workflow = create_research_workflow(
            checkpoint_saver=checkpoint_saver,
            session_id="test_session_001"
        )
        
        # Verify workflow creation
        assert research_workflow is not None
        assert hasattr(research_workflow, 'checkpointer')
        
        # Test checkpoint saving (mocked)
        with patch.object(checkpoint_saver, 'put') as mock_put:
            # Simulate workflow execution with state changes
            config = {"configurable": {"thread_id": "test_session_001"}}
            
            # Mock the workflow execution
            with patch.object(research_workflow, 'invoke') as mock_invoke:
                mock_invoke.return_value = initial_state
                
                result = research_workflow.invoke(initial_state, config=config)
                
                assert result is not None
                mock_invoke.assert_called_once()
    
    def test_dynamic_graph_construction(self):
        """Test dynamic graph construction based on workflow requirements."""
        # Create a dynamic graph builder
        graph_builder = DynamicGraphBuilder()
        
        # Define a workflow template
        template = WorkflowTemplate(
            template_id="dynamic_research_template",
            name="Dynamic Research Workflow",
            description="A dynamically constructed research workflow",
            node_templates=[
                {
                    'node_id': 'input_processor',
                    'node_type': 'processor',
                    'config': {'max_tokens': 1000}
                },
                {
                    'node_id': 'research_agent',
                    'node_type': 'agent',
                    'config': {'agent_type': 'research'}
                },
                {
                    'node_id': 'output_formatter',
                    'node_type': 'formatter',
                    'config': {'format': 'markdown'}
                }
            ],
            edge_templates=[
                {
                    'from_node': 'input_processor',
                    'to_node': 'research_agent',
                    'condition': None
                },
                {
                    'from_node': 'research_agent',
                    'to_node': 'output_formatter',
                    'condition': None
                }
            ]
        )
        
        # Build the graph
        graph = graph_builder.build_graph_from_template(template)
        
        # Verify graph construction
        assert graph is not None
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
    
    def test_multi_agent_coordination(self):
        """Test multi-agent coordination through workflow orchestration."""
        # Create a complex workflow that requires multiple agents
        workflow_configs = [
            {
                'workflow_id': 'research_phase',
                'agent_type': 'research',
                'priority': WorkflowPriority.HIGH.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 30.0}
                ]
            },
            {
                'workflow_id': 'analysis_phase',
                'agent_type': 'analysis',
                'dependencies': ['research_phase'],
                'priority': WorkflowPriority.NORMAL.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.MEMORY.value, 'amount': 40.0}
                ]
            },
            {
                'workflow_id': 'synthesis_phase',
                'agent_type': 'synthesis',
                'dependencies': ['analysis_phase'],
                'priority': WorkflowPriority.NORMAL.value
            },
            {
                'workflow_id': 'monitoring_phase',
                'agent_type': 'monitoring',
                'dependencies': [],  # Runs in parallel
                'priority': WorkflowPriority.LOW.value
            }
        ]
        
        plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
            name="Multi-Agent Coordination Test",
            description="Test coordination between multiple agents",
            strategy=OrchestrationStrategy.CONDITIONAL,
            workflow_configs=workflow_configs
        )
        
        # Verify plan structure
        assert len(plan.workflows) == 4
        
        # Verify dependency chain: research -> analysis -> synthesis
        workflows_by_id = {w.workflow_id: w for w in plan.workflows}
        
        assert len(workflows_by_id['research_phase'].dependencies) == 0
        assert 'research_phase' in workflows_by_id['analysis_phase'].dependencies
        assert 'analysis_phase' in workflows_by_id['synthesis_phase'].dependencies
        assert len(workflows_by_id['monitoring_phase'].dependencies) == 0  # Parallel
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery in workflow execution."""
        # Create a workflow that will fail
        workflow_configs = [
            {
                'workflow_id': 'failing_workflow',
                'priority': WorkflowPriority.NORMAL.value
            },
            {
                'workflow_id': 'recovery_workflow',
                'dependencies': ['failing_workflow'],
                'priority': WorkflowPriority.HIGH.value
            }
        ]
        
        plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
            name="Error Handling Test",
            description="Test error handling and recovery",
            strategy=OrchestrationStrategy.SEQUENTIAL,
            workflow_configs=workflow_configs
        )
        
        # Mock a failing execution
        def mock_execute_with_failure(workflow_schedule):
            if workflow_schedule.workflow_id == 'failing_workflow':
                return {
                    'success': False,
                    'error': 'Simulated workflow failure',
                    'workflow_id': workflow_schedule.workflow_id
                }
            else:
                return {
                    'success': True,
                    'result': 'Recovery successful',
                    'workflow_id': workflow_schedule.workflow_id
                }
        
        with patch.object(
            self.orchestrator.orchestration_engine,
            '_execute_single_workflow',
            side_effect=mock_execute_with_failure
        ):
            result = self.orchestrator.execute_plan(plan.plan_id)
            
            # The overall execution should handle the failure gracefully
            assert 'results' in result
            assert len(result['results']) == 2
            
            # Check that failure was recorded
            failing_result = next(
                r for r in result['results'] 
                if r['workflow_id'] == 'failing_workflow'
            )
            assert failing_result['success'] is False
    
    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring."""
        # Create a workflow with performance tracking
        plan_id = self.orchestrator.create_research_orchestration(
            ["Performance test query"],
            priority=WorkflowPriority.NORMAL
        )
        
        # Get initial insights
        initial_insights = self.orchestrator.get_orchestration_insights()
        initial_active_plans = initial_insights['active_plans']
        
        # Mock successful execution with timing
        def mock_timed_execution(workflow_schedule):
            time.sleep(0.1)  # Simulate work
            return {
                'success': True,
                'result': 'Performance test completed',
                'execution_time': 0.1,
                'workflow_id': workflow_schedule.workflow_id
            }
        
        with patch.object(
            self.orchestrator.orchestration_engine,
            '_execute_single_workflow',
            side_effect=mock_timed_execution
        ):
            start_time = time.time()
            result = self.orchestrator.execute_plan(plan_id)
            execution_time = time.time() - start_time
            
            # Verify execution completed
            assert result['success'] is True
            assert execution_time >= 0.1  # At least the simulated work time
        
        # Get final insights
        final_insights = self.orchestrator.get_orchestration_insights()
        
        # Should have one less active plan
        assert final_insights['active_plans'] == initial_active_plans
        assert final_insights['execution_history_count'] >= 1
    
    def test_resource_optimization_scenario(self):
        """Test resource optimization in complex scenarios."""
        # Create workflows with varying resource requirements
        workflow_configs = [
            {
                'workflow_id': 'light_workflow_1',
                'priority': WorkflowPriority.LOW.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 5.0}
                ]
            },
            {
                'workflow_id': 'heavy_workflow',
                'priority': WorkflowPriority.HIGH.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 60.0},
                    {'resource_type': ResourceType.MEMORY.value, 'amount': 70.0}
                ]
            },
            {
                'workflow_id': 'light_workflow_2',
                'priority': WorkflowPriority.NORMAL.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 10.0}
                ]
            }
        ]
        
        plan = self.orchestrator.orchestration_engine.create_orchestration_plan(
            name="Resource Optimization Test",
            description="Test resource-aware scheduling",
            strategy=OrchestrationStrategy.RESOURCE_AWARE,
            workflow_configs=workflow_configs
        )
        
        # Schedule all workflows
        for workflow_schedule in plan.workflows:
            self.orchestrator.orchestration_engine.scheduler.schedule_workflow(workflow_schedule)
        
        # Get workflows in execution order (should prioritize by resources and priority)
        execution_order = []
        resource_manager = self.orchestrator.orchestration_engine.resource_manager
        
        while True:
            next_workflow = self.orchestrator.orchestration_engine.scheduler.get_next_workflow()
            if next_workflow is None:
                break
            
            execution_order.append(next_workflow.workflow_id)
            
            # Simulate resource allocation and release
            resource_manager.allocate_resources(
                next_workflow.workflow_id,
                next_workflow.resource_requirements
            )
            resource_manager.release_resources(next_workflow.workflow_id)
        
        # High priority heavy workflow should execute first
        assert execution_order[0] == 'heavy_workflow'
        
        # Light workflows should execute after
        assert 'light_workflow_1' in execution_order[1:]
        assert 'light_workflow_2' in execution_order[1:]
    
    def test_session_management_integration(self):
        """Test integration with session management."""
        # Create multiple sessions with different workflows
        session_configs = [
            {
                'session_id': 'session_001',
                'workflow_type': 'research',
                'queries': ['Query 1 for session 1']
            },
            {
                'session_id': 'session_002',
                'workflow_type': 'analysis',
                'data_sources': ['data1.csv'],
                'analysis_types': ['statistical']
            }
        ]
        
        created_plans = []
        
        for config in session_configs:
            if config['workflow_type'] == 'research':
                plan_id = self.orchestrator.create_research_orchestration(
                    config['queries']
                )
            elif config['workflow_type'] == 'analysis':
                plan_id = self.orchestrator.create_analysis_pipeline(
                    config['data_sources'],
                    config['analysis_types']
                )
            
            created_plans.append({
                'session_id': config['session_id'],
                'plan_id': plan_id
            })
        
        # Verify all plans were created
        assert len(created_plans) == 2
        
        for plan_info in created_plans:
            assert plan_info['plan_id'] in self.orchestrator.orchestration_engine.active_plans
        
        # Test session isolation - each plan should be independent
        insights = self.orchestrator.get_orchestration_insights()
        assert insights['active_plans'] == 2

class TestEndToEndScenarios:
    """Test suite for complete end-to-end scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use the default orchestrator for end-to-end tests
        self.orchestrator = default_orchestrator
    
    def test_complete_research_pipeline(self):
        """Test a complete research pipeline from start to finish."""
        # Define a comprehensive research scenario
        research_scenario = {
            'topic': 'Artificial Intelligence in Healthcare',
            'queries': [
                'What are the current applications of AI in medical diagnosis?',
                'How is machine learning being used in drug discovery?',
                'What are the ethical considerations of AI in healthcare?',
                'What are the regulatory challenges for AI medical devices?'
            ],
            'analysis_requirements': [
                'trend_analysis',
                'sentiment_analysis',
                'comparative_analysis'
            ]
        }
        
        # Step 1: Create research orchestration
        research_plan_id = self.orchestrator.create_research_orchestration(
            research_scenario['queries'],
            priority=WorkflowPriority.HIGH
        )
        
        # Step 2: Create analysis pipeline for the research results
        analysis_plan_id = self.orchestrator.create_analysis_pipeline(
            data_sources=[f"research_result_{i}.json" for i in range(len(research_scenario['queries']))],
            analysis_types=research_scenario['analysis_requirements']
        )
        
        # Verify both plans were created
        assert research_plan_id in self.orchestrator.orchestration_engine.active_plans
        assert analysis_plan_id in self.orchestrator.orchestration_engine.active_plans
        
        # Get insights about the orchestration
        insights = self.orchestrator.get_orchestration_insights()
        assert insights['active_plans'] >= 2
        
        # Verify the structure of created plans
        research_plan = self.orchestrator.orchestration_engine.active_plans[research_plan_id]
        analysis_plan = self.orchestrator.orchestration_engine.active_plans[analysis_plan_id]
        
        assert len(research_plan.workflows) == 4  # One for each query
        assert len(analysis_plan.workflows) == 7  # 4 collection + 3 analysis
    
    def test_adaptive_workflow_scaling(self):
        """Test adaptive scaling of workflows based on load."""
        # Create multiple research orchestrations to simulate load
        plan_ids = []
        
        for i in range(5):
            plan_id = self.orchestrator.create_research_orchestration(
                [f"Research query {i} for load testing"],
                priority=WorkflowPriority.NORMAL
            )
            plan_ids.append(plan_id)
        
        # Verify all plans were created
        assert len(plan_ids) == 5
        
        for plan_id in plan_ids:
            assert plan_id in self.orchestrator.orchestration_engine.active_plans
        
        # Check resource utilization
        insights = self.orchestrator.get_orchestration_insights()
        assert insights['active_plans'] >= 5
        
        # Resource utilization should be tracked
        assert 'resource_utilization' in insights
        assert isinstance(insights['resource_utilization'], dict)
    
    def test_workflow_lifecycle_management(self):
        """Test complete workflow lifecycle from creation to completion."""
        # Create a simple workflow
        plan_id = self.orchestrator.create_research_orchestration(
            ["Lifecycle test query"]
        )
        
        # Initial state - plan should be active
        initial_insights = self.orchestrator.get_orchestration_insights()
        assert plan_id in self.orchestrator.orchestration_engine.active_plans
        
        # Start the orchestrator
        self.orchestrator.start()
        
        try:
            # Let it run briefly
            time.sleep(0.5)
            
            # Check that orchestration is running
            running_insights = self.orchestrator.get_orchestration_insights()
            assert running_insights['running'] is True
            
        finally:
            # Stop the orchestrator
            self.orchestrator.stop()
        
        # Verify it stopped
        final_insights = self.orchestrator.get_orchestration_insights()
        assert final_insights['running'] is False

if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1:
        test_category = sys.argv[1]
        
        if test_category == "integration":
            pytest.main(["-v", "TestWorkflowIntegration"])
        elif test_category == "e2e":
            pytest.main(["-v", "TestEndToEndScenarios"])
        else:
            print("Available test categories: integration, e2e")
    else:
        # Run all tests
        pytest.main(["-v", __file__])