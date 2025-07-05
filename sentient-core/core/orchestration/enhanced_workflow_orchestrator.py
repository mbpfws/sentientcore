"""Enhanced Workflow Orchestrator

This module provides an advanced orchestration layer that coordinates multiple
workflows, manages complex multi-agent interactions, and provides intelligent
workflow scheduling and resource management.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Type, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from pathlib import Path
import threading
from contextlib import contextmanager
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import heapq
import time
from abc import ABC, abstractmethod

from core.models import AppState, EnhancedTask, AgentType, TaskStatus, SessionState
from core.graphs.enhanced_langgraph_integration import (
    EnhancedLangGraphIntegration, DynamicGraphBuilder, WorkflowTemplate,
    GraphTopology, RoutingStrategy
)
from core.graphs.advanced_stateful_workflow import (
    StatefulWorkflowManager, WorkflowMetrics, WorkflowStatus
)
from core.services.enhanced_state_manager import EnhancedStateManager
from core.nlp.advanced_nlp_processor import AdvancedNLPProcessor
from core.services.enhanced_integration_service import EnhancedIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestrationStrategy(str, Enum):
    """Different orchestration strategies."""
    SEQUENTIAL = "sequential"          # Execute workflows one after another
    PARALLEL = "parallel"              # Execute workflows in parallel
    CONDITIONAL = "conditional"        # Execute based on conditions
    PRIORITY_BASED = "priority_based"  # Execute based on priority
    RESOURCE_AWARE = "resource_aware"  # Execute based on resource availability
    ADAPTIVE = "adaptive"              # Dynamically adapt strategy

class WorkflowPriority(int, Enum):
    """Workflow execution priorities."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class ResourceType(str, Enum):
    """Types of resources that can be managed."""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    CUSTOM = "custom"

@dataclass
class ResourceRequirement:
    """Resource requirement specification."""
    resource_type: ResourceType
    amount: float
    unit: str = "units"
    max_amount: Optional[float] = None
    priority: WorkflowPriority = WorkflowPriority.NORMAL

@dataclass
class WorkflowSchedule:
    """Workflow scheduling information."""
    workflow_id: str
    scheduled_time: datetime
    priority: WorkflowPriority
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: List[ResourceRequirement] = field(default_factory=list)
    max_execution_time: Optional[timedelta] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrchestrationPlan:
    """Complete orchestration plan for multiple workflows."""
    plan_id: str
    name: str
    description: str
    strategy: OrchestrationStrategy
    workflows: List[WorkflowSchedule]
    global_timeout: Optional[timedelta] = None
    rollback_strategy: Optional[str] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class ResourceManager:
    """Manages system resources for workflow execution."""
    
    def __init__(self, initial_resources: Optional[Dict[ResourceType, float]] = None):
        self.available_resources: Dict[ResourceType, float] = initial_resources or {
            ResourceType.CPU: 100.0,
            ResourceType.MEMORY: 100.0,
            ResourceType.NETWORK: 100.0,
            ResourceType.STORAGE: 100.0,
            ResourceType.API_CALLS: 1000.0
        }
        
        self.allocated_resources: Dict[str, Dict[ResourceType, float]] = {}
        self.resource_history: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        logger.info("Resource Manager initialized")
    
    def check_resource_availability(self, requirements: List[ResourceRequirement]) -> bool:
        """Check if required resources are available."""
        with self.lock:
            for req in requirements:
                available = self.available_resources.get(req.resource_type, 0.0)
                if available < req.amount:
                    return False
            return True
    
    def allocate_resources(self, workflow_id: str, requirements: List[ResourceRequirement]) -> bool:
        """Allocate resources for a workflow."""
        with self.lock:
            # Check availability first
            if not self.check_resource_availability(requirements):
                return False
            
            # Allocate resources
            allocated = {}
            for req in requirements:
                self.available_resources[req.resource_type] -= req.amount
                allocated[req.resource_type] = req.amount
            
            self.allocated_resources[workflow_id] = allocated
            
            # Record allocation
            self.resource_history.append({
                'timestamp': datetime.now(),
                'action': 'allocate',
                'workflow_id': workflow_id,
                'resources': allocated.copy()
            })
            
            logger.info(f"Resources allocated for workflow {workflow_id}: {allocated}")
            return True
    
    def release_resources(self, workflow_id: str) -> bool:
        """Release resources allocated to a workflow."""
        with self.lock:
            if workflow_id not in self.allocated_resources:
                return False
            
            allocated = self.allocated_resources[workflow_id]
            
            # Release resources
            for resource_type, amount in allocated.items():
                self.available_resources[resource_type] += amount
            
            del self.allocated_resources[workflow_id]
            
            # Record release
            self.resource_history.append({
                'timestamp': datetime.now(),
                'action': 'release',
                'workflow_id': workflow_id,
                'resources': allocated.copy()
            })
            
            logger.info(f"Resources released for workflow {workflow_id}: {allocated}")
            return True
    
    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization percentages."""
        with self.lock:
            utilization = {}
            for resource_type, total in self.available_resources.items():
                allocated_total = sum(
                    allocation.get(resource_type, 0.0)
                    for allocation in self.allocated_resources.values()
                )
                utilization[resource_type] = (allocated_total / (total + allocated_total)) * 100
            
            return utilization

class WorkflowScheduler:
    """Intelligent workflow scheduler."""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.scheduled_workflows: List[Tuple[float, WorkflowSchedule]] = []  # Priority queue
        self.running_workflows: Dict[str, Dict[str, Any]] = {}
        self.completed_workflows: Dict[str, Dict[str, Any]] = {}
        self.failed_workflows: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        logger.info("Workflow Scheduler initialized")
    
    def schedule_workflow(self, schedule: WorkflowSchedule) -> bool:
        """Schedule a workflow for execution."""
        with self.lock:
            # Calculate priority score (lower is higher priority)
            priority_score = self._calculate_priority_score(schedule)
            
            # Add to priority queue
            heapq.heappush(self.scheduled_workflows, (priority_score, schedule))
            
            logger.info(f"Workflow {schedule.workflow_id} scheduled with priority {priority_score}")
            return True
    
    def get_next_workflow(self) -> Optional[WorkflowSchedule]:
        """Get the next workflow to execute based on priority and resource availability."""
        with self.lock:
            while self.scheduled_workflows:
                priority_score, schedule = heapq.heappop(self.scheduled_workflows)
                
                # Check if it's time to execute
                if schedule.scheduled_time > datetime.now():
                    # Put it back and wait
                    heapq.heappush(self.scheduled_workflows, (priority_score, schedule))
                    return None
                
                # Check dependencies
                if self._check_dependencies(schedule):
                    # Check resource availability
                    if self.resource_manager.check_resource_availability(schedule.resource_requirements):
                        return schedule
                    else:
                        # Put back with slightly lower priority
                        heapq.heappush(self.scheduled_workflows, (priority_score + 0.1, schedule))
                else:
                    # Dependencies not met, put back
                    heapq.heappush(self.scheduled_workflows, (priority_score + 0.05, schedule))
            
            return None
    
    def mark_workflow_running(self, workflow_id: str, start_time: datetime):
        """Mark a workflow as currently running."""
        with self.lock:
            self.running_workflows[workflow_id] = {
                'start_time': start_time,
                'status': 'running'
            }
    
    def mark_workflow_completed(self, workflow_id: str, end_time: datetime, success: bool, result: Any = None):
        """Mark a workflow as completed."""
        with self.lock:
            if workflow_id in self.running_workflows:
                workflow_info = self.running_workflows[workflow_id]
                workflow_info.update({
                    'end_time': end_time,
                    'duration': (end_time - workflow_info['start_time']).total_seconds(),
                    'success': success,
                    'result': result
                })
                
                if success:
                    self.completed_workflows[workflow_id] = workflow_info
                else:
                    self.failed_workflows[workflow_id] = workflow_info
                
                del self.running_workflows[workflow_id]
    
    def _calculate_priority_score(self, schedule: WorkflowSchedule) -> float:
        """Calculate priority score for a workflow (lower is higher priority)."""
        base_score = float(schedule.priority.value)
        
        # Adjust for scheduled time (earlier = higher priority)
        time_factor = max(0, (schedule.scheduled_time - datetime.now()).total_seconds() / 3600)
        
        # Adjust for resource requirements (less resources = higher priority)
        resource_factor = sum(req.amount for req in schedule.resource_requirements) / 100.0
        
        return base_score + time_factor * 0.1 + resource_factor * 0.01
    
    def _check_dependencies(self, schedule: WorkflowSchedule) -> bool:
        """Check if workflow dependencies are satisfied."""
        for dep_id in schedule.dependencies:
            if dep_id not in self.completed_workflows:
                return False
        return True
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        with self.lock:
            return {
                'scheduled_count': len(self.scheduled_workflows),
                'running_count': len(self.running_workflows),
                'completed_count': len(self.completed_workflows),
                'failed_count': len(self.failed_workflows),
                'resource_utilization': self.resource_manager.get_resource_utilization()
            }

class OrchestrationEngine:
    """Core orchestration engine for managing complex workflows."""
    
    def __init__(self,
                 langgraph_integration: Optional[EnhancedLangGraphIntegration] = None,
                 nlp_processor: Optional[AdvancedNLPProcessor] = None,
                 state_manager: Optional[EnhancedStateManager] = None):
        
        self.langgraph_integration = langgraph_integration or EnhancedLangGraphIntegration()
        self.nlp_processor = nlp_processor or AdvancedNLPProcessor()
        self.state_manager = state_manager or EnhancedStateManager()
        
        # Core components
        self.resource_manager = ResourceManager()
        self.scheduler = WorkflowScheduler(self.resource_manager)
        
        # Orchestration state
        self.active_plans: Dict[str, OrchestrationPlan] = {}
        self.plan_execution_history: List[Dict[str, Any]] = []
        
        # Execution control
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.running = False
        self.orchestration_thread: Optional[threading.Thread] = None
        
        logger.info("Orchestration Engine initialized")
    
    def create_orchestration_plan(self,
                                 name: str,
                                 description: str,
                                 strategy: OrchestrationStrategy,
                                 workflow_configs: List[Dict[str, Any]]) -> OrchestrationPlan:
        """Create a comprehensive orchestration plan."""
        
        plan_id = str(uuid.uuid4())
        
        # Create workflow schedules
        workflows = []
        for i, config in enumerate(workflow_configs):
            schedule = WorkflowSchedule(
                workflow_id=config.get('workflow_id', f"workflow_{i}"),
                scheduled_time=config.get('scheduled_time', datetime.now()),
                priority=WorkflowPriority(config.get('priority', WorkflowPriority.NORMAL.value)),
                dependencies=config.get('dependencies', []),
                resource_requirements=[
                    ResourceRequirement(**req) for req in config.get('resource_requirements', [])
                ],
                max_execution_time=config.get('max_execution_time'),
                retry_policy=config.get('retry_policy', {}),
                metadata=config.get('metadata', {})
            )
            workflows.append(schedule)
        
        plan = OrchestrationPlan(
            plan_id=plan_id,
            name=name,
            description=description,
            strategy=strategy,
            workflows=workflows,
            global_timeout=workflow_configs[0].get('global_timeout') if workflow_configs else None,
            rollback_strategy=workflow_configs[0].get('rollback_strategy') if workflow_configs else None
        )
        
        self.active_plans[plan_id] = plan
        logger.info(f"Orchestration plan created: {plan_id}")
        
        return plan
    
    def execute_orchestration_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute an orchestration plan."""
        
        if plan_id not in self.active_plans:
            raise ValueError(f"Plan {plan_id} not found")
        
        plan = self.active_plans[plan_id]
        start_time = datetime.now()
        
        try:
            # Schedule all workflows in the plan
            for workflow_schedule in plan.workflows:
                self.scheduler.schedule_workflow(workflow_schedule)
            
            # Execute based on strategy
            if plan.strategy == OrchestrationStrategy.SEQUENTIAL:
                result = self._execute_sequential(plan)
            elif plan.strategy == OrchestrationStrategy.PARALLEL:
                result = self._execute_parallel(plan)
            elif plan.strategy == OrchestrationStrategy.CONDITIONAL:
                result = self._execute_conditional(plan)
            elif plan.strategy == OrchestrationStrategy.PRIORITY_BASED:
                result = self._execute_priority_based(plan)
            elif plan.strategy == OrchestrationStrategy.RESOURCE_AWARE:
                result = self._execute_resource_aware(plan)
            else:  # ADAPTIVE
                result = self._execute_adaptive(plan)
            
            # Record execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.plan_execution_history.append({
                'plan_id': plan_id,
                'start_time': start_time,
                'end_time': datetime.now(),
                'execution_time': execution_time,
                'success': result.get('success', False),
                'strategy': plan.strategy.value,
                'workflow_count': len(plan.workflows)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestration plan execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'plan_id': plan_id
            }
    
    def _execute_sequential(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute workflows sequentially."""
        
        results = []
        
        for workflow_schedule in plan.workflows:
            # Wait for dependencies
            self._wait_for_dependencies(workflow_schedule.dependencies)
            
            # Execute workflow
            result = self._execute_single_workflow(workflow_schedule)
            results.append(result)
            
            # Stop on failure if no retry policy
            if not result.get('success', False) and not workflow_schedule.retry_policy:
                break
        
        return {
            'success': all(r.get('success', False) for r in results),
            'results': results,
            'strategy': 'sequential'
        }
    
    def _execute_parallel(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute workflows in parallel."""
        
        futures = []
        
        for workflow_schedule in plan.workflows:
            future = self.executor.submit(self._execute_single_workflow, workflow_schedule)
            futures.append((workflow_schedule.workflow_id, future))
        
        results = []
        for workflow_id, future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                result['workflow_id'] = workflow_id
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel workflow {workflow_id} failed: {e}")
                results.append({
                    'workflow_id': workflow_id,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'success': all(r.get('success', False) for r in results),
            'results': results,
            'strategy': 'parallel'
        }
    
    def _execute_conditional(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute workflows based on conditions."""
        
        results = []
        
        for workflow_schedule in plan.workflows:
            # Check execution conditions
            should_execute = self._evaluate_execution_conditions(workflow_schedule)
            
            if should_execute:
                result = self._execute_single_workflow(workflow_schedule)
                results.append(result)
            else:
                results.append({
                    'workflow_id': workflow_schedule.workflow_id,
                    'success': True,
                    'skipped': True,
                    'reason': 'Conditions not met'
                })
        
        return {
            'success': all(r.get('success', False) for r in results),
            'results': results,
            'strategy': 'conditional'
        }
    
    def _execute_priority_based(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute workflows based on priority."""
        
        # Sort by priority
        sorted_workflows = sorted(plan.workflows, key=lambda w: w.priority.value)
        
        results = []
        for workflow_schedule in sorted_workflows:
            result = self._execute_single_workflow(workflow_schedule)
            results.append(result)
        
        return {
            'success': all(r.get('success', False) for r in results),
            'results': results,
            'strategy': 'priority_based'
        }
    
    def _execute_resource_aware(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute workflows based on resource availability."""
        
        results = []
        pending_workflows = plan.workflows.copy()
        
        while pending_workflows:
            executed_any = False
            
            for i, workflow_schedule in enumerate(pending_workflows):
                if self.resource_manager.check_resource_availability(workflow_schedule.resource_requirements):
                    result = self._execute_single_workflow(workflow_schedule)
                    results.append(result)
                    pending_workflows.pop(i)
                    executed_any = True
                    break
            
            if not executed_any:
                # Wait a bit for resources to become available
                time.sleep(1)
        
        return {
            'success': all(r.get('success', False) for r in results),
            'results': results,
            'strategy': 'resource_aware'
        }
    
    def _execute_adaptive(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """Execute workflows using adaptive strategy."""
        
        # Analyze plan characteristics and choose best strategy
        if len(plan.workflows) <= 2:
            return self._execute_sequential(plan)
        
        # Check resource requirements
        total_resources = sum(
            sum(req.amount for req in wf.resource_requirements)
            for wf in plan.workflows
        )
        
        if total_resources > 200:  # High resource usage
            return self._execute_resource_aware(plan)
        
        # Check dependencies
        has_dependencies = any(wf.dependencies for wf in plan.workflows)
        if has_dependencies:
            return self._execute_conditional(plan)
        
        # Default to parallel for independent workflows
        return self._execute_parallel(plan)
    
    def _execute_single_workflow(self, schedule: WorkflowSchedule) -> Dict[str, Any]:
        """Execute a single workflow."""
        
        workflow_id = schedule.workflow_id
        start_time = datetime.now()
        
        try:
            # Allocate resources
            if not self.resource_manager.allocate_resources(workflow_id, schedule.resource_requirements):
                return {
                    'workflow_id': workflow_id,
                    'success': False,
                    'error': 'Resource allocation failed'
                }
            
            # Mark as running
            self.scheduler.mark_workflow_running(workflow_id, start_time)
            
            # Create initial state
            initial_state = AppState(
                session_id=str(uuid.uuid4()),
                user_input=schedule.metadata.get('user_input', ''),
                conversation=[],
                tasks=[],
                logs=[]
            )
            
            # Execute workflow using LangGraph integration
            result = self.langgraph_integration.execute_workflow_with_analytics(
                workflow_id,
                initial_state,
                schedule.metadata.get('config', {})
            )
            
            end_time = datetime.now()
            
            # Mark as completed
            self.scheduler.mark_workflow_completed(
                workflow_id, end_time, result.get('success', False), result
            )
            
            # Release resources
            self.resource_manager.release_resources(workflow_id)
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            
            # Mark as failed
            self.scheduler.mark_workflow_completed(
                workflow_id, end_time, False, {'error': str(e)}
            )
            
            # Release resources
            self.resource_manager.release_resources(workflow_id)
            
            logger.error(f"Workflow {workflow_id} execution failed: {e}")
            return {
                'workflow_id': workflow_id,
                'success': False,
                'error': str(e)
            }
    
    def _wait_for_dependencies(self, dependencies: List[str]):
        """Wait for workflow dependencies to complete."""
        
        while dependencies:
            completed = []
            for dep_id in dependencies:
                if dep_id in self.scheduler.completed_workflows:
                    completed.append(dep_id)
            
            for dep_id in completed:
                dependencies.remove(dep_id)
            
            if dependencies:
                time.sleep(1)  # Wait before checking again
    
    def _evaluate_execution_conditions(self, schedule: WorkflowSchedule) -> bool:
        """Evaluate whether a workflow should be executed based on conditions."""
        
        conditions = schedule.metadata.get('execution_conditions', {})
        
        # Time-based conditions
        if 'time_window' in conditions:
            window = conditions['time_window']
            current_time = datetime.now().time()
            start_time = datetime.strptime(window['start'], '%H:%M').time()
            end_time = datetime.strptime(window['end'], '%H:%M').time()
            
            if not (start_time <= current_time <= end_time):
                return False
        
        # Resource-based conditions
        if 'min_resources' in conditions:
            min_resources = conditions['min_resources']
            utilization = self.resource_manager.get_resource_utilization()
            
            for resource_type, min_available in min_resources.items():
                if utilization.get(ResourceType(resource_type), 100) > (100 - min_available):
                    return False
        
        # State-based conditions
        if 'state_conditions' in conditions:
            # This would check global state conditions
            # Implementation depends on specific requirements
            pass
        
        return True
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status."""
        
        return {
            'active_plans': len(self.active_plans),
            'scheduler_status': self.scheduler.get_scheduler_status(),
            'resource_utilization': self.resource_manager.get_resource_utilization(),
            'execution_history_count': len(self.plan_execution_history),
            'running': self.running
        }
    
    def start_orchestration_loop(self):
        """Start the main orchestration loop."""
        
        if self.running:
            logger.warning("Orchestration loop already running")
            return
        
        self.running = True
        self.orchestration_thread = threading.Thread(target=self._orchestration_loop)
        self.orchestration_thread.start()
        
        logger.info("Orchestration loop started")
    
    def stop_orchestration_loop(self):
        """Stop the orchestration loop."""
        
        self.running = False
        if self.orchestration_thread:
            self.orchestration_thread.join()
        
        logger.info("Orchestration loop stopped")
    
    def _orchestration_loop(self):
        """Main orchestration loop."""
        
        while self.running:
            try:
                # Get next workflow to execute
                next_workflow = self.scheduler.get_next_workflow()
                
                if next_workflow:
                    # Execute workflow in background
                    self.executor.submit(self._execute_single_workflow, next_workflow)
                else:
                    # No workflows ready, wait a bit
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                time.sleep(5)  # Wait longer on error

class EnhancedWorkflowOrchestrator:
    """Main orchestrator class that provides a high-level interface."""
    
    def __init__(self,
                 integration_service: Optional[EnhancedIntegrationService] = None):
        
        self.integration_service = integration_service or EnhancedIntegrationService()
        self.orchestration_engine = OrchestrationEngine(
            self.integration_service.langgraph_integration,
            self.integration_service.nlp_processor,
            self.integration_service.state_manager
        )
        
        # High-level orchestration features
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        self.orchestration_history: List[Dict[str, Any]] = []
        
        logger.info("Enhanced Workflow Orchestrator initialized")
    
    def register_workflow_template(self, template: WorkflowTemplate):
        """Register a workflow template for reuse."""
        self.workflow_templates[template.name] = template
        logger.info(f"Workflow template registered: {template.name}")
    
    def create_research_orchestration(self,
                                    research_queries: List[str],
                                    priority: WorkflowPriority = WorkflowPriority.NORMAL) -> str:
        """Create an orchestration plan for multiple research tasks."""
        
        workflow_configs = []
        
        for i, query in enumerate(research_queries):
            config = {
                'workflow_id': f"research_{i}_{uuid.uuid4().hex[:8]}",
                'scheduled_time': datetime.now() + timedelta(seconds=i * 5),  # Stagger execution
                'priority': priority.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.API_CALLS.value, 'amount': 10},
                    {'resource_type': ResourceType.MEMORY.value, 'amount': 20}
                ],
                'metadata': {
                    'user_input': query,
                    'workflow_type': 'research'
                }
            }
            workflow_configs.append(config)
        
        plan = self.orchestration_engine.create_orchestration_plan(
            name=f"Research Orchestration - {len(research_queries)} queries",
            description="Parallel research execution for multiple queries",
            strategy=OrchestrationStrategy.PARALLEL,
            workflow_configs=workflow_configs
        )
        
        return plan.plan_id
    
    def create_analysis_pipeline(self,
                                data_sources: List[str],
                                analysis_types: List[str]) -> str:
        """Create an orchestration plan for data analysis pipeline."""
        
        workflow_configs = []
        
        # Data collection workflows
        for i, source in enumerate(data_sources):
            config = {
                'workflow_id': f"collect_{i}_{uuid.uuid4().hex[:8]}",
                'priority': WorkflowPriority.HIGH.value,
                'resource_requirements': [
                    {'resource_type': ResourceType.NETWORK.value, 'amount': 15},
                    {'resource_type': ResourceType.STORAGE.value, 'amount': 30}
                ],
                'metadata': {
                    'data_source': source,
                    'workflow_type': 'data_collection'
                }
            }
            workflow_configs.append(config)
        
        # Analysis workflows (depend on data collection)
        collection_ids = [config['workflow_id'] for config in workflow_configs]
        
        for analysis_type in analysis_types:
            config = {
                'workflow_id': f"analyze_{analysis_type}_{uuid.uuid4().hex[:8]}",
                'priority': WorkflowPriority.NORMAL.value,
                'dependencies': collection_ids,
                'resource_requirements': [
                    {'resource_type': ResourceType.CPU.value, 'amount': 40},
                    {'resource_type': ResourceType.MEMORY.value, 'amount': 50}
                ],
                'metadata': {
                    'analysis_type': analysis_type,
                    'workflow_type': 'analysis'
                }
            }
            workflow_configs.append(config)
        
        plan = self.orchestration_engine.create_orchestration_plan(
            name=f"Analysis Pipeline - {len(data_sources)} sources, {len(analysis_types)} analyses",
            description="Sequential data collection followed by parallel analysis",
            strategy=OrchestrationStrategy.CONDITIONAL,
            workflow_configs=workflow_configs
        )
        
        return plan.plan_id
    
    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute an orchestration plan."""
        return self.orchestration_engine.execute_orchestration_plan(plan_id)
    
    def get_orchestration_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about orchestration performance."""
        
        status = self.orchestration_engine.get_orchestration_status()
        
        # Add historical analysis
        history = self.orchestration_engine.plan_execution_history
        if history:
            avg_execution_time = np.mean([h['execution_time'] for h in history])
            success_rate = np.mean([h['success'] for h in history])
            
            status.update({
                'historical_performance': {
                    'total_executions': len(history),
                    'average_execution_time': avg_execution_time,
                    'success_rate': success_rate,
                    'strategy_distribution': self._analyze_strategy_distribution(history)
                }
            })
        
        return status
    
    def _analyze_strategy_distribution(self, history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze the distribution of orchestration strategies used."""
        
        distribution = defaultdict(int)
        for entry in history:
            distribution[entry.get('strategy', 'unknown')] += 1
        
        return dict(distribution)
    
    def start(self):
        """Start the orchestrator."""
        self.orchestration_engine.start_orchestration_loop()
    
    def stop(self):
        """Stop the orchestrator."""
        self.orchestration_engine.stop_orchestration_loop()

# Global orchestrator instance
default_orchestrator = EnhancedWorkflowOrchestrator()

# Export key components
__all__ = [
    'EnhancedWorkflowOrchestrator',
    'OrchestrationEngine',
    'ResourceManager',
    'WorkflowScheduler',
    'OrchestrationStrategy',
    'WorkflowPriority',
    'ResourceType',
    'OrchestrationPlan',
    'WorkflowSchedule',
    'ResourceRequirement',
    'default_orchestrator'
]