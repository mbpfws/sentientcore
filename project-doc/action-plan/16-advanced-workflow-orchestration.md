# 16 - Advanced Workflow Orchestration

## Overview

The Advanced Workflow Orchestration module provides sophisticated workflow management capabilities for complex multi-agent development scenarios. This includes dynamic workflow adaptation, parallel task execution, conditional branching, workflow templates, and real-time workflow monitoring with human-in-the-loop intervention points.

## Current State Analysis

### Workflow Requirements
- Dynamic workflow generation based on project complexity
- Parallel and sequential task execution management
- Conditional workflow branching and decision points
- Workflow template library for common development patterns
- Real-time workflow monitoring and intervention
- Workflow state persistence and recovery
- Performance optimization for large-scale workflows

### Integration Points
- Ultra Orchestrator Agent coordination
- Agent task distribution and load balancing
- Frontend workflow visualization
- State management service integration
- Memory service for workflow history

## Implementation Tasks

### Task 16.1: Advanced Workflow Engine

**File**: `core/workflows/advanced_engine.py`

**Advanced Workflow Engine**:
```python
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
import asyncio
from enum import Enum
from dataclasses import dataclass, field
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_FOR_INPUT = "waiting_for_input"

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    HUMAN_IN_LOOP = "human_in_loop"

@dataclass
class WorkflowTask:
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
    condition_id: str
    condition_type: str  # "success", "failure", "custom"
    condition_expression: str
    true_path: List[str]  # Task IDs to execute if condition is true
    false_path: List[str]  # Task IDs to execute if condition is false
    evaluation_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
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
        """Initialize the workflow engine"""
        await self._load_workflow_templates()
        await self._setup_condition_evaluators()
        await self._setup_human_intervention_handlers()
        
    async def create_workflow_from_template(self, template_name: str, parameters: Dict[str, Any]) -> WorkflowDefinition:
        """Create workflow from template with parameters"""
        template = self.workflow_templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Clone template and apply parameters
        workflow = self._clone_workflow_definition(template)
        workflow.id = str(uuid.uuid4())
        workflow = await self._apply_parameters_to_workflow(workflow, parameters)
        
        return workflow
    
    async def execute_workflow(self, workflow: WorkflowDefinition, context: Dict[str, Any] = None) -> str:
        """Execute workflow and return execution ID"""
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
        """Asynchronously execute workflow"""
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
        """Execute tasks based on dependency graph"""
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
        """Execute multiple tasks in parallel"""
        max_parallel = execution.workflow_definition.max_parallel_tasks
        
        # Split tasks into batches
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i:i + max_parallel]
            
            # Execute batch in parallel
            task_coroutines = [self._execute_single_task(execution, task) for task in batch]
            await asyncio.gather(*task_coroutines, return_exceptions=True)
    
    async def _execute_single_task(self, execution: WorkflowExecution, task: WorkflowTask):
        """Execute a single task"""
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
        """Execute task requiring human intervention"""
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
        """Wait for human input on a task"""
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
        """Provide human input for a waiting task"""
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
        """Evaluate workflow conditions and handle branching"""
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
        """Evaluate a single condition"""
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
        """Get tasks with no dependencies"""
        return [task for task in workflow.tasks if not task.dependencies]
    
    def _get_ready_tasks(self, execution: WorkflowExecution) -> List[WorkflowTask]:
        """Get tasks whose dependencies are satisfied"""
        ready_tasks = []
        
        for task in execution.workflow_definition.tasks:
            if (task.status == TaskStatus.PENDING and 
                all(dep_id in execution.completed_tasks for dep_id in task.dependencies)):
                ready_tasks.append(task)
        
        return ready_tasks
    
    async def _get_agent_for_task(self, task: WorkflowTask):
        """Get appropriate agent for task execution"""
        agents = await self.agent_registry.get_agents_by_type(task.agent_type)
        
        if not agents:
            return None
        
        # Simple load balancing - get least busy agent
        return min(agents, key=lambda a: len(getattr(a, 'current_tasks', [])))
    
    async def _update_execution_state(self, execution: WorkflowExecution):
        """Update execution state in state service"""
        await self.state_service.update_workflow_state(
            execution.execution_id,
            {
                "status": execution.status.value,
                "progress": execution.progress,
                "current_tasks": execution.current_tasks,
                "completed_tasks": execution.completed_tasks,
                "failed_tasks": execution.failed_tasks,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
    
    async def _finalize_execution(self, execution: WorkflowExecution):
        """Finalize workflow execution"""
        # Move to history
        self.execution_history.append(execution)
        
        # Remove from active executions
        if execution.execution_id in self.active_executions:
            del self.active_executions[execution.execution_id]
        
        # Store in memory service
        await self.memory_service.store_knowledge(
            "workflow_execution",
            {
                "execution_id": execution.execution_id,
                "workflow_name": execution.workflow_definition.name,
                "status": execution.status.value,
                "duration": (execution.end_time - execution.start_time).total_seconds() if execution.end_time else None,
                "task_count": len(execution.workflow_definition.tasks),
                "success_rate": len(execution.completed_tasks) / len(execution.workflow_definition.tasks),
                "execution_data": execution
            },
            {"type": "workflow_history"}
        )
    
    # Workflow template management
    async def _load_workflow_templates(self):
        """Load predefined workflow templates"""
        # Development workflow template
        dev_template = WorkflowDefinition(
            id="development_workflow",
            name="Standard Development Workflow",
            description="Standard workflow for feature development",
            version="1.0",
            tasks=[
                WorkflowTask(
                    id="requirements_analysis",
                    name="Requirements Analysis",
                    description="Analyze and refine requirements",
                    agent_type="ResearchAgent",
                    task_data={"type": "requirements_analysis"},
                    human_validation_required=True
                ),
                WorkflowTask(
                    id="architecture_planning",
                    name="Architecture Planning",
                    description="Plan system architecture",
                    agent_type="ArchitectPlannerAgent",
                    task_data={"type": "architecture_planning"},
                    dependencies=["requirements_analysis"],
                    human_validation_required=True
                ),
                WorkflowTask(
                    id="backend_development",
                    name="Backend Development",
                    description="Develop backend components",
                    agent_type="BackendDeveloperAgent",
                    task_data={"type": "backend_development"},
                    dependencies=["architecture_planning"],
                    execution_mode=ExecutionMode.PARALLEL
                ),
                WorkflowTask(
                    id="frontend_development",
                    name="Frontend Development",
                    description="Develop frontend components",
                    agent_type="FrontendDeveloperAgent",
                    task_data={"type": "frontend_development"},
                    dependencies=["architecture_planning"],
                    execution_mode=ExecutionMode.PARALLEL
                ),
                WorkflowTask(
                    id="integration",
                    name="Integration",
                    description="Integrate frontend and backend",
                    agent_type="CodingAgent",
                    task_data={"type": "integration"},
                    dependencies=["backend_development", "frontend_development"],
                    human_validation_required=True
                ),
                WorkflowTask(
                    id="testing",
                    name="Testing",
                    description="Execute comprehensive tests",
                    agent_type="TestingAgent",
                    task_data={"type": "testing"},
                    dependencies=["integration"]
                )
            ],
            human_intervention_points=["requirements_analysis", "architecture_planning", "integration"]
        )
        
        self.workflow_templates["development_workflow"] = dev_template
    
    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get current execution status"""
        execution = self.active_executions.get(execution_id)
        if not execution:
            # Check history
            for hist_execution in self.execution_history:
                if hist_execution.execution_id == execution_id:
                    execution = hist_execution
                    break
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        return {
            "execution_id": execution.execution_id,
            "workflow_name": execution.workflow_definition.name,
            "status": execution.status.value,
            "progress": execution.progress,
            "current_tasks": execution.current_tasks,
            "completed_tasks": execution.completed_tasks,
            "failed_tasks": execution.failed_tasks,
            "start_time": execution.start_time.isoformat() if execution.start_time else None,
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "intervention_history": execution.intervention_history
        }
    
    async def pause_execution(self, execution_id: str):
        """Pause workflow execution"""
        execution = self.active_executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.PAUSED
            await self._update_execution_state(execution)
    
    async def resume_execution(self, execution_id: str):
        """Resume paused workflow execution"""
        execution = self.active_executions.get(execution_id)
        if execution and execution.status == WorkflowStatus.PAUSED:
            execution.status = WorkflowStatus.RUNNING
            await self._update_execution_state(execution)
            # Continue execution
            asyncio.create_task(self._execute_workflow_async(execution))
    
    async def cancel_execution(self, execution_id: str):
        """Cancel workflow execution"""
        execution = self.active_executions.get(execution_id)
        if execution:
            execution.status = WorkflowStatus.CANCELLED
            execution.end_time = datetime.utcnow()
            await self._finalize_execution(execution)
```

### Task 16.2: Workflow Visualization and Monitoring

**File**: `frontend/components/workflow-monitor.tsx`

**Workflow Monitoring Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Play, Pause, Square, RefreshCw } from 'lucide-react';

interface WorkflowTask {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'blocked';
  agent_type: string;
  start_time?: string;
  end_time?: string;
  dependencies: string[];
  human_validation_required: boolean;
}

interface WorkflowExecution {
  execution_id: string;
  workflow_name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled' | 'waiting_for_input';
  progress: number;
  current_tasks: string[];
  completed_tasks: string[];
  failed_tasks: string[];
  start_time?: string;
  end_time?: string;
  intervention_history: any[];
}

interface WorkflowMonitorProps {
  executionId: string;
  onHumanInput?: (taskId: string, input: any) => void;
}

export const WorkflowMonitor: React.FC<WorkflowMonitorProps> = ({ 
  executionId, 
  onHumanInput 
}) => {
  const [execution, setExecution] = useState<WorkflowExecution | null>(null);
  const [tasks, setTasks] = useState<WorkflowTask[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchExecutionStatus();
    const interval = setInterval(fetchExecutionStatus, 2000); // Update every 2 seconds
    return () => clearInterval(interval);
  }, [executionId]);

  const fetchExecutionStatus = async () => {
    try {
      const response = await fetch(`/api/workflows/execution/${executionId}`);
      if (!response.ok) throw new Error('Failed to fetch execution status');
      
      const data = await response.json();
      setExecution(data.execution);
      setTasks(data.tasks || []);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleWorkflowControl = async (action: 'pause' | 'resume' | 'cancel') => {
    try {
      const response = await fetch(`/api/workflows/execution/${executionId}/${action}`, {
        method: 'POST'
      });
      if (!response.ok) throw new Error(`Failed to ${action} workflow`);
      await fetchExecutionStatus();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  };

  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'bg-gray-500',
      running: 'bg-blue-500',
      completed: 'bg-green-500',
      failed: 'bg-red-500',
      blocked: 'bg-yellow-500',
      paused: 'bg-orange-500',
      cancelled: 'bg-gray-600',
      waiting_for_input: 'bg-purple-500'
    };
    return colors[status] || 'bg-gray-500';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return <RefreshCw className="w-4 h-4 animate-spin" />;
      case 'completed':
        return <div className="w-4 h-4 rounded-full bg-green-500" />;
      case 'failed':
        return <div className="w-4 h-4 rounded-full bg-red-500" />;
      default:
        return <div className="w-4 h-4 rounded-full bg-gray-500" />;
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <RefreshCw className="w-6 h-6 animate-spin" />
            <span className="ml-2">Loading workflow status...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-red-600">Error: {error}</div>
        </CardContent>
      </Card>
    );
  }

  if (!execution) {
    return (
      <Card>
        <CardContent className="p-6">
          <div>Workflow execution not found</div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Workflow Overview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>{execution.workflow_name}</CardTitle>
            <div className="flex space-x-2">
              {execution.status === 'running' && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleWorkflowControl('pause')}
                >
                  <Pause className="w-4 h-4" />
                </Button>
              )}
              {execution.status === 'paused' && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleWorkflowControl('resume')}
                >
                  <Play className="w-4 h-4" />
                </Button>
              )}
              {['running', 'paused'].includes(execution.status) && (
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => handleWorkflowControl('cancel')}
                >
                  <Square className="w-4 h-4" />
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <Badge className={getStatusColor(execution.status)}>
                {execution.status.replace('_', ' ').toUpperCase()}
              </Badge>
              <span className="text-sm text-gray-600">
                ID: {execution.execution_id}
              </span>
            </div>
            
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Progress</span>
                <span>{Math.round(execution.progress * 100)}%</span>
              </div>
              <Progress value={execution.progress * 100} className="w-full" />
            </div>
            
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Completed:</span>
                <span className="ml-2 font-semibold">{execution.completed_tasks.length}</span>
              </div>
              <div>
                <span className="text-gray-600">Running:</span>
                <span className="ml-2 font-semibold">{execution.current_tasks.length}</span>
              </div>
              <div>
                <span className="text-gray-600">Failed:</span>
                <span className="ml-2 font-semibold text-red-600">{execution.failed_tasks.length}</span>
              </div>
            </div>
            
            {execution.start_time && (
              <div className="text-sm text-gray-600">
                Started: {new Date(execution.start_time).toLocaleString()}
                {execution.end_time && (
                  <span className="ml-4">
                    Ended: {new Date(execution.end_time).toLocaleString()}
                  </span>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Task List */}
      <Card>
        <CardHeader>
          <CardTitle>Tasks</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {tasks.map((task) => (
              <div
                key={task.id}
                className={`p-4 border rounded-lg ${
                  execution.current_tasks.includes(task.id) ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(task.status)}
                    <div>
                      <h4 className="font-semibold">{task.name}</h4>
                      <p className="text-sm text-gray-600">Agent: {task.agent_type}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <Badge variant="outline" className={getStatusColor(task.status)}>
                      {task.status.toUpperCase()}
                    </Badge>
                    {task.human_validation_required && (
                      <Badge variant="secondary">Human Validation</Badge>
                    )}
                  </div>
                </div>
                
                {task.dependencies.length > 0 && (
                  <div className="mt-2 text-sm text-gray-600">
                    Dependencies: {task.dependencies.join(', ')}
                  </div>
                )}
                
                {task.start_time && (
                  <div className="mt-2 text-sm text-gray-600">
                    Started: {new Date(task.start_time).toLocaleString()}
                    {task.end_time && (
                      <span className="ml-4">
                        Completed: {new Date(task.end_time).toLocaleString()}
                      </span>
                    )}
                  </div>
                )}
                
                {task.human_validation_required && 
                 task.status === 'completed' && 
                 execution.status === 'waiting_for_input' && (
                  <div className="mt-3 p-3 bg-yellow-50 border border-yellow-200 rounded">
                    <p className="text-sm text-yellow-800 mb-2">
                      This task requires human validation before proceeding.
                    </p>
                    <div className="flex space-x-2">
                      <Button
                        size="sm"
                        onClick={() => onHumanInput?.(task.id, { approved: true })}
                      >
                        Approve
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                          const reason = prompt('Please provide a reason for rejection:');
                          if (reason) {
                            onHumanInput?.(task.id, { approved: false, reason });
                          }
                        }}
                      >
                        Reject
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Human Intervention History */}
      {execution.intervention_history.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Human Interventions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {execution.intervention_history.map((intervention, index) => (
                <div key={index} className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <h5 className="font-semibold">{intervention.task_name}</h5>
                      <p className="text-sm text-gray-600">{intervention.description}</p>
                    </div>
                    <Badge 
                      variant={intervention.status === 'completed' ? 'default' : 'secondary'}
                    >
                      {intervention.status.toUpperCase()}
                    </Badge>
                  </div>
                  <div className="mt-2 text-sm text-gray-600">
                    Requested: {new Date(intervention.timestamp).toLocaleString()}
                    {intervention.response_timestamp && (
                      <span className="ml-4">
                        Responded: {new Date(intervention.response_timestamp).toLocaleString()}
                      </span>
                    )}
                  </div>
                  {intervention.response && (
                    <div className="mt-2 p-2 bg-gray-50 rounded text-sm">
                      Response: {JSON.stringify(intervention.response, null, 2)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
```

### Task 16.3: Backend API Integration

**File**: `app/api/workflows.py`

**Workflow API Endpoints**:
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from core.workflows.advanced_engine import AdvancedWorkflowEngine, WorkflowDefinition
from core.agents.registry import AgentRegistry
from core.services.state_service import StateService
from core.services.memory_service import MemoryService

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

# Initialize workflow engine (would be dependency injected)
workflow_engine = None

class WorkflowExecutionRequest(BaseModel):
    template_name: str
    parameters: Dict[str, Any] = {}
    context: Dict[str, Any] = {}

class HumanInputRequest(BaseModel):
    task_id: str
    input_data: Dict[str, Any]

@router.post("/execute")
async def execute_workflow(request: WorkflowExecutionRequest):
    """Execute workflow from template"""
    try:
        # Create workflow from template
        workflow = await workflow_engine.create_workflow_from_template(
            request.template_name,
            request.parameters
        )
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(
            workflow,
            request.context
        )
        
        return {
            "success": True,
            "execution_id": execution_id,
            "workflow_name": workflow.name
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/execution/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get workflow execution status"""
    try:
        status = await workflow_engine.get_execution_status(execution_id)
        return {
            "success": True,
            "execution": status
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/execution/{execution_id}/pause")
async def pause_execution(execution_id: str):
    """Pause workflow execution"""
    try:
        await workflow_engine.pause_execution(execution_id)
        return {"success": True, "message": "Workflow paused"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execution/{execution_id}/resume")
async def resume_execution(execution_id: str):
    """Resume workflow execution"""
    try:
        await workflow_engine.resume_execution(execution_id)
        return {"success": True, "message": "Workflow resumed"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execution/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel workflow execution"""
    try:
        await workflow_engine.cancel_execution(execution_id)
        return {"success": True, "message": "Workflow cancelled"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execution/{execution_id}/human-input")
async def provide_human_input(execution_id: str, request: HumanInputRequest):
    """Provide human input for workflow task"""
    try:
        await workflow_engine.provide_human_input(
            execution_id,
            request.task_id,
            request.input_data
        )
        return {"success": True, "message": "Human input provided"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/templates")
async def list_workflow_templates():
    """List available workflow templates"""
    try:
        templates = list(workflow_engine.workflow_templates.keys())
        return {
            "success": True,
            "templates": templates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/executions")
async def list_active_executions():
    """List active workflow executions"""
    try:
        executions = []
        for execution_id, execution in workflow_engine.active_executions.items():
            executions.append({
                "execution_id": execution_id,
                "workflow_name": execution.workflow_definition.name,
                "status": execution.status.value,
                "progress": execution.progress,
                "start_time": execution.start_time.isoformat() if execution.start_time else None
            })
        
        return {
            "success": True,
            "executions": executions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Strategy

### Unit Tests
- Workflow engine functionality
- Task execution and dependency management
- Condition evaluation
- Human intervention handling

### Integration Tests
- End-to-end workflow execution
- Agent coordination
- Frontend-backend communication
- State persistence

### Human Testing Scenarios
1. **Basic Workflow Execution**: Execute simple development workflow
2. **Human Intervention**: Test human validation points
3. **Parallel Task Execution**: Verify parallel task handling
4. **Workflow Control**: Test pause/resume/cancel functionality
5. **Error Recovery**: Test workflow error handling and recovery

## Validation Criteria

### Backend Validation
- [ ] Workflow engine executes tasks correctly
- [ ] Dependencies are respected
- [ ] Human intervention points work
- [ ] Workflow state is persisted
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Workflow monitor displays real-time status
- [ ] Human intervention UI works
- [ ] Workflow controls function properly
- [ ] Progress visualization is accurate

### Integration Validation
- [ ] Workflows coordinate multiple agents
- [ ] State synchronization works
- [ ] Error handling is robust
- [ ] Performance is acceptable

## Next Steps

After successful validation of advanced workflow orchestration, proceed to **17-performance-optimization-framework.md** for implementing comprehensive performance monitoring and optimization capabilities.

---

**Dependencies**: This phase requires the agent framework, state management, and memory services to be functional and builds upon the Ultra Orchestrator implementation.