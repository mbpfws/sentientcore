"""Advanced Workflow Management Module

Provides sophisticated workflow orchestration capabilities including:
- Dynamic workflow generation and adaptation
- Parallel and sequential task execution
- Conditional branching and decision points
- Human-in-the-loop workflow support
- Real-time monitoring and intervention
- Workflow state persistence and recovery
"""

from .advanced_engine import (
    AdvancedWorkflowEngine,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowTask,
    WorkflowCondition,
    WorkflowStatus,
    TaskStatus,
    ExecutionMode
)

__all__ = [
    'AdvancedWorkflowEngine',
    'WorkflowDefinition',
    'WorkflowExecution',
    'WorkflowTask',
    'WorkflowCondition',
    'WorkflowStatus',
    'TaskStatus',
    'ExecutionMode'
]