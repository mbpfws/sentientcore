"""Advanced Workflow Engine Package

This package provides advanced workflow orchestration capabilities including:
- Dynamic workflow generation
- Conditional branching
- Parallel and sequential execution
- Human-in-the-loop workflows
- State persistence and recovery
"""

from .advanced_engine import (
    WorkflowStatus,
    TaskStatus,
    ExecutionMode,
    WorkflowTask,
    WorkflowCondition,
    WorkflowDefinition,
    WorkflowExecution,
    AdvancedWorkflowEngine
)

from .workflow_factory import (
    create_advanced_workflow_engine,
    initialize_workflow_engine
)

from .dynamic_generator import (
    ProjectComplexity,
    WorkflowPattern,
    ProjectRequirements,
    WorkflowTemplate,
    DynamicWorkflowGenerator
)

from .template_system import (
    TemplateFormat,
    ValidationLevel,
    TemplateMetadata,
    TemplateValidationResult,
    WorkflowTemplateManager
)

from .conditional_branching import (
    ConditionType,
    ComparisonOperator,
    LogicalOperator,
    ConditionExpression,
    ComplexCondition,
    ConditionalBranch,
    BranchingContext,
    ConditionEvaluator,
    SimpleConditionEvaluator,
    TemporalConditionEvaluator,
    ConditionalBranchingEngine
)

__all__ = [
    # Core engine
    'WorkflowStatus',
    'TaskStatus', 
    'ExecutionMode',
    'WorkflowTask',
    'WorkflowCondition',
    'WorkflowDefinition',
    'WorkflowExecution',
    'AdvancedWorkflowEngine',
    'create_advanced_workflow_engine',
    'initialize_workflow_engine',
    
    # Dynamic generation
    'ProjectComplexity',
    'WorkflowPattern',
    'ProjectRequirements',
    'WorkflowTemplate',
    'DynamicWorkflowGenerator',
    
    # Template system
    'TemplateFormat',
    'ValidationLevel',
    'TemplateMetadata',
    'TemplateValidationResult',
    'WorkflowTemplateManager',
    
    # Conditional branching
    'ConditionType',
    'ComparisonOperator',
    'LogicalOperator',
    'ConditionExpression',
    'ComplexCondition',
    'ConditionalBranch',
    'BranchingContext',
    'ConditionEvaluator',
    'SimpleConditionEvaluator',
    'TemporalConditionEvaluator',
    'ConditionalBranchingEngine'
]