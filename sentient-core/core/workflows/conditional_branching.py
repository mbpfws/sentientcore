"""Conditional Branching System

This module provides advanced conditional branching capabilities for workflows,
including runtime condition evaluation, dynamic path selection, and context-aware
decision making.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import operator
from abc import ABC, abstractmethod

from .advanced_engine import WorkflowCondition, WorkflowExecution


class ConditionType(Enum):
    """Types of conditions supported."""
    SIMPLE = "simple"  # Simple comparisons
    COMPLEX = "complex"  # Complex expressions
    CUSTOM = "custom"  # Custom evaluator functions
    TEMPORAL = "temporal"  # Time-based conditions
    CONTEXTUAL = "contextual"  # Context-dependent conditions
    PROBABILISTIC = "probabilistic"  # Probability-based conditions


class ComparisonOperator(Enum):
    """Comparison operators for conditions."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES = "matches"  # Regex matching
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"


@dataclass
class ConditionExpression:
    """Represents a single condition expression."""
    left_operand: str
    operator: ComparisonOperator
    right_operand: Any
    data_type: str = "auto"  # auto, string, number, boolean, list, dict


@dataclass
class ComplexCondition:
    """Represents a complex condition with multiple expressions."""
    expressions: List[Union[ConditionExpression, 'ComplexCondition']]
    logical_operator: LogicalOperator
    parentheses: bool = False


@dataclass
class ConditionalBranch:
    """Represents a conditional branch in the workflow."""
    branch_id: str
    name: str
    condition: Union[ConditionExpression, ComplexCondition, str]
    true_path: List[str]  # Task IDs to execute if condition is true
    false_path: List[str]  # Task IDs to execute if condition is false
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BranchingContext:
    """Context for evaluating conditional branches."""
    workflow_id: str
    execution_id: str
    current_task_id: Optional[str]
    task_results: Dict[str, Any] = field(default_factory=dict)
    workflow_variables: Dict[str, Any] = field(default_factory=dict)
    system_context: Dict[str, Any] = field(default_factory=dict)
    user_context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ConditionEvaluator(ABC):
    """Abstract base class for condition evaluators."""
    
    @abstractmethod
    async def evaluate(
        self,
        condition: Union[ConditionExpression, ComplexCondition, str],
        context: BranchingContext
    ) -> bool:
        """Evaluate a condition against the given context."""
        pass


class SimpleConditionEvaluator(ConditionEvaluator):
    """Evaluates simple condition expressions."""
    
    def __init__(self):
        self.operators = {
            ComparisonOperator.EQUALS: operator.eq,
            ComparisonOperator.NOT_EQUALS: operator.ne,
            ComparisonOperator.GREATER_THAN: operator.gt,
            ComparisonOperator.GREATER_EQUAL: operator.ge,
            ComparisonOperator.LESS_THAN: operator.lt,
            ComparisonOperator.LESS_EQUAL: operator.le,
        }
        
    async def evaluate(
        self,
        condition: Union[ConditionExpression, ComplexCondition, str],
        context: BranchingContext
    ) -> bool:
        """Evaluate a simple condition expression."""
        
        if isinstance(condition, str):
            # Parse string condition
            condition = await self._parse_string_condition(condition)
            
        if isinstance(condition, ComplexCondition):
            return await self._evaluate_complex_condition(condition, context)
            
        if not isinstance(condition, ConditionExpression):
            raise ValueError(f"Unsupported condition type: {type(condition)}")
            
        # Get left operand value
        left_value = await self._resolve_operand(condition.left_operand, context)
        
        # Get right operand value
        right_value = condition.right_operand
        if isinstance(right_value, str) and right_value.startswith("$"):
            right_value = await self._resolve_operand(right_value, context)
            
        # Type conversion
        left_value, right_value = await self._convert_types(
            left_value, right_value, condition.data_type
        )
        
        # Evaluate condition
        return await self._apply_operator(condition.operator, left_value, right_value)
        
    async def _parse_string_condition(self, condition_str: str) -> ConditionExpression:
        """Parse a string condition into a ConditionExpression."""
        
        # Simple regex patterns for parsing
        patterns = {
            ComparisonOperator.GREATER_EQUAL: r'(.+?)\s*>=\s*(.+)',
            ComparisonOperator.LESS_EQUAL: r'(.+?)\s*<=\s*(.+)',
            ComparisonOperator.NOT_EQUALS: r'(.+?)\s*!=\s*(.+)',
            ComparisonOperator.EQUALS: r'(.+?)\s*==\s*(.+)',
            ComparisonOperator.GREATER_THAN: r'(.+?)\s*>\s*(.+)',
            ComparisonOperator.LESS_THAN: r'(.+?)\s*<\s*(.+)',
            ComparisonOperator.CONTAINS: r'(.+?)\s+contains\s+(.+)',
            ComparisonOperator.IN: r'(.+?)\s+in\s+(.+)',
            ComparisonOperator.MATCHES: r'(.+?)\s+matches\s+(.+)',
            ComparisonOperator.EXISTS: r'(.+?)\s+exists',
        }
        
        for op, pattern in patterns.items():
            match = re.match(pattern, condition_str.strip(), re.IGNORECASE)
            if match:
                if op == ComparisonOperator.EXISTS:
                    return ConditionExpression(
                        left_operand=match.group(1).strip(),
                        operator=op,
                        right_operand=True
                    )
                else:
                    return ConditionExpression(
                        left_operand=match.group(1).strip(),
                        operator=op,
                        right_operand=match.group(2).strip()
                    )
                    
        raise ValueError(f"Unable to parse condition: {condition_str}")
        
    async def _evaluate_complex_condition(
        self,
        condition: ComplexCondition,
        context: BranchingContext
    ) -> bool:
        """Evaluate a complex condition with multiple expressions."""
        
        if not condition.expressions:
            return True
            
        results = []
        for expr in condition.expressions:
            if isinstance(expr, (ConditionExpression, ComplexCondition)):
                result = await self.evaluate(expr, context)
                results.append(result)
            else:
                raise ValueError(f"Invalid expression type: {type(expr)}")
                
        # Apply logical operator
        if condition.logical_operator == LogicalOperator.AND:
            return all(results)
        elif condition.logical_operator == LogicalOperator.OR:
            return any(results)
        elif condition.logical_operator == LogicalOperator.NOT:
            return not results[0] if results else True
        elif condition.logical_operator == LogicalOperator.XOR:
            return sum(results) == 1
        else:
            raise ValueError(f"Unsupported logical operator: {condition.logical_operator}")
            
    async def _resolve_operand(self, operand: str, context: BranchingContext) -> Any:
        """Resolve an operand value from context."""
        
        if not isinstance(operand, str):
            return operand
            
        operand = operand.strip()
        
        # Variable references
        if operand.startswith("$"):
            var_name = operand[1:]
            
            # Check different context sources
            if var_name in context.workflow_variables:
                return context.workflow_variables[var_name]
            elif var_name in context.task_results:
                return context.task_results[var_name]
            elif var_name in context.system_context:
                return context.system_context[var_name]
            elif var_name in context.user_context:
                return context.user_context[var_name]
            else:
                # Try nested access (e.g., $task.result.status)
                parts = var_name.split('.')
                value = None
                
                for source in [context.task_results, context.workflow_variables, 
                              context.system_context, context.user_context]:
                    try:
                        value = source
                        for part in parts:
                            if isinstance(value, dict):
                                value = value[part]
                            else:
                                value = getattr(value, part)
                        break
                    except (KeyError, AttributeError, TypeError):
                        continue
                        
                if value is not None:
                    return value
                    
                raise ValueError(f"Variable not found: {var_name}")
                
        # Literal values
        if operand.lower() in ['true', 'false']:
            return operand.lower() == 'true'
        elif operand.lower() in ['null', 'none']:
            return None
        elif operand.startswith('"') and operand.endswith('"'):
            return operand[1:-1]  # String literal
        elif operand.startswith("'") and operand.endswith("'"):
            return operand[1:-1]  # String literal
        elif operand.startswith('[') and operand.endswith(']'):
            # List literal
            try:
                return json.loads(operand)
            except json.JSONDecodeError:
                return operand
        elif operand.startswith('{') and operand.endswith('}'):
            # Dict literal
            try:
                return json.loads(operand)
            except json.JSONDecodeError:
                return operand
        else:
            # Try to parse as number
            try:
                if '.' in operand:
                    return float(operand)
                else:
                    return int(operand)
            except ValueError:
                return operand  # Return as string
                
    async def _convert_types(
        self,
        left_value: Any,
        right_value: Any,
        data_type: str
    ) -> tuple:
        """Convert values to appropriate types for comparison."""
        
        if data_type == "auto":
            # Auto-detect type conversion
            if isinstance(left_value, type(right_value)):
                return left_value, right_value
            elif isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                return left_value, right_value
            elif isinstance(left_value, str) or isinstance(right_value, str):
                return str(left_value), str(right_value)
            else:
                return left_value, right_value
        elif data_type == "string":
            return str(left_value), str(right_value)
        elif data_type == "number":
            return float(left_value), float(right_value)
        elif data_type == "boolean":
            return bool(left_value), bool(right_value)
        else:
            return left_value, right_value
            
    async def _apply_operator(
        self,
        op: ComparisonOperator,
        left_value: Any,
        right_value: Any
    ) -> bool:
        """Apply comparison operator to values."""
        
        if op in self.operators:
            return self.operators[op](left_value, right_value)
        elif op == ComparisonOperator.CONTAINS:
            return right_value in left_value
        elif op == ComparisonOperator.NOT_CONTAINS:
            return right_value not in left_value
        elif op == ComparisonOperator.IN:
            return left_value in right_value
        elif op == ComparisonOperator.NOT_IN:
            return left_value not in right_value
        elif op == ComparisonOperator.MATCHES:
            return bool(re.search(str(right_value), str(left_value)))
        elif op == ComparisonOperator.EXISTS:
            return left_value is not None
        elif op == ComparisonOperator.NOT_EXISTS:
            return left_value is None
        else:
            raise ValueError(f"Unsupported operator: {op}")


class TemporalConditionEvaluator(ConditionEvaluator):
    """Evaluates time-based conditions."""
    
    async def evaluate(
        self,
        condition: Union[ConditionExpression, ComplexCondition, str],
        context: BranchingContext
    ) -> bool:
        """Evaluate temporal conditions."""
        
        if isinstance(condition, str):
            return await self._evaluate_temporal_string(condition, context)
            
        # For now, delegate to simple evaluator
        simple_evaluator = SimpleConditionEvaluator()
        return await simple_evaluator.evaluate(condition, context)
        
    async def _evaluate_temporal_string(self, condition: str, context: BranchingContext) -> bool:
        """Evaluate temporal string conditions."""
        
        condition = condition.lower().strip()
        current_time = context.timestamp
        
        # Time-based patterns
        if "after" in condition:
            # e.g., "after 14:00", "after 2023-01-01"
            time_str = condition.split("after")[1].strip()
            target_time = await self._parse_time(time_str, current_time)
            return current_time > target_time
        elif "before" in condition:
            time_str = condition.split("before")[1].strip()
            target_time = await self._parse_time(time_str, current_time)
            return current_time < target_time
        elif "between" in condition:
            # e.g., "between 09:00 and 17:00"
            parts = condition.split("between")[1].split("and")
            if len(parts) == 2:
                start_time = await self._parse_time(parts[0].strip(), current_time)
                end_time = await self._parse_time(parts[1].strip(), current_time)
                return start_time <= current_time <= end_time
        elif "weekday" in condition:
            weekday = current_time.weekday()  # 0=Monday, 6=Sunday
            return weekday < 5  # Monday to Friday
        elif "weekend" in condition:
            weekday = current_time.weekday()
            return weekday >= 5  # Saturday and Sunday
            
        return False
        
    async def _parse_time(self, time_str: str, reference_time: datetime) -> datetime:
        """Parse time string relative to reference time."""
        
        time_str = time_str.strip()
        
        # Try different time formats
        formats = [
            "%H:%M",
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                if fmt == "%H:%M":
                    # Time only - use reference date
                    time_obj = datetime.strptime(time_str, fmt).time()
                    return reference_time.replace(
                        hour=time_obj.hour,
                        minute=time_obj.minute,
                        second=0,
                        microsecond=0
                    )
                else:
                    return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
                
        raise ValueError(f"Unable to parse time: {time_str}")


class ConditionalBranchingEngine:
    """Main engine for managing conditional branching in workflows."""
    
    def __init__(self):
        self.evaluators: Dict[ConditionType, ConditionEvaluator] = {}
        self.custom_evaluators: Dict[str, Callable] = {}
        self.branches: Dict[str, ConditionalBranch] = {}
        
    async def initialize(self):
        """Initialize the conditional branching engine."""
        
        # Setup default evaluators
        self.evaluators[ConditionType.SIMPLE] = SimpleConditionEvaluator()
        self.evaluators[ConditionType.COMPLEX] = SimpleConditionEvaluator()
        self.evaluators[ConditionType.TEMPORAL] = TemporalConditionEvaluator()
        
    async def register_branch(
        self,
        branch: ConditionalBranch
    ):
        """Register a conditional branch."""
        self.branches[branch.branch_id] = branch
        
    async def evaluate_branch(
        self,
        branch_id: str,
        context: BranchingContext
    ) -> bool:
        """Evaluate a specific branch condition."""
        
        if branch_id not in self.branches:
            raise ValueError(f"Branch not found: {branch_id}")
            
        branch = self.branches[branch_id]
        return await self._evaluate_condition(branch.condition, context)
        
    async def get_next_tasks(
        self,
        branch_id: str,
        context: BranchingContext
    ) -> List[str]:
        """Get next tasks based on branch evaluation."""
        
        if branch_id not in self.branches:
            raise ValueError(f"Branch not found: {branch_id}")
            
        branch = self.branches[branch_id]
        condition_result = await self._evaluate_condition(branch.condition, context)
        
        if condition_result:
            return branch.true_path
        else:
            return branch.false_path
            
    async def evaluate_all_branches(
        self,
        context: BranchingContext
    ) -> Dict[str, bool]:
        """Evaluate all registered branches."""
        
        results = {}
        for branch_id, branch in self.branches.items():
            try:
                result = await self._evaluate_condition(branch.condition, context)
                results[branch_id] = result
            except Exception as e:
                print(f"Error evaluating branch {branch_id}: {e}")
                results[branch_id] = False
                
        return results
        
    async def register_custom_evaluator(
        self,
        name: str,
        evaluator: Callable[[Any, BranchingContext], bool]
    ):
        """Register a custom condition evaluator."""
        self.custom_evaluators[name] = evaluator
        
    async def _evaluate_condition(
        self,
        condition: Union[ConditionExpression, ComplexCondition, str],
        context: BranchingContext
    ) -> bool:
        """Evaluate a condition using appropriate evaluator."""
        
        # Determine condition type
        condition_type = await self._determine_condition_type(condition)
        
        # Use custom evaluator if specified
        if isinstance(condition, str) and condition.startswith("custom:"):
            evaluator_name = condition.split(":", 1)[1].split("(")[0]
            if evaluator_name in self.custom_evaluators:
                return await self.custom_evaluators[evaluator_name](condition, context)
                
        # Use appropriate evaluator
        if condition_type in self.evaluators:
            return await self.evaluators[condition_type].evaluate(condition, context)
        else:
            raise ValueError(f"No evaluator found for condition type: {condition_type}")
            
    async def _determine_condition_type(self, condition: Any) -> ConditionType:
        """Determine the type of condition."""
        
        if isinstance(condition, ConditionExpression):
            return ConditionType.SIMPLE
        elif isinstance(condition, ComplexCondition):
            return ConditionType.COMPLEX
        elif isinstance(condition, str):
            condition_lower = condition.lower()
            if any(word in condition_lower for word in ["after", "before", "between", "weekday", "weekend"]):
                return ConditionType.TEMPORAL
            elif condition.startswith("custom:"):
                return ConditionType.CUSTOM
            else:
                return ConditionType.SIMPLE
        else:
            return ConditionType.SIMPLE
            
    async def create_context_from_execution(
        self,
        execution: WorkflowExecution,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> BranchingContext:
        """Create branching context from workflow execution."""
        
        # Extract task results
        task_results = {}
        for task_id, result in execution.task_results.items():
            if hasattr(result, 'result') and result.result:
                task_results[task_id] = result.result
            else:
                task_results[task_id] = str(result)
                
        # Create context
        context = BranchingContext(
            workflow_id=execution.workflow_id,
            execution_id=execution.execution_id,
            current_task_id=execution.current_task_id,
            task_results=task_results,
            workflow_variables=execution.variables.copy(),
            system_context={
                "status": execution.status.value,
                "started_at": execution.started_at,
                "progress": len([r for r in execution.task_results.values() if r.status.value == "completed"]) / len(execution.task_results) if execution.task_results else 0
            },
            user_context=additional_context or {},
            timestamp=datetime.now()
        )
        
        return context