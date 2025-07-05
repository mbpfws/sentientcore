"""Interactive Task Breakdown Engine

Breaks down complex tasks into manageable, interactive workflow steps
with user input, approval gates, and dynamic refinement capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from pathlib import Path

from ..state.interactive_workflow_models import (
    InteractiveWorkflow, WorkflowStep, StepType, StepPriority,
    InteractionType, UserInteractionRequest, UserInteractionResponse
)
from ..models import EnhancedTask, TaskStatus, AgentType
from ..state.interactive_state_manager import InteractiveStateManager

class BreakdownStrategy(Enum):
    """Task breakdown strategies."""
    SEQUENTIAL = "sequential"  # Linear step-by-step breakdown
    PARALLEL = "parallel"  # Parallel execution branches
    HIERARCHICAL = "hierarchical"  # Nested sub-workflows
    ADAPTIVE = "adaptive"  # Dynamic breakdown based on complexity
    USER_GUIDED = "user_guided"  # User-driven breakdown process

class ComplexityLevel(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"  # 1-3 steps
    MODERATE = "moderate"  # 4-8 steps
    COMPLEX = "complex"  # 9-15 steps
    VERY_COMPLEX = "very_complex"  # 16+ steps

class TaskCategory(Enum):
    """Categories of tasks for specialized breakdown."""
    CODE_DEVELOPMENT = "code_development"
    SYSTEM_DESIGN = "system_design"
    DATA_PROCESSING = "data_processing"
    TESTING_QA = "testing_qa"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"
    DOCUMENTATION = "documentation"
    MAINTENANCE = "maintenance"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"

@dataclass
class BreakdownContext:
    """Context for task breakdown."""
    original_task: EnhancedTask
    user_id: str
    session_id: str
    target_complexity: ComplexityLevel
    preferred_strategy: BreakdownStrategy
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    existing_artifacts: List[str] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BreakdownResult:
    """Result of task breakdown."""
    workflow: InteractiveWorkflow
    breakdown_strategy: BreakdownStrategy
    complexity_analysis: Dict[str, Any]
    estimated_duration: timedelta
    confidence_score: float
    alternative_breakdowns: List['BreakdownResult'] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class TaskBreakdownStrategy(ABC):
    """Abstract base class for task breakdown strategies."""
    
    @abstractmethod
    def can_handle(self, task: EnhancedTask, context: BreakdownContext) -> bool:
        """Check if this strategy can handle the task."""
        pass
    
    @abstractmethod
    async def breakdown_task(self, task: EnhancedTask, context: BreakdownContext) -> BreakdownResult:
        """Break down the task into workflow steps."""
        pass
    
    @abstractmethod
    def estimate_complexity(self, task: EnhancedTask) -> ComplexityLevel:
        """Estimate task complexity."""
        pass

class CodeDevelopmentBreakdownStrategy(TaskBreakdownStrategy):
    """Strategy for breaking down code development tasks."""
    
    def can_handle(self, task: EnhancedTask, context: BreakdownContext) -> bool:
        """Check if this is a code development task."""
        keywords = ['code', 'implement', 'develop', 'create', 'build', 'program', 'function', 'class', 'api']
        task_text = f"{task.description} {' '.join(task.requirements)}".lower()
        return any(keyword in task_text for keyword in keywords)
    
    def estimate_complexity(self, task: EnhancedTask) -> ComplexityLevel:
        """Estimate complexity based on task characteristics."""
        complexity_indicators = {
            'api': 2, 'database': 3, 'authentication': 3, 'testing': 2,
            'integration': 3, 'deployment': 2, 'optimization': 3,
            'algorithm': 4, 'machine learning': 5, 'distributed': 4
        }
        
        task_text = f"{task.description} {' '.join(task.requirements)}".lower()
        complexity_score = sum(score for keyword, score in complexity_indicators.items() 
                             if keyword in task_text)
        
        # Add complexity based on requirements count
        complexity_score += len(task.requirements)
        
        if complexity_score <= 5:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 10:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 15:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    async def breakdown_task(self, task: EnhancedTask, context: BreakdownContext) -> BreakdownResult:
        """Break down code development task."""
        steps = []
        
        # 1. Analysis and Planning
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Analyze Requirements",
            description=f"Analyze and understand the requirements for: {task.description}",
            step_type=StepType.ANALYSIS,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(minutes=30),
            requires_approval=True,
            dependencies=[],
            metadata={
                "analysis_type": "requirements",
                "deliverables": ["requirements_analysis.md", "technical_specifications.md"]
            }
        ))
        
        # 2. Design Phase
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Design Architecture",
            description="Design the system architecture and component structure",
            step_type=StepType.DESIGN,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(hours=1),
            requires_approval=True,
            dependencies=[steps[0].id],
            metadata={
                "design_type": "architecture",
                "deliverables": ["architecture_diagram.md", "component_design.md"]
            }
        ))
        
        # 3. Environment Setup
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Setup Development Environment",
            description="Configure development environment and dependencies",
            step_type=StepType.CONFIGURATION,
            priority=StepPriority.MEDIUM,
            estimated_duration=timedelta(minutes=45),
            requires_approval=False,
            dependencies=[steps[1].id],
            metadata={
                "setup_type": "development",
                "deliverables": ["requirements.txt", "setup_instructions.md"]
            }
        ))
        
        # 4. Core Implementation
        core_features = self._identify_core_features(task)
        for i, feature in enumerate(core_features):
            steps.append(WorkflowStep(
                id=str(uuid.uuid4()),
                title=f"Implement {feature}",
                description=f"Develop and implement the {feature} functionality",
                step_type=StepType.CODE_GENERATION,
                priority=StepPriority.HIGH,
                estimated_duration=timedelta(hours=2),
                requires_approval=True,
                dependencies=[steps[2].id] if i == 0 else [steps[-2].id],
                metadata={
                    "feature_name": feature,
                    "implementation_type": "core",
                    "deliverables": [f"{feature.lower().replace(' ', '_')}.py"]
                }
            ))
        
        # 5. Testing
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Implement Unit Tests",
            description="Create comprehensive unit tests for all components",
            step_type=StepType.TESTING,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(hours=1),
            requires_approval=True,
            dependencies=[steps[-1].id],
            metadata={
                "test_type": "unit",
                "deliverables": ["test_suite.py", "test_report.html"]
            }
        ))
        
        # 6. Integration Testing
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Integration Testing",
            description="Perform integration testing and system validation",
            step_type=StepType.TESTING,
            priority=StepPriority.MEDIUM,
            estimated_duration=timedelta(minutes=45),
            requires_approval=True,
            dependencies=[steps[-1].id],
            metadata={
                "test_type": "integration",
                "deliverables": ["integration_test_report.html"]
            }
        ))
        
        # 7. Documentation
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Create Documentation",
            description="Generate comprehensive documentation and usage examples",
            step_type=StepType.DOCUMENTATION,
            priority=StepPriority.MEDIUM,
            estimated_duration=timedelta(minutes=30),
            requires_approval=False,
            dependencies=[steps[-1].id],
            metadata={
                "doc_type": "user_guide",
                "deliverables": ["README.md", "API_documentation.md", "examples/"]
            }
        ))
        
        # Create workflow
        workflow = InteractiveWorkflow(
            id=str(uuid.uuid4()),
            title=f"Development: {task.description}",
            description=f"Interactive development workflow for: {task.description}",
            user_id=context.user_id,
            session_id=context.session_id,
            steps=steps,
            metadata={
                "original_task_id": task.id,
                "breakdown_strategy": "code_development",
                "estimated_complexity": self.estimate_complexity(task).value
            }
        )
        
        # Calculate total estimated duration
        total_duration = sum((step.estimated_duration for step in steps), timedelta())
        
        # Complexity analysis
        complexity_analysis = {
            "level": self.estimate_complexity(task).value,
            "total_steps": len(steps),
            "approval_gates": sum(1 for step in steps if step.requires_approval),
            "estimated_duration_hours": total_duration.total_seconds() / 3600,
            "core_features_count": len(core_features),
            "risk_factors": self._identify_risk_factors(task)
        }
        
        return BreakdownResult(
            workflow=workflow,
            breakdown_strategy=BreakdownStrategy.SEQUENTIAL,
            complexity_analysis=complexity_analysis,
            estimated_duration=total_duration,
            confidence_score=0.85,
            recommendations=[
                "Review each approval gate carefully before proceeding",
                "Consider parallel execution for independent features",
                "Maintain comprehensive test coverage throughout development"
            ]
        )
    
    def _identify_core_features(self, task: EnhancedTask) -> List[str]:
        """Identify core features from task description."""
        # Simple feature extraction (in real implementation, this would be more sophisticated)
        features = []
        
        if 'api' in task.description.lower():
            features.append('API Endpoints')
        if 'database' in task.description.lower():
            features.append('Database Layer')
        if 'auth' in task.description.lower():
            features.append('Authentication System')
        if 'ui' in task.description.lower() or 'interface' in task.description.lower():
            features.append('User Interface')
        
        # Default core feature if none identified
        if not features:
            features.append('Core Functionality')
        
        return features
    
    def _identify_risk_factors(self, task: EnhancedTask) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        task_text = task.description.lower()
        
        if 'complex' in task_text or 'advanced' in task_text:
            risks.append('High complexity implementation')
        if 'integration' in task_text:
            risks.append('External system dependencies')
        if 'performance' in task_text or 'optimization' in task_text:
            risks.append('Performance requirements')
        if 'security' in task_text or 'auth' in task_text:
            risks.append('Security considerations')
        
        return risks

class SystemDesignBreakdownStrategy(TaskBreakdownStrategy):
    """Strategy for breaking down system design tasks."""
    
    def can_handle(self, task: EnhancedTask, context: BreakdownContext) -> bool:
        """Check if this is a system design task."""
        keywords = ['design', 'architecture', 'system', 'structure', 'framework', 'pattern']
        task_text = f"{task.description} {' '.join(task.requirements)}".lower()
        return any(keyword in task_text for keyword in keywords)
    
    def estimate_complexity(self, task: EnhancedTask) -> ComplexityLevel:
        """Estimate complexity for system design."""
        complexity_indicators = {
            'microservices': 4, 'distributed': 5, 'scalable': 3,
            'real-time': 4, 'high-availability': 4, 'security': 3,
            'integration': 3, 'api': 2, 'database': 2
        }
        
        task_text = f"{task.description} {' '.join(task.requirements)}".lower()
        complexity_score = sum(score for keyword, score in complexity_indicators.items() 
                             if keyword in task_text)
        
        if complexity_score <= 6:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 12:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 18:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.VERY_COMPLEX
    
    async def breakdown_task(self, task: EnhancedTask, context: BreakdownContext) -> BreakdownResult:
        """Break down system design task."""
        steps = []
        
        # 1. Requirements Analysis
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Analyze System Requirements",
            description="Gather and analyze functional and non-functional requirements",
            step_type=StepType.ANALYSIS,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(hours=1),
            requires_approval=True,
            dependencies=[],
            metadata={
                "analysis_type": "system_requirements",
                "deliverables": ["requirements_document.md", "constraints_analysis.md"]
            }
        ))
        
        # 2. High-Level Architecture
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Design High-Level Architecture",
            description="Create overall system architecture and component relationships",
            step_type=StepType.DESIGN,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(hours=2),
            requires_approval=True,
            dependencies=[steps[0].id],
            metadata={
                "design_type": "high_level_architecture",
                "deliverables": ["architecture_diagram.md", "component_overview.md"]
            }
        ))
        
        # 3. Detailed Component Design
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Design Individual Components",
            description="Detail the design of each system component",
            step_type=StepType.DESIGN,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(hours=3),
            requires_approval=True,
            dependencies=[steps[1].id],
            metadata={
                "design_type": "component_design",
                "deliverables": ["component_specifications.md", "interface_definitions.md"]
            }
        ))
        
        # 4. Data Architecture
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Design Data Architecture",
            description="Design data models, storage, and flow patterns",
            step_type=StepType.DESIGN,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(hours=1.5),
            requires_approval=True,
            dependencies=[steps[2].id],
            metadata={
                "design_type": "data_architecture",
                "deliverables": ["data_model.md", "database_schema.sql"]
            }
        ))
        
        # 5. Security Design
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Design Security Architecture",
            description="Define security measures, authentication, and authorization",
            step_type=StepType.DESIGN,
            priority=StepPriority.HIGH,
            estimated_duration=timedelta(hours=1),
            requires_approval=True,
            dependencies=[steps[3].id],
            metadata={
                "design_type": "security_architecture",
                "deliverables": ["security_design.md", "threat_model.md"]
            }
        ))
        
        # 6. Performance and Scalability
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Design for Performance and Scalability",
            description="Define performance requirements and scalability strategies",
            step_type=StepType.DESIGN,
            priority=StepPriority.MEDIUM,
            estimated_duration=timedelta(hours=1),
            requires_approval=True,
            dependencies=[steps[4].id],
            metadata={
                "design_type": "performance_scalability",
                "deliverables": ["performance_requirements.md", "scalability_plan.md"]
            }
        ))
        
        # 7. Implementation Plan
        steps.append(WorkflowStep(
            id=str(uuid.uuid4()),
            title="Create Implementation Plan",
            description="Develop detailed implementation roadmap and milestones",
            step_type=StepType.PLANNING,
            priority=StepPriority.MEDIUM,
            estimated_duration=timedelta(minutes=45),
            requires_approval=True,
            dependencies=[steps[5].id],
            metadata={
                "planning_type": "implementation",
                "deliverables": ["implementation_plan.md", "project_timeline.md"]
            }
        ))
        
        # Create workflow
        workflow = InteractiveWorkflow(
            id=str(uuid.uuid4()),
            title=f"System Design: {task.description}",
            description=f"Interactive system design workflow for: {task.description}",
            user_id=context.user_id,
            session_id=context.session_id,
            steps=steps,
            metadata={
                "original_task_id": task.id,
                "breakdown_strategy": "system_design",
                "estimated_complexity": self.estimate_complexity(task).value
            }
        )
        
        total_duration = sum((step.estimated_duration for step in steps), timedelta())
        
        complexity_analysis = {
            "level": self.estimate_complexity(task).value,
            "total_steps": len(steps),
            "approval_gates": sum(1 for step in steps if step.requires_approval),
            "estimated_duration_hours": total_duration.total_seconds() / 3600,
            "design_phases": 6,
            "risk_factors": self._identify_design_risks(task)
        }
        
        return BreakdownResult(
            workflow=workflow,
            breakdown_strategy=BreakdownStrategy.SEQUENTIAL,
            complexity_analysis=complexity_analysis,
            estimated_duration=total_duration,
            confidence_score=0.90,
            recommendations=[
                "Ensure stakeholder review at each design phase",
                "Consider iterative refinement of designs",
                "Validate designs against requirements frequently"
            ]
        )
    
    def _identify_design_risks(self, task: EnhancedTask) -> List[str]:
        """Identify design-specific risk factors."""
        risks = []
        task_text = task.description.lower()
        
        if 'distributed' in task_text or 'microservices' in task_text:
            risks.append('Distributed system complexity')
        if 'real-time' in task_text:
            risks.append('Real-time processing requirements')
        if 'scale' in task_text or 'high-load' in task_text:
            risks.append('Scalability challenges')
        if 'legacy' in task_text or 'migration' in task_text:
            risks.append('Legacy system integration')
        
        return risks

class InteractiveTaskBreakdownEngine:
    """Engine for breaking down complex tasks into interactive workflows."""
    
    def __init__(self, state_manager: InteractiveStateManager):
        self.state_manager = state_manager
        
        # Breakdown strategies
        self.strategies: List[TaskBreakdownStrategy] = [
            CodeDevelopmentBreakdownStrategy(),
            SystemDesignBreakdownStrategy()
        ]
        
        # Task categorization patterns
        self.category_patterns = {
            TaskCategory.CODE_DEVELOPMENT: ['code', 'implement', 'develop', 'program', 'function'],
            TaskCategory.SYSTEM_DESIGN: ['design', 'architecture', 'system', 'structure'],
            TaskCategory.TESTING_QA: ['test', 'quality', 'validation', 'verify'],
            TaskCategory.DEPLOYMENT: ['deploy', 'release', 'production', 'launch'],
            TaskCategory.DOCUMENTATION: ['document', 'guide', 'manual', 'readme']
        }
        
        # Metrics
        self.breakdown_metrics = {
            "tasks_broken_down": 0,
            "workflows_created": 0,
            "average_steps_per_workflow": 0.0,
            "user_interactions": 0,
            "refinements_requested": 0
        }
        
        print("Interactive Task Breakdown Engine initialized")
    
    def add_strategy(self, strategy: TaskBreakdownStrategy):
        """Add a custom breakdown strategy."""
        self.strategies.append(strategy)
    
    def categorize_task(self, task: EnhancedTask) -> TaskCategory:
        """Categorize a task based on its content."""
        task_text = f"{task.description} {' '.join(task.requirements)}".lower()
        
        # Score each category
        category_scores = {}
        for category, patterns in self.category_patterns.items():
            score = sum(1 for pattern in patterns if pattern in task_text)
            if score > 0:
                category_scores[category] = score
        
        # Return highest scoring category or default
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return TaskCategory.CODE_DEVELOPMENT  # Default
    
    def get_strategy_for_task(self, task: EnhancedTask, context: BreakdownContext) -> Optional[TaskBreakdownStrategy]:
        """Get the best strategy for breaking down a task."""
        for strategy in self.strategies:
            if strategy.can_handle(task, context):
                return strategy
        return None
    
    async def breakdown_task(self, task: EnhancedTask, user_id: str, session_id: str,
                           preferred_strategy: Optional[BreakdownStrategy] = None,
                           target_complexity: Optional[ComplexityLevel] = None,
                           constraints: Optional[Dict[str, Any]] = None) -> BreakdownResult:
        """Break down a task into an interactive workflow."""
        
        # Create breakdown context
        context = BreakdownContext(
            original_task=task,
            user_id=user_id,
            session_id=session_id,
            target_complexity=target_complexity or ComplexityLevel.MODERATE,
            preferred_strategy=preferred_strategy or BreakdownStrategy.SEQUENTIAL,
            constraints=constraints or {},
            preferences={},
            existing_artifacts=[],
            domain_knowledge={}
        )
        
        # Get appropriate strategy
        strategy = self.get_strategy_for_task(task, context)
        if not strategy:
            # Fallback to generic breakdown
            strategy = self._create_generic_strategy()
        
        # Perform breakdown
        result = await strategy.breakdown_task(task, context)
        
        # Update metrics
        self.breakdown_metrics["tasks_broken_down"] += 1
        self.breakdown_metrics["workflows_created"] += 1
        
        current_avg = self.breakdown_metrics["average_steps_per_workflow"]
        total_workflows = self.breakdown_metrics["workflows_created"]
        new_avg = ((current_avg * (total_workflows - 1)) + len(result.workflow.steps)) / total_workflows
        self.breakdown_metrics["average_steps_per_workflow"] = new_avg
        
        return result
    
    async def request_user_refinement(self, workflow: InteractiveWorkflow, 
                                     refinement_request: str, user_id: str, 
                                     session_id: str) -> UserInteractionRequest:
        """Request user input for workflow refinement."""
        
        interaction_request = UserInteractionRequest(
            id=str(uuid.uuid4()),
            interaction_type=InteractionType.REFINEMENT_REQUEST,
            title="Workflow Refinement Needed",
            description=refinement_request,
            user_id=user_id,
            session_id=session_id,
            metadata={
                "workflow_id": workflow.id,
                "current_steps_count": len(workflow.steps),
                "refinement_type": "user_guided"
            }
        )
        
        # Update metrics
        self.breakdown_metrics["user_interactions"] += 1
        self.breakdown_metrics["refinements_requested"] += 1
        
        return interaction_request
    
    async def apply_user_refinement(self, workflow: InteractiveWorkflow, 
                                   user_response: UserInteractionResponse) -> InteractiveWorkflow:
        """Apply user refinements to a workflow."""
        
        if not user_response.response_data:
            return workflow
        
        refinements = user_response.response_data
        
        # Apply step modifications
        if 'step_modifications' in refinements:
            for step_mod in refinements['step_modifications']:
                step_id = step_mod.get('step_id')
                step = workflow.get_step_by_id(step_id)
                if step:
                    if 'title' in step_mod:
                        step.title = step_mod['title']
                    if 'description' in step_mod:
                        step.description = step_mod['description']
                    if 'requires_approval' in step_mod:
                        step.requires_approval = step_mod['requires_approval']
                    if 'priority' in step_mod:
                        step.priority = StepPriority(step_mod['priority'])
        
        # Add new steps
        if 'new_steps' in refinements:
            for new_step_data in refinements['new_steps']:
                new_step = WorkflowStep(
                    id=str(uuid.uuid4()),
                    title=new_step_data['title'],
                    description=new_step_data['description'],
                    step_type=StepType(new_step_data.get('step_type', 'analysis')),
                    priority=StepPriority(new_step_data.get('priority', 'medium')),
                    estimated_duration=timedelta(minutes=new_step_data.get('duration_minutes', 30)),
                    requires_approval=new_step_data.get('requires_approval', True),
                    dependencies=new_step_data.get('dependencies', [])
                )
                
                # Insert at specified position or append
                position = new_step_data.get('position', len(workflow.steps))
                workflow.steps.insert(position, new_step)
        
        # Remove steps
        if 'remove_steps' in refinements:
            step_ids_to_remove = set(refinements['remove_steps'])
            workflow.steps = [step for step in workflow.steps if step.id not in step_ids_to_remove]
        
        # Update workflow metadata
        workflow.metadata['last_refined_at'] = datetime.now().isoformat()
        workflow.metadata['refinement_count'] = workflow.metadata.get('refinement_count', 0) + 1
        
        return workflow
    
    def _create_generic_strategy(self) -> TaskBreakdownStrategy:
        """Create a generic breakdown strategy for unrecognized tasks."""
        
        class GenericBreakdownStrategy(TaskBreakdownStrategy):
            def can_handle(self, task: EnhancedTask, context: BreakdownContext) -> bool:
                return True  # Can handle any task
            
            def estimate_complexity(self, task: EnhancedTask) -> ComplexityLevel:
                # Simple heuristic based on description length and requirements
                desc_length = len(task.description)
                req_count = len(task.requirements)
                
                complexity_score = (desc_length // 50) + (req_count * 2)
                
                if complexity_score <= 3:
                    return ComplexityLevel.SIMPLE
                elif complexity_score <= 6:
                    return ComplexityLevel.MODERATE
                elif complexity_score <= 10:
                    return ComplexityLevel.COMPLEX
                else:
                    return ComplexityLevel.VERY_COMPLEX
            
            async def breakdown_task(self, task: EnhancedTask, context: BreakdownContext) -> BreakdownResult:
                steps = [
                    WorkflowStep(
                        id=str(uuid.uuid4()),
                        title="Analyze Task",
                        description=f"Analyze and understand: {task.description}",
                        step_type=StepType.ANALYSIS,
                        priority=StepPriority.HIGH,
                        estimated_duration=timedelta(minutes=30),
                        requires_approval=True,
                        dependencies=[]
                    ),
                    WorkflowStep(
                        id=str(uuid.uuid4()),
                        title="Plan Implementation",
                        description="Create detailed implementation plan",
                        step_type=StepType.PLANNING,
                        priority=StepPriority.HIGH,
                        estimated_duration=timedelta(minutes=45),
                        requires_approval=True,
                        dependencies=[]
                    ),
                    WorkflowStep(
                        id=str(uuid.uuid4()),
                        title="Execute Task",
                        description="Execute the main task implementation",
                        step_type=StepType.IMPLEMENTATION,
                        priority=StepPriority.HIGH,
                        estimated_duration=timedelta(hours=2),
                        requires_approval=True,
                        dependencies=[]
                    ),
                    WorkflowStep(
                        id=str(uuid.uuid4()),
                        title="Validate Results",
                        description="Validate and verify task completion",
                        step_type=StepType.VALIDATION,
                        priority=StepPriority.MEDIUM,
                        estimated_duration=timedelta(minutes=30),
                        requires_approval=True,
                        dependencies=[]
                    )
                ]
                
                # Set up dependencies
                for i in range(1, len(steps)):
                    steps[i].dependencies = [steps[i-1].id]
                
                workflow = InteractiveWorkflow(
                    id=str(uuid.uuid4()),
                    title=f"Generic Task: {task.description}",
                    description=f"Generic workflow for: {task.description}",
                    user_id=context.user_id,
                    session_id=context.session_id,
                    steps=steps,
                    metadata={
                        "original_task_id": task.id,
                        "breakdown_strategy": "generic",
                        "estimated_complexity": self.estimate_complexity(task).value
                    }
                )
                
                total_duration = sum((step.estimated_duration for step in steps), timedelta())
                
                return BreakdownResult(
                    workflow=workflow,
                    breakdown_strategy=BreakdownStrategy.SEQUENTIAL,
                    complexity_analysis={
                        "level": self.estimate_complexity(task).value,
                        "total_steps": len(steps),
                        "approval_gates": len(steps),
                        "estimated_duration_hours": total_duration.total_seconds() / 3600
                    },
                    estimated_duration=total_duration,
                    confidence_score=0.70,
                    recommendations=[
                        "Consider refining this generic breakdown for better specificity",
                        "Add domain-specific steps as needed"
                    ]
                )
        
        return GenericBreakdownStrategy()
    
    def get_breakdown_metrics(self) -> Dict[str, Any]:
        """Get breakdown engine metrics."""
        return self.breakdown_metrics.copy()
    
    def get_supported_categories(self) -> List[TaskCategory]:
        """Get list of supported task categories."""
        return list(self.category_patterns.keys())

# Factory function
def create_task_breakdown_engine(state_manager: InteractiveStateManager) -> InteractiveTaskBreakdownEngine:
    """Create a task breakdown engine with default configuration."""
    return InteractiveTaskBreakdownEngine(state_manager)