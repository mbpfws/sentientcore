"""Dynamic Workflow Generation Module

This module provides dynamic workflow generation capabilities based on:
- Project complexity analysis
- User requirements
- Available resources
- Historical patterns
- Conditional branching logic
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

from .advanced_engine import (
    WorkflowDefinition,
    WorkflowTask,
    WorkflowCondition,
    ExecutionMode,
    TaskStatus
)


class ProjectComplexity(Enum):
    """Project complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ENTERPRISE = "enterprise"


class WorkflowPattern(Enum):
    """Common workflow patterns."""
    LINEAR = "linear"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"
    HYBRID = "hybrid"


@dataclass
class ProjectRequirements:
    """Project requirements for workflow generation."""
    project_type: str
    complexity: ProjectComplexity
    timeline: timedelta
    team_size: int
    technologies: List[str]
    constraints: Dict[str, Any] = field(default_factory=dict)
    priorities: List[str] = field(default_factory=list)
    risk_tolerance: str = "medium"  # low, medium, high
    quality_requirements: str = "standard"  # basic, standard, high, critical


@dataclass
class WorkflowTemplate:
    """Template for generating workflows."""
    template_id: str
    name: str
    description: str
    pattern: WorkflowPattern
    complexity_range: List[ProjectComplexity]
    base_tasks: List[Dict[str, Any]]
    conditional_tasks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    decision_points: List[Dict[str, Any]] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: timedelta = field(default_factory=lambda: timedelta(hours=8))


class DynamicWorkflowGenerator:
    """Generates workflows dynamically based on project requirements."""
    
    def __init__(self):
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.complexity_analyzers: Dict[str, callable] = {}
        self.decision_engines: Dict[str, callable] = {}
        self.optimization_strategies: Dict[str, callable] = {}
        
    async def initialize(self):
        """Initialize the dynamic workflow generator."""
        await self._load_templates()
        await self._setup_analyzers()
        await self._setup_decision_engines()
        await self._setup_optimization_strategies()
        
    async def generate_workflow(
        self,
        requirements: ProjectRequirements,
        template_id: Optional[str] = None
    ) -> WorkflowDefinition:
        """Generate a workflow based on project requirements."""
        
        # Analyze project complexity if not provided
        if not template_id:
            template_id = await self._select_best_template(requirements)
            
        template = self.templates.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
            
        # Generate base workflow
        workflow = await self._generate_base_workflow(template, requirements)
        
        # Apply conditional logic
        workflow = await self._apply_conditional_logic(workflow, requirements, template)
        
        # Optimize workflow
        workflow = await self._optimize_workflow(workflow, requirements)
        
        # Validate workflow
        await self._validate_workflow(workflow)
        
        return workflow
        
    async def _load_templates(self):
        """Load predefined workflow templates."""
        
        # Feature Development Template
        self.templates["feature_development"] = WorkflowTemplate(
            template_id="feature_development",
            name="Feature Development Workflow",
            description="Complete feature development lifecycle",
            pattern=WorkflowPattern.HYBRID,
            complexity_range=[ProjectComplexity.SIMPLE, ProjectComplexity.MODERATE, ProjectComplexity.COMPLEX],
            base_tasks=[
                {
                    "id": "requirements_analysis",
                    "name": "Requirements Analysis",
                    "agent_type": "research_agent",
                    "priority": "high",
                    "estimated_duration": 2
                },
                {
                    "id": "architecture_design",
                    "name": "Architecture Design",
                    "agent_type": "architect_agent",
                    "dependencies": ["requirements_analysis"],
                    "priority": "high",
                    "estimated_duration": 4
                },
                {
                    "id": "implementation",
                    "name": "Feature Implementation",
                    "agent_type": "coding_agent",
                    "dependencies": ["architecture_design"],
                    "priority": "high",
                    "estimated_duration": 8
                },
                {
                    "id": "testing",
                    "name": "Testing & Validation",
                    "agent_type": "testing_agent",
                    "dependencies": ["implementation"],
                    "priority": "high",
                    "estimated_duration": 4
                }
            ],
            conditional_tasks={
                "high_complexity": [
                    {
                        "id": "performance_optimization",
                        "name": "Performance Optimization",
                        "agent_type": "optimization_agent",
                        "dependencies": ["testing"],
                        "estimated_duration": 3
                    },
                    {
                        "id": "security_review",
                        "name": "Security Review",
                        "agent_type": "security_agent",
                        "dependencies": ["testing"],
                        "estimated_duration": 2
                    }
                ],
                "enterprise_grade": [
                    {
                        "id": "documentation",
                        "name": "Documentation",
                        "agent_type": "documentation_agent",
                        "dependencies": ["testing"],
                        "estimated_duration": 3
                    },
                    {
                        "id": "deployment_planning",
                        "name": "Deployment Planning",
                        "agent_type": "deployment_agent",
                        "dependencies": ["testing"],
                        "estimated_duration": 2
                    }
                ]
            },
            decision_points=[
                {
                    "point_id": "complexity_check",
                    "condition": "complexity >= complex",
                    "true_path": "high_complexity",
                    "false_path": "continue"
                },
                {
                    "point_id": "enterprise_check",
                    "condition": "complexity == enterprise OR quality_requirements == critical",
                    "true_path": "enterprise_grade",
                    "false_path": "continue"
                }
            ]
        )
        
        # Research Workflow Template
        self.templates["research_workflow"] = WorkflowTemplate(
            template_id="research_workflow",
            name="Research & Analysis Workflow",
            description="Comprehensive research and analysis workflow",
            pattern=WorkflowPattern.ITERATIVE,
            complexity_range=[ProjectComplexity.SIMPLE, ProjectComplexity.MODERATE],
            base_tasks=[
                {
                    "id": "initial_research",
                    "name": "Initial Research",
                    "agent_type": "research_agent",
                    "priority": "high",
                    "estimated_duration": 3
                },
                {
                    "id": "data_analysis",
                    "name": "Data Analysis",
                    "agent_type": "analysis_agent",
                    "dependencies": ["initial_research"],
                    "priority": "high",
                    "estimated_duration": 4
                },
                {
                    "id": "synthesis",
                    "name": "Knowledge Synthesis",
                    "agent_type": "synthesis_agent",
                    "dependencies": ["data_analysis"],
                    "priority": "high",
                    "estimated_duration": 2
                }
            ],
            conditional_tasks={
                "deep_research": [
                    {
                        "id": "expert_consultation",
                        "name": "Expert Consultation",
                        "agent_type": "consultation_agent",
                        "dependencies": ["initial_research"],
                        "estimated_duration": 2
                    },
                    {
                        "id": "comparative_analysis",
                        "name": "Comparative Analysis",
                        "agent_type": "analysis_agent",
                        "dependencies": ["expert_consultation"],
                        "estimated_duration": 3
                    }
                ]
            },
            decision_points=[
                {
                    "point_id": "depth_check",
                    "condition": "complexity >= moderate OR priorities contains 'thorough'",
                    "true_path": "deep_research",
                    "false_path": "continue"
                }
            ]
        )
        
        # Bug Fix Workflow Template
        self.templates["bug_fix_workflow"] = WorkflowTemplate(
            template_id="bug_fix_workflow",
            name="Bug Fix Workflow",
            description="Systematic bug identification and resolution",
            pattern=WorkflowPattern.LINEAR,
            complexity_range=[ProjectComplexity.SIMPLE, ProjectComplexity.MODERATE],
            base_tasks=[
                {
                    "id": "bug_reproduction",
                    "name": "Bug Reproduction",
                    "agent_type": "testing_agent",
                    "priority": "critical",
                    "estimated_duration": 1
                },
                {
                    "id": "root_cause_analysis",
                    "name": "Root Cause Analysis",
                    "agent_type": "analysis_agent",
                    "dependencies": ["bug_reproduction"],
                    "priority": "critical",
                    "estimated_duration": 2
                },
                {
                    "id": "fix_implementation",
                    "name": "Fix Implementation",
                    "agent_type": "coding_agent",
                    "dependencies": ["root_cause_analysis"],
                    "priority": "critical",
                    "estimated_duration": 3
                },
                {
                    "id": "fix_validation",
                    "name": "Fix Validation",
                    "agent_type": "testing_agent",
                    "dependencies": ["fix_implementation"],
                    "priority": "critical",
                    "estimated_duration": 1
                }
            ]
        )
        
    async def _setup_analyzers(self):
        """Setup complexity analyzers."""
        
        async def analyze_project_complexity(requirements: ProjectRequirements) -> ProjectComplexity:
            """Analyze project complexity based on requirements."""
            complexity_score = 0
            
            # Technology complexity
            if len(requirements.technologies) > 5:
                complexity_score += 2
            elif len(requirements.technologies) > 3:
                complexity_score += 1
                
            # Team size factor
            if requirements.team_size > 10:
                complexity_score += 2
            elif requirements.team_size > 5:
                complexity_score += 1
                
            # Timeline pressure
            if requirements.timeline < timedelta(days=7):
                complexity_score += 2
            elif requirements.timeline < timedelta(days=30):
                complexity_score += 1
                
            # Quality requirements
            if requirements.quality_requirements == "critical":
                complexity_score += 2
            elif requirements.quality_requirements == "high":
                complexity_score += 1
                
            # Map score to complexity
            if complexity_score >= 6:
                return ProjectComplexity.ENTERPRISE
            elif complexity_score >= 4:
                return ProjectComplexity.COMPLEX
            elif complexity_score >= 2:
                return ProjectComplexity.MODERATE
            else:
                return ProjectComplexity.SIMPLE
                
        self.complexity_analyzers["default"] = analyze_project_complexity
        
    async def _setup_decision_engines(self):
        """Setup decision engines for conditional logic."""
        
        async def evaluate_condition(condition: str, context: Dict[str, Any]) -> bool:
            """Evaluate a condition string against context."""
            try:
                # Simple condition evaluation
                # In production, use a proper expression evaluator
                
                if "complexity >= complex" in condition:
                    return context.get("complexity") in [ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]
                elif "complexity == enterprise" in condition:
                    return context.get("complexity") == ProjectComplexity.ENTERPRISE
                elif "quality_requirements == critical" in condition:
                    return context.get("quality_requirements") == "critical"
                elif "priorities contains" in condition:
                    priority = condition.split("'")[1]
                    return priority in context.get("priorities", [])
                elif "complexity >= moderate" in condition:
                    return context.get("complexity") in [ProjectComplexity.MODERATE, ProjectComplexity.COMPLEX, ProjectComplexity.ENTERPRISE]
                    
                return False
            except Exception:
                return False
                
        self.decision_engines["default"] = evaluate_condition
        
    async def _setup_optimization_strategies(self):
        """Setup workflow optimization strategies."""
        
        async def optimize_for_speed(workflow: WorkflowDefinition) -> WorkflowDefinition:
            """Optimize workflow for speed."""
            # Identify tasks that can run in parallel
            for task in workflow.tasks:
                if not task.dependencies and task.execution_mode == ExecutionMode.SEQUENTIAL:
                    task.execution_mode = ExecutionMode.PARALLEL
                    
            return workflow
            
        async def optimize_for_quality(workflow: WorkflowDefinition) -> WorkflowDefinition:
            """Optimize workflow for quality."""
            # Add additional validation steps
            validation_tasks = []
            for task in workflow.tasks:
                if task.agent_type == "coding_agent":
                    validation_task = WorkflowTask(
                        id=f"{task.id}_validation",
                        name=f"{task.name} Validation",
                        description=f"Validate {task.name}",
                        agent_type="validation_agent",
                        task_data={"validate_task": task.id},
                        dependencies=[task.id],
                        execution_mode=ExecutionMode.SEQUENTIAL
                    )
                    validation_tasks.append(validation_task)
                    
            workflow.tasks.extend(validation_tasks)
            return workflow
            
        self.optimization_strategies["speed"] = optimize_for_speed
        self.optimization_strategies["quality"] = optimize_for_quality
        
    async def _select_best_template(self, requirements: ProjectRequirements) -> str:
        """Select the best template based on requirements."""
        
        # Analyze complexity
        complexity = await self.complexity_analyzers["default"](requirements)
        
        # Match project type to template
        if requirements.project_type in ["feature", "enhancement", "development"]:
            return "feature_development"
        elif requirements.project_type in ["research", "analysis", "investigation"]:
            return "research_workflow"
        elif requirements.project_type in ["bug", "fix", "hotfix"]:
            return "bug_fix_workflow"
        else:
            # Default to feature development
            return "feature_development"
            
    async def _generate_base_workflow(
        self,
        template: WorkflowTemplate,
        requirements: ProjectRequirements
    ) -> WorkflowDefinition:
        """Generate base workflow from template."""
        
        workflow_id = f"{template.template_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert template tasks to workflow tasks
        tasks = []
        for task_data in template.base_tasks:
            task = WorkflowTask(
                id=task_data["id"],
                name=task_data["name"],
                description=task_data.get("description", task_data["name"]),
                agent_type=task_data["agent_type"],
                task_data={
                    "requirements": requirements.__dict__,
                    "priority": task_data.get("priority", "medium"),
                    "estimated_duration": task_data.get("estimated_duration", 2)
                },
                dependencies=task_data.get("dependencies", []),
                execution_mode=ExecutionMode.SEQUENTIAL,
                timeout=task_data.get("estimated_duration", 2) * 3600  # Convert hours to seconds
            )
            tasks.append(task)
            
        return WorkflowDefinition(
            id=workflow_id,
            name=f"{template.name} - {requirements.project_type}",
            description=f"Generated workflow for {requirements.project_type}",
            version="1.0",
            tasks=tasks,
            conditions=[],
            metadata={
                "template_id": template.template_id,
                "generated_at": datetime.now().isoformat(),
                "requirements": requirements.__dict__
            }
        )
        
    async def _apply_conditional_logic(
        self,
        workflow: WorkflowDefinition,
        requirements: ProjectRequirements,
        template: WorkflowTemplate
    ) -> WorkflowDefinition:
        """Apply conditional logic to add/remove tasks."""
        
        context = {
            "complexity": requirements.complexity,
            "quality_requirements": requirements.quality_requirements,
            "priorities": requirements.priorities,
            "team_size": requirements.team_size,
            "timeline": requirements.timeline
        }
        
        # Process decision points
        for decision_point in template.decision_points:
            condition = decision_point["condition"]
            
            if await self.decision_engines["default"](condition, context):
                # Add conditional tasks
                true_path = decision_point["true_path"]
                if true_path in template.conditional_tasks:
                    conditional_tasks = template.conditional_tasks[true_path]
                    
                    for task_data in conditional_tasks:
                        task = WorkflowTask(
                            id=task_data["id"],
                            name=task_data["name"],
                            description=task_data.get("description", task_data["name"]),
                            agent_type=task_data["agent_type"],
                            task_data={
                                "requirements": requirements.__dict__,
                                "conditional": True,
                                "decision_point": decision_point["point_id"]
                            },
                            dependencies=task_data.get("dependencies", []),
                            execution_mode=ExecutionMode.SEQUENTIAL,
                            timeout=task_data.get("estimated_duration", 2) * 3600
                        )
                        workflow.tasks.append(task)
                        
                # Add workflow condition
                workflow_condition = WorkflowCondition(
                    condition_id=decision_point["point_id"],
                    condition_type="custom",
                    condition_expression=condition,
                    true_path=[task["id"] for task in template.conditional_tasks.get(true_path, [])],
                    false_path=[]
                )
                workflow.conditions.append(workflow_condition)
                
        return workflow
        
    async def _optimize_workflow(
        self,
        workflow: WorkflowDefinition,
        requirements: ProjectRequirements
    ) -> WorkflowDefinition:
        """Optimize workflow based on requirements."""
        
        # Determine optimization strategy
        if requirements.timeline < timedelta(days=7):
            # Optimize for speed
            workflow = await self.optimization_strategies["speed"](workflow)
        elif requirements.quality_requirements in ["high", "critical"]:
            # Optimize for quality
            workflow = await self.optimization_strategies["quality"](workflow)
            
        return workflow
        
    async def _validate_workflow(self, workflow: WorkflowDefinition):
        """Validate the generated workflow."""
        
        # Check for circular dependencies
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = next((t for t in workflow.tasks if t.id == task_id), None)
            if task:
                for dep in task.dependencies:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True
                        
            rec_stack.remove(task_id)
            return False
            
        for task in workflow.tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    raise ValueError(f"Circular dependency detected in workflow {workflow.id}")
                    
        # Check for orphaned tasks
        all_dependencies = set()
        for task in workflow.tasks:
            all_dependencies.update(task.dependencies)
            
        task_ids = {task.id for task in workflow.tasks}
        orphaned_deps = all_dependencies - task_ids
        
        if orphaned_deps:
            raise ValueError(f"Orphaned dependencies found: {orphaned_deps}")
            
    async def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available workflow templates."""
        return [
            {
                "id": template.template_id,
                "name": template.name,
                "description": template.description,
                "pattern": template.pattern.value,
                "complexity_range": [c.value for c in template.complexity_range],
                "estimated_duration": str(template.estimated_duration)
            }
            for template in self.templates.values()
        ]
        
    async def analyze_project_requirements(
        self,
        project_description: str,
        additional_context: Dict[str, Any] = None
    ) -> ProjectRequirements:
        """Analyze project description to extract requirements."""
        
        # Simple keyword-based analysis
        # In production, use NLP/LLM for better analysis
        
        description_lower = project_description.lower()
        
        # Determine project type
        if any(word in description_lower for word in ["bug", "fix", "error", "issue"]):
            project_type = "bug"
        elif any(word in description_lower for word in ["research", "analyze", "investigate"]):
            project_type = "research"
        else:
            project_type = "feature"
            
        # Determine complexity
        complexity_indicators = {
            "simple": ["simple", "basic", "quick", "small"],
            "moderate": ["moderate", "medium", "standard"],
            "complex": ["complex", "advanced", "sophisticated", "large"],
            "enterprise": ["enterprise", "critical", "mission-critical", "production"]
        }
        
        complexity = ProjectComplexity.MODERATE  # Default
        for level, indicators in complexity_indicators.items():
            if any(indicator in description_lower for indicator in indicators):
                complexity = ProjectComplexity(level)
                break
                
        # Extract technologies
        tech_keywords = {
            "python": ["python", "fastapi", "django", "flask"],
            "javascript": ["javascript", "js", "node", "react", "vue", "angular"],
            "database": ["database", "sql", "postgres", "mysql", "mongodb"],
            "ai": ["ai", "ml", "machine learning", "llm", "gpt", "groq"]
        }
        
        technologies = []
        for tech, keywords in tech_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                technologies.append(tech)
                
        # Default values with context override
        context = additional_context or {}
        
        return ProjectRequirements(
            project_type=project_type,
            complexity=complexity,
            timeline=timedelta(days=context.get("timeline_days", 14)),
            team_size=context.get("team_size", 3),
            technologies=technologies,
            constraints=context.get("constraints", {}),
            priorities=context.get("priorities", []),
            risk_tolerance=context.get("risk_tolerance", "medium"),
            quality_requirements=context.get("quality_requirements", "standard")
        )