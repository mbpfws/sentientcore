"""Workflow Template System

This module provides a comprehensive template system for workflow generation,
including template management, validation, and customization capabilities.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import yaml
from datetime import datetime, timedelta

from .advanced_engine import WorkflowDefinition, WorkflowTask, WorkflowCondition, ExecutionMode
from .dynamic_generator import WorkflowTemplate, ProjectComplexity, WorkflowPattern


class TemplateFormat(Enum):
    """Supported template formats."""
    JSON = "json"
    YAML = "yaml"
    PYTHON = "python"


class ValidationLevel(Enum):
    """Template validation levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass
class TemplateMetadata:
    """Metadata for workflow templates."""
    author: str
    version: str
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    license: str = "MIT"
    documentation_url: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    compatibility: Dict[str, str] = field(default_factory=dict)


@dataclass
class TemplateValidationResult:
    """Result of template validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class WorkflowTemplateManager:
    """Manages workflow templates including loading, saving, and validation."""
    
    def __init__(self, template_directory: Optional[Path] = None):
        self.template_directory = template_directory or Path("templates")
        self.templates: Dict[str, WorkflowTemplate] = {}
        self.template_metadata: Dict[str, TemplateMetadata] = {}
        self.validation_rules: Dict[str, callable] = {}
        
    async def initialize(self):
        """Initialize the template manager."""
        await self._setup_validation_rules()
        await self._ensure_template_directory()
        await self._load_all_templates()
        
    async def load_template(
        self,
        template_id: str,
        file_path: Optional[Path] = None
    ) -> WorkflowTemplate:
        """Load a single template from file or registry."""
        
        if template_id in self.templates:
            return self.templates[template_id]
            
        if file_path:
            template = await self._load_template_from_file(file_path)
        else:
            # Try to find template file
            template_file = self._find_template_file(template_id)
            if not template_file:
                raise FileNotFoundError(f"Template {template_id} not found")
            template = await self._load_template_from_file(template_file)
            
        # Validate template
        validation_result = await self.validate_template(template)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid template: {validation_result.errors}")
            
        self.templates[template.template_id] = template
        return template
        
    async def save_template(
        self,
        template: WorkflowTemplate,
        metadata: Optional[TemplateMetadata] = None,
        format_type: TemplateFormat = TemplateFormat.YAML
    ) -> Path:
        """Save a template to file."""
        
        # Validate template before saving
        validation_result = await self.validate_template(template)
        if not validation_result.is_valid:
            raise ValueError(f"Cannot save invalid template: {validation_result.errors}")
            
        # Prepare metadata
        if not metadata:
            metadata = TemplateMetadata(
                author="system",
                version="1.0.0",
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
        # Determine file path
        file_extension = format_type.value
        file_path = self.template_directory / f"{template.template_id}.{file_extension}"
        
        # Save template
        await self._save_template_to_file(template, metadata, file_path, format_type)
        
        # Update registry
        self.templates[template.template_id] = template
        self.template_metadata[template.template_id] = metadata
        
        return file_path
        
    async def validate_template(
        self,
        template: WorkflowTemplate,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> TemplateValidationResult:
        """Validate a workflow template."""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Basic validation
        if not template.template_id:
            errors.append("Template ID is required")
        if not template.name:
            errors.append("Template name is required")
        if not template.base_tasks:
            errors.append("Template must have at least one base task")
            
        # Validate tasks
        task_ids = set()
        for task_data in template.base_tasks:
            if "id" not in task_data:
                errors.append("Task missing required 'id' field")
                continue
                
            task_id = task_data["id"]
            if task_id in task_ids:
                errors.append(f"Duplicate task ID: {task_id}")
            task_ids.add(task_id)
            
            if "name" not in task_data:
                errors.append(f"Task {task_id} missing required 'name' field")
            if "agent_type" not in task_data:
                errors.append(f"Task {task_id} missing required 'agent_type' field")
                
            # Validate dependencies
            dependencies = task_data.get("dependencies", [])
            for dep in dependencies:
                if dep not in task_ids and dep not in [t["id"] for t in template.base_tasks]:
                    # Check if dependency is in conditional tasks
                    found_in_conditional = False
                    for conditional_tasks in template.conditional_tasks.values():
                        if dep in [t["id"] for t in conditional_tasks]:
                            found_in_conditional = True
                            break
                    if not found_in_conditional:
                        warnings.append(f"Task {task_id} depends on unknown task: {dep}")
                        
        # Validate conditional tasks
        for condition_name, conditional_tasks in template.conditional_tasks.items():
            conditional_task_ids = set()
            for task_data in conditional_tasks:
                if "id" not in task_data:
                    errors.append(f"Conditional task in {condition_name} missing 'id' field")
                    continue
                    
                task_id = task_data["id"]
                if task_id in conditional_task_ids:
                    errors.append(f"Duplicate conditional task ID in {condition_name}: {task_id}")
                conditional_task_ids.add(task_id)
                
        # Validate decision points
        for decision_point in template.decision_points:
            if "point_id" not in decision_point:
                errors.append("Decision point missing 'point_id' field")
            if "condition" not in decision_point:
                errors.append("Decision point missing 'condition' field")
            if "true_path" not in decision_point:
                errors.append("Decision point missing 'true_path' field")
                
            true_path = decision_point.get("true_path")
            if true_path and true_path not in template.conditional_tasks and true_path != "continue":
                warnings.append(f"Decision point references unknown conditional path: {true_path}")
                
        # Standard validation
        if validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            # Check for circular dependencies
            if await self._has_circular_dependencies(template):
                errors.append("Template contains circular dependencies")
                
            # Check for unreachable tasks
            unreachable = await self._find_unreachable_tasks(template)
            if unreachable:
                warnings.extend([f"Unreachable task: {task_id}" for task_id in unreachable])
                
        # Strict validation
        if validation_level == ValidationLevel.STRICT:
            # Check for best practices
            if len(template.base_tasks) > 20:
                suggestions.append("Consider breaking down large workflows into smaller templates")
                
            # Check for missing descriptions
            for task_data in template.base_tasks:
                if "description" not in task_data:
                    suggestions.append(f"Task {task_data.get('id')} missing description")
                    
            # Check for missing estimated durations
            for task_data in template.base_tasks:
                if "estimated_duration" not in task_data:
                    suggestions.append(f"Task {task_data.get('id')} missing estimated_duration")
                    
        return TemplateValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
        
    async def create_template_from_workflow(
        self,
        workflow: WorkflowDefinition,
        template_id: str,
        name: str,
        description: str
    ) -> WorkflowTemplate:
        """Create a template from an existing workflow."""
        
        # Convert workflow tasks to template format
        base_tasks = []
        for task in workflow.tasks:
            task_data = {
                "id": task.id,
                "name": task.name,
                "description": task.description,
                "agent_type": task.agent_type,
                "dependencies": task.dependencies,
                "estimated_duration": task.timeout // 3600 if task.timeout else 2  # Convert seconds to hours
            }
            
            # Extract priority from task_data if available
            if hasattr(task, 'task_data') and task.task_data:
                if "priority" in task.task_data:
                    task_data["priority"] = task.task_data["priority"]
                    
            base_tasks.append(task_data)
            
        # Convert workflow conditions to decision points
        decision_points = []
        for condition in workflow.conditions:
            decision_point = {
                "point_id": condition.condition_id,
                "condition": condition.condition_expression,
                "true_path": "continue" if not condition.true_path else condition.true_path[0],
                "false_path": "continue" if not condition.false_path else condition.false_path[0]
            }
            decision_points.append(decision_point)
            
        template = WorkflowTemplate(
            template_id=template_id,
            name=name,
            description=description,
            pattern=WorkflowPattern.HYBRID,  # Default pattern
            complexity_range=[ProjectComplexity.SIMPLE, ProjectComplexity.MODERATE],
            base_tasks=base_tasks,
            conditional_tasks={},
            decision_points=decision_points,
            resource_requirements={},
            estimated_duration=timedelta(hours=sum(task.get("estimated_duration", 2) for task in base_tasks))
        )
        
        return template
        
    async def customize_template(
        self,
        template_id: str,
        customizations: Dict[str, Any]
    ) -> WorkflowTemplate:
        """Customize an existing template."""
        
        if template_id not in self.templates:
            raise ValueError(f"Template {template_id} not found")
            
        template = self.templates[template_id]
        
        # Create a copy for customization
        customized_template = WorkflowTemplate(
            template_id=f"{template.template_id}_custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=customizations.get("name", f"{template.name} (Customized)"),
            description=customizations.get("description", template.description),
            pattern=template.pattern,
            complexity_range=template.complexity_range.copy(),
            base_tasks=template.base_tasks.copy(),
            conditional_tasks=template.conditional_tasks.copy(),
            decision_points=template.decision_points.copy(),
            resource_requirements=template.resource_requirements.copy(),
            estimated_duration=template.estimated_duration
        )
        
        # Apply customizations
        if "add_tasks" in customizations:
            customized_template.base_tasks.extend(customizations["add_tasks"])
            
        if "remove_tasks" in customizations:
            task_ids_to_remove = set(customizations["remove_tasks"])
            customized_template.base_tasks = [
                task for task in customized_template.base_tasks
                if task["id"] not in task_ids_to_remove
            ]
            
        if "modify_tasks" in customizations:
            for task_id, modifications in customizations["modify_tasks"].items():
                for task in customized_template.base_tasks:
                    if task["id"] == task_id:
                        task.update(modifications)
                        break
                        
        if "add_conditional_tasks" in customizations:
            for condition_name, tasks in customizations["add_conditional_tasks"].items():
                if condition_name in customized_template.conditional_tasks:
                    customized_template.conditional_tasks[condition_name].extend(tasks)
                else:
                    customized_template.conditional_tasks[condition_name] = tasks
                    
        if "add_decision_points" in customizations:
            customized_template.decision_points.extend(customizations["add_decision_points"])
            
        # Validate customized template
        validation_result = await self.validate_template(customized_template)
        if not validation_result.is_valid:
            raise ValueError(f"Customized template is invalid: {validation_result.errors}")
            
        return customized_template
        
    async def get_template_library(self) -> Dict[str, Dict[str, Any]]:
        """Get the complete template library with metadata."""
        
        library = {}
        for template_id, template in self.templates.items():
            metadata = self.template_metadata.get(template_id)
            
            library[template_id] = {
                "template": {
                    "id": template.template_id,
                    "name": template.name,
                    "description": template.description,
                    "pattern": template.pattern.value,
                    "complexity_range": [c.value for c in template.complexity_range],
                    "task_count": len(template.base_tasks),
                    "estimated_duration": str(template.estimated_duration)
                },
                "metadata": asdict(metadata) if metadata else None
            }
            
        return library
        
    async def search_templates(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[WorkflowTemplate]:
        """Search templates by query and filters."""
        
        results = []
        query_lower = query.lower()
        
        for template in self.templates.values():
            # Text search
            if (query_lower in template.name.lower() or
                query_lower in template.description.lower() or
                query_lower in template.template_id.lower()):
                
                # Apply filters
                if filters:
                    if "pattern" in filters and template.pattern.value != filters["pattern"]:
                        continue
                    if "complexity" in filters and ProjectComplexity(filters["complexity"]) not in template.complexity_range:
                        continue
                    if "max_tasks" in filters and len(template.base_tasks) > filters["max_tasks"]:
                        continue
                        
                results.append(template)
                
        return results
        
    async def _setup_validation_rules(self):
        """Setup validation rules for templates."""
        
        async def validate_task_structure(task_data: Dict[str, Any]) -> List[str]:
            """Validate individual task structure."""
            errors = []
            required_fields = ["id", "name", "agent_type"]
            
            for field in required_fields:
                if field not in task_data:
                    errors.append(f"Task missing required field: {field}")
                    
            return errors
            
        self.validation_rules["task_structure"] = validate_task_structure
        
    async def _ensure_template_directory(self):
        """Ensure template directory exists."""
        self.template_directory.mkdir(parents=True, exist_ok=True)
        
    async def _load_all_templates(self):
        """Load all templates from the template directory."""
        
        if not self.template_directory.exists():
            return
            
        for file_path in self.template_directory.glob("*.yaml"):
            try:
                template = await self._load_template_from_file(file_path)
                self.templates[template.template_id] = template
            except Exception as e:
                print(f"Failed to load template from {file_path}: {e}")
                
        for file_path in self.template_directory.glob("*.json"):
            try:
                template = await self._load_template_from_file(file_path)
                self.templates[template.template_id] = template
            except Exception as e:
                print(f"Failed to load template from {file_path}: {e}")
                
    async def _load_template_from_file(self, file_path: Path) -> WorkflowTemplate:
        """Load template from a specific file."""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
                data = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        # Convert data to WorkflowTemplate
        template = WorkflowTemplate(
            template_id=data["template_id"],
            name=data["name"],
            description=data["description"],
            pattern=WorkflowPattern(data.get("pattern", "hybrid")),
            complexity_range=[ProjectComplexity(c) for c in data.get("complexity_range", ["simple", "moderate"])],
            base_tasks=data["base_tasks"],
            conditional_tasks=data.get("conditional_tasks", {}),
            decision_points=data.get("decision_points", []),
            resource_requirements=data.get("resource_requirements", {}),
            estimated_duration=timedelta(hours=data.get("estimated_duration_hours", 8))
        )
        
        # Load metadata if available
        if "metadata" in data:
            metadata_data = data["metadata"]
            metadata = TemplateMetadata(
                author=metadata_data.get("author", "unknown"),
                version=metadata_data.get("version", "1.0.0"),
                created_at=datetime.fromisoformat(metadata_data.get("created_at", datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(metadata_data.get("updated_at", datetime.now().isoformat())),
                tags=metadata_data.get("tags", []),
                category=metadata_data.get("category", "general"),
                license=metadata_data.get("license", "MIT"),
                documentation_url=metadata_data.get("documentation_url"),
                dependencies=metadata_data.get("dependencies", []),
                compatibility=metadata_data.get("compatibility", {})
            )
            self.template_metadata[template.template_id] = metadata
            
        return template
        
    async def _save_template_to_file(
        self,
        template: WorkflowTemplate,
        metadata: TemplateMetadata,
        file_path: Path,
        format_type: TemplateFormat
    ):
        """Save template to file."""
        
        data = {
            "template_id": template.template_id,
            "name": template.name,
            "description": template.description,
            "pattern": template.pattern.value,
            "complexity_range": [c.value for c in template.complexity_range],
            "base_tasks": template.base_tasks,
            "conditional_tasks": template.conditional_tasks,
            "decision_points": template.decision_points,
            "resource_requirements": template.resource_requirements,
            "estimated_duration_hours": template.estimated_duration.total_seconds() / 3600,
            "metadata": {
                "author": metadata.author,
                "version": metadata.version,
                "created_at": metadata.created_at.isoformat(),
                "updated_at": metadata.updated_at.isoformat(),
                "tags": metadata.tags,
                "category": metadata.category,
                "license": metadata.license,
                "documentation_url": metadata.documentation_url,
                "dependencies": metadata.dependencies,
                "compatibility": metadata.compatibility
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format_type == TemplateFormat.YAML:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format_type == TemplateFormat.JSON:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
    def _find_template_file(self, template_id: str) -> Optional[Path]:
        """Find template file by ID."""
        
        for extension in ["yaml", "yml", "json"]:
            file_path = self.template_directory / f"{template_id}.{extension}"
            if file_path.exists():
                return file_path
                
        return None
        
    async def _has_circular_dependencies(self, template: WorkflowTemplate) -> bool:
        """Check if template has circular dependencies."""
        
        # Build dependency graph
        graph = {}
        for task_data in template.base_tasks:
            task_id = task_data["id"]
            dependencies = task_data.get("dependencies", [])
            graph[task_id] = dependencies
            
        # Add conditional tasks
        for conditional_tasks in template.conditional_tasks.values():
            for task_data in conditional_tasks:
                task_id = task_data["id"]
                dependencies = task_data.get("dependencies", [])
                graph[task_id] = dependencies
                
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node)
            return False
            
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
                    
        return False
        
    async def _find_unreachable_tasks(self, template: WorkflowTemplate) -> List[str]:
        """Find tasks that are unreachable from root tasks."""
        
        # Find root tasks (tasks with no dependencies)
        all_tasks = {task["id"] for task in template.base_tasks}
        for conditional_tasks in template.conditional_tasks.values():
            all_tasks.update(task["id"] for task in conditional_tasks)
            
        root_tasks = set()
        for task_data in template.base_tasks:
            if not task_data.get("dependencies"):
                root_tasks.add(task_data["id"])
                
        # BFS to find reachable tasks
        reachable = set(root_tasks)
        queue = list(root_tasks)
        
        while queue:
            current = queue.pop(0)
            
            # Find tasks that depend on current task
            for task_data in template.base_tasks:
                task_id = task_data["id"]
                if current in task_data.get("dependencies", []) and task_id not in reachable:
                    reachable.add(task_id)
                    queue.append(task_id)
                    
            # Check conditional tasks
            for conditional_tasks in template.conditional_tasks.values():
                for task_data in conditional_tasks:
                    task_id = task_data["id"]
                    if current in task_data.get("dependencies", []) and task_id not in reachable:
                        reachable.add(task_id)
                        queue.append(task_id)
                        
        return list(all_tasks - reachable)