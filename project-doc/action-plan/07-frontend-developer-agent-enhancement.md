# Frontend Developer Agent Enhancement

## Overview

This phase enhances the Frontend Developer Agent to provide advanced frontend development capabilities, including modern React patterns, component library integration, responsive design, accessibility, performance optimization, and automated testing. The agent will leverage Next.js 15, React 19, and Shadcn UI to create production-ready frontend applications.

## Implementation Tasks

### 1. Enhanced Frontend Developer Agent

**File**: `core/agents/frontend_developer_agent.py`

```python
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from core.agents.base_agent import BaseAgent
from core.models import (
    FrontendTask, ComponentSpec, DesignSystem, 
    ResponsiveBreakpoint, AccessibilityRequirement,
    PerformanceMetric, TestingStrategy
)
from core.services.llm_service import LLMService
from core.services.memory_service import MemoryService
from core.services.state_service import StateService

class FrontendDeveloperAgent(BaseAgent):
    """Enhanced Frontend Developer Agent for modern React development."""
    
    def __init__(self):
        super().__init__()
        self.agent_type = "frontend_developer"
        self.capabilities = [
            "component_development",
            "responsive_design",
            "accessibility_implementation",
            "performance_optimization",
            "testing_automation",
            "design_system_integration",
            "state_management",
            "routing_implementation",
            "api_integration",
            "build_optimization"
        ]
        self.supported_frameworks = [
            "react", "nextjs", "typescript", "tailwindcss", "shadcn"
        ]
        self.llm_service: Optional[LLMService] = None
        self.memory_service: Optional[MemoryService] = None
        self.state_service: Optional[StateService] = None
    
    async def initialize(self, llm_service: LLMService, memory_service: MemoryService, state_service: StateService):
        """Initialize the Frontend Developer Agent with required services."""
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.state_service = state_service
        await self.update_status("initialized")
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if the agent can handle the given task."""
        task_type = task.get('type', '').lower()
        frontend_tasks = [
            'component_development', 'ui_implementation', 'frontend_development',
            'responsive_design', 'accessibility_audit', 'performance_optimization',
            'testing_implementation', 'design_system_setup'
        ]
        return task_type in frontend_tasks
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a frontend development task."""
        await self.update_status("processing")
        
        try:
            task_type = task.get('type')
            
            if task_type == 'component_development':
                result = await self._develop_component(task)
            elif task_type == 'ui_implementation':
                result = await self._implement_ui(task)
            elif task_type == 'responsive_design':
                result = await self._implement_responsive_design(task)
            elif task_type == 'accessibility_audit':
                result = await self._audit_accessibility(task)
            elif task_type == 'performance_optimization':
                result = await self._optimize_performance(task)
            elif task_type == 'testing_implementation':
                result = await self._implement_testing(task)
            elif task_type == 'design_system_setup':
                result = await self._setup_design_system(task)
            else:
                result = await self._handle_generic_frontend_task(task)
            
            # Store results in memory
            await self.memory_service.store_knowledge(
                f"frontend_task_{datetime.now().isoformat()}",
                {
                    "task": task,
                    "result": result,
                    "agent": self.agent_type,
                    "timestamp": datetime.now().isoformat()
                },
                "frontend_development"
            )
            
            await self.update_status("completed")
            return result
            
        except Exception as e:
            await self.update_status("error", str(e))
            raise
    
    async def _develop_component(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Develop a React component based on specifications."""
        component_spec = ComponentSpec(**task.get('component_spec', {}))
        
        # Analyze component requirements
        requirements_analysis = await self._analyze_component_requirements(component_spec)
        
        # Generate component architecture
        component_architecture = await self._design_component_architecture(
            component_spec, requirements_analysis
        )
        
        # Generate component code
        component_code = await self._generate_component_code(
            component_spec, component_architecture
        )
        
        # Generate component tests
        component_tests = await self._generate_component_tests(
            component_spec, component_code
        )
        
        # Generate component documentation
        component_docs = await self._generate_component_documentation(
            component_spec, component_code
        )
        
        # Generate Storybook stories
        storybook_stories = await self._generate_storybook_stories(
            component_spec, component_code
        )
        
        return {
            "session_id": f"component_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "component_name": component_spec.name,
            "requirements_analysis": requirements_analysis,
            "component_architecture": component_architecture,
            "component_code": component_code,
            "component_tests": component_tests,
            "component_docs": component_docs,
            "storybook_stories": storybook_stories,
            "file_structure": self._generate_component_file_structure(component_spec),
            "implementation_notes": self._generate_implementation_notes(component_spec),
            "confidence_score": 0.9
        }
    
    async def _implement_ui(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement UI based on design specifications."""
        ui_spec = task.get('ui_spec', {})
        design_tokens = task.get('design_tokens', {})
        
        # Analyze UI requirements
        ui_analysis = await self._analyze_ui_requirements(ui_spec, design_tokens)
        
        # Generate page/layout structure
        page_structure = await self._generate_page_structure(ui_spec, ui_analysis)
        
        # Generate responsive layouts
        responsive_layouts = await self._generate_responsive_layouts(ui_spec)
        
        # Generate component implementations
        component_implementations = await self._generate_ui_components(ui_spec, design_tokens)
        
        # Generate styling solutions
        styling_solutions = await self._generate_styling_solutions(ui_spec, design_tokens)
        
        # Generate accessibility implementations
        accessibility_implementations = await self._generate_accessibility_features(ui_spec)
        
        return {
            "session_id": f"ui_impl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ui_analysis": ui_analysis,
            "page_structure": page_structure,
            "responsive_layouts": responsive_layouts,
            "component_implementations": component_implementations,
            "styling_solutions": styling_solutions,
            "accessibility_implementations": accessibility_implementations,
            "implementation_roadmap": self._generate_ui_implementation_roadmap(ui_spec),
            "confidence_score": 0.85
        }
    
    async def _implement_responsive_design(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement responsive design patterns."""
        breakpoints = task.get('breakpoints', [])
        design_requirements = task.get('design_requirements', {})
        
        # Analyze responsive requirements
        responsive_analysis = await self._analyze_responsive_requirements(
            breakpoints, design_requirements
        )
        
        # Generate breakpoint strategies
        breakpoint_strategies = await self._generate_breakpoint_strategies(responsive_analysis)
        
        # Generate responsive components
        responsive_components = await self._generate_responsive_components(
            breakpoints, design_requirements
        )
        
        # Generate CSS/Tailwind solutions
        styling_solutions = await self._generate_responsive_styling(
            breakpoints, responsive_analysis
        )
        
        # Generate testing strategies
        responsive_testing = await self._generate_responsive_testing_strategies(
            breakpoints
        )
        
        return {
            "session_id": f"responsive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "responsive_analysis": responsive_analysis,
            "breakpoint_strategies": breakpoint_strategies,
            "responsive_components": responsive_components,
            "styling_solutions": styling_solutions,
            "responsive_testing": responsive_testing,
            "implementation_guide": self._generate_responsive_implementation_guide(),
            "confidence_score": 0.88
        }
    
    async def _audit_accessibility(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Audit and implement accessibility features."""
        target_level = task.get('accessibility_level', 'AA')
        components_to_audit = task.get('components', [])
        
        # Analyze accessibility requirements
        accessibility_analysis = await self._analyze_accessibility_requirements(
            target_level, components_to_audit
        )
        
        # Generate accessibility implementations
        accessibility_implementations = await self._generate_accessibility_implementations(
            accessibility_analysis
        )
        
        # Generate ARIA patterns
        aria_patterns = await self._generate_aria_patterns(components_to_audit)
        
        # Generate keyboard navigation
        keyboard_navigation = await self._generate_keyboard_navigation_patterns(
            components_to_audit
        )
        
        # Generate accessibility testing
        accessibility_testing = await self._generate_accessibility_testing(
            target_level, components_to_audit
        )
        
        return {
            "session_id": f"a11y_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "accessibility_level": target_level,
            "accessibility_analysis": accessibility_analysis,
            "accessibility_implementations": accessibility_implementations,
            "aria_patterns": aria_patterns,
            "keyboard_navigation": keyboard_navigation,
            "accessibility_testing": accessibility_testing,
            "compliance_checklist": self._generate_accessibility_checklist(target_level),
            "confidence_score": 0.92
        }
    
    async def _optimize_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize frontend performance."""
        performance_targets = task.get('performance_targets', {})
        current_metrics = task.get('current_metrics', {})
        
        # Analyze performance requirements
        performance_analysis = await self._analyze_performance_requirements(
            performance_targets, current_metrics
        )
        
        # Generate optimization strategies
        optimization_strategies = await self._generate_optimization_strategies(
            performance_analysis
        )
        
        # Generate code splitting strategies
        code_splitting = await self._generate_code_splitting_strategies(
            performance_analysis
        )
        
        # Generate caching strategies
        caching_strategies = await self._generate_caching_strategies(
            performance_analysis
        )
        
        # Generate performance monitoring
        performance_monitoring = await self._generate_performance_monitoring(
            performance_targets
        )
        
        return {
            "session_id": f"perf_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "performance_analysis": performance_analysis,
            "optimization_strategies": optimization_strategies,
            "code_splitting": code_splitting,
            "caching_strategies": caching_strategies,
            "performance_monitoring": performance_monitoring,
            "implementation_roadmap": self._generate_performance_roadmap(),
            "confidence_score": 0.87
        }
    
    async def _implement_testing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Implement comprehensive frontend testing."""
        testing_strategy = TestingStrategy(**task.get('testing_strategy', {}))
        components_to_test = task.get('components', [])
        
        # Analyze testing requirements
        testing_analysis = await self._analyze_testing_requirements(
            testing_strategy, components_to_test
        )
        
        # Generate unit tests
        unit_tests = await self._generate_unit_tests(
            components_to_test, testing_strategy
        )
        
        # Generate integration tests
        integration_tests = await self._generate_integration_tests(
            components_to_test, testing_strategy
        )
        
        # Generate E2E tests
        e2e_tests = await self._generate_e2e_tests(
            testing_strategy
        )
        
        # Generate visual regression tests
        visual_tests = await self._generate_visual_regression_tests(
            components_to_test
        )
        
        # Generate testing utilities
        testing_utilities = await self._generate_testing_utilities(
            testing_strategy
        )
        
        return {
            "session_id": f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "testing_analysis": testing_analysis,
            "unit_tests": unit_tests,
            "integration_tests": integration_tests,
            "e2e_tests": e2e_tests,
            "visual_tests": visual_tests,
            "testing_utilities": testing_utilities,
            "testing_configuration": self._generate_testing_configuration(),
            "confidence_score": 0.91
        }
    
    async def _setup_design_system(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Setup and configure design system."""
        design_system = DesignSystem(**task.get('design_system', {}))
        
        # Analyze design system requirements
        design_analysis = await self._analyze_design_system_requirements(design_system)
        
        # Generate design tokens
        design_tokens = await self._generate_design_tokens(design_system)
        
        # Generate component library structure
        component_library = await self._generate_component_library_structure(
            design_system
        )
        
        # Generate theme configuration
        theme_configuration = await self._generate_theme_configuration(
            design_system, design_tokens
        )
        
        # Generate documentation structure
        documentation_structure = await self._generate_design_system_documentation(
            design_system
        )
        
        return {
            "session_id": f"design_sys_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "design_analysis": design_analysis,
            "design_tokens": design_tokens,
            "component_library": component_library,
            "theme_configuration": theme_configuration,
            "documentation_structure": documentation_structure,
            "implementation_guide": self._generate_design_system_guide(),
            "confidence_score": 0.89
        }
    
    async def _handle_generic_frontend_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic frontend development tasks."""
        task_description = task.get('description', '')
        requirements = task.get('requirements', [])
        
        prompt = f"""
        Analyze this frontend development task and provide a comprehensive solution:
        
        Task Description: {task_description}
        Requirements: {', '.join(requirements)}
        
        Please provide:
        1. Task analysis and approach
        2. Implementation strategy
        3. Code examples and patterns
        4. Testing recommendations
        5. Performance considerations
        6. Accessibility considerations
        
        Focus on modern React patterns, TypeScript, and best practices.
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "analysis": response,
                "implementation_notes": "Detailed analysis provided"
            }
        
        return {
            "session_id": f"frontend_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "task_analysis": result,
            "confidence_score": 0.75
        }
    
    # Helper methods for component analysis and generation
    async def _analyze_component_requirements(self, component_spec: ComponentSpec) -> Dict[str, Any]:
        """Analyze component requirements and constraints."""
        prompt = f"""
        Analyze the requirements for this React component:
        
        Component Name: {component_spec.name}
        Description: {component_spec.description}
        Props: {component_spec.props}
        Functionality: {component_spec.functionality}
        
        Provide analysis including:
        1. Component complexity assessment
        2. Required React patterns (hooks, context, etc.)
        3. State management needs
        4. Performance considerations
        5. Accessibility requirements
        6. Testing strategy
        
        Return as JSON.
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            max_tokens=1500,
            temperature=0.6
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"analysis": response}
    
    async def _design_component_architecture(self, component_spec: ComponentSpec, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Design component architecture and structure."""
        prompt = f"""
        Design the architecture for this React component:
        
        Component: {component_spec.name}
        Requirements Analysis: {json.dumps(requirements, indent=2)}
        
        Provide architecture including:
        1. Component structure and hierarchy
        2. Hook usage strategy
        3. State management approach
        4. Event handling patterns
        5. Performance optimization strategies
        6. Reusability considerations
        
        Return as JSON with detailed architecture plan.
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            max_tokens=1500,
            temperature=0.6
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"architecture": response}
    
    async def _generate_component_code(self, component_spec: ComponentSpec, architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete component code."""
        prompt = f"""
        Generate complete React component code with TypeScript:
        
        Component: {component_spec.name}
        Architecture: {json.dumps(architecture, indent=2)}
        
        Generate:
        1. Main component file with TypeScript interfaces
        2. Custom hooks if needed
        3. Utility functions
        4. Styled components or Tailwind classes
        5. Export statements
        
        Use modern React patterns, TypeScript, and Tailwind CSS.
        Ensure code is production-ready and follows best practices.
        
        Return as JSON with separate code files.
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            max_tokens=2500,
            temperature=0.5
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"code": response}
    
    def _generate_component_file_structure(self, component_spec: ComponentSpec) -> Dict[str, Any]:
        """Generate recommended file structure for component."""
        component_name = component_spec.name
        
        return {
            "structure": {
                f"components/{component_name}/": {
                    f"{component_name}.tsx": "Main component file",
                    f"{component_name}.test.tsx": "Component tests",
                    f"{component_name}.stories.tsx": "Storybook stories",
                    f"{component_name}.types.ts": "TypeScript interfaces",
                    "hooks/": {
                        f"use{component_name}.ts": "Custom hooks"
                    },
                    "utils/": {
                        f"{component_name.lower()}-utils.ts": "Utility functions"
                    },
                    "index.ts": "Export file"
                }
            },
            "description": f"Recommended file structure for {component_name} component"
        }
    
    def _generate_implementation_notes(self, component_spec: ComponentSpec) -> List[str]:
        """Generate implementation notes and best practices."""
        return [
            "Use TypeScript for type safety",
            "Implement proper error boundaries",
            "Add comprehensive prop validation",
            "Include accessibility attributes",
            "Optimize for performance with React.memo if needed",
            "Follow naming conventions consistently",
            "Add proper documentation and comments",
            "Implement responsive design patterns",
            "Include loading and error states",
            "Test all user interactions"
        ]
```

### 2. Enhanced Pydantic Models

**File**: `core/models.py` (additions)

```python
# Frontend Development Models

class ComponentSpec(BaseModel):
    """Specification for React component development."""
    name: str
    description: str
    props: Dict[str, Any] = {}
    functionality: List[str] = []
    styling_requirements: Dict[str, Any] = {}
    accessibility_requirements: List[str] = []
    performance_requirements: Dict[str, Any] = {}
    testing_requirements: List[str] = []

class DesignSystem(BaseModel):
    """Design system configuration."""
    name: str
    version: str
    color_palette: Dict[str, str] = {}
    typography: Dict[str, Any] = {}
    spacing: Dict[str, str] = {}
    breakpoints: Dict[str, str] = {}
    components: List[str] = []
    tokens: Dict[str, Any] = {}

class ResponsiveBreakpoint(BaseModel):
    """Responsive design breakpoint."""
    name: str
    min_width: Optional[int] = None
    max_width: Optional[int] = None
    css_query: str
    design_considerations: List[str] = []

class AccessibilityRequirement(BaseModel):
    """Accessibility requirement specification."""
    level: str  # A, AA, AAA
    guideline: str
    description: str
    implementation_notes: List[str] = []
    testing_criteria: List[str] = []

class PerformanceMetric(BaseModel):
    """Performance metric specification."""
    metric_name: str
    target_value: float
    current_value: Optional[float] = None
    measurement_method: str
    optimization_strategies: List[str] = []

class TestingStrategy(BaseModel):
    """Frontend testing strategy."""
    unit_testing: bool = True
    integration_testing: bool = True
    e2e_testing: bool = True
    visual_regression: bool = False
    accessibility_testing: bool = True
    performance_testing: bool = False
    testing_frameworks: List[str] = []
    coverage_targets: Dict[str, float] = {}

class FrontendTask(BaseModel):
    """Frontend development task."""
    type: str
    component_spec: Optional[ComponentSpec] = None
    ui_spec: Dict[str, Any] = {}
    design_tokens: Dict[str, Any] = {}
    breakpoints: List[ResponsiveBreakpoint] = []
    accessibility_level: str = "AA"
    performance_targets: Dict[str, Any] = {}
    testing_strategy: Optional[TestingStrategy] = None
    design_system: Optional[DesignSystem] = None
    requirements: List[str] = []
    description: str = ""

class FrontendResult(BaseModel):
    """Frontend development result."""
    session_id: str
    task_type: str
    component_code: Optional[Dict[str, Any]] = None
    ui_implementation: Optional[Dict[str, Any]] = None
    responsive_design: Optional[Dict[str, Any]] = None
    accessibility_implementation: Optional[Dict[str, Any]] = None
    performance_optimization: Optional[Dict[str, Any]] = None
    testing_implementation: Optional[Dict[str, Any]] = None
    design_system_setup: Optional[Dict[str, Any]] = None
    file_structure: Dict[str, Any] = {}
    implementation_notes: List[str] = []
    confidence_score: float
    created_at: datetime = Field(default_factory=datetime.now)
```

### 3. Backend API Endpoints

**File**: `app/api/frontend.py`

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.agents.frontend_developer_agent import FrontendDeveloperAgent
from core.models import FrontendTask, FrontendResult
from core.services.llm_service import LLMService
from core.services.memory_service import MemoryService
from core.services.state_service import StateService

router = APIRouter(prefix="/api/frontend", tags=["frontend"])

# Global services (to be injected)
frontend_agent = FrontendDeveloperAgent()
llm_service = LLMService()
memory_service = MemoryService()
state_service = StateService()

@router.post("/develop-component")
async def develop_component(
    component_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Develop a React component based on specifications."""
    try:
        # Initialize agent if not already done
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        # Create frontend task
        task = {
            "type": "component_development",
            **component_request
        }
        
        # Process task in background
        background_tasks.add_task(frontend_agent.process_task, task)
        
        return {
            "status": "component_development_initiated",
            "session_id": f"component_dev_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "Component development started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/implement-ui")
async def implement_ui(
    ui_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Implement UI based on design specifications."""
    try:
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        task = {
            "type": "ui_implementation",
            **ui_request
        }
        
        background_tasks.add_task(frontend_agent.process_task, task)
        
        return {
            "status": "ui_implementation_initiated",
            "session_id": f"ui_impl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "UI implementation started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/responsive-design")
async def implement_responsive_design(
    responsive_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Implement responsive design patterns."""
    try:
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        task = {
            "type": "responsive_design",
            **responsive_request
        }
        
        background_tasks.add_task(frontend_agent.process_task, task)
        
        return {
            "status": "responsive_design_initiated",
            "session_id": f"responsive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "Responsive design implementation started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/accessibility-audit")
async def audit_accessibility(
    accessibility_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Audit and implement accessibility features."""
    try:
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        task = {
            "type": "accessibility_audit",
            **accessibility_request
        }
        
        background_tasks.add_task(frontend_agent.process_task, task)
        
        return {
            "status": "accessibility_audit_initiated",
            "session_id": f"a11y_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "Accessibility audit started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize-performance")
async def optimize_performance(
    performance_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Optimize frontend performance."""
    try:
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        task = {
            "type": "performance_optimization",
            **performance_request
        }
        
        background_tasks.add_task(frontend_agent.process_task, task)
        
        return {
            "status": "performance_optimization_initiated",
            "session_id": f"perf_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "Performance optimization started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/implement-testing")
async def implement_testing(
    testing_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Implement comprehensive frontend testing."""
    try:
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        task = {
            "type": "testing_implementation",
            **testing_request
        }
        
        background_tasks.add_task(frontend_agent.process_task, task)
        
        return {
            "status": "testing_implementation_initiated",
            "session_id": f"testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "Testing implementation started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/setup-design-system")
async def setup_design_system(
    design_system_request: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Setup and configure design system."""
    try:
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        task = {
            "type": "design_system_setup",
            **design_system_request
        }
        
        background_tasks.add_task(frontend_agent.process_task, task)
        
        return {
            "status": "design_system_setup_initiated",
            "session_id": f"design_sys_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": "Design system setup started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}")
async def get_frontend_session_result(session_id: str) -> Dict[str, Any]:
    """Get frontend development session results."""
    try:
        # Retrieve from memory service
        results = await memory_service.retrieve_knowledge(
            f"frontend_task_{session_id}",
            "frontend_development"
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_frontend_development_history(
    limit: int = 10,
    task_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get frontend development history."""
    try:
        # Retrieve from memory service
        history = await memory_service.retrieve_knowledge(
            "frontend_development_history",
            "frontend_development"
        )
        
        if not history:
            return []
        
        # Filter by task type if specified
        if task_type:
            history = [h for h in history if h.get('task_type') == task_type]
        
        # Return limited results
        return history[:limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_frontend_templates() -> Dict[str, Any]:
    """Get frontend development templates and patterns."""
    return {
        "component_templates": {
            "basic_component": {
                "name": "Basic Component",
                "description": "Simple functional component with props",
                "template": "basic-component-template"
            },
            "form_component": {
                "name": "Form Component",
                "description": "Form component with validation",
                "template": "form-component-template"
            },
            "data_display": {
                "name": "Data Display",
                "description": "Component for displaying data",
                "template": "data-display-template"
            }
        },
        "design_patterns": {
            "responsive_grid": "CSS Grid responsive patterns",
            "mobile_first": "Mobile-first responsive design",
            "accessibility_patterns": "Common accessibility patterns",
            "performance_patterns": "Performance optimization patterns"
        },
        "testing_patterns": {
            "unit_test_template": "Jest/React Testing Library template",
            "integration_test_template": "Integration test template",
            "e2e_test_template": "Playwright E2E test template"
        }
    }

@router.post("/analyze-component")
async def analyze_existing_component(
    component_analysis_request: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze existing component for improvements."""
    try:
        if not frontend_agent.llm_service:
            await frontend_agent.initialize(llm_service, memory_service, state_service)
        
        component_code = component_analysis_request.get('component_code', '')
        analysis_type = component_analysis_request.get('analysis_type', 'general')
        
        prompt = f"""
        Analyze this React component and provide recommendations:
        
        Component Code:
        {component_code}
        
        Analysis Type: {analysis_type}
        
        Provide analysis for:
        1. Code quality and best practices
        2. Performance optimization opportunities
        3. Accessibility improvements
        4. Testing recommendations
        5. Refactoring suggestions
        
        Return as JSON with detailed recommendations.
        """
        
        response = await llm_service.generate_response(
            prompt,
            max_tokens=2000,
            temperature=0.6
        )
        
        try:
            analysis_result = json.loads(response)
        except json.JSONDecodeError:
            analysis_result = {"analysis": response}
        
        return {
            "analysis_result": analysis_result,
            "session_id": f"component_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/best-practices")
async def get_frontend_best_practices() -> Dict[str, Any]:
    """Get frontend development best practices."""
    return {
        "react_best_practices": [
            "Use functional components with hooks",
            "Implement proper error boundaries",
            "Optimize with React.memo for expensive components",
            "Use useCallback and useMemo appropriately",
            "Follow the rules of hooks",
            "Implement proper key props for lists",
            "Use TypeScript for type safety"
        ],
        "performance_best_practices": [
            "Implement code splitting with React.lazy",
            "Optimize bundle size with tree shaking",
            "Use proper image optimization",
            "Implement virtual scrolling for large lists",
            "Minimize re-renders with proper state management",
            "Use service workers for caching",
            "Implement proper loading states"
        ],
        "accessibility_best_practices": [
            "Use semantic HTML elements",
            "Implement proper ARIA attributes",
            "Ensure keyboard navigation support",
            "Provide alternative text for images",
            "Maintain proper color contrast ratios",
            "Implement focus management",
            "Test with screen readers"
        ],
        "testing_best_practices": [
            "Write tests that test behavior, not implementation",
            "Use React Testing Library for component tests",
            "Implement integration tests for user flows",
            "Use visual regression testing for UI consistency",
            "Test accessibility with automated tools",
            "Implement E2E tests for critical paths",
            "Maintain high test coverage"
        ]
    }
```

### 4. Frontend Components

**File**: `frontend/components/frontend-developer-dashboard.tsx`

```typescript
'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

interface FrontendDeveloperDashboardProps {
  onTaskComplete?: (result: any) => void;
}

interface ComponentSpec {
  name: string;
  description: string;
  props: Record<string, any>;
  functionality: string[];
  styling_requirements: Record<string, any>;
  accessibility_requirements: string[];
}

interface FrontendTask {
  type: string;
  component_spec?: ComponentSpec;
  ui_spec?: Record<string, any>;
  breakpoints?: any[];
  accessibility_level?: string;
  performance_targets?: Record<string, any>;
  testing_strategy?: Record<string, any>;
}

interface TaskResult {
  session_id: string;
  task_type: string;
  component_code?: Record<string, any>;
  ui_implementation?: Record<string, any>;
  confidence_score: number;
  created_at: string;
}

export const FrontendDeveloperDashboard: React.FC<FrontendDeveloperDashboardProps> = ({
  onTaskComplete
}) => {
  const [activeTab, setActiveTab] = useState('component-dev');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentTask, setCurrentTask] = useState<FrontendTask | null>(null);
  const [taskResult, setTaskResult] = useState<TaskResult | null>(null);
  const [taskHistory, setTaskHistory] = useState<TaskResult[]>([]);
  
  // Component Development State
  const [componentName, setComponentName] = useState('');
  const [componentDescription, setComponentDescription] = useState('');
  const [componentProps, setComponentProps] = useState<string[]>(['']);
  const [componentFunctionality, setComponentFunctionality] = useState<string[]>(['']);
  
  // UI Implementation State
  const [uiDescription, setUiDescription] = useState('');
  const [designTokens, setDesignTokens] = useState('');
  
  // Responsive Design State
  const [breakpoints, setBreakpoints] = useState<string[]>(['mobile', 'tablet', 'desktop']);
  const [responsiveRequirements, setResponsiveRequirements] = useState('');
  
  // Accessibility State
  const [accessibilityLevel, setAccessibilityLevel] = useState('AA');
  const [componentsToAudit, setComponentsToAudit] = useState<string[]>(['']);
  
  // Performance State
  const [performanceTargets, setPerformanceTargets] = useState('');
  
  // Testing State
  const [testingTypes, setTestingTypes] = useState<string[]>(['unit', 'integration']);
  
  useEffect(() => {
    fetchTaskHistory();
  }, []);
  
  const fetchTaskHistory = async () => {
    try {
      const response = await fetch('/api/frontend/history');
      if (response.ok) {
        const history = await response.json();
        setTaskHistory(history);
      }
    } catch (error) {
      console.error('Failed to fetch task history:', error);
    }
  };
  
  const addArrayItem = (array: string[], setArray: (arr: string[]) => void) => {
    setArray([...array, '']);
  };
  
  const updateArrayItem = (array: string[], setArray: (arr: string[]) => void, index: number, value: string) => {
    const newArray = [...array];
    newArray[index] = value;
    setArray(newArray);
  };
  
  const removeArrayItem = (array: string[], setArray: (arr: string[]) => void, index: number) => {
    if (array.length > 1) {
      const newArray = array.filter((_, i) => i !== index);
      setArray(newArray);
    }
  };
  
  const startComponentDevelopment = async () => {
    if (!componentName.trim() || !componentDescription.trim()) {
      alert('Please provide component name and description');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const componentSpec: ComponentSpec = {
        name: componentName,
        description: componentDescription,
        props: componentProps.reduce((acc, prop, index) => {
          if (prop.trim()) {
            acc[`prop${index + 1}`] = prop;
          }
          return acc;
        }, {} as Record<string, any>),
        functionality: componentFunctionality.filter(f => f.trim()),
        styling_requirements: {},
        accessibility_requirements: []
      };
      
      const task: FrontendTask = {
        type: 'component_development',
        component_spec: componentSpec
      };
      
      setCurrentTask(task);
      
      const response = await fetch('/api/frontend/develop-component', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(task),
      });
      
      if (response.ok) {
        const result = await response.json();
        
        // Poll for results
        setTimeout(() => {
          fetchTaskResult(result.session_id);
        }, 3000);
      } else {
        throw new Error('Failed to start component development');
      }
    } catch (error) {
      console.error('Failed to start component development:', error);
      setIsProcessing(false);
    }
  };
  
  const startUIImplementation = async () => {
    if (!uiDescription.trim()) {
      alert('Please provide UI description');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const task: FrontendTask = {
        type: 'ui_implementation',
        ui_spec: {
          description: uiDescription,
          requirements: uiDescription.split('\n').filter(line => line.trim())
        },
        design_tokens: designTokens ? JSON.parse(designTokens) : {}
      };
      
      setCurrentTask(task);
      
      const response = await fetch('/api/frontend/implement-ui', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(task),
      });
      
      if (response.ok) {
        const result = await response.json();
        setTimeout(() => {
          fetchTaskResult(result.session_id);
        }, 3000);
      } else {
        throw new Error('Failed to start UI implementation');
      }
    } catch (error) {
      console.error('Failed to start UI implementation:', error);
      setIsProcessing(false);
    }
  };
  
  const startResponsiveDesign = async () => {
    if (!responsiveRequirements.trim()) {
      alert('Please provide responsive design requirements');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const task: FrontendTask = {
        type: 'responsive_design',
        breakpoints: breakpoints.map(bp => ({
          name: bp,
          css_query: `@media (min-width: ${bp === 'mobile' ? '320px' : bp === 'tablet' ? '768px' : '1024px'})`
        })),
        ui_spec: {
          responsive_requirements: responsiveRequirements
        }
      };
      
      setCurrentTask(task);
      
      const response = await fetch('/api/frontend/responsive-design', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(task),
      });
      
      if (response.ok) {
        const result = await response.json();
        setTimeout(() => {
          fetchTaskResult(result.session_id);
        }, 3000);
      } else {
        throw new Error('Failed to start responsive design');
      }
    } catch (error) {
      console.error('Failed to start responsive design:', error);
      setIsProcessing(false);
    }
  };
  
  const startAccessibilityAudit = async () => {
    const validComponents = componentsToAudit.filter(c => c.trim());
    if (validComponents.length === 0) {
      alert('Please specify components to audit');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const task: FrontendTask = {
        type: 'accessibility_audit',
        accessibility_level: accessibilityLevel,
        ui_spec: {
          components: validComponents
        }
      };
      
      setCurrentTask(task);
      
      const response = await fetch('/api/frontend/accessibility-audit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(task),
      });
      
      if (response.ok) {
        const result = await response.json();
        setTimeout(() => {
          fetchTaskResult(result.session_id);
        }, 3000);
      } else {
        throw new Error('Failed to start accessibility audit');
      }
    } catch (error) {
      console.error('Failed to start accessibility audit:', error);
      setIsProcessing(false);
    }
  };
  
  const startPerformanceOptimization = async () => {
    if (!performanceTargets.trim()) {
      alert('Please provide performance targets');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const task: FrontendTask = {
        type: 'performance_optimization',
        performance_targets: JSON.parse(performanceTargets)
      };
      
      setCurrentTask(task);
      
      const response = await fetch('/api/frontend/optimize-performance', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(task),
      });
      
      if (response.ok) {
        const result = await response.json();
        setTimeout(() => {
          fetchTaskResult(result.session_id);
        }, 3000);
      } else {
        throw new Error('Failed to start performance optimization');
      }
    } catch (error) {
      console.error('Failed to start performance optimization:', error);
      setIsProcessing(false);
    }
  };
  
  const startTestingImplementation = async () => {
    if (testingTypes.length === 0) {
      alert('Please select testing types');
      return;
    }
    
    setIsProcessing(true);
    
    try {
      const task: FrontendTask = {
        type: 'testing_implementation',
        testing_strategy: {
          unit_testing: testingTypes.includes('unit'),
          integration_testing: testingTypes.includes('integration'),
          e2e_testing: testingTypes.includes('e2e'),
          visual_regression: testingTypes.includes('visual'),
          accessibility_testing: testingTypes.includes('accessibility')
        }
      };
      
      setCurrentTask(task);
      
      const response = await fetch('/api/frontend/implement-testing', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(task),
      });
      
      if (response.ok) {
        const result = await response.json();
        setTimeout(() => {
          fetchTaskResult(result.session_id);
        }, 3000);
      } else {
        throw new Error('Failed to start testing implementation');
      }
    } catch (error) {
      console.error('Failed to start testing implementation:', error);
      setIsProcessing(false);
    }
  };
  
  const fetchTaskResult = async (sessionId: string) => {
    try {
      const response = await fetch(`/api/frontend/session/${sessionId}`);
      if (response.ok) {
        const result = await response.json();
        setTaskResult(result.result);
        setIsProcessing(false);
        
        if (onTaskComplete) {
          onTaskComplete(result.result);
        }
        
        fetchTaskHistory();
      } else {
        // Still processing, try again
        setTimeout(() => {
          fetchTaskResult(sessionId);
        }, 2000);
      }
    } catch (error) {
      console.error('Failed to fetch task result:', error);
      setIsProcessing(false);
    }
  };
  
  const renderComponentDevelopment = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">Component Name</label>
          <Input
            value={componentName}
            onChange={(e) => setComponentName(e.target.value)}
            placeholder="e.g., UserCard, SearchInput"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-2">Component Type</label>
          <Select>
            <SelectTrigger>
              <SelectValue placeholder="Select component type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="functional">Functional Component</SelectItem>
              <SelectItem value="form">Form Component</SelectItem>
              <SelectItem value="display">Display Component</SelectItem>
              <SelectItem value="layout">Layout Component</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Component Description</label>
        <Textarea
          value={componentDescription}
          onChange={(e) => setComponentDescription(e.target.value)}
          placeholder="Describe the component's purpose and behavior"
          rows={3}
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Component Props</label>
        <div className="space-y-2">
          {componentProps.map((prop, index) => (
            <div key={index} className="flex gap-2">
              <Input
                value={prop}
                onChange={(e) => updateArrayItem(componentProps, setComponentProps, index, e.target.value)}
                placeholder="e.g., title: string, onClick: () => void"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => removeArrayItem(componentProps, setComponentProps, index)}
                disabled={componentProps.length === 1}
              >
                Remove
              </Button>
            </div>
          ))}
          <Button
            variant="outline"
            onClick={() => addArrayItem(componentProps, setComponentProps)}
          >
            Add Prop
          </Button>
        </div>
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Functionality Requirements</label>
        <div className="space-y-2">
          {componentFunctionality.map((func, index) => (
            <div key={index} className="flex gap-2">
              <Input
                value={func}
                onChange={(e) => updateArrayItem(componentFunctionality, setComponentFunctionality, index, e.target.value)}
                placeholder="e.g., Handle form validation, Display loading state"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => removeArrayItem(componentFunctionality, setComponentFunctionality, index)}
                disabled={componentFunctionality.length === 1}
              >
                Remove
              </Button>
            </div>
          ))}
          <Button
            variant="outline"
            onClick={() => addArrayItem(componentFunctionality, setComponentFunctionality)}
          >
            Add Functionality
          </Button>
        </div>
      </div>
      
      <Button
        onClick={startComponentDevelopment}
        disabled={isProcessing || !componentName.trim() || !componentDescription.trim()}
        className="w-full"
      >
        {isProcessing ? 'Developing Component...' : 'Start Component Development'}
      </Button>
    </div>
  );
  
  const renderUIImplementation = () => (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">UI Description</label>
        <Textarea
          value={uiDescription}
          onChange={(e) => setUiDescription(e.target.value)}
          placeholder="Describe the UI layout, components, and interactions"
          rows={4}
        />
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Design Tokens (JSON)</label>
        <Textarea
          value={designTokens}
          onChange={(e) => setDesignTokens(e.target.value)}
          placeholder='{"colors": {"primary": "#007bff"}, "spacing": {"sm": "8px"}}'
          rows={4}
        />
      </div>
      
      <Button
        onClick={startUIImplementation}
        disabled={isProcessing || !uiDescription.trim()}
        className="w-full"
      >
        {isProcessing ? 'Implementing UI...' : 'Start UI Implementation'}
      </Button>
    </div>
  );
  
  const renderResponsiveDesign = () => (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Breakpoints</label>
        <div className="flex flex-wrap gap-2">
          {['mobile', 'tablet', 'desktop', 'wide'].map((bp) => (
            <Badge
              key={bp}
              variant={breakpoints.includes(bp) ? 'default' : 'outline'}
              className="cursor-pointer"
              onClick={() => {
                if (breakpoints.includes(bp)) {
                  setBreakpoints(breakpoints.filter(b => b !== bp));
                } else {
                  setBreakpoints([...breakpoints, bp]);
                }
              }}
            >
              {bp}
            </Badge>
          ))}
        </div>
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Responsive Requirements</label>
        <Textarea
          value={responsiveRequirements}
          onChange={(e) => setResponsiveRequirements(e.target.value)}
          placeholder="Describe responsive behavior for different screen sizes"
          rows={4}
        />
      </div>
      
      <Button
        onClick={startResponsiveDesign}
        disabled={isProcessing || !responsiveRequirements.trim()}
        className="w-full"
      >
        {isProcessing ? 'Implementing Responsive Design...' : 'Start Responsive Design'}
      </Button>
    </div>
  );
  
  const renderAccessibilityAudit = () => (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Accessibility Level</label>
        <Select value={accessibilityLevel} onValueChange={setAccessibilityLevel}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="A">WCAG Level A</SelectItem>
            <SelectItem value="AA">WCAG Level AA</SelectItem>
            <SelectItem value="AAA">WCAG Level AAA</SelectItem>
          </SelectContent>
        </Select>
      </div>
      
      <div>
        <label className="block text-sm font-medium mb-2">Components to Audit</label>
        <div className="space-y-2">
          {componentsToAudit.map((component, index) => (
            <div key={index} className="flex gap-2">
              <Input
                value={component}
                onChange={(e) => updateArrayItem(componentsToAudit, setComponentsToAudit, index, e.target.value)}
                placeholder="Component name or selector"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={() => removeArrayItem(componentsToAudit, setComponentsToAudit, index)}
                disabled={componentsToAudit.length === 1}
              >
                Remove
              </Button>
            </div>
          ))}
          <Button
            variant="outline"
            onClick={() => addArrayItem(componentsToAudit, setComponentsToAudit)}
          >
            Add Component
          </Button>
        </div>
      </div>
      
      <Button
        onClick={startAccessibilityAudit}
        disabled={isProcessing || componentsToAudit.filter(c => c.trim()).length === 0}
        className="w-full"
      >
        {isProcessing ? 'Running Accessibility Audit...' : 'Start Accessibility Audit'}
      </Button>
    </div>
  );
  
  const renderPerformanceOptimization = () => (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Performance Targets (JSON)</label>
        <Textarea
          value={performanceTargets}
          onChange={(e) => setPerformanceTargets(e.target.value)}
          placeholder='{"lcp": 2.5, "fid": 100, "cls": 0.1, "bundle_size": "500kb"}'
          rows={4}
        />
      </div>
      
      <Alert>
        <AlertDescription>
          Define performance targets for metrics like Largest Contentful Paint (LCP), 
          First Input Delay (FID), Cumulative Layout Shift (CLS), and bundle size.
        </AlertDescription>
      </Alert>
      
      <Button
        onClick={startPerformanceOptimization}
        disabled={isProcessing || !performanceTargets.trim()}
        className="w-full"
      >
        {isProcessing ? 'Optimizing Performance...' : 'Start Performance Optimization'}
      </Button>
    </div>
  );
  
  const renderTestingImplementation = () => (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Testing Types</label>
        <div className="flex flex-wrap gap-2">
          {['unit', 'integration', 'e2e', 'visual', 'accessibility', 'performance'].map((type) => (
            <Badge
              key={type}
              variant={testingTypes.includes(type) ? 'default' : 'outline'}
              className="cursor-pointer"
              onClick={() => {
                if (testingTypes.includes(type)) {
                  setTestingTypes(testingTypes.filter(t => t !== type));
                } else {
                  setTestingTypes([...testingTypes, type]);
                }
              }}
            >
              {type}
            </Badge>
          ))}
        </div>
      </div>
      
      <Alert>
        <AlertDescription>
          Select the types of tests to implement. Unit and integration tests are recommended 
          for all components, while E2E tests are best for critical user flows.
        </AlertDescription>
      </Alert>
      
      <Button
        onClick={startTestingImplementation}
        disabled={isProcessing || testingTypes.length === 0}
        className="w-full"
      >
        {isProcessing ? 'Implementing Tests...' : 'Start Testing Implementation'}
      </Button>
    </div>
  );
  
  const renderTaskResult = () => {
    if (!taskResult) return null;
    
    return (
      <Card>
        <CardHeader>
          <CardTitle>Task Result</CardTitle>
          <Badge variant="outline">
            Confidence: {Math.round((taskResult.confidence_score || 0) * 100)}%
          </Badge>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible>
            {taskResult.component_code && (
              <AccordionItem value="component-code">
                <AccordionTrigger>Component Code</AccordionTrigger>
                <AccordionContent>
                  <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
                    {JSON.stringify(taskResult.component_code, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>
            )}
            
            {taskResult.ui_implementation && (
              <AccordionItem value="ui-implementation">
                <AccordionTrigger>UI Implementation</AccordionTrigger>
                <AccordionContent>
                  <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
                    {JSON.stringify(taskResult.ui_implementation, null, 2)}
                  </pre>
                </AccordionContent>
              </AccordionItem>
            )}
          </Accordion>
        </CardContent>
      </Card>
    );
  };
  
  const renderTaskHistory = () => (
    <div className="space-y-4">
      {taskHistory.length === 0 ? (
        <p className="text-gray-500 text-center py-8">No task history available</p>
      ) : (
        taskHistory.map((task, index) => (
          <Card key={index}>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="text-lg">{task.task_type}</CardTitle>
                <Badge variant="outline">
                  {Math.round((task.confidence_score || 0) * 100)}%
                </Badge>
              </div>
              <p className="text-sm text-gray-500">
                {new Date(task.created_at).toLocaleString()}
              </p>
            </CardHeader>
            <CardContent>
              <p className="text-sm">Session ID: {task.session_id}</p>
            </CardContent>
          </Card>
        ))
      )}
    </div>
  );
  
  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">Frontend Developer Agent</h1>
        <p className="text-gray-600">
          Advanced frontend development with React, TypeScript, and modern patterns
        </p>
      </div>
      
      {isProcessing && (
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex items-center space-x-4">
              <Progress value={33} className="flex-1" />
              <span className="text-sm font-medium">Processing...</span>
            </div>
            <p className="text-sm text-gray-500 mt-2">
              The Frontend Developer Agent is working on your task. This may take a few moments.
            </p>
          </CardContent>
        </Card>
      )}
      
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-7">
          <TabsTrigger value="component-dev">Components</TabsTrigger>
          <TabsTrigger value="ui-impl">UI Implementation</TabsTrigger>
          <TabsTrigger value="responsive">Responsive</TabsTrigger>
          <TabsTrigger value="accessibility">A11y</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="testing">Testing</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
        </TabsList>
        
        <TabsContent value="component-dev">
          <Card>
            <CardHeader>
              <CardTitle>Component Development</CardTitle>
            </CardHeader>
            <CardContent>
              {renderComponentDevelopment()}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="ui-impl">
          <Card>
            <CardHeader>
              <CardTitle>UI Implementation</CardTitle>
            </CardHeader>
            <CardContent>
              {renderUIImplementation()}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="responsive">
          <Card>
            <CardHeader>
              <CardTitle>Responsive Design</CardTitle>
            </CardHeader>
            <CardContent>
              {renderResponsiveDesign()}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="accessibility">
          <Card>
            <CardHeader>
              <CardTitle>Accessibility Audit</CardTitle>
            </CardHeader>
            <CardContent>
              {renderAccessibilityAudit()}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="performance">
          <Card>
            <CardHeader>
              <CardTitle>Performance Optimization</CardTitle>
            </CardHeader>
            <CardContent>
              {renderPerformanceOptimization()}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="testing">
          <Card>
            <CardHeader>
              <CardTitle>Testing Implementation</CardTitle>
            </CardHeader>
            <CardContent>
              {renderTestingImplementation()}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Task History</CardTitle>
            </CardHeader>
            <CardContent>
              {renderTaskHistory()}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
      
      {taskResult && (
        <div className="mt-6">
          {renderTaskResult()}
        </div>
      )}
    </div>
  );
};

export default FrontendDeveloperDashboard;
```

## 6. Testing Strategy

### 6.1 Unit Tests for Frontend Developer Agent

```python
# tests/test_frontend_developer_agent.py
import pytest
from unittest.mock import Mock, patch
from core.agents.frontend_developer_agent import FrontendDeveloperAgent
from core.models import FrontendTask, ComponentSpec, DesignSystem

class TestFrontendDeveloperAgent:
    @pytest.fixture
    def agent(self):
        return FrontendDeveloperAgent()
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly"""
        assert agent.name == "Frontend Developer Agent"
        assert agent.description is not None
        assert hasattr(agent, 'llm_service')
    
    @pytest.mark.asyncio
    async def test_develop_component(self, agent):
        """Test component development functionality"""
        component_spec = ComponentSpec(
            name="UserCard",
            type="functional",
            props=[{"name": "user", "type": "User", "required": True}],
            styling="tailwind",
            accessibility_requirements=["aria-label", "keyboard-navigation"]
        )
        
        task = FrontendTask(
            task_type="develop-component",
            component_spec=component_spec
        )
        
        with patch.object(agent.llm_service, 'generate_response') as mock_llm:
            mock_llm.return_value = {
                "component_code": {
                    "tsx": "const UserCard = ({ user }) => <div>{user.name}</div>",
                    "styles": ".user-card { padding: 1rem; }",
                    "types": "interface User { name: string; }"
                },
                "tests": {
                    "unit": "test('renders user name', () => { ... })"
                }
            }
            
            result = await agent.develop_component(task)
            
            assert result.component_code is not None
            assert "UserCard" in result.component_code["tsx"]
            assert result.confidence_score > 0.7
    
    @pytest.mark.asyncio
    async def test_implement_ui(self, agent):
        """Test UI implementation functionality"""
        design_system = DesignSystem(
            colors={"primary": "#007bff", "secondary": "#6c757d"},
            typography={"font_family": "Inter", "base_size": "16px"},
            spacing={"sm": "8px", "md": "16px", "lg": "24px"}
        )
        
        task = FrontendTask(
            task_type="implement-ui",
            ui_description="Create a modern dashboard layout",
            design_system=design_system
        )
        
        with patch.object(agent.llm_service, 'generate_response') as mock_llm:
            mock_llm.return_value = {
                "ui_implementation": {
                    "layout": "<div className='dashboard'>...</div>",
                    "components": ["Header", "Sidebar", "MainContent"],
                    "styles": ".dashboard { display: grid; ... }"
                }
            }
            
            result = await agent.implement_ui(task)
            
            assert result.ui_implementation is not None
            assert "dashboard" in result.ui_implementation["layout"]
    
    @pytest.mark.asyncio
    async def test_responsive_design(self, agent):
        """Test responsive design implementation"""
        task = FrontendTask(
            task_type="responsive-design",
            breakpoints=["mobile", "tablet", "desktop"],
            responsive_requirements="Mobile-first approach with collapsible sidebar"
        )
        
        with patch.object(agent.llm_service, 'generate_response') as mock_llm:
            mock_llm.return_value = {
                "responsive_implementation": {
                    "breakpoints": {
                        "mobile": "@media (max-width: 768px)",
                        "tablet": "@media (min-width: 769px) and (max-width: 1024px)"
                    },
                    "adaptive_components": ["Sidebar", "Navigation"]
                }
            }
            
            result = await agent.implement_responsive_design(task)
            
            assert result.responsive_implementation is not None
            assert "mobile" in result.responsive_implementation["breakpoints"]
    
    @pytest.mark.asyncio
    async def test_accessibility_audit(self, agent):
        """Test accessibility audit functionality"""
        task = FrontendTask(
            task_type="accessibility-audit",
            accessibility_level="AA",
            components_to_audit=["Button", "Form", "Navigation"]
        )
        
        with patch.object(agent.llm_service, 'generate_response') as mock_llm:
            mock_llm.return_value = {
                "accessibility_report": {
                    "issues": [
                        {"component": "Button", "issue": "Missing aria-label", "severity": "high"}
                    ],
                    "recommendations": ["Add aria-labels to all interactive elements"]
                }
            }
            
            result = await agent.audit_accessibility(task)
            
            assert result.accessibility_report is not None
            assert len(result.accessibility_report["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_performance_optimization(self, agent):
        """Test performance optimization functionality"""
        task = FrontendTask(
            task_type="optimize-performance",
            performance_targets={"lcp": 2.5, "fid": 100, "cls": 0.1}
        )
        
        with patch.object(agent.llm_service, 'generate_response') as mock_llm:
            mock_llm.return_value = {
                "optimization_report": {
                    "optimizations": [
                        {"type": "code-splitting", "impact": "30% bundle size reduction"},
                        {"type": "lazy-loading", "impact": "Improved LCP by 1.2s"}
                    ],
                    "metrics": {"lcp": 2.1, "fid": 85, "cls": 0.08}
                }
            }
            
            result = await agent.optimize_performance(task)
            
            assert result.optimization_report is not None
            assert len(result.optimization_report["optimizations"]) > 0
    
    @pytest.mark.asyncio
    async def test_implement_testing(self, agent):
        """Test testing implementation functionality"""
        task = FrontendTask(
            task_type="implement-testing",
            testing_types=["unit", "integration", "e2e"]
        )
        
        with patch.object(agent.llm_service, 'generate_response') as mock_llm:
            mock_llm.return_value = {
                "testing_implementation": {
                    "unit_tests": "describe('Component', () => { ... })",
                    "integration_tests": "test('user flow', () => { ... })",
                    "e2e_tests": "cy.visit('/'); cy.get('[data-testid=button]').click();"
                }
            }
            
            result = await agent.implement_testing(task)
            
            assert result.testing_implementation is not None
            assert "unit_tests" in result.testing_implementation
    
    @pytest.mark.asyncio
    async def test_process_task(self, agent):
        """Test overall task processing"""
        task = FrontendTask(
            task_type="develop-component",
            component_spec=ComponentSpec(
                name="TestComponent",
                type="functional",
                props=[],
                styling="tailwind"
            )
        )
        
        with patch.object(agent, 'develop_component') as mock_develop:
            mock_develop.return_value = Mock(confidence_score=0.9)
            
            result = await agent.process_task(task)
            
            assert result is not None
            mock_develop.assert_called_once_with(task)
```

### 6.2 Integration Tests for Frontend API

```python
# tests/test_frontend_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestFrontendAPI:
    def test_develop_component_endpoint(self):
        """Test component development endpoint"""
        payload = {
            "component_spec": {
                "name": "TestButton",
                "type": "functional",
                "props": [{"name": "onClick", "type": "function", "required": True}],
                "styling": "tailwind",
                "accessibility_requirements": ["aria-label"]
            }
        }
        
        response = client.post("/api/frontend/develop-component", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "task_id" in data
    
    def test_implement_ui_endpoint(self):
        """Test UI implementation endpoint"""
        payload = {
            "ui_description": "Create a user profile page",
            "design_system": {
                "colors": {"primary": "#007bff"},
                "typography": {"font_family": "Inter"},
                "spacing": {"sm": "8px"}
            }
        }
        
        response = client.post("/api/frontend/implement-ui", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
    
    def test_responsive_design_endpoint(self):
        """Test responsive design endpoint"""
        payload = {
            "breakpoints": ["mobile", "tablet", "desktop"],
            "responsive_requirements": "Mobile-first design"
        }
        
        response = client.post("/api/frontend/responsive-design", json=payload)
        
        assert response.status_code == 200
    
    def test_accessibility_audit_endpoint(self):
        """Test accessibility audit endpoint"""
        payload = {
            "accessibility_level": "AA",
            "components_to_audit": ["Button", "Form"]
        }
        
        response = client.post("/api/frontend/accessibility-audit", json=payload)
        
        assert response.status_code == 200
    
    def test_optimize_performance_endpoint(self):
        """Test performance optimization endpoint"""
        payload = {
            "performance_targets": {"lcp": 2.5, "fid": 100, "cls": 0.1}
        }
        
        response = client.post("/api/frontend/optimize-performance", json=payload)
        
        assert response.status_code == 200
    
    def test_implement_testing_endpoint(self):
        """Test testing implementation endpoint"""
        payload = {
            "testing_types": ["unit", "integration"]
        }
        
        response = client.post("/api/frontend/implement-testing", json=payload)
        
        assert response.status_code == 200
    
    def test_get_session_result(self):
        """Test getting session results"""
        # First create a session
        payload = {"component_spec": {"name": "Test", "type": "functional", "props": [], "styling": "tailwind"}}
        create_response = client.post("/api/frontend/develop-component", json=payload)
        session_id = create_response.json()["session_id"]
        
        # Then get the result
        response = client.get(f"/api/frontend/sessions/{session_id}/result")
        
        assert response.status_code in [200, 202]  # 202 if still processing
    
    def test_get_history(self):
        """Test getting frontend development history"""
        response = client.get("/api/frontend/history")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_templates(self):
        """Test getting component templates"""
        response = client.get("/api/frontend/templates")
        
        assert response.status_code == 200
        data = response.json()
        assert "component_templates" in data
    
    def test_analyze_existing_component(self):
        """Test analyzing existing components"""
        payload = {
            "component_path": "/components/Button.tsx"
        }
        
        response = client.post("/api/frontend/analyze-component", json=payload)
        
        assert response.status_code == 200
    
    def test_get_best_practices(self):
        """Test getting frontend best practices"""
        response = client.get("/api/frontend/best-practices")
        
        assert response.status_code == 200
        data = response.json()
        assert "practices" in data
```

## 7. Validation Criteria

### 7.1 Backend Validation

**Core Functionality:**
- [ ] FrontendDeveloperAgent successfully processes all task types
- [ ] Component development generates valid React/TypeScript code
- [ ] UI implementation follows design system specifications
- [ ] Responsive design includes proper breakpoints and media queries
- [ ] Accessibility audit identifies WCAG compliance issues
- [ ] Performance optimization provides actionable recommendations
- [ ] Testing implementation generates comprehensive test suites

**API Endpoints:**
- [ ] All endpoints return appropriate HTTP status codes
- [ ] Request/response schemas match Pydantic models
- [ ] Error handling provides meaningful error messages
- [ ] Session management works correctly
- [ ] History tracking maintains task records

**Integration:**
- [ ] LLM service integration functions properly
- [ ] Database operations (if applicable) work correctly
- [ ] Logging captures important events
- [ ] Configuration management is flexible

### 7.2 Frontend Validation

**User Interface:**
- [ ] Dashboard loads without errors
- [ ] All tabs are functional and display correct content
- [ ] Form inputs validate user data appropriately
- [ ] Task results display in readable format
- [ ] History shows previous tasks with details

**User Experience:**
- [ ] Interface is intuitive and easy to navigate
- [ ] Loading states provide clear feedback
- [ ] Error messages are helpful and actionable
- [ ] Responsive design works on different screen sizes
- [ ] Accessibility features are implemented

**Functionality:**
- [ ] All task types can be initiated from the UI
- [ ] Real-time updates show task progress
- [ ] Results can be viewed and downloaded
- [ ] Templates and best practices are accessible
- [ ] Component analysis works with file uploads

### 7.3 Integration Validation

**End-to-End Workflows:**
- [ ] Complete component development workflow
- [ ] UI implementation with design system
- [ ] Responsive design implementation
- [ ] Accessibility audit and remediation
- [ ] Performance optimization cycle
- [ ] Testing implementation and validation

**Data Flow:**
- [ ] Frontend correctly sends requests to backend
- [ ] Backend processes requests and returns results
- [ ] Real-time updates work properly
- [ ] Error handling works across the stack
- [ ] Session persistence maintains state

## 8. Human Testing Scenarios

### 8.1 Component Development Scenario
**Objective:** Test the complete component development workflow

**Steps:**
1. Navigate to the Frontend Developer Dashboard
2. Go to the "Components" tab
3. Fill in component details:
   - Name: "ProductCard"
   - Type: "Functional"
   - Props: name (string), price (number), image (string)
   - Styling: "Tailwind CSS"
   - Accessibility: "ARIA labels, keyboard navigation"
4. Click "Start Component Development"
5. Wait for processing to complete
6. Review the generated component code
7. Check that TypeScript interfaces are included
8. Verify accessibility features are implemented

**Expected Results:**
- Valid React component with TypeScript
- Proper prop types and interfaces
- Tailwind CSS classes applied
- Accessibility attributes included
- Unit tests generated

### 8.2 UI Implementation Scenario
**Objective:** Test UI implementation with design system

**Steps:**
1. Go to the "UI Implementation" tab
2. Enter UI description: "Create a modern e-commerce product listing page"
3. Configure design system:
   - Colors: primary (#007bff), secondary (#6c757d)
   - Typography: Inter font family
   - Spacing: 8px, 16px, 24px scale
4. Click "Start UI Implementation"
5. Review the generated layout and components
6. Check design system consistency

**Expected Results:**
- Complete page layout with components
- Design system tokens properly applied
- Responsive grid or flexbox layout
- Consistent spacing and typography

### 8.3 Accessibility Audit Scenario
**Objective:** Test accessibility auditing capabilities

**Steps:**
1. Go to the "Accessibility" tab
2. Select WCAG Level AA
3. Add components to audit: "Navigation", "ProductCard", "CheckoutForm"
4. Click "Start Accessibility Audit"
5. Review the accessibility report
6. Check recommendations for improvements

**Expected Results:**
- Detailed accessibility issues identified
- WCAG guideline references provided
- Actionable recommendations given
- Severity levels assigned to issues

### 8.4 Performance Optimization Scenario
**Objective:** Test performance optimization features

**Steps:**
1. Go to the "Performance" tab
2. Set performance targets:
   - LCP: 2.5 seconds
   - FID: 100 milliseconds
   - CLS: 0.1
   - Bundle size: 500KB
3. Click "Start Performance Optimization"
4. Review optimization recommendations
5. Check impact estimates for each optimization

**Expected Results:**
- Specific optimization strategies provided
- Code splitting recommendations
- Image optimization suggestions
- Bundle analysis and improvements
- Performance metrics predictions

---

## Next Steps

After completing the Frontend Developer Agent enhancement, proceed to:

**Next Action Plan:** `08-backend-developer-agent-enhancement.md`

This will focus on enhancing the Backend Developer Agent with advanced API development, database design, microservices architecture, and deployment capabilities.