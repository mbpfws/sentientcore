# 04 - Ultra Orchestrator Implementation

## Overview

The Ultra Orchestrator serves as the central coordination hub for the entire multi-agent system. It manages workflow execution, task distribution, agent coordination, and maintains the overall system state. This phase transforms the existing basic orchestrator into a sophisticated workflow management system.

## Current State Analysis

### Existing File
- `core/agents/ultra_orchestrator.py` - Basic orchestration logic

### Enhancement Requirements
- LangGraph integration for workflow orchestration
- Advanced task planning and decomposition
- Dynamic agent assignment and load balancing
- Workflow state management
- Real-time monitoring and adaptation
- Integration with all core services

## Implementation Tasks

### Task 4.1: Enhanced Ultra Orchestrator Agent

**File**: `core/agents/ultra_orchestrator.py` (Complete Rewrite)

**Ultra Orchestrator Implementation**:
```python
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
from enum import Enum

from langgraph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from .base_agent import BaseAgent, AgentStatus
from ..models import WorkflowState, TaskDefinition, AgentProfile
from ..services.memory_service import MemoryService
from ..services.state_service import StateService
from ..services.llm_service import LLMService

class WorkflowPhase(Enum):
    INITIALIZATION = "initialization"
    REQUIREMENTS_ANALYSIS = "requirements_analysis"
    RESEARCH = "research"
    ARCHITECTURE_PLANNING = "architecture_planning"
    FRONTEND_DEVELOPMENT = "frontend_development"
    BACKEND_DEVELOPMENT = "backend_development"
    INTEGRATION = "integration"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMPLETION = "completion"

class UltraOrchestratorAgent(BaseAgent):
    def __init__(self, agent_id: str = "ultra_orchestrator"):
        super().__init__(
            agent_id=agent_id,
            name="Ultra Orchestrator",
            description="Central coordination agent for multi-agent workflows"
        )
        self.capabilities = [
            "workflow_orchestration",
            "task_decomposition",
            "agent_coordination",
            "progress_monitoring",
            "adaptive_planning"
        ]
        
        self.active_workflows = {}
        self.agent_registry = None
        self.message_bus = None
        self.workflow_graph = None
        self.task_queue = asyncio.Queue()
        
    async def initialize(self, agent_registry, message_bus):
        """Initialize orchestrator with required dependencies"""
        self.agent_registry = agent_registry
        self.message_bus = message_bus
        await self._build_workflow_graph()
        await self.update_status(AgentStatus.IDLE)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming task and initiate workflow"""
        try:
            await self.update_status(AgentStatus.THINKING)
            
            # Create new workflow
            workflow_id = f"workflow_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            workflow = await self._create_workflow(workflow_id, task)
            
            # Store workflow state
            self.active_workflows[workflow_id] = workflow
            await self.state_service.update_workflow_state(workflow_id, workflow.dict())
            
            # Start workflow execution
            await self.update_status(AgentStatus.WORKING)
            result = await self._execute_workflow(workflow)
            
            await self.update_status(AgentStatus.COMPLETED)
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "result": result
            }
            
        except Exception as e:
            await self.update_status(AgentStatus.ERROR, str(e))
            raise
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Ultra Orchestrator can handle any high-level task"""
        return task.get("type") in [
            "development_request",
            "feature_implementation",
            "system_enhancement",
            "workflow_execution"
        ]
    
    async def _create_workflow(self, workflow_id: str, task: Dict[str, Any]) -> WorkflowState:
        """Create workflow from task requirements"""
        # Analyze task and decompose into phases
        phases = await self._analyze_and_decompose_task(task)
        
        # Create task definitions for each phase
        tasks = []
        for i, phase in enumerate(phases):
            task_def = TaskDefinition(
                task_id=f"{workflow_id}_task_{i+1}",
                task_type=phase["type"],
                description=phase["description"],
                parameters=phase["parameters"],
                priority=phase.get("priority", 1),
                dependencies=phase.get("dependencies", []),
                created_at=datetime.utcnow()
            )
            tasks.append(task_def)
        
        workflow = WorkflowState(
            workflow_id=workflow_id,
            name=task.get("name", "Development Workflow"),
            description=task.get("description", ""),
            current_phase=WorkflowPhase.INITIALIZATION.value,
            tasks=tasks,
            agent_assignments={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return workflow
    
    async def _analyze_and_decompose_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use LLM to analyze task and decompose into actionable phases"""
        prompt = f"""
        Analyze the following development task and decompose it into specific, actionable phases:
        
        Task: {task.get('description', '')}
        Requirements: {task.get('requirements', [])}
        Technology Stack: {task.get('tech_stack', {})}
        
        Decompose this into phases following this structure:
        1. Requirements Analysis
        2. Research (if needed)
        3. Architecture Planning
        4. Frontend Development (if applicable)
        5. Backend Development (if applicable)
        6. Integration
        7. Testing
        8. Deployment (if applicable)
        
        For each phase, provide:
        - type: phase type
        - description: detailed description
        - parameters: specific parameters needed
        - priority: 1-5 (5 being highest)
        - dependencies: list of dependent phase types
        - estimated_duration: estimated time in hours
        - required_agents: list of agent types needed
        
        Return as JSON array.
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.3
        )
        
        # Parse LLM response and validate
        try:
            import json
            phases = json.loads(response)
            return phases
        except json.JSONDecodeError:
            # Fallback to default phases if LLM response is invalid
            return self._get_default_phases(task)
    
    async def _execute_workflow(self, workflow: WorkflowState) -> Dict[str, Any]:
        """Execute workflow using LangGraph"""
        # Initialize workflow state for LangGraph
        initial_state = {
            "workflow_id": workflow.workflow_id,
            "current_phase": workflow.current_phase,
            "tasks": [task.dict() for task in workflow.tasks],
            "completed_tasks": [],
            "agent_assignments": workflow.agent_assignments,
            "context": {},
            "errors": []
        }
        
        # Execute workflow graph
        result = await self.workflow_graph.ainvoke(initial_state)
        
        return result
    
    async def _build_workflow_graph(self):
        """Build LangGraph workflow for orchestration"""
        workflow = StateGraph(dict)
        
        # Add nodes for each workflow phase
        workflow.add_node("initialize", self._initialize_workflow)
        workflow.add_node("analyze_requirements", self._analyze_requirements)
        workflow.add_node("conduct_research", self._conduct_research)
        workflow.add_node("plan_architecture", self._plan_architecture)
        workflow.add_node("develop_frontend", self._develop_frontend)
        workflow.add_node("develop_backend", self._develop_backend)
        workflow.add_node("integrate_components", self._integrate_components)
        workflow.add_node("run_tests", self._run_tests)
        workflow.add_node("deploy_system", self._deploy_system)
        workflow.add_node("monitor_system", self._monitor_system)
        
        # Define workflow edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "analyze_requirements")
        workflow.add_conditional_edges(
            "analyze_requirements",
            self._should_conduct_research,
            {
                "research": "conduct_research",
                "planning": "plan_architecture"
            }
        )
        workflow.add_edge("conduct_research", "plan_architecture")
        workflow.add_conditional_edges(
            "plan_architecture",
            self._determine_development_path,
            {
                "frontend_first": "develop_frontend",
                "backend_first": "develop_backend",
                "parallel": "develop_frontend"  # Start with frontend for parallel
            }
        )
        workflow.add_edge("develop_frontend", "develop_backend")
        workflow.add_edge("develop_backend", "integrate_components")
        workflow.add_edge("integrate_components", "run_tests")
        workflow.add_conditional_edges(
            "run_tests",
            self._should_deploy,
            {
                "deploy": "deploy_system",
                "monitor": "monitor_system"
            }
        )
        workflow.add_edge("deploy_system", "monitor_system")
        workflow.add_edge("monitor_system", END)
        
        self.workflow_graph = workflow.compile()
    
    # Workflow Phase Implementations
    async def _initialize_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize workflow execution"""
        await self.log_activity(f"Initializing workflow {state['workflow_id']}")
        
        # Set up workflow context
        state["context"]["start_time"] = datetime.utcnow().isoformat()
        state["context"]["phase_history"] = []
        
        # Store initial state
        await self.memory_service.store_knowledge(
            "workflow_context",
            state,
            {"workflow_id": state["workflow_id"], "phase": "initialization"}
        )
        
        return state
    
    async def _analyze_requirements(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and refine requirements"""
        await self.log_activity("Analyzing requirements")
        
        # Find and assign requirements analysis task to appropriate agent
        analysis_tasks = [task for task in state["tasks"] if task["task_type"] == "requirements_analysis"]
        
        if analysis_tasks:
            task = analysis_tasks[0]
            
            # Find best agent for requirements analysis
            suitable_agents = await self.agent_registry.find_agents_by_capability("requirements_analysis")
            if not suitable_agents:
                # Fallback to research agent
                suitable_agents = await self.agent_registry.get_agents_by_type("ResearchAgent")
            
            if suitable_agents:
                assigned_agent = suitable_agents[0]
                result = await assigned_agent.process_task(task)
                
                # Update state with analysis results
                state["context"]["requirements_analysis"] = result
                state["completed_tasks"].append(task["task_id"])
                
                # Update workflow state
                await self.state_service.update_workflow_state(
                    state["workflow_id"],
                    {"current_phase": "requirements_analysis", "progress": 0.1}
                )
        
        return state
    
    async def _conduct_research(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct research if needed"""
        await self.log_activity("Conducting research")
        
        # Find research tasks
        research_tasks = [task for task in state["tasks"] if task["task_type"] == "research"]
        
        for task in research_tasks:
            # Assign to research agent
            research_agents = await self.agent_registry.get_agents_by_type("ResearchAgent")
            if research_agents:
                assigned_agent = research_agents[0]
                result = await assigned_agent.process_task(task)
                
                state["context"]["research_results"] = result
                state["completed_tasks"].append(task["task_id"])
        
        # Update progress
        await self.state_service.update_workflow_state(
            state["workflow_id"],
            {"current_phase": "research", "progress": 0.2}
        )
        
        return state
    
    async def _plan_architecture(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Plan system architecture"""
        await self.log_activity("Planning architecture")
        
        # Find architecture planning tasks
        planning_tasks = [task for task in state["tasks"] if task["task_type"] == "architecture_planning"]
        
        for task in planning_tasks:
            # Assign to architect planner agent
            planner_agents = await self.agent_registry.get_agents_by_type("ArchitectPlannerAgent")
            if planner_agents:
                assigned_agent = planner_agents[0]
                result = await assigned_agent.process_task(task)
                
                state["context"]["architecture_plan"] = result
                state["completed_tasks"].append(task["task_id"])
        
        # Update progress
        await self.state_service.update_workflow_state(
            state["workflow_id"],
            {"current_phase": "architecture_planning", "progress": 0.3}
        )
        
        return state
    
    async def _develop_frontend(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Develop frontend components"""
        await self.log_activity("Developing frontend")
        
        # Find frontend development tasks
        frontend_tasks = [task for task in state["tasks"] if task["task_type"] == "frontend_development"]
        
        for task in frontend_tasks:
            # Assign to frontend developer agent
            frontend_agents = await self.agent_registry.get_agents_by_type("FrontendDeveloperAgent")
            if frontend_agents:
                assigned_agent = frontend_agents[0]
                result = await assigned_agent.process_task(task)
                
                state["context"]["frontend_artifacts"] = result
                state["completed_tasks"].append(task["task_id"])
        
        # Update progress
        await self.state_service.update_workflow_state(
            state["workflow_id"],
            {"current_phase": "frontend_development", "progress": 0.5}
        )
        
        return state
    
    async def _develop_backend(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Develop backend components"""
        await self.log_activity("Developing backend")
        
        # Find backend development tasks
        backend_tasks = [task for task in state["tasks"] if task["task_type"] == "backend_development"]
        
        for task in backend_tasks:
            # Assign to backend developer agent
            backend_agents = await self.agent_registry.get_agents_by_type("BackendDeveloperAgent")
            if backend_agents:
                assigned_agent = backend_agents[0]
                result = await assigned_agent.process_task(task)
                
                state["context"]["backend_artifacts"] = result
                state["completed_tasks"].append(task["task_id"])
        
        # Update progress
        await self.state_service.update_workflow_state(
            state["workflow_id"],
            {"current_phase": "backend_development", "progress": 0.7}
        )
        
        return state
    
    async def _integrate_components(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate frontend and backend components"""
        await self.log_activity("Integrating components")
        
        # Find integration tasks
        integration_tasks = [task for task in state["tasks"] if task["task_type"] == "integration"]
        
        for task in integration_tasks:
            # Assign to coding agent for integration
            coding_agents = await self.agent_registry.get_agents_by_type("CodingAgent")
            if coding_agents:
                assigned_agent = coding_agents[0]
                result = await assigned_agent.process_task(task)
                
                state["context"]["integration_results"] = result
                state["completed_tasks"].append(task["task_id"])
        
        # Update progress
        await self.state_service.update_workflow_state(
            state["workflow_id"],
            {"current_phase": "integration", "progress": 0.8}
        )
        
        return state
    
    async def _run_tests(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive tests"""
        await self.log_activity("Running tests")
        
        # Find testing tasks
        testing_tasks = [task for task in state["tasks"] if task["task_type"] == "testing"]
        
        for task in testing_tasks:
            # Assign to coding agent for testing
            coding_agents = await self.agent_registry.get_agents_by_type("CodingAgent")
            if coding_agents:
                assigned_agent = coding_agents[0]
                result = await assigned_agent.process_task(task)
                
                state["context"]["test_results"] = result
                state["completed_tasks"].append(task["task_id"])
        
        # Update progress
        await self.state_service.update_workflow_state(
            state["workflow_id"],
            {"current_phase": "testing", "progress": 0.9}
        )
        
        return state
    
    async def _deploy_system(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy system if applicable"""
        await self.log_activity("Deploying system")
        
        # Find deployment tasks
        deployment_tasks = [task for task in state["tasks"] if task["task_type"] == "deployment"]
        
        for task in deployment_tasks:
            # Assign to coding agent for deployment
            coding_agents = await self.agent_registry.get_agents_by_type("CodingAgent")
            if coding_agents:
                assigned_agent = coding_agents[0]
                result = await assigned_agent.process_task(task)
                
                state["context"]["deployment_results"] = result
                state["completed_tasks"].append(task["task_id"])
        
        return state
    
    async def _monitor_system(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor system and finalize workflow"""
        await self.log_activity("Monitoring and finalizing")
        
        # Assign monitoring agent
        monitoring_agents = await self.agent_registry.get_agents_by_type("MonitoringAgent")
        if monitoring_agents:
            monitoring_agent = monitoring_agents[0]
            monitoring_task = {
                "task_type": "monitoring",
                "description": "Monitor workflow completion and system health",
                "parameters": {"workflow_id": state["workflow_id"]}
            }
            result = await monitoring_agent.process_task(monitoring_task)
            state["context"]["monitoring_results"] = result
        
        # Finalize workflow
        state["context"]["end_time"] = datetime.utcnow().isoformat()
        state["context"]["total_duration"] = (
            datetime.fromisoformat(state["context"]["end_time"]) - 
            datetime.fromisoformat(state["context"]["start_time"])
        ).total_seconds()
        
        # Update final progress
        await self.state_service.update_workflow_state(
            state["workflow_id"],
            {"current_phase": "completed", "progress": 1.0, "status": "completed"}
        )
        
        return state
    
    # Conditional Edge Functions
    async def _should_conduct_research(self, state: Dict[str, Any]) -> str:
        """Determine if research phase is needed"""
        research_tasks = [task for task in state["tasks"] if task["task_type"] == "research"]
        return "research" if research_tasks else "planning"
    
    async def _determine_development_path(self, state: Dict[str, Any]) -> str:
        """Determine development approach (frontend-first, backend-first, or parallel)"""
        # Default to frontend-first for user validation
        return "frontend_first"
    
    async def _should_deploy(self, state: Dict[str, Any]) -> str:
        """Determine if deployment is needed"""
        deployment_tasks = [task for task in state["tasks"] if task["task_type"] == "deployment"]
        return "deploy" if deployment_tasks else "monitor"
    
    def _get_default_phases(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get default phases if LLM analysis fails"""
        return [
            {
                "type": "requirements_analysis",
                "description": "Analyze and refine requirements",
                "parameters": {"task": task},
                "priority": 5,
                "dependencies": [],
                "estimated_duration": 2,
                "required_agents": ["ResearchAgent"]
            },
            {
                "type": "architecture_planning",
                "description": "Plan system architecture",
                "parameters": {"requirements": task.get("requirements", [])},
                "priority": 4,
                "dependencies": ["requirements_analysis"],
                "estimated_duration": 4,
                "required_agents": ["ArchitectPlannerAgent"]
            },
            {
                "type": "frontend_development",
                "description": "Develop frontend components",
                "parameters": {"tech_stack": task.get("tech_stack", {})},
                "priority": 3,
                "dependencies": ["architecture_planning"],
                "estimated_duration": 8,
                "required_agents": ["FrontendDeveloperAgent"]
            },
            {
                "type": "backend_development",
                "description": "Develop backend components",
                "parameters": {"tech_stack": task.get("tech_stack", {})},
                "priority": 3,
                "dependencies": ["architecture_planning"],
                "estimated_duration": 8,
                "required_agents": ["BackendDeveloperAgent"]
            },
            {
                "type": "integration",
                "description": "Integrate all components",
                "parameters": {},
                "priority": 2,
                "dependencies": ["frontend_development", "backend_development"],
                "estimated_duration": 4,
                "required_agents": ["CodingAgent"]
            },
            {
                "type": "testing",
                "description": "Run comprehensive tests",
                "parameters": {},
                "priority": 2,
                "dependencies": ["integration"],
                "estimated_duration": 4,
                "required_agents": ["CodingAgent"]
            }
        ]
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            state = await self.state_service.get_workflow_state(workflow_id)
            return {
                "workflow_id": workflow_id,
                "status": workflow.status,
                "current_phase": workflow.current_phase,
                "progress": workflow.progress,
                "tasks_completed": len([t for t in workflow.tasks if t.status == "completed"]),
                "total_tasks": len(workflow.tasks),
                "state": state
            }
        return {"error": "Workflow not found"}
    
    async def pause_workflow(self, workflow_id: str):
        """Pause workflow execution"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = "paused"
            await self.state_service.update_workflow_state(workflow_id, {"status": "paused"})
    
    async def resume_workflow(self, workflow_id: str):
        """Resume paused workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = "active"
            await self.state_service.update_workflow_state(workflow_id, {"status": "active"})
    
    async def cancel_workflow(self, workflow_id: str):
        """Cancel workflow execution"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = "cancelled"
            await self.state_service.update_workflow_state(workflow_id, {"status": "cancelled"})
            del self.active_workflows[workflow_id]
```

### Task 4.2: Workflow Management Service

**File**: `core/services/workflow_service.py`

**Workflow Service Implementation**:
```python
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from ..models import WorkflowState, TaskDefinition
from .state_service import StateService
from .memory_service import MemoryService

class WorkflowService:
    def __init__(self, state_service: StateService, memory_service: MemoryService):
        self.state_service = state_service
        self.memory_service = memory_service
        self.active_workflows = {}
        self.workflow_templates = {}
    
    async def create_workflow_template(self, template_name: str, template_config: Dict[str, Any]):
        """Create reusable workflow template"""
        self.workflow_templates[template_name] = template_config
        await self.memory_service.store_knowledge(
            'workflow_templates',
            template_config,
            {'template_name': template_name}
        )
    
    async def instantiate_workflow_from_template(self, template_name: str, parameters: Dict[str, Any]) -> WorkflowState:
        """Create workflow instance from template"""
        if template_name not in self.workflow_templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.workflow_templates[template_name]
        
        # Generate unique workflow ID
        workflow_id = f"{template_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create tasks from template
        tasks = []
        for task_template in template.get('tasks', []):
            task = TaskDefinition(
                task_id=f"{workflow_id}_{task_template['name']}",
                task_type=task_template['type'],
                description=task_template['description'].format(**parameters),
                parameters={**task_template.get('parameters', {}), **parameters},
                priority=task_template.get('priority', 1),
                dependencies=task_template.get('dependencies', []),
                created_at=datetime.utcnow()
            )
            tasks.append(task)
        
        workflow = WorkflowState(
            workflow_id=workflow_id,
            name=template['name'].format(**parameters),
            description=template['description'].format(**parameters),
            current_phase="initialization",
            tasks=tasks,
            agent_assignments={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        return workflow
    
    async def get_workflow_metrics(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed workflow metrics"""
        workflow_state = await self.state_service.get_workflow_state(workflow_id)
        
        if not workflow_state:
            return {"error": "Workflow not found"}
        
        # Calculate metrics
        total_tasks = len(workflow_state.get('tasks', []))
        completed_tasks = len([t for t in workflow_state.get('tasks', []) if t.get('status') == 'completed'])
        
        start_time = datetime.fromisoformat(workflow_state.get('created_at', datetime.utcnow().isoformat()))
        current_time = datetime.utcnow()
        duration = (current_time - start_time).total_seconds()
        
        return {
            'workflow_id': workflow_id,
            'progress_percentage': (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'pending_tasks': total_tasks - completed_tasks,
            'duration_seconds': duration,
            'current_phase': workflow_state.get('current_phase'),
            'status': workflow_state.get('status'),
            'agent_assignments': workflow_state.get('agent_assignments', {})
        }
    
    async def get_workflow_history(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        history = await self.memory_service.retrieve_knowledge(
            'workflow_context',
            {'workflow_id': workflow_id}
        )
        return history
```

### Task 4.3: Backend API Integration

**File**: `app/api/workflows.py`

**Workflow API Endpoints**:
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

@router.post("/create")
async def create_workflow(task: Dict[str, Any], background_tasks: BackgroundTasks):
    """Create and start new workflow"""
    try:
        # Get Ultra Orchestrator
        orchestrator = await get_ultra_orchestrator()
        
        # Start workflow in background
        background_tasks.add_task(orchestrator.process_task, task)
        
        return {
            "status": "workflow_started",
            "message": "Workflow execution initiated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    try:
        orchestrator = await get_ultra_orchestrator()
        status = await orchestrator.get_workflow_status(workflow_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{workflow_id}/pause")
async def pause_workflow(workflow_id: str):
    """Pause workflow execution"""
    try:
        orchestrator = await get_ultra_orchestrator()
        await orchestrator.pause_workflow(workflow_id)
        return {"status": "paused"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{workflow_id}/resume")
async def resume_workflow(workflow_id: str):
    """Resume workflow execution"""
    try:
        orchestrator = await get_ultra_orchestrator()
        await orchestrator.resume_workflow(workflow_id)
        return {"status": "resumed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{workflow_id}/cancel")
async def cancel_workflow(workflow_id: str):
    """Cancel workflow execution"""
    try:
        orchestrator = await get_ultra_orchestrator()
        await orchestrator.cancel_workflow(workflow_id)
        return {"status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workflow_id}/metrics")
async def get_workflow_metrics(workflow_id: str):
    """Get detailed workflow metrics"""
    try:
        workflow_service = await get_workflow_service()
        metrics = await workflow_service.get_workflow_metrics(workflow_id)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_workflows():
    """List all workflows"""
    try:
        # Implementation to list all workflows
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_ultra_orchestrator():
    """Get Ultra Orchestrator instance"""
    # Implementation to get orchestrator from registry
    pass

async def get_workflow_service():
    """Get Workflow Service instance"""
    # Implementation to get workflow service
    pass
```

### Task 4.4: Frontend Workflow Interface

**File**: `frontend/components/workflow-dashboard.tsx`

**Workflow Dashboard Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';

interface WorkflowStatus {
  workflow_id: string;
  status: string;
  current_phase: string;
  progress: number;
  tasks_completed: number;
  total_tasks: number;
}

interface WorkflowDashboardProps {
  onWorkflowSelect: (workflowId: string) => void;
}

export const WorkflowDashboard: React.FC<WorkflowDashboardProps> = ({ onWorkflowSelect }) => {
  const [workflows, setWorkflows] = useState<WorkflowStatus[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  
  useEffect(() => {
    fetchWorkflows();
    const interval = setInterval(fetchWorkflows, 3000); // Update every 3 seconds
    return () => clearInterval(interval);
  }, []);
  
  const fetchWorkflows = async () => {
    try {
      const response = await fetch('/api/workflows');
      const data = await response.json();
      setWorkflows(data);
    } catch (error) {
      console.error('Failed to fetch workflows:', error);
    }
  };
  
  const createWorkflow = async () => {
    setIsCreating(true);
    try {
      const task = {
        type: 'development_request',
        name: 'New Development Task',
        description: 'User-initiated development workflow',
        requirements: [],
        tech_stack: {
          frontend: 'Next.js 15, React 19, Shadcn UI',
          backend: 'FastAPI, Python 3.11+'
        }
      };
      
      const response = await fetch('/api/workflows/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(task)
      });
      
      if (response.ok) {
        await fetchWorkflows();
      }
    } catch (error) {
      console.error('Failed to create workflow:', error);
    } finally {
      setIsCreating(false);
    }
  };
  
  const pauseWorkflow = async (workflowId: string) => {
    try {
      await fetch(`/api/workflows/${workflowId}/pause`, { method: 'POST' });
      await fetchWorkflows();
    } catch (error) {
      console.error('Failed to pause workflow:', error);
    }
  };
  
  const resumeWorkflow = async (workflowId: string) => {
    try {
      await fetch(`/api/workflows/${workflowId}/resume`, { method: 'POST' });
      await fetchWorkflows();
    } catch (error) {
      console.error('Failed to resume workflow:', error);
    }
  };
  
  const cancelWorkflow = async (workflowId: string) => {
    try {
      await fetch(`/api/workflows/${workflowId}/cancel`, { method: 'POST' });
      await fetchWorkflows();
    } catch (error) {
      console.error('Failed to cancel workflow:', error);
    }
  };
  
  const getStatusColor = (status: string) => {
    const colors = {
      active: 'bg-blue-500',
      paused: 'bg-yellow-500',
      completed: 'bg-green-500',
      cancelled: 'bg-red-500',
      error: 'bg-red-600'
    };
    return colors[status] || 'bg-gray-500';
  };
  
  return (
    <div className="workflow-dashboard space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Workflow Dashboard</h2>
        <Button 
          onClick={createWorkflow} 
          disabled={isCreating}
          className="bg-blue-600 hover:bg-blue-700"
        >
          {isCreating ? 'Creating...' : 'New Workflow'}
        </Button>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {workflows.map(workflow => (
          <Card 
            key={workflow.workflow_id}
            className={`cursor-pointer transition-all hover:shadow-lg ${
              selectedWorkflow === workflow.workflow_id ? 'ring-2 ring-blue-500' : ''
            }`}
            onClick={() => {
              setSelectedWorkflow(workflow.workflow_id);
              onWorkflowSelect(workflow.workflow_id);
            }}
          >
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start">
                <CardTitle className="text-lg">
                  {workflow.workflow_id.split('_').slice(-2).join('_')}
                </CardTitle>
                <Badge className={getStatusColor(workflow.status)}>
                  {workflow.status}
                </Badge>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Progress</span>
                  <span>{Math.round(workflow.progress * 100)}%</span>
                </div>
                <Progress value={workflow.progress * 100} className="h-2" />
              </div>
              
              <div className="text-sm text-gray-600">
                <div>Phase: {workflow.current_phase}</div>
                <div>Tasks: {workflow.tasks_completed}/{workflow.total_tasks}</div>
              </div>
              
              <div className="flex gap-2 mt-4">
                {workflow.status === 'active' && (
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={(e) => {
                      e.stopPropagation();
                      pauseWorkflow(workflow.workflow_id);
                    }}
                  >
                    Pause
                  </Button>
                )}
                
                {workflow.status === 'paused' && (
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={(e) => {
                      e.stopPropagation();
                      resumeWorkflow(workflow.workflow_id);
                    }}
                  >
                    Resume
                  </Button>
                )}
                
                {['active', 'paused'].includes(workflow.status) && (
                  <Button 
                    size="sm" 
                    variant="destructive"
                    onClick={(e) => {
                      e.stopPropagation();
                      cancelWorkflow(workflow.workflow_id);
                    }}
                  >
                    Cancel
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
      
      {workflows.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500 mb-4">No workflows found</p>
          <Button onClick={createWorkflow} disabled={isCreating}>
            Create Your First Workflow
          </Button>
        </div>
      )}
    </div>
  );
};
```

### Task 4.5: Workflow Visualization Component

**File**: `frontend/components/workflow-visualizer.tsx`

**Workflow Visualization**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface WorkflowVisualizerProps {
  workflowId: string;
}

interface TaskNode {
  task_id: string;
  task_type: string;
  description: string;
  status: string;
  dependencies: string[];
  assigned_agent?: string;
}

export const WorkflowVisualizer: React.FC<WorkflowVisualizerProps> = ({ workflowId }) => {
  const [workflowData, setWorkflowData] = useState<any>(null);
  const [tasks, setTasks] = useState<TaskNode[]>([]);
  
  useEffect(() => {
    if (workflowId) {
      fetchWorkflowData();
      const interval = setInterval(fetchWorkflowData, 5000);
      return () => clearInterval(interval);
    }
  }, [workflowId]);
  
  const fetchWorkflowData = async () => {
    try {
      const response = await fetch(`/api/workflows/${workflowId}/status`);
      const data = await response.json();
      setWorkflowData(data);
      setTasks(data.state?.tasks || []);
    } catch (error) {
      console.error('Failed to fetch workflow data:', error);
    }
  };
  
  const getTaskStatusColor = (status: string) => {
    const colors = {
      pending: 'bg-gray-400',
      in_progress: 'bg-blue-500',
      completed: 'bg-green-500',
      failed: 'bg-red-500',
      blocked: 'bg-orange-500'
    };
    return colors[status] || 'bg-gray-400';
  };
  
  const getPhaseColor = (phase: string) => {
    const colors = {
      initialization: 'bg-purple-500',
      requirements_analysis: 'bg-blue-500',
      research: 'bg-indigo-500',
      architecture_planning: 'bg-cyan-500',
      frontend_development: 'bg-green-500',
      backend_development: 'bg-yellow-500',
      integration: 'bg-orange-500',
      testing: 'bg-red-500',
      deployment: 'bg-pink-500',
      monitoring: 'bg-gray-500',
      completed: 'bg-emerald-500'
    };
    return colors[phase] || 'bg-gray-500';
  };
  
  if (!workflowData) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center">Loading workflow data...</div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <div className="workflow-visualizer space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Workflow: {workflowId}</span>
            <Badge className={getPhaseColor(workflowData.current_phase)}>
              {workflowData.current_phase}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {Math.round(workflowData.progress * 100)}%
              </div>
              <div className="text-sm text-gray-600">Progress</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {workflowData.tasks_completed}
              </div>
              <div className="text-sm text-gray-600">Completed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {workflowData.total_tasks - workflowData.tasks_completed}
              </div>
              <div className="text-sm text-gray-600">Remaining</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {workflowData.total_tasks}
              </div>
              <div className="text-sm text-gray-600">Total Tasks</div>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle>Task Flow</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {tasks.map((task, index) => (
              <div key={task.task_id} className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-bold ${
                    getTaskStatusColor(task.status)
                  }`}>
                    {index + 1}
                  </div>
                </div>
                
                <div className="flex-grow">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">{task.task_type.replace('_', ' ').toUpperCase()}</h4>
                    <Badge variant="outline">{task.status}</Badge>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{task.description}</p>
                  {task.assigned_agent && (
                    <p className="text-xs text-blue-600 mt-1">Assigned to: {task.assigned_agent}</p>
                  )}
                </div>
                
                {index < tasks.length - 1 && (
                  <div className="flex-shrink-0">
                    <div className="w-px h-8 bg-gray-300"></div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
```

## Testing Strategy

### Task 4.6: Orchestrator Testing

**Unit Tests**:
```python
# test_ultra_orchestrator.py
class TestUltraOrchestrator:
    async def test_workflow_creation(self):
        pass
    
    async def test_task_decomposition(self):
        pass
    
    async def test_agent_assignment(self):
        pass
    
    async def test_workflow_execution(self):
        pass
    
    async def test_error_handling(self):
        pass
```

**Integration Tests**:
- End-to-end workflow execution
- Agent coordination and communication
- State persistence and recovery
- Frontend workflow management

## Validation Criteria

### Backend Validation
- [ ] Ultra Orchestrator processes tasks correctly
- [ ] Workflow decomposition works accurately
- [ ] Agent assignment and coordination functional
- [ ] LangGraph workflow execution successful
- [ ] State management and persistence working
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Workflow dashboard displays real-time data
- [ ] Workflow creation and management functional
- [ ] Task visualization shows correct status
- [ ] Control buttons (pause/resume/cancel) work
- [ ] Real-time updates reflect backend changes

### Integration Validation
- [ ] Complete workflow execution from start to finish
- [ ] Agent coordination through orchestrator
- [ ] Frontend reflects backend workflow state
- [ ] Error handling and recovery mechanisms
- [ ] Performance under multiple concurrent workflows

## Human Testing Scenarios

1. **Simple Workflow Test**: Create a basic development workflow and monitor execution
2. **Complex Workflow Test**: Create multi-phase workflow with dependencies
3. **Workflow Control Test**: Pause, resume, and cancel workflows
4. **Error Recovery Test**: Trigger errors and verify recovery mechanisms
5. **Concurrent Workflows Test**: Run multiple workflows simultaneously
6. **Frontend Integration Test**: Use dashboard to manage workflows end-to-end

## Next Steps

After successful validation of the Ultra Orchestrator, proceed to **05-research-agent-enhancement.md** for implementing the enhanced Research Agent with advanced search capabilities and knowledge synthesis.

---

**Dependencies**: This phase requires the agent framework from Phase 3 and core services from Phase 2 to be fully functional.