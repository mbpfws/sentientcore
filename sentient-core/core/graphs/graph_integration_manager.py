"""Graph Integration Manager for Multi-Agent RAG System.

This module provides a unified interface for managing and coordinating all specialized graphs
in the system, including orchestrator, research, architecture, frontend/backend development,
build execution, and knowledge synthesis graphs.
"""

from typing import Dict, Any, List, Optional, Type, Union
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from datetime import datetime
import json
import asyncio
from enum import Enum

from .orchestrator_graph import create_orchestrator_graph, OrchestratorState
from .research_graph import create_research_graph, ResearchState
from .project_architecture_graph import create_project_architecture_graph, ProjectArchitectureState
from .frontend_development_graph import create_frontend_development_graph, FrontendDevelopmentState
from .backend_development_graph import create_backend_development_graph, BackendDevelopmentState
from .build_execution_graph import create_build_execution_graph, BuildExecutionState
from .knowledge_synthesis_graph import create_knowledge_synthesis_graph, KnowledgeSynthesisState

from ..services.llm_service import EnhancedLLMService
from ..services.memory_service import MemoryService
from ..services.state_service import StateService


class GraphType(Enum):
    """Enumeration of available graph types."""
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    PROJECT_ARCHITECTURE = "project_architecture"
    FRONTEND_DEVELOPMENT = "frontend_development"
    BACKEND_DEVELOPMENT = "backend_development"
    BUILD_EXECUTION = "build_execution"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"


class GraphExecutionStatus(Enum):
    """Enumeration of graph execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class GraphExecutionResult(BaseModel):
    """Result of graph execution."""
    graph_type: GraphType
    execution_id: str
    status: GraphExecutionStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    final_state: Optional[Dict[str, Any]] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphIntegrationState(BaseModel):
    """State for graph integration management."""
    session_id: str
    active_graphs: Dict[str, GraphExecutionResult] = Field(default_factory=dict)
    execution_history: List[GraphExecutionResult] = Field(default_factory=list)
    graph_dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    shared_context: Dict[str, Any] = Field(default_factory=dict)
    global_state: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class GraphIntegrationManager:
    """Manager for coordinating multiple specialized graphs."""
    
    def __init__(
        self,
        llm_service: EnhancedLLMService,
        memory_service: MemoryService,
        state_service: StateService
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.state_service = state_service
        
        # Initialize graphs
        self.graphs: Dict[GraphType, StateGraph] = {}
        self.state_types: Dict[GraphType, Type] = {
            GraphType.ORCHESTRATOR: OrchestratorState,
            GraphType.RESEARCH: ResearchState,
            GraphType.PROJECT_ARCHITECTURE: ProjectArchitectureState,
            GraphType.FRONTEND_DEVELOPMENT: FrontendDevelopmentState,
            GraphType.BACKEND_DEVELOPMENT: BackendDevelopmentState,
            GraphType.BUILD_EXECUTION: BuildExecutionState,
            GraphType.KNOWLEDGE_SYNTHESIS: KnowledgeSynthesisState
        }
        
        # Integration state
        self.integration_states: Dict[str, GraphIntegrationState] = {}
        
        # Initialize all graphs
        self._initialize_graphs()
    
    def _initialize_graphs(self) -> None:
        """Initialize all specialized graphs."""
        try:
            self.graphs[GraphType.ORCHESTRATOR] = create_orchestrator_graph(
                self.llm_service, self.memory_service, self.state_service
            )
            
            self.graphs[GraphType.RESEARCH] = create_research_graph(
                self.llm_service, self.memory_service, self.state_service
            )
            
            self.graphs[GraphType.PROJECT_ARCHITECTURE] = create_project_architecture_graph(
                self.llm_service, self.memory_service, self.state_service
            )
            
            self.graphs[GraphType.FRONTEND_DEVELOPMENT] = create_frontend_development_graph(
                self.llm_service, self.memory_service, self.state_service
            )
            
            self.graphs[GraphType.BACKEND_DEVELOPMENT] = create_backend_development_graph(
                self.llm_service, self.memory_service, self.state_service
            )
            
            self.graphs[GraphType.BUILD_EXECUTION] = create_build_execution_graph(
                self.llm_service, self.memory_service, self.state_service
            )
            
            self.graphs[GraphType.KNOWLEDGE_SYNTHESIS] = create_knowledge_synthesis_graph(
                self.llm_service, self.memory_service, self.state_service
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize graphs: {str(e)}")
    
    def create_session(self, session_id: str) -> GraphIntegrationState:
        """Create a new integration session."""
        if session_id in self.integration_states:
            return self.integration_states[session_id]
        
        state = GraphIntegrationState(session_id=session_id)
        self.integration_states[session_id] = state
        
        # Store in memory service
        self.memory_service.store_conversation_memory(
            session_id=session_id,
            message_type="session_created",
            content={"session_id": session_id},
            metadata={"timestamp": state.created_at.isoformat()}
        )
        
        return state
    
    def get_session(self, session_id: str) -> Optional[GraphIntegrationState]:
        """Get an existing integration session."""
        return self.integration_states.get(session_id)
    
    def execute_graph(
        self,
        session_id: str,
        graph_type: GraphType,
        initial_state: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> GraphExecutionResult:
        """Execute a specific graph with given initial state."""
        if session_id not in self.integration_states:
            self.create_session(session_id)
        
        integration_state = self.integration_states[session_id]
        
        if execution_id is None:
            execution_id = f"{graph_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create execution result
        result = GraphExecutionResult(
            graph_type=graph_type,
            execution_id=execution_id,
            status=GraphExecutionStatus.PENDING,
            start_time=datetime.now()
        )
        
        try:
            # Update status to running
            result.status = GraphExecutionStatus.RUNNING
            integration_state.active_graphs[execution_id] = result
            
            # Get the appropriate graph and state type
            graph = self.graphs[graph_type]
            state_type = self.state_types[graph_type]
            
            # Create initial state with session context
            enhanced_initial_state = {
                **initial_state,
                "session_id": session_id,
                **integration_state.shared_context
            }
            
            # Convert to proper state object
            if hasattr(state_type, 'model_validate'):
                state_obj = state_type.model_validate(enhanced_initial_state)
            else:
                state_obj = state_type(**enhanced_initial_state)
            
            # Execute the graph
            final_state = graph.invoke(state_obj)
            
            # Update result
            result.status = GraphExecutionStatus.COMPLETED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.final_state = final_state.model_dump() if hasattr(final_state, 'model_dump') else dict(final_state)
            
            # Extract outputs based on graph type
            result.outputs = self._extract_graph_outputs(graph_type, final_state)
            
            # Update shared context with relevant outputs
            self._update_shared_context(integration_state, graph_type, result.outputs)
            
        except Exception as e:
            result.status = GraphExecutionStatus.FAILED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            result.errors.append({
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "phase": "execution"
            })
        
        finally:
            # Move from active to history
            if execution_id in integration_state.active_graphs:
                del integration_state.active_graphs[execution_id]
            integration_state.execution_history.append(result)
            integration_state.updated_at = datetime.now()
            
            # Store execution result
            self.memory_service.store_conversation_memory(
                session_id=session_id,
                message_type="graph_execution",
                content=result.model_dump(),
                metadata={
                    "graph_type": graph_type.value,
                    "execution_id": execution_id,
                    "status": result.status.value
                }
            )
        
        return result
    
    async def execute_graph_async(
        self,
        session_id: str,
        graph_type: GraphType,
        initial_state: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> GraphExecutionResult:
        """Execute a graph asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.execute_graph,
            session_id,
            graph_type,
            initial_state,
            execution_id
        )
    
    def execute_workflow(
        self,
        session_id: str,
        workflow_definition: List[Dict[str, Any]]
    ) -> List[GraphExecutionResult]:
        """Execute a workflow consisting of multiple graphs."""
        results = []
        
        for step in workflow_definition:
            graph_type = GraphType(step["graph_type"])
            initial_state = step.get("initial_state", {})
            execution_id = step.get("execution_id")
            
            # Check dependencies
            dependencies = step.get("dependencies", [])
            if dependencies:
                self._wait_for_dependencies(session_id, dependencies)
            
            # Execute the graph
            result = self.execute_graph(
                session_id=session_id,
                graph_type=graph_type,
                initial_state=initial_state,
                execution_id=execution_id
            )
            
            results.append(result)
            
            # Stop workflow if execution failed
            if result.status == GraphExecutionStatus.FAILED:
                break
        
        return results
    
    def _extract_graph_outputs(self, graph_type: GraphType, final_state: Any) -> Dict[str, Any]:
        """Extract relevant outputs from graph execution."""
        outputs = {}
        
        try:
            if hasattr(final_state, 'model_dump'):
                state_dict = final_state.model_dump()
            else:
                state_dict = dict(final_state)
            
            if graph_type == GraphType.ORCHESTRATOR:
                outputs = {
                    "conversation_context": state_dict.get("conversation_context", {}),
                    "task_orchestration": state_dict.get("task_orchestration", {}),
                    "agent_coordination": state_dict.get("agent_coordination", {}),
                    "system_control": state_dict.get("system_control", {})
                }
            
            elif graph_type == GraphType.RESEARCH:
                outputs = {
                    "research_results": state_dict.get("research_results", {}),
                    "synthesized_report": state_dict.get("synthesized_report", {}),
                    "knowledge_base": state_dict.get("knowledge_base", {})
                }
            
            elif graph_type == GraphType.PROJECT_ARCHITECTURE:
                outputs = {
                    "architecture_design": state_dict.get("architecture_design", {}),
                    "component_specifications": state_dict.get("component_specifications", {}),
                    "integration_plan": state_dict.get("integration_plan", {}),
                    "technical_documentation": state_dict.get("technical_documentation", {})
                }
            
            elif graph_type == GraphType.FRONTEND_DEVELOPMENT:
                outputs = {
                    "ui_components": state_dict.get("ui_components", {}),
                    "frontend_code": state_dict.get("frontend_code", {}),
                    "user_interface": state_dict.get("user_interface", {}),
                    "testing_results": state_dict.get("testing_results", {})
                }
            
            elif graph_type == GraphType.BACKEND_DEVELOPMENT:
                outputs = {
                    "api_implementation": state_dict.get("api_implementation", {}),
                    "database_schema": state_dict.get("database_schema", {}),
                    "backend_services": state_dict.get("backend_services", {}),
                    "deployment_config": state_dict.get("deployment_config", {})
                }
            
            elif graph_type == GraphType.BUILD_EXECUTION:
                outputs = {
                    "build_artifacts": state_dict.get("build_artifacts", {}),
                    "deployment_package": state_dict.get("deployment_package", {}),
                    "test_results": state_dict.get("test_results", {}),
                    "performance_metrics": state_dict.get("performance_metrics", {})
                }
            
            elif graph_type == GraphType.KNOWLEDGE_SYNTHESIS:
                outputs = {
                    "final_knowledge_artifacts": state_dict.get("final_knowledge_artifacts", {}),
                    "synthesis_summary": state_dict.get("synthesis_summary", {}),
                    "knowledge_base_entries": state_dict.get("knowledge_base_entries", [])
                }
            
        except Exception as e:
            outputs["extraction_error"] = str(e)
        
        return outputs
    
    def _update_shared_context(
        self,
        integration_state: GraphIntegrationState,
        graph_type: GraphType,
        outputs: Dict[str, Any]
    ) -> None:
        """Update shared context with graph outputs."""
        context_key = f"{graph_type.value}_outputs"
        integration_state.shared_context[context_key] = outputs
        
        # Update global state with key information
        if graph_type == GraphType.ORCHESTRATOR:
            integration_state.global_state.update({
                "current_task": outputs.get("task_orchestration", {}).get("current_task"),
                "system_status": outputs.get("system_control", {}).get("status")
            })
        
        elif graph_type == GraphType.PROJECT_ARCHITECTURE:
            integration_state.global_state.update({
                "architecture": outputs.get("architecture_design"),
                "components": outputs.get("component_specifications")
            })
        
        elif graph_type == GraphType.RESEARCH:
            integration_state.global_state.update({
                "research_data": outputs.get("research_results"),
                "knowledge_base": outputs.get("knowledge_base")
            })
    
    def _wait_for_dependencies(self, session_id: str, dependencies: List[str]) -> None:
        """Wait for dependent graph executions to complete."""
        integration_state = self.integration_states.get(session_id)
        if not integration_state:
            return
        
        # Check if all dependencies are completed
        for dep_id in dependencies:
            # Look for the dependency in execution history
            dep_found = False
            for result in integration_state.execution_history:
                if result.execution_id == dep_id:
                    if result.status != GraphExecutionStatus.COMPLETED:
                        raise RuntimeError(f"Dependency {dep_id} failed or is not completed")
                    dep_found = True
                    break
            
            if not dep_found:
                raise RuntimeError(f"Dependency {dep_id} not found in execution history")
    
    def get_execution_status(self, session_id: str, execution_id: str) -> Optional[GraphExecutionResult]:
        """Get the status of a specific execution."""
        integration_state = self.integration_states.get(session_id)
        if not integration_state:
            return None
        
        # Check active graphs first
        if execution_id in integration_state.active_graphs:
            return integration_state.active_graphs[execution_id]
        
        # Check execution history
        for result in integration_state.execution_history:
            if result.execution_id == execution_id:
                return result
        
        return None
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of the integration session."""
        integration_state = self.integration_states.get(session_id)
        if not integration_state:
            return {}
        
        active_count = len(integration_state.active_graphs)
        completed_count = len([r for r in integration_state.execution_history if r.status == GraphExecutionStatus.COMPLETED])
        failed_count = len([r for r in integration_state.execution_history if r.status == GraphExecutionStatus.FAILED])
        
        return {
            "session_id": session_id,
            "created_at": integration_state.created_at.isoformat(),
            "updated_at": integration_state.updated_at.isoformat(),
            "active_graphs": active_count,
            "completed_executions": completed_count,
            "failed_executions": failed_count,
            "total_executions": len(integration_state.execution_history),
            "shared_context_keys": list(integration_state.shared_context.keys()),
            "global_state": integration_state.global_state
        }
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up a session and its resources."""
        if session_id in self.integration_states:
            del self.integration_states[session_id]
            
            # Store cleanup event
            self.memory_service.store_conversation_memory(
                session_id=session_id,
                message_type="session_cleanup",
                content={"session_id": session_id},
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
            return True
        return False
    
    def get_available_graphs(self) -> List[Dict[str, Any]]:
        """Get information about available graphs."""
        return [
            {
                "type": graph_type.value,
                "name": graph_type.value.replace("_", " ").title(),
                "state_type": state_type.__name__,
                "available": graph_type in self.graphs
            }
            for graph_type, state_type in self.state_types.items()
        ]
    
    def create_end_to_end_workflow(self, user_request: str) -> List[Dict[str, Any]]:
        """Create an end-to-end workflow based on user request."""
        workflow = [
            {
                "graph_type": GraphType.ORCHESTRATOR.value,
                "initial_state": {
                    "conversation_context": {
                        "user_input": user_request,
                        "conversation_history": []
                    }
                },
                "execution_id": "orchestrator_initial"
            },
            {
                "graph_type": GraphType.RESEARCH.value,
                "initial_state": {
                    "research_query": user_request,
                    "search_parameters": {"max_results": 10}
                },
                "execution_id": "research_phase",
                "dependencies": ["orchestrator_initial"]
            },
            {
                "graph_type": GraphType.PROJECT_ARCHITECTURE.value,
                "initial_state": {
                    "requirements": {"user_request": user_request}
                },
                "execution_id": "architecture_phase",
                "dependencies": ["research_phase"]
            },
            {
                "graph_type": GraphType.FRONTEND_DEVELOPMENT.value,
                "initial_state": {
                    "requirements": {"user_request": user_request}
                },
                "execution_id": "frontend_phase",
                "dependencies": ["architecture_phase"]
            },
            {
                "graph_type": GraphType.BACKEND_DEVELOPMENT.value,
                "initial_state": {
                    "requirements": {"user_request": user_request}
                },
                "execution_id": "backend_phase",
                "dependencies": ["architecture_phase"]
            },
            {
                "graph_type": GraphType.BUILD_EXECUTION.value,
                "initial_state": {
                    "build_requirements": {"user_request": user_request}
                },
                "execution_id": "build_phase",
                "dependencies": ["frontend_phase", "backend_phase"]
            },
            {
                "graph_type": GraphType.KNOWLEDGE_SYNTHESIS.value,
                "initial_state": {
                    "user_requirements": {"user_request": user_request}
                },
                "execution_id": "synthesis_phase",
                "dependencies": ["build_phase"]
            }
        ]
        
        return workflow


def create_graph_integration_manager(
    llm_service: EnhancedLLMService,
    memory_service: MemoryService,
    state_service: StateService
) -> GraphIntegrationManager:
    """Factory function to create a graph integration manager."""
    return GraphIntegrationManager(llm_service, memory_service, state_service)