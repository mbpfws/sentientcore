"""Orchestrator Graph Implementation

This module implements the main orchestrator graph that manages overall system flow,
user interaction, and high-level task delegation as specified in the improvement-rag.md document.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from langgraph import StateGraph, END
from langgraph.graph import CompiledGraph
from pydantic import BaseModel, Field

from ..models import AppState, TaskStatus, AgentType, SessionState
from ..agents.ultra_orchestrator import UltraOrchestrator
from ..agents.monitoring_agent import MonitoringAgent
from ..services.llm_service import EnhancedLLMService
from ..utils.logger import get_logger

logger = get_logger(__name__)

class OrchestratorState(BaseModel):
    """State model for the orchestrator graph."""
    
    # Core conversation state
    user_input: str = ""
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    current_language: str = "en"
    
    # Task orchestration state
    orchestration_phase: Literal[
        "initial_engagement",
        "clarifying_request", 
        "research_and_planning",
        "prototyping",
        "development",
        "monitoring",
        "completion"
    ] = "initial_engagement"
    
    # Agent coordination
    active_agents: List[AgentType] = Field(default_factory=list)
    required_subgraphs: List[str] = Field(default_factory=list)
    
    # Request refinement
    request_clarity_score: float = 0.0
    refinement_iterations: int = 0
    max_refinement_iterations: int = 5
    
    # Task delegation
    delegated_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    task_dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    
    # System control
    should_use_external_search: bool = False
    fallback_model_required: bool = False
    
    # Output and results
    orchestrator_decision: str = ""
    next_action: str = ""
    artifacts_generated: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Error handling
    error_count: int = 0
    max_errors: int = 3
    last_error: Optional[str] = None

class OrchestratorGraphBuilder:
    """Builder class for creating the orchestrator graph."""
    
    def __init__(self, llm_service: EnhancedLLMService):
        self.llm_service = llm_service
        self.ultra_orchestrator = UltraOrchestrator(llm_service)
        self.monitoring_agent = MonitoringAgent(llm_service)
        
    def create_graph(self) -> CompiledGraph:
        """Create and compile the orchestrator graph."""
        
        # Create the state graph
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("initial_engagement", self._initial_engagement_node)
        workflow.add_node("clarify_request", self._clarify_request_node)
        workflow.add_node("analyze_requirements", self._analyze_requirements_node)
        workflow.add_node("delegate_tasks", self._delegate_tasks_node)
        workflow.add_node("coordinate_agents", self._coordinate_agents_node)
        workflow.add_node("monitor_progress", self._monitor_progress_node)
        workflow.add_node("synthesize_results", self._synthesize_results_node)
        workflow.add_node("handle_user_feedback", self._handle_user_feedback_node)
        workflow.add_node("error_recovery", self._error_recovery_node)
        
        # Set entry point
        workflow.set_entry_point("initial_engagement")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "initial_engagement",
            self._route_from_initial_engagement,
            {
                "clarify": "clarify_request",
                "analyze": "analyze_requirements",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "clarify_request",
            self._route_from_clarification,
            {
                "continue_clarifying": "clarify_request",
                "analyze": "analyze_requirements",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_requirements",
            self._route_from_analysis,
            {
                "delegate": "delegate_tasks",
                "clarify_more": "clarify_request",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "delegate_tasks",
            self._route_from_delegation,
            {
                "coordinate": "coordinate_agents",
                "monitor": "monitor_progress",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "coordinate_agents",
            self._route_from_coordination,
            {
                "monitor": "monitor_progress",
                "delegate_more": "delegate_tasks",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "monitor_progress",
            self._route_from_monitoring,
            {
                "synthesize": "synthesize_results",
                "coordinate": "coordinate_agents",
                "await_feedback": "handle_user_feedback",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "synthesize_results",
            self._route_from_synthesis,
            {
                "complete": END,
                "await_feedback": "handle_user_feedback",
                "continue_work": "coordinate_agents",
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "handle_user_feedback",
            self._route_from_feedback,
            {
                "clarify": "clarify_request",
                "delegate": "delegate_tasks",
                "coordinate": "coordinate_agents",
                "complete": END,
                "error": "error_recovery"
            }
        )
        
        workflow.add_conditional_edges(
            "error_recovery",
            self._route_from_error_recovery,
            {
                "retry": "initial_engagement",
                "escalate": END,
                "continue": "coordinate_agents"
            }
        )
        
        return workflow.compile()
    
    async def _initial_engagement_node(self, state: OrchestratorState) -> OrchestratorState:
        """Handle initial user engagement and language detection."""
        try:
            logger.info("Starting initial engagement phase")
            
            # Detect language and set conversation context
            language_result = await self.ultra_orchestrator.detect_language(state.user_input)
            state.current_language = language_result.get("language", "en")
            
            # Analyze initial request clarity
            clarity_analysis = await self.ultra_orchestrator.analyze_request_clarity(
                state.user_input, state.current_language
            )
            
            state.request_clarity_score = clarity_analysis.get("clarity_score", 0.0)
            state.orchestration_phase = "initial_engagement"
            
            # Add to conversation history
            state.conversation_history.append({
                "role": "user",
                "content": state.user_input,
                "timestamp": self._get_timestamp(),
                "language": state.current_language
            })
            
            logger.info(f"Initial engagement completed. Clarity score: {state.request_clarity_score}")
            
        except Exception as e:
            logger.error(f"Error in initial engagement: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _clarify_request_node(self, state: OrchestratorState) -> OrchestratorState:
        """Clarify user request through iterative refinement."""
        try:
            logger.info("Starting request clarification")
            
            state.orchestration_phase = "clarifying_request"
            state.refinement_iterations += 1
            
            # Generate clarification questions
            clarification_result = await self.ultra_orchestrator.generate_clarification_questions(
                state.user_input,
                state.conversation_history,
                state.current_language
            )
            
            # Update state with clarification
            state.orchestrator_decision = clarification_result.get("clarification_response", "")
            
            # Add orchestrator response to conversation
            state.conversation_history.append({
                "role": "assistant",
                "content": state.orchestrator_decision,
                "timestamp": self._get_timestamp(),
                "phase": "clarification"
            })
            
            logger.info(f"Clarification iteration {state.refinement_iterations} completed")
            
        except Exception as e:
            logger.error(f"Error in request clarification: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _analyze_requirements_node(self, state: OrchestratorState) -> OrchestratorState:
        """Analyze requirements and determine needed agents and subgraphs."""
        try:
            logger.info("Starting requirements analysis")
            
            state.orchestration_phase = "research_and_planning"
            
            # Analyze requirements and determine strategy
            analysis_result = await self.ultra_orchestrator.analyze_requirements(
                state.user_input,
                state.conversation_history
            )
            
            # Determine required agents and subgraphs
            state.active_agents = analysis_result.get("required_agents", [])
            state.required_subgraphs = analysis_result.get("required_subgraphs", [])
            state.should_use_external_search = analysis_result.get("use_external_search", False)
            
            # Create task breakdown
            state.delegated_tasks = analysis_result.get("task_breakdown", [])
            state.task_dependencies = analysis_result.get("task_dependencies", {})
            
            logger.info(f"Requirements analysis completed. Agents: {state.active_agents}")
            
        except Exception as e:
            logger.error(f"Error in requirements analysis: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _delegate_tasks_node(self, state: OrchestratorState) -> OrchestratorState:
        """Delegate tasks to appropriate agents and subgraphs."""
        try:
            logger.info("Starting task delegation")
            
            # Delegate tasks to agents
            delegation_result = await self.ultra_orchestrator.delegate_tasks(
                state.delegated_tasks,
                state.active_agents,
                state.required_subgraphs
            )
            
            # Update task status
            for task in state.delegated_tasks:
                task["status"] = "delegated"
                task["assigned_agent"] = delegation_result.get(task["id"], {}).get("agent")
                task["assigned_subgraph"] = delegation_result.get(task["id"], {}).get("subgraph")
            
            state.next_action = "coordinate_agents"
            
            logger.info("Task delegation completed")
            
        except Exception as e:
            logger.error(f"Error in task delegation: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _coordinate_agents_node(self, state: OrchestratorState) -> OrchestratorState:
        """Coordinate multiple agents and manage inter-agent communication."""
        try:
            logger.info("Starting agent coordination")
            
            # Coordinate agent activities
            coordination_result = await self.ultra_orchestrator.coordinate_agents(
                state.active_agents,
                state.delegated_tasks,
                state.task_dependencies
            )
            
            # Update task progress
            for task_update in coordination_result.get("task_updates", []):
                for task in state.delegated_tasks:
                    if task["id"] == task_update["task_id"]:
                        task.update(task_update)
            
            state.next_action = "monitor_progress"
            
            logger.info("Agent coordination completed")
            
        except Exception as e:
            logger.error(f"Error in agent coordination: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _monitor_progress_node(self, state: OrchestratorState) -> OrchestratorState:
        """Monitor progress of all active tasks and agents."""
        try:
            logger.info("Starting progress monitoring")
            
            state.orchestration_phase = "monitoring"
            
            # Monitor task progress
            monitoring_result = await self.monitoring_agent.monitor_system_progress(
                state.delegated_tasks,
                state.active_agents
            )
            
            # Update state with monitoring results
            state.next_action = monitoring_result.get("recommended_action", "continue")
            
            # Check for completed tasks
            completed_tasks = [task for task in state.delegated_tasks if task.get("status") == "completed"]
            if completed_tasks:
                state.artifacts_generated.extend(
                    monitoring_result.get("generated_artifacts", [])
                )
            
            logger.info(f"Progress monitoring completed. Next action: {state.next_action}")
            
        except Exception as e:
            logger.error(f"Error in progress monitoring: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _synthesize_results_node(self, state: OrchestratorState) -> OrchestratorState:
        """Synthesize results from all completed tasks."""
        try:
            logger.info("Starting result synthesis")
            
            state.orchestration_phase = "completion"
            
            # Synthesize results
            synthesis_result = await self.ultra_orchestrator.synthesize_results(
                state.delegated_tasks,
                state.artifacts_generated,
                state.conversation_history
            )
            
            # Generate final response
            state.orchestrator_decision = synthesis_result.get("final_response", "")
            
            # Add final response to conversation
            state.conversation_history.append({
                "role": "assistant",
                "content": state.orchestrator_decision,
                "timestamp": self._get_timestamp(),
                "phase": "completion",
                "artifacts": state.artifacts_generated
            })
            
            logger.info("Result synthesis completed")
            
        except Exception as e:
            logger.error(f"Error in result synthesis: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _handle_user_feedback_node(self, state: OrchestratorState) -> OrchestratorState:
        """Handle user feedback and determine next steps."""
        try:
            logger.info("Handling user feedback")
            
            # Process user feedback
            feedback_result = await self.ultra_orchestrator.process_user_feedback(
                state.user_input,
                state.conversation_history,
                state.artifacts_generated
            )
            
            # Determine next action based on feedback
            state.next_action = feedback_result.get("next_action", "complete")
            
            # Update tasks if needed
            if feedback_result.get("task_modifications"):
                for modification in feedback_result["task_modifications"]:
                    for task in state.delegated_tasks:
                        if task["id"] == modification["task_id"]:
                            task.update(modification["updates"])
            
            logger.info(f"User feedback processed. Next action: {state.next_action}")
            
        except Exception as e:
            logger.error(f"Error handling user feedback: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            
        return state
    
    async def _error_recovery_node(self, state: OrchestratorState) -> OrchestratorState:
        """Handle errors and attempt recovery."""
        try:
            logger.info(f"Starting error recovery. Error count: {state.error_count}")
            
            if state.error_count >= state.max_errors:
                state.next_action = "escalate"
                state.orchestrator_decision = f"Maximum error limit reached. Last error: {state.last_error}"
            else:
                # Attempt recovery
                recovery_result = await self.ultra_orchestrator.attempt_error_recovery(
                    state.last_error,
                    state.orchestration_phase
                )
                
                state.next_action = recovery_result.get("recovery_action", "retry")
                
                # Reset error if recovery successful
                if recovery_result.get("recovery_successful", False):
                    state.error_count = 0
                    state.last_error = None
            
            logger.info(f"Error recovery completed. Next action: {state.next_action}")
            
        except Exception as e:
            logger.error(f"Error in error recovery: {str(e)}")
            state.error_count += 1
            state.last_error = str(e)
            state.next_action = "escalate"
            
        return state
    
    def _route_from_initial_engagement(self, state: OrchestratorState) -> str:
        """Route from initial engagement based on request clarity."""
        if state.error_count > 0:
            return "error"
        elif state.request_clarity_score < 0.6:
            return "clarify"
        else:
            return "analyze"
    
    def _route_from_clarification(self, state: OrchestratorState) -> str:
        """Route from clarification based on clarity improvement."""
        if state.error_count > 0:
            return "error"
        elif (state.refinement_iterations < state.max_refinement_iterations and 
              state.request_clarity_score < 0.7):
            return "continue_clarifying"
        else:
            return "analyze"
    
    def _route_from_analysis(self, state: OrchestratorState) -> str:
        """Route from analysis based on requirements clarity."""
        if state.error_count > 0:
            return "error"
        elif not state.active_agents or not state.delegated_tasks:
            return "clarify_more"
        else:
            return "delegate"
    
    def _route_from_delegation(self, state: OrchestratorState) -> str:
        """Route from delegation based on task complexity."""
        if state.error_count > 0:
            return "error"
        elif len(state.active_agents) > 1:
            return "coordinate"
        else:
            return "monitor"
    
    def _route_from_coordination(self, state: OrchestratorState) -> str:
        """Route from coordination based on agent status."""
        if state.error_count > 0:
            return "error"
        elif state.next_action == "delegate_more":
            return "delegate_more"
        else:
            return "monitor"
    
    def _route_from_monitoring(self, state: OrchestratorState) -> str:
        """Route from monitoring based on task completion status."""
        if state.error_count > 0:
            return "error"
        elif state.next_action == "synthesize":
            return "synthesize"
        elif state.next_action == "await_feedback":
            return "await_feedback"
        else:
            return "coordinate"
    
    def _route_from_synthesis(self, state: OrchestratorState) -> str:
        """Route from synthesis based on completion status."""
        if state.error_count > 0:
            return "error"
        elif state.next_action == "complete":
            return "complete"
        elif state.next_action == "await_feedback":
            return "await_feedback"
        else:
            return "continue_work"
    
    def _route_from_feedback(self, state: OrchestratorState) -> str:
        """Route from feedback based on user input."""
        if state.error_count > 0:
            return "error"
        elif state.next_action == "clarify":
            return "clarify"
        elif state.next_action == "delegate":
            return "delegate"
        elif state.next_action == "coordinate":
            return "coordinate"
        elif state.next_action == "complete":
            return "complete"
        else:
            return "coordinate"
    
    def _route_from_error_recovery(self, state: OrchestratorState) -> str:
        """Route from error recovery based on recovery success."""
        if state.next_action == "escalate":
            return "escalate"
        elif state.next_action == "retry":
            return "retry"
        else:
            return "continue"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

# Factory function for creating orchestrator graph
def create_orchestrator_graph(llm_service: EnhancedLLMService) -> CompiledGraph:
    """Create and return a compiled orchestrator graph."""
    builder = OrchestratorGraphBuilder(llm_service)
    return builder.create_graph()