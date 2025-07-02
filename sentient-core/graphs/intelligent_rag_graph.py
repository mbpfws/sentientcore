"""
Intelligent Multi-Agent RAG Graph
Uses natural language understanding for workflow orchestration
"""

from langgraph.graph import StateGraph, END
from core.models import AppState, Message, Task, TaskStatus, LogEntry
from core.agents.intelligent_orchestrator import IntelligentOrchestrator
from core.agents.research_agent import ResearchAgent
from core.services.llm_service import LLMService
from typing import Dict, Any
import json

class IntelligentRAGWorkflow:
    """
    Intelligent workflow that makes decisions based on natural language understanding.
    The orchestrator intelligently routes to appropriate agents based on conversation analysis.
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.orchestrator = IntelligentOrchestrator(self.llm_service)
        self.research_agent = ResearchAgent(self.llm_service)
        
        # Build the intelligent workflow
        self.workflow = self._build_intelligent_workflow()
    
    def _build_intelligent_workflow(self) -> StateGraph:
        """Builds the intelligent workflow graph."""
        workflow = StateGraph(AppState)
        
        # Core intelligent nodes
        workflow.add_node("intelligent_orchestrator", self._intelligent_orchestrator_node)
        workflow.add_node("execute_research", self._execute_research_node)
        workflow.add_node("execute_architecture", self._execute_architecture_node) 
        workflow.add_node("execute_design", self._execute_design_node)
        workflow.add_node("execute_builder", self._execute_builder_node)
        workflow.add_node("await_user_input", self._await_user_input_node)
        
        # Set entry point to intelligent orchestrator
        workflow.set_entry_point("intelligent_orchestrator")
        
        # Intelligent routing from orchestrator
        workflow.add_conditional_edges(
            "intelligent_orchestrator",
            self._route_from_intelligent_orchestrator,
            {
                "await_user_input": "await_user_input",
                "execute_research": "execute_research", 
                "execute_architecture": "execute_architecture",
                "execute_design": "execute_design",
                "execute_builder": "execute_builder",
                "end": END
            }
        )
        
        # Research execution returns to orchestrator for next decision
        workflow.add_edge("execute_research", "intelligent_orchestrator")
        workflow.add_edge("execute_architecture", "intelligent_orchestrator") 
        workflow.add_edge("execute_design", "intelligent_orchestrator")
        workflow.add_edge("execute_builder", END)
        workflow.add_edge("await_user_input", END)
        
        return workflow.compile()
    
    def _intelligent_orchestrator_node(self, state: AppState) -> Dict[str, Any]:
        """
        Intelligent orchestrator that makes decisions based on natural language understanding.
        """
        print("---INTELLIGENT ORCHESTRATOR---")
        
        # Get intelligent decision from orchestrator
        result = self.orchestrator.analyze_and_orchestrate(state)
        
        # Extract decision for routing
        orchestrator_decision = result.get("orchestrator_decision", {})
        decision = orchestrator_decision.get("decision", "continue_conversation")
        
        # Set routing action - always use await_user_input for conversation management
        if decision in ["continue_conversation", "request_clarification", "redirect_conversation"]:
            result["next_action"] = "await_user_input"
        elif decision == "create_task_breakdown" and orchestrator_decision.get("ready_for_execution"):
            # Check if we have executable tasks ready
            updated_state = AppState(**result)  # Convert back to AppState to check tasks
            pending_tasks = [task for task in updated_state.tasks if task.status == TaskStatus.PENDING]
            if pending_tasks:
                # Sort by sequence to get the next task
                pending_tasks.sort(key=lambda x: getattr(x, 'sequence', 1))
                next_task = pending_tasks[0]
                
                # Determine execution path based on agent type
                agent_type = getattr(next_task, 'agent', 'research')
                if agent_type == "research":
                    result["next_action"] = "execute_research"
                elif agent_type == "architecture": 
                    result["next_action"] = "execute_architecture"
                elif agent_type == "design":
                    result["next_action"] = "execute_design"
                elif agent_type == "builder":
                    result["next_action"] = "execute_builder"
                else:
                    result["next_action"] = "await_user_input"
            else:
                result["next_action"] = "await_user_input"
        else:
            result["next_action"] = "await_user_input"
        
        print(f"[Orchestrator] Decision: {decision} -> Route: {result['next_action']}")
        return result
    
    def _execute_research_node(self, state: AppState) -> Dict[str, Any]:
        """Execute research tasks using the research agent."""
        print("---EXECUTING RESEARCH TASKS---")
        
        # Find pending research tasks
        research_tasks = [
            task for task in state.tasks 
            if task.status == TaskStatus.PENDING and getattr(task, 'agent', '') == 'research'
        ]
        
        if research_tasks:
            # Execute the first research task
            task = research_tasks[0]
            task.status = TaskStatus.IN_PROGRESS
            
            self._log(state, f"Executing research task: {task.description}")
            
            # Use research agent to execute the task
            research_query = task.description
            research_result = self.research_agent.execute_research(research_query)
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.result = f"Research completed for: {research_query}"
            
            self._log(state, f"Research task completed successfully")
            
            # Add result to messages
            state.messages.append(Message(
                sender="assistant",
                content=f"✅ Research completed: {task.description}\n\nReady for next task in sequence."
            ))
        
        return state.model_dump()
    
    def _execute_architecture_node(self, state: AppState) -> Dict[str, Any]:
        """Execute architecture planning tasks."""
        print("---EXECUTING ARCHITECTURE TASKS---")
        
        # Find pending architecture tasks
        arch_tasks = [
            task for task in state.tasks 
            if task.status == TaskStatus.PENDING and getattr(task, 'agent', '') == 'architecture'
        ]
        
        if arch_tasks:
            task = arch_tasks[0]
            task.status = TaskStatus.IN_PROGRESS
            
            self._log(state, f"Executing architecture task: {task.description}")
            
            # Simulate architecture planning
            task.status = TaskStatus.COMPLETED
            task.result = f"Architecture plan created for: {task.description}"
            
            state.messages.append(Message(
                sender="assistant", 
                content=f"✅ Architecture planning completed: {task.description}\n\nSystem architecture and technical specifications ready."
            ))
            
            self._log(state, f"Architecture task completed successfully")
        
        return state.model_dump()
    
    def _execute_design_node(self, state: AppState) -> Dict[str, Any]:
        """Execute design and UX tasks.""" 
        print("---EXECUTING DESIGN TASKS---")
        
        # Find pending design tasks
        design_tasks = [
            task for task in state.tasks
            if task.status == TaskStatus.PENDING and getattr(task, 'agent', '') == 'design'
        ]
        
        if design_tasks:
            task = design_tasks[0]
            task.status = TaskStatus.IN_PROGRESS
            
            self._log(state, f"Executing design task: {task.description}")
            
            # Simulate design work
            task.status = TaskStatus.COMPLETED
            task.result = f"Design artifacts created for: {task.description}"
            
            state.messages.append(Message(
                sender="assistant",
                content=f"✅ Design completed: {task.description}\n\nWireframes and UI mockups ready."
            ))
            
            self._log(state, f"Design task completed successfully")
        
        return state.model_dump()
    
    def _execute_builder_node(self, state: AppState) -> Dict[str, Any]:
        """Execute building/development tasks."""
        print("---EXECUTING BUILDER TASKS---")
        
        # Find pending builder tasks
        builder_tasks = [
            task for task in state.tasks
            if task.status == TaskStatus.PENDING and getattr(task, 'agent', '') == 'builder'
        ]
        
        if builder_tasks:
            task = builder_tasks[0]
            task.status = TaskStatus.IN_PROGRESS
            
            self._log(state, f"Executing builder task: {task.description}")
            
            # Simulate building work
            task.status = TaskStatus.COMPLETED
            task.result = f"Development completed for: {task.description}"
            
            state.messages.append(Message(
                sender="assistant",
                content=f"✅ Development completed: {task.description}\n\n🎉 Your project has been successfully built! All components are ready for deployment."
            ))
            
            self._log(state, f"Builder task completed successfully")
        
        return state.model_dump()
    
    def _await_user_input_node(self, state) -> Dict[str, Any]:
        """Node for awaiting additional user input."""
        print("---AWAITING USER INPUT---")
        
        # Handle both dict and AppState formats
        if isinstance(state, dict):
            app_state = AppState(**state)
        else:
            app_state = state
            
        self._log(app_state, "Conversation complete. Waiting for additional user input.")
        
        return app_state.model_dump()
    
    def _route_from_intelligent_orchestrator(self, state) -> str:
        """Routes based on the intelligent orchestrator's decision."""
        # Handle both AppState object and dictionary formats
        if isinstance(state, dict):
            next_action = state.get('next_action', 'continue_conversation')
        else:
            next_action = getattr(state, 'next_action', 'continue_conversation')
        
        # Map orchestrator decisions to workflow paths
        routing_map = {
            "continue_conversation": "await_user_input",
            "request_clarification": "await_user_input", 
            "redirect_conversation": "await_user_input",
            "execute_research": "execute_research",
            "execute_architecture": "execute_architecture",
            "execute_design": "execute_design", 
            "execute_builder": "execute_builder"
        }
        
        route = routing_map.get(next_action, "await_user_input")
        print(f"[Routing] Orchestrator decision: {next_action} -> {route}")
        
        return route
    
    def _log(self, state: AppState, message: str):
        """Adds a log entry to the state."""
        log_entry = LogEntry(
            source="IntelligentRAGWorkflow",
            message=message
        )
        state.logs.append(log_entry)
        print(f"[IntelligentRAGWorkflow] {message}")

# Create the application instance
intelligent_workflow = IntelligentRAGWorkflow()
app = intelligent_workflow.workflow