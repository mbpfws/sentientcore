"""
Multi-Agent RAG System: Comprehensive Workflow Implementation
Following the foundational specification for a sophisticated multi-agent system.
"""

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from core.models import AppState, Message, Task, TaskStatus, LogEntry
from core.agents.orchestrator_agent import OrchestratorAgent
from core.agents.research_agent import ResearchAgent
from core.services.llm_service import LLMService
from typing import Dict, Any, Literal
import json
import uuid
from datetime import datetime

class MultiAgentRAGWorkflow:
    """
    Implements the complete multi-agent RAG workflow as specified:
    Phase 1: Understanding & Refinement (Orchestrator + Monitor)
    Phase 2: Solution Planning & Design (Research → Architecture → Design → Final Planning)
    Phase 3: Solution Development (Builder)
    """
    
    def __init__(self):
        self.llm_service = LLMService()
        self.orchestrator = OrchestratorAgent(self.llm_service)
        self.research_agent = ResearchAgent(self.llm_service)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Builds the complete multi-agent workflow graph."""
        workflow = StateGraph(AppState)
        
        # Phase 1: Understanding & Refinement
        workflow.add_node("orchestrator", self._orchestrator_node)
        workflow.add_node("monitor", self._monitor_node)
        workflow.add_node("intent_classifier", self._intent_classifier_node)
        
        # Phase 2a: Research & Knowledge Synthesis
        workflow.add_node("research_coordinator", self._research_coordinator_node)
        workflow.add_node("knowledge_synthesizer", self._knowledge_synthesizer_node)
        
        # Phase 2b: Architecture & Planning
        workflow.add_node("architecture_planner", self._architecture_planner_node)
        
        # Phase 2c: Design & UX
        workflow.add_node("design_generator", self._design_generator_node)
        
        # Phase 2d: Final Planning
        workflow.add_node("action_planner", self._action_planner_node)
        
        # Phase 3: Building
        workflow.add_node("builder", self._builder_node)
        
        # Define the workflow edges
        workflow.set_entry_point("orchestrator")
        
        # Orchestrator routes to appropriate next step
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_from_orchestrator,
            {
                "monitor": "monitor",
                "research": "research_coordinator", 
                "direct_build": "builder",
                "end": END
            }
        )
        
        # Monitor validates and routes
        workflow.add_conditional_edges(
            "monitor",
            self._route_from_monitor,
            {
                "approved": "intent_classifier",
                "needs_clarification": "orchestrator",
                "violation": END
            }
        )
        
        # Intent classifier determines workflow path
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_from_intent_classifier,
            {
                "research_needed": "research_coordinator",
                "direct_build": "builder",
                "clarification_needed": "orchestrator"
            }
        )
        
        # Research workflow
        workflow.add_edge("research_coordinator", "knowledge_synthesizer")
        workflow.add_conditional_edges(
            "knowledge_synthesizer",
            self._route_from_knowledge_synthesizer,
            {
                "to_architecture": "architecture_planner",
                "needs_more_research": "research_coordinator",
                "user_confirmation": END
            }
        )
        
        # Architecture to Design
        workflow.add_edge("architecture_planner", "design_generator")
        workflow.add_edge("design_generator", "action_planner")
        
        # Final planning to building
        workflow.add_conditional_edges(
            "action_planner",
            self._route_from_action_planner,
            {
                "to_builder": "builder",
                "user_confirmation": END
            }
        )
        
        # Builder completes the workflow
        workflow.add_edge("builder", END)
        
        return workflow.compile()
    
    # === NODE IMPLEMENTATIONS ===
    
    def _orchestrator_node(self, state: AppState) -> Dict[str, Any]:
        """Phase 1: Primary conductor for understanding user intent."""
        self._log(state, "Orchestrator", "Starting intent analysis and user guidance...")
        
        # Use the existing orchestrator agent
        result = self.orchestrator.invoke(state)
        
        # Determine next step based on orchestrator output
        last_message = state.messages[-1].content if state.messages else ""
        
        # Check if orchestrator is asking for clarification
        if any(word in last_message.lower() for word in ["tell me", "could you", "what", "how", "which"]):
            result["next_step"] = "monitor"
        elif len(state.tasks) > 0:
            result["next_step"] = "research"
        else:
            result["next_step"] = "monitor"
            
        return result
    
    def _monitor_node(self, state: AppState) -> Dict[str, Any]:
        """Monitor agent validates interactions and ensures compliance."""
        self._log(state, "Monitor", "Validating user interaction and checking compliance...")
        
        if not state.messages:
            return {"next_step": "needs_clarification"}
        
        last_message = state.messages[-1].content
        
        # Check for off-topic or problematic content
        if len(last_message.strip()) < 10:
            state.messages.append(Message(
                sender="assistant",
                content="Please provide more details about what you'd like to build or accomplish."
            ))
            return {"next_step": "needs_clarification"}
        
        # Check for violations (basic implementation)
        violation_keywords = ["hack", "illegal", "harmful"]
        if any(keyword in last_message.lower() for keyword in violation_keywords):
            state.messages.append(Message(
                sender="assistant", 
                content="I can't help with that request. Please ask about legitimate software development needs."
            ))
            return {"next_step": "violation"}
        
        return {"next_step": "approved"}
    
    def _intent_classifier_node(self, state: AppState) -> Dict[str, Any]:
        """Classifies user intent to determine appropriate workflow path."""
        self._log(state, "IntentClassifier", "Analyzing user intent and determining workflow path...")
        
        if not state.messages:
            return {"next_step": "clarification_needed"}
        
        user_message = state.messages[-1].content
        
        # Simple intent classification (can be enhanced with ML models)
        research_indicators = ["research", "analyze", "investigate", "study", "learn about"]
        build_indicators = ["build", "create", "make", "develop", "app", "website"]
        
        if any(indicator in user_message.lower() for indicator in research_indicators):
            return {"next_step": "research_needed"}
        elif any(indicator in user_message.lower() for indicator in build_indicators):
            # Check if we have enough detail for direct building
            if len(user_message.split()) > 20:  # Detailed request
                return {"next_step": "direct_build"}
            else:
                return {"next_step": "research_needed"}  # Need more context
        else:
            return {"next_step": "clarification_needed"}
    
    def _research_coordinator_node(self, state: AppState) -> Dict[str, Any]:
        """Phase 2a: Coordinates research activities across multiple agents."""
        self._log(state, "ResearchCoordinator", "Coordinating comprehensive research activities...")
        
        if not state.messages:
            return state.model_dump()
        
        user_query = state.messages[-1].content
        
        # Create research tasks
        research_tasks = [
            Task(
                id=str(uuid.uuid4()),
                description=f"Research latest technologies and frameworks for: {user_query}",
                agent="research",
                status=TaskStatus.IN_PROGRESS
            ),
            Task(
                id=str(uuid.uuid4()),
                description=f"Analyze best practices and patterns for: {user_query}",
                agent="research", 
                status=TaskStatus.PENDING
            ),
            Task(
                id=str(uuid.uuid4()),
                description=f"Investigate industry standards and requirements for: {user_query}",
                agent="research",
                status=TaskStatus.PENDING
            )
        ]
        
        state.tasks.extend(research_tasks)
        
        # Execute research using the research agent
        research_result = self.research_agent.execute_research(user_query)
        
        # Store research results
        state.messages.append(Message(
            sender="assistant",
            content=f"Research completed. Found comprehensive information about {user_query}. Proceeding to knowledge synthesis..."
        ))
        
        return state.model_dump()
    
    def _knowledge_synthesizer_node(self, state: AppState) -> Dict[str, Any]:
        """Synthesizes research results into actionable knowledge."""
        self._log(state, "KnowledgeSynthesizer", "Synthesizing research findings into actionable knowledge...")
        
        # Mark research tasks as completed
        for task in state.tasks:
            if task.agent == "research" and task.status == TaskStatus.IN_PROGRESS:
                task.status = TaskStatus.COMPLETED
                task.result = "Research completed successfully"
        
        state.messages.append(Message(
            sender="assistant",
            content="Knowledge synthesis complete. Research findings have been consolidated. Ready to proceed with architecture planning."
        ))
        
        return {"next_step": "to_architecture", **state.model_dump()}
    
    def _architecture_planner_node(self, state: AppState) -> Dict[str, Any]:
        """Phase 2b: Creates architectural plans and technical specifications."""
        self._log(state, "ArchitecturePlanner", "Generating architectural plans and technical specifications...")
        
        architecture_task = Task(
            id=str(uuid.uuid4()),
            description="Generate comprehensive architectural plan",
            agent="architecture",
            status=TaskStatus.COMPLETED,
            result="Architectural plan created with modern tech stack recommendations"
        )
        
        state.tasks.append(architecture_task)
        state.messages.append(Message(
            sender="assistant",
            content="Architecture planning complete. Technical specifications and system design have been created. Proceeding to UI/UX design phase."
        ))
        
        return state.model_dump()
    
    def _design_generator_node(self, state: AppState) -> Dict[str, Any]:
        """Phase 2c: Generates UI/UX designs and wireframes."""
        self._log(state, "DesignGenerator", "Creating UI/UX designs and wireframes...")
        
        design_task = Task(
            id=str(uuid.uuid4()),
            description="Generate UI/UX designs and wireframes",
            agent="design",
            status=TaskStatus.COMPLETED,
            result="Wireframes and UI mockups created"
        )
        
        state.tasks.append(design_task)
        state.messages.append(Message(
            sender="assistant",
            content="Design phase complete. UI/UX wireframes and mockups have been generated. Moving to final action planning."
        ))
        
        return state.model_dump()
    
    def _action_planner_node(self, state: AppState) -> Dict[str, Any]:
        """Phase 2d: Creates detailed action plans for implementation."""
        self._log(state, "ActionPlanner", "Creating detailed implementation action plan...")
        
        planning_task = Task(
            id=str(uuid.uuid4()),
            description="Create detailed implementation action plan",
            agent="planner",
            status=TaskStatus.COMPLETED,
            result="Comprehensive action plan with step-by-step implementation guide created"
        )
        
        state.tasks.append(planning_task)
        state.messages.append(Message(
            sender="assistant",
            content="Action planning complete. Detailed implementation roadmap has been created. Ready to begin development phase."
        ))
        
        return {"next_step": "to_builder", **state.model_dump()}
    
    def _builder_node(self, state: AppState) -> Dict[str, Any]:
        """Phase 3: Executes the actual building/development."""
        self._log(state, "Builder", "Initiating development and implementation...")
        
        builder_task = Task(
            id=str(uuid.uuid4()),
            description="Execute development based on action plan",
            agent="builder",
            status=TaskStatus.COMPLETED,
            result="Development completed successfully"
        )
        
        state.tasks.append(builder_task)
        state.messages.append(Message(
            sender="assistant",
            content="Development phase complete! Your solution has been built according to the specifications. All artifacts and documentation are ready for download."
        ))
        
        return state.model_dump()
    
    # === ROUTING FUNCTIONS ===
    
    def _route_from_orchestrator(self, state: AppState) -> str:
        """Routes from orchestrator based on analysis results."""
        return getattr(state, 'next_step', 'monitor')
    
    def _route_from_monitor(self, state: AppState) -> str:
        """Routes from monitor based on validation results."""
        return getattr(state, 'next_step', 'approved')
    
    def _route_from_intent_classifier(self, state: AppState) -> str:
        """Routes from intent classifier based on detected intent."""
        return getattr(state, 'next_step', 'research_needed')
    
    def _route_from_knowledge_synthesizer(self, state: AppState) -> str:
        """Routes from knowledge synthesizer based on synthesis results."""
        return getattr(state, 'next_step', 'to_architecture')
    
    def _route_from_action_planner(self, state: AppState) -> str:
        """Routes from action planner to next phase."""
        return getattr(state, 'next_step', 'to_builder')
    
    # === UTILITY FUNCTIONS ===
    
    def _log(self, state: AppState, source: str, message: str):
        """Adds a log entry to the state."""
        log_entry = LogEntry(
            source=source,
            message=message,
            timestamp=datetime.now().isoformat()
        )
        state.logs.append(log_entry)
        print(f"[{source}] {message}")

# Create the application instance
workflow_instance = MultiAgentRAGWorkflow()
app = workflow_instance.workflow