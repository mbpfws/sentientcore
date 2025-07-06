"""  
The Sentient Workflow Graph - Build 1: Core Conversation & Orchestration Loop
Foundational workflow with conversational guidance, clarification-first approach, and state management.
"""

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from core.models import AppState, Message, LogEntry
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.services.enhanced_llm_service import EnhancedLLMService
from core.services.session_persistence_service import SessionPersistenceService
from core.services.memory_service import MemoryService
from typing import Dict, Any, Optional, List
import logging
import asyncio
import time
import uuid
import json
import os

# Global variables for lazy initialization
_llm_service = None
_ultra_orchestrator = None
_session_persistence = None

def get_llm_service():
    """Get or create LLM service instance"""
    global _llm_service
    if _llm_service is None:
        try:
            print("Attempting to initialize EnhancedLLMService...")
            _llm_service = EnhancedLLMService()
            print("✓ EnhancedLLMService initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize LLM service: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            # Create a mock service for testing
            _llm_service = MockLLMService()
            print("Using MockLLMService as fallback")
    return _llm_service

def get_ultra_orchestrator():
    """Get or create Ultra Orchestrator instance"""
    global _ultra_orchestrator
    if _ultra_orchestrator is None:
        try:
            print("Attempting to initialize UltraOrchestrator...")
            llm_service = get_llm_service()
            print(f"LLM Service type: {type(llm_service).__name__}")
            _ultra_orchestrator = UltraOrchestrator(llm_service)
            print("✓ UltraOrchestrator initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize Ultra Orchestrator: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            # Create a mock orchestrator for testing
            _ultra_orchestrator = MockUltraOrchestrator()
            print("Using MockUltraOrchestrator as fallback")
    return _ultra_orchestrator

def get_session_persistence():
    """Get or create Session Persistence service instance"""
    global _session_persistence
    if _session_persistence is None:
        try:
            _session_persistence = SessionPersistenceService()
            print("✓ SessionPersistenceService initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize Session Persistence: {e}")
            _session_persistence = MockSessionPersistence()
    return _session_persistence

# Mock classes for fallback when initialization fails
class MockLLMService:
    """Mock LLM service for testing when real service fails to initialize"""
    async def invoke(self, system_prompt: str, user_prompt: str, model: str, **kwargs):
        return "Mock response: LLM service is not available"

class MockUltraOrchestrator:
    """Mock Ultra Orchestrator for testing when real orchestrator fails to initialize"""
    async def invoke(self, state: AppState) -> AppState:
        state.messages.append(Message(
            sender="assistant",
            content="System is initializing. Please try again in a moment."
        ))
        state.logs.append(LogEntry(
            source="MockUltraOrchestrator",
            message="Mock orchestrator invoked - system initializing"
        ))
        state.next_action = "end"
        return state
    
    async def invoke_state(self, state: AppState) -> AppState:
        """Mock invoke_state method for compatibility"""
        return await self.invoke(state)

class MockSessionPersistence:
    """Mock Session Persistence for testing when real service fails to initialize"""
    async def save_session(self, session_id: str, state: AppState):
        print(f"Mock: Would save session {session_id}")
        return True
    
    async def load_session(self, session_id: str):
        print(f"Mock: Would load session {session_id}")
        return None



# Function to save session state with enhanced conversation history persistence
async def save_session(state: AppState) -> None:
    """
    Build 1: Save session state with enhanced conversation history persistence.
    Ensures all conversation context and artifacts are properly stored.
    """
    try:
        session_persistence = get_session_persistence()
        
        # Build 1: Ensure conversation_history is up to date before saving
        if hasattr(state, 'conversation_history') and state.conversation_history is not None:
            # Validate conversation history integrity
            if len(state.conversation_history) != len(state.messages):
                # Rebuild conversation history from messages if mismatch
                state.conversation_history = []
                for msg in state.messages:
                    state.conversation_history.append({
                        "role": msg.sender,
                        "content": msg.content,
                        "timestamp": msg.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    })
        
        await session_persistence.save_session(state.session_id, state)
        
        # Add enhanced log entry for successful save
        state.logs.append(LogEntry(
            source="Build1_WorkflowGraph",
            message=f"[BUILD 1] Session {state.session_id} saved with {len(state.messages)} messages, {len(state.conversation_history)} history entries, and {len(state.logs)} log entries"
        ))
        
    except Exception as e:
        # Enhanced error logging with more context
        state.logs.append(LogEntry(
            source="Build1_WorkflowGraph",
            message=f"[BUILD 1] Failed to save session {state.session_id}: {str(e)} - State may be lost"
        ))

# Function to load existing session state
async def load_session_if_exists(session_id: str) -> AppState:
    """
    Build 1: Load existing session state with conversation history or create new AppState.
    Enhanced for robust state management and conversation context preservation.
    """
    session_persistence = get_session_persistence()
    
    try:
        existing_state = await session_persistence.load_session(session_id)
        if existing_state:
            print(f"✓ Loaded existing session: {session_id}")
            
            # Build 1: Ensure conversation_history is initialized
            if not hasattr(existing_state, 'conversation_history') or existing_state.conversation_history is None:
                existing_state.conversation_history = []
                
                # Rebuild conversation history from messages if missing
                for msg in existing_state.messages:
                    existing_state.conversation_history.append({
                        "role": msg.sender,
                        "content": msg.content,
                        "timestamp": msg.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ")
                    })
            
            existing_state.logs.append(LogEntry(
                source="Build1_WorkflowGraph",
                message=f"[BUILD 1] Session {session_id} restored with {len(existing_state.messages)} messages and {len(existing_state.conversation_history)} history entries"
            ))
            return existing_state
    except Exception as e:
        print(f"Warning: Failed to load session {session_id}: {e}")
    
    # Create new state if loading failed or session doesn't exist
    new_state = AppState(
        messages=[],
        user_prompt="",
        image=None,
        conversation_history=[],  # Build 1: Initialize conversation history
        tasks=[],
        logs=[LogEntry(
            source="Build1_WorkflowGraph",
            message=f"[BUILD 1] New session initialized: {session_id}"
        )],
        session_id=session_id
    )
    return new_state

# Create the main workflow graph for Build 1: Core Conversation & Orchestration Loop
workflow = StateGraph(AppState)

# --- Define the graph nodes ---

# Async wrapper for the Ultra Orchestrator with session persistence
async def ultra_orchestrator_node(state: AppState) -> AppState:
    """
    Build 1: Enhanced core conversation processing node with robust state management.
    Handles user messages, maintains conversation context, and ensures persistence.
    """
    # Get or create session ID
    session_id = state.session_id or str(uuid.uuid4())
    state.session_id = session_id
    
    # Initialize conversation_history if not present (Build 1 requirement)
    if not hasattr(state, 'conversation_history') or state.conversation_history is None:
        state.conversation_history = []
    
    # Log the processing start with enhanced context
    state.logs.append(LogEntry(
        source="Build1_WorkflowGraph",
        message=f"[BUILD 1] Processing message in session {session_id} - Messages: {len(state.messages)}"
    ))
    
    try:
        # Get the orchestrator instance
        orchestrator = get_ultra_orchestrator()
        session_persistence = get_session_persistence()
        
        # Build 1: Ensure conversation history is maintained
        # Add current user message to conversation history
        if state.messages:
            latest_message = state.messages[-1]
            if latest_message.sender == "user":
                state.conversation_history.append({
                    "role": "user",
                    "content": latest_message.content,
                    "timestamp": latest_message.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        
        # Process the message through the orchestrator with enhanced state
        result_state = await orchestrator.invoke_state(state)
        
        # Build 1: Add assistant response to conversation history
        if result_state.messages:
            latest_assistant_message = None
            for msg in reversed(result_state.messages):
                if msg.sender == "assistant":
                    latest_assistant_message = msg
                    break
            
            if latest_assistant_message:
                result_state.conversation_history.append({
                    "role": "assistant",
                    "content": latest_assistant_message.content,
                    "timestamp": latest_assistant_message.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ")
                })
        
        # Build 1: Enhanced session persistence with conversation history
        try:
            await session_persistence.save_session(result_state.session_id, result_state)
            result_state.logs.append(LogEntry(
                source="Build1_WorkflowGraph",
                message=f"Session {result_state.session_id} saved successfully"
            ))
        except Exception as e:
            result_state.logs.append(LogEntry(
                source="Build1_WorkflowGraph",
                message=f"Warning: Failed to save session {result_state.session_id}: {str(e)}"
            ))
        
        # Log successful processing
        result_state.logs.append(LogEntry(
            source="Build1_WorkflowGraph",
            message=f"[BUILD 1] Successfully processed message - Total history: {len(result_state.conversation_history)} entries"
        ))
        
        # Set next action to end for Build 1 (orchestrator handles clarification internally)
        result_state.next_action = "end"
        
        return result_state
        
    except Exception as e:
        # Enhanced error handling and logging
        error_message = f"[BUILD 1] Error in ultra_orchestrator_node: {str(e)}"
        state.logs.append(LogEntry(
            source="Build1_WorkflowGraph",
            message=error_message
        ))
        
        # Add error message to conversation
        error_response = Message(
            sender="assistant",
            content="I encountered an error processing your request. Please try again or provide more details.",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        state.messages.append(error_response)
        
        # Add error to conversation history
        state.conversation_history.append({
            "role": "assistant",
            "content": error_response.content,
            "timestamp": error_response.created_at,
            "error": True
        })
        
        state.next_action = "end"
        return state

# Build 2: Research node for handling research tasks and artifact generation
async def research_node(state: AppState) -> AppState:
    """
    Build 2: Enhanced research node with artifact generation and memory persistence.
    Handles autonomous research tasks and saves artifacts to memory.
    """
    try:
        # Log research start
        state.logs.append(LogEntry(
            source="Build2_ResearchAgent",
            message=f"[BUILD 2] Starting research task for session {state.session_id}"
        ))
        
        # Get the research agent
        from core.agents.build2_research_agent import Build2ResearchAgent
        research_agent = Build2ResearchAgent()
        
        # Extract research query from the latest user message
        research_query = ""
        if state.messages:
            latest_message = state.messages[-1]
            if latest_message.sender == "user":
                research_query = latest_message.content
        
        # Conduct research with the query
        research_result = await research_agent.conduct_research(research_query)
        
        # Add research result to messages
        research_response = Message(
            sender="assistant",
            content=f"Research completed successfully. {research_result.get('summary', 'Research artifacts have been generated and saved.')}",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        state.messages.append(research_response)
        
        # Add to conversation history
        state.conversation_history.append({
            "role": "assistant",
            "content": research_response.content,
            "timestamp": research_response.created_at,
            "research_artifacts": research_result.get('artifacts', [])
        })
        
        # Log research completion
        state.logs.append(LogEntry(
            source="Build2_ResearchAgent",
            message=f"[BUILD 2] Research completed with {len(research_result.get('artifacts', []))} artifacts generated"
        ))
        
        # Set next action based on research results
        state.next_action = "process_message"  # Return to orchestrator for next steps
        
        return state
        
    except Exception as e:
        # Enhanced error handling
        error_message = f"[BUILD 2] Error in research_node: {str(e)}"
        state.logs.append(LogEntry(
            source="Build2_ResearchAgent",
            message=error_message
        ))
        
        # Add error message to conversation
        error_response = Message(
            sender="assistant",
            content="I encountered an error during research. Please try rephrasing your request or provide more specific details.",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        state.messages.append(error_response)
        
        state.conversation_history.append({
            "role": "assistant",
            "content": error_response.content,
            "timestamp": error_response.created_at,
            "error": True
        })
        
        state.next_action = "process_message"
        return state

# Build 3: Planning node for handling planning tasks and PRD generation
async def planning_node(state: AppState) -> AppState:
    """
    Build 3: Enhanced planning node with PRD generation and architecture planning.
    Handles project planning tasks and generates comprehensive documentation.
    """
    try:
        # Log planning start
        state.logs.append(LogEntry(
            source="Build3_PlanningAgent",
            message=f"[BUILD 3] Starting planning task for session {state.session_id}"
        ))
        
        # Get the planning agent
        from core.agents.architect_planner_agent import ArchitectPlannerAgent
        planning_agent = ArchitectPlannerAgent()
        
        # Extract planning request from the latest user message
        planning_request = ""
        if state.messages:
            latest_message = state.messages[-1]
            if latest_message.sender == "user":
                planning_request = latest_message.content
        
        # Generate planning response
        planning_result = await planning_agent.invoke(planning_request)
        
        # Add planning result to messages
        planning_response = Message(
            sender="assistant",
            content=planning_result,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        state.messages.append(planning_response)
        
        # Add to conversation history
        state.conversation_history.append({
            "role": "assistant",
            "content": planning_response.content,
            "timestamp": planning_response.created_at,
            "planning_artifacts": True
        })
        
        # Log planning completion
        state.logs.append(LogEntry(
            source="Build3_PlanningAgent",
            message="[BUILD 3] Planning completed successfully"
        ))
        
        # Set next action
        state.next_action = "process_message"  # Return to orchestrator for next steps
        
        return state
        
    except Exception as e:
        # Enhanced error handling
        error_message = f"[BUILD 3] Error in planning_node: {str(e)}"
        state.logs.append(LogEntry(
            source="Build3_PlanningAgent",
            message=error_message
        ))
        
        # Add error message to conversation
        error_response = Message(
            sender="assistant",
            content="I encountered an error during planning. Please try rephrasing your request or provide more details about your project requirements.",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        state.messages.append(error_response)
        
        state.conversation_history.append({
            "role": "assistant",
            "content": error_response.content,
            "timestamp": error_response.created_at,
            "error": True
        })
        
        state.next_action = "process_message"
        return state

# Implementation node placeholder for future builds
async def implementation_node(state: AppState) -> AppState:
    """
    Implementation node for code generation and feature implementation.
    Currently returns to orchestrator for further processing.
    """
    state.logs.append(LogEntry(
        source="ImplementationAgent",
        message="Implementation node called - returning to orchestrator"
    ))
    state.next_action = "process_message"
    return state

# Node for the Ultra Orchestrator - the core of Build 1 (handles conversational guidance internally)
workflow.add_node("orchestrator", ultra_orchestrator_node)

# Build 2 & 3: Add research and planning nodes
workflow.add_node("research", research_node)
workflow.add_node("planning", planning_node)
workflow.add_node("implementation", implementation_node)

# --- Define the graph edges for Builds 1, 2, and 3 ---

# The workflow starts with the orchestrator node (Ultra Orchestrator)
workflow.set_entry_point("orchestrator")

# Build 1, 2, 3: Enhanced routing based on next_action
def route_next_action(state: AppState) -> str:
    """
    Route to the appropriate node based on the state's next_action.
    Supports all three builds with intelligent routing.
    """
    action = state.next_action
    
    if action == "research":
        return "research"
    elif action == "planning":
        return "planning"
    elif action == "implementation":
        return "implementation"
    else:
        return END

# Add conditional edges from orchestrator
workflow.add_conditional_edges(
    "orchestrator",
    route_next_action,
    {
        "research": "research",
        "planning": "planning", 
        "implementation": "implementation",
        END: END
    }
)

# Build 2: Research node routes back to orchestrator or ends
def route_after_research(state: AppState) -> str:
    """
    Route after research completion based on state.
    """
    if state.next_action == "process_message":
        return "orchestrator"
    else:
        return END

workflow.add_conditional_edges(
    "research",
    route_after_research,
    {
        "orchestrator": "orchestrator",
        END: END
    }
)

# Build 3: Planning node routes back to orchestrator or ends
def route_after_planning(state: AppState) -> str:
    """
    Route after planning completion based on state.
    """
    if state.next_action == "process_message":
        return "orchestrator"
    else:
        return END

workflow.add_conditional_edges(
    "planning",
    route_after_planning,
    {
        "orchestrator": "orchestrator",
        END: END
    }
)

# Implementation node routes back to orchestrator or ends
def route_after_implementation(state: AppState) -> str:
    """
    Route after implementation completion based on state.
    """
    if state.next_action == "process_message":
        return "orchestrator"
    else:
        return END

workflow.add_conditional_edges(
    "implementation",
    route_after_implementation,
    {
        "orchestrator": "orchestrator",
        END: END
    }
)


# Function to safely compile the workflow
def get_sentient_workflow_app():
    """Get or create the compiled workflow app with error handling"""
    try:
        return workflow.compile()
    except Exception as e:
        print(f"Warning: Failed to compile workflow: {e}")
        # Return a mock workflow for testing
        return MockWorkflowApp()

class MockWorkflowApp:
    """Mock workflow app for testing when compilation fails"""
    async def ainvoke(self, state: AppState, **kwargs):
        """Mock async invoke method"""
        state.messages.append(Message(
            sender="assistant",
            content="System is currently initializing. Please try again in a moment."
        ))
        state.logs.append(LogEntry(
            source="MockWorkflowApp",
            message="Mock workflow app invoked - system initializing"
        ))
        return state

# Lazy compilation - only compile when needed
_sentient_workflow_app = None

def get_compiled_workflow():
    """Get the compiled workflow app with lazy initialization"""
    global _sentient_workflow_app
    if _sentient_workflow_app is None:
        _sentient_workflow_app = get_sentient_workflow_app()
    return _sentient_workflow_app

# Main workflow graph function
async def sentient_workflow_graph(state: AppState) -> AppState:
    """
    Builds 1, 2, 3: Main workflow graph with enhanced state management and conversation persistence.
    Routes between different agents based on state with robust error handling and session persistence.
    Supports orchestrator, research, and planning workflows with intelligent routing.
    """
    try:
        # Enhanced logging for all builds
        state.logs.append(LogEntry(
            source="SentientWorkflowGraph",
            message=f"[BUILDS 1-3] Starting workflow for session {state.session_id} with action: {state.next_action}"
        ))
        
        # Get the compiled workflow app
        workflow_app = get_compiled_workflow()
        
        # Execute the workflow with the compiled graph
        result_state = await workflow_app.ainvoke(state)
        
        # Enhanced final save with comprehensive logging
        await save_session(result_state)
        
        # Log workflow completion with detailed metrics
        result_state.logs.append(LogEntry(
            source="SentientWorkflowGraph",
            message=f"[BUILDS 1-3] Workflow completed for session {result_state.session_id}. Final action: {result_state.next_action}. Messages: {len(result_state.messages)}, History: {len(result_state.conversation_history)}"
        ))
        
        return result_state
        
    except Exception as e:
        # Enhanced error handling with detailed logging
        error_msg = f"[BUILDS 1-3] Workflow error in session {state.session_id}: {str(e)}"
        state.logs.append(LogEntry(
            source="SentientWorkflowGraph",
            message=error_msg
        ))
        
        # Still try to save session even on error for state preservation
        try:
            await save_session(state)
        except Exception as save_error:
            state.logs.append(LogEntry(
                source="SentientWorkflowGraph",
                message=f"[BUILDS 1-3] Critical: Failed to save session after error: {str(save_error)}"
            ))
            
        return state

# For backward compatibility, create a property that compiles on first access
class LazyWorkflowApp:
    def __getattr__(self, name):
        app = get_compiled_workflow()
        return getattr(app, name)
    
    async def ainvoke(self, *args, **kwargs):
        app = get_compiled_workflow()
        return await app.ainvoke(*args, **kwargs)

sentient_workflow_app = LazyWorkflowApp()