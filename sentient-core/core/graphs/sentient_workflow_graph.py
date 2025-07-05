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
from core.services.enhanced_llm_service_main import EnhancedLLMService
from core.services.session_persistence_service import SessionPersistenceService
from typing import Dict, Any
import uuid
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
            llm_service = get_llm_service()
            _ultra_orchestrator = UltraOrchestrator(llm_service)
        except Exception as e:
            print(f"Warning: Failed to initialize Ultra Orchestrator: {e}")
            # Create a mock orchestrator for testing
            _ultra_orchestrator = MockUltraOrchestrator()
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

class MockSessionPersistence:
    """Mock Session Persistence for testing when real service fails to initialize"""
    async def save_session(self, session_id: str, state: AppState):
        print(f"Mock: Would save session {session_id}")
        return True
    
    async def load_session(self, session_id: str):
        print(f"Mock: Would load session {session_id}")
        return None



# Function to load existing session state
async def load_session_if_exists(session_id: str) -> AppState:
    """Load existing session state or create new AppState"""
    session_persistence = get_session_persistence()
    
    try:
        existing_state = await session_persistence.load_session(session_id)
        if existing_state:
            print(f"✓ Loaded existing session: {session_id}")
            existing_state.logs.append(LogEntry(
                source="Build1_WorkflowGraph",
                message=f"Session {session_id} restored from persistent storage"
            ))
            return existing_state
    except Exception as e:
        print(f"Warning: Failed to load session {session_id}: {e}")
    
    # Create new state if loading failed or session doesn't exist
    new_state = AppState(
        messages=[],
        user_prompt="",
        image=None,
        conversation_history=[],
        tasks=[],
        logs=[LogEntry(
            source="Build1_WorkflowGraph",
            message=f"New session initialized: {session_id}"
        )],
        session_id=session_id
    )
    return new_state

# Create the main workflow graph for Build 1: Core Conversation & Orchestration Loop
workflow = StateGraph(AppState)

# --- Define the graph nodes ---

# Async wrapper for the Ultra Orchestrator with session persistence
async def ultra_orchestrator_node(state: AppState) -> AppState:
    """Build 1: Core conversation processing with conversational guidance and clarification-first approach."""
    orchestrator = get_ultra_orchestrator()
    session_persistence = get_session_persistence()
    
    # Ensure session ID exists
    if not hasattr(state, 'session_id') or not state.session_id:
        state.session_id = str(uuid.uuid4())
        state.logs.append(LogEntry(
            source="Build1_WorkflowGraph",
            message=f"New session created: {state.session_id}"
        ))
    
    # Log incoming message for Build 1 processing
    latest_message = state.messages[-1].content if state.messages else "No message"
    state.logs.append(LogEntry(
        source="Build1_WorkflowGraph",
        message=f"Build 1: Processing message in session {state.session_id}: '{latest_message[:100]}...'"
    ))
    
    # Process through Build 1 orchestrator (conversational guidance with clarification-first)
    result_state = await orchestrator.invoke_state(state)
    
    # Save session state after processing
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
    
    # Log Build 1 processing completion
    result_state.logs.append(LogEntry(
        source="Build1_WorkflowGraph",
        message=f"Build 1: Processing completed. Messages: {len(result_state.messages)}, Action: {result_state.next_action}"
    ))
    
    # Set next action to end (orchestrator handles clarification internally)
    result_state.next_action = "end"
    
    return result_state

# Node for the Ultra Orchestrator - the core of Build 1 (handles conversational guidance internally)
workflow.add_node("orchestrator", ultra_orchestrator_node)

# --- Define the graph edges for Build 2: Orchestrator with Internal Delegation ---

# The workflow starts with the orchestrator node (Ultra Orchestrator)
workflow.set_entry_point("orchestrator")

# For Build 1, we have a simple linear flow: orchestrator -> end
# The orchestrator internally handles conversational guidance and clarification requests
# This maintains the stateful conversation while providing collaborative guidance
workflow.add_edge("orchestrator", END)


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

# Compile the workflow with error handling
sentient_workflow_app = get_sentient_workflow_app()