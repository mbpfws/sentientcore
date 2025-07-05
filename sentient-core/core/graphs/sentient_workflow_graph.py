"""  
The Sentient Workflow Graph - Build 1: The Core Conversation & Orchestration Loop
Foundational workflow with persistent conversation management and state handling.
"""

from langgraph.graph import StateGraph, END
from core.models import AppState, Message, LogEntry
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.services.enhanced_llm_service_main import EnhancedLLMService
from typing import Dict, Any

# Global variables for lazy initialization
_llm_service = None
_ultra_orchestrator = None

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



# Create the main workflow graph for Build 2: Conversation & Research Delegation
workflow = StateGraph(AppState)

# --- Define the graph nodes ---

# Async wrapper for the Ultra Orchestrator
async def ultra_orchestrator_node(state: AppState) -> AppState:
    """Build 1: Core conversation processing with persistent state management."""
    orchestrator = get_ultra_orchestrator()
    
    # Log incoming message for Build 1 conversation tracking
    latest_message = state.messages[-1].content if state.messages else "No message"
    state.logs.append(LogEntry(
        source="Build1_WorkflowGraph",
        message=f"Build 1: Processing conversation message: '{latest_message[:100]}...'"
    ))
    
    # Process through Build 1 orchestrator (simple conversation with context)
    result_state = await orchestrator.invoke(state)
    
    # Log Build 1 conversation completion
    result_state.logs.append(LogEntry(
        source="Build1_WorkflowGraph",
        message=f"Build 1: Conversation processed. Total messages: {len(result_state.messages)}, Context maintained: {bool(result_state.conversation_history)}"
    ))
    
    # Set next action to end for Build 1 (simple conversation loop)
    result_state.next_action = "end"
    
    return result_state

# Node for the Ultra Orchestrator - the core of Build 2 (handles delegation internally)
workflow.add_node("orchestrator", ultra_orchestrator_node)

# --- Define the graph edges for Build 2: Orchestrator with Internal Delegation ---

# The workflow starts with the orchestrator node (Ultra Orchestrator)
workflow.set_entry_point("orchestrator")

# For Build 2, we have a simple linear flow: orchestrator -> end
# The orchestrator internally handles research delegation to the Research Agent
# This maintains the stateful conversation while enabling specialized task delegation
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