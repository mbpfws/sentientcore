"""  
The Sentient Workflow Graph - Build 1: Core Conversation & Orchestration Loop
Simplified workflow focusing on basic conversation management and state persistence.
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
            _llm_service = EnhancedLLMService()
        except Exception as e:
            print(f"Warning: Failed to initialize LLM service: {e}")
            # Create a mock service for testing
            _llm_service = MockLLMService()
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



# Create the main workflow graph for Build 1: Core Conversation Loop
workflow = StateGraph(AppState)

# --- Define the graph nodes ---

# Async wrapper for the Ultra Orchestrator
async def ultra_orchestrator_node(state: AppState) -> AppState:
    """Async wrapper for the Ultra Orchestrator invoke method."""
    orchestrator = get_ultra_orchestrator()
    
    # Add conversation history logging for Build 1
    state.logs.append(LogEntry(
        source="WorkflowGraph",
        message=f"Processing user message: '{state.user_prompt}'"
    ))
    
    # Process through orchestrator
    result_state = await orchestrator.invoke(state)
    
    # Log the conversation turn completion
    result_state.logs.append(LogEntry(
        source="WorkflowGraph",
        message=f"Conversation turn completed. Messages in history: {len(result_state.messages)}"
    ))
    
    return result_state

# Node for the Ultra Orchestrator - the core of Build 1
workflow.add_node("conversation", ultra_orchestrator_node)

# --- Define the graph edges for Build 1: Simple Conversation Loop ---

# The workflow starts with the conversation node (Ultra Orchestrator)
workflow.set_entry_point("conversation")

# For Build 1, we have a simple linear flow: conversation -> end
# This creates a stateful conversation where each user input goes through
# the orchestrator and the conversation history is maintained in AppState
workflow.add_edge("conversation", END)


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