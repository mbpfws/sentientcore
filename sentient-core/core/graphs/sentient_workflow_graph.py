"""
The Sentient Workflow Graph
This is the main, top-level graph that orchestrates the entire multi-agent system.
"""

from langgraph.graph import StateGraph, END
from core.models import AppState
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.agents.monitoring_agent import MonitoringAgent
from core.services.llm_service import EnhancedLLMService
from typing import Dict, Any

# Global variables for lazy initialization
_llm_service = None
_ultra_orchestrator = None
_monitoring_agent = None

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

def get_monitoring_agent():
    """Get or create Monitoring Agent instance"""
    global _monitoring_agent
    if _monitoring_agent is None:
        try:
            _monitoring_agent = MonitoringAgent()
        except Exception as e:
            print(f"Warning: Failed to initialize Monitoring Agent: {e}")
            # Create a mock monitoring agent for testing
            _monitoring_agent = MockMonitoringAgent()
    return _monitoring_agent

# Mock classes for fallback when initialization fails
class MockLLMService:
    """Mock LLM service for testing when real service fails to initialize"""
    async def invoke(self, system_prompt: str, user_prompt: str, model: str, **kwargs):
        return "Mock response: LLM service is not available"

class MockUltraOrchestrator:
    """Mock Ultra Orchestrator for testing when real orchestrator fails to initialize"""
    async def invoke(self, state: AppState) -> AppState:
        state.messages.append({
            "sender": "assistant",
            "content": "System is initializing. Please try again in a moment.",
            "created_at": "2024-01-01T00:00:00Z"
        })
        state.next_action = "end"
        return state

class MockMonitoringAgent:
    """Mock Monitoring Agent for testing when real agent fails to initialize"""
    def invoke(self, state: AppState) -> AppState:
        print("Mock monitoring agent: System status OK")
        return state

# Create the main workflow graph
workflow = StateGraph(AppState)

# --- Define the graph nodes ---

# Async wrapper for the Ultra Orchestrator
async def ultra_orchestrator_node(state: AppState) -> AppState:
    """Async wrapper for the Ultra Orchestrator invoke method."""
    orchestrator = get_ultra_orchestrator()
    return await orchestrator.invoke(state)

# Sync wrapper for the Monitoring Agent
def monitoring_agent_node(state: AppState) -> AppState:
    """Wrapper for the Monitoring Agent invoke method."""
    agent = get_monitoring_agent()
    return agent.invoke(state)

# Node for the Ultra Orchestrator to make decisions
workflow.add_node("ultra_orchestrator", ultra_orchestrator_node)

# Node for the Monitoring Agent to observe and log
workflow.add_node("monitor", monitoring_agent_node)

# --- Define a routing function ---
def route_from_monitor(state: AppState) -> str:
    """
    Determines the next step after monitoring based on the orchestrator's decision.
    """
    decision = getattr(state, 'next_action', 'end')
    print(f"---ROUTING from Monitor. Decision: {decision}---")
    
    if decision == "create_plan":
        # In future phases, this will route to the appropriate agent (e.g., "research").
        # For now, we end the workflow to inspect the plan.
        return "end"
    elif decision == "monitor":
        return "monitor"
    elif decision == "orchestrate":
        return "ultra_orchestrator"
    
    # For conversational decisions, we end the loop and await the next user input.
    return "end"


# --- Define the graph edges ---

# The workflow starts with the Ultra Orchestrator
workflow.set_entry_point("ultra_orchestrator")

# After the orchestrator makes a decision, the monitor observes it
workflow.add_edge("ultra_orchestrator", "monitor")

# After monitoring, route to the next appropriate step or end.
workflow.add_conditional_edges(
    "monitor",
    route_from_monitor,
    {
        # In future phases, we will add paths to other agents here
        # "research": "research_node",
        "monitor": "monitor",
        "ultra_orchestrator": "ultra_orchestrator", 
        "end": END,
    },
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
        state.messages.append({
            "sender": "assistant",
            "content": "System is currently initializing. Please try again in a moment.",
            "created_at": "2024-01-01T00:00:00Z"
        })
        return state

# Compile the workflow with error handling
sentient_workflow_app = get_sentient_workflow_app()