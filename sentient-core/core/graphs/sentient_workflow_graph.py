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

# Initialize services and core agents
llm_service = EnhancedLLMService()
ultra_orchestrator = UltraOrchestrator(llm_service)
monitoring_agent = MonitoringAgent()

# Create the main workflow graph
workflow = StateGraph(AppState)

# --- Define the graph nodes ---

# Async wrapper for the Ultra Orchestrator
async def ultra_orchestrator_node(state: AppState) -> AppState:
    """Async wrapper for the Ultra Orchestrator invoke method."""
    return await ultra_orchestrator.invoke(state)

# Sync wrapper for the Monitoring Agent
def monitoring_agent_node(state: AppState) -> AppState:
    """Wrapper for the Monitoring Agent invoke method."""
    return monitoring_agent.invoke(state)

# Node for the Ultra Orchestrator to make decisions
workflow.add_node("ultra_orchestrator", ultra_orchestrator_node)

# Node for the Monitoring Agent to observe and log
workflow.add_node("monitor", monitoring_agent_node)

# --- Define a routing function ---
def route_from_monitor(state: AppState) -> str:
    """
    Determines the next step after monitoring based on the orchestrator's decision.
    """
    decision = state.next_action
    print(f"---ROUTING from Monitor. Decision: {decision}---")
    
    if decision == "create_plan":
        # In future phases, this will route to the appropriate agent (e.g., "research").
        # For now, we end the workflow to inspect the plan.
        return "end"
    
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
        "end": END,
    },
)


# Compile the graph into a runnable application
app = workflow.compile()