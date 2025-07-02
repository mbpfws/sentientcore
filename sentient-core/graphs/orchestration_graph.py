from langgraph.graph import StateGraph, END
from core.models import AppState, TaskStatus, ResearchState, LogEntry
from core.agents.orchestrator_agent import OrchestratorAgent
from core.agents.research_agent import ResearchAgent
from graphs.research_graph import research_app
from core.services.llm_service import LLMService
import json

# --- Placeholder Design Agent ---
class DesignAgent:
    def invoke(self, state: AppState) -> AppState:
        print("---DESIGN AGENT (Placeholder)---")
        task = next((t for t in state.tasks if t.id == state.task_to_run_id), None)
        if task:
            task.result = "This is a placeholder for the design agent's output. The full agent has not been implemented yet."
            task.status = TaskStatus.COMPLETED
            state.logs.append(LogEntry(source="DesignAgent", message=f"Completed placeholder task: {task.description}"))
        return state

# Initialize services and agents
llm_service = LLMService()
orchestrator_agent = OrchestratorAgent(llm_service)
design_agent = DesignAgent()

def run_research_graph(state: AppState) -> AppState:
    """
    This node runs the separate research graph for a pending research task.
    """
    state.logs.append(LogEntry(source="OrchestrationGraph", message="Entering research sub-graph..."))
    print("---ORCHESTRATION: RUNNING RESEARCH GRAPH---")
    
    task_to_process = next((t for t in state.tasks if t.id == state.task_to_run_id), None)

    if task_to_process:
        task_to_process.status = TaskStatus.IN_PROGRESS
        
        # Initialize the research state
        research_state = ResearchState(original_query=task_to_process.description)
        
        # Invoke the research graph
        final_research_state = research_app.invoke(research_state)
        
        # Merge logs from the research state into the main app state
        if "logs" in final_research_state:
            state.logs.extend(final_research_state["logs"])

        # --- Robustly unpack the research result ---
        report_content = final_research_state.get("final_report", "No report generated.")
        suggestions = final_research_state.get("continual_search_suggestions", [])

        # If the report content is a string that looks like JSON, parse it.
        # This handles cases where the LLM fails to return a clean dict.
        if isinstance(report_content, str) and report_content.strip().startswith("{"):
            try:
                report_data = json.loads(report_content)
                report_content = report_data.get("report", report_content)
                suggestions = report_data.get("continual_search_suggestions", suggestions)
            except json.JSONDecodeError:
                # If parsing fails, we just use the raw string as the report.
                state.logs.append(LogEntry(source="OrchestrationGraph", message="Could not parse final_report as JSON."))
                pass
            
        # Store the final report and suggestions in the task result
        task_to_process.result = report_content
        task_to_process.follow_up_questions = suggestions
        task_to_process.status = TaskStatus.COMPLETED
        state.logs.append(LogEntry(source="OrchestrationGraph", message="Research sub-graph complete. Report and suggestions generated."))
        print("---ORCHESTRATION: RESEARCH GRAPH COMPLETED---")
        
    return state

def route_tasks(state: AppState) -> str:
    """
    Determines the routing logic.
    If a specific task is selected to run, route to the corresponding agent.
    Otherwise, route to the orchestrator to plan tasks from the user prompt.
    """
    print("---ROUTING---")
    if state.task_to_run_id:
        task_to_run = next((task for task in state.tasks if task.id == state.task_to_run_id), None)
        if task_to_run:
            log_msg = f"Routing to selected agent: {task_to_run.agent}"
            state.logs.append(LogEntry(source="OrchestrationGraph", message=log_msg))
            print(log_msg)
            return task_to_run.agent
    
    log_msg = "No specific task selected. Routing to orchestrator for planning."
    state.logs.append(LogEntry(source="OrchestrationGraph", message=log_msg))
    print(log_msg)
    return "orchestrator"

# Define the graph
workflow = StateGraph(AppState)

# A router node that does nothing but allow the graph to start
def router_node(state: AppState) -> dict:
    """A dummy node that serves as the entry point for routing."""
    return {}

# Add the nodes
workflow.add_node("router", router_node)
workflow.add_node("orchestrator", orchestrator_agent.invoke)
workflow.add_node("research", run_research_graph)
workflow.add_node("design", design_agent.invoke)

# Set the entry point
workflow.set_entry_point("router")

# Add conditional routing from the router node
workflow.add_conditional_edges(
    "router",
    route_tasks,
    {
        "orchestrator": "orchestrator",
        "research": "research",
        "design": "design",
        "end": END, # Added for safety, though route_tasks should not return "end"
    },
)

# The orchestrator's job is now just to plan, so it always ends the turn.
workflow.add_edge("orchestrator", END)
workflow.add_edge("research", END)
workflow.add_edge("design", END)

# Compile the graph into a runnable app
app = workflow.compile()

# You can now run this app by calling app.invoke(initial_state)
