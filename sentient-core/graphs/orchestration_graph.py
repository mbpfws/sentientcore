from langgraph.graph import StateGraph, END
from core.models import AppState, TaskStatus, ResearchState
from core.agents.orchestrator_agent import OrchestratorAgent
from graphs.research_graph import research_app
from core.services.llm_service import LLMService

# Initialize services and agents
llm_service = LLMService()
orchestrator_agent = OrchestratorAgent(llm_service)

def run_research_graph(state: AppState) -> AppState:
    """
    This node runs the separate research graph for a pending research task.
    """
    print("---ORCHESTRATION: RUNNING RESEARCH GRAPH---")
    
    # Find the pending research task
    task_to_process = next((task for task in state.tasks if task.status == TaskStatus.PENDING and task.agent == "research"), None)

    if task_to_process:
        task_to_process.status = TaskStatus.IN_PROGRESS
        
        # Initialize the research state
        research_state = ResearchState(original_query=task_to_process.description)
        
        # Invoke the research graph
        final_research_state = research_app.invoke(research_state)
        
        # Store the final report in the task result
        task_to_process.result = final_research_state.get("final_report", "No report generated.")
        task_to_process.status = TaskStatus.COMPLETED
        print("---ORCHESTRATION: RESEARCH GRAPH COMPLETED---")
        
    return state


# Define the graph
workflow = StateGraph(AppState)

# Add the nodes for each agent
workflow.add_node("orchestrator", orchestrator_agent.invoke)
workflow.add_node("research", run_research_graph)

# Set the entry point
workflow.set_entry_point("orchestrator")

# Define the routing logic
def route_tasks(state: AppState) -> str:
    """
    Determines the next step based on the status of tasks.
    If there's a pending task, route to the appropriate agent.
    If not, end the workflow for this turn.
    """
    # Find the first pending task
    next_task = next((task for task in state.tasks if task.status == TaskStatus.PENDING), None)

    if next_task:
        print(f"Routing to agent: {next_task.agent}")
        # In the future, this can be a more complex routing table
        if next_task.agent == "research":
            return "research"
        else:
            # If agent is unknown, end for now
            return "end"
    else:
        # No pending tasks, so we end the current workflow
        print("No pending tasks. Ending workflow.")
        return "end"

# Add the conditional edges for routing
workflow.add_conditional_edges(
    "orchestrator",
    route_tasks,
    {
        "research": "research",
        "end": END,
    },
)

# After the research agent runs, route back to the orchestrator to decide the next step
# This creates a loop where the orchestrator can continue to dispatch tasks
# The 'route_tasks' function will then decide if another agent runs or if the loop ends.
workflow.add_conditional_edges(
    "research",
    route_tasks,
    {
        "research": "research", # Allow chaining multiple research tasks
        "end": END,
    },
)

# Compile the graph into a runnable app
app = workflow.compile()

# You can now run this app by calling app.invoke(initial_state)
