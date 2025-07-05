from langgraph.graph import StateGraph, END
from core.models import ResearchState
from core.agents.research_agent import ResearchAgent
from core.services.llm_service import EnhancedLLMService

def should_continue(state: ResearchState) -> str:
    """
    Determines the next step in the research process.
    """
    if any(step.status == "pending" for step in state.steps):
        return "execute_search"
    else:
        return "synthesize_report"

def create_execute_search_with_streaming(research_agent):
    """Create execute_search_with_streaming function with research_agent closure."""
    async def execute_search_with_streaming(state: ResearchState) -> ResearchState:
        """
        Wrapper function to execute search with streaming support.
        """
        # Get streaming callback from state if available
        stream_callback = getattr(state, 'stream_callback', None)
        return await research_agent.execute_search(state, stream_callback)
    return execute_search_with_streaming

def create_research_graph(llm_service: EnhancedLLMService):
    """Create and return a compiled research graph."""
    research_agent = ResearchAgent(llm_service)
    execute_search_with_streaming = create_execute_search_with_streaming(research_agent)
    
    # Define the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("plan_steps", research_agent.plan_steps)
    workflow.add_node("execute_search", execute_search_with_streaming)
    workflow.add_node("synthesize_report", research_agent.synthesize_report)
    
    # Set entry and exit points
    workflow.set_entry_point("plan_steps")
    workflow.add_edge("synthesize_report", END)
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "plan_steps",
        should_continue,
        {
            "execute_search": "execute_search",
            "synthesize_report": "synthesize_report",
        },
    )
    workflow.add_conditional_edges(
        "execute_search",
        should_continue,
        {
            "execute_search": "execute_search",
            "synthesize_report": "synthesize_report",
        },
    )
    
    # Compile and return the graph
    return workflow.compile()
# Create a default research_app instance using EnhancedLLMService
llm_service = EnhancedLLMService()
research_app = create_research_graph(llm_service)
