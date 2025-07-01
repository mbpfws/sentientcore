from core.models import Task, AppState, TaskStatus, ResearchState, ResearchStep, LogEntry
from core.services.llm_service import LLMService
import json

class ResearchAgent:
    """
    The research agent uses a multi-step process to answer complex questions.
    1. Plan: Break down the user's question into a series of research steps.
    2. Execute: Perform a web search for each step.
    3. Synthesize: Combine the findings into a final report.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def plan_steps(self, state: ResearchState) -> ResearchState:
        """
        Generates a series of research steps to answer the original query.
        """
        state.logs.append(LogEntry(source="ResearchAgent", message="Starting research planning..."))
        print("---RESEARCH AGENT: PLANNING---")
        
        system_prompt = """
You are a world-class research planner. Your role is to break down a complex user query
into a series of simple, targeted web search queries.

- The user's query is: "{query}"
- Generate a JSON array of 3 to 5 search queries that will collectively answer this query.
- Each query should be a simple string.
- Your output MUST be a valid JSON object with a single key "queries" containing the array of strings.

Example:
{
  "queries": [
    "latest advancements in LLM agentic workflows",
    "LangGraph vs AutoGen for multi-agent systems",
    "how to implement stateful agents in LangGraph"
  ]
}
"""
        
        response_json_str = self.llm_service.invoke(
            system_prompt=system_prompt.format(query=state.original_query),
            user_prompt=f"Please generate a research plan for the query: {state.original_query}",
            model="llama-3.3-70b-versatile" # Use a powerful model for planning
        )

        try:
            response_data = json.loads(response_json_str)
            queries = response_data.get("queries", [])
            state.steps = [ResearchStep(query=q) for q in queries]
            log_msg = f"Generated {len(state.steps)} research steps."
            state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
            print(log_msg)
        except (json.JSONDecodeError, KeyError) as e:
            log_msg = f"Error parsing research plan: {e}. Falling back to a single search."
            state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
            print(log_msg)
            # Handle error, maybe create a single step as a fallback
            state.steps = [ResearchStep(query=state.original_query)]
            
        return state

    def execute_search(self, state: ResearchState) -> ResearchState:
        """
        Executes the next pending web search.
        """
        print("---RESEARCH AGENT: EXECUTING SEARCH---")
        
        pending_step = next((step for step in state.steps if step.status == "pending"), None)
        if not pending_step:
            state.logs.append(LogEntry(source="ResearchAgent", message="No pending search steps found."))
            print("No pending search steps found.")
            return state

        log_msg = f"Executing search for: '{pending_step.query}'"
        state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
        print(log_msg)

        system_prompt = """
You are a world-class research analyst with access to a web search tool.
Your goal is to provide a comprehensive and objective summary of the information
found for the given query. Focus on facts, figures, and key insights.

Synthesize the findings into a clear and concise report.
"""
        
        # We use compound-beta as it's designed for this kind of tool-using search
        search_result = self.llm_service.invoke(
            system_prompt=system_prompt,
            user_prompt=pending_step.query,
            model="compound-beta"
        )
        
        pending_step.result = search_result
        pending_step.status = "completed"
        log_msg = f"Search for '{pending_step.query}' completed."
        state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
        print(log_msg)

        return state

    def synthesize_report(self, state: ResearchState) -> ResearchState:
        """
        Synthesizes the results from all research steps into a final report.
        """
        state.logs.append(LogEntry(source="ResearchAgent", message="Synthesizing final report..."))
        print("---RESEARCH AGENT: SYNTHESIZING REPORT---")

        all_results = []
        for i, step in enumerate(state.steps):
            all_results.append(f"Research Step {i+1}: Query: {step.query}\nResult: {step.result}\n---")
        
        synthesis_prompt = """
You are a world-class synthesis expert. You have been provided with a series of
research findings from multiple search queries. Your task is to synthesize this
information into a single, comprehensive, and well-structured report that directly
answers the user's original question.

The user's original question was: "{original_query}"

The research findings are as follows:
{research_data}

Generate a final, user-facing report based on this data. The report should be written
in clear, accessible language and be presented in markdown format.
"""

        final_report = self.llm_service.invoke(
            system_prompt=synthesis_prompt.format(
                original_query=state.original_query,
                research_data="\n".join(all_results)
            ),
            user_prompt="Please generate the final synthesized report.",
            model="llama-3.3-70b-versatile" # Use a powerful model for synthesis
        )

        state.final_report = final_report
        state.logs.append(LogEntry(source="ResearchAgent", message="Final report generated."))
        print("Final report generated.")
        
        return state
