from core.models import EnhancedTask, AppState, TaskStatus, ResearchState, ResearchStep, LogEntry
from core.services.llm_service import EnhancedLLMService
from core.agents.base_agent import BaseAgent, AgentCapability, ActivityType
from core.tools.web_search_tools import WebSearchTools
import json
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ResearchMode(Enum):
    """Research modes that determine the depth and approach of research."""
    KNOWLEDGE = "knowledge"  # Multiple sources, keywords search, consolidated report
    DEEP = "deep"  # In-depth analysis with reasoning, citations, sophisticated evaluation
    BEST_IN_CLASS = "best_in_class"  # Gather options/alternatives, consolidate what's best

class ResearchAgent(BaseAgent):
    """
    Advanced research agent with three sophisticated research modes:
    1. Knowledge Research: Multi-source keyword search with consolidation
    2. Deep Research: In-depth analysis with citations and sophisticated reasoning
    3. Best-in-Class Research: Comparative analysis to find optimal solutions
    """

    def __init__(self, llm_service: EnhancedLLMService, agent_id: str = "research_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Research Agent",
            capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
            description="Advanced research agent with multiple research modes and agentic tooling"
        )
        self.llm_service = llm_service
        self.web_search_tools = WebSearchTools()
        
        # Model selection based on research complexity - using Groq compound-beta models
        self.models = {
            "planning": "compound-beta",  # Best for planning and tool use
            "search": "compound-beta",  # Use compound-beta for agentic search with tools
            "synthesis": "compound-beta",  # Best for complex synthesis
            "reasoning": "compound-beta"  # Best for deep reasoning
        }
        
        # Register web search tools with LLM service
        self._register_search_tools()
    
    def _register_search_tools(self):
        """Register web search tools with the LLM service for agentic tooling"""
        try:
            # Register tool functions
            agentic_tools = self.web_search_tools.get_tool_functions()
            for tool in agentic_tools:
                self.llm_service.register_tool(tool)
            
            # Note: Cannot use await in __init__, will log during first task execution
            print("Web search tools registered successfully")
        except Exception as e:
            print(f"Failed to register web search tools: {str(e)}")

    def can_handle_task(self, task: EnhancedTask) -> bool:
        """
        Determines if this agent can handle the given task.
        """
        research_keywords = ['research', 'analyze', 'investigate', 'study', 'explore', 'find', 'search']
        task_description = task.description.lower()
        return any(keyword in task_description for keyword in research_keywords)
    
    async def process_task(self, task: EnhancedTask, state: Optional[AppState] = None) -> Dict:
        """
        Processes a research task by executing the full research workflow.
        """
        try:
            await self.log_activity(ActivityType.TASK_START, f"Processing research task: {task.description}")
            
            # Create research state from task
            research_state = ResearchState(
                original_query=task.description,
                steps=[],
                final_report="",
                continual_search_suggestions=[],
                logs=[]
            )
            
            # Execute research workflow
            research_state = await self.plan_steps(research_state)
            
            # Execute all searches
            while any(step.status == "pending" for step in research_state.steps):
                research_state = await self.execute_search(research_state)
            
            # Synthesize final report
            research_state = await self.synthesize_report(research_state)
            
            await self.log_activity(ActivityType.TASK_COMPLETED, f"Research task completed: {task.description}")
            
            return {
                "status": "completed",
                "result": research_state.final_report,
                "suggestions": research_state.continual_search_suggestions,
                "logs": research_state.logs
            }
            
        except Exception as e:
            await self.handle_error(e, f"Error processing research task: {task.description}")
            return {
                "status": "error",
                "error": str(e),
                "logs": []
            }

    async def determine_research_mode(self, query: str) -> ResearchMode:
        """
        Intelligently determines the appropriate research mode based on query analysis.
        """
        # Use compound-beta for intelligent mode detection
        mode_prompt = f"""
Analyze this research query and determine the most appropriate research mode:

Query: "{query}"

Research Modes:
1. KNOWLEDGE: For general information gathering, factual queries, "what is", "how does", basic explanations
2. DEEP: For complex analysis, "why", "analyze", "evaluate", "compare in detail", academic research
3. BEST_IN_CLASS: For finding optimal solutions, "best", "recommend", "which should I choose", comparative evaluation

Respond with only one word: KNOWLEDGE, DEEP, or BEST_IN_CLASS
"""
        
        response = await self.llm_service.invoke(
            system_prompt="You are an expert at categorizing research queries.",
            user_prompt=mode_prompt,
            model=self.models["planning"]
        )
        
        # Parse response and default to KNOWLEDGE if unclear
        response_clean = response.strip().upper()
        if "DEEP" in response_clean:
            return ResearchMode.DEEP
        elif "BEST_IN_CLASS" in response_clean or "BEST" in response_clean:
            return ResearchMode.BEST_IN_CLASS
        else:
            return ResearchMode.KNOWLEDGE

    async def plan_steps(self, state: ResearchState) -> ResearchState:
        """
        Generates research steps based on the determined research mode.
        """
        await self.log_activity(ActivityType.TASK_PROGRESS, "Starting intelligent research planning")
        state.logs.append(LogEntry(source="ResearchAgent", message="Starting intelligent research planning..."))
        print("---RESEARCH AGENT: INTELLIGENT PLANNING---")
        
        # Determine research mode
        research_mode = await self.determine_research_mode(state.original_query)
        await self.log_activity(ActivityType.TASK_PROGRESS, f"Selected research mode: {research_mode.value}")
        state.logs.append(LogEntry(source="ResearchAgent", message=f"Selected research mode: {research_mode.value}"))
        print(f"Research mode selected: {research_mode.value}")
        
        # Generate mode-specific research plan
        if research_mode == ResearchMode.KNOWLEDGE:
            return await self._plan_knowledge_research(state)
        elif research_mode == ResearchMode.DEEP:
            return await self._plan_deep_research(state)
        else:  # BEST_IN_CLASS
            return await self._plan_best_in_class_research(state)

    async def _plan_knowledge_research(self, state: ResearchState) -> ResearchState:
        """Plans multi-source keyword search for knowledge gathering."""
        system_prompt = """
You are a research planning expert. Create a comprehensive knowledge research plan.
Break down the query into 4-6 specific search queries that will gather information from multiple sources and perspectives.

**CRITICAL:** Output ONLY valid JSON with this exact structure:
{
  "queries": [
    "specific search query 1",
    "specific search query 2",
    "specific search query 3"
  ]
}
"""
        
        response = await self._get_json_response(system_prompt, state.original_query, "planning")
        queries = response.get("queries", [state.original_query])
        state.steps = [ResearchStep(query=q) for q in queries]
        
        log_msg = f"Generated {len(state.steps)} knowledge research steps."
        state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
        return state

    async def _plan_deep_research(self, state: ResearchState) -> ResearchState:
        """Plans in-depth research with reasoning and analysis."""
        system_prompt = """
You are a deep research strategist. Create a sophisticated research plan for in-depth analysis.
Include background research, current state analysis, expert perspectives, and future implications.

**CRITICAL:** Output ONLY valid JSON with this exact structure:
{
  "queries": [
    "background and historical context of [topic]",
    "current state and recent developments in [topic]",
    "expert opinions and academic research on [topic]",
    "challenges and limitations of [topic]",
    "future trends and implications of [topic]"
  ]
}
"""
        
        response = await self._get_json_response(system_prompt, state.original_query, "planning")
        queries = response.get("queries", [state.original_query])
        state.steps = [ResearchStep(query=q) for q in queries]
        
        log_msg = f"Generated {len(state.steps)} deep research steps."
        state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
        return state

    async def _plan_best_in_class_research(self, state: ResearchState) -> ResearchState:
        """Plans comparative research to find optimal solutions."""
        system_prompt = """
You are a comparative analysis expert. Create a research plan to identify the best options and alternatives.
Focus on gathering different approaches, solutions, or tools, then evaluating their strengths and weaknesses.

**CRITICAL:** Output ONLY valid JSON with this exact structure:
{
  "queries": [
    "top options and alternatives for [topic]",
    "comparison of leading solutions in [topic]",
    "pros and cons of different [topic] approaches",
    "expert recommendations for [topic]",
    "case studies and real-world examples of [topic]"
  ]
}
"""
        
        response = await self._get_json_response(system_prompt, state.original_query, "planning")
        queries = response.get("queries", [state.original_query])
        state.steps = [ResearchStep(query=q) for q in queries]
        
        log_msg = f"Generated {len(state.steps)} comparative research steps."
        state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
        return state

    async def execute_search(self, state: ResearchState, stream_callback=None) -> ResearchState:
        """
        Executes the next pending web search using Groq's compound-beta model with agentic tooling.
        Supports streaming for real-time updates and verbose search results.
        """
        print("---RESEARCH AGENT: EXECUTING AGENTIC SEARCH---")
        
        pending_step = next((step for step in state.steps if step.status == "pending"), None)
        if not pending_step:
            await self.log_activity(ActivityType.TASK_PROGRESS, "No pending search steps found")
            state.logs.append(LogEntry(source="ResearchAgent", message="No pending search steps found."))
            return state

        log_msg = f"Executing agentic search for: '{pending_step.query}'"
        await self.log_activity(ActivityType.PROCESSING, log_msg)
        state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
        print(log_msg)
        
        # Notify frontend about search start
        if stream_callback:
            stream_callback({
                "type": "search_start",
                "query": pending_step.query,
                "message": log_msg
            })

        # Use compound-beta with agentic tooling for comprehensive search
        system_prompt = """
You are a world-class research analyst with access to advanced web search tools.
Conduct thorough, multi-faceted research using the available search tools.

Instructions:
1. Use multiple search tools to gather comprehensive information
2. Perform both general and technical searches when relevant
3. Look for current, accurate information from multiple sources
4. Include specific facts, figures, data points, and examples
5. Note any conflicting information, debates, or different perspectives
6. Provide detailed source context and credibility assessment
7. Be verbose and thorough in your analysis

Search Strategy:
- Start with general web search for broad understanding
- Use technical search for detailed/specialized information
- Use research search for academic or in-depth analysis
- Synthesize findings from all searches into comprehensive results

Format your response with clear structure, detailed findings, and source attribution.
"""
        
        try:
            # Use generate_with_tools for agentic search capabilities
            search_result = await self.llm_service.generate_with_tools(
                system_prompt=system_prompt,
                user_prompt=f"Research this query thoroughly using all available search tools: {pending_step.query}",
                model=self.models["search"],
                stream=stream_callback is not None
            )
            
            # Handle streaming response
            if stream_callback and hasattr(search_result, '__iter__'):
                accumulated_result = ""
                for chunk in search_result:
                    if hasattr(chunk, 'choices') and chunk.choices[0].delta.content:
                        content_chunk = chunk.choices[0].delta.content
                        accumulated_result += content_chunk
                        
                        # Stream to frontend
                        stream_callback({
                            "type": "search_chunk",
                            "query": pending_step.query,
                            "chunk": content_chunk,
                            "accumulated": accumulated_result
                        })
                search_result = accumulated_result
                        
        except Exception as e:
            # Fallback to regular generation without tools
            print(f"Agentic search failed, falling back to regular search: {e}")
            await self.log_activity(ActivityType.ERROR, f"Agentic search failed: {str(e)}")
            
            search_result = await self.llm_service.generate(
                system_prompt="You are a research analyst. Provide comprehensive research findings.",
                user_prompt=f"Research this query thoroughly: {pending_step.query}",
                model=self.models["search"]
            )
        
        pending_step.result = search_result
        pending_step.status = "completed"
        
        log_msg = f"Agentic search for '{pending_step.query}' completed with verbose results."
        await self.log_activity(ActivityType.PROCESSING, log_msg)
        state.logs.append(LogEntry(source="ResearchAgent", message=log_msg))
        print(log_msg)
        
        # Notify frontend about search completion
        if stream_callback:
            stream_callback({
                "type": "search_complete",
                "query": pending_step.query,
                "result": search_result,
                "message": log_msg
            })

        return state

    async def synthesize_report(self, state: ResearchState) -> ResearchState:
        """
        Synthesizes research findings into a comprehensive report based on research mode.
        """
        await self.log_activity(ActivityType.PROCESSING, "Synthesizing comprehensive report")
        state.logs.append(LogEntry(source="ResearchAgent", message="Synthesizing comprehensive report..."))
        print("---RESEARCH AGENT: ADVANCED SYNTHESIS---")

        # Determine research mode from the number and type of steps
        research_mode = self._infer_research_mode(state)
        
        all_results = []
        for i, step in enumerate(state.steps):
            all_results.append(f"**Research Finding {i+1}:** {step.query}\n\n{step.result}\n\n---\n")
        
        if research_mode == ResearchMode.KNOWLEDGE:
            return await self._synthesize_knowledge_report(state, all_results)
        elif research_mode == ResearchMode.DEEP:
            return await self._synthesize_deep_report(state, all_results)
        else:  # BEST_IN_CLASS
            return await self._synthesize_best_in_class_report(state, all_results)

    async def _synthesize_knowledge_report(self, state: ResearchState, all_results: List[str]) -> ResearchState:
        """Synthesizes a knowledge-focused report."""
        synthesis_prompt = f"""
You are a knowledge synthesis expert. Create a comprehensive, well-structured report that consolidates multiple sources of information.

Original Query: "{state.original_query}"

Research Findings:
{chr(10).join(all_results)}

Create a report with:
1. Executive Summary
2. Key Findings (organized by themes)
3. Important Facts and Figures
4. Conclusion
5. Sources and References (where mentioned in findings)

**CRITICAL:** Output ONLY valid JSON:
{{
  "report": "# Knowledge Report: [Title]\\n\\n## Executive Summary\\n[content]\\n\\n## Key Findings\\n[content]\\n\\n## Important Facts and Figures\\n[content]\\n\\n## Conclusion\\n[content]\\n\\n## Sources and References\\n[content]",
  "continual_search_suggestions": [
    "Follow-up question 1",
    "Follow-up question 2",
    "Follow-up question 3",
    "Follow-up question 4"
  ]
}}
"""
        
        response = await self._get_json_response(synthesis_prompt, "", "synthesis")
        state.final_report = response.get("report", "Failed to generate knowledge report.")
        state.continual_search_suggestions = response.get("continual_search_suggestions", [])
        
        await self.log_activity(ActivityType.PROCESSING, "Knowledge report synthesized")
        state.logs.append(LogEntry(source="ResearchAgent", message="Knowledge report synthesized."))
        return state

    async def _synthesize_deep_report(self, state: ResearchState, all_results: List[str]) -> ResearchState:
        """Synthesizes an in-depth analytical report."""
        synthesis_prompt = f"""
You are a deep research analyst. Create a sophisticated, in-depth report with critical analysis and reasoning.

Original Query: "{state.original_query}"

Research Findings:
{chr(10).join(all_results)}

Create a comprehensive analytical report with:
1. Executive Summary
2. Background and Context
3. Current State Analysis
4. Critical Analysis and Evaluation
5. Expert Perspectives and Debates
6. Challenges and Limitations
7. Future Implications and Trends
8. Conclusions and Recommendations
9. Citations and Sources

Use sophisticated reasoning, identify patterns, contradictions, and provide deep insights.

**CRITICAL:** Output ONLY valid JSON:
{{
  "report": "# Deep Research Analysis: [Title]\\n\\n## Executive Summary\\n[content]\\n\\n## Background and Context\\n[content]\\n\\n## Current State Analysis\\n[content]\\n\\n## Critical Analysis and Evaluation\\n[content]\\n\\n## Expert Perspectives and Debates\\n[content]\\n\\n## Challenges and Limitations\\n[content]\\n\\n## Future Implications and Trends\\n[content]\\n\\n## Conclusions and Recommendations\\n[content]\\n\\n## Citations and Sources\\n[content]",
  "continual_search_suggestions": [
    "Deep follow-up question 1",
    "Deep follow-up question 2", 
    "Deep follow-up question 3",
    "Deep follow-up question 4"
  ]
}}
"""
        
        response = await self._get_json_response(synthesis_prompt, "", "synthesis")
        state.final_report = response.get("report", "Failed to generate deep research report.")
        state.continual_search_suggestions = response.get("continual_search_suggestions", [])
        
        await self.log_activity(ActivityType.PROCESSING, "Deep research report synthesized")
        state.logs.append(LogEntry(source="ResearchAgent", message="Deep research report synthesized."))
        return state

    async def _synthesize_best_in_class_report(self, state: ResearchState, all_results: List[str]) -> ResearchState:
        """Synthesizes a best-in-class comparative report."""
        synthesis_prompt = f"""
You are a comparative analysis expert. Create a definitive report that identifies the best options and provides clear recommendations.

Original Query: "{state.original_query}"

Research Findings:
{chr(10).join(all_results)}

Create a comparative analysis report with:
1. Executive Summary with Top Recommendation
2. Available Options and Alternatives
3. Detailed Comparison Matrix
4. Strengths and Weaknesses Analysis
5. Use Case Scenarios
6. Expert Recommendations
7. Final Verdict and Rationale
8. Implementation Considerations

Focus on helping the reader make the best decision for their specific context.

**CRITICAL:** Output ONLY valid JSON:
{{
  "report": "# Best-in-Class Analysis: [Title]\\n\\n## Executive Summary and Top Recommendation\\n[content]\\n\\n## Available Options and Alternatives\\n[content]\\n\\n## Detailed Comparison Matrix\\n[content]\\n\\n## Strengths and Weaknesses Analysis\\n[content]\\n\\n## Use Case Scenarios\\n[content]\\n\\n## Expert Recommendations\\n[content]\\n\\n## Final Verdict and Rationale\\n[content]\\n\\n## Implementation Considerations\\n[content]",
  "continual_search_suggestions": [
    "Implementation question 1",
    "Alternative comparison question 2",
    "Cost/benefit analysis question 3",
    "Specific use case question 4"
  ]
}}
"""
        
        response = await self._get_json_response(synthesis_prompt, "", "synthesis")
        state.final_report = response.get("report", "Failed to generate best-in-class report.")
        state.continual_search_suggestions = response.get("continual_search_suggestions", [])
        
        await self.log_activity(ActivityType.PROCESSING, "Best-in-class report synthesized")
        state.logs.append(LogEntry(source="ResearchAgent", message="Best-in-class report synthesized."))
        return state

    async def _get_json_response(self, system_prompt: str, user_prompt: str, model_type: str) -> Dict:
        """Helper method to get and parse JSON responses with error handling."""
        try:
            response = await self.llm_service.invoke(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.models[model_type]
            )
            
            # Clean response and extract JSON
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
            
        except json.JSONDecodeError as e:
            await self.log_activity(ActivityType.ERROR, f"JSON parsing error: {e}")
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response}")
            return {"queries": [user_prompt]} if model_type == "planning" else {"report": response, "continual_search_suggestions": []}
        except Exception as e:
            await self.log_activity(ActivityType.ERROR, f"Error getting response: {e}")
            print(f"Error getting response: {e}")
            return {"queries": [user_prompt]} if model_type == "planning" else {"report": f"Error: {e}", "continual_search_suggestions": []}

    def _clean_json_response(self, response: str) -> str:
        """Cleans LLM response to extract valid JSON."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Find JSON object boundaries
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start != -1 and end > start:
            return response[start:end]
        
        return response.strip()

    def _infer_research_mode(self, state: ResearchState) -> ResearchMode:
        """Infers research mode from the types of research steps."""
        step_queries = " ".join([step.query for step in state.steps]).lower()
        
        if any(keyword in step_queries for keyword in ["compare", "best", "alternatives", "options", "versus"]):
            return ResearchMode.BEST_IN_CLASS
        elif any(keyword in step_queries for keyword in ["deep", "analysis", "evaluate", "expert", "academic", "implications"]):
            return ResearchMode.DEEP
        else:
            return ResearchMode.KNOWLEDGE