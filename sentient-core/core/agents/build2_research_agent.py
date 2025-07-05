"""Build 2 Research Agent with Groq Agentic Tooling

This agent implements autonomous research capabilities using Groq's compound-beta models
with built-in tool_use functionality for web search and knowledge synthesis.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from groq import Groq
from core.models import AppState, LogEntry, ResearchState, ResearchStep
from core.services.enhanced_llm_service import EnhancedLLMService
from core.services.memory_service import MemoryService
from core.services.session_persistence_service import get_session_persistence_service


class Build2ResearchAgent:
    """
    Advanced Research Agent for Build 2 that uses Groq's agentic tooling
    to autonomously conduct research with verbose logging and artifact generation.
    """
    
    def __init__(self):
        self.groq_client = Groq()
        self.llm_service = EnhancedLLMService()
        self.memory_service = MemoryService()
        self.persistence_service = get_session_persistence_service()
        
        # Research artifacts storage
        self.research_docs_path = Path("./memory/layer1_research_docs")
        self.research_docs_path.mkdir(parents=True, exist_ok=True)
        
        # Groq models optimized for agentic tooling
        self.research_model = "compound-beta"  # Primary model for research
        self.synthesis_model = "compound-beta-mini"  # Faster model for synthesis
        
        print("Build 2 Research Agent initialized with Groq agentic tooling")
    
    def _get_search_tools(self) -> List[Dict[str, Any]]:
        """
        Define the search tools available to the Groq model for agentic tooling.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information on a specific topic or question. Use this when you need current information, facts, or research on any subject.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to execute. Be specific and focused."
                            },
                            "focus_area": {
                                "type": "string",
                                "description": "The specific aspect or area to focus on (e.g., 'technical solutions', 'best practices', 'recent developments')"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    async def _execute_search_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a search tool call made by the Groq model.
        """
        try:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            if function_name == "web_search":
                query = arguments["query"]
                focus_area = arguments.get("focus_area", "general")
                
                print(f"[Research Agent]: Executing search - Query: '{query}', Focus: '{focus_area}'")
                
                # Use the existing LLM service's web search capability
                search_results = await self.llm_service.web_search(
                    query=query,
                    num_results=5
                )
                
                return {
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({
                        "query": query,
                        "focus_area": focus_area,
                        "results": search_results,
                        "timestamp": datetime.now().isoformat()
                    })
                }
            
            else:
                return {
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps({"error": f"Unknown function: {function_name}"})
                }
                
        except Exception as e:
            print(f"Error executing search tool: {e}")
            return {
                "tool_call_id": tool_call["id"],
                "content": json.dumps({"error": str(e)})
            }
    
    async def invoke(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for the research agent - compatible with UltraOrchestrator interface.
        """
        try:
            # Create a new AppState for this research session
            from core.models import Message
            state = AppState(
                messages=[Message(sender="user", content=user_message)],
                logs=[],
                session_id=session_id or "default"
            )
            
            # Conduct the research
            result_state = await self.conduct_research(user_message, state, session_id or "default")
            
            # Extract the research results from the state
            research_logs = [log.message for log in result_state.logs if log.source == "Build2_ResearchAgent"]
            
            return {
                "status": "success",
                "research_query": user_message,
                "logs": research_logs,
                "session_id": session_id,
                "message": "Research completed successfully. Detailed findings have been generated and saved."
            }
            
        except Exception as e:
            error_msg = f"Research agent error: {str(e)}"
            print(error_msg)
            return {
                "status": "error",
                "research_query": user_message,
                "error": error_msg,
                "session_id": session_id,
                "message": f"I encountered an issue while conducting research: {str(e)}"
            }
    
    async def conduct_research(self, query: str, state: AppState, session_id: str) -> AppState:
        """
        Conduct autonomous research using Groq's agentic tooling.
        """
        try:
            # Initialize research state
            research_state = ResearchState(
                original_query=query,
                steps=[],
                logs=[],
                final_report=None
            )
            
            # Log research initiation
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message=f"[Research Agent]: Starting autonomous research for query: '{query}'"
            ))
            
            # Phase 1: Research Planning and Execution
            planning_prompt = f"""
You are an expert research agent with access to web search tools. Your task is to conduct comprehensive research on the following query:

QUERY: {query}

Your approach should be:
1. Break down the query into specific research areas
2. Use the web_search tool to gather information on each area
3. Think step-by-step and be verbose about your research process
4. Generate multiple targeted search queries to get comprehensive coverage
5. Analyze and synthesize the information you find

Be extremely verbose about your thought process. Explain each search you're about to make and why.

Start by analyzing the query and planning your research approach.
"""
            
            # Initial research conversation
            messages = [
                {"role": "system", "content": "You are a thorough research agent. Be verbose and explain your thinking process."},
                {"role": "user", "content": planning_prompt}
            ]
            
            # Conduct iterative research with tool calls
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                state.logs.append(LogEntry(
                    source="Build2_ResearchAgent",
                    message=f"[Research Agent]: Research iteration {iteration}/{max_iterations}"
                ))
                
                # Call Groq with tool use capability
                response = self.groq_client.chat.completions.create(
                    model=self.research_model,
                    messages=messages,
                    tools=self._get_search_tools(),
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=2000
                )
                
                message = response.choices[0].message
                
                # Log the agent's reasoning
                if message.content:
                    state.logs.append(LogEntry(
                        source="Build2_ResearchAgent",
                        message=f"[Research Agent]: {message.content}"
                    ))
                
                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                # Execute tool calls if any
                if message.tool_calls:
                    tool_results = []
                    
                    for tool_call in message.tool_calls:
                        # Log the search being executed
                        try:
                            args = json.loads(tool_call.function.arguments)
                            query_text = args.get("query", "unknown")
                            state.logs.append(LogEntry(
                                source="Build2_ResearchAgent",
                                message=f"[Research Agent]: Executing search query: '{query_text}'"
                            ))
                        except:
                            pass
                        
                        # Execute the tool call
                        result = await self._execute_search_tool(tool_call.dict())
                        tool_results.append(result)
                        
                        # Log search completion
                        state.logs.append(LogEntry(
                            source="Build2_ResearchAgent",
                            message=f"[Research Agent]: Search completed, processing results..."
                        ))
                    
                    # Add tool results to conversation
                    for result in tool_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": result["content"]
                        })
                    
                    # Continue the conversation to get analysis
                    messages.append({
                        "role": "user",
                        "content": "Please analyze the search results and determine if you need more information or if you can proceed to synthesize your findings. If you need more information, make additional searches. If you have enough, provide a comprehensive analysis."
                    })
                
                else:
                    # No more tool calls, research is complete
                    break
            
            # Phase 2: Consolidation and Report Generation
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message="[Research Agent]: Research phase complete, generating consolidated report..."
            ))
            
            # Generate final report
            final_report = await self._generate_final_report(messages, query, state)
            
            # Save research artifacts
            await self._save_research_artifacts(query, final_report, session_id, state)
            
            # Update state with research completion
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message="[Research Agent]: Research completed successfully. Report generated and saved."
            ))
            
            return state
            
        except Exception as e:
            error_msg = f"Error during research: {str(e)}"
            print(error_msg)
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message=f"[Research Agent]: ERROR - {error_msg}"
            ))
            return state
    
    async def _generate_final_report(self, conversation_messages: List[Dict], original_query: str, state: AppState) -> str:
        """
        Generate a comprehensive final report from the research conversation.
        """
        try:
            synthesis_prompt = f"""
Based on the research conversation above, generate a comprehensive, well-structured research report for the original query: "{original_query}"

The report should include:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Recommendations
5. Sources and References
6. Conclusion

Format the report in clean Markdown with proper headings, bullet points, and structure.
Make it professional and actionable.
"""
            
            # Add synthesis request to conversation
            synthesis_messages = conversation_messages + [
                {"role": "user", "content": synthesis_prompt}
            ]
            
            # Generate final report using synthesis model
            response = self.groq_client.chat.completions.create(
                model=self.synthesis_model,
                messages=synthesis_messages,
                temperature=0.1,
                max_tokens=3000
            )
            
            final_report = response.choices[0].message.content
            
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message="[Research Agent]: Final report generated successfully"
            ))
            
            return final_report
            
        except Exception as e:
            error_msg = f"Error generating final report: {str(e)}"
            print(error_msg)
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message=f"[Research Agent]: ERROR - {error_msg}"
            ))
            return f"# Research Report\n\nError generating report: {error_msg}"
    
    async def _save_research_artifacts(self, query: str, report: str, session_id: str, state: AppState):
        """
        Save research artifacts to long-term memory and generate PDF.
        """
        try:
            # Create filename-safe query string
            safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_query = safe_query.replace(' ', '_')[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save Markdown report
            md_filename = f"research_{timestamp}_{safe_query}.md"
            md_path = self.research_docs_path / md_filename
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message=f"[Research Agent]: Markdown report saved to {md_filename}"
            ))
            
            # Generate PDF using markdown library
            try:
                import markdown
                import pdfkit
                
                # Convert markdown to HTML
                html_content = markdown.markdown(report, extensions=['tables', 'fenced_code'])
                
                # Add basic CSS styling
                styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Research Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        h1 {{ border-bottom: 2px solid #333; }}
        h2 {{ border-bottom: 1px solid #666; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; border-radius: 3px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""
                
                # Save PDF
                pdf_filename = f"research_{timestamp}_{safe_query}.pdf"
                pdf_path = self.research_docs_path / pdf_filename
                
                # Configure pdfkit options
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': "UTF-8",
                    'no-outline': None
                }
                
                pdfkit.from_string(styled_html, str(pdf_path), options=options)
                
                state.logs.append(LogEntry(
                    source="Build2_ResearchAgent",
                    message=f"[Research Agent]: PDF report saved to {pdf_filename}"
                ))
                
            except ImportError:
                state.logs.append(LogEntry(
                    source="Build2_ResearchAgent",
                    message="[Research Agent]: PDF generation skipped (pdfkit not available)"
                ))
            except Exception as pdf_error:
                state.logs.append(LogEntry(
                    source="Build2_ResearchAgent",
                    message=f"[Research Agent]: PDF generation failed: {str(pdf_error)}"
                ))
            
            # Save to memory service
            await self.memory_service.store_research_finding(
                query=query,
                findings=report,
                source="Build2_ResearchAgent",
                session_id=session_id
            )
            
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message="[Research Agent]: Research findings stored in long-term memory"
            ))
            
        except Exception as e:
            error_msg = f"Error saving research artifacts: {str(e)}"
            print(error_msg)
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message=f"[Research Agent]: ERROR - {error_msg}"
            ))


# Global instance
_build2_research_agent: Optional[Build2ResearchAgent] = None


def get_build2_research_agent() -> Build2ResearchAgent:
    """Get the global Build 2 research agent instance."""
    global _build2_research_agent
    if _build2_research_agent is None:
        _build2_research_agent = Build2ResearchAgent()
    return _build2_research_agent