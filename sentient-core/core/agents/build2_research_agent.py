"""Build 2 Research Agent with Groq Agentic Tooling

This agent implements autonomous research capabilities using Groq's compound-beta models
with built-in agentic tooling for web search and code execution.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from groq import Groq
from core.models import AppState, LogEntry, ResearchState, ResearchStep, MemoryLayer
from core.services.memory_service import MemoryService, MemoryType
from core.services.session_persistence_service import get_session_persistence_service


class Build2ResearchAgent:
    """
    Advanced Research Agent for Build 2 that uses Groq's compound-beta models
    with built-in agentic tooling for autonomous research and knowledge synthesis.
    """
    
    def __init__(self):
        self.groq_client = Groq()
        self.memory_service = MemoryService()
        self.persistence_service = get_session_persistence_service()
        
        # Research artifacts storage
        self.research_docs_path = Path("./memory/layer1_research_docs")
        self.research_docs_path.mkdir(parents=True, exist_ok=True)
        
        # Groq compound models with built-in agentic tooling
        self.research_model = "compound-beta"  # Primary model with built-in web search and code execution
        self.synthesis_model = "compound-beta-mini"  # Faster model for synthesis with single tool call support
        
        print("Build 2 Research Agent initialized with Groq compound-beta agentic tooling")
    
    # Note: compound-beta models have built-in agentic tooling
    # No custom tool definitions needed - web search and code execution are built-in
    
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
            
            # Phase 1: Research Planning and Execution using compound-beta
            research_prompt = f"""
You are an expert research agent with built-in web search capabilities. Your task is to conduct comprehensive research on the following query:

QUERY: {query}

Your approach should be:
1. Break down the query into specific research areas
2. Use your built-in web search capabilities to gather current information
3. Think step-by-step and be verbose about your research process
4. Generate multiple targeted searches to get comprehensive coverage
5. Analyze and synthesize the information you find
6. Provide a thorough, well-structured research report

Be extremely verbose about your thought process. Explain each search you're about to make and why. Use your web search capabilities to find the most current and relevant information.

Start by analyzing the query and conducting your research.
"""
            
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message="[Research Agent]: Starting research with compound-beta agentic tooling"
            ))
            
            # Use compound-beta model with built-in agentic tooling
            try:
                response = self.groq_client.chat.completions.create(
                    model=self.research_model,
                    messages=[
                        {"role": "system", "content": "You are a thorough research agent with built-in web search capabilities. Be verbose and explain your thinking process."},
                        {"role": "user", "content": research_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=4000
                )
                
                research_content = response.choices[0].message.content
                
                # Log the research process
                state.logs.append(LogEntry(
                    source="Build2_ResearchAgent",
                    message="[Research Agent]: Research completed using compound-beta agentic tooling"
                ))
                
                # Check if tools were executed
                if hasattr(response.choices[0].message, 'executed_tools') and response.choices[0].message.executed_tools:
                    executed_tools = response.choices[0].message.executed_tools
                    state.logs.append(LogEntry(
                        source="Build2_ResearchAgent",
                        message=f"[Research Agent]: Executed {len(executed_tools)} tool calls during research"
                    ))
                    
                    # Log details of executed tools
                    for i, tool in enumerate(executed_tools, 1):
                        # Handle ExecutedTool object properly
                        if hasattr(tool, 'type'):
                            tool_type = tool.type
                        elif hasattr(tool, 'name'):
                            tool_type = tool.name
                        elif isinstance(tool, dict):
                            tool_type = tool.get('type', 'unknown')
                        else:
                            tool_type = str(type(tool).__name__)
                        
                        state.logs.append(LogEntry(
                            source="Build2_ResearchAgent",
                            message=f"[Research Agent]: Tool {i}: {tool_type}"
                        ))
                
            except Exception as e:
                error_msg = f"Error during compound-beta research: {str(e)}"
                print(error_msg)
                state.logs.append(LogEntry(
                    source="Build2_ResearchAgent",
                    message=f"[Research Agent]: ERROR - {error_msg}"
                ))
                research_content = f"Research failed due to error: {error_msg}"
            
            # Phase 2: Consolidation and Report Generation
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message="[Research Agent]: Research phase complete, generating consolidated report..."
            ))
            
            # Generate final report
            final_report = await self._generate_final_report(research_content, query, state)
            
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
    
    async def _generate_final_report(self, research_content: str, query: str, state: AppState) -> str:
        """
        Generate a comprehensive final report from the research content.
        """
        try:
            synthesis_prompt = f"""
Based on the following research content, generate a comprehensive, well-structured research report for the query: "{query}"

RESEARCH CONTENT:
{research_content}

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
            
            state.logs.append(LogEntry(
                source="Build2_ResearchAgent",
                message="[Research Agent]: Generating final report with compound-beta-mini"
            ))
            
            # Generate final report using compound-beta-mini directly
            response = self.groq_client.chat.completions.create(
                model=self.synthesis_model,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst. Generate comprehensive, well-structured reports based on research findings."},
                    {"role": "user", "content": synthesis_prompt}
                ],
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
            
            # Save to memory service using the correct method
            await self.memory_service.store_memory(
                layer=MemoryLayer.KNOWLEDGE_SYNTHESIS,
                memory_type=MemoryType.RESEARCH_FINDING,
                content=report,
                metadata={
                    "query": query,
                    "source": "Build2_ResearchAgent",
                    "session_id": session_id,
                    "topic": safe_query,
                    "timestamp": timestamp
                },
                tags=["research", "build2", "groq_agent"]
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