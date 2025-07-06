from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel
import asyncio
import json
import uuid
import time
from datetime import datetime
import os
import sys
from io import BytesIO
from pathlib import Path
import markdown

# Import SSE infrastructure
from .sse_events import sse_manager, sse_event_handler

# Optional WeasyPrint import for PDF generation - temporarily disabled
# try:
#     from weasyprint import HTML, CSS
#     WEASYPRINT_AVAILABLE = True
# except ImportError:
WEASYPRINT_AVAILABLE = False
HTML = None
CSS = None

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.agents.research_agent import ResearchAgent, ResearchMode
from core.services.enhanced_llm_service import EnhancedLLMService
from core.services.memory_service import MemoryService, MemoryType
from core.models import LogEntry

router = APIRouter(prefix="/research", tags=["research"])

# SSE Integration Functions
def get_sse_manager():
    """Get the global SSE connection manager"""
    return sse_manager

def get_sse_event_handler():
    """Get the global SSE event handler"""
    return sse_event_handler

async def send_research_update(research_id: str, data: dict):
    """Send research update via SSE"""
    await sse_manager.broadcast_to_research(data, research_id)

# Request/Response Models
class ResearchRequest(BaseModel):
    query: str
    mode: str = "knowledge"  # knowledge, deep, best_in_class
    workflow_id: str
    context: Optional[str] = None
    max_searches: Optional[int] = 5
    enable_verbose: bool = True

class ResearchProgress(BaseModel):
    research_id: str
    status: str
    progress: float
    current_step: str
    verbose_log: List[str]
    search_results: List[Dict[str, Any]]
    timestamp: str

class ResearchResult(BaseModel):
    id: str
    query: str
    mode: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    created_at: str
    completed_at: Optional[str] = None
    verbose_log: List[str]
    workflow_id: str
    search_count: int = 0

class ExportRequest(BaseModel):
    result_id: str
    format: str = "pdf"  # pdf or markdown

# In-memory storage for research results (will be replaced with persistent storage)
research_storage: Dict[str, ResearchResult] = {}
active_research: Dict[str, Dict[str, Any]] = {}

# Research results directory
RESEARCH_RESULTS_DIR = Path("memory/research_results")
RESEARCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_research_result_to_file(result: ResearchResult) -> None:
    """Save research result to file system for persistence"""
    try:
        file_path = RESEARCH_RESULTS_DIR / f"{result.id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(result.dict(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving research result to file: {e}")

def load_research_result_from_file(result_id: str) -> Optional[ResearchResult]:
    """Load research result from file system"""
    try:
        file_path = RESEARCH_RESULTS_DIR / f"{result_id}.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return ResearchResult(**data)
    except Exception as e:
        print(f"Error loading research result from file: {e}")
    return None

def load_all_research_results() -> Dict[str, ResearchResult]:
    """Load all research results from file system"""
    results = {}
    try:
        for file_path in RESEARCH_RESULTS_DIR.glob("*.json"):
            result_id = file_path.stem
            result = load_research_result_from_file(result_id)
            if result:
                results[result_id] = result
    except Exception as e:
        print(f"Error loading research results: {e}")
    return results

def get_research_results_by_workflow(workflow_id: str) -> List[ResearchResult]:
    """Get all research results for a specific workflow"""
    all_results = load_all_research_results()
    return [result for result in all_results.values() if result.workflow_id == workflow_id]

# Initialize services
llm_service = EnhancedLLMService()
memory_service = MemoryService()

# Load existing research results from file system on startup
def initialize_research_storage():
    """Load existing research results from file system into memory"""
    global research_storage
    try:
        file_results = load_all_research_results()
        research_storage.update(file_results)
        print(f"Loaded {len(file_results)} research results from file system")
    except Exception as e:
        print(f"Error loading research results on startup: {e}")

# Initialize on module load
initialize_research_storage()

@router.post("/start")
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """Start a new research task with Groq's agentic tooling"""
    try:
        # Generate unique research ID
        research_id = str(uuid.uuid4())
        
        # Create research result entry
        research_result = ResearchResult(
            id=research_id,
            query=request.query,
            mode=request.mode,
            status="pending",
            progress=0.0,
            created_at=datetime.now().isoformat(),
            verbose_log=[f"Research initiated: {request.query}"],
            workflow_id=request.workflow_id,
            search_count=0
        )
        
        # Store in memory and persistent storage
        research_storage[research_id] = research_result
        
        # Store in memory service for persistence
        await memory_service.store_memory(
            memory_type=MemoryType.CONVERSATION_HISTORY,
            content={
                "type": "research_started",
                "research_id": research_id,
                "query": request.query,
                "mode": request.mode,
                "workflow_id": request.workflow_id
            },
            metadata={
                "research_id": research_id,
                "workflow_id": request.workflow_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Start research in background
        background_tasks.add_task(
            execute_research_task,
            research_id,
            request
        )
        
        return JSONResponse({
            "success": True,
            "data": research_result.dict(),
            "message": "Research started successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start research: {str(e)}")

async def execute_research_task(research_id: str, request: ResearchRequest):
    """Execute the research task with verbose logging and progress updates"""
    try:
        # Initialize research agent with Groq agentic tooling
        research_agent = ResearchAgent(llm_service)
        
        # Update status to searching
        await update_research_status(
            research_id,
            "searching",
            10.0,
            "Initializing research with Groq agentic tooling",
            ["Starting research agent with compound-beta model"]
        )
        
        # Determine research mode
        mode_map = {
            "knowledge": ResearchMode.KNOWLEDGE,
            "deep": ResearchMode.DEEP,
            "best_in_class": ResearchMode.BEST_IN_CLASS
        }
        research_mode = mode_map.get(request.mode, ResearchMode.KNOWLEDGE)
        
        # Execute research with verbose logging
        verbose_logs = []
        search_results = []
        
        # Create research state
        from core.agents.research_agent import ResearchState
        state = ResearchState(
            original_query=request.query,
            research_mode=research_mode,
            steps=[],
            final_report="",
            continual_search_suggestions=[],
            logs=[]
        )
        
        # Plan research steps
        await update_research_status(
            research_id,
            "searching",
            20.0,
            "Planning research steps",
            ["Using Groq compound-beta for intelligent research planning"]
        )
        
        state = await research_agent.plan_research_steps(state)
        verbose_logs.extend([log.message for log in state.logs])
        
        # Execute search steps with progress updates
        total_steps = len(state.steps)
        for i, step in enumerate(state.steps):
            progress = 20.0 + (i / total_steps) * 60.0  # 20% to 80%
            
            await update_research_status(
                research_id,
                "searching",
                progress,
                f"Executing search: {step.query}",
                [f"Search {i+1}/{total_steps}: {step.query}"]
            )
            
            # Execute individual search
            state = await research_agent.execute_agentic_search(state)
            
            # Update verbose logs
            new_logs = [log.message for log in state.logs[len(verbose_logs):]]
            verbose_logs.extend(new_logs)
            
            # Store search result
            if state.steps and len(state.steps) > i:
                search_results.append({
                    "query": step.query,
                    "result": state.steps[i].result,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Send real-time update via SSE
            await send_research_update(research_id, {
                "type": "search_progress",
                "progress": progress,
                "current_step": f"Search {i+1}/{total_steps}",
                "search_result": search_results[-1] if search_results else None
            })
        
        # Synthesize final report
        await update_research_status(
            research_id,
            "synthesizing",
            85.0,
            "Synthesizing comprehensive report",
            ["Generating final report with Groq compound-beta"]
        )
        
        state = await research_agent.synthesize_report(state)
        verbose_logs.extend([log.message for log in state.logs[len(verbose_logs):]])
        
        # Prepare final results
        final_results = {
            "summary": state.final_report,
            "sources": [],
            "insights": [],
            "recommendations": state.continual_search_suggestions,
            "citations": [],
            "search_results": search_results
        }
        
        # Extract sources from search results
        for search_result in search_results:
            if "sources" in search_result.get("result", ""):
                # Parse sources from result text (simplified)
                final_results["sources"].append({
                    "title": search_result["query"],
                    "url": "#",
                    "snippet": search_result["result"][:200] + "...",
                    "relevance_score": 0.8
                })
        
        # Complete research
        await update_research_status(
            research_id,
            "completed",
            100.0,
            "Research completed successfully",
            ["Research synthesis completed"],
            final_results
        )
        
        # Store final result in memory service
        await memory_service.store_memory(
            memory_type=MemoryType.KNOWLEDGE_SYNTHESIS,
            content={
                "type": "research_completed",
                "research_id": research_id,
                "query": request.query,
                "mode": request.mode,
                "results": final_results,
                "workflow_id": request.workflow_id
            },
            metadata={
                "research_id": research_id,
                "workflow_id": request.workflow_id,
                "completed_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        await update_research_status(
            research_id,
            "error",
            0.0,
            f"Research failed: {str(e)}",
            [f"Error: {str(e)}"]
        )

async def update_research_status(
    research_id: str,
    status: str,
    progress: float,
    current_step: str,
    new_logs: List[str],
    results: Optional[Dict[str, Any]] = None
):
    """Update research status and send real-time updates"""
    if research_id in research_storage:
        research_result = research_storage[research_id]
        research_result.status = status
        research_result.progress = progress
        research_result.verbose_log.extend(new_logs)
        
        if results:
            research_result.results = results
            research_result.completed_at = datetime.now().isoformat()
        
        # Save to file system for persistence
        save_research_result_to_file(research_result)
        
        # Send SSE update
        await send_research_update(research_id, {
            "type": "research_update",
            "result": research_result.dict(),
            "research_id": research_id,
            "status": status,
            "progress": progress,
            "current_step": current_step,
            "verbose_log": new_logs,
            "timestamp": datetime.now().isoformat()
        })

@router.get("/results")
async def get_research_results(workflow_id: str) -> JSONResponse:
    """Get all research results for a workflow"""
    try:
        # Get from file system (persistent storage)
        file_results = get_research_results_by_workflow(workflow_id)
        
        # Get from memory storage (current session)
        memory_results = [
            result for result in research_storage.values()
            if result.workflow_id == workflow_id
        ]
        
        # Combine results, prioritizing memory (more recent) over file
        combined_results = {}
        
        # Add file results first
        for result in file_results:
            combined_results[result.id] = result
        
        # Override with memory results (more recent)
        for result in memory_results:
            combined_results[result.id] = result
        
        # Sort by created_at (newest first)
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        
        return JSONResponse({
            "success": True,
            "data": [result.dict() for result in sorted_results],
            "total_count": len(sorted_results),
            "message": "Research results retrieved successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get research results: {str(e)}")

@router.get("/result/{result_id}")
async def get_research_result(result_id: str) -> JSONResponse:
    """Get a specific research result"""
    try:
        # First check memory storage
        result = research_storage.get(result_id)
        
        # If not in memory, check file system
        if not result:
            result = load_research_result_from_file(result_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Research result not found")
        
        return JSONResponse({
            "success": True,
            "data": result.dict(),
            "message": "Research result retrieved successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get research result: {str(e)}")

@router.post("/export/pdf")
async def export_research_pdf(request: ExportRequest) -> StreamingResponse:
    """Export research result as PDF"""
    try:
        if not WEASYPRINT_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="PDF export is not available. WeasyPrint dependencies are not installed."
            )
        
        # First check memory storage
        result = research_storage.get(request.result_id)
        
        # If not in memory, check file system
        if not result:
            result = load_research_result_from_file(request.result_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Research result not found")
        
        # Generate HTML content
        html_content = generate_research_html(result)
        
        # Convert to PDF
        pdf_buffer = BytesIO()
        HTML(string=html_content).write_pdf(pdf_buffer)
        pdf_buffer.seek(0)
        
        # Return as streaming response
        return StreamingResponse(
            BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=research_{result.id}.pdf"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export PDF: {str(e)}")

@router.post("/export/markdown")
async def export_research_markdown(request: ExportRequest) -> JSONResponse:
    """Export research result as Markdown"""
    try:
        # First check memory storage
        result = research_storage.get(request.result_id)
        
        # If not in memory, check file system
        if not result:
            result = load_research_result_from_file(request.result_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Research result not found")
        markdown_content = generate_research_markdown(result)
        
        return JSONResponse({
            "success": True,
            "markdown": markdown_content,
            "message": "Markdown exported successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export Markdown: {str(e)}")

# WebSocket endpoint removed - research updates now use SSE
# Clients should connect to /api/sse/research/{research_id} for real-time updates

def generate_research_html(result: ResearchResult) -> str:
    """Generate HTML content for PDF export"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Research Report: {result.query}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; }}
            h2 {{ color: #666; }}
            .meta {{ background: #f5f5f5; padding: 15px; margin: 20px 0; }}
            .source {{ margin: 10px 0; padding: 10px; border-left: 3px solid #007acc; }}
        </style>
    </head>
    <body>
        <h1>Research Report: {result.query}</h1>
        <div class="meta">
            <p><strong>Mode:</strong> {result.mode.replace('_', ' ').title()}</p>
            <p><strong>Created:</strong> {result.created_at}</p>
            <p><strong>Status:</strong> {result.status}</p>
        </div>
    """
    
    if result.results:
        html += f"<h2>Summary</h2><p>{result.results.get('summary', 'No summary available')}</p>"
        
        if result.results.get('sources'):
            html += "<h2>Sources</h2>"
            for i, source in enumerate(result.results['sources']):
                html += f'<div class="source"><strong>{i+1}. {source["title"]}</strong><br>{source["snippet"]}</div>'
    
    html += "</body></html>"
    return html

def generate_research_markdown(result: ResearchResult) -> str:
    """Generate Markdown content for export"""
    markdown_content = f"""# Research Report: {result.query}

**Mode:** {result.mode.replace('_', ' ').title()}
**Created:** {result.created_at}
**Status:** {result.status}

"""
    
    if result.results:
        markdown_content += f"## Summary\n\n{result.results.get('summary', 'No summary available')}\n\n"
        
        if result.results.get('sources'):
            markdown_content += "## Sources\n\n"
            for i, source in enumerate(result.results['sources']):
                markdown_content += f"{i+1}. **{source['title']}**\n   {source['snippet']}\n\n"
    
    return markdown_content