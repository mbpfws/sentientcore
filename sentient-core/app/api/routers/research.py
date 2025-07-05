from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
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
import markdown
from weasyprint import HTML, CSS

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.agents.research_agent import ResearchAgent, ResearchMode
from core.services.enhanced_llm_service_main import EnhancedLLMService
from core.services.memory_service import MemoryService, MemoryType
from core.models import LogEntry

router = APIRouter(prefix="/research", tags=["research"])

# Global connections manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, research_id: str):
        await websocket.accept()
        self.active_connections[research_id] = websocket
        
    def disconnect(self, research_id: str):
        if research_id in self.active_connections:
            del self.active_connections[research_id]
            
    async def send_update(self, research_id: str, data: dict):
        if research_id in self.active_connections:
            try:
                await self.active_connections[research_id].send_text(json.dumps(data))
            except:
                self.disconnect(research_id)

manager = ConnectionManager()

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

# Initialize services
llm_service = EnhancedLLMService()
memory_service = MemoryService()

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
            
            # Send real-time update
            await manager.send_update(research_id, {
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
        
        # Send WebSocket update
        await manager.send_update(research_id, {
            "type": "status_update",
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
        # Get from memory storage
        workflow_results = [
            result for result in research_storage.values()
            if result.workflow_id == workflow_id
        ]
        
        # Also get from persistent memory
        memories = await memory_service.retrieve_memories(
            memory_type=MemoryType.KNOWLEDGE_SYNTHESIS,
            limit=50
        )
        
        # Filter memories for this workflow
        persistent_results = [
            memory for memory in memories
            if memory.metadata.get("workflow_id") == workflow_id
        ]
        
        return JSONResponse({
            "success": True,
            "data": [result.dict() for result in workflow_results],
            "persistent_count": len(persistent_results),
            "message": "Research results retrieved successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get research results: {str(e)}")

@router.get("/result/{result_id}")
async def get_research_result(result_id: str) -> JSONResponse:
    """Get a specific research result"""
    try:
        if result_id not in research_storage:
            raise HTTPException(status_code=404, detail="Research result not found")
        
        result = research_storage[result_id]
        return JSONResponse({
            "success": True,
            "data": result.dict(),
            "message": "Research result retrieved successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get research result: {str(e)}")

@router.post("/export/pdf")
async def export_research_pdf(request: ExportRequest) -> StreamingResponse:
    """Export research result as PDF"""
    try:
        if request.result_id not in research_storage:
            raise HTTPException(status_code=404, detail="Research result not found")
        
        result = research_storage[request.result_id]
        
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
        if request.result_id not in research_storage:
            raise HTTPException(status_code=404, detail="Research result not found")
        
        result = research_storage[request.result_id]
        markdown_content = generate_research_markdown(result)
        
        return JSONResponse({
            "success": True,
            "markdown": markdown_content,
            "message": "Markdown exported successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export Markdown: {str(e)}")

@router.websocket("/ws/{research_id}")
async def websocket_endpoint(websocket: WebSocket, research_id: str):
    """WebSocket endpoint for real-time research updates"""
    await manager.connect(websocket, research_id)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(research_id)

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