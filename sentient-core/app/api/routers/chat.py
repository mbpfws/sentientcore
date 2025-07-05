from fastapi import APIRouter, HTTPException, Request, Body, Form, File, UploadFile
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import sys
import time
import glob

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AppState, Message, LogEntry
from core.graphs.sentient_workflow_graph import get_sentient_workflow_app, load_session_if_exists
from core.services.session_persistence_service import SessionPersistenceService
import uuid

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str
    workflow_mode: str = "intelligent"  # Default to intelligent workflow
    image_data: Optional[bytes] = None
    research_mode: Optional[str] = None  # knowledge, deep, or best_in_class
    session_id: Optional[str] = None  # Build 2: Session persistence support

class MessageResponse(BaseModel):
    """Response model for chat messages"""
    id: str
    content: str
    sender: str
    created_at: str
    session_id: Optional[str] = None  # Build 2: Include session ID in response

class ApiResponse(BaseModel):
    """Generic API response wrapper"""
    data: Any
    success: bool = True
    message: str = "Success"

async def _process_message_internal(
    message: str,
    workflow_mode: str = "intelligent",
    research_mode: Optional[str] = None,
    task_id: Optional[str] = None,
    image_data: Optional[bytes] = None,
    session_id: Optional[str] = None
) -> MessageResponse:
    """Internal function to process chat messages with Build 2 session persistence"""
    # Build 2: Handle session persistence
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Load existing session or create new app state
    app_state = await load_session_if_exists(session_id)
    
    # Add research mode prefix if specified
    message_text = message
    if research_mode:
        research_prefix = {
            "knowledge": "Please conduct a Knowledge Research",
            "deep": "Please conduct a Deep Research",
            "best_in_class": "Please conduct a Best-in-Class Research"
        }.get(research_mode, "")
        
        if research_prefix:
            message_text = f"{research_prefix}: {message_text}"
    
    # Add user message to state
    app_state.messages.append(
        Message(sender="user", content=message_text, image=image_data)
    )
    
    # Set image if provided
    if image_data:
        app_state.image = image_data
    
    # Process through the sentient workflow graph
    try:
        workflow_app = get_sentient_workflow_app()
        result = await workflow_app.ainvoke(app_state)
    except Exception as workflow_error:
        # Fallback to a helpful error message if workflow fails
        error_response = Message(
            sender="assistant",
            content=f"I encountered an issue processing your request: {str(workflow_error)}. Please try again or contact support if the issue persists.",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        app_state.messages.append(error_response)
        result = app_state
    
    # Convert result to AppState model and extract assistant response
    if isinstance(result, dict):
        result_state = AppState(**result)
    else:
        result_state = result
    
    # Get the last assistant message from the result
    assistant_messages = [msg for msg in result_state.messages if msg.sender == "assistant"]
    if assistant_messages:
        last_message = assistant_messages[-1]
        response_message = MessageResponse(
            id=f"msg_{int(time.time() * 1000)}",
            content=last_message.content,
            sender="assistant",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=session_id  # Build 2: Include session ID
        )
    else:
        # Fallback response if no assistant message found
        response_message = MessageResponse(
            id=f"msg_{int(time.time() * 1000)}",
            content="I received your message but couldn't generate a response. Please try again.",
            sender="assistant",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=session_id  # Build 2: Include session ID
        )
    
    return response_message

@router.post("/message")
async def process_chat_message(
    message: str = Form(...),
    workflow_mode: str = Form("intelligent"),
    research_mode: Optional[str] = Form(None),
    task_id: Optional[str] = Form(None),
    image_data: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)  # Build 2: Session persistence
):
    """Process a chat message with optional image attachment"""
    try:
        # Handle image data if provided
        image_bytes = None
        if image_data:
            image_bytes = await image_data.read()
        
        response_message = await _process_message_internal(
            message=message,
            workflow_mode=workflow_mode,
            research_mode=research_mode,
            task_id=task_id,
            image_data=image_bytes,
            session_id=session_id  # Build 2: Pass session ID
        )
        
        return ApiResponse(data=response_message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@router.post("/message/json")
async def process_chat_message_json(chat_request: ChatRequest):
    """Process a chat message using JSON (backward compatibility)"""
    try:
        response_message = await _process_message_internal(
            message=chat_request.message,
            workflow_mode=chat_request.workflow_mode,
            research_mode=chat_request.research_mode,
            task_id=getattr(chat_request, 'task_id', None),
            image_data=chat_request.image_data,
            session_id=chat_request.session_id  # Build 2: Pass session ID
        )
        
        return ApiResponse(data=response_message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

class ChatHistory(BaseModel):
    """Chat history response model"""
    messages: List[MessageResponse]
    workflow_mode: str
    task_id: Optional[str] = None

@router.get("/history")
async def get_chat_history(
    workflow_mode: Optional[str] = None, 
    task_id: Optional[str] = None,
    session_id: Optional[str] = None  # Build 2: Session-based history
):
    """Get the chat history for Build 2 with session persistence"""
    try:
        messages = []
        
        # Build 2: Load session history if session_id provided
        if session_id:
            try:
                app_state = await load_session_if_exists(session_id)
                # Convert AppState messages to MessageResponse format
                for msg in app_state.messages:
                    messages.append(MessageResponse(
                        id=f"msg_{int(time.time() * 1000)}_{len(messages)}",
                        content=msg.content,
                        sender=msg.sender,
                        created_at=msg.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        session_id=session_id
                    ))
            except Exception as e:
                print(f"Warning: Failed to load session history for {session_id}: {e}")
        
        chat_history = ChatHistory(
            messages=messages,
            workflow_mode=workflow_mode or "intelligent",
            task_id=task_id
        )
        
        return ApiResponse(data=chat_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

# Build 2: Research artifact download endpoints
@router.get("/download/research/{filename}")
async def download_research_artifact(filename: str):
    """Download research artifacts (markdown or PDF files) from long-term memory"""
    try:
        # Define the research documents directory
        research_dir = os.path.join(project_root, "memory", "layer1_research_docs")
        
        # Security check: ensure filename doesn't contain path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        # Check for both .md and .pdf extensions
        if not (filename.endswith('.md') or filename.endswith('.pdf')):
            raise HTTPException(status_code=400, detail="Only .md and .pdf files are supported")
        
        file_path = os.path.join(research_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Research artifact '{filename}' not found")
        
        # Determine media type
        media_type = "application/pdf" if filename.endswith('.pdf') else "text/markdown"
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading research artifact: {str(e)}")

@router.get("/research/artifacts")
async def list_research_artifacts():
    """List available research artifacts in long-term memory"""
    try:
        research_dir = os.path.join(project_root, "memory", "layer1_research_docs")
        
        # Create directory if it doesn't exist
        os.makedirs(research_dir, exist_ok=True)
        
        # Find all .md and .pdf files
        md_files = glob.glob(os.path.join(research_dir, "*.md"))
        pdf_files = glob.glob(os.path.join(research_dir, "*.pdf"))
        
        artifacts = []
        for file_path in md_files + pdf_files:
            filename = os.path.basename(file_path)
            file_stats = os.stat(file_path)
            artifacts.append({
                "filename": filename,
                "type": "markdown" if filename.endswith('.md') else "pdf",
                "size": file_stats.st_size,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(file_stats.st_ctime)),
                "download_url": f"/chat/download/research/{filename}"
            })
        
        return ApiResponse(data={
            "artifacts": artifacts,
            "total_count": len(artifacts)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing research artifacts: {str(e)}")

# Build 2: Session management endpoints
@router.get("/sessions")
async def list_sessions():
    """List all available sessions with metadata"""
    try:
        from core.services.session_persistence_service import SessionPersistenceService
        
        # Initialize session persistence service
        session_service = SessionPersistenceService()
        
        # Get all sessions
        sessions = await session_service.list_sessions()
        
        return ApiResponse(data={
            "sessions": sessions,
            "total_count": len(sessions)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session and its data"""
    try:
        from core.services.session_persistence_service import SessionPersistenceService
        
        # Security check: ensure session_id doesn't contain path traversal
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Initialize session persistence service
        session_service = SessionPersistenceService()
        
        # Delete the session
        success = await session_service.delete_session(session_id)
        
        if success:
            return ApiResponse(data={"message": f"Session {session_id} deleted successfully"})
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@router.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a specific session"""
    try:
        from core.services.session_persistence_service import SessionPersistenceService
        
        # Security check: ensure session_id doesn't contain path traversal
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Initialize session persistence service
        session_service = SessionPersistenceService()
        
        # Get session statistics
        stats = await session_service.get_session_stats(session_id)
        
        if stats:
            return ApiResponse(data=stats)
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session stats: {str(e)}")
