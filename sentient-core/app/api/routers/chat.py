from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import FileResponse
import time
import os
import glob
import uuid
from core.models import AppState, Message
from core.services.session_persistence_service import get_session_persistence_service
from core.config import ROOT_DIR
from core.graphs.sentient_workflow_graph import get_sentient_workflow_app

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
    session_id: Optional[str] = None
    message_type: Optional[str] = "text"  # text, confirmation, artifact
    metadata: Optional[Dict[str, Any]] = None

class ConversationContext(BaseModel):
    """Model for conversation context tracking"""
    current_focus: str = "general_inquiry"  # general_inquiry, requirements_gathering, research, planning, development
    user_intent: str = "unknown"  # development, research, planning, general_inquiry
    requirements_gathered: bool = False
    research_needed: bool = False
    project_type: Optional[str] = None
    last_updated: str = time.strftime("%Y-%m-%dT%H:%M:%SZ")

class PendingConfirmation(BaseModel):
    """Model for pending user confirmations"""
    confirmation_id: str
    message: str
    action_type: str  # start_research, transition_planning, create_workflow
    action_data: Dict[str, Any]
    created_at: str = time.strftime("%Y-%m-%dT%H:%M:%SZ")

class ConfirmationRequest(BaseModel):
    """Request model for confirmation responses"""
    confirmation_id: str
    confirmed: bool
    session_id: Optional[str] = None

class ContextUpdateRequest(BaseModel):
    """Request model for updating conversation context"""
    session_id: str
    context: ConversationContext  # Build 2: Include session ID in response

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
    session_service = get_session_persistence_service()
    app_state = await session_service.load_session(session_id)
    if app_state is None:
        app_state = AppState(session_id=session_id)
    
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
            content=f"I encountered an issue processing your request: {str(workflow_error)}. Please try again or contact support if the issue persists."
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
        assistant_content = last_message.content
        message_type = "text"
        metadata = {}
        
        # Check for special message types in content
        if "[CONFIRMATION_REQUIRED]" in assistant_content:
            message_type = "confirmation"
            # Generate confirmation ID
            confirmation_id = str(uuid.uuid4())
            metadata = {
                "requires_confirmation": True,
                "confirmation_id": confirmation_id
            }
            # Store pending confirmation in session
            if not hasattr(result_state, 'pending_confirmations'):
                result_state.pending_confirmations = []
            
            # Extract action details from content (simplified)
            action_type = "general"
            if "research" in assistant_content.lower():
                action_type = "start_research"
            elif "planning" in assistant_content.lower():
                action_type = "transition_planning"
            elif "development" in assistant_content.lower():
                action_type = "create_workflow"
            
            result_state.pending_confirmations.append({
                "confirmation_id": confirmation_id,
                "message": assistant_content.replace("[CONFIRMATION_REQUIRED]", "").strip(),
                "action_type": action_type,
                "action_data": {},
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
            
            # Clean the content for display
            assistant_content = assistant_content.replace("[CONFIRMATION_REQUIRED]", "").strip()
        
        elif "[ARTIFACT]" in assistant_content:
            message_type = "artifact"
            # Extract artifact information
            if "research_report" in assistant_content.lower():
                metadata = {
                    "artifact_type": "research_report",
                    "download_url": "/chat/download/research/latest_research_report.md"
                }
            elif "plan" in assistant_content.lower():
                metadata = {
                    "artifact_type": "plan",
                    "download_url": "/chat/download/research/latest_plan.md"
                }
            
            # Clean the content for display
            assistant_content = assistant_content.replace("[ARTIFACT]", "").strip()
        
        response_message = MessageResponse(
            id=f"msg_{int(time.time() * 1000)}",
            content=assistant_content,
            sender="assistant",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=session_id,
            message_type=message_type,
            metadata=metadata if metadata else None
        )
        
        # Save updated session with pending confirmations
        if session_id:
            await session_service.save_session(session_id, result_state)
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
                session_service = get_session_persistence_service()
                app_state = await session_service.load_session(session_id)
                # Convert AppState messages to MessageResponse format
                if app_state:
                    for msg in app_state.messages:
                        messages.append(MessageResponse(
                            id=f"msg_{int(time.time() * 1000)}_{len(messages)}",
                            content=msg.content,
                            sender=msg.sender,
                            created_at=getattr(msg, 'created_at', None).strftime("%Y-%m-%dT%H:%M:%SZ") if getattr(msg, 'created_at', None) else time.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        research_dir = os.path.join(ROOT_DIR, "sentient-core", "memory", "layer1_research_docs")
        
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
        research_dir = os.path.join(ROOT_DIR, "sentient-core", "memory", "layer1_research_docs")
        
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

# Enhanced conversation context and confirmation endpoints
@router.post("/confirmation")
async def handle_confirmation(confirmation_request: ConfirmationRequest):
    """Handle user confirmation responses"""
    try:
        session_id = confirmation_request.session_id
        confirmation_id = confirmation_request.confirmation_id
        confirmed = confirmation_request.confirmed
        
        # Load session to get pending confirmations
        if session_id:
            session_service = get_session_persistence_service()
            app_state = await session_service.load_session(session_id)
            if app_state is None:
                app_state = AppState(session_id=session_id)
        else:
            app_state = AppState()
        
        # Find the pending confirmation
        pending_confirmation = None
        for conf in getattr(app_state, 'pending_confirmations', []):
            if conf.get('confirmation_id') == confirmation_id:
                pending_confirmation = conf
                break
        
        if not pending_confirmation:
            raise HTTPException(status_code=404, detail="Confirmation not found")
        
        response_data = {
            "confirmation_id": confirmation_id,
            "confirmed": confirmed,
            "action_executed": False,
            "message": ""
        }
        
        if confirmed:
            # Execute the confirmed action
            action_type = pending_confirmation.get('action_type')
            action_data = pending_confirmation.get('action_data', {})
            
            if action_type == "start_research":
                # Trigger research workflow
                response_data["action_executed"] = True
                response_data["message"] = "Research workflow initiated successfully."
                
            elif action_type == "transition_planning":
                # Transition to planning phase
                response_data["action_executed"] = True
                response_data["message"] = "Transitioning to planning phase."
                
            elif action_type == "create_workflow":
                # Create development workflow
                response_data["action_executed"] = True
                response_data["message"] = "Development workflow created successfully."
        else:
            response_data["message"] = "Action cancelled by user."
        
        # Remove the confirmation from pending list
        if hasattr(app_state, 'pending_confirmations'):
            app_state.pending_confirmations = [
                conf for conf in app_state.pending_confirmations 
                if conf.get('confirmation_id') != confirmation_id
            ]
        
        # Save updated session
        if session_id:
            session_service = get_session_persistence_service()
            await session_service.save_session(session_id, app_state)
        
        return ApiResponse(data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling confirmation: {str(e)}")

@router.get("/sessions/{session_id}/context")
async def get_conversation_context(session_id: str):
    """Get conversation context for a session"""
    try:
        # Security check
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Load session
        session_service = get_session_persistence_service()
        app_state = await session_service.load_session(session_id)
        if app_state is None:
            app_state = AppState(session_id=session_id)
        
        # Get conversation context from session or create default
        context = getattr(app_state, 'conversation_context', {
            "current_focus": "general_inquiry",
            "user_intent": "unknown",
            "requirements_gathered": False,
            "research_needed": False,
            "project_type": None,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        
        return ApiResponse(data=context)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation context: {str(e)}")

@router.put("/sessions/{session_id}/context")
async def update_conversation_context(session_id: str, context_update: ContextUpdateRequest):
    """Update conversation context for a session"""
    try:
        # Security check
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Load session
        session_service = get_session_persistence_service()
        app_state = await session_service.load_session(session_id)
        if app_state is None:
            app_state = AppState(session_id=session_id)
        
        # Update conversation context
        context_dict = context_update.context.dict()
        context_dict["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        app_state.conversation_context = context_dict
        
        # Save updated session
        await session_service.save_session(session_id, app_state)
        
        return ApiResponse(data={
            "message": "Conversation context updated successfully",
            "context": context_dict
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating conversation context: {str(e)}")

@router.get("/sessions/{session_id}/confirmations")
async def get_pending_confirmations(session_id: str):
    """Get pending confirmations for a session"""
    try:
        # Security check
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Load session
        session_service = get_session_persistence_service()
        app_state = await session_service.load_session(session_id)
        if app_state is None:
            app_state = AppState(session_id=session_id)
        
        # Get pending confirmations
        confirmations = getattr(app_state, 'pending_confirmations', [])
        
        return ApiResponse(data={
            "confirmations": confirmations,
            "total_count": len(confirmations)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting pending confirmations: {str(e)}")
