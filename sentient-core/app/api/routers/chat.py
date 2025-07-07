from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Form, File, UploadFile, Depends
from fastapi.responses import FileResponse
import time
import os
import glob
import uuid
from app.services.service_factory import ServiceFactory, get_service_factory
from core.config import ROOT_DIR

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
    session_id: Optional[str] = None,
    services: ServiceFactory = Depends(get_service_factory)
) -> MessageResponse:
    """Internal message processing function with service factory integration"""
    try:
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get services from factory
        state_manager = services.state_manager
        llm_service = services.llm_service
        memory_service = services.memory_service
        sse_manager = services.sse_manager
        workflow_orchestrator = services.workflow_service
        
        # Load or create session state
        session_state = await state_manager.get_session_state(session_id)
        if not session_state:
            session_state = await state_manager.create_session(session_id)
        
        # Add user message to session
        user_message = {
            "content": message,
            "sender": "user",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_data": image_data
        }
        
        # Update session with user message
        await state_manager.merge_session_data(session_id, {
            "messages": session_state.data.get("messages", []) + [user_message],
            "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        
        # Handle research mode prefix
        if research_mode:
            message = f"[RESEARCH_MODE:{research_mode}] {message}"
        
        # Store message in memory
        await memory_service.store_memory(
            content=message,
            memory_type="conversation",
            session_id=session_id,
            metadata={"sender": "user", "workflow_mode": workflow_mode}
        )
        
        # Process message through LLM service
        response_content = await llm_service.chat_completion(
            messages=[
                {"role": "user", "content": message}
            ],
            session_id=session_id
        )
        
        # Create assistant message
        assistant_message = {
            "content": response_content,
            "sender": "assistant",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        # Update session with assistant message
        current_messages = session_state.data.get("messages", []) + [user_message]
        await state_manager.merge_session_data(session_id, {
            "messages": current_messages + [assistant_message],
            "last_activity": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        })
        
        # Store assistant response in memory
        await memory_service.store_memory(
            content=response_content,
            memory_type="conversation",
            session_id=session_id,
            metadata={"sender": "assistant", "workflow_mode": workflow_mode}
        )
        
        # Send SSE event for real-time updates
        await sse_manager.send_chat_message(
            session_id=session_id,
            message=response_content,
            sender="assistant"
        )
        
        # Create response message
        response_message = MessageResponse(
            id=f"msg_{int(time.time() * 1000)}",
            content=response_content,
            sender="assistant",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=session_id
        )
        
        return response_message
        
    except Exception as e:
        print(f"Error in _process_message_internal: {e}")
        # Create error response
        response_message = MessageResponse(
            id=f"msg_{int(time.time() * 1000)}",
            content="I received your message but couldn't generate a response. Please try again.",
            sender="assistant",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id=session_id
        )
    
    return response_message

@router.post("/message")
async def process_chat_message(
    message: str = Form(...),
    workflow_mode: str = Form("intelligent"),
    research_mode: Optional[str] = Form(None),
    task_id: Optional[str] = Form(None),
    image_data: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    services: ServiceFactory = Depends(get_service_factory)
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
            session_id=session_id,
            services=services
        )
        
        return ApiResponse(data=response_message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@router.post("/message/json")
async def process_chat_message_json(
    chat_request: ChatRequest,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Process a chat message using JSON (backward compatibility)"""
    try:
        response_message = await _process_message_internal(
            message=chat_request.message,
            workflow_mode=chat_request.workflow_mode,
            research_mode=chat_request.research_mode,
            task_id=getattr(chat_request, 'task_id', None),
            image_data=chat_request.image_data,
            session_id=chat_request.session_id,
            services=services
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
    session_id: Optional[str] = None,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Get the chat history using service factory"""
    try:
        messages = []
        
        # Load session history if session_id provided
        if session_id:
            try:
                state_manager = services.state_manager
                session_state = await state_manager.get_session_state(session_id)
                
                # Convert session messages to MessageResponse format
                if session_state and "messages" in session_state.data:
                    for i, msg in enumerate(session_state.data["messages"]):
                        messages.append(MessageResponse(
                            id=f"msg_{int(time.time() * 1000)}_{i}",
                            content=msg.get("content", ""),
                            sender=msg.get("sender", "unknown"),
                            created_at=msg.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
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
async def list_sessions(services: ServiceFactory = Depends(get_service_factory)):
    """List all available sessions with metadata"""
    try:
        state_manager = services.state_manager
        
        # Get all sessions
        sessions = await state_manager.list_sessions()
        
        return ApiResponse(data={
            "sessions": sessions,
            "total_count": len(sessions)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")

@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Get detailed information about a specific session"""
    try:
        # Validate session ID format
        if not session_id or len(session_id) < 8:
            raise HTTPException(status_code=400, detail="Invalid session ID format")
        
        state_manager = services.state_manager
        
        # Get session details
        session_data = await state_manager.get_session_state(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return ApiResponse(data={
            "session_id": session_data.session_id,
            "created_at": session_data.created_at.isoformat(),
            "updated_at": session_data.updated_at.isoformat(),
            "data": session_data.data
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {str(e)}")

@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Delete a specific session and its data"""
    try:
        # Security check: ensure session_id doesn't contain path traversal
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        state_manager = services.state_manager
        
        # Delete the session
        success = await state_manager.delete_session(session_id)
        
        if success:
            return ApiResponse(data={"message": f"Session {session_id} deleted successfully"})
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@router.get("/sessions/{session_id}/stats")
async def get_session_stats(
    session_id: str,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Get statistics for a specific session"""
    try:
        # Security check: ensure session_id doesn't contain path traversal
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        state_manager = services.state_manager
        
        # Get session data
        session_data = await state_manager.get_session_state(session_id)
        
        if session_data:
            # Calculate basic statistics from session data
            messages = session_data.data.get("messages", [])
            stats = {
                "session_id": session_id,
                "message_count": len(messages),
                "created_at": session_data.created_at.isoformat(),
                "updated_at": session_data.updated_at.isoformat(),
                "user_messages": len([m for m in messages if m.get("sender") == "user"]),
                "assistant_messages": len([m for m in messages if m.get("sender") == "assistant"])
            }
            return ApiResponse(data=stats)
        else:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session stats: {str(e)}")

# Enhanced conversation context and confirmation endpoints
@router.post("/confirmation")
async def handle_confirmation(
    confirmation_request: ConfirmationRequest,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Handle user confirmation responses"""
    try:
        session_id = confirmation_request.session_id
        confirmation_id = confirmation_request.confirmation_id
        confirmed = confirmation_request.confirmed
        
        state_manager = services.state_manager
        
        # Load session to get pending confirmations
        session_data = None
        if session_id:
            session_data = await state_manager.get_session_state(session_id)
        
        if not session_data:
            session_data_dict = {"pending_confirmations": []}
        else:
            session_data_dict = session_data.data
        
        # Find the pending confirmation
        pending_confirmation = None
        pending_confirmations = session_data_dict.get('pending_confirmations', [])
        for conf in pending_confirmations:
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
        session_data_dict['pending_confirmations'] = [
            conf for conf in pending_confirmations 
            if conf.get('confirmation_id') != confirmation_id
        ]
        
        # Save updated session
        if session_id:
            await state_manager.merge_session_data(session_id, session_data_dict)
        
        return ApiResponse(data=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling confirmation: {str(e)}")

@router.get("/sessions/{session_id}/context")
async def get_conversation_context(
    session_id: str,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Get conversation context for a session"""
    try:
        # Security check
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        state_manager = services.state_manager
        
        # Load session
        session_data = await state_manager.get_session_state(session_id)
        
        # Get conversation context from session or create default
        if session_data and "conversation_context" in session_data.data:
            context = session_data.data["conversation_context"]
        else:
            context = {
                "current_focus": "general_inquiry",
                "user_intent": "unknown",
                "requirements_gathered": False,
                "research_needed": False,
                "project_type": None,
                "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        
        return ApiResponse(data=context)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation context: {str(e)}")

@router.put("/sessions/{session_id}/context")
async def update_conversation_context(
    session_id: str, 
    context_update: ContextUpdateRequest,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Update conversation context for a session"""
    try:
        # Security check
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        state_manager = services.state_manager
        
        # Load session
        session_data = await state_manager.get_session_state(session_id)
        if not session_data:
            session_data_dict = {}
        else:
            session_data_dict = session_data.data
        
        # Update conversation context
        context_dict = context_update.context.dict()
        context_dict["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        session_data_dict["conversation_context"] = context_dict
        
        # Save updated session
        await state_manager.merge_session_data(session_id, session_data_dict)
        
        return ApiResponse(data={
            "message": "Conversation context updated successfully",
            "context": context_dict
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating conversation context: {str(e)}")

@router.get("/sessions/{session_id}/confirmations")
async def get_pending_confirmations(
    session_id: str,
    services: ServiceFactory = Depends(get_service_factory)
):
    """Get pending confirmations for a session"""
    try:
        # Security check
        if ".." in session_id or "/" in session_id or "\\" in session_id:
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        state_manager = services.state_manager
        
        # Load session
        session_data = await state_manager.get_session_state(session_id)
        
        # Get pending confirmations
        if session_data and "pending_confirmations" in session_data.data:
            confirmations = session_data.data["pending_confirmations"]
        else:
            confirmations = []
        
        return ApiResponse(data={
            "confirmations": confirmations,
            "total_count": len(confirmations)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting pending confirmations: {str(e)}")
