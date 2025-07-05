from fastapi import APIRouter, HTTPException, Request, Body, Form, File, UploadFile
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import sys
import time

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AppState, Message, LogEntry
from core.graphs.sentient_workflow_graph import get_sentient_workflow_app

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str
    workflow_mode: str = "intelligent"  # Default to intelligent workflow
    image_data: Optional[bytes] = None
    research_mode: Optional[str] = None  # knowledge, deep, or best_in_class

class MessageResponse(BaseModel):
    """Response model for chat messages"""
    id: str
    content: str
    sender: str
    created_at: str

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
    image_data: Optional[bytes] = None
) -> MessageResponse:
    """Internal function to process chat messages"""
    # Create initial app state
    app_state = AppState()
    
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
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    else:
        # Fallback response if no assistant message found
        response_message = MessageResponse(
            id=f"msg_{int(time.time() * 1000)}",
            content="I received your message but couldn't generate a response. Please try again.",
            sender="assistant",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    
    return response_message

@router.post("/message")
async def process_chat_message(
    message: str = Form(...),
    workflow_mode: str = Form("intelligent"),
    research_mode: Optional[str] = Form(None),
    task_id: Optional[str] = Form(None),
    image_data: Optional[UploadFile] = File(None)
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
            image_data=image_bytes
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
            image_data=chat_request.image_data
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
async def get_chat_history(workflow_mode: Optional[str] = None, task_id: Optional[str] = None):
    """Get the chat history"""
    try:
        # For now, return empty history as we don't have persistent storage
        # In a real implementation, this would fetch from a database
        chat_history = ChatHistory(
            messages=[],
            workflow_mode=workflow_mode or "intelligent",
            task_id=task_id
        )
        
        return ApiResponse(data=chat_history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")
