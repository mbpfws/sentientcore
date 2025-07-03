from fastapi import APIRouter, HTTPException, Request, Body
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
# from graphs.intelligent_rag_graph import app as intelligent_rag_app

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

@router.post("/message")
async def process_chat_message(chat_request: ChatRequest):
    """Process a chat message and return the response message"""
    try:
        # Create initial app state
        app_state = AppState()
        
        # Add research mode prefix if specified
        message_text = chat_request.message
        if chat_request.research_mode:
            research_prefix = {
                "knowledge": "Please conduct a Knowledge Research",
                "deep": "Please conduct a Deep Research",
                "best_in_class": "Please conduct a Best-in-Class Research"
            }.get(chat_request.research_mode, "")
            
            if research_prefix:
                message_text = f"{research_prefix}: {message_text}"
        
        # Add user message to state
        app_state.messages.append(
            Message(sender="user", content=message_text, image=chat_request.image_data)
        )
        
        # Set image if provided
        if chat_request.image_data:
            app_state.image = chat_request.image_data
        
        # Process through appropriate workflow - temporarily disabled
        # Temporarily return a mock response since graph modules are missing
        mock_response = Message(
            sender="assistant",
            content=f"I received your message: '{message_text}'. The workflow system is currently being set up.",
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        )
        app_state.messages.append(mock_response)
        result = app_state
        
        # if chat_request.workflow_mode == "intelligent":
        #     result = intelligent_rag_app.invoke(app_state.model_dump())
        # elif chat_request.workflow_mode == "multi_agent":
        #     from graphs.multi_agent_rag_graph import app as multi_agent_app
        #     result = multi_agent_app.invoke(app_state.model_dump())
        # elif chat_request.workflow_mode == "legacy":
        #     from graphs.orchestration_graph import app as orchestration_app
        #     result = orchestration_app.invoke(app_state.model_dump())
        # else:
        #     raise HTTPException(status_code=400, detail=f"Invalid workflow mode: {chat_request.workflow_mode}")
        
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
                id=last_message.id or f"msg_{int(time.time() * 1000)}",
                content=last_message.content,
                sender="assistant",
                created_at=last_message.created_at or time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            )
        else:
            # Fallback response if no assistant message found
            response_message = MessageResponse(
                id=f"msg_{int(time.time() * 1000)}",
                content="I received your message but couldn't generate a response. Please try again.",
                sender="assistant",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
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
