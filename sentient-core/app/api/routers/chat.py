from fastapi import APIRouter, HTTPException, Request, Body
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AppState, Message, LogEntry
from graphs.intelligent_rag_graph import app as intelligent_rag_app

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str
    workflow_mode: str = "intelligent"  # Default to intelligent workflow
    image_data: Optional[bytes] = None
    research_mode: Optional[str] = None  # knowledge, deep, or best_in_class

@router.post("/", response_model=AppState)
async def process_chat_message(chat_request: ChatRequest):
    """Process a chat message and return the updated application state"""
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
        
        # Process through appropriate workflow
        if chat_request.workflow_mode == "intelligent":
            result = intelligent_rag_app.invoke(app_state.model_dump())
        elif chat_request.workflow_mode == "multi_agent":
            from graphs.multi_agent_rag_graph import app as multi_agent_app
            result = multi_agent_app.invoke(app_state.model_dump())
        elif chat_request.workflow_mode == "legacy":
            from graphs.orchestration_graph import app as orchestration_app
            result = orchestration_app.invoke(app_state.model_dump())
        else:
            raise HTTPException(status_code=400, detail=f"Invalid workflow mode: {chat_request.workflow_mode}")
        
        # Convert result to AppState model
        if isinstance(result, dict):
            # Create a new AppState from the result
            return AppState(**result)
        
        return app_state
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat message: {str(e)}")

@router.get("/history")
async def get_chat_history(app_state: AppState = Body(...)):
    """Get the chat history from the application state"""
    try:
        # Prepare chat history with proper message formatting
        history = [{
            "sender": msg.sender,
            "content": msg.content,
            "has_image": msg.image is not None
        } for msg in app_state.messages]
        
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")
