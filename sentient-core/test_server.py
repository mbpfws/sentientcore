#!/usr/bin/env python3
"""
Minimal test server to verify Build 2 research functionality
"""

import asyncio
import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our components
from core.models import AppState, Message
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.services.enhanced_llm_service_main import EnhancedLLMService

app = FastAPI(title="Test Server for Build 2")

class TestRequest(BaseModel):
    message: str
    workflow_mode: str = "intelligent"
    research_mode: Optional[str] = None

@app.post("/test/research")
async def test_research(request: TestRequest):
    """Test endpoint for Build 2 research functionality"""
    try:
        # Create LLM service and orchestrator
        llm_service = EnhancedLLMService()
        orchestrator = UltraOrchestrator(llm_service)
        
        # Create app state
        app_state = AppState()
        
        # Add research mode prefix if specified
        message_text = request.message
        if request.research_mode:
            research_prefix = {
                "knowledge": "Please conduct a Knowledge Research",
                "deep": "Please conduct a Deep Research", 
                "best_in_class": "Please conduct a Best-in-Class Research"
            }.get(request.research_mode, "")
            
            if research_prefix:
                message_text = f"{research_prefix}: {message_text}"
        
        # Add user message
        app_state.messages.append(
            Message(sender="user", content=message_text)
        )
        
        # Process with orchestrator
        result = await orchestrator.invoke_state(app_state)
        
        # Extract response
        assistant_messages = [msg for msg in result.messages if msg.sender == "assistant"]
        if assistant_messages:
            response_content = assistant_messages[-1].content
        else:
            response_content = "No response generated"
            
        return {
            "success": True,
            "response": response_content,
            "research_mode": request.research_mode,
            "message_count": len(result.messages)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "research_mode": request.research_mode
        }

@app.get("/")
async def root():
    return {"status": "Test server running", "build": "Build 2 - Research Delegation"}

if __name__ == "__main__":
    print("Starting Build 2 test server...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")