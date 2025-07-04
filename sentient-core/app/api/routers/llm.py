from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json

# Add project root to path to allow absolute imports
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.services.llm_service import EnhancedLLMService

router = APIRouter(prefix="/llm", tags=["llm"])

# Dependency to get a single instance of the LLM service
llm_service = EnhancedLLMService()

def get_llm_service():
    return llm_service

class LLMRequest(BaseModel):
    model: str
    system_prompt: str
    user_prompt: str
    # Add any other parameters like temperature, max_tokens etc. if needed
    temperature: float = 0.7

async def stream_generator(response_stream):
    """Generator function to format SSE events."""
    try:
        async for chunk in response_stream:
            # Ensure chunk is not empty or just whitespace
            if chunk and chunk.strip():
                # Send data chunks as JSON objects for easier parsing on the frontend
                yield f"data: {json.dumps({'content': chunk})}\n\n"
    except Exception as e:
        print(f"Error during stream generation: {e}")
        # Yield a structured error message
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Signal the end of the stream to the client
        yield "data: [DONE]\n\n"


@router.post("/stream")
async def stream_llm_response(
    request: LLMRequest,
    service: EnhancedLLMService = Depends(get_llm_service)
):
    """
    Endpoint to get a streaming response from the LLM service.
    Uses Server-Sent Events (SSE) to stream data.
    """
    response_stream = service.invoke(
        system_prompt=request.system_prompt,
        user_prompt=request.user_prompt,
        model=request.model,
        stream=True,
        temperature=request.temperature
    )

    return StreamingResponse(
        stream_generator(response_stream),
        media_type="text/event-stream"
    )
