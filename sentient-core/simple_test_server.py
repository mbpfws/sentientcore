#!/usr/bin/env python3
"""
Simple Test Server for AI Orchestrator Connection Fix
A minimal FastAPI server to test the frontend-backend connection
"""

import time
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="Simple Test Server",
    description="Minimal server for testing AI Orchestrator connection fix",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server_start_time = time.time()

class ChatRequest(BaseModel):
    message: str
    model: str = "test-model"

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: float

@app.get("/")
async def root():
    return {
        "message": "Simple Test Server",
        "status": "running",
        "uptime": time.time() - server_start_time
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - server_start_time
    }

@app.get("/api/chat/message/json")
async def chat_get():
    return {
        "response": "Test response from simple server",
        "model": "test-model",
        "timestamp": time.time()
    }

@app.post("/api/chat/message/json")
async def chat_post(request: ChatRequest):
    return ChatResponse(
        response=f"Echo: {request.message}",
        model=request.model,
        timestamp=time.time()
    )

@app.get("/api/status")
async def status():
    return {
        "status": "operational",
        "uptime": time.time() - server_start_time,
        "server": "simple_test_server"
    }

@app.get("/api/test")
async def test():
    return {
        "test": "success",
        "timestamp": time.time(),
        "message": "Simple test server is working"
    }

if __name__ == "__main__":
    print("🚀 Starting Simple Test Server on http://127.0.0.1:8008")
    print("📚 Endpoints: /health, /api/chat/message/json, /api/status, /api/test")
    print("🛑 Press Ctrl+C to stop")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8008,
        log_level="info"
    )