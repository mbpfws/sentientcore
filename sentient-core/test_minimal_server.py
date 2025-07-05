#!/usr/bin/env python3
"""
Minimal test server to isolate endpoint issues
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Minimal Test Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Minimal test server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is operational"}

@app.get("/docs")
async def docs_redirect():
    return {"message": "API documentation available"}

@app.get("/api/state")
async def get_system_state():
    return {
        "status": "active",
        "agents": [],
        "workflows": [],
        "timestamp": "2025-01-01T00:00:00Z"
    }

@app.get("/api/memory/status")
async def get_memory_status():
    return {
        "status": "operational",
        "total_memories": 0,
        "memory_usage": "0MB"
    }

@app.post("/api/memory/store")
async def store_memory(data: dict):
    return {
        "success": True,
        "memory_id": "test-memory-123",
        "message": "Memory stored successfully"
    }

@app.get("/api/memory/retrieve")
async def retrieve_memory(query: str = "test"):
    return {
        "memories": [],
        "total": 0,
        "query": query
    }

@app.post("/api/agents/execute")
async def execute_agent(request: dict):
    return {
        "success": True,
        "agent_type": request.get("agent_type", "unknown"),
        "task": request.get("task", "unknown"),
        "result": "Mock execution completed",
        "execution_time": 0.1
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)