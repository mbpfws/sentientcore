from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from typing import List, Dict, Any, Optional

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AppState, Message, EnhancedTask, TaskStatus, LogEntry

app = FastAPI(
    title="Sentient Core API",
    description="API for the Sentient Core Multi-Agent RAG System",
    version="0.1.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from app.api.routers import agents, workflows, chat, core_services

# Register routers with /api prefix
app.include_router(agents.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(core_services.router, prefix="/api")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Sentient Core API is running"}
