#!/usr/bin/env python3
"""
Test script to isolate the exact import causing the hang in app.py
"""

import os
import sys
from dotenv import load_dotenv

print("Starting app.py import simulation...")

try:
    print("1. Loading environment variables...")
    load_dotenv()
    print("✓ Environment variables loaded")
    
    print("2. Setting up Python path...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print("✓ Python path configured")
    
    print("3. Testing FastAPI import...")
    from fastapi import FastAPI, Depends, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    print("✓ FastAPI imported")
    
    print("4. Testing core.models import...")
    from core.models import AppState, Message, EnhancedTask, TaskStatus, LogEntry
    print("✓ core.models imported")
    
    print("5. Creating FastAPI app...")
    app = FastAPI(
        title="Sentient Core API",
        description="API for the Sentient Core Multi-Agent RAG System",
        version="0.1.0"
    )
    print("✓ FastAPI app created")
    
    print("6. Adding CORS middleware...")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("✓ CORS middleware added")
    
    print("7. Testing router imports one by one...")
    
    print("7a. Importing agents router...")
    from app.api.routers import agents
    print("✓ agents router imported")
    
    print("7b. Importing workflows router...")
    from app.api.routers import workflows
    print("✓ workflows router imported")
    
    print("7c. Importing chat router...")
    from app.api.routers import chat
    print("✓ chat router imported")
    
    print("7d. Importing core_services router...")
    from app.api.routers import core_services
    print("✓ core_services router imported")
    
    print("7e. Importing api_endpoints router...")
    from app.api.routers import api_endpoints
    print("✓ api_endpoints router imported")
    
    print("7f. Importing interactive_workflows router...")
    from app.api.routers import interactive_workflows
    print("✓ interactive_workflows router imported")
    
    print("7g. Importing research router...")
    from app.api.routers import research
    print("✓ research router imported")
    
    print("7h. Importing monitoring router...")
    from app.api.routers import monitoring
    print("✓ monitoring router imported")
    
    print("7i. Importing implementation router...")
    from app.api.routers import implementation
    print("✓ implementation router imported")
    
    print("8. Testing router registration...")
    app.include_router(agents.router, prefix="/api")
    app.include_router(workflows.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(core_services.router, prefix="/api")
    app.include_router(api_endpoints.router, prefix="/api")
    app.include_router(interactive_workflows.router, prefix="/api")
    app.include_router(research.router, prefix="/api")
    app.include_router(monitoring.router, prefix="/api")
    app.include_router(implementation.router, prefix="/api")
    print("✓ All routers registered")
    
    print("\n🎉 App import simulation completed successfully!")
    
except Exception as e:
    print(f"❌ Error during app import simulation: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    sys.exit(1)