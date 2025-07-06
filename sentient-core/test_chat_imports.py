#!/usr/bin/env python3
"""
Test script to replicate the exact import chain used by the chat router.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting chat router import chain test...")

try:
    print("1. Loading environment variables...")
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment variables loaded")
    
    print("2. Testing core.models import...")
    from core.models import AppState, Message, LogEntry
    print("✓ Core models imported")
    
    print("3. Testing core.agents.ultra_orchestrator import...")
    from core.agents.ultra_orchestrator import UltraOrchestrator
    print("✓ UltraOrchestrator imported")
    
    print("4. Testing core.services.enhanced_llm_service import...")
    from core.services.enhanced_llm_service import EnhancedLLMService
    print("✓ EnhancedLLMService imported")
    
    print("5. Testing core.services.session_persistence_service import...")
    from core.services.session_persistence_service import SessionPersistenceService
    print("✓ SessionPersistenceService imported")
    
    print("6. Testing core.services.memory_service import...")
    from core.services.memory_service import MemoryService
    print("✓ MemoryService imported")
    
    print("7. Testing sentient_workflow_graph import...")
    from core.graphs.sentient_workflow_graph import get_sentient_workflow_app
    print("✓ sentient_workflow_graph imported")
    
    print("8. Testing get_sentient_workflow_app call...")
    workflow_app = get_sentient_workflow_app()
    print(f"✓ get_sentient_workflow_app() returned: {type(workflow_app).__name__}")
    
    print("9. Testing FastAPI imports...")
    from fastapi import APIRouter, HTTPException, UploadFile, File, Form
    from fastapi.responses import JSONResponse, FileResponse
    print("✓ FastAPI imports successful")
    
    print("10. Testing Pydantic imports...")
    from pydantic import BaseModel, Field
    print("✓ Pydantic imports successful")
    
    print("11. Testing other imports...")
    from typing import Optional, List, Dict, Any
    import json
    import uuid
    import base64
    import asyncio
    import os
    from datetime import datetime
    print("✓ Other imports successful")
    
    print("\n🎉 All chat router imports completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during chat router import chain: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    sys.exit(1)