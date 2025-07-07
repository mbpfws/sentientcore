from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from app.services.service_factory import ServiceFactory, get_service_factory

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

@router.get("/logs/{session_id}")
async def get_session_logs(
    session_id: str,
    limit: Optional[int] = Query(100, description="Maximum number of logs to return"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR)"),
    service_factory: ServiceFactory = Depends(get_service_factory)
) -> Dict[str, Any]:
    """
    Get logs for a specific session with optional filtering.
    Supports all three builds with enhanced log display.
    """
    try:
        state_manager = service_factory.state_manager
        session_data = await state_manager.get_session_state(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        logs = session_data.data.get('logs', []) if session_data.data else []
        
        # Filter by level if specified
        if level:
            logs = [log for log in logs if log.get('level', 'INFO') == level.upper()]
        
        # Limit results
        logs = logs[-limit:] if limit else logs
        
        # Convert logs to dict format for JSON serialization
        log_data = []
        for log in logs:
            if isinstance(log, dict):
                log_dict = {
                    "timestamp": log.get('timestamp', datetime.now().isoformat()),
                    "level": log.get('level', 'INFO'),
                    "source": log.get('source', 'Unknown'),
                    "message": log.get('message', str(log)),
                    "activity_type": log.get('activity_type', None)
                }
            else:
                log_dict = {
                    "timestamp": getattr(log, 'timestamp', datetime.now().isoformat()),
                    "level": getattr(log, 'level', 'INFO'),
                    "source": getattr(log, 'source', 'Unknown'),
                    "message": getattr(log, 'message', str(log)),
                    "activity_type": getattr(log, 'activity_type', None)
                }
            log_data.append(log_dict)
        
        return {
            "session_id": session_id,
            "total_logs": len(session_data.data.get('logs', [])) if session_data.data else 0,
            "filtered_logs": len(log_data),
            "logs": log_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving logs: {str(e)}")

@router.get("/artifacts/{session_id}")
async def get_session_artifacts(
    session_id: str,
    service_factory: ServiceFactory = Depends(get_service_factory)
) -> Dict[str, Any]:
    """
    Get research and planning artifacts for a specific session.
    Supports Build 2 (research) and Build 3 (planning) artifacts.
    """
    try:
        memory_service = service_factory.memory_service
        artifacts = {
            "research_artifacts": [],
            "planning_artifacts": [],
            "session_id": session_id
        }
        
        # Get research artifacts from Layer 1 memory
        try:
            research_docs_path = Path("./memory/layer1_research_docs")
            if research_docs_path.exists():
                for file_path in research_docs_path.glob("*.md"):
                    if session_id in file_path.name or "research" in file_path.name.lower():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        artifacts["research_artifacts"].append({
                            "filename": file_path.name,
                            "path": str(file_path),
                            "type": "research",
                            "size": len(content),
                            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                            "preview": content[:500] + "..." if len(content) > 500 else content
                        })
        except Exception as e:
            print(f"Error loading research artifacts: {e}")
        
        # Get planning artifacts from Layer 2 memory
        try:
            planning_docs_path = Path("./memory/layer2_planning_docs")
            if planning_docs_path.exists():
                for file_path in planning_docs_path.glob("*.md"):
                    if session_id in file_path.name or "prd" in file_path.name.lower():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        artifacts["planning_artifacts"].append({
                            "filename": file_path.name,
                            "path": str(file_path),
                            "type": "planning",
                            "size": len(content),
                            "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                            "preview": content[:500] + "..." if len(content) > 500 else content
                        })
        except Exception as e:
            print(f"Error loading planning artifacts: {e}")
        
        return artifacts
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving artifacts: {str(e)}")

@router.get("/download/{artifact_type}/{filename}")
async def download_artifact(
    artifact_type: str,
    filename: str
) -> Dict[str, Any]:
    """
    Download a specific artifact file (research or planning).
    Returns file content for PDF/Markdown download.
    """
    try:
        if artifact_type == "research":
            base_path = Path("./memory/layer1_research_docs")
        elif artifact_type == "planning":
            base_path = Path("./memory/layer2_planning_docs")
        else:
            raise HTTPException(status_code=400, detail="Invalid artifact type. Use 'research' or 'planning'")
        
        file_path = base_path / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Artifact {filename} not found")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "filename": filename,
            "type": artifact_type,
            "content": content,
            "size": len(content),
            "download_url": f"/api/monitoring/download/{artifact_type}/{filename}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading artifact: {str(e)}")

@router.get("/conversation/{session_id}")
async def get_conversation_history(
    session_id: str,
    limit: Optional[int] = Query(50, description="Maximum number of messages to return"),
    service_factory: ServiceFactory = Depends(get_service_factory)
) -> Dict[str, Any]:
    """
    Get enhanced conversation history for a session.
    Supports Build 1 conversation persistence with artifacts metadata.
    """
    try:
        state_manager = service_factory.state_manager
        session_data = await state_manager.get_session_state(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Get conversation history with enhanced metadata
        conversation_history = session_data.data.get('conversation_history', []) if session_data.data else []
        messages = session_data.data.get('messages', []) if session_data.data else []
        
        # Limit results
        conversation_history = conversation_history[-limit:] if limit else conversation_history
        
        return {
            "session_id": session_id,
            "total_messages": len(messages),
            "conversation_history": conversation_history,
            "has_research_artifacts": any(msg.get('research_artifacts') for msg in conversation_history if isinstance(msg, dict)),
            "has_planning_artifacts": any(msg.get('planning_artifacts') for msg in conversation_history if isinstance(msg, dict))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation history: {str(e)}")

@router.get("/status/{session_id}")
async def get_session_status(
    session_id: str,
    service_factory: ServiceFactory = Depends(get_service_factory)
) -> Dict[str, Any]:
    """
    Get comprehensive session status including all builds metrics.
    """
    try:
        state_manager = service_factory.state_manager
        session_data = await state_manager.load_session_data(session_id)
        
        if not session_data:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        # Calculate metrics
        messages = session_data.data.get('messages', []) if session_data.data else []
        logs = session_data.data.get('logs', []) if session_data.data else []
        conversation_history = session_data.data.get('conversation_history', []) if session_data.data else []
        
        # Count different types of activities
        research_logs = []
        planning_logs = []
        error_logs = []
        
        for log in logs:
            if isinstance(log, dict):
                message = log.get('message', '')
                source = log.get('source', '').lower()
                level = log.get('level', 'INFO')
            else:
                message = getattr(log, 'message', '')
                source = getattr(log, 'source', '').lower()
                level = getattr(log, 'level', 'INFO')
            
            if 'BUILD 2' in message or 'research' in source:
                research_logs.append(log)
            if 'BUILD 3' in message or 'planning' in source:
                planning_logs.append(log)
            if level == 'ERROR':
                error_logs.append(log)
        
        return {
            "session_id": session_id,
            "next_action": session_data.data.get('next_action', 'unknown') if session_data.data else 'unknown',
            "metrics": {
                "total_messages": len(messages),
                "total_logs": len(logs),
                "conversation_entries": len(conversation_history),
                "research_activities": len(research_logs),
                "planning_activities": len(planning_logs),
                "error_count": len(error_logs)
            },
            "last_activity": logs[-1].get('timestamp') if logs and isinstance(logs[-1], dict) else (getattr(logs[-1], 'timestamp', None) if logs else None),
            "builds_active": {
                "build1_conversation": len(conversation_history) > 0,
                "build2_research": len(research_logs) > 0,
                "build3_planning": len(planning_logs) > 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session status: {str(e)}")