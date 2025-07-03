from fastapi import APIRouter, HTTPException, Request, Body
from typing import List, Dict, Any, Optional
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AppState, TaskStatus, EnhancedTask, LogEntry
# from graphs.intelligent_rag_graph import app as intelligent_rag_app
# from graphs.multi_agent_rag_graph import app as multi_agent_app
# from graphs.orchestration_graph import app as orchestration_app

router = APIRouter(prefix="/workflows", tags=["workflows"])

@router.get("/")
async def list_workflows():
    """List available workflow modes"""
    return [
        {"id": "intelligent", "name": "Intelligent RAG", "description": "Natural language orchestration with intelligent workflow management"},
        {"id": "multi_agent", "name": "Multi-Agent RAG", "description": "Collaborative multi-agent system with specialized workflows"},
        {"id": "legacy", "name": "Legacy Orchestration", "description": "Standard orchestration workflow"}
    ]

@router.post("/{workflow_id}/execute")
async def execute_workflow(workflow_id: str, app_state: AppState):
    """Execute the specified workflow with the provided application state"""
    try:
        # Temporarily disabled - missing graph modules
        raise HTTPException(status_code=501, detail=f"Workflow execution temporarily disabled - missing graph modules")
        
        # if workflow_id == "intelligent":
        #     # Use the intelligent RAG workflow
        #     result = intelligent_rag_app.invoke(app_state.model_dump())
        # elif workflow_id == "multi_agent":
        #     # Use the multi-agent RAG workflow
        #     result = multi_agent_app.invoke(app_state.model_dump())
        # elif workflow_id == "legacy":
        #     # Use legacy orchestration workflow
        #     result = orchestration_app.invoke(app_state.model_dump())
        # else:
        #     raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        # 
        # # Return the updated state
        # return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution error: {str(e)}")

@router.post("/tasks/{task_id}/execute")
async def execute_task(task_id: str, app_state: AppState):
    """Execute a specific task within the workflow"""
    try:
        # Temporarily disabled - missing graph modules
        raise HTTPException(status_code=501, detail=f"Task execution temporarily disabled - missing graph modules")
        
        # # Set the task to execute
        # app_state.task_to_run_id = task_id
        # 
        # # Execute through orchestration graph
        # result = orchestration_app.invoke(app_state.model_dump())
        # 
        # return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution error: {str(e)}")

@router.get("/tasks/status")
async def get_tasks_status(app_state: AppState = Body(...)):
    """Get the status of all tasks in the current workflow"""
    try:
        task_stats = {
            "total": len(app_state.tasks),
            "pending": sum(1 for task in app_state.tasks if task.status == TaskStatus.PENDING),
            "in_progress": sum(1 for task in app_state.tasks if task.status == TaskStatus.IN_PROGRESS),
            "completed": sum(1 for task in app_state.tasks if task.status == TaskStatus.COMPLETED),
            "failed": sum(1 for task in app_state.tasks if task.status == TaskStatus.FAILED),
            "tasks": [{
                "id": task.id,
                "title": task.title,
                "description": task.description,
                "status": task.status,
                "agent": task.agent_type,
                "sequence": task.sequence,
                "created_at": task.created_at.isoformat() if task.created_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            } for task in app_state.tasks]
        }
        return task_stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting task status: {str(e)}")
