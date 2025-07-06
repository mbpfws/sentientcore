from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Set, Optional, Any
import asyncio
import json
import uuid
from datetime import datetime
import logging

# Import the interactive workflow components
from core.state.interactive_workflow_models import (
    InteractiveWorkflowStatus,
    UserApprovalState,
    WorkflowStep,
    UserInteractionRequest
)
from core.state.interactive_state_manager import InteractiveStateManager, InteractiveWorkflowEventBus
from core.workflow.step_by_step_orchestrator import StepByStepOrchestrator

# Import SSE infrastructure
from .sse_events import sse_manager, sse_event_handler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflow-events"])

# SSE Integration Functions
def get_sse_manager():
    """Get the global SSE connection manager"""
    return sse_manager

def get_sse_event_handler():
    """Get the global SSE event handler"""
    return sse_event_handler

# SSE Event Handler Functions
async def handle_approval_request(workflow_id: str, step_id: str, approval_data: dict):
    """Handle workflow approval request via SSE"""
    await sse_event_handler.handle_approval_request(workflow_id, step_id, approval_data)
    logger.info(f"Approval request sent for workflow {workflow_id}, step {step_id}")

async def handle_approval_submission(workflow_id: str, step_id: str, approval_result: dict):
    """Handle workflow approval submission via SSE"""
    await sse_event_handler.handle_approval_submission(workflow_id, step_id, approval_result)
    logger.info(f"Approval submitted for workflow {workflow_id}, step {step_id}")

async def handle_workflow_pause(workflow_id: str, reason: str = None):
    """Handle workflow pause event via SSE"""
    await sse_event_handler.handle_workflow_pause(workflow_id, reason)
    logger.info(f"Workflow {workflow_id} paused: {reason}")

async def handle_workflow_resume(workflow_id: str):
    """Handle workflow resume event via SSE"""
    await sse_event_handler.handle_workflow_resume(workflow_id)
    logger.info(f"Workflow {workflow_id} resumed")

async def handle_workflow_completion(workflow_id: str, result: dict):
    """Handle workflow completion event via SSE"""
    await sse_event_handler.handle_workflow_completion(workflow_id, result)
    logger.info(f"Workflow {workflow_id} completed")

async def handle_workflow_error(workflow_id: str, error: str, step_id: str = None):
    """Handle workflow error event via SSE"""
    await sse_event_handler.handle_workflow_error(workflow_id, error, step_id)
    logger.error(f"Workflow {workflow_id} error: {error}")

async def handle_step_update(workflow_id: str, step_id: str, step_data: dict):
    """Handle workflow step update via SSE"""
    await sse_event_handler.handle_step_update(workflow_id, step_id, step_data)
    logger.info(f"Step update for workflow {workflow_id}, step {step_id}")

# SSE Message Handling
# Note: SSE is unidirectional, so client subscriptions are handled via HTTP endpoints
# in the sse_events.py module

# SSE endpoints are now handled in sse_events.py
# This file maintains backward compatibility for existing workflow event functions

# Initialize event bus integration
async def setup_event_bus_integration():
    """Set up integration with the interactive workflow event bus"""
    try:
        # This would be called during app startup
        # For now, we'll create a placeholder that can be extended
        logger.info("SSE event bus integration initialized")
    except Exception as e:
        logger.error(f"Failed to setup event bus integration: {e}")

# Utility endpoints for SSE management
@router.get("/connections/stats")
async def get_connection_stats():
    """Get SSE connection statistics"""
    return sse_manager.get_connection_stats()

@router.post("/broadcast")
async def broadcast_message(
    message: dict,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Broadcast a message to SSE connections (admin endpoint)"""
    try:
        if workflow_id:
            await sse_manager.broadcast_to_workflow(message, workflow_id)
        elif user_id:
            await sse_manager.broadcast_to_user(message, user_id)
        else:
            await sse_manager.broadcast_to_all(message)
        
        return {"status": "success", "message": "Message broadcasted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast message: {str(e)}")