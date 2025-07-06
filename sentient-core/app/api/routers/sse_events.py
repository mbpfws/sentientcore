from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Dict, List, Set, Optional, Any, AsyncGenerator
import asyncio
import json
import uuid
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Import the interactive workflow components
from core.state.interactive_workflow_models import (
    InteractiveWorkflowStatus,
    UserApprovalState,
    WorkflowStep,
    UserInteractionRequest
)
from core.state.interactive_state_manager import InteractiveStateManager, InteractiveWorkflowEventBus
from core.workflow.step_by_step_orchestrator import StepByStepOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sse", tags=["server-sent-events"])

# SSE Connection management
class SSEConnectionManager:
    """Manages Server-Sent Events connections for real-time updates"""
    
    def __init__(self):
        # Active connections: {connection_id: connection_info}
        self.active_connections: Dict[str, Dict[str, Any]] = {}
        # Workflow subscriptions: {workflow_id: set of connection_ids}
        self.workflow_subscriptions: Dict[str, Set[str]] = {}
        # User subscriptions: {user_id: set of connection_ids}
        self.user_subscriptions: Dict[str, Set[str]] = {}
        # Research subscriptions: {research_id: set of connection_ids}
        self.research_subscriptions: Dict[str, Set[str]] = {}
        # Message queues for each connection: {connection_id: asyncio.Queue}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        # Connection metadata: {connection_id: metadata}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_connection(self, connection_id: str, user_id: Optional[str] = None) -> str:
        """Create a new SSE connection"""
        # Create message queue for this connection
        self.message_queues[connection_id] = asyncio.Queue()
        
        # Store connection info
        self.active_connections[connection_id] = {
            "user_id": user_id,
            "connected_at": datetime.now(),
            "last_ping": datetime.now()
        }
        
        # Store connection metadata
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "connected_at": datetime.now(),
            "subscribed_workflows": set(),
            "subscribed_research": set()
        }
        
        # Subscribe to user-specific updates if user_id provided
        if user_id:
            if user_id not in self.user_subscriptions:
                self.user_subscriptions[user_id] = set()
            self.user_subscriptions[user_id].add(connection_id)
        
        logger.info(f"SSE connection {connection_id} created for user {user_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove an SSE connection"""
        if connection_id in self.active_connections:
            # Remove from workflow subscriptions
            for workflow_id, subscribers in self.workflow_subscriptions.items():
                subscribers.discard(connection_id)
            
            # Remove from research subscriptions
            for research_id, subscribers in self.research_subscriptions.items():
                subscribers.discard(connection_id)
            
            # Remove from user subscriptions
            metadata = self.connection_metadata.get(connection_id, {})
            user_id = metadata.get("user_id")
            if user_id and user_id in self.user_subscriptions:
                self.user_subscriptions[user_id].discard(connection_id)
                if not self.user_subscriptions[user_id]:
                    del self.user_subscriptions[user_id]
            
            # Clean up
            del self.active_connections[connection_id]
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            if connection_id in self.message_queues:
                del self.message_queues[connection_id]
            
            logger.info(f"SSE connection {connection_id} disconnected")
    
    def subscribe_to_workflow(self, connection_id: str, workflow_id: str):
        """Subscribe a connection to workflow updates"""
        if workflow_id not in self.workflow_subscriptions:
            self.workflow_subscriptions[workflow_id] = set()
        
        self.workflow_subscriptions[workflow_id].add(connection_id)
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscribed_workflows"].add(workflow_id)
        
        logger.info(f"Connection {connection_id} subscribed to workflow {workflow_id}")
    
    def subscribe_to_research(self, connection_id: str, research_id: str):
        """Subscribe a connection to research updates"""
        if research_id not in self.research_subscriptions:
            self.research_subscriptions[research_id] = set()
        
        self.research_subscriptions[research_id].add(connection_id)
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscribed_research"].add(research_id)
        
        logger.info(f"Connection {connection_id} subscribed to research {research_id}")
    
    def unsubscribe_from_workflow(self, connection_id: str, workflow_id: str):
        """Unsubscribe a connection from workflow updates"""
        if workflow_id in self.workflow_subscriptions:
            self.workflow_subscriptions[workflow_id].discard(connection_id)
            if not self.workflow_subscriptions[workflow_id]:
                del self.workflow_subscriptions[workflow_id]
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscribed_workflows"].discard(workflow_id)
        
        logger.info(f"Connection {connection_id} unsubscribed from workflow {workflow_id}")
    
    def unsubscribe_from_research(self, connection_id: str, research_id: str):
        """Unsubscribe a connection from research updates"""
        if research_id in self.research_subscriptions:
            self.research_subscriptions[research_id].discard(connection_id)
            if not self.research_subscriptions[research_id]:
                del self.research_subscriptions[research_id]
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscribed_research"].discard(research_id)
        
        logger.info(f"Connection {connection_id} unsubscribed from research {research_id}")
    
    async def send_message(self, connection_id: str, message: dict):
        """Send a message to a specific connection"""
        if connection_id in self.message_queues:
            try:
                await self.message_queues[connection_id].put(message)
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_to_workflow(self, message: dict, workflow_id: str):
        """Broadcast a message to all connections subscribed to a workflow"""
        if workflow_id in self.workflow_subscriptions:
            subscribers = self.workflow_subscriptions[workflow_id].copy()
            for connection_id in subscribers:
                await self.send_message(connection_id, message)
    
    async def broadcast_to_research(self, message: dict, research_id: str):
        """Broadcast a message to all connections subscribed to research"""
        if research_id in self.research_subscriptions:
            subscribers = self.research_subscriptions[research_id].copy()
            for connection_id in subscribers:
                await self.send_message(connection_id, message)
    
    async def broadcast_to_user(self, message: dict, user_id: str):
        """Broadcast a message to all connections for a specific user"""
        if user_id in self.user_subscriptions:
            subscribers = self.user_subscriptions[user_id].copy()
            for connection_id in subscribers:
                await self.send_message(connection_id, message)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active connections"""
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.send_message(connection_id, message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections"""
        return {
            "total_connections": len(self.active_connections),
            "workflow_subscriptions": {wf_id: len(subs) for wf_id, subs in self.workflow_subscriptions.items()},
            "research_subscriptions": {res_id: len(subs) for res_id, subs in self.research_subscriptions.items()},
            "user_subscriptions": {user_id: len(subs) for user_id, subs in self.user_subscriptions.items()},
            "active_workflows": len(self.workflow_subscriptions),
            "active_research": len(self.research_subscriptions)
        }
    
    async def get_message_stream(self, connection_id: str) -> AsyncGenerator[str, None]:
        """Get message stream for a specific connection"""
        if connection_id not in self.message_queues:
            return
        
        queue = self.message_queues[connection_id]
        
        try:
            # Send initial connection message
            initial_message = {
                "type": "connected",
                "connection_id": connection_id,
                "message": "Connected to SSE stream",
                "timestamp": datetime.now().isoformat()
            }
            yield f"data: {json.dumps(initial_message)}\n\n"
            
            # Send periodic heartbeat and process messages
            while connection_id in self.active_connections:
                try:
                    # Wait for message with timeout for heartbeat
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(message)}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat
                    heartbeat = {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                except Exception as e:
                    logger.error(f"Error in message stream for {connection_id}: {e}")
                    break
        except Exception as e:
            logger.error(f"Error in SSE stream for {connection_id}: {e}")
        finally:
            self.disconnect(connection_id)

# Global SSE connection manager
sse_manager = SSEConnectionManager()

# Event handlers for workflow events
class SSEEventHandler:
    """Handles workflow events and broadcasts them via SSE"""
    
    def __init__(self, connection_manager: SSEConnectionManager):
        self.manager = connection_manager
    
    async def handle_workflow_created(self, event_data: Dict[str, Any]):
        """Handle workflow creation events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "workflow_created",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_all(message)
    
    async def handle_workflow_started(self, event_data: Dict[str, Any]):
        """Handle workflow start events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "workflow_started",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_step_started(self, event_data: Dict[str, Any]):
        """Handle step start events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "step_started",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_step_completed(self, event_data: Dict[str, Any]):
        """Handle step completion events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "step_completed",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_approval_requested(self, event_data: Dict[str, Any]):
        """Handle approval request events"""
        workflow_id = event_data.get("workflow_id")
        user_id = event_data.get("user_id")  # If specific user approval needed
        
        message = {
            "type": "approval_requested",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
            "requires_action": True
        }
        
        if user_id:
            await self.manager.broadcast_to_user(message, user_id)
        else:
            await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_approval_submitted(self, event_data: Dict[str, Any]):
        """Handle approval submission events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "approval_submitted",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_workflow_paused(self, event_data: Dict[str, Any]):
        """Handle workflow pause events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "workflow_paused",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_workflow_resumed(self, event_data: Dict[str, Any]):
        """Handle workflow resume events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "workflow_resumed",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_workflow_completed(self, event_data: Dict[str, Any]):
        """Handle workflow completion events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "workflow_completed",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_workflow_error(self, event_data: Dict[str, Any]):
        """Handle workflow error events"""
        workflow_id = event_data.get("workflow_id")
        message = {
            "type": "workflow_error",
            "workflow_id": workflow_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat(),
            "severity": "error"
        }
        await self.manager.broadcast_to_workflow(message, workflow_id)
    
    async def handle_research_update(self, event_data: Dict[str, Any]):
        """Handle research update events"""
        research_id = event_data.get("research_id")
        message = {
            "type": "research_update",
            "research_id": research_id,
            "data": event_data,
            "timestamp": datetime.now().isoformat()
        }
        await self.manager.broadcast_to_research(message, research_id)

# Global event handler
sse_event_handler = SSEEventHandler(sse_manager)

# SSE Endpoints
@router.get("/workflows")
async def workflow_sse_endpoint(
    request: Request,
    user_id: Optional[str] = None
):
    """SSE endpoint for real-time workflow updates"""
    connection_id = str(uuid.uuid4())
    
    # Create connection
    sse_manager.create_connection(connection_id, user_id)
    
    # Return SSE stream
    return StreamingResponse(
        sse_manager.get_message_stream(connection_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/research/{research_id}")
async def research_sse_endpoint(
    request: Request,
    research_id: str
):
    """SSE endpoint for real-time research updates"""
    connection_id = str(uuid.uuid4())
    
    # Create connection and subscribe to research
    sse_manager.create_connection(connection_id)
    sse_manager.subscribe_to_research(connection_id, research_id)
    
    # Return SSE stream
    return StreamingResponse(
        sse_manager.get_message_stream(connection_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

# Management endpoints
@router.post("/subscribe/workflow/{workflow_id}")
async def subscribe_to_workflow(
    workflow_id: str,
    connection_id: str
):
    """Subscribe a connection to workflow updates"""
    try:
        sse_manager.subscribe_to_workflow(connection_id, workflow_id)
        return {"status": "success", "message": f"Subscribed to workflow {workflow_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to subscribe: {str(e)}")

@router.post("/subscribe/research/{research_id}")
async def subscribe_to_research(
    research_id: str,
    connection_id: str
):
    """Subscribe a connection to research updates"""
    try:
        sse_manager.subscribe_to_research(connection_id, research_id)
        return {"status": "success", "message": f"Subscribed to research {research_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to subscribe: {str(e)}")

@router.get("/connections/stats")
async def get_connection_stats():
    """Get SSE connection statistics"""
    return sse_manager.get_connection_stats()

@router.post("/broadcast")
async def broadcast_message(
    message: dict,
    workflow_id: Optional[str] = None,
    research_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Broadcast a message to SSE connections (admin endpoint)"""
    try:
        if workflow_id:
            await sse_manager.broadcast_to_workflow(message, workflow_id)
        elif research_id:
            await sse_manager.broadcast_to_research(message, research_id)
        elif user_id:
            await sse_manager.broadcast_to_user(message, user_id)
        else:
            await sse_manager.broadcast_to_all(message)
        
        return {"status": "success", "message": "Message broadcasted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast message: {str(e)}")

# Initialize event bus integration
async def setup_sse_event_bus_integration():
    """Set up integration with the interactive workflow event bus"""
    try:
        # This would be called during app startup
        # For now, we'll create a placeholder that can be extended
        logger.info("SSE event bus integration initialized")
    except Exception as e:
        logger.error(f"Failed to setup SSE event bus integration: {e}")