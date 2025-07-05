from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websockets"])

# Connection management
class ConnectionManager:
    """Manages WebSocket connections for real-time workflow updates"""
    
    def __init__(self):
        # Active connections: {connection_id: websocket}
        self.active_connections: Dict[str, WebSocket] = {}
        # Workflow subscriptions: {workflow_id: set of connection_ids}
        self.workflow_subscriptions: Dict[str, Set[str]] = {}
        # User subscriptions: {user_id: set of connection_ids}
        self.user_subscriptions: Dict[str, Set[str]] = {}
        # Connection metadata: {connection_id: metadata}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        # Store connection metadata
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "connected_at": datetime.now(),
            "subscribed_workflows": set()
        }
        
        # Subscribe to user-specific updates if user_id provided
        if user_id:
            if user_id not in self.user_subscriptions:
                self.user_subscriptions[user_id] = set()
            self.user_subscriptions[user_id].add(connection_id)
        
        logger.info(f"WebSocket connection {connection_id} established for user {user_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.active_connections:
            # Remove from workflow subscriptions
            for workflow_id, subscribers in self.workflow_subscriptions.items():
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
            
            logger.info(f"WebSocket connection {connection_id} disconnected")
    
    def subscribe_to_workflow(self, connection_id: str, workflow_id: str):
        """Subscribe a connection to workflow updates"""
        if workflow_id not in self.workflow_subscriptions:
            self.workflow_subscriptions[workflow_id] = set()
        
        self.workflow_subscriptions[workflow_id].add(connection_id)
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscribed_workflows"].add(workflow_id)
        
        logger.info(f"Connection {connection_id} subscribed to workflow {workflow_id}")
    
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
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send a message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast_to_workflow(self, message: dict, workflow_id: str):
        """Broadcast a message to all connections subscribed to a workflow"""
        if workflow_id in self.workflow_subscriptions:
            subscribers = self.workflow_subscriptions[workflow_id].copy()
            for connection_id in subscribers:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_user(self, message: dict, user_id: str):
        """Broadcast a message to all connections for a specific user"""
        if user_id in self.user_subscriptions:
            subscribers = self.user_subscriptions[user_id].copy()
            for connection_id in subscribers:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active connections"""
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about active connections"""
        return {
            "total_connections": len(self.active_connections),
            "workflow_subscriptions": {wf_id: len(subs) for wf_id, subs in self.workflow_subscriptions.items()},
            "user_subscriptions": {user_id: len(subs) for user_id, subs in self.user_subscriptions.items()},
            "active_workflows": len(self.workflow_subscriptions)
        }

# Global connection manager
manager = ConnectionManager()

# Event handlers for workflow events
class WorkflowEventHandler:
    """Handles workflow events and broadcasts them via WebSocket"""
    
    def __init__(self, connection_manager: ConnectionManager):
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

# Global event handler
event_handler = WorkflowEventHandler(manager)

# WebSocket message handlers
async def handle_websocket_message(websocket: WebSocket, connection_id: str, message: dict):
    """Handle incoming WebSocket messages"""
    try:
        message_type = message.get("type")
        
        if message_type == "subscribe_workflow":
            workflow_id = message.get("workflow_id")
            if workflow_id:
                manager.subscribe_to_workflow(connection_id, workflow_id)
                await manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "workflow_id": workflow_id,
                    "message": f"Subscribed to workflow {workflow_id}"
                }, connection_id)
        
        elif message_type == "unsubscribe_workflow":
            workflow_id = message.get("workflow_id")
            if workflow_id:
                manager.unsubscribe_from_workflow(connection_id, workflow_id)
                await manager.send_personal_message({
                    "type": "unsubscription_confirmed",
                    "workflow_id": workflow_id,
                    "message": f"Unsubscribed from workflow {workflow_id}"
                }, connection_id)
        
        elif message_type == "ping":
            await manager.send_personal_message({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }, connection_id)
        
        elif message_type == "get_stats":
            stats = manager.get_connection_stats()
            await manager.send_personal_message({
                "type": "stats",
                "data": stats
            }, connection_id)
        
        else:
            await manager.send_personal_message({
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }, connection_id)
    
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": f"Error processing message: {str(e)}"
        }, connection_id)

@router.websocket("/workflows")
async def workflow_websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = None
):
    """WebSocket endpoint for real-time workflow updates"""
    connection_id = str(uuid.uuid4())
    
    try:
        # Accept connection
        await manager.connect(websocket, connection_id, user_id)
        
        # Send welcome message
        await manager.send_personal_message({
            "type": "connected",
            "connection_id": connection_id,
            "message": "Connected to workflow updates",
            "timestamp": datetime.now().isoformat()
        }, connection_id)
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await handle_websocket_message(websocket, connection_id, message)
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format"
                }, connection_id)
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Internal error: {str(e)}"
                }, connection_id)
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(connection_id)

# Initialize event bus integration
async def setup_event_bus_integration():
    """Set up integration with the interactive workflow event bus"""
    try:
        # This would be called during app startup
        # For now, we'll create a placeholder that can be extended
        logger.info("WebSocket event bus integration initialized")
    except Exception as e:
        logger.error(f"Failed to setup event bus integration: {e}")

# Utility endpoints for WebSocket management
@router.get("/connections/stats")
async def get_connection_stats():
    """Get WebSocket connection statistics"""
    return manager.get_connection_stats()

@router.post("/broadcast")
async def broadcast_message(
    message: dict,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Broadcast a message to WebSocket connections (admin endpoint)"""
    try:
        if workflow_id:
            await manager.broadcast_to_workflow(message, workflow_id)
        elif user_id:
            await manager.broadcast_to_user(message, user_id)
        else:
            await manager.broadcast_to_all(message)
        
        return {"status": "success", "message": "Message broadcasted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast message: {str(e)}")