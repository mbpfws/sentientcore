from typing import Dict, List, Optional, Any, AsyncGenerator, Set
import asyncio
import json
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import Request
from fastapi.responses import StreamingResponse
import logging


class EventType(Enum):
    """Types of SSE events"""
    CHAT_MESSAGE = "chat_message"
    RESEARCH_PROGRESS = "research_progress"
    RESEARCH_COMPLETE = "research_complete"
    WORKFLOW_UPDATE = "workflow_update"
    AGENT_STATUS = "agent_status"
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SESSION_UPDATE = "session_update"
    MEMORY_UPDATE = "memory_update"
    STATE_CHANGE = "state_change"


@dataclass
class SSEEvent:
    """Server-Sent Event data structure"""
    id: str
    event_type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    session_id: Optional[str] = None
    target_clients: Optional[Set[str]] = None  # Specific client IDs, None = broadcast
    
    def to_sse_format(self) -> str:
        """Convert to SSE format string"""
        lines = []
        lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event_type.value}")
        
        # Prepare data payload
        payload = {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            **self.data
        }
        
        # Split data into multiple lines if needed
        data_json = json.dumps(payload)
        for line in data_json.split('\n'):
            lines.append(f"data: {line}")
        
        lines.append("")  # Empty line to end the event
        return "\n".join(lines)


@dataclass
class SSEClient:
    """Represents a connected SSE client"""
    id: str
    session_id: Optional[str]
    connected_at: datetime
    last_heartbeat: datetime
    event_queue: asyncio.Queue
    subscribed_events: Set[EventType]
    metadata: Dict[str, Any]
    
    def is_interested_in_event(self, event: SSEEvent) -> bool:
        """Check if client is interested in this event"""
        # Check event type subscription
        if event.event_type not in self.subscribed_events:
            return False
        
        # Check session filtering
        if event.session_id and self.session_id and event.session_id != self.session_id:
            return False
        
        # Check target client filtering
        if event.target_clients and self.id not in event.target_clients:
            return False
        
        return True


class SSEConnectionManager:
    """Manages Server-Sent Events connections and broadcasting"""
    
    def __init__(self, heartbeat_interval: int = 30, max_queue_size: int = 100):
        self.heartbeat_interval = heartbeat_interval
        self.max_queue_size = max_queue_size
        
        # Connected clients
        self._clients: Dict[str, SSEClient] = {}
        self._clients_lock = asyncio.Lock()
        
        # Event broadcasting
        self._event_history: List[SSEEvent] = []
        self._max_history = 1000
        
        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self._stats = {
            "total_connections": 0,
            "active_connections": 0,
            "events_sent": 0,
            "events_dropped": 0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def connect_client(
        self,
        request: Request,
        session_id: Optional[str] = None,
        subscribed_events: Optional[List[EventType]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StreamingResponse:
        """Connect a new SSE client"""
        client_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Default to all events if none specified
        if subscribed_events is None:
            subscribed_events = list(EventType)
        
        client = SSEClient(
            id=client_id,
            session_id=session_id,
            connected_at=now,
            last_heartbeat=now,
            event_queue=asyncio.Queue(maxsize=self.max_queue_size),
            subscribed_events=set(subscribed_events),
            metadata=metadata or {}
        )
        
        async with self._clients_lock:
            self._clients[client_id] = client
            self._stats["total_connections"] += 1
            self._stats["active_connections"] = len(self._clients)
        
        self.logger.info(f"SSE client connected: {client_id} (session: {session_id})")
        
        # Send connection confirmation
        await self._send_to_client(
            client,
            SSEEvent(
                id=str(uuid.uuid4()),
                event_type=EventType.SYSTEM_STATUS,
                data={
                    "status": "connected",
                    "client_id": client_id,
                    "message": "SSE connection established"
                },
                timestamp=now,
                session_id=session_id
            )
        )
        
        # Return streaming response
        return StreamingResponse(
            self._client_event_stream(client),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
    
    async def disconnect_client(self, client_id: str):
        """Disconnect a client"""
        async with self._clients_lock:
            client = self._clients.pop(client_id, None)
            if client:
                self._stats["active_connections"] = len(self._clients)
                self.logger.info(f"SSE client disconnected: {client_id}")
    
    async def broadcast_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
        target_clients: Optional[Set[str]] = None
    ) -> int:
        """Broadcast an event to interested clients"""
        event = SSEEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            session_id=session_id,
            target_clients=target_clients
        )
        
        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Send to interested clients
        sent_count = 0
        async with self._clients_lock:
            clients_to_remove = []
            
            for client_id, client in self._clients.items():
                if client.is_interested_in_event(event):
                    try:
                        await self._send_to_client(client, event)
                        sent_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to send event to client {client_id}: {e}")
                        clients_to_remove.append(client_id)
            
            # Remove failed clients
            for client_id in clients_to_remove:
                self._clients.pop(client_id, None)
            
            if clients_to_remove:
                self._stats["active_connections"] = len(self._clients)
        
        self._stats["events_sent"] += sent_count
        return sent_count
    
    async def send_to_session(
        self,
        session_id: str,
        event_type: EventType,
        data: Dict[str, Any]
    ) -> int:
        """Send event to all clients in a specific session"""
        return await self.broadcast_event(
            event_type=event_type,
            data=data,
            session_id=session_id
        )
    
    async def send_to_client(
        self,
        client_id: str,
        event_type: EventType,
        data: Dict[str, Any]
    ) -> bool:
        """Send event to a specific client"""
        return await self.broadcast_event(
            event_type=event_type,
            data=data,
            target_clients={client_id}
        ) > 0
    
    async def _client_event_stream(self, client: SSEClient) -> AsyncGenerator[str, None]:
        """Generate SSE stream for a client"""
        try:
            while True:
                try:
                    # Wait for event with timeout for heartbeat
                    event = await asyncio.wait_for(
                        client.event_queue.get(),
                        timeout=self.heartbeat_interval
                    )
                    yield event.to_sse_format()
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    heartbeat = SSEEvent(
                        id=str(uuid.uuid4()),
                        event_type=EventType.HEARTBEAT,
                        data={"timestamp": datetime.now().isoformat()},
                        timestamp=datetime.now()
                    )
                    yield heartbeat.to_sse_format()
                    client.last_heartbeat = datetime.now()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in client event stream: {e}")
        finally:
            await self.disconnect_client(client.id)
    
    async def _send_to_client(self, client: SSEClient, event: SSEEvent):
        """Send event to a specific client"""
        try:
            client.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            # Drop oldest event and add new one
            try:
                client.event_queue.get_nowait()
                client.event_queue.put_nowait(event)
                self._stats["events_dropped"] += 1
            except asyncio.QueueEmpty:
                pass
    
    async def _heartbeat_loop(self):
        """Background task to send heartbeats and detect disconnected clients"""
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                now = datetime.now()
                timeout_threshold = now.timestamp() - (self.heartbeat_interval * 3)
                
                async with self._clients_lock:
                    disconnected_clients = [
                        client_id for client_id, client in self._clients.items()
                        if client.last_heartbeat.timestamp() < timeout_threshold
                    ]
                    
                    for client_id in disconnected_clients:
                        self._clients.pop(client_id, None)
                        self.logger.info(f"Client {client_id} timed out")
                    
                    if disconnected_clients:
                        self._stats["active_connections"] = len(self._clients)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    async def _cleanup_loop(self):
        """Background task for periodic cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup old events from history
                if len(self._event_history) > self._max_history:
                    self._event_history = self._event_history[-self._max_history:]
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection and event statistics"""
        return {
            **self._stats,
            "active_connections": len(self._clients),
            "event_history_size": len(self._event_history)
        }
    
    def get_connected_clients(self) -> List[Dict[str, Any]]:
        """Get information about connected clients"""
        clients_info = []
        for client in self._clients.values():
            clients_info.append({
                "id": client.id,
                "session_id": client.session_id,
                "connected_at": client.connected_at.isoformat(),
                "last_heartbeat": client.last_heartbeat.isoformat(),
                "subscribed_events": [event.value for event in client.subscribed_events],
                "queue_size": client.event_queue.qsize(),
                "metadata": client.metadata
            })
        return clients_info
    
    async def shutdown(self):
        """Shutdown the SSE manager"""
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Disconnect all clients
        async with self._clients_lock:
            client_ids = list(self._clients.keys())
            for client_id in client_ids:
                await self.disconnect_client(client_id)
        
        self.logger.info("SSE Connection Manager shutdown complete")


# Global SSE manager instance
sse_manager = SSEConnectionManager()


# Convenience functions for common event types
async def broadcast_chat_message(
    message: str,
    sender: str,
    session_id: str,
    message_type: str = "assistant"
):
    """Broadcast a chat message event"""
    await sse_manager.broadcast_event(
        event_type=EventType.CHAT_MESSAGE,
        data={
            "message": message,
            "sender": sender,
            "message_type": message_type
        },
        session_id=session_id
    )


async def broadcast_research_progress(
    task_id: str,
    progress: float,
    status: str,
    details: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Broadcast research progress event"""
    await sse_manager.broadcast_event(
        event_type=EventType.RESEARCH_PROGRESS,
        data={
            "task_id": task_id,
            "progress": progress,
            "status": status,
            "details": details
        },
        session_id=session_id
    )


async def broadcast_workflow_update(
    workflow_id: str,
    status: str,
    step: Optional[str] = None,
    session_id: Optional[str] = None
):
    """Broadcast workflow update event"""
    await sse_manager.broadcast_event(
        event_type=EventType.WORKFLOW_UPDATE,
        data={
            "workflow_id": workflow_id,
            "status": status,
            "step": step
        },
        session_id=session_id
    )


async def broadcast_agent_status(
    agent_id: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
):
    """Broadcast agent status event"""
    await sse_manager.broadcast_event(
        event_type=EventType.AGENT_STATUS,
        data={
            "agent_id": agent_id,
            "status": status,
            "details": details or {}
        },
        session_id=session_id
    )


async def broadcast_error(
    error_type: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
):
    """Broadcast error event"""
    await sse_manager.broadcast_event(
        event_type=EventType.ERROR,
        data={
            "error_type": error_type,
            "message": message,
            "details": details or {}
        },
        session_id=session_id
    )