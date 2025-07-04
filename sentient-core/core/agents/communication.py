"""Agent Communication Protocol and Message Bus

This module implements the inter-agent communication system including:
- AgentMessage for structured communication
- MessageBus for routing and delivery
- Event-driven communication patterns
- Message persistence and replay capabilities
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from dataclasses import dataclass, asdict
import json
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    """Types of messages that can be sent between agents"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    ERROR_NOTIFICATION = "error_notification"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    COORDINATION = "coordination"
    BROADCAST = "broadcast"
    HEARTBEAT = "heartbeat"
    WORKFLOW_EVENT = "workflow_event"
    CUSTOM = "custom"

class MessagePriority(str, Enum):
    """Message priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"

@dataclass
class AgentMessage:
    """Structured message for inter-agent communication"""
    id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast messages
    message_type: MessageType
    priority: MessagePriority
    payload: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # For request-response correlation
    reply_to: Optional[str] = None  # For response messages
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def create_response(self, 
                       sender_id: str, 
                       payload: Dict[str, Any],
                       message_type: MessageType = MessageType.TASK_RESPONSE) -> 'AgentMessage':
        """Create a response message to this message"""
        return AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=self.sender_id,
            message_type=message_type,
            priority=self.priority,
            payload=payload,
            timestamp=datetime.utcnow(),
            correlation_id=self.correlation_id or self.id,
            reply_to=self.id
        )

class MessageHandler:
    """Base class for message handlers"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.subscriptions: Set[MessageType] = set()
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming message and optionally return a response"""
        raise NotImplementedError
    
    def subscribe(self, message_type: MessageType):
        """Subscribe to a message type"""
        self.subscriptions.add(message_type)
    
    def unsubscribe(self, message_type: MessageType):
        """Unsubscribe from a message type"""
        self.subscriptions.discard(message_type)

class MessageBus:
    """Central message bus for agent communication
    
    Provides:
    - Message routing and delivery
    - Subscription management
    - Message persistence
    - Broadcast capabilities
    - Message replay and recovery
    """
    
    def __init__(self, max_history: int = 10000):
        self.handlers: Dict[str, MessageHandler] = {}
        self.subscriptions: Dict[MessageType, Set[str]] = defaultdict(set)
        self.message_history: deque = deque(maxlen=max_history)
        self.pending_messages: Dict[str, List[AgentMessage]] = defaultdict(list)
        self.correlation_map: Dict[str, List[AgentMessage]] = defaultdict(list)
        self.running = False
        self.delivery_queue: asyncio.Queue = asyncio.Queue()
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "broadcasts_sent": 0
        }
        
        logger.info("MessageBus initialized")
    
    async def start(self):
        """Start the message bus"""
        self.running = True
        # Start message delivery task
        asyncio.create_task(self._delivery_worker())
        logger.info("MessageBus started")
    
    async def stop(self):
        """Stop the message bus"""
        self.running = False
        logger.info("MessageBus stopped")
    
    def register_handler(self, handler: MessageHandler):
        """Register a message handler for an agent"""
        self.handlers[handler.agent_id] = handler
        
        # Update subscriptions
        for message_type in handler.subscriptions:
            self.subscriptions[message_type].add(handler.agent_id)
        
        logger.info(f"Registered handler for agent {handler.agent_id}")
    
    def unregister_handler(self, agent_id: str):
        """Unregister a message handler"""
        if agent_id in self.handlers:
            handler = self.handlers[agent_id]
            
            # Remove from subscriptions
            for message_type in handler.subscriptions:
                self.subscriptions[message_type].discard(agent_id)
            
            del self.handlers[agent_id]
            logger.info(f"Unregistered handler for agent {agent_id}")
    
    def subscribe(self, agent_id: str, message_type: MessageType):
        """Subscribe an agent to a message type"""
        self.subscriptions[message_type].add(agent_id)
        
        if agent_id in self.handlers:
            self.handlers[agent_id].subscribe(message_type)
        
        logger.debug(f"Agent {agent_id} subscribed to {message_type.value}")
    
    def unsubscribe(self, agent_id: str, message_type: MessageType):
        """Unsubscribe an agent from a message type"""
        self.subscriptions[message_type].discard(agent_id)
        
        if agent_id in self.handlers:
            self.handlers[agent_id].unsubscribe(message_type)
        
        logger.debug(f"Agent {agent_id} unsubscribed from {message_type.value}")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the bus
        
        Args:
            message: Message to send
            
        Returns:
            True if message was queued successfully
        """
        try:
            # Check if message is expired
            if message.is_expired():
                logger.warning(f"Message {message.id} expired before sending")
                return False
            
            # Add to history
            self.message_history.append(message)
            
            # Add correlation tracking
            if message.correlation_id:
                self.correlation_map[message.correlation_id].append(message)
            
            # Queue for delivery
            await self.delivery_queue.put(message)
            
            self.stats["messages_sent"] += 1
            
            logger.debug(f"Message {message.id} queued for delivery")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {str(e)}")
            self.stats["messages_failed"] += 1
            return False
    
    async def broadcast_message(self, 
                               sender_id: str,
                               message_type: MessageType,
                               payload: Dict[str, Any],
                               priority: MessagePriority = MessagePriority.NORMAL) -> bool:
        """Broadcast a message to all subscribed agents
        
        Args:
            sender_id: ID of the sending agent
            message_type: Type of message to broadcast
            payload: Message payload
            priority: Message priority
            
        Returns:
            True if broadcast was successful
        """
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            message_type=message_type,
            priority=priority,
            payload=payload,
            timestamp=datetime.utcnow()
        )
        
        success = await self.send_message(message)
        if success:
            self.stats["broadcasts_sent"] += 1
        
        return success
    
    async def _delivery_worker(self):
        """Background worker for message delivery"""
        while self.running:
            try:
                # Get next message with timeout
                message = await asyncio.wait_for(
                    self.delivery_queue.get(), 
                    timeout=1.0
                )
                
                await self._deliver_message(message)
                
            except asyncio.TimeoutError:
                # No messages to process, continue
                continue
            except Exception as e:
                logger.error(f"Error in delivery worker: {str(e)}")
    
    async def _deliver_message(self, message: AgentMessage):
        """Deliver a message to its recipients"""
        try:
            # Check if message is expired
            if message.is_expired():
                logger.warning(f"Message {message.id} expired during delivery")
                return
            
            recipients = []
            
            if message.recipient_id:
                # Direct message
                if message.recipient_id in self.handlers:
                    recipients.append(message.recipient_id)
                else:
                    # Store for later delivery
                    self.pending_messages[message.recipient_id].append(message)
                    logger.debug(f"Message {message.id} stored for offline agent {message.recipient_id}")
            else:
                # Broadcast message
                recipients = list(self.subscriptions.get(message.message_type, set()))
            
            # Deliver to each recipient
            for recipient_id in recipients:
                if recipient_id == message.sender_id:
                    continue  # Don't deliver to sender
                
                handler = self.handlers.get(recipient_id)
                if handler:
                    try:
                        # Handle message and get optional response
                        response = await handler.handle_message(message)
                        
                        # If handler returns a response, send it
                        if response:
                            await self.send_message(response)
                        
                        logger.debug(f"Message {message.id} delivered to {recipient_id}")
                        
                    except Exception as e:
                        logger.error(f"Error delivering message {message.id} to {recipient_id}: {str(e)}")
            
            self.stats["messages_delivered"] += 1
            
        except Exception as e:
            logger.error(f"Failed to deliver message {message.id}: {str(e)}")
            self.stats["messages_failed"] += 1
    
    async def deliver_pending_messages(self, agent_id: str):
        """Deliver pending messages to an agent that just came online"""
        if agent_id in self.pending_messages:
            pending = self.pending_messages[agent_id]
            self.pending_messages[agent_id] = []
            
            for message in pending:
                if not message.is_expired():
                    await self.delivery_queue.put(message)
            
            logger.info(f"Queued {len(pending)} pending messages for {agent_id}")
    
    def get_message_history(self, 
                           agent_id: Optional[str] = None,
                           message_type: Optional[MessageType] = None,
                           limit: int = 100) -> List[AgentMessage]:
        """Get message history with optional filtering
        
        Args:
            agent_id: Filter by sender or recipient
            message_type: Filter by message type
            limit: Maximum number of messages to return
            
        Returns:
            List of messages matching criteria
        """
        messages = list(self.message_history)
        
        # Apply filters
        if agent_id:
            messages = [
                msg for msg in messages 
                if msg.sender_id == agent_id or msg.recipient_id == agent_id
            ]
        
        if message_type:
            messages = [msg for msg in messages if msg.message_type == message_type]
        
        # Sort by timestamp (newest first) and limit
        messages.sort(key=lambda x: x.timestamp, reverse=True)
        return messages[:limit]
    
    def get_conversation(self, correlation_id: str) -> List[AgentMessage]:
        """Get all messages in a conversation by correlation ID
        
        Args:
            correlation_id: Correlation ID to search for
            
        Returns:
            List of messages in the conversation
        """
        messages = self.correlation_map.get(correlation_id, [])
        messages.sort(key=lambda x: x.timestamp)
        return messages
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        return {
            **self.stats,
            "active_handlers": len(self.handlers),
            "total_subscriptions": sum(len(agents) for agents in self.subscriptions.values()),
            "pending_messages": sum(len(msgs) for msgs in self.pending_messages.values()),
            "history_size": len(self.message_history),
            "queue_size": self.delivery_queue.qsize()
        }
    
    def clear_history(self):
        """Clear message history"""
        self.message_history.clear()
        self.correlation_map.clear()
        logger.info("Message history cleared")

# Global message bus instance
_message_bus: Optional[MessageBus] = None

def get_message_bus() -> MessageBus:
    """Get the global message bus instance"""
    global _message_bus
    if _message_bus is None:
        _message_bus = MessageBus()
    return _message_bus

async def initialize_message_bus() -> MessageBus:
    """Initialize and start the global message bus"""
    bus = get_message_bus()
    if not bus.running:
        await bus.start()
    return bus

async def shutdown_message_bus():
    """Shutdown the global message bus"""
    global _message_bus
    if _message_bus and _message_bus.running:
        await _message_bus.stop()
        _message_bus = None