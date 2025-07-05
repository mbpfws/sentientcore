"""Agent Registry for Multi-Agent System

This module provides centralized agent registration, discovery, and lifecycle management.
It maintains a registry of all active agents and their capabilities, enabling dynamic
agent discovery and coordination.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import json
import logging
from collections import defaultdict

from .base_agent import AgentCapability, BaseAgent
from .communication import MessageBus, AgentMessage, MessageType, MessagePriority, get_message_bus
from ..services.state_service import StateService, AgentStatus

logger = logging.getLogger(__name__)

class RegistrationStatus(str, Enum):
    """Agent registration status"""
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

@dataclass
class AgentProfile:
    """Comprehensive agent profile for registry"""
    agent_id: str
    name: str
    description: str
    agent_type: str
    capabilities: List[AgentCapability]
    status: RegistrationStatus
    registration_time: datetime
    last_heartbeat: datetime
    last_activity: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    health_status: str = "unknown"
    version: str = "1.0.0"
    dependencies: List[str] = None
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.resource_requirements is None:
            self.resource_requirements = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        data = asdict(self)
        data['capabilities'] = [cap.value for cap in self.capabilities]
        data['registration_time'] = self.registration_time.isoformat()
        data['last_heartbeat'] = self.last_heartbeat.isoformat()
        if self.last_activity:
            data['last_activity'] = self.last_activity.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        """Create profile from dictionary"""
        data['capabilities'] = [AgentCapability(cap) for cap in data['capabilities']]
        data['registration_time'] = datetime.fromisoformat(data['registration_time'])
        data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        if data.get('last_activity'):
            data['last_activity'] = datetime.fromisoformat(data['last_activity'])
        return cls(**data)
    
    def is_healthy(self, heartbeat_timeout: timedelta = timedelta(minutes=5)) -> bool:
        """Check if agent is considered healthy based on heartbeat"""
        return (datetime.utcnow() - self.last_heartbeat) < heartbeat_timeout
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp"""
        self.last_heartbeat = datetime.utcnow()
    
    def update_activity(self):
        """Update the last activity timestamp"""
        self.last_activity = datetime.utcnow()

class AgentRegistry:
    """Central registry for agent management
    
    Provides:
    - Agent registration and deregistration
    - Agent discovery by capabilities
    - Health monitoring and heartbeat tracking
    - Agent lifecycle management
    - Performance metrics aggregation
    """
    
    def __init__(self, 
                 state_service: Optional[StateService] = None,
                 message_bus: Optional[MessageBus] = None,
                 heartbeat_interval: int = 60,
                 heartbeat_timeout: int = 300):
        self.state_service = state_service
        self.message_bus = message_bus or get_message_bus()
        self.heartbeat_interval = heartbeat_interval  # seconds
        self.heartbeat_timeout = timedelta(seconds=heartbeat_timeout)
        
        # Registry storage
        self.agents: Dict[str, AgentProfile] = {}
        self.capability_index: Dict[AgentCapability, Set[str]] = defaultdict(set)
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Monitoring
        self.health_checks: Dict[str, Callable] = {}
        self.event_listeners: List[Callable] = []
        
        # Background tasks
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        # Statistics
        self.stats = {
            "total_registrations": 0,
            "active_agents": 0,
            "failed_health_checks": 0,
            "heartbeat_timeouts": 0
        }
        
        logger.info("AgentRegistry initialized")
    
    async def start(self):
        """Start the agent registry"""
        self.running = True
        
        # Start background monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("AgentRegistry started")
    
    async def stop(self):
        """Stop the agent registry"""
        self.running = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AgentRegistry stopped")
    
    async def register_agent(self, 
                            agent: BaseAgent,
                            agent_type: str = None,
                            metadata: Dict[str, Any] = None) -> bool:
        """Register an agent with the registry
        
        Args:
            agent: Agent instance to register
            agent_type: Type classification for the agent
            metadata: Additional metadata
            
        Returns:
            True if registration was successful
        """
        try:
            agent_type = agent_type or agent.__class__.__name__
            
            # Create agent profile
            profile = AgentProfile(
                agent_id=agent.agent_id,
                name=agent.name,
                description=agent.description,
                agent_type=agent_type,
                capabilities=agent.capabilities,
                status=RegistrationStatus.ACTIVE,
                registration_time=datetime.utcnow(),
                last_heartbeat=datetime.utcnow(),
                metadata=metadata or {},
                performance_metrics=agent.performance_metrics.copy()
            )
            
            # Store in registry
            self.agents[agent.agent_id] = profile
            
            # Update indexes
            for capability in agent.capabilities:
                self.capability_index[capability].add(agent.agent_id)
            self.type_index[agent_type].add(agent.agent_id)
            
            # Update statistics
            self.stats["total_registrations"] += 1
            self.stats["active_agents"] += 1
            
            # Notify event listeners
            await self._notify_event("agent_registered", {
                "agent_id": agent.agent_id,
                "agent_type": agent_type,
                "capabilities": [cap.value for cap in agent.capabilities]
            })
            
            # Send registration broadcast
            if self.message_bus:
                await self.message_bus.broadcast_message(
                    sender_id="registry",
                    message_type=MessageType.WORKFLOW_EVENT,
                    payload={
                        "event": "agent_registered",
                        "agent_id": agent.agent_id,
                        "agent_type": agent_type,
                        "capabilities": [cap.value for cap in agent.capabilities]
                    },
                    priority=MessagePriority.NORMAL
                )
            
            logger.info(f"Agent {agent.agent_id} ({agent_type}) registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {str(e)}")
            return False
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent from the registry
        
        Args:
            agent_id: ID of agent to deregister
            
        Returns:
            True if deregistration was successful
        """
        try:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found in registry")
                return False
            
            profile = self.agents[agent_id]
            
            # Remove from indexes
            for capability in profile.capabilities:
                self.capability_index[capability].discard(agent_id)
            self.type_index[profile.agent_type].discard(agent_id)
            
            # Update status and remove
            profile.status = RegistrationStatus.TERMINATED
            del self.agents[agent_id]
            
            # Update statistics
            self.stats["active_agents"] -= 1
            
            # Notify event listeners
            await self._notify_event("agent_deregistered", {
                "agent_id": agent_id,
                "agent_type": profile.agent_type
            })
            
            # Send deregistration broadcast
            if self.message_bus:
                await self.message_bus.broadcast_message(
                    sender_id="registry",
                    message_type=MessageType.WORKFLOW_EVENT,
                    payload={
                        "event": "agent_deregistered",
                        "agent_id": agent_id,
                        "agent_type": profile.agent_type
                    },
                    priority=MessagePriority.NORMAL
                )
            
            logger.info(f"Agent {agent_id} deregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister agent {agent_id}: {str(e)}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile by ID
        
        Args:
            agent_id: Agent ID to lookup
            
        Returns:
            Agent profile if found, None otherwise
        """
        return self.agents.get(agent_id)
    
    def find_agents_by_capability(self, capability: AgentCapability) -> List[AgentProfile]:
        """Find agents with a specific capability
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of agent profiles with the capability
        """
        agent_ids = self.capability_index.get(capability, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def find_agents_by_type(self, agent_type: str) -> List[AgentProfile]:
        """Find agents by type
        
        Args:
            agent_type: Agent type to search for
            
        Returns:
            List of agent profiles of the specified type
        """
        agent_ids = self.type_index.get(agent_type, set())
        return [self.agents[agent_id] for agent_id in agent_ids if agent_id in self.agents]
    
    def find_healthy_agents(self, 
                           capability: Optional[AgentCapability] = None,
                           agent_type: Optional[str] = None) -> List[AgentProfile]:
        """Find healthy agents with optional filtering
        
        Args:
            capability: Optional capability filter
            agent_type: Optional type filter
            
        Returns:
            List of healthy agent profiles
        """
        agents = list(self.agents.values())
        
        # Filter by health
        agents = [agent for agent in agents if agent.is_healthy(self.heartbeat_timeout)]
        
        # Filter by capability
        if capability:
            agents = [agent for agent in agents if capability in agent.capabilities]
        
        # Filter by type
        if agent_type:
            agents = [agent for agent in agents if agent.agent_type == agent_type]
        
        return agents
    
    def get_all_agents(self, include_inactive: bool = False) -> List[AgentProfile]:
        """Get all registered agents
        
        Args:
            include_inactive: Whether to include inactive agents
            
        Returns:
            List of agent profiles
        """
        agents = list(self.agents.values())
        
        if not include_inactive:
            agents = [agent for agent in agents if agent.status == RegistrationStatus.ACTIVE]
        
        return agents
    
    async def update_agent_heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if update was successful
        """
        if agent_id in self.agents:
            self.agents[agent_id].update_heartbeat()
            return True
        return False
    
    async def update_agent_activity(self, agent_id: str) -> bool:
        """Update agent activity timestamp
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if update was successful
        """
        if agent_id in self.agents:
            self.agents[agent_id].update_activity()
            return True
        return False
    
    async def update_agent_metrics(self, 
                                  agent_id: str, 
                                  metrics: Dict[str, float]) -> bool:
        """Update agent performance metrics
        
        Args:
            agent_id: Agent ID
            metrics: Performance metrics to update
            
        Returns:
            True if update was successful
        """
        if agent_id in self.agents:
            self.agents[agent_id].performance_metrics.update(metrics)
            return True
        return False
    
    def add_event_listener(self, listener: Callable):
        """Add an event listener for registry events
        
        Args:
            listener: Async function to call on events
        """
        self.event_listeners.append(listener)
    
    def remove_event_listener(self, listener: Callable):
        """Remove an event listener
        
        Args:
            listener: Listener function to remove
        """
        if listener in self.event_listeners:
            self.event_listeners.remove(listener)
    
    async def _notify_event(self, event_type: str, data: Dict[str, Any]):
        """Notify all event listeners of an event"""
        for listener in self.event_listeners:
            try:
                await listener(event_type, data)
            except Exception as e:
                logger.error(f"Error in event listener: {str(e)}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop for health checks"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retrying
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered agents"""
        current_time = datetime.utcnow()
        unhealthy_agents = []
        
        for agent_id, profile in self.agents.items():
            # Check heartbeat timeout
            if (current_time - profile.last_heartbeat) > self.heartbeat_timeout:
                unhealthy_agents.append(agent_id)
                self.stats["heartbeat_timeouts"] += 1
                
                logger.warning(f"Agent {agent_id} heartbeat timeout")
                
                # Update status
                profile.status = RegistrationStatus.INACTIVE
                profile.health_status = "heartbeat_timeout"
                
                # Notify event listeners
                await self._notify_event("agent_unhealthy", {
                    "agent_id": agent_id,
                    "reason": "heartbeat_timeout",
                    "last_heartbeat": profile.last_heartbeat.isoformat()
                })
        
        # Update statistics
        self.stats["active_agents"] = len([a for a in self.agents.values() 
                                          if a.status == RegistrationStatus.ACTIVE])
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics
        
        Returns:
            Dictionary containing registry statistics
        """
        capability_counts = {}
        for capability, agents in self.capability_index.items():
            capability_counts[capability.value] = len(agents)
        
        type_counts = {}
        for agent_type, agents in self.type_index.items():
            type_counts[agent_type] = len(agents)
        
        return {
            **self.stats,
            "total_agents": len(self.agents),
            "capability_distribution": capability_counts,
            "type_distribution": type_counts,
            "healthy_agents": len([a for a in self.agents.values() 
                                  if a.is_healthy(self.heartbeat_timeout)]),
            "registry_uptime": datetime.utcnow().isoformat() if self.running else None
        }
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry data for persistence
        
        Returns:
            Dictionary containing all registry data
        """
        return {
            "agents": {agent_id: profile.to_dict() 
                      for agent_id, profile in self.agents.items()},
            "stats": self.stats,
            "export_time": datetime.utcnow().isoformat()
        }
    
    def import_registry(self, data: Dict[str, Any]):
        """Import registry data from persistence
        
        Args:
            data: Registry data to import
        """
        try:
            # Clear existing data
            self.agents.clear()
            self.capability_index.clear()
            self.type_index.clear()
            
            # Import agents
            for agent_id, profile_data in data.get("agents", {}).items():
                profile = AgentProfile.from_dict(profile_data)
                self.agents[agent_id] = profile
                
                # Rebuild indexes
                for capability in profile.capabilities:
                    self.capability_index[capability].add(agent_id)
                self.type_index[profile.agent_type].add(agent_id)
            
            # Import stats
            if "stats" in data:
                self.stats.update(data["stats"])
            
            logger.info(f"Imported registry with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to import registry data: {str(e)}")

# Global registry instance
_agent_registry: Optional[AgentRegistry] = None

def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance"""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry

async def initialize_agent_registry(state_service: Optional[StateService] = None) -> AgentRegistry:
    """Initialize and start the global agent registry"""
    registry = get_agent_registry()
    if state_service:
        registry.state_service = state_service
    if not registry.running:
        await registry.start()
    return registry

async def shutdown_agent_registry():
    """Shutdown the global agent registry"""
    global _agent_registry
    if _agent_registry and _agent_registry.running:
        await _agent_registry.stop()
        _agent_registry = None