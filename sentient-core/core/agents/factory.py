"""Agent Factory for Multi-Agent System

This module provides centralized agent creation, configuration, and lifecycle management.
It handles agent instantiation, dependency injection, and initialization workflows.
"""

import asyncio
import logging
from typing import Dict, Any, Type, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import importlib
import inspect

from .base_agent import BaseAgent, AgentCapability
from .registry import AgentRegistry, get_agent_registry
from .communication import MessageBus, get_message_bus
from ..services.state_service import StateService
from ..models import AgentType

logger = logging.getLogger(__name__)

class AgentCreationError(Exception):
    """Exception raised when agent creation fails"""
    pass

class AgentConfigurationError(Exception):
    """Exception raised when agent configuration is invalid"""
    pass

@dataclass
class AgentConfiguration:
    """Configuration for agent creation"""
    agent_type: str
    name: str
    description: str = ""
    capabilities: List[AgentCapability] = None
    config: Dict[str, Any] = None
    dependencies: List[str] = None
    resource_requirements: Dict[str, Any] = None
    auto_start: bool = True
    auto_register: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.config is None:
            self.config = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.resource_requirements is None:
            self.resource_requirements = {}
        if self.metadata is None:
            self.metadata = {}
    
    def validate(self) -> bool:
        """Validate the configuration
        
        Returns:
            True if configuration is valid
            
        Raises:
            AgentConfigurationError: If configuration is invalid
        """
        if not self.agent_type:
            raise AgentConfigurationError("agent_type is required")
        
        if not self.name:
            raise AgentConfigurationError("name is required")
        
        # Validate capabilities
        for cap in self.capabilities:
            if not isinstance(cap, AgentCapability):
                raise AgentConfigurationError(f"Invalid capability: {cap}")
        
        # Validate resource requirements
        if self.resource_requirements:
            valid_resources = {'memory', 'cpu', 'storage', 'network'}
            for resource in self.resource_requirements:
                if resource not in valid_resources:
                    logger.warning(f"Unknown resource requirement: {resource}")
        
        return True

class AgentFactory:
    """Factory for creating and managing agent instances
    
    Provides:
    - Agent type registration and discovery
    - Agent instantiation with dependency injection
    - Configuration validation and management
    - Lifecycle management integration
    - Resource allocation and monitoring
    """
    
    def __init__(self, 
                 registry: Optional[AgentRegistry] = None,
                 message_bus: Optional[MessageBus] = None,
                 state_service: Optional[StateService] = None):
        self.registry = registry or get_agent_registry()
        self.message_bus = message_bus or get_message_bus()
        self.state_service = state_service
        
        # Agent type registry
        self.agent_types: Dict[str, Type[BaseAgent]] = {}
        self.agent_configs: Dict[str, Dict[str, Any]] = {}
        self.creation_hooks: Dict[str, List[Callable]] = {}
        
        # Active agents
        self.active_agents: Dict[str, BaseAgent] = {}
        
        # Statistics
        self.stats = {
            "total_created": 0,
            "creation_failures": 0,
            "active_count": 0,
            "destroyed_count": 0
        }
        
        # Register built-in agent types
        self._register_builtin_agents()
        
        logger.info("AgentFactory initialized")
    
    def _register_builtin_agents(self):
        """Register built-in agent types"""
        try:
            # Import and register built-in agents
            builtin_agents = {
                "ResearchAgent": "research_agent.ResearchAgent",
                "ArchitectPlannerAgent": "architect_planner_agent.ArchitectPlannerAgent",
                "FrontendDeveloperAgent": "frontend_developer_agent.FrontendDeveloperAgent",
                "BackendDeveloperAgent": "backend_developer_agent.BackendDeveloperAgent",
                "CodingAgent": "coding_agent.CodingAgent",
                "MonitoringAgent": "monitoring_agent.MonitoringAgent",
                "UltraOrchestrator": "ultra_orchestrator.UltraOrchestrator"
            }
            
            for agent_name, module_path in builtin_agents.items():
                try:
                    module_name, class_name = module_path.rsplit('.', 1)
                    module = importlib.import_module(f"core.agents.{module_name}")
                    agent_class = getattr(module, class_name)
                    
                    if issubclass(agent_class, BaseAgent):
                        self.register_agent_type(agent_name, agent_class)
                        logger.debug(f"Registered built-in agent: {agent_name}")
                    else:
                        logger.warning(f"Class {class_name} is not a BaseAgent subclass")
                        
                except (ImportError, AttributeError) as e:
                    logger.warning(f"Could not register built-in agent {agent_name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error registering built-in agents: {str(e)}")
    
    def register_agent_type(self, 
                           agent_type: str, 
                           agent_class: Type[BaseAgent],
                           default_config: Dict[str, Any] = None):
        """Register an agent type with the factory
        
        Args:
            agent_type: Unique identifier for the agent type
            agent_class: Agent class to register
            default_config: Default configuration for this agent type
        """
        if not issubclass(agent_class, BaseAgent):
            raise AgentCreationError(f"Agent class must inherit from BaseAgent")
        
        self.agent_types[agent_type] = agent_class
        self.agent_configs[agent_type] = default_config or {}
        self.creation_hooks[agent_type] = []
        
        logger.info(f"Registered agent type: {agent_type}")
    
    def unregister_agent_type(self, agent_type: str):
        """Unregister an agent type
        
        Args:
            agent_type: Agent type to unregister
        """
        if agent_type in self.agent_types:
            del self.agent_types[agent_type]
            del self.agent_configs[agent_type]
            del self.creation_hooks[agent_type]
            logger.info(f"Unregistered agent type: {agent_type}")
    
    def get_registered_types(self) -> List[str]:
        """Get list of registered agent types
        
        Returns:
            List of registered agent type names
        """
        return list(self.agent_types.keys())
    
    def add_creation_hook(self, agent_type: str, hook: Callable):
        """Add a creation hook for an agent type
        
        Args:
            agent_type: Agent type to add hook for
            hook: Function to call after agent creation
        """
        if agent_type in self.creation_hooks:
            self.creation_hooks[agent_type].append(hook)
    
    async def create_agent(self, config: AgentConfiguration) -> BaseAgent:
        """Create an agent instance
        
        Args:
            config: Agent configuration
            
        Returns:
            Created agent instance
            
        Raises:
            AgentCreationError: If agent creation fails
        """
        try:
            # Validate configuration
            config.validate()
            
            # Check if agent type is registered
            if config.agent_type not in self.agent_types:
                raise AgentCreationError(f"Unknown agent type: {config.agent_type}")
            
            agent_class = self.agent_types[config.agent_type]
            
            # Merge default config with provided config
            merged_config = {**self.agent_configs[config.agent_type], **config.config}
            
            # Prepare constructor arguments
            constructor_args = self._prepare_constructor_args(
                agent_class, config, merged_config
            )
            
            # Create agent instance
            agent = agent_class(**constructor_args)
            
            # Set additional properties
            agent.name = config.name
            agent.description = config.description
            agent.capabilities = config.capabilities.copy()
            
            # Initialize agent
            if config.auto_start:
                await agent.initialize()
                await agent.start()
            
            # Register with registry
            if config.auto_register and self.registry:
                await self.registry.register_agent(
                    agent, 
                    config.agent_type, 
                    config.metadata
                )
            
            # Store in active agents
            self.active_agents[agent.agent_id] = agent
            
            # Run creation hooks
            for hook in self.creation_hooks.get(config.agent_type, []):
                try:
                    await hook(agent, config)
                except Exception as e:
                    logger.error(f"Error in creation hook: {str(e)}")
            
            # Update statistics
            self.stats["total_created"] += 1
            self.stats["active_count"] += 1
            
            logger.info(f"Created agent {agent.agent_id} of type {config.agent_type}")
            return agent
            
        except Exception as e:
            self.stats["creation_failures"] += 1
            logger.error(f"Failed to create agent: {str(e)}")
            raise AgentCreationError(f"Agent creation failed: {str(e)}") from e
    
    def _prepare_constructor_args(self, 
                                 agent_class: Type[BaseAgent], 
                                 config: AgentConfiguration,
                                 merged_config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare constructor arguments for agent creation
        
        Args:
            agent_class: Agent class to create
            config: Agent configuration
            merged_config: Merged configuration
            
        Returns:
            Dictionary of constructor arguments
        """
        # Get constructor signature
        sig = inspect.signature(agent_class.__init__)
        args = {}
        
        # Map common parameters
        param_mapping = {
            'name': config.name,
            'description': config.description,
            'capabilities': config.capabilities,
            'config': merged_config,
            'message_bus': self.message_bus,
            'state_service': self.state_service,
            'registry': self.registry
        }
        
        # Add parameters that exist in constructor
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            if param_name in param_mapping:
                args[param_name] = param_mapping[param_name]
            elif param_name in merged_config:
                args[param_name] = merged_config[param_name]
            elif param.default == inspect.Parameter.empty:
                # Required parameter not provided
                logger.warning(f"Required parameter {param_name} not provided for {agent_class.__name__}")
        
        return args
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Destroy an agent instance
        
        Args:
            agent_id: ID of agent to destroy
            
        Returns:
            True if agent was destroyed successfully
        """
        try:
            if agent_id not in self.active_agents:
                logger.warning(f"Agent {agent_id} not found in active agents")
                return False
            
            agent = self.active_agents[agent_id]
            
            # Stop agent
            await agent.stop()
            
            # Deregister from registry
            if self.registry:
                await self.registry.deregister_agent(agent_id)
            
            # Remove from active agents
            del self.active_agents[agent_id]
            
            # Update statistics
            self.stats["active_count"] -= 1
            self.stats["destroyed_count"] += 1
            
            logger.info(f"Destroyed agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to destroy agent {agent_id}: {str(e)}")
            return False
    
    async def restart_agent(self, agent_id: str) -> bool:
        """Restart an agent instance
        
        Args:
            agent_id: ID of agent to restart
            
        Returns:
            True if agent was restarted successfully
        """
        try:
            if agent_id not in self.active_agents:
                logger.warning(f"Agent {agent_id} not found in active agents")
                return False
            
            agent = self.active_agents[agent_id]
            
            # Stop and start agent
            await agent.stop()
            await agent.start()
            
            logger.info(f"Restarted agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {str(e)}")
            return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an active agent by ID
        
        Args:
            agent_id: Agent ID to lookup
            
        Returns:
            Agent instance if found, None otherwise
        """
        return self.active_agents.get(agent_id)
    
    def get_active_agents(self) -> List[BaseAgent]:
        """Get all active agents
        
        Returns:
            List of active agent instances
        """
        return list(self.active_agents.values())
    
    def get_agents_by_type(self, agent_type: str) -> List[BaseAgent]:
        """Get active agents by type
        
        Args:
            agent_type: Agent type to filter by
            
        Returns:
            List of agents of the specified type
        """
        return [agent for agent in self.active_agents.values() 
                if agent.__class__.__name__ == agent_type]
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[BaseAgent]:
        """Get active agents by capability
        
        Args:
            capability: Capability to filter by
            
        Returns:
            List of agents with the specified capability
        """
        return [agent for agent in self.active_agents.values() 
                if capability in agent.capabilities]
    
    async def shutdown_all_agents(self):
        """Shutdown all active agents"""
        logger.info("Shutting down all active agents")
        
        # Create list of agent IDs to avoid modification during iteration
        agent_ids = list(self.active_agents.keys())
        
        # Shutdown agents in parallel
        tasks = [self.destroy_agent(agent_id) for agent_id in agent_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("All agents shut down")
    
    def get_factory_stats(self) -> Dict[str, Any]:
        """Get factory statistics
        
        Returns:
            Dictionary containing factory statistics
        """
        type_counts = {}
        for agent in self.active_agents.values():
            agent_type = agent.__class__.__name__
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        capability_counts = {}
        for agent in self.active_agents.values():
            for capability in agent.capabilities:
                cap_name = capability.value
                capability_counts[cap_name] = capability_counts.get(cap_name, 0) + 1
        
        return {
            **self.stats,
            "registered_types": len(self.agent_types),
            "type_distribution": type_counts,
            "capability_distribution": capability_counts,
            "registered_type_names": list(self.agent_types.keys())
        }

# Global factory instance
_agent_factory: Optional[AgentFactory] = None

def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance"""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
    return _agent_factory

async def initialize_agent_factory(
    registry: Optional[AgentRegistry] = None,
    message_bus: Optional[MessageBus] = None,
    state_service: Optional[StateService] = None
) -> AgentFactory:
    """Initialize the global agent factory"""
    factory = get_agent_factory()
    if registry:
        factory.registry = registry
    if message_bus:
        factory.message_bus = message_bus
    if state_service:
        factory.state_service = state_service
    return factory

async def shutdown_agent_factory():
    """Shutdown the global agent factory"""
    global _agent_factory
    if _agent_factory:
        await _agent_factory.shutdown_all_agents()
        _agent_factory = None

# Convenience functions for common agent creation patterns

async def create_research_agent(name: str, 
                               config: Dict[str, Any] = None) -> BaseAgent:
    """Create a research agent with default configuration"""
    factory = get_agent_factory()
    agent_config = AgentConfiguration(
        agent_type="ResearchAgent",
        name=name,
        description="Research and knowledge gathering agent",
        capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
        config=config or {}
    )
    return await factory.create_agent(agent_config)

async def create_coding_agent(name: str, 
                             config: Dict[str, Any] = None) -> BaseAgent:
    """Create a coding agent with default configuration"""
    factory = get_agent_factory()
    agent_config = AgentConfiguration(
        agent_type="CodingAgent",
        name=name,
        description="Code generation and development agent",
        capabilities=[AgentCapability.CODE_GENERATION, AgentCapability.TESTING],
        config=config or {}
    )
    return await factory.create_agent(agent_config)

async def create_monitoring_agent(name: str, 
                                 config: Dict[str, Any] = None) -> BaseAgent:
    """Create a monitoring agent with default configuration"""
    factory = get_agent_factory()
    agent_config = AgentConfiguration(
        agent_type="MonitoringAgent",
        name=name,
        description="System monitoring and health tracking agent",
        capabilities=[AgentCapability.MONITORING, AgentCapability.ANALYSIS],
        config=config or {}
    )
    return await factory.create_agent(agent_config)