"""Agent System Module

This module provides the complete agent system including:
- Base agent infrastructure
- Specialized agent implementations
- Agent factory and registry
- Communication and error handling
"""

# Base infrastructure
from .base_agent import BaseAgent, AgentCapability, ActivityType
from .factory import AgentFactory, AgentConfiguration, AgentCreationError, AgentConfigurationError
from .registry import AgentRegistry, AgentProfile, RegistrationStatus
from .communication import MessageBus, AgentMessage, MessageType, MessagePriority
from .error_handling import (
    AgentError, AgentTimeoutError, AgentResourceError,
    ErrorHandler, ErrorSeverity, ErrorCategory, RecoveryStrategy,
    ErrorInfo, RecoveryAction, handle_agent_errors
)

# Specialized agents
from .research_agent import ResearchAgent
from .architect_planner_agent import ArchitectPlannerAgent
from .frontend_developer_agent import FrontendDeveloperAgent
from .backend_developer_agent import BackendDeveloperAgent
from .coding_agent import CodingAgent
from .monitoring_agent import MonitoringAgent
from .ultra_orchestrator import UltraOrchestrator

# Integration module
from .integration import (
    AgentSystemIntegration,
    get_agent_system,
    initialize_agent_system,
    shutdown_agent_system
)

# Convenience functions
def get_agent_factory() -> AgentFactory:
    """Get the default agent factory instance"""
    return AgentFactory()

def get_agent_registry() -> AgentRegistry:
    """Get the default agent registry instance"""
    return AgentRegistry()

__all__ = [
    # Base infrastructure
    'BaseAgent',
    'AgentCapability',
    'ActivityType',
    'AgentFactory',
    'AgentConfiguration',
    'AgentCreationError',
    'AgentConfigurationError',
    'AgentRegistry',
    'AgentProfile',
    'RegistrationStatus',
    'MessageBus',
    'AgentMessage',
    'MessageType',
    'MessagePriority',
    'AgentError',
    'AgentTimeoutError',
    'AgentResourceError',
    'ErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'RecoveryStrategy',
    'ErrorInfo',
    'RecoveryAction',
    'handle_agent_errors',
    
    # Specialized agents
    'ResearchAgent',
    'ArchitectPlannerAgent',
    'FrontendDeveloperAgent',
    'BackendDeveloperAgent',
    'CodingAgent',
    'MonitoringAgent',
    'UltraOrchestrator',
    
    # Integration module
    'AgentSystemIntegration',
    'get_agent_system',
    'initialize_agent_system',
    'shutdown_agent_system',
    
    # Convenience functions
    'get_agent_factory',
    'get_agent_registry',
]