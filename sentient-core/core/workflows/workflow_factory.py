"""Workflow Factory Module

Provides factory functions for creating and configuring workflow engines
with proper integration to the existing system services.
"""

from typing import Optional
import logging

from .advanced_engine import AdvancedWorkflowEngine
from ..services.state_service import StateService
from ..services.memory_service import MemoryService
from ..agents.registry import AgentRegistry

logger = logging.getLogger(__name__)


def create_advanced_workflow_engine(
    state_service: StateService,
    memory_service: MemoryService,
    agent_registry: Optional[AgentRegistry] = None
) -> AdvancedWorkflowEngine:
    """Create and configure an AdvancedWorkflowEngine instance.
    
    Args:
        state_service: State management service
        memory_service: Memory management service
        agent_registry: Agent registry for task execution (optional)
    
    Returns:
        Configured AdvancedWorkflowEngine instance
    """
    # Create agent registry if not provided
    if agent_registry is None:
        agent_registry = _create_default_agent_registry()
    
    # Create workflow engine
    engine = AdvancedWorkflowEngine(
        agent_registry=agent_registry,
        state_service=state_service,
        memory_service=memory_service
    )
    
    logger.info("Advanced workflow engine created successfully")
    return engine


def _create_default_agent_registry() -> AgentRegistry:
    """Create a default agent registry with basic agent types.
    
    Returns:
        Configured AgentRegistry instance
    """
    from ..agents.registry import AgentRegistry
    from ..agents.research_agent import ResearchAgent
    from ..agents.backend_developer_agent import BackendDeveloperAgent
    from ..agents.frontend_developer_agent import FrontendDeveloperAgent
    from ..agents.architect_planner_agent import ArchitectPlannerAgent
    
    registry = AgentRegistry()
    
    # Register default agent types
    registry.register_agent_type("research_agent", ResearchAgent)
    registry.register_agent_type("backend_developer", BackendDeveloperAgent)
    registry.register_agent_type("frontend_developer", FrontendDeveloperAgent)
    registry.register_agent_type("architect_agent", ArchitectPlannerAgent)
    
    logger.info("Default agent registry created with basic agent types")
    return registry


async def initialize_workflow_engine(
    engine: AdvancedWorkflowEngine,
    load_templates: bool = True
) -> None:
    """Initialize a workflow engine with default configuration.
    
    Args:
        engine: AdvancedWorkflowEngine instance to initialize
        load_templates: Whether to load default workflow templates
    """
    try:
        await engine.initialize()
        
        if load_templates:
            await _load_default_templates(engine)
        
        logger.info("Workflow engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize workflow engine: {e}")
        raise


async def _load_default_templates(engine: AdvancedWorkflowEngine) -> None:
    """Load default workflow templates into the engine.
    
    Args:
        engine: AdvancedWorkflowEngine instance
    """
    # Templates are already loaded in the engine's initialize method
    # This function can be extended to load additional templates from files
    pass