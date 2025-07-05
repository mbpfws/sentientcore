"""Agent System Integration Module

Provides integration between the agent system and the main application,
including initialization, workflow orchestration, and state management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .base_agent import BaseAgent, AgentCapability, AgentStatus
from .factory import AgentFactory, AgentConfiguration
from .registry import AgentRegistry
from .communication import MessageBus, MessageType, MessagePriority
from .error_handling import ErrorHandler, ErrorSeverity, ErrorCategory
from .ultra_orchestrator import UltraOrchestrator
from .research_agent import ResearchAgent
from .architect_planner_agent import ArchitectPlannerAgent
from .coding_agent import CodingAgent
from .monitoring_agent import MonitoringAgent

from ..services.state_service import StateService
from ..services.llm_service import EnhancedLLMService
from ..models import AppState, Message, EnhancedTask, TaskStatus, LogEntry

logger = logging.getLogger(__name__)

class AgentSystemIntegration:
    """Main integration class for the agent system"""
    
    def __init__(self):
        self.factory: Optional[AgentFactory] = None
        self.registry: Optional[AgentRegistry] = None
        self.message_bus: Optional[MessageBus] = None
        self.state_service: Optional[StateService] = None
        self.llm_service: Optional[EnhancedLLMService] = None
        self.orchestrator: Optional[UltraOrchestrator] = None
        self.agents: Dict[str, BaseAgent] = {}
        self.initialized = False
        self.running = False
    
    async def initialize(self, 
                        state_service: StateService,
                        llm_service: EnhancedLLMService) -> bool:
        """Initialize the agent system
        
        Args:
            state_service: State management service
            llm_service: Enhanced LLM service
            
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing agent system...")
            
            # Store services
            self.state_service = state_service
            self.llm_service = llm_service
            
            # Initialize core components
            self.message_bus = MessageBus()
            await self.message_bus.start()
            
            self.registry = AgentRegistry(
                state_service=state_service,
                message_bus=self.message_bus
            )
            await self.registry.start()
            
            self.factory = AgentFactory(
                registry=self.registry,
                message_bus=self.message_bus,
                state_service=state_service
            )
            
            # Create and register core agents
            await self._create_core_agents()
            
            self.initialized = True
            logger.info("Agent system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent system: {e}")
            return False
    
    async def _create_core_agents(self):
        """Create and register core agents"""
        try:
            # Create Ultra Orchestrator
            orchestrator_config = AgentConfiguration(
                agent_type="UltraOrchestrator",
                name="Ultra Orchestrator",
                description="Main orchestration agent for workflow management",
                capabilities=[AgentCapability.ORCHESTRATION, AgentCapability.PLANNING],
                auto_start=True,
                auto_register=True
            )
            
            self.orchestrator = await self.factory.create_agent(
                orchestrator_config,
                llm_service=self.llm_service
            )
            self.agents["orchestrator"] = self.orchestrator
            
            # Create Research Agent
            research_config = AgentConfiguration(
                agent_type="ResearchAgent",
                name="Research Agent",
                description="Specialized agent for research and knowledge gathering",
                capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
                auto_start=True,
                auto_register=True
            )
            
            research_agent = await self.factory.create_agent(
                research_config,
                llm_service=self.llm_service
            )
            self.agents["research"] = research_agent
            
            # Create Architect Planner Agent
            architect_config = AgentConfiguration(
                agent_type="ArchitectPlannerAgent",
                name="Architect Planner Agent",
                description="Specialized agent for architecture and planning",
                capabilities=[AgentCapability.PLANNING, AgentCapability.ARCHITECTURE],
                auto_start=True,
                auto_register=True
            )
            
            architect_agent = await self.factory.create_agent(
                architect_config,
                llm_service=self.llm_service
            )
            self.agents["architect"] = architect_agent
            
            # Create Coding Agent
            coding_config = AgentConfiguration(
                agent_type="CodingAgent",
                name="Coding Agent",
                description="Specialized agent for code generation and development",
                capabilities=[AgentCapability.CODING, AgentCapability.TESTING, AgentCapability.DEBUGGING],
                auto_start=True,
                auto_register=True
            )
            
            coding_agent = await self.factory.create_agent(
                coding_config,
                llm_service=self.llm_service
            )
            self.agents["coding"] = coding_agent
            
            # Create Monitoring Agent
            monitoring_config = AgentConfiguration(
                agent_type="MonitoringAgent",
                name="Monitoring Agent",
                description="Specialized agent for system monitoring and analysis",
                capabilities=[AgentCapability.MONITORING, AgentCapability.ANALYSIS],
                auto_start=True,
                auto_register=True
            )
            
            monitoring_agent = await self.factory.create_agent(
                monitoring_config,
                llm_service=self.llm_service
            )
            self.agents["monitoring"] = monitoring_agent
            
            logger.info(f"Created {len(self.agents)} core agents")
            
        except Exception as e:
            logger.error(f"Failed to create core agents: {e}")
            raise
    
    async def start(self) -> bool:
        """Start the agent system
        
        Returns:
            True if started successfully
        """
        if not self.initialized:
            logger.error("Agent system not initialized")
            return False
        
        try:
            # Start all agents
            for agent_id, agent in self.agents.items():
                await agent.start()
                logger.info(f"Started agent: {agent_id}")
            
            self.running = True
            logger.info("Agent system started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent system: {e}")
            return False
    
    async def stop(self):
        """Stop the agent system"""
        try:
            logger.info("Stopping agent system...")
            
            # Stop all agents
            for agent_id, agent in self.agents.items():
                await agent.stop()
                logger.info(f"Stopped agent: {agent_id}")
            
            # Stop core components
            if self.factory:
                await self.factory.shutdown_all_agents()
            
            if self.registry:
                await self.registry.stop()
            
            if self.message_bus:
                await self.message_bus.stop()
            
            self.running = False
            logger.info("Agent system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping agent system: {e}")
    
    async def process_user_request(self, 
                                 user_input: str, 
                                 app_state: AppState,
                                 workflow_mode: str = "intelligent",
                                 image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Process a user request through the agent system
        
        Args:
            user_input: User's input text
            app_state: Current application state
            workflow_mode: Workflow mode (intelligent, multi_agent, legacy)
            image_data: Optional image data
            
        Returns:
            Dictionary with updated messages, tasks, and logs
        """
        if not self.running:
            logger.error("Agent system not running")
            return {"messages": [], "tasks": [], "logs": []}
        
        try:
            logger.info(f"Processing user request in {workflow_mode} mode")
            
            # Create context for the orchestrator
            context = {
                "user_input": user_input,
                "workflow_mode": workflow_mode,
                "image_data": image_data,
                "conversation_history": [msg.dict() for msg in app_state.messages],
                "current_tasks": [task.dict() for task in app_state.tasks],
                "logs": [log.dict() for log in app_state.logs]
            }
            
            # Process through orchestrator
            result = await self.orchestrator.invoke(context)
            
            # Convert result to expected format
            messages = []
            tasks = []
            logs = []
            
            if isinstance(result, dict):
                # Extract orchestrator response
                if "response" in result:
                    messages.append({
                        "sender": "assistant",
                        "content": result["response"],
                        "image": None
                    })
                
                # Extract tasks if any
                if "tasks" in result:
                    for task_data in result["tasks"]:
                        tasks.append({
                            "id": task_data.get("id", f"task_{len(tasks)}"),
                            "description": task_data.get("description", ""),
                            "agent": task_data.get("agent", "unknown"),
                            "status": TaskStatus.PENDING.value,
                            "result": None,
                            "follow_up_questions": task_data.get("follow_up_questions", [])
                        })
                
                # Add processing log
                logs.append({
                    "source": "Orchestrator",
                    "message": f"Processed request in {workflow_mode} mode",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return {
                "messages": messages,
                "tasks": tasks,
                "logs": logs
            }
            
        except Exception as e:
            logger.error(f"Error processing user request: {e}")
            return {
                "messages": [{
                    "sender": "assistant",
                    "content": f"I encountered an error processing your request: {str(e)}",
                    "image": None
                }],
                "tasks": [],
                "logs": [{
                    "source": "System",
                    "message": f"Error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
    
    async def execute_agent_task(self, 
                               agent_name: str, 
                               task_description: str, 
                               context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task through a specific agent
        
        Args:
            agent_name: Name of the agent to execute the task
            task_description: Description of the task to execute
            context: Additional context for the task
            
        Returns:
            Dictionary with success status, result, and any errors
        """
        if not self.running:
            logger.error("Agent system not running")
            return {"success": False, "error": "Agent system not running"}
        
        try:
            # Get the specified agent
            agent = self.agents.get(agent_name)
            if not agent:
                logger.error(f"Agent '{agent_name}' not found")
                return {"success": False, "error": f"Agent '{agent_name}' not found"}
            
            logger.info(f"Executing task through {agent_name}: {task_description}")
            
            # Handle UltraOrchestrator differently (it uses invoke method)
            if agent_name == "orchestrator" and hasattr(agent, 'invoke'):
                # Create a minimal AppState for the orchestrator
                from ..models import AppState, Message
                app_state = AppState(
                    messages=[Message(sender="user", content=task_description)],
                    tasks=[],
                    logs=[]
                )
                result = await agent.invoke(app_state)
                return {
                    "success": True,
                    "result": result,
                    "agent_name": agent_name
                }
            
            # For other agents that inherit from BaseAgent
            if hasattr(agent, 'process_task'):
                # Create an EnhancedTask object
                task_id = context.get("task_id", f"task_{datetime.utcnow().timestamp()}")
                enhanced_task = EnhancedTask(
                    id=task_id,
                    description=task_description,
                    agent=agent_name,
                    status=TaskStatus.PENDING
                )
                
                # Execute the task through the agent
                result = await agent.process_task(enhanced_task)
                
                return {
                    "success": True,
                    "result": result if isinstance(result, str) else str(result),
                    "agent_name": agent_name
                }
            else:
                # Fallback for agents without process_task method
                logger.error(f"Agent '{agent_name}' does not have a process_task method")
                return {
                    "success": False,
                    "error": f"Agent '{agent_name}' does not support task execution",
                    "agent_name": agent_name
                }
            
        except Exception as e:
            logger.error(f"Error executing task through {agent_name}: {e}")
            return {
                "success": False,
                "error": f"Task execution error: {str(e)}",
                "agent_name": agent_name
            }
    
    async def execute_task(self, task_id: str, app_state: AppState) -> Dict[str, Any]:
        """Execute a specific task
        
        Args:
            task_id: ID of the task to execute
            app_state: Current application state
            
        Returns:
            Dictionary with updated task result and logs
        """
        if not self.running:
            logger.error("Agent system not running")
            return {"result": None, "logs": []}
        
        try:
            # Find the task
            task = None
            for t in app_state.tasks:
                if t.id == task_id:
                    task = t
                    break
            
            if not task:
                logger.error(f"Task {task_id} not found")
                return {"result": None, "logs": []}
            
            logger.info(f"Executing task: {task.description}")
            
            # Determine which agent should handle the task
            agent_mapping = {
                "research": self.agents.get("research"),
                "architect": self.agents.get("architect"),
                "coding": self.agents.get("coding"),
                "monitoring": self.agents.get("monitoring")
            }
            
            agent = agent_mapping.get(task.agent)
            if not agent:
                logger.error(f"No agent found for task type: {task.agent}")
                return {"result": None, "logs": []}
            
            # Execute the task
            context = {
                "task_description": task.description,
                "task_id": task.id,
                "conversation_history": [msg.dict() for msg in app_state.messages]
            }
            
            result = await agent.process_task(context)
            
            logs = [{
                "source": agent.name,
                "message": f"Completed task: {task.description}",
                "timestamp": datetime.utcnow().isoformat()
            }]
            
            return {
                "result": result if isinstance(result, str) else str(result),
                "logs": logs
            }
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            return {
                "result": f"Error executing task: {str(e)}",
                "logs": [{
                    "source": "System",
                    "message": f"Task execution error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status
        
        Returns:
            Dictionary with system status information
        """
        return {
            "initialized": self.initialized,
            "running": self.running,
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "status": agent.status.value if agent.status else "unknown",
                    "capabilities": [cap.value for cap in agent.capabilities]
                }
                for agent_id, agent in self.agents.items()
            },
            "registry_stats": self.registry.get_registry_stats() if self.registry else {},
            "factory_stats": self.factory.get_factory_stats() if self.factory else {}
        }

# Global instance
_agent_system: Optional[AgentSystemIntegration] = None

def get_agent_system() -> AgentSystemIntegration:
    """Get the global agent system instance"""
    global _agent_system
    if _agent_system is None:
        _agent_system = AgentSystemIntegration()
    return _agent_system

async def initialize_agent_system(state_service: StateService, 
                                llm_service: EnhancedLLMService) -> AgentSystemIntegration:
    """Initialize the global agent system
    
    Args:
        state_service: State management service
        llm_service: Enhanced LLM service
        
    Returns:
        Initialized agent system instance
    """
    system = get_agent_system()
    if not system.initialized:
        await system.initialize(state_service, llm_service)
        await system.start()
    return system

async def shutdown_agent_system():
    """Shutdown the global agent system"""
    global _agent_system
    if _agent_system and _agent_system.running:
        await _agent_system.stop()
    _agent_system = None