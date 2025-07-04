"""Workflow Orchestration Module

Provides orchestration for different workflow modes including intelligent RAG,
multi-agent RAG, and legacy orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

from ..agents.integration import AgentSystemIntegration, get_agent_system
from ..services.state_service import StateService
from ..services.enhanced_llm_service import EnhancedLLMService
from ..models import AppState, Message, EnhancedTask, TaskStatus, LogEntry

logger = logging.getLogger(__name__)

class WorkflowMode(Enum):
    """Available workflow modes"""
    INTELLIGENT = "intelligent"
    MULTI_AGENT = "multi_agent"
    LEGACY = "legacy"

class ResearchMode(Enum):
    """Available research modes"""
    KNOWLEDGE = "knowledge"
    DEEP = "deep"
    BEST_IN_CLASS = "best-in-class"

class WorkflowOrchestrator:
    """Main workflow orchestrator for handling different execution modes"""
    
    def __init__(self, 
                 state_service: StateService,
                 llm_service: EnhancedLLMService):
        self.state_service = state_service
        self.llm_service = llm_service
        self.agent_system: Optional[AgentSystemIntegration] = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the workflow orchestrator
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing workflow orchestrator...")
            
            # Initialize agent system
            from ..agents.integration import initialize_agent_system
            self.agent_system = await initialize_agent_system(
                self.state_service, 
                self.llm_service
            )
            
            self.initialized = True
            logger.info("Workflow orchestrator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow orchestrator: {e}")
            return False
    
    async def execute_workflow(self, 
                             user_input: str,
                             app_state: AppState,
                             workflow_mode: str = "intelligent",
                             research_mode: str = "knowledge",
                             image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Execute a workflow based on the specified mode
        
        Args:
            user_input: User's input text
            app_state: Current application state
            workflow_mode: Workflow mode (intelligent, multi_agent, legacy)
            research_mode: Research mode (knowledge, deep, best-in-class)
            image_data: Optional image data
            
        Returns:
            Dictionary with updated messages, tasks, and logs
        """
        if not self.initialized:
            logger.error("Workflow orchestrator not initialized")
            return self._create_error_response("System not initialized")
        
        try:
            logger.info(f"Executing {workflow_mode} workflow with {research_mode} research mode")
            
            # Route to appropriate workflow handler
            if workflow_mode == WorkflowMode.INTELLIGENT.value:
                return await self._execute_intelligent_workflow(
                    user_input, app_state, research_mode, image_data
                )
            elif workflow_mode == WorkflowMode.MULTI_AGENT.value:
                return await self._execute_multi_agent_workflow(
                    user_input, app_state, research_mode, image_data
                )
            elif workflow_mode == WorkflowMode.LEGACY.value:
                return await self._execute_legacy_workflow(
                    user_input, app_state, research_mode, image_data
                )
            else:
                logger.error(f"Unknown workflow mode: {workflow_mode}")
                return self._create_error_response(f"Unknown workflow mode: {workflow_mode}")
                
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return self._create_error_response(f"Workflow execution error: {str(e)}")
    
    async def _execute_intelligent_workflow(self, 
                                          user_input: str,
                                          app_state: AppState,
                                          research_mode: str,
                                          image_data: Optional[bytes]) -> Dict[str, Any]:
        """Execute intelligent RAG workflow
        
        This mode uses advanced reasoning and context-aware processing
        with the Ultra Orchestrator for optimal results.
        """
        logger.info("Executing intelligent RAG workflow")
        
        try:
            # Enhanced context preparation for intelligent mode
            context = {
                "user_input": user_input,
                "workflow_mode": "intelligent",
                "research_mode": research_mode,
                "image_data": image_data,
                "conversation_history": [msg.dict() for msg in app_state.messages],
                "current_tasks": [task.dict() for task in app_state.tasks],
                "logs": [log.dict() for log in app_state.logs],
                "intelligent_features": {
                    "context_awareness": True,
                    "multi_step_reasoning": True,
                    "adaptive_planning": True,
                    "quality_optimization": True
                }
            }
            
            # Process through agent system with intelligent enhancements
            result = await self.agent_system.process_user_request(
                user_input, app_state, "intelligent", image_data
            )
            
            # Add intelligent workflow specific enhancements
            if result.get("messages"):
                for message in result["messages"]:
                    if message.get("sender") == "assistant":
                        message["content"] = self._enhance_response_quality(
                            message["content"], research_mode
                        )
            
            # Add intelligent workflow log
            result["logs"].append({
                "source": "Intelligent Workflow",
                "message": f"Processed with {research_mode} research mode using advanced reasoning",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in intelligent workflow: {e}")
            return self._create_error_response(f"Intelligent workflow error: {str(e)}")
    
    async def _execute_multi_agent_workflow(self, 
                                          user_input: str,
                                          app_state: AppState,
                                          research_mode: str,
                                          image_data: Optional[bytes]) -> Dict[str, Any]:
        """Execute multi-agent RAG workflow
        
        This mode coordinates multiple specialized agents for comprehensive
        task handling and collaborative problem-solving.
        """
        logger.info("Executing multi-agent RAG workflow")
        
        try:
            # Multi-agent context preparation
            context = {
                "user_input": user_input,
                "workflow_mode": "multi_agent",
                "research_mode": research_mode,
                "image_data": image_data,
                "conversation_history": [msg.dict() for msg in app_state.messages],
                "current_tasks": [task.dict() for task in app_state.tasks],
                "logs": [log.dict() for log in app_state.logs],
                "multi_agent_features": {
                    "collaborative_processing": True,
                    "specialized_agents": True,
                    "parallel_execution": True,
                    "cross_agent_communication": True
                }
            }
            
            # Process through agent system with multi-agent coordination
            result = await self.agent_system.process_user_request(
                user_input, app_state, "multi_agent", image_data
            )
            
            # Enhance with multi-agent specific features
            if result.get("tasks"):
                # Distribute tasks across different agents
                result["tasks"] = self._distribute_tasks_to_agents(
                    result["tasks"], research_mode
                )
            
            # Add multi-agent workflow log
            result["logs"].append({
                "source": "Multi-Agent Workflow",
                "message": f"Coordinated multiple agents with {research_mode} research mode",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-agent workflow: {e}")
            return self._create_error_response(f"Multi-agent workflow error: {str(e)}")
    
    async def _execute_legacy_workflow(self, 
                                     user_input: str,
                                     app_state: AppState,
                                     research_mode: str,
                                     image_data: Optional[bytes]) -> Dict[str, Any]:
        """Execute legacy orchestration workflow
        
        This mode provides backward compatibility and simpler processing
        for basic use cases.
        """
        logger.info("Executing legacy orchestration workflow")
        
        try:
            # Legacy context preparation (simplified)
            context = {
                "user_input": user_input,
                "workflow_mode": "legacy",
                "research_mode": research_mode,
                "image_data": image_data,
                "conversation_history": [msg.dict() for msg in app_state.messages[-5:]],  # Limited history
                "legacy_features": {
                    "simple_processing": True,
                    "backward_compatibility": True,
                    "minimal_overhead": True
                }
            }
            
            # Use simplified processing through LLM service directly
            response = await self._process_legacy_request(user_input, context)
            
            result = {
                "messages": [{
                    "sender": "assistant",
                    "content": response,
                    "image": None
                }],
                "tasks": [],
                "logs": [{
                    "source": "Legacy Workflow",
                    "message": f"Processed with {research_mode} research mode using legacy orchestration",
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in legacy workflow: {e}")
            return self._create_error_response(f"Legacy workflow error: {str(e)}")
    
    async def _process_legacy_request(self, user_input: str, context: Dict[str, Any]) -> str:
        """Process request using legacy method (direct LLM)"""
        try:
            # Prepare prompt for legacy processing
            prompt = f"""
User Request: {user_input}

Research Mode: {context.get('research_mode', 'knowledge')}

Please provide a helpful response based on the user's request.
Keep the response concise and focused.
"""
            
            # Add conversation history if available
            if context.get('conversation_history'):
                history_text = "\n".join([
                    f"{msg['sender']}: {msg['content']}"
                    for msg in context['conversation_history'][-3:]  # Last 3 messages
                ])
                prompt += f"\n\nRecent Conversation:\n{history_text}"
            
            # Generate response using LLM service
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in legacy processing: {e}")
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def _enhance_response_quality(self, response: str, research_mode: str) -> str:
        """Enhance response quality for intelligent mode"""
        # Add research mode specific enhancements
        if research_mode == ResearchMode.DEEP.value:
            prefix = "🔍 **Deep Research Analysis:** "
        elif research_mode == ResearchMode.BEST_IN_CLASS.value:
            prefix = "⭐ **Best-in-Class Solution:** "
        else:
            prefix = "💡 **Knowledge-Based Response:** "
        
        return f"{prefix}{response}"
    
    def _distribute_tasks_to_agents(self, tasks: List[Dict[str, Any]], research_mode: str) -> List[Dict[str, Any]]:
        """Distribute tasks to appropriate agents based on task type"""
        agent_mapping = {
            "research": ["research", "analysis", "investigation", "study"],
            "architect": ["design", "architecture", "planning", "structure"],
            "coding": ["code", "implement", "develop", "program", "build"],
            "monitoring": ["monitor", "track", "observe", "analyze"]
        }
        
        for task in tasks:
            description_lower = task.get("description", "").lower()
            
            # Determine best agent based on task description
            best_agent = "research"  # Default
            max_matches = 0
            
            for agent, keywords in agent_mapping.items():
                matches = sum(1 for keyword in keywords if keyword in description_lower)
                if matches > max_matches:
                    max_matches = matches
                    best_agent = agent
            
            task["agent"] = best_agent
            
            # Add research mode context
            task["research_mode"] = research_mode
        
        return tasks
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        return {
            "messages": [{
                "sender": "assistant",
                "content": f"I apologize, but I encountered an error: {error_message}",
                "image": None
            }],
            "tasks": [],
            "logs": [{
                "source": "Workflow Orchestrator",
                "message": f"Error: {error_message}",
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
    
    async def execute_task(self,
                          task_id: str,
                          task_description: str,
                          agent_source: str = "orchestrator") -> Dict[str, Any]:
        """Execute a specific task through the appropriate agent
        
        Args:
            task_id: Unique identifier for the task
            task_description: Description of the task to execute
            agent_source: Source agent that should handle the task
            
        Returns:
            Dictionary with success status, result, and any errors
        """
        if not self.initialized:
            logger.error("Workflow orchestrator not initialized")
            return {"success": False, "error": "System not initialized"}
        
        try:
            logger.info(f"Executing task {task_id}: {task_description}")
            
            # Route task to appropriate agent based on agent_source
            if agent_source in ["research", "researcher"]:
                result = await self.agent_system.execute_agent_task(
                    "research", task_description, {"task_id": task_id}
                )
            elif agent_source in ["architect", "planner"]:
                result = await self.agent_system.execute_agent_task(
                    "architect", task_description, {"task_id": task_id}
                )
            elif agent_source in ["coding", "developer"]:
                result = await self.agent_system.execute_agent_task(
                    "coding", task_description, {"task_id": task_id}
                )
            elif agent_source in ["monitoring", "monitor"]:
                result = await self.agent_system.execute_agent_task(
                    "monitoring", task_description, {"task_id": task_id}
                )
            else:
                # Default to orchestrator for unknown agent sources
                result = await self.agent_system.execute_agent_task(
                    "orchestrator", task_description, {"task_id": task_id}
                )
            
            if result and result.get("success", False):
                return {
                    "success": True,
                    "result": result.get("result", "Task completed successfully"),
                    "agent_source": agent_source
                }
            else:
                error_msg = result.get("error", "Task execution failed") if result else "No result returned"
                return {
                    "success": False,
                    "error": error_msg,
                    "agent_source": agent_source
                }
                
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            return {
                "success": False,
                "error": f"Task execution error: {str(e)}",
                "agent_source": agent_source
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status
        
        Returns:
            Dictionary with system status information
        """
        status = {
            "orchestrator_initialized": self.initialized,
            "agent_system_status": None
        }
        
        if self.agent_system:
            status["agent_system_status"] = self.agent_system.get_system_status()
        
        return status
    
    async def shutdown(self):
        """Shutdown the workflow orchestrator"""
        try:
            logger.info("Shutting down workflow orchestrator...")
            
            if self.agent_system:
                await self.agent_system.stop()
            
            self.initialized = False
            logger.info("Workflow orchestrator shut down successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down workflow orchestrator: {e}")

# Global instance
_workflow_orchestrator: Optional[WorkflowOrchestrator] = None

def get_workflow_orchestrator(state_service: StateService, 
                            llm_service: EnhancedLLMService) -> WorkflowOrchestrator:
    """Get the global workflow orchestrator instance"""
    global _workflow_orchestrator
    if _workflow_orchestrator is None:
        _workflow_orchestrator = WorkflowOrchestrator(state_service, llm_service)
    return _workflow_orchestrator

async def initialize_workflow_orchestrator(state_service: StateService, 
                                         llm_service: EnhancedLLMService) -> WorkflowOrchestrator:
    """Initialize the global workflow orchestrator
    
    Args:
        state_service: State management service
        llm_service: Enhanced LLM service
        
    Returns:
        Initialized workflow orchestrator instance
    """
    orchestrator = get_workflow_orchestrator(state_service, llm_service)
    if not orchestrator.initialized:
        await orchestrator.initialize()
    return orchestrator

async def shutdown_workflow_orchestrator():
    """Shutdown the global workflow orchestrator"""
    global _workflow_orchestrator
    if _workflow_orchestrator and _workflow_orchestrator.initialized:
        await _workflow_orchestrator.shutdown()
    _workflow_orchestrator = None