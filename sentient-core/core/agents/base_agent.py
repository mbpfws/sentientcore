"""Base Agent Class for Multi-Agent System

This module provides the foundational BaseAgent class that all agents inherit from,
establishing consistent interfaces for agent lifecycle management, activity tracking,
status reporting, and inter-agent communication.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
import logging

from ..services.state_service import StateService, AgentStatus
from ..models import LogEntry

# Configure logging
logger = logging.getLogger(__name__)

class AgentCapability(str, Enum):
    """Enumeration of agent capabilities"""
    RESEARCH = "research"
    PLANNING = "planning"
    CODING = "coding"
    MONITORING = "monitoring"
    FRONTEND_DEVELOPMENT = "frontend_development"
    BACKEND_DEVELOPMENT = "backend_development"
    ORCHESTRATION = "orchestration"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    ERROR_HANDLING = "error_handling"
    TESTING = "testing"
    DOCUMENTATION = "documentation"

class ActivityType(str, Enum):
    """Types of agent activities for tracking"""
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    STATUS_CHANGE = "status_change"
    ERROR_OCCURRED = "error_occurred"
    COMMUNICATION_SENT = "communication_sent"
    COMMUNICATION_RECEIVED = "communication_received"
    RESOURCE_ACCESS = "resource_access"
    DECISION_MADE = "decision_made"
    CUSTOM = "custom"

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system
    
    Provides common functionality for:
    - Agent lifecycle management
    - Activity tracking and logging
    - Status reporting and updates
    - Performance metrics collection
    - Error handling and recovery
    - Inter-agent communication
    """
    
    def __init__(self, 
                 agent_id: str = None,
                 name: str = None,
                 description: str = None,
                 capabilities: List[AgentCapability] = None):
        """Initialize base agent
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            description: Description of agent's purpose and functionality
            capabilities: List of agent capabilities
        """
        self.agent_id = agent_id or f"{self.__class__.__name__.lower()}_{uuid.uuid4().hex[:8]}"
        self.name = name or self.__class__.__name__
        self.description = description or f"Instance of {self.__class__.__name__}"
        self.capabilities = capabilities or []
        
        # State management
        self.status = AgentStatus.OFFLINE
        self.current_task: Optional[str] = None
        self.state_service: Optional[StateService] = None
        
        # Activity tracking
        self.activity_log: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_response_time": 0.0,
            "success_rate": 1.0,
            "total_runtime": 0.0,
            "last_activity": None
        }
        
        # Communication
        self.message_handlers: Dict[str, Callable] = {}
        self.subscribed_events: List[str] = []
        
        # Error handling
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.recovery_attempts = 0
        
        # Lifecycle timestamps
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.last_activity_at: Optional[datetime] = None
        
        logger.info(f"Agent {self.agent_id} ({self.name}) initialized")
    
    async def initialize(self, state_service: StateService, **kwargs):
        """Initialize agent with required services
        
        Args:
            state_service: State management service instance
            **kwargs: Additional initialization parameters
        """
        self.state_service = state_service
        await self.update_status(AgentStatus.IDLE)
        await self.log_activity(ActivityType.STATUS_CHANGE, "Agent initialized")
        
        # Register with state service
        await self.state_service.update_agent_state(
            self.agent_id,
            status=self.status,
            metadata={
                "name": self.name,
                "description": self.description,
                "capabilities": [cap.value for cap in self.capabilities],
                "created_at": self.created_at.isoformat()
            }
        )
        
        logger.info(f"Agent {self.agent_id} initialized with state service")
    
    async def start(self):
        """Start the agent"""
        self.started_at = datetime.utcnow()
        await self.update_status(AgentStatus.IDLE)
        await self.log_activity(ActivityType.STATUS_CHANGE, "Agent started")
        logger.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent"""
        await self.update_status(AgentStatus.OFFLINE)
        await self.log_activity(ActivityType.STATUS_CHANGE, "Agent stopped")
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def update_status(self, new_status: AgentStatus, details: str = None):
        """Update agent status
        
        Args:
            new_status: New status to set
            details: Optional details about the status change
        """
        old_status = self.status
        self.status = new_status
        self.last_activity_at = datetime.utcnow()
        
        # Update state service if available
        if self.state_service:
            await self.state_service.update_agent_state(
                self.agent_id,
                status=new_status,
                last_activity=self.last_activity_at,
                current_task=self.current_task
            )
        
        # Log status change
        await self.log_activity(
            ActivityType.STATUS_CHANGE,
            f"Status changed from {old_status.value} to {new_status.value}",
            {"old_status": old_status.value, "new_status": new_status.value, "details": details}
        )
        
        logger.debug(f"Agent {self.agent_id} status: {old_status.value} -> {new_status.value}")
    
    async def log_activity(self, 
                          activity_type: ActivityType, 
                          message: str, 
                          metadata: Dict[str, Any] = None,
                          level: str = "info"):
        """Log agent activity
        
        Args:
            activity_type: Type of activity being logged
            message: Activity description
            metadata: Additional activity metadata
            level: Log level (info, warning, error)
        """
        timestamp = datetime.utcnow()
        
        activity_entry = {
            "timestamp": timestamp.isoformat(),
            "agent_id": self.agent_id,
            "activity_type": activity_type.value,
            "message": message,
            "metadata": metadata or {},
            "level": level
        }
        
        # Add to local activity log
        self.activity_log.append(activity_entry)
        
        # Keep only last 1000 activities
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-1000:]
        
        # Update performance metrics
        self.performance_metrics["last_activity"] = timestamp.isoformat()
        
        # Log to system logger
        log_message = f"[{self.agent_id}] {message}"
        if level == "error":
            logger.error(log_message)
        elif level == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    async def handle_error(self, error: Exception, context: str = None):
        """Handle agent errors
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
        """
        self.error_count += 1
        self.last_error = str(error)
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "error_count": self.error_count
        }
        
        await self.log_activity(
            ActivityType.ERROR_OCCURRED,
            f"Error in {context or 'unknown context'}: {str(error)}",
            error_details,
            "error"
        )
        
        # Update status to error if not already
        if self.status != AgentStatus.ERROR:
            await self.update_status(AgentStatus.ERROR, f"Error: {str(error)}")
        
        # Update state service
        if self.state_service:
            await self.state_service.update_agent_state(
                self.agent_id,
                error_info=str(error),
                performance_metrics=self.performance_metrics
            )
        
        logger.error(f"Agent {self.agent_id} error in {context}: {str(error)}")
    
    async def update_performance_metrics(self, metrics: Dict[str, Any]):
        """Update agent performance metrics
        
        Args:
            metrics: Dictionary of metrics to update
        """
        self.performance_metrics.update(metrics)
        
        # Calculate success rate
        total_tasks = self.performance_metrics.get("tasks_completed", 0) + self.performance_metrics.get("tasks_failed", 0)
        if total_tasks > 0:
            self.performance_metrics["success_rate"] = self.performance_metrics.get("tasks_completed", 0) / total_tasks
        
        # Update state service
        if self.state_service:
            await self.state_service.update_agent_state(
                self.agent_id,
                performance_metrics=self.performance_metrics
            )
        
        await self.log_activity(
            ActivityType.TASK_PROGRESS,
            "Performance metrics updated",
            metrics
        )
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "current_task": self.current_task,
            "capabilities": [cap.value for cap in self.capabilities],
            "performance_metrics": self.performance_metrics.copy(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_activity_at": self.last_activity_at.isoformat() if self.last_activity_at else None,
            "activity_count": len(self.activity_log)
        }
    
    def get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent agent activities
        
        Args:
            limit: Maximum number of activities to return
            
        Returns:
            List of recent activity entries
        """
        return self.activity_log[-limit:] if self.activity_log else []
    
    @abstractmethod
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Determine if agent can handle a specific task
        
        Args:
            task: Task specification
            
        Returns:
            True if agent can handle the task, False otherwise
        """
        pass
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task assigned to the agent
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        pass
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with full lifecycle management
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        task_id = task.get("id", f"task_{uuid.uuid4().hex[:8]}")
        self.current_task = task_id
        
        start_time = datetime.utcnow()
        
        try:
            # Update status and log task start
            await self.update_status(AgentStatus.BUSY, f"Starting task {task_id}")
            await self.log_activity(
                ActivityType.TASK_START,
                f"Starting task: {task.get('description', task_id)}",
                {"task_id": task_id, "task_type": task.get('type')}
            )
            
            # Check if agent can handle the task
            if not await self.can_handle_task(task):
                raise ValueError(f"Agent {self.agent_id} cannot handle task type: {task.get('type')}")
            
            # Process the task
            result = await self.process_task(task)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update performance metrics
            await self.update_performance_metrics({
                "tasks_completed": self.performance_metrics.get("tasks_completed", 0) + 1,
                "avg_response_time": (
                    (self.performance_metrics.get("avg_response_time", 0.0) * 
                     (self.performance_metrics.get("tasks_completed", 0))) + execution_time
                ) / (self.performance_metrics.get("tasks_completed", 0) + 1),
                "total_runtime": self.performance_metrics.get("total_runtime", 0.0) + execution_time
            })
            
            # Log task completion
            await self.log_activity(
                ActivityType.TASK_COMPLETE,
                f"Completed task: {task.get('description', task_id)}",
                {
                    "task_id": task_id,
                    "execution_time": execution_time,
                    "result_summary": str(result)[:200] if result else None
                }
            )
            
            # Update status back to idle
            await self.update_status(AgentStatus.IDLE, f"Completed task {task_id}")
            
            return {
                "success": True,
                "task_id": task_id,
                "execution_time": execution_time,
                "result": result
            }
            
        except Exception as e:
            # Handle task failure
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            await self.handle_error(e, f"task execution: {task_id}")
            
            # Update failure metrics
            await self.update_performance_metrics({
                "tasks_failed": self.performance_metrics.get("tasks_failed", 0) + 1
            })
            
            return {
                "success": False,
                "task_id": task_id,
                "execution_time": execution_time,
                "error": str(e)
            }
            
        finally:
            self.current_task = None