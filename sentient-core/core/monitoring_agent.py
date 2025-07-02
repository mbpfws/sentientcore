"""
Monitoring Agent for Multi-Agent RAG System
Comprehensive monitoring of all agent activities, task sequencing,
and real-time system state tracking with verbose frontend feedback.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import json
import uuid
from enum import Enum

from core.enhanced_models import (
    SystemState, EnhancedTask, TaskStatus, AgentType, AgentInstance, AgentStatus,
    MonitoringEvent, WorkflowMetrics, MemoryLayer, MemoryEntry, SystemConfiguration
)
from core.services.llm_service import LLMService


class MonitoringLevel(str, Enum):
    MINIMAL = "minimal"
    STANDARD = "standard"
    VERBOSE = "verbose"
    DEBUG = "debug"


class MonitoringAgent:
    """
    Advanced monitoring agent that:
    - Tracks all agent activities in real-time
    - Enforces sequential task execution
    - Provides comprehensive system health monitoring
    - Generates verbose feedback for frontend display
    - Manages workflow coordination and optimization
    """
    
    def __init__(
        self, 
        llm_service: LLMService, 
        system_config: SystemConfiguration,
        monitoring_level: MonitoringLevel = MonitoringLevel.VERBOSE
    ):
        self.llm_service = llm_service
        self.config = system_config
        self.monitoring_level = monitoring_level
        self.agent_id = str(uuid.uuid4())
        
        # Event tracking
        self.events: List[MonitoringEvent] = []
        self.metrics_history: List[WorkflowMetrics] = []
        
        # Real-time callbacks for frontend
        self.frontend_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Sequential execution control
        self.execution_lock = asyncio.Lock()
        self.current_executing_task: Optional[str] = None
        
        # Agent performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = {}
        
    def register_frontend_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register a callback function for real-time frontend updates."""
        self.frontend_callbacks.append(callback)
    
    def emit_to_frontend(self, event_data: Dict[str, Any]):
        """Emit real-time data to all registered frontend callbacks."""
        for callback in self.frontend_callbacks:
            try:
                callback(event_data)
            except Exception as e:
                self._log_event("frontend_callback_error", {"error": str(e)})
    
    async def start_monitoring(self, system_state: SystemState):
        """Initialize monitoring for a session."""
        self._log_event(
            "monitoring_started",
            {
                "session_id": system_state.session.session_id,
                "monitoring_level": self.monitoring_level,
                "agent_id": self.agent_id
            }
        )
        
        # Initialize metrics
        initial_metrics = WorkflowMetrics(
            session_id=system_state.session.session_id
        )
        self.metrics_history.append(initial_metrics)
        
        # Emit initial state to frontend
        self.emit_to_frontend({
            "type": "monitoring_initialized",
            "data": {
                "session_id": system_state.session.session_id,
                "monitoring_agent_id": self.agent_id,
                "level": self.monitoring_level,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def monitor_task_execution(
        self, 
        task: EnhancedTask, 
        system_state: SystemState
    ) -> Dict[str, Any]:
        """
        Monitor and control task execution with strict sequencing.
        Ensures only one task executes at a time.
        """
        async with self.execution_lock:
            return await self._execute_monitored_task(task, system_state)
    
    async def _execute_monitored_task(
        self, 
        task: EnhancedTask, 
        system_state: SystemState
    ) -> Dict[str, Any]:
        """Execute a single task with comprehensive monitoring."""
        
        # Pre-execution validation
        validation_result = await self._validate_task_execution(task, system_state)
        if not validation_result["can_execute"]:
            return validation_result
        
        # Mark task as running
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.current_executing_task = task.id
        
        # Log task start
        self._log_event(
            "task_started",
            {
                "task_id": task.id,
                "task_title": task.title,
                "agent_type": task.agent_type.value,
                "priority": task.priority,
                "estimated_duration": task.estimated_duration
            }
        )
        
        # Emit to frontend
        self.emit_to_frontend({
            "type": "task_started",
            "data": {
                "task": task.model_dump(),
                "system_state": self._get_system_summary(system_state)
            }
        })
        
        try:
            # Execute task based on agent type
            execution_result = await self._execute_task_by_agent_type(task, system_state)
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.actual_duration = int((task.completed_at - task.started_at).total_seconds() / 60)
            task.output = execution_result
            
            # Update agent performance
            self._update_agent_performance(task, True)
            
            # Log successful completion
            self._log_event(
                "task_completed",
                {
                    "task_id": task.id,
                    "task_title": task.title,
                    "actual_duration": task.actual_duration,
                    "success": True
                }
            )
            
            # Emit completion to frontend
            self.emit_to_frontend({
                "type": "task_completed",
                "data": {
                    "task": task.model_dump(),
                    "result": execution_result,
                    "system_state": self._get_system_summary(system_state)
                }
            })
            
            return {
                "success": True,
                "task_id": task.id,
                "result": execution_result,
                "duration": task.actual_duration
            }
            
        except Exception as e:
            # Handle task failure
            task.status = TaskStatus.FAILED
            task.error_log.append(f"Execution failed: {str(e)}")
            
            self._update_agent_performance(task, False)
            
            # Log failure
            self._log_event(
                "task_failed",
                {
                    "task_id": task.id,
                    "task_title": task.title,
                    "error": str(e),
                    "agent_type": task.agent_type.value
                },
                severity="error"
            )
            
            # Emit failure to frontend
            self.emit_to_frontend({
                "type": "task_failed",
                "data": {
                    "task": task.model_dump(),
                    "error": str(e),
                    "system_state": self._get_system_summary(system_state)
                }
            })
            
            return {
                "success": False,
                "task_id": task.id,
                "error": str(e)
            }
        
        finally:
            self.current_executing_task = None
            
            # Update system metrics
            await self._update_system_metrics(system_state)
    
    async def _validate_task_execution(
        self, 
        task: EnhancedTask, 
        system_state: SystemState
    ) -> Dict[str, Any]:
        """Validate if a task can be executed now."""
        
        # Check dependencies
        for dep_id in task.dependencies:
            dep_task = next((t for t in system_state.tasks if t.id == dep_id), None)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return {
                    "can_execute": False,
                    "reason": f"Dependency task {dep_id} not completed",
                    "blocking_task": dep_id
                }
        
        # Check if another task is running
        if self.current_executing_task and self.current_executing_task != task.id:
            return {
                "can_execute": False,
                "reason": "Another task is currently executing",
                "blocking_task": self.current_executing_task
            }
        
        # Check agent availability
        agent_available = await self._check_agent_availability(task.agent_type, system_state)
        if not agent_available:
            return {
                "can_execute": False,
                "reason": f"No available {task.agent_type.value} agent",
                "suggested_wait_time": 60  # seconds
            }
        
        return {"can_execute": True}
    
    async def _execute_task_by_agent_type(
        self, 
        task: EnhancedTask, 
        system_state: SystemState
    ) -> Dict[str, Any]:
        """Execute task based on its assigned agent type."""
        
        # This would integrate with actual agent implementations
        # For now, we'll simulate execution with LLM-based processing
        
        execution_prompt = self._get_task_execution_prompt(task, system_state)
        
        # Select appropriate model for task type
        model = self._select_model_for_task(task.agent_type)
        
        # Execute task with progress tracking
        progress_callback = lambda progress: self._emit_task_progress(task.id, progress)
        
        result = await self._execute_with_progress_tracking(
            execution_prompt, model, task, progress_callback
        )
        
        return result
    
    async def _execute_with_progress_tracking(
        self,
        prompt: str,
        model: str,
        task: EnhancedTask,
        progress_callback: Callable[[float], None]
    ) -> Dict[str, Any]:
        """Execute task with real-time progress updates."""
        
        # Simulate progress updates during execution
        progress_steps = [10, 25, 50, 75, 90, 100]
        
        for i, progress in enumerate(progress_steps):
            # Update progress
            task.progress_percentage = progress
            progress_callback(progress)
            
            # Simulate work time
            await asyncio.sleep(1)  # In real implementation, this would be actual work
            
            if progress == 100:
                # Final execution result
                try:
                    response = self.llm_service.generate_response(
                        model_name=model,
                        prompt=prompt,
                        stream=False
                    )
                    
                    return {
                        "status": "completed",
                        "response": str(response),
                        "model_used": model,
                        "execution_metadata": {
                            "prompt_length": len(prompt),
                            "response_length": len(str(response))
                        }
                    }
                    
                except Exception as e:
                    raise Exception(f"LLM execution failed: {str(e)}")
        
        return {"status": "completed", "response": "Task executed successfully"}
    
    def _emit_task_progress(self, task_id: str, progress: float):
        """Emit task progress to frontend."""
        self.emit_to_frontend({
            "type": "task_progress",
            "data": {
                "task_id": task_id,
                "progress": progress,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    def _get_task_execution_prompt(self, task: EnhancedTask, system_state: SystemState) -> str:
        """Generate execution prompt for a task."""
        
        # Get relevant context from memory layers
        relevant_context = self._get_relevant_context(task, system_state)
        
        prompt = f"""
You are a {task.agent_type.value} executing the following task:

TASK: {task.title}
DESCRIPTION: {task.description}
PRIORITY: {task.priority}/5
ESTIMATED DURATION: {task.estimated_duration} minutes

RELEVANT CONTEXT:
{relevant_context}

TASK CONTEXT:
{json.dumps(task.context, indent=2)}

Please execute this task thoroughly and provide detailed results. Your response should include:
1. Step-by-step execution process
2. Key findings or outputs
3. Any issues encountered
4. Recommendations for next steps
5. Quality assessment of the work completed

Focus on delivering high-quality, actionable results that advance the overall project goals.
"""
        
        return prompt
    
    def _get_relevant_context(self, task: EnhancedTask, system_state: SystemState) -> str:
        """Get relevant context from memory layers for task execution."""
        
        context_parts = []
        
        # Get recent memory from relevant layers
        if task.agent_type == AgentType.RESEARCH_AGENT:
            knowledge_entries = system_state.get_memory_by_layer(MemoryLayer.KNOWLEDGE_SYNTHESIS, 5)
            context_parts.append("KNOWLEDGE SYNTHESIS:")
            for entry in knowledge_entries:
                context_parts.append(f"- {entry.content[:200]}...")
        
        elif task.agent_type in [AgentType.FRONTEND_AGENT, AgentType.BACKEND_AGENT, AgentType.CODE_AGENT]:
            codebase_entries = system_state.get_memory_by_layer(MemoryLayer.CODEBASE_KNOWLEDGE, 5)
            context_parts.append("CODEBASE KNOWLEDGE:")
            for entry in codebase_entries:
                context_parts.append(f"- {entry.content[:200]}...")
                
            stack_entries = system_state.get_memory_by_layer(MemoryLayer.STACK_DEPENDENCIES, 3)
            context_parts.append("STACK DEPENDENCIES:")
            for entry in stack_entries:
                context_parts.append(f"- {entry.content[:200]}...")
        
        # Always include recent conversation context
        build_entries = system_state.get_memory_by_layer(MemoryLayer.BUILD_CONVERSATION, 3)
        context_parts.append("BUILD CONVERSATION:")
        for entry in build_entries:
            context_parts.append(f"- {entry.content[:200]}...")
        
        return "\n".join(context_parts) if context_parts else "No specific context available."
    
    def _select_model_for_task(self, agent_type: AgentType) -> str:
        """Select appropriate model based on agent type."""
        
        model_mapping = {
            AgentType.RESEARCH_AGENT: self.config.primary_models.get("research_agent", "llama-3.3-70b-versatile"),
            AgentType.CODE_AGENT: self.config.primary_models.get("code_agent", "gemini-2.5-flash"),
            AgentType.FRONTEND_AGENT: self.config.primary_models.get("code_agent", "gemini-2.5-flash"),
            AgentType.BACKEND_AGENT: self.config.primary_models.get("code_agent", "gemini-2.5-flash"),
            AgentType.ARCHITECT_PLANNER: self.config.primary_models.get("ultra_orchestrator", "gemini-2.5-flash")
        }
        
        return model_mapping.get(agent_type, "llama-3.3-70b-versatile")
    
    async def _check_agent_availability(
        self, 
        agent_type: AgentType, 
        system_state: SystemState
    ) -> bool:
        """Check if an agent of the specified type is available."""
        
        # Count active tasks for this agent type
        active_tasks = [
            t for t in system_state.tasks 
            if t.agent_type == agent_type and t.status == TaskStatus.RUNNING
        ]
        
        # For now, allow one task per agent type
        return len(active_tasks) == 0
    
    def _update_agent_performance(self, task: EnhancedTask, success: bool):
        """Update performance metrics for the agent type."""
        
        agent_type_str = task.agent_type.value
        
        if agent_type_str not in self.agent_performance:
            self.agent_performance[agent_type_str] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "total_duration": 0,
                "average_duration": 0,
                "success_rate": 1.0
            }
        
        perf = self.agent_performance[agent_type_str]
        perf["total_tasks"] += 1
        
        if success:
            perf["successful_tasks"] += 1
        else:
            perf["failed_tasks"] += 1
        
        if task.actual_duration:
            perf["total_duration"] += task.actual_duration
            perf["average_duration"] = perf["total_duration"] / perf["total_tasks"]
        
        perf["success_rate"] = perf["successful_tasks"] / perf["total_tasks"]
    
    async def _update_system_metrics(self, system_state: SystemState):
        """Update comprehensive system metrics."""
        
        if not self.metrics_history:
            return
        
        current_metrics = self.metrics_history[-1]
        
        # Update task counts
        current_metrics.total_tasks = len(system_state.tasks)
        current_metrics.completed_tasks = len([t for t in system_state.tasks if t.status == TaskStatus.COMPLETED])
        current_metrics.failed_tasks = len([t for t in system_state.tasks if t.status == TaskStatus.FAILED])
        
        # Calculate average task duration
        completed_tasks_with_duration = [
            t for t in system_state.tasks 
            if t.status == TaskStatus.COMPLETED and t.actual_duration
        ]
        if completed_tasks_with_duration:
            total_duration = sum(t.actual_duration for t in completed_tasks_with_duration)
            current_metrics.average_task_duration = total_duration / len(completed_tasks_with_duration)
        
        # Update agent utilization
        for agent_type_str, perf in self.agent_performance.items():
            current_metrics.agent_utilization[agent_type_str] = perf["success_rate"]
        
        # Calculate conversation efficiency
        if system_state.session.total_turns > 0:
            goals_achieved = current_metrics.completed_tasks
            current_metrics.conversation_efficiency = min(goals_achieved / system_state.session.total_turns, 1.0)
        
        # Count errors and warnings
        current_metrics.errors_count = len([e for e in self.events if e.severity == "error"])
        current_metrics.warnings_count = len([e for e in self.events if e.severity == "warning"])
        
        # Emit metrics to frontend
        self.emit_to_frontend({
            "type": "metrics_updated",
            "data": {
                "metrics": current_metrics.model_dump(),
                "agent_performance": self.agent_performance,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    def _get_system_summary(self, system_state: SystemState) -> Dict[str, Any]:
        """Get a summary of current system state for frontend display."""
        
        active_tasks = system_state.get_active_tasks()
        
        return {
            "session_id": system_state.session.session_id,
            "total_tasks": len(system_state.tasks),
            "active_tasks": len(active_tasks),
            "completed_tasks": len([t for t in system_state.tasks if t.status == TaskStatus.COMPLETED]),
            "failed_tasks": len([t for t in system_state.tasks if t.status == TaskStatus.FAILED]),
            "current_executing_task": self.current_executing_task,
            "agent_count": len(system_state.agents),
            "memory_layers": {
                "knowledge_synthesis": len(system_state.knowledge_synthesis),
                "build_conversation": len(system_state.build_conversation),
                "codebase_knowledge": len(system_state.codebase_knowledge),
                "stack_dependencies": len(system_state.stack_dependencies)
            },
            "recent_events": [e.model_dump() for e in self.events[-5:]],
            "system_health": self._assess_system_health(system_state)
        }
    
    def _assess_system_health(self, system_state: SystemState) -> Dict[str, Any]:
        """Assess overall system health and performance."""
        
        # Count recent errors
        recent_errors = len([
            e for e in self.events 
            if e.severity in ["error", "critical"] and
            e.timestamp > datetime.now() - timedelta(minutes=30)
        ])
        
        # Calculate success rate
        total_tasks = len(system_state.tasks)
        failed_tasks = len([t for t in system_state.tasks if t.status == TaskStatus.FAILED])
        success_rate = (total_tasks - failed_tasks) / max(total_tasks, 1)
        
        # Determine health status
        if recent_errors > 5 or success_rate < 0.7:
            health_status = "critical"
        elif recent_errors > 2 or success_rate < 0.9:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "success_rate": success_rate,
            "recent_errors": recent_errors,
            "uptime_minutes": int((datetime.now() - system_state.session.created_at).total_seconds() / 60)
        }
    
    def _log_event(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        severity: str = "info"
    ):
        """Log a monitoring event."""
        
        event = MonitoringEvent(
            event_type=event_type,
            source_agent_id=self.agent_id,
            data=data,
            severity=severity
        )
        
        self.events.append(event)
        
        # Keep only recent events to manage memory
        if len(self.events) > 1000:
            self.events = self.events[-500:]
        
        # Print to console for debugging
        if self.monitoring_level in [MonitoringLevel.VERBOSE, MonitoringLevel.DEBUG]:
            print(f"[MONITOR] {event_type}: {json.dumps(data, indent=2)}")
    
    async def get_monitoring_report(self, system_state: SystemState) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        
        recent_events = [e.model_dump() for e in self.events[-20:]]
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "monitoring_agent_id": self.agent_id,
            "monitoring_level": self.monitoring_level,
            "system_summary": self._get_system_summary(system_state),
            "recent_events": recent_events,
            "current_metrics": current_metrics.model_dump() if current_metrics else None,
            "agent_performance": self.agent_performance,
            "system_health": self._assess_system_health(system_state),
            "recommendations": await self._generate_recommendations(system_state)
        }
    
    async def _generate_recommendations(self, system_state: SystemState) -> List[Dict[str, Any]]:
        """Generate intelligent recommendations for system optimization."""
        
        recommendations = []
        
        # Check for failed tasks
        failed_tasks = [t for t in system_state.tasks if t.status == TaskStatus.FAILED]
        if failed_tasks:
            recommendations.append({
                "type": "failed_tasks",
                "priority": "high",
                "message": f"{len(failed_tasks)} tasks have failed. Consider reviewing error logs and retrying.",
                "action": "review_failed_tasks"
            })
        
        # Check for blocked tasks
        blocked_tasks = [t for t in system_state.tasks if t.status == TaskStatus.BLOCKED]
        if blocked_tasks:
            recommendations.append({
                "type": "blocked_tasks",
                "priority": "medium",
                "message": f"{len(blocked_tasks)} tasks are blocked. Check dependencies.",
                "action": "resolve_dependencies"
            })
        
        # Check conversation efficiency
        if system_state.session.total_turns > 10 and system_state.session.goal_clarity_score < 0.5:
            recommendations.append({
                "type": "conversation_efficiency",
                "priority": "medium",
                "message": "Conversation may need more focus. Consider clarifying goals.",
                "action": "clarify_goals"
            })
        
        # Check agent performance
        for agent_type, perf in self.agent_performance.items():
            if perf["success_rate"] < 0.8:
                recommendations.append({
                    "type": "agent_performance",
                    "priority": "medium",
                    "message": f"{agent_type} agent has low success rate ({perf['success_rate']:.1%})",
                    "action": "optimize_agent_config"
                })
        
        return recommendations
    
    async def handle_session_reset(self):
        """Handle session reset by clearing monitoring state."""
        
        self._log_event("session_reset", {
            "previous_events_count": len(self.events),
            "previous_metrics_count": len(self.metrics_history)
        })
        
        # Clear monitoring data
        self.events.clear()
        self.metrics_history.clear()
        self.agent_performance.clear()
        self.current_executing_task = None
        
        # Notify frontend
        self.emit_to_frontend({
            "type": "session_reset",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "monitoring_agent_id": self.agent_id
            }
        }) 