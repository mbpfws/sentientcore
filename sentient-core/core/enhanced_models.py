"""
Enhanced Models for Multi-Agent RAG System
Supports 4-layer memory architecture, task dependency management, 
and comprehensive system state tracking.
"""

from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
import uuid
from enum import Enum
from pydantic import BaseModel, Field

# Enhanced Task Status with more granular states
class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# Agent Types with specialized roles
class AgentType(str, Enum):
    ULTRA_ORCHESTRATOR = "ultra_orchestrator"
    MONITORING_AGENT = "monitoring_agent"
    RESEARCH_AGENT = "research_agent"
    ARCHITECT_PLANNER = "architect_planner"
    FRONTEND_AGENT = "frontend_agent"
    BACKEND_AGENT = "backend_agent"
    CODE_AGENT = "code_agent"
    SPECIALIZED_AGENT = "specialized_agent"

# Research Types for Research Agent
class ResearchType(str, Enum):
    KNOWLEDGE_RESEARCH = "knowledge_research"
    DEEP_RESEARCH = "deep_research"  
    BEST_IN_CLASS = "best_in_class"
    MARKET_ANALYSIS = "market_analysis"
    TECHNICAL_EVALUATION = "technical_evaluation"

# Memory Layer Types
class MemoryLayer(str, Enum):
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    BUILD_CONVERSATION = "build_conversation"
    CODEBASE_KNOWLEDGE = "codebase_knowledge"
    STACK_DEPENDENCIES = "stack_dependencies"

# Session States
class SessionState(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed" 
    TERMINATED = "terminated"
    BANNED = "banned"

class MemoryEntry(BaseModel):
    """Individual memory entry in the 4-layer system"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    layer: MemoryLayer
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)

class ConversationTurn(BaseModel):
    """Individual conversation turn with enhanced metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str  # "user", "ultra_orchestrator", "monitoring_agent", etc.
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    language: str = "en"
    agent_id: Optional[str] = None
    attachments: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SessionContext(BaseModel):
    """Session management and context"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    user_ip: str = "unknown"
    state: SessionState = SessionState.ACTIVE
    created_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)
    total_turns: int = 0
    detected_language: str = "en"
    user_expertise_level: Optional[str] = None  # "beginner", "intermediate", "expert"
    industry_context: Optional[str] = None
    goal_clarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    conversation_turns: List[ConversationTurn] = Field(default_factory=list)
    warning_count: int = 0

class EnhancedTask(BaseModel):
    """Enhanced task model with dependencies and monitoring"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    agent_type: AgentType
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=3, ge=1, le=5)  # 1=highest, 5=lowest
    estimated_duration: int = Field(default=30)  # minutes
    actual_duration: Optional[int] = None
    dependencies: List[str] = Field(default_factory=list)  # Task IDs
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent_id: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    output: Optional[Dict[str, Any]] = None
    error_log: List[str] = Field(default_factory=list)

class TaskDependencyGraph(BaseModel):
    """Manages task dependencies and execution order"""
    tasks: Dict[str, EnhancedTask] = Field(default_factory=dict)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)  # task_id -> [dependency_ids]
    
    def add_task(self, task: EnhancedTask):
        """Add a task to the dependency graph"""
        self.tasks[task.id] = task
        self.dependencies[task.id] = task.dependencies.copy()
    
    def get_next_executable_task(self) -> Optional[str]:
        """Get the next task that can be executed (all dependencies completed)."""
        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check if all dependencies are completed
            dependencies = self.dependencies.get(task_id, [])
            if all(self.tasks.get(dep_id, EnhancedTask(title="", description="", agent_type=AgentType.SPECIALIZED_AGENT)).status == TaskStatus.COMPLETED 
                   for dep_id in dependencies):
                return task_id
        
        return None
    
    def mark_task_completed(self, task_id: str):
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].completed_at = datetime.now()

class AgentStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    BLOCKED = "blocked"
    ERROR = "error"
    OFFLINE = "offline"

class AgentInstance(BaseModel):
    """Individual agent state and capabilities"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    total_tasks_completed: int = 0
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GraphState(BaseModel):
    """State for individual graphs and sub-graphs"""
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    graph_type: str  # "orchestrator", "research", "architecture", etc.
    current_node: str
    execution_path: List[str] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["running", "paused", "completed", "error"] = "running"
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SystemState(BaseModel):
    """Comprehensive system state including all agents, tasks, and memory layers."""
    session: SessionContext
    tasks: List[EnhancedTask] = Field(default_factory=list)
    agents: List[AgentInstance] = Field(default_factory=list)
    
    # 4-Layer Memory Architecture
    knowledge_synthesis: List[MemoryEntry] = Field(default_factory=list)
    build_conversation: List[MemoryEntry] = Field(default_factory=list)
    codebase_knowledge: List[MemoryEntry] = Field(default_factory=list)
    stack_dependencies: List[MemoryEntry] = Field(default_factory=list)
    
    # System tracking
    active_graphs: List[str] = Field(default_factory=list)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    warning_log: List[Dict[str, Any]] = Field(default_factory=list)
    
    def get_memory_by_layer(self, layer: MemoryLayer, limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memory entries from a specific layer."""
        if layer == MemoryLayer.KNOWLEDGE_SYNTHESIS:
            return sorted(self.knowledge_synthesis, key=lambda x: x.timestamp, reverse=True)[:limit]
        elif layer == MemoryLayer.BUILD_CONVERSATION:
            return sorted(self.build_conversation, key=lambda x: x.timestamp, reverse=True)[:limit]
        elif layer == MemoryLayer.CODEBASE_KNOWLEDGE:
            return sorted(self.codebase_knowledge, key=lambda x: x.timestamp, reverse=True)[:limit]
        elif layer == MemoryLayer.STACK_DEPENDENCIES:
            return sorted(self.stack_dependencies, key=lambda x: x.timestamp, reverse=True)[:limit]
        else:
            return []
    
    def get_active_tasks(self) -> List[EnhancedTask]:
        """Get all active (non-completed) tasks."""
        return [task for task in self.tasks if task.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]]
    
    def get_tasks_by_agent_type(self, agent_type: AgentType) -> List[EnhancedTask]:
        """Get tasks assigned to a specific agent type."""
        return [task for task in self.tasks if task.agent_type == agent_type]

class MonitoringEvent(BaseModel):
    """Events tracked by the monitoring agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str  # "task_started", "task_completed", "agent_status_change", etc.
    source_agent_id: str
    target_task_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    severity: Literal["info", "warning", "error", "critical"] = "info"

class WorkflowMetrics(BaseModel):
    """Comprehensive workflow performance metrics."""
    session_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration: float = 0.0
    agent_utilization: Dict[str, float] = Field(default_factory=dict)  # agent_id -> utilization %
    conversation_efficiency: float = 0.0  # goals achieved vs turns taken
    user_satisfaction_score: Optional[float] = None
    system_response_time: float = 0.0  # average response time in seconds
    memory_usage_mb: float = 0.0
    errors_count: int = 0
    warnings_count: int = 0

# Enhanced configuration for the system
class SystemConfiguration(BaseModel):
    """System-wide configuration"""
    primary_models: Dict[str, str] = Field(default_factory=lambda: {
        "ultra_orchestrator": "gemini-2.5-flash",
        "monitoring_agent": "llama-3.3-70b-versatile",
        "research_agent": "llama-3.3-70b-versatile",
        "code_agent": "gemini-2.5-flash"
    })
    
    fallback_models: Dict[str, List[str]] = Field(default_factory=lambda: {
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "google": ["gemini-2.5-flash", "gemini-1.5-pro"]
    })
    
    warning_threshold: int = 7
    ban_duration_hours: int = 1
    max_concurrent_tasks: int = 5
    memory_retention_days: int = 30
    search_providers: List[str] = Field(default_factory=lambda: ["tavily", "exa"])
    e2b_config: Dict[str, Any] = Field(default_factory=dict)