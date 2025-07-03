"""
Core Pydantic Models for the Sentient-Core Multi-Agent RAG System

This file defines the primary data structures that govern the state and workflow
of the entire application. It serves as the single source of truth for models
representing tasks, agents, memory, and the overall system state.
"""

from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
import uuid
from enum import Enum
from pydantic import BaseModel, Field

# --- Core Enums for State Management ---

class TaskStatus(str, Enum):
    """Defines the lifecycle of a task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    WAITING_DEPENDENCY = "waiting_dependency"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REVIEWING = "reviewing"
    DONE = "done"  # State for UI to know result has been seen

class AgentType(str, Enum):
    """Enumerates the specialized roles of agents in the system."""
    ULTRA_ORCHESTRATOR = "ultra_orchestrator"
    MONITORING_AGENT = "monitoring_agent"
    RESEARCH_AGENT = "research_agent"
    ARCHITECT_PLANNER = "architect_planner"
    FRONTEND_DEVELOPER = "frontend_developer"
    BACKEND_DEVELOPER = "backend_developer"
    CODING_AGENT = "coding_agent"
    SPECIALIZED_AGENT = "specialized_agent"

class MemoryLayer(str, Enum):
    """Defines the layers of the long-term memory system."""
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    BUILD_CONVERSATION = "build_conversation"
    CODEBASE_KNOWLEDGE = "codebase_knowledge"
    STACK_DEPENDENCIES = "stack_dependencies"

class SessionState(str, Enum):
    """Represents the overall state of a user's session."""
    ACTIVE = "active"
    CLARIFYING = "clarifying"
    RESEARCHING = "researching"
    PLANNING = "planning"
    BUILDING = "building"
    REVIEWING = "reviewing"
    COMPLETED = "completed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"

# --- Foundational Data Structures ---

class EnhancedTask(BaseModel):
    """
    A robust task model supporting dependencies, sequencing, and detailed tracking.
    This replaces the previous simple 'Task' model.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str
    agent_type: AgentType
    status: TaskStatus = TaskStatus.PENDING
    
    # Execution Flow
    sequence: int = 1
    dependencies: List[str] = Field(default_factory=list)  # Task IDs this depends on
    
    # Results & Artifacts
    result: Optional[str] = None
    artifacts: List[str] = Field(default_factory=list)  # File paths or URLs
    follow_up_questions: List[str] = Field(default_factory=list)

    # Timestamps & Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # For compatibility with old Task agent field
    @property
    def agent(self) -> str:
        return self.agent_type.value

class Message(BaseModel):
    """Represents a single message in the conversation history."""
    sender: str
    content: str
    image: Optional[bytes] = None

class LogEntry(BaseModel):
    """Represents a single log entry for system monitoring and UI display."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: str  # e.g., "UltraOrchestrator", "ResearchGraph", "UI"
    message: str

class ResearchStep(BaseModel):
    """Represents a single step in a research process."""
    query: str
    status: str = "pending"
    result: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class ResearchState(BaseModel):
    """Manages the state of a multi-step research task for the research sub-graph."""
    original_query: str
    steps: List[ResearchStep] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    final_report: Optional[str] = None
    continual_search_suggestions: List[str] = Field(default_factory=list)

# --- Main Application State ---

class AppState(BaseModel):
    """
    The primary state object for the LangGraph workflow.
    This is a comprehensive model that holds the entire state of the application,
    including session info, messages, tasks, logs, and user inputs.
    """
    # Conversation & User Input
    messages: List[Message] = Field(default_factory=list)
    user_prompt: str = ""
    image: Optional[bytes] = None
    language: str = "en"
    
    # Task Management
    tasks: List[EnhancedTask] = Field(default_factory=list)
    task_to_run_id: Optional[str] = None
    
    # System & Workflow
    logs: List[LogEntry] = Field(default_factory=list)
    
    # Used for routing decisions within the graph
    next_action: Optional[str] = None
    
    # Stores the raw JSON decision from the orchestrator for other agents/UI to use
    orchestrator_decision: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True