from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime
import uuid
from enum import Enum
from pydantic import BaseModel, Field

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DONE = "done"
    FAILED = "failed"

class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    agent: str
    status: str = TaskStatus.PENDING
    result: str | None = None
    follow_up_questions: List[str] | None = None
    # Additional attributes for intelligent orchestration
    sequence: int = 1
    dependencies: List[str] = Field(default_factory=list)
    estimated_duration: str = "medium"  # "short", "medium", "long"
    user_confirmation_required: bool = True

class Message(BaseModel):
    sender: str
    content: str
    image: bytes | None = None

class ResearchStep(BaseModel):
    """Represents a single step in a research plan."""
    query: str
    status: Literal["pending", "completed"] = "pending"
    result: str | None = None

class LogEntry(BaseModel):
    """Represents a single log entry for UI display."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    source: str  # e.g., "Orchestrator", "ResearchGraph", "UI"
    message: str

class ResearchState(BaseModel):
    """Manages the state of a multi-step research task."""
    original_query: str
    steps: List[ResearchStep] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    final_report: str | None = None
    continual_search_suggestions: List[str] = Field(default_factory=list)
    stream_callback: object = Field(default=None, exclude=True)  # Exclude from serialization

class ResearchResult(BaseModel):
    """Represents the final output of the research agent."""
    report: str
    continual_search_suggestions: List[str]

class AppState(BaseModel):
    """
    The state of the application, including all messages, tasks, and user input.
    """
    messages: list[Message] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    logs: List[LogEntry] = Field(default_factory=list)
    user_prompt: str = ""
    image: bytes | None = None
    language: str = "en"
    task_to_run_id: str | None = None 