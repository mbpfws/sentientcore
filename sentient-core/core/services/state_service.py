"""State Management Service for Multi-Agent System

This service provides comprehensive state management capabilities for agents,
workflows, and conversations with persistence and recovery features.
"""

import asyncio
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

class WorkflowStatus(Enum):
    """Workflow status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentState:
    """Agent state data structure"""
    agent_id: str
    status: AgentStatus
    current_task: Optional[str] = None
    last_activity: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    error_info: Optional[str] = None
    
    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if self.performance_metrics is None:
            self.performance_metrics = {
                "tasks_completed": 0,
                "avg_response_time": 0.0,
                "success_rate": 1.0
            }

@dataclass
class WorkflowState:
    """Workflow state data structure"""
    workflow_id: str
    status: WorkflowStatus
    current_step: Optional[str] = None
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    steps_completed: List[str] = None
    steps_remaining: List[str] = None
    metadata: Dict[str, Any] = None
    error_info: Optional[str] = None
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.utcnow()
        if self.steps_completed is None:
            self.steps_completed = []
        if self.steps_remaining is None:
            self.steps_remaining = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ConversationState:
    """Conversation state data structure"""
    conversation_id: str
    user_id: Optional[str] = None
    context: Dict[str, Any] = None
    active_agents: List[str] = None
    last_interaction: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.active_agents is None:
            self.active_agents = []
        if self.last_interaction is None:
            self.last_interaction = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class StateCheckpoint:
    """State checkpoint data structure"""
    checkpoint_id: str
    timestamp: datetime
    agent_states: Dict[str, AgentState]
    workflow_states: Dict[str, WorkflowState]
    conversation_state: Optional[ConversationState]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class StateService:
    """Enhanced State Management Service
    
    Provides comprehensive state management for agents, workflows, and conversations
    with persistence, recovery, and real-time monitoring capabilities.
    """
    
    def __init__(self, db_path: str = "state_management.db", checkpoint_interval: int = 300):
        """Initialize the State Management Service
        
        Args:
            db_path: Path to SQLite database file
            checkpoint_interval: Automatic checkpoint interval in seconds
        """
        self.db_path = Path(db_path)
        self.checkpoint_interval = checkpoint_interval
        
        # In-memory state storage
        self.agent_states: Dict[str, AgentState] = {}
        self.workflow_states: Dict[str, WorkflowState] = {}
        self.conversation_state: Optional[ConversationState] = None
        
        # Event listeners
        self.state_listeners: Dict[str, List[Callable]] = {
            "agent_update": [],
            "workflow_update": [],
            "conversation_update": []
        }
        
        # Threading and persistence
        self._lock = threading.RLock()
        self._checkpoint_task = None
        self._running = False
        
        # Performance tracking
        self.performance_stats = {
            "state_updates": 0,
            "checkpoints_created": 0,
            "recovery_operations": 0,
            "avg_update_time": 0.0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info(f"StateService initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Agent states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    agent_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_task TEXT,
                    last_activity TEXT,
                    metadata TEXT,
                    performance_metrics TEXT,
                    error_info TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Workflow states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_states (
                    workflow_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_step TEXT,
                    progress REAL DEFAULT 0.0,
                    started_at TEXT,
                    completed_at TEXT,
                    steps_completed TEXT,
                    steps_remaining TEXT,
                    metadata TEXT,
                    error_info TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Conversation states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_states (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    context TEXT,
                    active_agents TEXT,
                    last_interaction TEXT,
                    metadata TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Checkpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    agent_states TEXT,
                    workflow_states TEXT,
                    conversation_state TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database tables initialized successfully")
    
    async def start(self):
        """Start the state service with automatic checkpointing"""
        self._running = True
        
        # Load existing state from database
        await self._load_state_from_db()
        
        # Start automatic checkpointing
        if self.checkpoint_interval > 0:
            self._checkpoint_task = asyncio.create_task(self._auto_checkpoint())
        
        logger.info("StateService started successfully")
    
    async def stop(self):
        """Stop the state service and save final checkpoint"""
        self._running = False
        
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
        
        # Save final checkpoint
        await self.checkpoint_state(f"final_{datetime.utcnow().isoformat()}")
        
        logger.info("StateService stopped successfully")
    
    async def update_agent_state(self, agent_id: str, **updates) -> AgentState:
        """Update individual agent state
        
        Args:
            agent_id: Unique agent identifier
            **updates: State fields to update
            
        Returns:
            Updated AgentState object
        """
        start_time = datetime.utcnow()
        
        with self._lock:
            # Get existing state or create new one
            current_state = self.agent_states.get(agent_id)
            if current_state is None:
                current_state = AgentState(
                    agent_id=agent_id,
                    status=AgentStatus.IDLE
                )
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(current_state, field):
                    if field == 'status' and isinstance(value, str):
                        value = AgentStatus(value)
                    setattr(current_state, field, value)
            
            # Update last activity
            current_state.last_activity = datetime.utcnow()
            
            # Store updated state
            self.agent_states[agent_id] = current_state
        
        # Persist to database
        await self._persist_agent_state(current_state)
        
        # Notify listeners
        await self._notify_listeners("agent_update", agent_id, current_state)
        
        # Update performance stats
        update_time = (datetime.utcnow() - start_time).total_seconds()
        self._update_performance_stats("state_updates", update_time)
        
        logger.debug(f"Agent state updated: {agent_id} -> {current_state.status.value}")
        return current_state
    
    async def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """Get current agent state
        
        Args:
            agent_id: Unique agent identifier
            
        Returns:
            AgentState object or None if not found
        """
        with self._lock:
            return self.agent_states.get(agent_id)
    
    async def get_all_agent_states(self) -> Dict[str, AgentState]:
        """Get all current agent states
        
        Returns:
            Dictionary of agent_id -> AgentState
        """
        with self._lock:
            return self.agent_states.copy()
    
    async def update_workflow_state(self, workflow_id: str, **updates) -> WorkflowState:
        """Update workflow state
        
        Args:
            workflow_id: Unique workflow identifier
            **updates: State fields to update
            
        Returns:
            Updated WorkflowState object
        """
        start_time = datetime.utcnow()
        
        with self._lock:
            # Get existing state or create new one
            current_state = self.workflow_states.get(workflow_id)
            if current_state is None:
                current_state = WorkflowState(
                    workflow_id=workflow_id,
                    status=WorkflowStatus.PENDING
                )
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(current_state, field):
                    if field == 'status' and isinstance(value, str):
                        value = WorkflowStatus(value)
                    setattr(current_state, field, value)
            
            # Auto-complete workflow if progress reaches 100%
            if current_state.progress >= 100.0 and current_state.status == WorkflowStatus.RUNNING:
                current_state.status = WorkflowStatus.COMPLETED
                current_state.completed_at = datetime.utcnow()
            
            # Store updated state
            self.workflow_states[workflow_id] = current_state
        
        # Persist to database
        await self._persist_workflow_state(current_state)
        
        # Notify listeners
        await self._notify_listeners("workflow_update", workflow_id, current_state)
        
        # Update performance stats
        update_time = (datetime.utcnow() - start_time).total_seconds()
        self._update_performance_stats("state_updates", update_time)
        
        logger.debug(f"Workflow state updated: {workflow_id} -> {current_state.status.value}")
        return current_state
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowState]:
        """Get current workflow execution status
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            WorkflowState object or None if not found
        """
        with self._lock:
            return self.workflow_states.get(workflow_id)
    
    async def get_all_workflow_states(self) -> Dict[str, WorkflowState]:
        """Get all current workflow states
        
        Returns:
            Dictionary of workflow_id -> WorkflowState
        """
        with self._lock:
            return self.workflow_states.copy()
    
    async def update_conversation_state(self, conversation_id: str, **updates) -> ConversationState:
        """Update conversation state
        
        Args:
            conversation_id: Unique conversation identifier
            **updates: State fields to update
            
        Returns:
            Updated ConversationState object
        """
        start_time = datetime.utcnow()
        
        with self._lock:
            # Get existing state or create new one
            if self.conversation_state is None or self.conversation_state.conversation_id != conversation_id:
                self.conversation_state = ConversationState(
                    conversation_id=conversation_id
                )
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(self.conversation_state, field):
                    setattr(self.conversation_state, field, value)
            
            # Update last interaction
            self.conversation_state.last_interaction = datetime.utcnow()
        
        # Persist to database
        await self._persist_conversation_state(self.conversation_state)
        
        # Notify listeners
        await self._notify_listeners("conversation_update", conversation_id, self.conversation_state)
        
        # Update performance stats
        update_time = (datetime.utcnow() - start_time).total_seconds()
        self._update_performance_stats("state_updates", update_time)
        
        logger.debug(f"Conversation state updated: {conversation_id}")
        return self.conversation_state
    
    async def get_conversation_state(self) -> Optional[ConversationState]:
        """Get current conversation state
        
        Returns:
            ConversationState object or None if not found
        """
        with self._lock:
            return self.conversation_state
    
    async def checkpoint_state(self, checkpoint_id: str, metadata: Dict[str, Any] = None) -> StateCheckpoint:
        """Create state checkpoint for recovery
        
        Args:
            checkpoint_id: Unique checkpoint identifier
            metadata: Optional checkpoint metadata
            
        Returns:
            StateCheckpoint object
        """
        start_time = datetime.utcnow()
        
        with self._lock:
            checkpoint = StateCheckpoint(
                checkpoint_id=checkpoint_id,
                timestamp=datetime.utcnow(),
                agent_states=self.agent_states.copy(),
                workflow_states=self.workflow_states.copy(),
                conversation_state=self.conversation_state,
                metadata=metadata or {}
            )
        
        # Persist checkpoint to database
        await self._persist_checkpoint(checkpoint)
        
        # Update performance stats
        checkpoint_time = (datetime.utcnow() - start_time).total_seconds()
        self.performance_stats["checkpoints_created"] += 1
        
        logger.info(f"State checkpoint created: {checkpoint_id}")
        return checkpoint
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from checkpoint
        
        Args:
            checkpoint_id: Checkpoint identifier to restore from
            
        Returns:
            True if restoration successful, False otherwise
        """
        try:
            checkpoint = await self._load_checkpoint(checkpoint_id)
            if checkpoint is None:
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            with self._lock:
                self.agent_states = checkpoint.agent_states
                self.workflow_states = checkpoint.workflow_states
                self.conversation_state = checkpoint.conversation_state
            
            self.performance_stats["recovery_operations"] += 1
            logger.info(f"State restored from checkpoint: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint {checkpoint_id}: {e}")
            return False
    
    def add_state_listener(self, event_type: str, callback: Callable):
        """Add event listener for state changes
        
        Args:
            event_type: Type of event ('agent_update', 'workflow_update', 'conversation_update')
            callback: Callback function to execute on event
        """
        if event_type in self.state_listeners:
            self.state_listeners[event_type].append(callback)
            logger.debug(f"State listener added for {event_type}")
    
    def remove_state_listener(self, event_type: str, callback: Callable):
        """Remove event listener
        
        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if event_type in self.state_listeners and callback in self.state_listeners[event_type]:
            self.state_listeners[event_type].remove(callback)
            logger.debug(f"State listener removed for {event_type}")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics
        
        Returns:
            Dictionary containing performance metrics
        """
        with self._lock:
            stats = self.performance_stats.copy()
            stats.update({
                "active_agents": len(self.agent_states),
                "active_workflows": len(self.workflow_states),
                "has_conversation": self.conversation_state is not None,
                "uptime_seconds": (datetime.utcnow() - datetime.utcnow()).total_seconds()  # Placeholder
            })
            return stats
    
    async def cleanup_old_states(self, max_age_hours: int = 24):
        """Clean up old inactive states
        
        Args:
            max_age_hours: Maximum age in hours for keeping inactive states
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        with self._lock:
            # Clean up old agent states
            agents_to_remove = []
            for agent_id, state in self.agent_states.items():
                if (state.last_activity and state.last_activity < cutoff_time and 
                    state.status in [AgentStatus.IDLE, AgentStatus.OFFLINE]):
                    agents_to_remove.append(agent_id)
            
            for agent_id in agents_to_remove:
                del self.agent_states[agent_id]
                cleaned_count += 1
            
            # Clean up completed workflows
            workflows_to_remove = []
            for workflow_id, state in self.workflow_states.items():
                if (state.completed_at and state.completed_at < cutoff_time and 
                    state.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]):
                    workflows_to_remove.append(workflow_id)
            
            for workflow_id in workflows_to_remove:
                del self.workflow_states[workflow_id]
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old states")
    
    # Private helper methods
    
    async def _auto_checkpoint(self):
        """Automatic checkpoint creation task"""
        while self._running:
            try:
                await asyncio.sleep(self.checkpoint_interval)
                if self._running:
                    checkpoint_id = f"auto_{datetime.utcnow().isoformat()}"
                    await self.checkpoint_state(checkpoint_id, {"type": "automatic"})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-checkpoint failed: {e}")
    
    async def _notify_listeners(self, event_type: str, entity_id: str, state: Any):
        """Notify registered event listeners"""
        for callback in self.state_listeners.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(entity_id, state)
                else:
                    callback(entity_id, state)
            except Exception as e:
                logger.error(f"State listener error: {e}")
    
    def _update_performance_stats(self, operation: str, duration: float):
        """Update performance statistics"""
        if operation == "state_updates":
            self.performance_stats["state_updates"] += 1
            # Update rolling average
            current_avg = self.performance_stats["avg_update_time"]
            count = self.performance_stats["state_updates"]
            self.performance_stats["avg_update_time"] = ((current_avg * (count - 1)) + duration) / count
    
    async def _load_state_from_db(self):
        """Load existing state from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Load agent states
                cursor.execute("SELECT * FROM agent_states")
                for row in cursor.fetchall():
                    agent_state = self._row_to_agent_state(row)
                    self.agent_states[agent_state.agent_id] = agent_state
                
                # Load workflow states
                cursor.execute("SELECT * FROM workflow_states")
                for row in cursor.fetchall():
                    workflow_state = self._row_to_workflow_state(row)
                    self.workflow_states[workflow_state.workflow_id] = workflow_state
                
                # Load conversation state (latest)
                cursor.execute("SELECT * FROM conversation_states ORDER BY updated_at DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    self.conversation_state = self._row_to_conversation_state(row)
                
                logger.info(f"Loaded {len(self.agent_states)} agent states, {len(self.workflow_states)} workflow states")
                
        except Exception as e:
            logger.error(f"Failed to load state from database: {e}")
    
    async def _persist_agent_state(self, state: AgentState):
        """Persist agent state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO agent_states 
                    (agent_id, status, current_task, last_activity, metadata, performance_metrics, error_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.agent_id,
                    state.status.value,
                    state.current_task,
                    state.last_activity.isoformat() if state.last_activity else None,
                    json.dumps(state.metadata),
                    json.dumps(state.performance_metrics),
                    state.error_info
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist agent state: {e}")
    
    async def _persist_workflow_state(self, state: WorkflowState):
        """Persist workflow state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO workflow_states 
                    (workflow_id, status, current_step, progress, started_at, completed_at, 
                     steps_completed, steps_remaining, metadata, error_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.workflow_id,
                    state.status.value,
                    state.current_step,
                    state.progress,
                    state.started_at.isoformat() if state.started_at else None,
                    state.completed_at.isoformat() if state.completed_at else None,
                    json.dumps(state.steps_completed),
                    json.dumps(state.steps_remaining),
                    json.dumps(state.metadata),
                    state.error_info
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist workflow state: {e}")
    
    async def _persist_conversation_state(self, state: ConversationState):
        """Persist conversation state to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO conversation_states 
                    (conversation_id, user_id, context, active_agents, last_interaction, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    state.conversation_id,
                    state.user_id,
                    json.dumps(state.context),
                    json.dumps(state.active_agents),
                    state.last_interaction.isoformat() if state.last_interaction else None,
                    json.dumps(state.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist conversation state: {e}")
    
    async def _persist_checkpoint(self, checkpoint: StateCheckpoint):
        """Persist checkpoint to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert states to JSON with proper enum handling
                def convert_state_to_dict(state):
                    """Convert state object to dict with enum values as strings"""
                    state_dict = asdict(state)
                    # Convert enum values to strings
                    if 'status' in state_dict and hasattr(state_dict['status'], 'value'):
                        state_dict['status'] = state_dict['status'].value
                    # Convert datetime objects to ISO format strings
                    for key, value in state_dict.items():
                        if isinstance(value, datetime):
                            state_dict[key] = value.isoformat()
                    return state_dict
                
                agent_states_json = json.dumps({
                    k: convert_state_to_dict(v) for k, v in checkpoint.agent_states.items()
                })
                workflow_states_json = json.dumps({
                    k: convert_state_to_dict(v) for k, v in checkpoint.workflow_states.items()
                })
                
                conversation_state_json = None
                if checkpoint.conversation_state:
                    conv_dict = convert_state_to_dict(checkpoint.conversation_state)
                    conversation_state_json = json.dumps(conv_dict)
                
                cursor.execute("""
                    INSERT INTO state_checkpoints 
                    (checkpoint_id, timestamp, agent_states, workflow_states, conversation_state, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint.checkpoint_id,
                    checkpoint.timestamp.isoformat(),
                    agent_states_json,
                    workflow_states_json,
                    conversation_state_json,
                    json.dumps(checkpoint.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist checkpoint: {e}")
            logger.error(f"Checkpoint data: {checkpoint}")
            raise
    
    async def _load_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """Load checkpoint from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM state_checkpoints WHERE checkpoint_id = ?", 
                    (checkpoint_id,)
                )
                row = cursor.fetchone()
                if row:
                    return self._row_to_checkpoint(row)
                return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _row_to_agent_state(self, row) -> AgentState:
        """Convert database row to AgentState"""
        return AgentState(
            agent_id=row[0],
            status=AgentStatus(row[1]),
            current_task=row[2],
            last_activity=datetime.fromisoformat(row[3]) if row[3] else None,
            metadata=json.loads(row[4]) if row[4] else {},
            performance_metrics=json.loads(row[5]) if row[5] else {},
            error_info=row[6]
        )
    
    def _row_to_workflow_state(self, row) -> WorkflowState:
        """Convert database row to WorkflowState"""
        return WorkflowState(
            workflow_id=row[0],
            status=WorkflowStatus(row[1]),
            current_step=row[2],
            progress=row[3] or 0.0,
            started_at=datetime.fromisoformat(row[4]) if row[4] else None,
            completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
            steps_completed=json.loads(row[6]) if row[6] else [],
            steps_remaining=json.loads(row[7]) if row[7] else [],
            metadata=json.loads(row[8]) if row[8] else {},
            error_info=row[9]
        )
    
    def _row_to_conversation_state(self, row) -> ConversationState:
        """Convert database row to ConversationState"""
        return ConversationState(
            conversation_id=row[0],
            user_id=row[1],
            context=json.loads(row[2]) if row[2] else {},
            active_agents=json.loads(row[3]) if row[3] else [],
            last_interaction=datetime.fromisoformat(row[4]) if row[4] else None,
            metadata=json.loads(row[5]) if row[5] else {}
        )
    
    def _row_to_checkpoint(self, row) -> StateCheckpoint:
        """Convert database row to StateCheckpoint"""
        try:
            # Parse agent states
            agent_states_data = json.loads(row[2]) if row[2] else {}
            agent_states = {}
            for k, v in agent_states_data.items():
                # Handle enum conversion
                if isinstance(v['status'], str):
                    v['status'] = AgentStatus(v['status'])
                # Handle datetime conversion
                if v.get('last_activity') and isinstance(v['last_activity'], str):
                    v['last_activity'] = datetime.fromisoformat(v['last_activity'])
                agent_states[k] = AgentState(**v)
            
            # Parse workflow states
            workflow_states_data = json.loads(row[3]) if row[3] else {}
            workflow_states = {}
            for k, v in workflow_states_data.items():
                # Handle enum conversion
                if isinstance(v['status'], str):
                    v['status'] = WorkflowStatus(v['status'])
                # Handle datetime conversions
                if v.get('started_at') and isinstance(v['started_at'], str):
                    v['started_at'] = datetime.fromisoformat(v['started_at'])
                if v.get('completed_at') and isinstance(v['completed_at'], str):
                    v['completed_at'] = datetime.fromisoformat(v['completed_at'])
                workflow_states[k] = WorkflowState(**v)
            
            # Parse conversation state
            conversation_state = None
            if row[4] and row[4] != 'null':
                conv_data = json.loads(row[4])
                if conv_data and conv_data.get('last_interaction') and isinstance(conv_data['last_interaction'], str):
                    conv_data['last_interaction'] = datetime.fromisoformat(conv_data['last_interaction'])
                conversation_state = ConversationState(**conv_data)
            
            return StateCheckpoint(
                checkpoint_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                agent_states=agent_states,
                workflow_states=workflow_states,
                conversation_state=conversation_state,
                metadata=json.loads(row[5]) if row[5] else {}
            )
        except Exception as e:
            logger.error(f"Error converting row to checkpoint: {e}")
            logger.error(f"Row data: {row}")
            raise

# Context manager for easy service lifecycle management
@asynccontextmanager
async def state_service_context(db_path: str = "state_management.db", checkpoint_interval: int = 300):
    """Context manager for StateService lifecycle
    
    Args:
        db_path: Path to SQLite database file
        checkpoint_interval: Automatic checkpoint interval in seconds
        
    Yields:
        StateService instance
    """
    service = StateService(db_path, checkpoint_interval)
    try:
        await service.start()
        yield service
    finally:
        await service.stop()

# Example usage and testing
if __name__ == "__main__":
    async def main():
        async with state_service_context() as state_service:
            # Test agent state management
            await state_service.update_agent_state(
                "agent_1", 
                status="active", 
                current_task="processing_request"
            )
            
            # Test workflow state management
            await state_service.update_workflow_state(
                "workflow_1",
                status="running",
                current_step="step_2",
                progress=50.0
            )
            
            # Test conversation state
            await state_service.update_conversation_state(
                "conv_1",
                user_id="user_123",
                active_agents=["agent_1"]
            )
            
            # Create checkpoint
            await state_service.checkpoint_state("test_checkpoint")
            
            # Get performance stats
            stats = await state_service.get_performance_stats()
            print(f"Performance stats: {stats}")
    
    asyncio.run(main())