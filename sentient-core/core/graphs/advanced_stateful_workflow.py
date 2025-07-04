"""Advanced LangGraph-based Stateful Workflows with Persistence and Checkpointing

This module implements a comprehensive state management system that enhances the existing
LangGraph implementation with advanced features including:
- Persistent state storage with multiple backends
- Automatic checkpointing and recovery
- Multi-agent coordination and state synchronization
- Advanced routing and conditional logic
- State versioning and rollback capabilities
- Cross-session state sharing
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Type, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from pathlib import Path
import sqlite3
import pickle
import threading
from contextlib import contextmanager

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.prebuilt import ToolExecutor
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Advanced state management will be limited.")

from core.models import AppState, EnhancedTask, AgentType, TaskStatus, SessionState
from core.services.enhanced_state_manager import EnhancedStateManager, StateConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CheckpointType(str, Enum):
    """Types of checkpoints for state persistence."""
    AUTOMATIC = "automatic"  # System-generated checkpoints
    MANUAL = "manual"       # User-triggered checkpoints
    MILESTONE = "milestone"  # Important workflow milestones
    RECOVERY = "recovery"    # Recovery points after errors
    BRANCH = "branch"       # Branching points for parallel execution

class WorkflowStatus(str, Enum):
    """Status of workflow execution."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"

@dataclass
class StateCheckpoint:
    """Represents a state checkpoint for recovery and rollback."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    workflow_id: str = ""
    session_id: str = ""
    agent_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution monitoring."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_nodes_executed: int = 0
    total_state_transitions: int = 0
    checkpoint_count: int = 0
    error_count: int = 0
    recovery_count: int = 0
    average_node_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    
    @property
    def execution_duration(self) -> timedelta:
        end = self.end_time or datetime.now()
        return end - self.start_time

class AdvancedCheckpointSaver(BaseCheckpointSaver):
    """Enhanced checkpoint saver with advanced persistence features."""
    
    def __init__(self, db_path: str = "checkpoints.db", max_checkpoints: int = 100):
        self.db_path = db_path
        self.max_checkpoints = max_checkpoints
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the checkpoint database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT,
                    session_id TEXT,
                    timestamp TEXT,
                    checkpoint_type TEXT,
                    state_data BLOB,
                    metadata TEXT,
                    description TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_session 
                ON checkpoints(workflow_id, session_id, timestamp)
            """)
    
    def save_checkpoint(self, checkpoint: StateCheckpoint) -> bool:
        """Save a checkpoint to persistent storage."""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO checkpoints 
                        (id, workflow_id, session_id, timestamp, checkpoint_type, 
                         state_data, metadata, description)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        checkpoint.id,
                        checkpoint.workflow_id,
                        checkpoint.session_id,
                        checkpoint.timestamp.isoformat(),
                        checkpoint.checkpoint_type.value,
                        pickle.dumps(checkpoint.state_snapshot),
                        json.dumps(checkpoint.metadata),
                        checkpoint.description
                    ))
                    
                    # Clean up old checkpoints if needed
                    self._cleanup_old_checkpoints(conn, checkpoint.workflow_id)
                    
            logger.info(f"Checkpoint saved: {checkpoint.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """Load a specific checkpoint."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return StateCheckpoint(
                        id=row[0],
                        workflow_id=row[1],
                        session_id=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        checkpoint_type=CheckpointType(row[4]),
                        state_snapshot=pickle.loads(row[5]),
                        metadata=json.loads(row[6]),
                        description=row[7]
                    )
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
        return None
    
    def get_latest_checkpoint(self, workflow_id: str, session_id: str) -> Optional[StateCheckpoint]:
        """Get the most recent checkpoint for a workflow/session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM checkpoints 
                    WHERE workflow_id = ? AND session_id = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (workflow_id, session_id))
                row = cursor.fetchone()
                
                if row:
                    return StateCheckpoint(
                        id=row[0],
                        workflow_id=row[1],
                        session_id=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        checkpoint_type=CheckpointType(row[4]),
                        state_snapshot=pickle.loads(row[5]),
                        metadata=json.loads(row[6]),
                        description=row[7]
                    )
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
        return None
    
    def list_checkpoints(self, workflow_id: str, session_id: str) -> List[StateCheckpoint]:
        """List all checkpoints for a workflow/session."""
        checkpoints = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM checkpoints 
                    WHERE workflow_id = ? AND session_id = ?
                    ORDER BY timestamp DESC
                """, (workflow_id, session_id))
                
                for row in cursor.fetchall():
                    checkpoints.append(StateCheckpoint(
                        id=row[0],
                        workflow_id=row[1],
                        session_id=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        checkpoint_type=CheckpointType(row[4]),
                        state_snapshot=pickle.loads(row[5]),
                        metadata=json.loads(row[6]),
                        description=row[7]
                    ))
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
        return checkpoints
    
    def _cleanup_old_checkpoints(self, conn, workflow_id: str):
        """Remove old checkpoints to maintain storage limits."""
        cursor = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE workflow_id = ?", 
            (workflow_id,)
        )
        count = cursor.fetchone()[0]
        
        if count > self.max_checkpoints:
            conn.execute("""
                DELETE FROM checkpoints WHERE id IN (
                    SELECT id FROM checkpoints 
                    WHERE workflow_id = ?
                    ORDER BY timestamp ASC 
                    LIMIT ?
                )
            """, (workflow_id, count - self.max_checkpoints))

class StatefulWorkflowManager:
    """Advanced workflow manager with state persistence and multi-agent coordination."""
    
    def __init__(self, 
                 checkpoint_saver: Optional[AdvancedCheckpointSaver] = None,
                 state_manager: Optional[EnhancedStateManager] = None):
        self.checkpoint_saver = checkpoint_saver or AdvancedCheckpointSaver()
        self.state_manager = state_manager or EnhancedStateManager(
            StateConfig(persistence_type="hybrid")
        )
        
        # Workflow management
        self.active_workflows: Dict[str, StateGraph] = {}
        self.workflow_states: Dict[str, AppState] = {}
        self.workflow_metrics: Dict[str, WorkflowMetrics] = {}
        self.workflow_status: Dict[str, WorkflowStatus] = {}
        
        # Multi-agent coordination
        self.agent_sessions: Dict[str, Dict[str, Any]] = {}
        self.shared_context: Dict[str, Any] = {}
        self.coordination_locks: Dict[str, threading.Lock] = {}
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.state_change_listeners: List[Callable] = []
        
        logger.info("Stateful Workflow Manager initialized")
    
    def create_workflow(self, 
                       workflow_id: str,
                       state_class: Type = AppState,
                       auto_checkpoint: bool = True,
                       checkpoint_interval: int = 5) -> StateGraph:
        """Create a new stateful workflow with advanced features."""
        
        if not LANGGRAPH_AVAILABLE:
            raise RuntimeError("LangGraph is required for stateful workflows")
        
        # Create the workflow graph
        workflow = StateGraph(state_class)
        
        # Initialize workflow tracking
        self.active_workflows[workflow_id] = workflow
        self.workflow_metrics[workflow_id] = WorkflowMetrics()
        self.workflow_status[workflow_id] = WorkflowStatus.INITIALIZED
        self.coordination_locks[workflow_id] = threading.Lock()
        
        # Set up automatic checkpointing if enabled
        if auto_checkpoint:
            self._setup_auto_checkpointing(workflow_id, checkpoint_interval)
        
        logger.info(f"Created workflow: {workflow_id}")
        return workflow
    
    def register_agent_node(self, 
                           workflow_id: str,
                           node_name: str,
                           agent_func: Callable,
                           agent_type: AgentType,
                           checkpoint_before: bool = True,
                           checkpoint_after: bool = True) -> None:
        """Register an agent node with automatic state management."""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Wrap the agent function with state management
        wrapped_func = self._wrap_agent_function(
            workflow_id, node_name, agent_func, agent_type,
            checkpoint_before, checkpoint_after
        )
        
        workflow.add_node(node_name, wrapped_func)
        logger.info(f"Registered agent node: {node_name} in workflow {workflow_id}")
    
    def _wrap_agent_function(self, 
                            workflow_id: str,
                            node_name: str,
                            agent_func: Callable,
                            agent_type: AgentType,
                            checkpoint_before: bool,
                            checkpoint_after: bool) -> Callable:
        """Wrap an agent function with state management capabilities."""
        
        async def wrapped_agent(state: AppState) -> AppState:
            session_id = getattr(state, 'session_id', 'default')
            
            # Pre-execution checkpoint
            if checkpoint_before:
                await self._create_checkpoint(
                    workflow_id, session_id, state,
                    CheckpointType.AUTOMATIC,
                    f"Before {node_name} execution"
                )
            
            # Update metrics
            metrics = self.workflow_metrics[workflow_id]
            start_time = datetime.now()
            
            try:
                # Execute the agent function
                if asyncio.iscoroutinefunction(agent_func):
                    result_state = await agent_func(state)
                else:
                    result_state = agent_func(state)
                
                # Update execution metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                metrics.total_nodes_executed += 1
                metrics.average_node_execution_time = (
                    (metrics.average_node_execution_time * (metrics.total_nodes_executed - 1) + execution_time) /
                    metrics.total_nodes_executed
                )
                
                # Post-execution checkpoint
                if checkpoint_after:
                    await self._create_checkpoint(
                        workflow_id, session_id, result_state,
                        CheckpointType.AUTOMATIC,
                        f"After {node_name} execution"
                    )
                
                # Notify state change listeners
                await self._notify_state_change(workflow_id, node_name, state, result_state)
                
                return result_state
                
            except Exception as e:
                metrics.error_count += 1
                logger.error(f"Error in {node_name}: {e}")
                
                # Create recovery checkpoint
                await self._create_checkpoint(
                    workflow_id, session_id, state,
                    CheckpointType.RECOVERY,
                    f"Error recovery point for {node_name}: {str(e)}"
                )
                
                # Attempt recovery
                recovered_state = await self._attempt_recovery(workflow_id, session_id, state, e)
                return recovered_state or state
        
        return wrapped_agent
    
    async def _create_checkpoint(self, 
                                workflow_id: str,
                                session_id: str,
                                state: AppState,
                                checkpoint_type: CheckpointType,
                                description: str) -> str:
        """Create a state checkpoint."""
        
        checkpoint = StateCheckpoint(
            workflow_id=workflow_id,
            session_id=session_id,
            checkpoint_type=checkpoint_type,
            state_snapshot=state.dict(),
            description=description,
            metadata={
                'workflow_status': self.workflow_status[workflow_id].value,
                'metrics': self.workflow_metrics[workflow_id].__dict__
            }
        )
        
        success = self.checkpoint_saver.save_checkpoint(checkpoint)
        if success:
            self.workflow_metrics[workflow_id].checkpoint_count += 1
            return checkpoint.id
        
        return ""
    
    async def _attempt_recovery(self, 
                               workflow_id: str,
                               session_id: str,
                               current_state: AppState,
                               error: Exception) -> Optional[AppState]:
        """Attempt to recover from an error using checkpoints."""
        
        try:
            # Get the latest recovery checkpoint
            latest_checkpoint = self.checkpoint_saver.get_latest_checkpoint(
                workflow_id, session_id
            )
            
            if latest_checkpoint:
                # Restore state from checkpoint
                recovered_state = AppState(**latest_checkpoint.state_snapshot)
                
                # Update recovery metrics
                self.workflow_metrics[workflow_id].recovery_count += 1
                
                logger.info(f"Recovered workflow {workflow_id} from checkpoint {latest_checkpoint.id}")
                return recovered_state
                
        except Exception as recovery_error:
            logger.error(f"Recovery failed for workflow {workflow_id}: {recovery_error}")
        
        return None
    
    async def _notify_state_change(self, 
                                  workflow_id: str,
                                  node_name: str,
                                  old_state: AppState,
                                  new_state: AppState) -> None:
        """Notify registered listeners of state changes."""
        
        for listener in self.state_change_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(workflow_id, node_name, old_state, new_state)
                else:
                    listener(workflow_id, node_name, old_state, new_state)
            except Exception as e:
                logger.error(f"State change listener error: {e}")
    
    def add_state_change_listener(self, listener: Callable) -> None:
        """Add a listener for state changes."""
        self.state_change_listeners.append(listener)
    
    def setup_multi_agent_coordination(self, 
                                      workflow_id: str,
                                      coordination_rules: Dict[str, Any]) -> None:
        """Set up coordination rules between agents."""
        
        self.agent_sessions[workflow_id] = {
            'coordination_rules': coordination_rules,
            'shared_state': {},
            'agent_locks': {},
            'message_queue': deque(),
            'coordination_history': []
        }
        
        logger.info(f"Multi-agent coordination set up for workflow {workflow_id}")
    
    async def coordinate_agents(self, 
                               workflow_id: str,
                               requesting_agent: str,
                               coordination_type: str,
                               data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate between multiple agents."""
        
        session = self.agent_sessions.get(workflow_id, {})
        coordination_rules = session.get('coordination_rules', {})
        
        # Apply coordination rules
        if coordination_type in coordination_rules:
            rule = coordination_rules[coordination_type]
            
            # Handle different coordination patterns
            if rule.get('type') == 'sequential':
                return await self._handle_sequential_coordination(workflow_id, requesting_agent, data)
            elif rule.get('type') == 'parallel':
                return await self._handle_parallel_coordination(workflow_id, requesting_agent, data)
            elif rule.get('type') == 'conditional':
                return await self._handle_conditional_coordination(workflow_id, requesting_agent, data)
        
        return {'status': 'no_coordination_rule', 'data': data}
    
    async def _handle_sequential_coordination(self, 
                                            workflow_id: str,
                                            requesting_agent: str,
                                            data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sequential agent coordination."""
        
        with self.coordination_locks[workflow_id]:
            session = self.agent_sessions[workflow_id]
            
            # Add to coordination history
            session['coordination_history'].append({
                'timestamp': datetime.now(),
                'agent': requesting_agent,
                'type': 'sequential',
                'data': data
            })
            
            # Update shared state
            session['shared_state'].update(data)
            
            return {
                'status': 'coordinated',
                'shared_state': session['shared_state'].copy(),
                'next_agent': data.get('next_agent')
            }
    
    async def _handle_parallel_coordination(self, 
                                          workflow_id: str,
                                          requesting_agent: str,
                                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle parallel agent coordination."""
        
        session = self.agent_sessions[workflow_id]
        
        # Create agent-specific lock if needed
        if requesting_agent not in session['agent_locks']:
            session['agent_locks'][requesting_agent] = threading.Lock()
        
        with session['agent_locks'][requesting_agent]:
            # Update agent-specific state
            agent_key = f"{requesting_agent}_state"
            session['shared_state'][agent_key] = data
            
            return {
                'status': 'parallel_coordinated',
                'agent_state': data,
                'all_agent_states': {k: v for k, v in session['shared_state'].items() if k.endswith('_state')}
            }
    
    async def _handle_conditional_coordination(self, 
                                             workflow_id: str,
                                             requesting_agent: str,
                                             data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conditional agent coordination."""
        
        session = self.agent_sessions[workflow_id]
        coordination_rules = session['coordination_rules']
        
        # Evaluate conditions
        conditions = coordination_rules.get('conditional', {}).get('conditions', [])
        
        for condition in conditions:
            if self._evaluate_condition(condition, data, session['shared_state']):
                # Execute the action for this condition
                action = condition.get('action', {})
                return await self._execute_coordination_action(workflow_id, requesting_agent, action, data)
        
        # Default action if no conditions match
        return {'status': 'no_condition_matched', 'data': data}
    
    def _evaluate_condition(self, 
                           condition: Dict[str, Any],
                           current_data: Dict[str, Any],
                           shared_state: Dict[str, Any]) -> bool:
        """Evaluate a coordination condition."""
        
        condition_type = condition.get('type')
        
        if condition_type == 'state_value':
            key = condition.get('key')
            expected_value = condition.get('value')
            actual_value = shared_state.get(key)
            return actual_value == expected_value
        
        elif condition_type == 'data_contains':
            key = condition.get('key')
            return key in current_data
        
        elif condition_type == 'agent_completed':
            agent = condition.get('agent')
            agent_key = f"{agent}_completed"
            return shared_state.get(agent_key, False)
        
        return False
    
    async def _execute_coordination_action(self, 
                                          workflow_id: str,
                                          requesting_agent: str,
                                          action: Dict[str, Any],
                                          data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a coordination action."""
        
        action_type = action.get('type')
        
        if action_type == 'wait_for_agent':
            target_agent = action.get('agent')
            return {'status': 'waiting', 'waiting_for': target_agent}
        
        elif action_type == 'notify_agents':
            agents = action.get('agents', [])
            return {'status': 'notified', 'notified_agents': agents, 'data': data}
        
        elif action_type == 'merge_states':
            session = self.agent_sessions[workflow_id]
            merged_state = {**session['shared_state'], **data}
            session['shared_state'] = merged_state
            return {'status': 'states_merged', 'merged_state': merged_state}
        
        return {'status': 'unknown_action', 'action': action}
    
    def compile_workflow(self, 
                        workflow_id: str,
                        checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
        """Compile a workflow with advanced checkpointing."""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Use custom checkpointer if provided, otherwise use the default
        if checkpointer is None:
            checkpointer = MemorySaver()  # Default LangGraph checkpointer
        
        # Compile with checkpointing enabled
        compiled_workflow = workflow.compile(checkpointer=checkpointer)
        
        # Update workflow status
        self.workflow_status[workflow_id] = WorkflowStatus.RUNNING
        
        logger.info(f"Compiled workflow: {workflow_id}")
        return compiled_workflow
    
    def get_workflow_metrics(self, workflow_id: str) -> Optional[WorkflowMetrics]:
        """Get metrics for a specific workflow."""
        return self.workflow_metrics.get(workflow_id)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get the current status of a workflow."""
        return self.workflow_status.get(workflow_id)
    
    def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs."""
        return list(self.active_workflows.keys())
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id in self.workflow_status:
            self.workflow_status[workflow_id] = WorkflowStatus.PAUSED
            logger.info(f"Paused workflow: {workflow_id}")
            return True
        return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id in self.workflow_status:
            self.workflow_status[workflow_id] = WorkflowStatus.RUNNING
            logger.info(f"Resumed workflow: {workflow_id}")
            return True
        return False
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id in self.workflow_status:
            self.workflow_status[workflow_id] = WorkflowStatus.CANCELLED
            
            # Clean up resources
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            if workflow_id in self.agent_sessions:
                del self.agent_sessions[workflow_id]
            if workflow_id in self.coordination_locks:
                del self.coordination_locks[workflow_id]
            
            logger.info(f"Cancelled workflow: {workflow_id}")
            return True
        return False
    
    def _setup_auto_checkpointing(self, workflow_id: str, interval: int) -> None:
        """Set up automatic checkpointing for a workflow."""
        
        async def auto_checkpoint_task():
            while (self.workflow_status.get(workflow_id) == WorkflowStatus.RUNNING and
                   workflow_id in self.active_workflows):
                
                await asyncio.sleep(interval)
                
                # Create automatic checkpoint if workflow is still active
                if workflow_id in self.workflow_states:
                    state = self.workflow_states[workflow_id]
                    session_id = getattr(state, 'session_id', 'default')
                    
                    await self._create_checkpoint(
                        workflow_id, session_id, state,
                        CheckpointType.AUTOMATIC,
                        f"Automatic checkpoint (interval: {interval}s)"
                    )
        
        # Start the auto-checkpoint task
        asyncio.create_task(auto_checkpoint_task())
        logger.info(f"Auto-checkpointing enabled for workflow {workflow_id} (interval: {interval}s)")

# Convenience functions for creating common workflow patterns

def create_research_workflow(workflow_manager: StatefulWorkflowManager,
                           workflow_id: str = "research_workflow") -> StateGraph:
    """Create a research workflow with advanced state management."""
    
    from core.models import ResearchState
    from core.agents.research_agent import ResearchAgent
    from core.services.llm_service import EnhancedLLMService
    
    # Create workflow
    workflow = workflow_manager.create_workflow(workflow_id, ResearchState)
    
    # Initialize research agent
    llm_service = EnhancedLLMService()
    research_agent = ResearchAgent(llm_service)
    
    # Register agent nodes with checkpointing
    workflow_manager.register_agent_node(
        workflow_id, "plan_steps", research_agent.plan_steps, 
        AgentType.RESEARCH_AGENT, checkpoint_before=True, checkpoint_after=True
    )
    
    workflow_manager.register_agent_node(
        workflow_id, "execute_search", research_agent.execute_search,
        AgentType.RESEARCH_AGENT, checkpoint_before=False, checkpoint_after=True
    )
    
    workflow_manager.register_agent_node(
        workflow_id, "synthesize_report", research_agent.synthesize_report,
        AgentType.RESEARCH_AGENT, checkpoint_before=False, checkpoint_after=True
    )
    
    # Set up workflow structure
    workflow.set_entry_point("plan_steps")
    workflow.add_edge("synthesize_report", END)
    
    # Add conditional routing with state-aware logic
    def should_continue_research(state: ResearchState) -> str:
        if any(step.status == "pending" for step in state.steps):
            return "execute_search"
        else:
            return "synthesize_report"
    
    workflow.add_conditional_edges(
        "plan_steps",
        should_continue_research,
        {
            "execute_search": "execute_search",
            "synthesize_report": "synthesize_report",
        },
    )
    
    workflow.add_conditional_edges(
        "execute_search",
        should_continue_research,
        {
            "execute_search": "execute_search",
            "synthesize_report": "synthesize_report",
        },
    )
    
    return workflow

def create_orchestrator_workflow(workflow_manager: StatefulWorkflowManager,
                                workflow_id: str = "orchestrator_workflow") -> StateGraph:
    """Create an orchestrator workflow with multi-agent coordination."""
    
    from core.agents.ultra_orchestrator import UltraOrchestrator
    from core.agents.monitoring_agent import MonitoringAgent
    from core.services.llm_service import EnhancedLLMService
    
    # Create workflow
    workflow = workflow_manager.create_workflow(workflow_id, AppState)
    
    # Initialize agents
    llm_service = EnhancedLLMService()
    ultra_orchestrator = UltraOrchestrator(llm_service)
    monitoring_agent = MonitoringAgent()
    
    # Register agent nodes
    workflow_manager.register_agent_node(
        workflow_id, "ultra_orchestrator", ultra_orchestrator.invoke,
        AgentType.ULTRA_ORCHESTRATOR, checkpoint_before=True, checkpoint_after=True
    )
    
    workflow_manager.register_agent_node(
        workflow_id, "monitor", monitoring_agent.invoke,
        AgentType.MONITORING_AGENT, checkpoint_before=False, checkpoint_after=True
    )
    
    # Set up multi-agent coordination
    coordination_rules = {
        'task_delegation': {
            'type': 'sequential',
            'agents': ['ultra_orchestrator', 'monitor']
        },
        'error_handling': {
            'type': 'conditional',
            'conditions': [
                {
                    'type': 'state_value',
                    'key': 'error_occurred',
                    'value': True,
                    'action': {
                        'type': 'notify_agents',
                        'agents': ['monitor']
                    }
                }
            ]
        }
    }
    
    workflow_manager.setup_multi_agent_coordination(workflow_id, coordination_rules)
    
    # Set up workflow structure
    workflow.set_entry_point("ultra_orchestrator")
    workflow.add_edge("ultra_orchestrator", "monitor")
    
    # Add conditional routing
    def route_from_monitor(state: AppState) -> str:
        decision = state.next_action
        if decision == "create_plan":
            return "end"
        return "end"
    
    workflow.add_conditional_edges(
        "monitor",
        route_from_monitor,
        {"end": END}
    )
    
    return workflow

# Global workflow manager instance
default_workflow_manager = StatefulWorkflowManager()

# Export key components
__all__ = [
    'StatefulWorkflowManager',
    'AdvancedCheckpointSaver',
    'StateCheckpoint',
    'WorkflowMetrics',
    'CheckpointType',
    'WorkflowStatus',
    'create_research_workflow',
    'create_orchestrator_workflow',
    'default_workflow_manager'
]