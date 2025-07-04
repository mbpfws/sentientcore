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

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.prebuilt import ToolExecutor
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("Warning: LangGraph not available. Graph functionality will be limited.")

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty

# Import existing models
try:
    from ..models import AppState, AgentType, TaskStatus, SessionState
    from ..services.memory_service import MemoryService
    from ..nlp.advanced_nlp_processor import get_nlp_processor
except ImportError:
    print("Warning: Some imports failed. Using fallback implementations.")
    
    class AppState:
        pass
    
    class AgentType(Enum):
        ORCHESTRATOR = "orchestrator"
        RESEARCH = "research"
        CODING = "coding"
        MONITORING = "monitoring"
        PLANNING = "planning"
        EXECUTION = "execution"
    
    class TaskStatus(Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        CANCELLED = "cancelled"
    
    class SessionState(Enum):
        ACTIVE = "active"
        PAUSED = "paused"
        COMPLETED = "completed"
        ERROR = "error"

class GraphNodeType(Enum):
    """Types of graph nodes."""
    AGENT = "agent"
    ROUTER = "router"
    CONDITION = "condition"
    TRANSFORMER = "transformer"
    AGGREGATOR = "aggregator"
    VALIDATOR = "validator"
    MONITOR = "monitor"
    CHECKPOINT = "checkpoint"

class EdgeType(Enum):
    """Types of graph edges."""
    SEQUENTIAL = "sequential"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP = "loop"
    FALLBACK = "fallback"
    PRIORITY = "priority"

class ExecutionMode(Enum):
    """Graph execution modes."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"
    BATCH = "batch"

class NodeStatus(Enum):
    """Node execution status."""
    IDLE = "idle"
    WAITING = "waiting"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class GraphNode:
    """Represents a node in the execution graph."""
    id: str
    name: str
    node_type: GraphNodeType
    agent_type: Optional[AgentType] = None
    function: Optional[Callable] = None
    conditions: List[Callable] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.IDLE
    execution_count: int = 0
    last_execution: Optional[datetime] = None
    execution_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

@dataclass
class GraphEdge:
    """Represents an edge in the execution graph."""
    id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    condition: Optional[Callable] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    traversal_count: int = 0
    last_traversal: Optional[datetime] = None

@dataclass
class ExecutionContext:
    """Context for graph execution."""
    session_id: str
    execution_id: str
    mode: ExecutionMode
    state: Any
    start_time: datetime
    current_node: Optional[str] = None
    execution_path: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoints: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class GraphMetrics:
    """Graph execution metrics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    average_execution_time: float = 0.0
    node_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edge_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

class DynamicRouter:
    """Dynamic routing logic for graph execution."""
    
    def __init__(self):
        self.routing_rules: Dict[str, List[Callable]] = defaultdict(list)
        self.fallback_routes: Dict[str, str] = {}
        self.priority_routes: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.nlp_processor = None
        
        try:
            self.nlp_processor = get_nlp_processor()
        except Exception as e:
            print(f"NLP processor not available for routing: {e}")
    
    def add_routing_rule(self, from_node: str, rule: Callable[[Any], Optional[str]]):
        """Add a routing rule for a node."""
        self.routing_rules[from_node].append(rule)
    
    def add_fallback_route(self, from_node: str, to_node: str):
        """Add a fallback route for a node."""
        self.fallback_routes[from_node] = to_node
    
    def add_priority_route(self, from_node: str, to_node: str, priority: int):
        """Add a priority route for a node."""
        self.priority_routes[from_node].append((to_node, priority))
        self.priority_routes[from_node].sort(key=lambda x: x[1], reverse=True)
    
    async def route(self, from_node: str, state: Any, context: ExecutionContext) -> Optional[str]:
        """Determine the next node based on routing rules."""
        # Apply routing rules in order
        for rule in self.routing_rules[from_node]:
            try:
                next_node = rule(state)
                if next_node:
                    return next_node
            except Exception as e:
                print(f"Routing rule failed: {e}")
                continue
        
        # Check priority routes
        if from_node in self.priority_routes:
            for to_node, priority in self.priority_routes[from_node]:
                # Simple availability check - in practice, you'd check node status
                return to_node
        
        # Use fallback route
        if from_node in self.fallback_routes:
            return self.fallback_routes[from_node]
        
        return None
    
    def add_intent_based_routing(self, from_node: str, intent_mapping: Dict[str, str]):
        """Add intent-based routing using NLP analysis."""
        def intent_router(state) -> Optional[str]:
            if not self.nlp_processor:
                return None
            
            # Extract text from state (this would need to be adapted based on your state structure)
            text = ""
            if hasattr(state, 'user_input'):
                text = state.user_input
            elif hasattr(state, 'messages') and state.messages:
                text = state.messages[-1].content if hasattr(state.messages[-1], 'content') else str(state.messages[-1])
            
            if not text:
                return None
            
            try:
                # Analyze intent
                import asyncio
                analysis = asyncio.run(self.nlp_processor.analyze_text(text, include_intent=True))
                
                if analysis.intent and analysis.intent.confidence > 0.6:
                    intent_type = analysis.intent.intent_type.value
                    return intent_mapping.get(intent_type)
            except Exception as e:
                print(f"Intent-based routing failed: {e}")
            
            return None
        
        self.add_routing_rule(from_node, intent_router)
    
    def add_entity_based_routing(self, from_node: str, entity_mapping: Dict[str, str]):
        """Add entity-based routing using NLP analysis."""
        def entity_router(state) -> Optional[str]:
            if not self.nlp_processor:
                return None
            
            # Extract text from state
            text = ""
            if hasattr(state, 'user_input'):
                text = state.user_input
            elif hasattr(state, 'messages') and state.messages:
                text = state.messages[-1].content if hasattr(state.messages[-1], 'content') else str(state.messages[-1])
            
            if not text:
                return None
            
            try:
                # Analyze entities
                import asyncio
                analysis = asyncio.run(self.nlp_processor.analyze_text(text, include_entities=True))
                
                for entity in analysis.entities:
                    entity_type = entity.entity_type.value
                    if entity_type in entity_mapping and entity.confidence > 0.7:
                        return entity_mapping[entity_type]
            except Exception as e:
                print(f"Entity-based routing failed: {e}")
            
            return None
        
        self.add_routing_rule(from_node, entity_router)

class StateManager:
    """Advanced state management for graph execution."""
    
    def __init__(self):
        self.state_history: Dict[str, List[Any]] = defaultdict(list)
        self.state_snapshots: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.state_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.state_validators: List[Callable[[Any], bool]] = []
        self.state_transformers: Dict[str, List[Callable]] = defaultdict(list)
        self.change_listeners: List[Callable[[str, Any, Any], None]] = []
    
    def add_validator(self, validator: Callable[[Any], bool]):
        """Add a state validator."""
        self.state_validators.append(validator)
    
    def add_transformer(self, node_id: str, transformer: Callable[[Any], Any]):
        """Add a state transformer for a specific node."""
        self.state_transformers[node_id].append(transformer)
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Add a state change listener."""
        self.change_listeners.append(listener)
    
    async def update_state(self, session_id: str, node_id: str, new_state: Any) -> bool:
        """Update state with validation and transformation."""
        with self.state_locks[session_id]:
            # Get current state
            current_state = self.get_current_state(session_id)
            
            # Apply transformers
            transformed_state = new_state
            for transformer in self.state_transformers[node_id]:
                try:
                    transformed_state = transformer(transformed_state)
                except Exception as e:
                    print(f"State transformer failed: {e}")
                    return False
            
            # Validate state
            for validator in self.state_validators:
                try:
                    if not validator(transformed_state):
                        print(f"State validation failed for session {session_id}")
                        return False
                except Exception as e:
                    print(f"State validator error: {e}")
                    return False
            
            # Update state history
            self.state_history[session_id].append(transformed_state)
            
            # Notify listeners
            for listener in self.change_listeners:
                try:
                    listener(session_id, current_state, transformed_state)
                except Exception as e:
                    print(f"State change listener error: {e}")
            
            return True
    
    def get_current_state(self, session_id: str) -> Any:
        """Get the current state for a session."""
        history = self.state_history[session_id]
        return history[-1] if history else None
    
    def get_state_history(self, session_id: str, limit: Optional[int] = None) -> List[Any]:
        """Get state history for a session."""
        history = self.state_history[session_id]
        if limit:
            return history[-limit:]
        return history.copy()
    
    def create_snapshot(self, session_id: str, snapshot_name: str):
        """Create a named snapshot of the current state."""
        current_state = self.get_current_state(session_id)
        if current_state:
            self.state_snapshots[session_id][snapshot_name] = current_state
    
    def restore_snapshot(self, session_id: str, snapshot_name: str) -> bool:
        """Restore state from a named snapshot."""
        if snapshot_name in self.state_snapshots[session_id]:
            snapshot_state = self.state_snapshots[session_id][snapshot_name]
            self.state_history[session_id].append(snapshot_state)
            return True
        return False
    
    def rollback(self, session_id: str, steps: int = 1) -> bool:
        """Rollback state by specified number of steps."""
        history = self.state_history[session_id]
        if len(history) > steps:
            # Remove the last 'steps' states
            for _ in range(steps):
                history.pop()
            return True
        return False

class GraphExecutor:
    """Advanced graph execution engine."""
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.router = DynamicRouter()
        self.state_manager = StateManager()
        self.metrics = GraphMetrics()
        self.execution_contexts: Dict[str, ExecutionContext] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.event_queue: Queue = Queue()
        self.checkpoint_manager = None
        
        # Initialize checkpoint manager if LangGraph is available
        if LANGGRAPH_AVAILABLE:
            try:
                self.checkpoint_manager = MemorySaver()
            except Exception as e:
                print(f"Failed to initialize checkpoint manager: {e}")
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        self.nodes[node.id] = node
        self.metrics.node_metrics[node.id] = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "avg_execution_time": 0.0,
            "last_execution": None
        }
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        self.edges[edge.id] = edge
        self.metrics.edge_metrics[edge.id] = {
            "traversals": 0,
            "last_traversal": None
        }
    
    def remove_node(self, node_id: str):
        """Remove a node and its associated edges."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            
            # Remove associated edges
            edges_to_remove = [
                edge_id for edge_id, edge in self.edges.items()
                if edge.source_node == node_id or edge.target_node == node_id
            ]
            
            for edge_id in edges_to_remove:
                del self.edges[edge_id]
    
    def get_outgoing_edges(self, node_id: str) -> List[GraphEdge]:
        """Get all outgoing edges from a node."""
        return [edge for edge in self.edges.values() if edge.source_node == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[GraphEdge]:
        """Get all incoming edges to a node."""
        return [edge for edge in self.edges.values() if edge.target_node == node_id]
    
    async def execute_graph(self, session_id: str, initial_state: Any, 
                           start_node: str, mode: ExecutionMode = ExecutionMode.SYNCHRONOUS) -> Any:
        """Execute the graph starting from a specific node."""
        execution_id = str(uuid.uuid4())
        
        context = ExecutionContext(
            session_id=session_id,
            execution_id=execution_id,
            mode=mode,
            state=initial_state,
            start_time=datetime.now(),
            current_node=start_node
        )
        
        self.execution_contexts[execution_id] = context
        
        try:
            if mode == ExecutionMode.SYNCHRONOUS:
                return await self._execute_synchronous(context)
            elif mode == ExecutionMode.ASYNCHRONOUS:
                return await self._execute_asynchronous(context)
            elif mode == ExecutionMode.STREAMING:
                return self._execute_streaming(context)
            elif mode == ExecutionMode.BATCH:
                return await self._execute_batch(context)
            else:
                raise ValueError(f"Unsupported execution mode: {mode}")
        
        except Exception as e:
            context.errors.append({
                "error": str(e),
                "timestamp": datetime.now(),
                "node": context.current_node
            })
            self.metrics.failed_executions += 1
            raise
        
        finally:
            # Update metrics
            execution_time = (datetime.now() - context.start_time).total_seconds()
            self.metrics.total_executions += 1
            
            if not context.errors:
                self.metrics.successful_executions += 1
            
            # Update average execution time
            total_time = self.metrics.average_execution_time * (self.metrics.total_executions - 1) + execution_time
            self.metrics.average_execution_time = total_time / self.metrics.total_executions
            self.metrics.last_updated = datetime.now()
    
    async def _execute_synchronous(self, context: ExecutionContext) -> Any:
        """Execute graph synchronously."""
        current_node_id = context.current_node
        
        while current_node_id:
            context.execution_path.append(current_node_id)
            
            # Execute current node
            node = self.nodes[current_node_id]
            context.current_node = current_node_id
            
            try:
                start_time = datetime.now()
                node.status = NodeStatus.EXECUTING
                
                # Execute node function
                if node.function:
                    if asyncio.iscoroutinefunction(node.function):
                        context.state = await node.function(context.state)
                    else:
                        context.state = node.function(context.state)
                
                # Update state
                await self.state_manager.update_state(context.session_id, current_node_id, context.state)
                
                # Update node metrics
                execution_time = (datetime.now() - start_time).total_seconds()
                node.execution_time = execution_time
                node.execution_count += 1
                node.last_execution = datetime.now()
                node.status = NodeStatus.COMPLETED
                
                # Update metrics
                node_metrics = self.metrics.node_metrics[current_node_id]
                node_metrics["executions"] += 1
                node_metrics["successes"] += 1
                node_metrics["avg_execution_time"] = (
                    (node_metrics["avg_execution_time"] * (node_metrics["executions"] - 1) + execution_time) /
                    node_metrics["executions"]
                )
                node_metrics["last_execution"] = datetime.now()
                
            except Exception as e:
                node.status = NodeStatus.FAILED
                node.error_count += 1
                node.last_error = str(e)
                
                # Update metrics
                self.metrics.node_metrics[current_node_id]["failures"] += 1
                
                context.errors.append({
                    "error": str(e),
                    "timestamp": datetime.now(),
                    "node": current_node_id
                })
                
                raise
            
            # Determine next node
            next_node_id = await self.router.route(current_node_id, context.state, context)
            
            if next_node_id:
                # Update edge metrics
                edge = next((e for e in self.get_outgoing_edges(current_node_id) 
                           if e.target_node == next_node_id), None)
                if edge:
                    edge.traversal_count += 1
                    edge.last_traversal = datetime.now()
                    self.metrics.edge_metrics[edge.id]["traversals"] += 1
                    self.metrics.edge_metrics[edge.id]["last_traversal"] = datetime.now()
            
            current_node_id = next_node_id
        
        return context.state
    
    async def _execute_asynchronous(self, context: ExecutionContext) -> Any:
        """Execute graph asynchronously with parallel node execution where possible."""
        # This is a simplified implementation - in practice, you'd need more sophisticated
        # dependency analysis to determine which nodes can run in parallel
        
        pending_nodes = {context.current_node}
        completed_nodes = set()
        futures = {}
        
        while pending_nodes or futures:
            # Start execution for ready nodes
            ready_nodes = pending_nodes.copy()
            pending_nodes.clear()
            
            for node_id in ready_nodes:
                if node_id not in completed_nodes:
                    future = self.thread_pool.submit(self._execute_node, node_id, context)
                    futures[future] = node_id
            
            # Wait for at least one node to complete
            if futures:
                done_futures = []
                for future in list(futures.keys()):
                    if future.done():
                        done_futures.append(future)
                
                if not done_futures:
                    # Wait for the first future to complete
                    import concurrent.futures
                    done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    done_futures = list(done)
                
                for future in done_futures:
                    node_id = futures.pop(future)
                    try:
                        result = future.result()
                        completed_nodes.add(node_id)
                        
                        # Determine next nodes
                        next_nodes = await self._get_next_nodes(node_id, context)
                        pending_nodes.update(next_nodes)
                        
                    except Exception as e:
                        context.errors.append({
                            "error": str(e),
                            "timestamp": datetime.now(),
                            "node": node_id
                        })
        
        return context.state
    
    def _execute_streaming(self, context: ExecutionContext):
        """Execute graph in streaming mode, yielding intermediate results."""
        def stream_generator():
            current_node_id = context.current_node
            
            while current_node_id:
                try:
                    # Execute node
                    node = self.nodes[current_node_id]
                    if node.function:
                        if asyncio.iscoroutinefunction(node.function):
                            # For async functions, we need to run them in the event loop
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                context.state = loop.run_until_complete(node.function(context.state))
                            finally:
                                loop.close()
                        else:
                            context.state = node.function(context.state)
                    
                    # Yield intermediate result
                    yield {
                        "node_id": current_node_id,
                        "state": context.state,
                        "timestamp": datetime.now()
                    }
                    
                    # Get next node
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        next_node_id = loop.run_until_complete(
                            self.router.route(current_node_id, context.state, context)
                        )
                    finally:
                        loop.close()
                    
                    current_node_id = next_node_id
                    
                except Exception as e:
                    yield {
                        "error": str(e),
                        "node_id": current_node_id,
                        "timestamp": datetime.now()
                    }
                    break
        
        return stream_generator()
    
    async def _execute_batch(self, context: ExecutionContext) -> Any:
        """Execute graph in batch mode with checkpointing."""
        batch_size = context.metadata.get("batch_size", 5)
        current_node_id = context.current_node
        batch_count = 0
        
        while current_node_id:
            # Execute node
            await self._execute_node_async(current_node_id, context)
            batch_count += 1
            
            # Create checkpoint every batch_size nodes
            if batch_count % batch_size == 0:
                checkpoint_name = f"batch_{batch_count // batch_size}"
                context.checkpoints[checkpoint_name] = {
                    "state": context.state,
                    "node": current_node_id,
                    "timestamp": datetime.now()
                }
            
            # Get next node
            current_node_id = await self.router.route(current_node_id, context.state, context)
        
        return context.state
    
    def _execute_node(self, node_id: str, context: ExecutionContext) -> Any:
        """Execute a single node (synchronous version for thread pool)."""
        node = self.nodes[node_id]
        
        if node.function:
            if asyncio.iscoroutinefunction(node.function):
                # Run async function in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(node.function(context.state))
                finally:
                    loop.close()
            else:
                return node.function(context.state)
        
        return context.state
    
    async def _execute_node_async(self, node_id: str, context: ExecutionContext):
        """Execute a single node asynchronously."""
        node = self.nodes[node_id]
        context.execution_path.append(node_id)
        
        try:
            start_time = datetime.now()
            node.status = NodeStatus.EXECUTING
            
            if node.function:
                if asyncio.iscoroutinefunction(node.function):
                    context.state = await node.function(context.state)
                else:
                    context.state = node.function(context.state)
            
            # Update state
            await self.state_manager.update_state(context.session_id, node_id, context.state)
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            node.execution_time = execution_time
            node.execution_count += 1
            node.last_execution = datetime.now()
            node.status = NodeStatus.COMPLETED
            
        except Exception as e:
            node.status = NodeStatus.FAILED
            node.error_count += 1
            node.last_error = str(e)
            raise
    
    async def _get_next_nodes(self, current_node_id: str, context: ExecutionContext) -> Set[str]:
        """Get the next nodes to execute based on routing logic."""
        next_node = await self.router.route(current_node_id, context.state, context)
        return {next_node} if next_node else set()
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific execution."""
        if execution_id in self.execution_contexts:
            context = self.execution_contexts[execution_id]
            return {
                "execution_id": execution_id,
                "session_id": context.session_id,
                "mode": context.mode.value,
                "start_time": context.start_time,
                "current_node": context.current_node,
                "execution_path": context.execution_path,
                "errors": context.errors,
                "checkpoints": list(context.checkpoints.keys())
            }
        return None
    
    def get_graph_metrics(self) -> Dict[str, Any]:
        """Get comprehensive graph metrics."""
        return {
            "total_executions": self.metrics.total_executions,
            "successful_executions": self.metrics.successful_executions,
            "failed_executions": self.metrics.failed_executions,
            "success_rate": self.metrics.successful_executions / max(1, self.metrics.total_executions),
            "average_execution_time": self.metrics.average_execution_time,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "active_executions": len(self.execution_contexts),
            "node_metrics": self.metrics.node_metrics,
            "edge_metrics": self.metrics.edge_metrics,
            "last_updated": self.metrics.last_updated
        }

class AdvancedGraphManager:
    """
    Advanced graph manager that integrates with existing LangGraph implementation
    and provides enhanced capabilities for multi-agent coordination.
    """
    
    def __init__(self, memory_service: Optional[MemoryService] = None):
        self.executor = GraphExecutor()
        self.memory_service = memory_service
        self.agent_graphs: Dict[AgentType, Any] = {}  # Store LangGraph instances
        self.session_managers: Dict[str, StateManager] = {}
        self.active_sessions: Dict[str, SessionState] = {}
        self.nlp_processor = None
        
        try:
            self.nlp_processor = get_nlp_processor()
        except Exception as e:
            print(f"NLP processor not available: {e}")
        
        # Initialize default routing rules
        self._setup_default_routing()
        
        print("Advanced Graph Manager initialized")
    
    def _setup_default_routing(self):
        """Setup default routing rules based on common patterns."""
        # Intent-based routing from orchestrator
        if self.nlp_processor:
            self.executor.router.add_intent_based_routing("orchestrator", {
                "question": "research",
                "request": "planning",
                "command": "execution",
                "feedback": "monitoring"
            })
            
            # Entity-based routing for technical requests
            self.executor.router.add_entity_based_routing("planning", {
                "programming_language": "coding",
                "framework": "coding",
                "tool": "execution",
                "file_path": "coding"
            })
        
        # Fallback routes
        self.executor.router.add_fallback_route("orchestrator", "monitoring")
        self.executor.router.add_fallback_route("research", "planning")
        self.executor.router.add_fallback_route("planning", "execution")
        self.executor.router.add_fallback_route("execution", "monitoring")
        self.executor.router.add_fallback_route("coding", "monitoring")
    
    def register_agent_graph(self, agent_type: AgentType, graph: Any):
        """Register a LangGraph instance for an agent type."""
        self.agent_graphs[agent_type] = graph
        
        # Create corresponding graph node
        node = GraphNode(
            id=agent_type.value,
            name=f"{agent_type.value.title()} Agent",
            node_type=GraphNodeType.AGENT,
            agent_type=agent_type,
            function=self._create_agent_function(agent_type)
        )
        
        self.executor.add_node(node)
    
    def _create_agent_function(self, agent_type: AgentType) -> Callable:
        """Create a function that executes the agent's graph."""
        async def agent_function(state: Any) -> Any:
            if agent_type in self.agent_graphs:
                graph = self.agent_graphs[agent_type]
                
                # Execute the agent's graph
                if hasattr(graph, 'ainvoke'):
                    result = await graph.ainvoke(state)
                elif hasattr(graph, 'invoke'):
                    result = graph.invoke(state)
                else:
                    # Fallback for custom graph implementations
                    result = await graph(state)
                
                # Store result in memory if available
                if self.memory_service:
                    try:
                        await self.memory_service.store_conversation_memory(
                            session_id=getattr(state, 'session_id', 'default'),
                            user_message=getattr(state, 'user_input', ''),
                            assistant_message=str(result),
                            metadata={
                                "agent_type": agent_type.value,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                    except Exception as e:
                        print(f"Failed to store conversation memory: {e}")
                
                return result
            else:
                print(f"No graph registered for agent type: {agent_type}")
                return state
        
        return agent_function
    
    async def start_session(self, session_id: str, initial_state: Any) -> str:
        """Start a new graph execution session."""
        self.active_sessions[session_id] = SessionState.ACTIVE
        self.session_managers[session_id] = StateManager()
        
        # Store initial state
        await self.session_managers[session_id].update_state(session_id, "start", initial_state)
        
        return session_id
    
    async def execute_session(self, session_id: str, 
                             start_agent: AgentType = AgentType.ORCHESTRATOR,
                             mode: ExecutionMode = ExecutionMode.SYNCHRONOUS) -> Any:
        """Execute a session starting from a specific agent."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        if self.active_sessions[session_id] != SessionState.ACTIVE:
            raise ValueError(f"Session {session_id} is not active")
        
        try:
            # Get current state
            current_state = self.session_managers[session_id].get_current_state(session_id)
            
            # Execute graph
            result = await self.executor.execute_graph(
                session_id=session_id,
                initial_state=current_state,
                start_node=start_agent.value,
                mode=mode
            )
            
            self.active_sessions[session_id] = SessionState.COMPLETED
            return result
            
        except Exception as e:
            self.active_sessions[session_id] = SessionState.ERROR
            raise
    
    async def pause_session(self, session_id: str):
        """Pause a session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id] = SessionState.PAUSED
    
    async def resume_session(self, session_id: str):
        """Resume a paused session."""
        if session_id in self.active_sessions and self.active_sessions[session_id] == SessionState.PAUSED:
            self.active_sessions[session_id] = SessionState.ACTIVE
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a session."""
        if session_id in self.active_sessions:
            state_manager = self.session_managers.get(session_id)
            return {
                "session_id": session_id,
                "status": self.active_sessions[session_id].value,
                "state_history_length": len(state_manager.get_state_history(session_id)) if state_manager else 0,
                "snapshots": list(state_manager.state_snapshots[session_id].keys()) if state_manager else []
            }
        return None
    
    def add_custom_routing_rule(self, from_agent: AgentType, to_agent: AgentType, 
                               condition: Callable[[Any], bool]):
        """Add a custom routing rule between agents."""
        def routing_rule(state) -> Optional[str]:
            try:
                if condition(state):
                    return to_agent.value
            except Exception as e:
                print(f"Custom routing rule failed: {e}")
            return None
        
        self.executor.router.add_routing_rule(from_agent.value, routing_rule)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics for the graph manager."""
        return {
            "graph_metrics": self.executor.get_graph_metrics(),
            "active_sessions": len([s for s in self.active_sessions.values() if s == SessionState.ACTIVE]),
            "total_sessions": len(self.active_sessions),
            "registered_agents": list(self.agent_graphs.keys()),
            "session_states": {sid: state.value for sid, state in self.active_sessions.items()},
            "nlp_available": self.nlp_processor is not None,
            "memory_service_available": self.memory_service is not None
        }

# Global instance
_graph_manager = None

def get_graph_manager(memory_service: Optional[MemoryService] = None) -> AdvancedGraphManager:
    """Get the global graph manager instance."""
    global _graph_manager
    if _graph_manager is None:
        _graph_manager = AdvancedGraphManager(memory_service)
    return _graph_manager

def initialize_graph_manager(memory_service: Optional[MemoryService] = None) -> AdvancedGraphManager:
    """Initialize and return the graph manager."""
    global _graph_manager
    _graph_manager = AdvancedGraphManager(memory_service)
    return _graph_manager