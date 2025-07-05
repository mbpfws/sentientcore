import asyncio
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import threading
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field, validator
from ..models import AppState, TaskStatus, AgentType, MemoryLayer, SessionState

class StateChangeType(Enum):
    """Types of state changes for tracking and rollback."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    ROLLBACK = "rollback"

class StatePersistenceMode(Enum):
    """State persistence modes."""
    NONE = "none"  # No persistence
    MEMORY = "memory"  # In-memory only
    FILE = "file"  # File-based persistence
    DATABASE = "database"  # Database persistence
    HYBRID = "hybrid"  # Memory + File backup

class StateValidationLevel(Enum):
    """State validation levels."""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"

@dataclass
class StateChange:
    """Represents a state change for history tracking."""
    id: str
    timestamp: datetime
    change_type: StateChangeType
    path: str  # JSON path to the changed field
    old_value: Any
    new_value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class StateSnapshot:
    """Represents a complete state snapshot."""
    id: str
    timestamp: datetime
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    
    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for state integrity verification."""
        state_str = json.dumps(self.state_data, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify state integrity using checksum."""
        return self.checksum == self._calculate_checksum()

class StateValidator:
    """Validates state changes and integrity."""
    
    def __init__(self, validation_level: StateValidationLevel = StateValidationLevel.BASIC):
        self.validation_level = validation_level
        self.custom_validators: Dict[str, Callable] = {}
        self.schema_validators: Dict[str, Type[BaseModel]] = {}
    
    def register_validator(self, path: str, validator: Callable[[Any], bool]):
        """Register a custom validator for a specific state path."""
        self.custom_validators[path] = validator
    
    def register_schema_validator(self, path: str, schema: Type[BaseModel]):
        """Register a Pydantic schema validator for a specific state path."""
        self.schema_validators[path] = schema
    
    def validate_change(self, path: str, old_value: Any, new_value: Any) -> tuple[bool, Optional[str]]:
        """Validate a state change."""
        if self.validation_level == StateValidationLevel.NONE:
            return True, None
        
        # Basic validation
        if self.validation_level in [StateValidationLevel.BASIC, StateValidationLevel.STRICT]:
            # Type consistency check
            if old_value is not None and new_value is not None:
                if type(old_value) != type(new_value) and not self._is_compatible_type(old_value, new_value):
                    return False, f"Type mismatch: {type(old_value)} -> {type(new_value)}"
        
        # Custom validation
        if path in self.custom_validators:
            try:
                if not self.custom_validators[path](new_value):
                    return False, f"Custom validation failed for path: {path}"
            except Exception as e:
                return False, f"Custom validator error: {str(e)}"
        
        # Schema validation
        if path in self.schema_validators:
            try:
                self.schema_validators[path](**new_value if isinstance(new_value, dict) else {"value": new_value})
            except Exception as e:
                return False, f"Schema validation failed: {str(e)}"
        
        return True, None
    
    def _is_compatible_type(self, old_value: Any, new_value: Any) -> bool:
        """Check if types are compatible for state updates."""
        # Allow None to any type
        if old_value is None or new_value is None:
            return True
        
        # Allow numeric conversions
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            return True
        
        # Allow string to any serializable type
        if isinstance(old_value, str) or isinstance(new_value, str):
            return True
        
        return False

class StateEventBus:
    """Event bus for state change notifications."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.global_subscribers: List[Callable] = []
    
    def subscribe(self, path: str, callback: Callable[[StateChange], None]):
        """Subscribe to state changes at a specific path."""
        self.subscribers[path].append(callback)
    
    def subscribe_global(self, callback: Callable[[StateChange], None]):
        """Subscribe to all state changes."""
        self.global_subscribers.append(callback)
    
    def unsubscribe(self, path: str, callback: Callable):
        """Unsubscribe from state changes."""
        if path in self.subscribers and callback in self.subscribers[path]:
            self.subscribers[path].remove(callback)
    
    def unsubscribe_global(self, callback: Callable):
        """Unsubscribe from global state changes."""
        if callback in self.global_subscribers:
            self.global_subscribers.remove(callback)
    
    async def emit(self, change: StateChange):
        """Emit a state change event."""
        # Notify path-specific subscribers
        for subscriber in self.subscribers.get(change.path, []):
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(change)
                else:
                    subscriber(change)
            except Exception as e:
                print(f"Error in state change subscriber: {e}")
        
        # Notify global subscribers
        for subscriber in self.global_subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(change)
                else:
                    subscriber(change)
            except Exception as e:
                print(f"Error in global state change subscriber: {e}")

class StatePersistenceManager:
    """Manages state persistence across different storage backends."""
    
    def __init__(self, mode: StatePersistenceMode = StatePersistenceMode.HYBRID,
                 storage_path: Optional[Path] = None):
        self.mode = mode
        self.storage_path = storage_path or Path("./state_storage")
        self.storage_path.mkdir(exist_ok=True)
        
        # In-memory cache for hybrid mode
        self.memory_cache: Dict[str, StateSnapshot] = {}
        self.cache_ttl = timedelta(hours=1)
    
    async def save_snapshot(self, snapshot: StateSnapshot) -> bool:
        """Save a state snapshot."""
        try:
            if self.mode in [StatePersistenceMode.MEMORY, StatePersistenceMode.HYBRID]:
                self.memory_cache[snapshot.id] = snapshot
            
            if self.mode in [StatePersistenceMode.FILE, StatePersistenceMode.HYBRID]:
                file_path = self.storage_path / f"{snapshot.id}.json"
                with open(file_path, 'w') as f:
                    json.dump({
                        'id': snapshot.id,
                        'timestamp': snapshot.timestamp.isoformat(),
                        'state_data': snapshot.state_data,
                        'metadata': snapshot.metadata,
                        'checksum': snapshot.checksum
                    }, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error saving state snapshot: {e}")
            return False
    
    async def load_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Load a state snapshot."""
        try:
            # Try memory cache first
            if self.mode in [StatePersistenceMode.MEMORY, StatePersistenceMode.HYBRID]:
                if snapshot_id in self.memory_cache:
                    snapshot = self.memory_cache[snapshot_id]
                    # Check TTL for hybrid mode
                    if self.mode == StatePersistenceMode.HYBRID:
                        if datetime.now() - snapshot.timestamp > self.cache_ttl:
                            del self.memory_cache[snapshot_id]
                        else:
                            return snapshot
                    else:
                        return snapshot
            
            # Try file storage
            if self.mode in [StatePersistenceMode.FILE, StatePersistenceMode.HYBRID]:
                file_path = self.storage_path / f"{snapshot_id}.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    snapshot = StateSnapshot(
                        id=data['id'],
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        state_data=data['state_data'],
                        metadata=data['metadata'],
                        checksum=data['checksum']
                    )
                    
                    # Verify integrity
                    if snapshot.verify_integrity():
                        # Cache in memory for hybrid mode
                        if self.mode == StatePersistenceMode.HYBRID:
                            self.memory_cache[snapshot_id] = snapshot
                        return snapshot
                    else:
                        print(f"State snapshot {snapshot_id} failed integrity check")
            
            return None
        except Exception as e:
            print(f"Error loading state snapshot: {e}")
            return None
    
    async def list_snapshots(self) -> List[str]:
        """List available snapshots."""
        snapshots = set()
        
        # From memory cache
        if self.mode in [StatePersistenceMode.MEMORY, StatePersistenceMode.HYBRID]:
            snapshots.update(self.memory_cache.keys())
        
        # From file storage
        if self.mode in [StatePersistenceMode.FILE, StatePersistenceMode.HYBRID]:
            for file_path in self.storage_path.glob("*.json"):
                snapshots.add(file_path.stem)
        
        return list(snapshots)
    
    async def cleanup_old_snapshots(self, max_age: timedelta = timedelta(days=7)):
        """Clean up old snapshots."""
        cutoff_time = datetime.now() - max_age
        
        # Clean memory cache
        if self.mode in [StatePersistenceMode.MEMORY, StatePersistenceMode.HYBRID]:
            to_remove = []
            for snapshot_id, snapshot in self.memory_cache.items():
                if snapshot.timestamp < cutoff_time:
                    to_remove.append(snapshot_id)
            
            for snapshot_id in to_remove:
                del self.memory_cache[snapshot_id]
        
        # Clean file storage
        if self.mode in [StatePersistenceMode.FILE, StatePersistenceMode.HYBRID]:
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if timestamp < cutoff_time:
                        file_path.unlink()
                except Exception as e:
                    print(f"Error cleaning up snapshot {file_path}: {e}")

class EnhancedStateManager:
    """
    Advanced state management system with:
    - Hierarchical state organization
    - Change tracking and rollback
    - State validation and integrity
    - Event-driven notifications
    - Persistence and recovery
    - Concurrent access control
    - Performance optimization
    """
    
    def __init__(self, 
                 persistence_mode: StatePersistenceMode = StatePersistenceMode.HYBRID,
                 validation_level: StateValidationLevel = StateValidationLevel.BASIC,
                 storage_path: Optional[Path] = None,
                 max_history_size: int = 1000):
        
        # Core state storage
        self.current_state: AppState = AppState()
        self.state_lock = asyncio.Lock()
        
        # Change tracking
        self.change_history: deque[StateChange] = deque(maxlen=max_history_size)
        self.snapshots: Dict[str, StateSnapshot] = {}
        
        # Components
        self.validator = StateValidator(validation_level)
        self.event_bus = StateEventBus()
        self.persistence_manager = StatePersistenceManager(persistence_mode, storage_path)
        
        # Configuration
        self.auto_snapshot_interval = timedelta(minutes=5)
        self.last_snapshot_time = datetime.now()
        
        # Performance tracking
        self.operation_stats = defaultdict(int)
        self.performance_metrics = defaultdict(list)
        
        print(f"Enhanced State Manager initialized with {persistence_mode.value} persistence")
    
    async def get_state(self) -> AppState:
        """Get the current state (thread-safe)."""
        async with self.state_lock:
            return self.current_state.copy(deep=True)
    
    async def update_state(self, 
                          path: str, 
                          value: Any, 
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a specific path in the state."""
        start_time = datetime.now()
        
        async with self.state_lock:
            try:
                # Get current value
                old_value = self._get_nested_value(self.current_state.dict(), path)
                
                # Validate change
                is_valid, error_msg = self.validator.validate_change(path, old_value, value)
                if not is_valid:
                    print(f"State validation failed: {error_msg}")
                    return False
                
                # Apply change
                self._set_nested_value(self.current_state, path, value)
                
                # Create change record
                change = StateChange(
                    id=f"change_{len(self.change_history)}_{int(start_time.timestamp())}",
                    timestamp=start_time,
                    change_type=StateChangeType.UPDATE,
                    path=path,
                    old_value=old_value,
                    new_value=value,
                    metadata=metadata or {},
                    user_id=user_id,
                    session_id=session_id
                )
                
                # Add to history
                self.change_history.append(change)
                
                # Emit event
                await self.event_bus.emit(change)
                
                # Auto-snapshot if needed
                await self._check_auto_snapshot()
                
                # Track performance
                operation_time = (datetime.now() - start_time).total_seconds()
                self.operation_stats['update'] += 1
                self.performance_metrics['update_time'].append(operation_time)
                
                return True
                
            except Exception as e:
                print(f"Error updating state: {e}")
                return False
    
    async def merge_state(self, 
                         state_dict: Dict[str, Any],
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None) -> bool:
        """Merge a dictionary into the current state."""
        start_time = datetime.now()
        
        async with self.state_lock:
            try:
                # Create a deep copy for rollback
                old_state_dict = self.current_state.dict()
                
                # Merge the state
                merged_dict = self._deep_merge(old_state_dict, state_dict)
                
                # Validate the merged state
                try:
                    new_state = AppState(**merged_dict)
                except Exception as e:
                    print(f"Invalid merged state: {e}")
                    return False
                
                # Apply the new state
                self.current_state = new_state
                
                # Create change record
                change = StateChange(
                    id=f"merge_{len(self.change_history)}_{int(start_time.timestamp())}",
                    timestamp=start_time,
                    change_type=StateChangeType.MERGE,
                    path="root",
                    old_value=old_state_dict,
                    new_value=merged_dict,
                    user_id=user_id,
                    session_id=session_id
                )
                
                self.change_history.append(change)
                await self.event_bus.emit(change)
                await self._check_auto_snapshot()
                
                # Track performance
                operation_time = (datetime.now() - start_time).total_seconds()
                self.operation_stats['merge'] += 1
                self.performance_metrics['merge_time'].append(operation_time)
                
                return True
                
            except Exception as e:
                print(f"Error merging state: {e}")
                return False
    
    async def create_snapshot(self, 
                             snapshot_id: Optional[str] = None,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a state snapshot."""
        async with self.state_lock:
            if snapshot_id is None:
                snapshot_id = f"snapshot_{int(datetime.now().timestamp())}"
            
            snapshot = StateSnapshot(
                id=snapshot_id,
                timestamp=datetime.now(),
                state_data=self.current_state.dict(),
                metadata=metadata or {}
            )
            
            # Save snapshot
            self.snapshots[snapshot_id] = snapshot
            await self.persistence_manager.save_snapshot(snapshot)
            
            self.last_snapshot_time = datetime.now()
            self.operation_stats['snapshot'] += 1
            
            return snapshot_id
    
    async def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore state from a snapshot."""
        async with self.state_lock:
            try:
                # Try to load from memory first
                snapshot = self.snapshots.get(snapshot_id)
                
                # If not in memory, try persistence
                if snapshot is None:
                    snapshot = await self.persistence_manager.load_snapshot(snapshot_id)
                
                if snapshot is None:
                    print(f"Snapshot {snapshot_id} not found")
                    return False
                
                # Verify integrity
                if not snapshot.verify_integrity():
                    print(f"Snapshot {snapshot_id} failed integrity check")
                    return False
                
                # Create rollback change record
                old_state_dict = self.current_state.dict()
                
                # Restore state
                self.current_state = AppState(**snapshot.state_data)
                
                # Record the rollback
                change = StateChange(
                    id=f"rollback_{len(self.change_history)}_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    change_type=StateChangeType.ROLLBACK,
                    path="root",
                    old_value=old_state_dict,
                    new_value=snapshot.state_data,
                    metadata={"snapshot_id": snapshot_id}
                )
                
                self.change_history.append(change)
                await self.event_bus.emit(change)
                
                self.operation_stats['restore'] += 1
                
                return True
                
            except Exception as e:
                print(f"Error restoring snapshot: {e}")
                return False
    
    async def rollback_changes(self, steps: int = 1) -> bool:
        """Rollback the last N changes."""
        async with self.state_lock:
            if len(self.change_history) < steps:
                print(f"Cannot rollback {steps} steps, only {len(self.change_history)} changes available")
                return False
            
            try:
                # Get changes to rollback (in reverse order)
                changes_to_rollback = list(self.change_history)[-steps:]
                changes_to_rollback.reverse()
                
                # Apply rollbacks
                for change in changes_to_rollback:
                    if change.change_type == StateChangeType.UPDATE:
                        self._set_nested_value(self.current_state, change.path, change.old_value)
                    elif change.change_type == StateChangeType.MERGE:
                        self.current_state = AppState(**change.old_value)
                
                # Remove rolled back changes from history
                for _ in range(steps):
                    self.change_history.pop()
                
                # Create rollback record
                rollback_change = StateChange(
                    id=f"rollback_{len(self.change_history)}_{int(datetime.now().timestamp())}",
                    timestamp=datetime.now(),
                    change_type=StateChangeType.ROLLBACK,
                    path="multiple",
                    old_value=None,
                    new_value=None,
                    metadata={"steps_rolled_back": steps}
                )
                
                self.change_history.append(rollback_change)
                await self.event_bus.emit(rollback_change)
                
                self.operation_stats['rollback'] += 1
                
                return True
                
            except Exception as e:
                print(f"Error rolling back changes: {e}")
                return False
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a nested value using dot notation path."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, obj: Any, path: str, value: Any):
        """Set a nested value using dot notation path."""
        keys = path.split('.')
        current = obj
        
        for key in keys[:-1]:
            if hasattr(current, key):
                current = getattr(current, key)
            else:
                raise ValueError(f"Invalid path: {path}")
        
        setattr(current, keys[-1], value)
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    async def _check_auto_snapshot(self):
        """Check if an automatic snapshot should be created."""
        if datetime.now() - self.last_snapshot_time > self.auto_snapshot_interval:
            await self.create_snapshot(
                snapshot_id=f"auto_{int(datetime.now().timestamp())}",
                metadata={"type": "automatic"}
            )
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get state management analytics."""
        return {
            "operation_stats": dict(self.operation_stats),
            "change_history_size": len(self.change_history),
            "snapshots_count": len(self.snapshots),
            "current_state_size": len(str(self.current_state.dict())),
            "performance_metrics": {
                metric: {
                    "count": len(times),
                    "avg_time": sum(times) / len(times) if times else 0,
                    "max_time": max(times) if times else 0,
                    "min_time": min(times) if times else 0
                }
                for metric, times in self.performance_metrics.items()
            },
            "persistence_mode": self.persistence_manager.mode.value,
            "validation_level": self.validator.validation_level.value
        }

# Global instance
_enhanced_state_manager = None

def get_enhanced_state_manager() -> EnhancedStateManager:
    """Get the global enhanced state manager instance."""
    global _enhanced_state_manager
    if _enhanced_state_manager is None:
        _enhanced_state_manager = EnhancedStateManager()
    return _enhanced_state_manager

def initialize_enhanced_state_manager(**kwargs) -> EnhancedStateManager:
    """Initialize and return the enhanced state manager."""
    global _enhanced_state_manager
    _enhanced_state_manager = EnhancedStateManager(**kwargs)
    return _enhanced_state_manager