from typing import Dict, List, Optional, Any, Union, Callable
import json
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import gzip
from contextlib import asynccontextmanager


class PersistenceMode(Enum):
    """State persistence modes"""
    MEMORY_ONLY = "memory_only"  # No persistence
    IMMEDIATE = "immediate"  # Save immediately on change
    BATCH = "batch"  # Save in batches
    MANUAL = "manual"  # Manual save only


class StateChangeType(Enum):
    """Types of state changes for tracking"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    RESET = "reset"


@dataclass
class StateChange:
    """Represents a single state change event"""
    id: str
    session_id: str
    change_type: StateChangeType
    path: str  # JSON path to the changed data
    old_value: Any
    new_value: Any
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'session_id': self.session_id,
            'change_type': self.change_type.value,
            'path': self.path,
            'old_value': json.dumps(self.old_value) if self.old_value is not None else None,
            'new_value': json.dumps(self.new_value) if self.new_value is not None else None,
            'timestamp': self.timestamp.isoformat(),
            'metadata': json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateChange':
        return cls(
            id=data['id'],
            session_id=data['session_id'],
            change_type=StateChangeType(data['change_type']),
            path=data['path'],
            old_value=json.loads(data['old_value']) if data['old_value'] else None,
            new_value=json.loads(data['new_value']) if data['new_value'] else None,
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=json.loads(data['metadata'])
        )


@dataclass
class SessionState:
    """Complete session state container"""
    session_id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'data': self.data,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        return cls(
            session_id=data['session_id'],
            data=data['data'],
            metadata=data['metadata'],
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            version=data.get('version', 1)
        )


class EnhancedStateManager:
    """Enhanced state management with hybrid persistence and change tracking"""
    
    def __init__(
        self, 
        db_path: str = "app/state_management.db",
        storage_path: str = "app/state_storage",
        persistence_mode: PersistenceMode = PersistenceMode.IMMEDIATE,
        max_history_per_session: int = 1000
    ):
        self.db_path = Path(db_path)
        self.storage_path = Path(storage_path)
        self.persistence_mode = persistence_mode
        self.max_history_per_session = max_history_per_session
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory state cache
        self._state_cache: Dict[str, SessionState] = {}
        self._change_listeners: List[Callable[[StateChange], None]] = []
        self._batch_changes: List[StateChange] = []
        self._batch_lock = asyncio.Lock()
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for state management"""
        with sqlite3.connect(self.db_path) as conn:
            # Session states table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_states (
                    session_id TEXT PRIMARY KEY,
                    data_path TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    compressed BOOLEAN DEFAULT 1
                )
            """)
            
            # State changes history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_changes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES session_states (session_id)
                )
            """)
            
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_changes ON state_changes(session_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_change_type ON state_changes(change_type)
            """)
            
            conn.commit()
    
    async def create_session(self, session_id: str, initial_data: Optional[Dict[str, Any]] = None) -> SessionState:
        """Create a new session state"""
        now = datetime.now()
        
        session_state = SessionState(
            session_id=session_id,
            data=initial_data or {},
            metadata={"created_by": "EnhancedStateManager"},
            created_at=now,
            updated_at=now
        )
        
        # Store in cache
        self._state_cache[session_id] = session_state
        
        # Persist if needed
        if self.persistence_mode != PersistenceMode.MEMORY_ONLY:
            await self._persist_session_state(session_state)
        
        # Track creation
        await self._track_change(
            session_id=session_id,
            change_type=StateChangeType.CREATE,
            path="/",
            old_value=None,
            new_value=session_state.data,
            metadata={"action": "session_created"}
        )
        
        return session_state
    
    async def get_session_state(self, session_id: str) -> Optional[SessionState]:
        """Get session state by ID"""
        # Check cache first
        if session_id in self._state_cache:
            return self._state_cache[session_id]
        
        # Load from storage
        session_state = await self._load_session_state(session_id)
        if session_state:
            self._state_cache[session_id] = session_state
        
        return session_state
    
    async def update_session_data(
        self, 
        session_id: str, 
        path: str, 
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update specific data in session state"""
        session_state = await self.get_session_state(session_id)
        if not session_state:
            return False
        
        # Get old value for change tracking
        old_value = self._get_nested_value(session_state.data, path)
        
        # Update the data
        self._set_nested_value(session_state.data, path, value)
        session_state.updated_at = datetime.now()
        session_state.version += 1
        
        # Track the change
        await self._track_change(
            session_id=session_id,
            change_type=StateChangeType.UPDATE,
            path=path,
            old_value=old_value,
            new_value=value,
            metadata=metadata or {}
        )
        
        # Persist if needed
        if self.persistence_mode == PersistenceMode.IMMEDIATE:
            await self._persist_session_state(session_state)
        
        return True
    
    async def merge_session_data(
        self, 
        session_id: str, 
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Merge data into session state"""
        session_state = await self.get_session_state(session_id)
        if not session_state:
            return False
        
        old_data = session_state.data.copy()
        
        # Deep merge the data
        self._deep_merge(session_state.data, data)
        session_state.updated_at = datetime.now()
        session_state.version += 1
        
        # Track the change
        await self._track_change(
            session_id=session_id,
            change_type=StateChangeType.MERGE,
            path="/",
            old_value=old_data,
            new_value=session_state.data,
            metadata=metadata or {}
        )
        
        # Persist if needed
        if self.persistence_mode == PersistenceMode.IMMEDIATE:
            await self._persist_session_state(session_state)
        
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and its history"""
        session_state = await self.get_session_state(session_id)
        if not session_state:
            return False
        
        # Track deletion
        await self._track_change(
            session_id=session_id,
            change_type=StateChangeType.DELETE,
            path="/",
            old_value=session_state.data,
            new_value=None,
            metadata={"action": "session_deleted"}
        )
        
        # Remove from cache
        self._state_cache.pop(session_id, None)
        
        # Remove from storage
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM state_changes WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM session_states WHERE session_id = ?", (session_id,))
            conn.commit()
        
        # Remove state file
        state_file = self.storage_path / f"{session_id}.state"
        if state_file.exists():
            state_file.unlink()
        
        return True
    
    async def get_session_history(
        self, 
        session_id: str, 
        limit: int = 100,
        change_type: Optional[StateChangeType] = None
    ) -> List[StateChange]:
        """Get change history for a session"""
        query = "SELECT * FROM state_changes WHERE session_id = ?"
        params = [session_id]
        
        if change_type:
            query += " AND change_type = ?"
            params.append(change_type.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        changes = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                changes.append(StateChange.from_dict(dict(row)))
        
        return changes
    
    async def revert_to_version(
        self, 
        session_id: str, 
        target_version: int
    ) -> bool:
        """Revert session state to a specific version"""
        # Get all changes since target version
        changes = await self.get_session_history(session_id, limit=1000)
        
        session_state = await self.get_session_state(session_id)
        if not session_state:
            return False
        
        # Find changes to revert
        changes_to_revert = [
            change for change in changes 
            if change.timestamp > session_state.created_at
        ]
        
        # Apply reverse changes
        for change in reversed(changes_to_revert):
            if change.change_type in [StateChangeType.UPDATE, StateChangeType.CREATE]:
                if change.old_value is not None:
                    self._set_nested_value(session_state.data, change.path, change.old_value)
                else:
                    self._delete_nested_value(session_state.data, change.path)
        
        session_state.version = target_version
        session_state.updated_at = datetime.now()
        
        # Persist the reverted state
        await self._persist_session_state(session_state)
        
        return True
    
    def add_change_listener(self, listener: Callable[[StateChange], None]):
        """Add a listener for state changes"""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[StateChange], None]):
        """Remove a change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    async def save_all_sessions(self) -> int:
        """Manually save all cached sessions"""
        saved_count = 0
        for session_state in self._state_cache.values():
            await self._persist_session_state(session_state)
            saved_count += 1
        return saved_count
    
    async def _persist_session_state(self, session_state: SessionState):
        """Persist session state to storage"""
        # Save data to compressed file
        state_file = self.storage_path / f"{session_state.session_id}.state"
        
        with gzip.open(state_file, 'wb') as f:
            pickle.dump(session_state.data, f)
        
        # Update database record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO session_states 
                (session_id, data_path, metadata, created_at, updated_at, version, compressed)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_state.session_id,
                str(state_file),
                json.dumps(session_state.metadata),
                session_state.created_at.isoformat(),
                session_state.updated_at.isoformat(),
                session_state.version,
                True
            ))
            conn.commit()
    
    async def _load_session_state(self, session_id: str) -> Optional[SessionState]:
        """Load session state from storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM session_states WHERE session_id = ?", 
                (session_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Load data from file
            state_file = Path(row['data_path'])
            if not state_file.exists():
                return None
            
            try:
                if row['compressed']:
                    with gzip.open(state_file, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(state_file, 'rb') as f:
                        data = pickle.load(f)
                
                return SessionState(
                    session_id=row['session_id'],
                    data=data,
                    metadata=json.loads(row['metadata']),
                    created_at=datetime.fromisoformat(row['created_at']),
                    updated_at=datetime.fromisoformat(row['updated_at']),
                    version=row['version']
                )
            except Exception:
                return None
    
    async def _track_change(
        self,
        session_id: str,
        change_type: StateChangeType,
        path: str,
        old_value: Any,
        new_value: Any,
        metadata: Dict[str, Any]
    ):
        """Track a state change"""
        change = StateChange(
            id=str(uuid.uuid4()),
            session_id=session_id,
            change_type=change_type,
            path=path,
            old_value=old_value,
            new_value=new_value,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        # Notify listeners
        for listener in self._change_listeners:
            try:
                listener(change)
            except Exception:
                pass  # Don't let listener errors break the system
        
        # Store change based on persistence mode
        if self.persistence_mode == PersistenceMode.IMMEDIATE:
            await self._persist_change(change)
        elif self.persistence_mode == PersistenceMode.BATCH:
            async with self._batch_lock:
                self._batch_changes.append(change)
        
        # Cleanup old changes if needed
        await self._cleanup_old_changes(session_id)
    
    async def _persist_change(self, change: StateChange):
        """Persist a single change to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO state_changes 
                (id, session_id, change_type, path, old_value, new_value, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                change.id,
                change.session_id,
                change.change_type.value,
                change.path,
                json.dumps(change.old_value) if change.old_value is not None else None,
                json.dumps(change.new_value) if change.new_value is not None else None,
                change.timestamp.isoformat(),
                json.dumps(change.metadata)
            ))
            conn.commit()
    
    async def _cleanup_old_changes(self, session_id: str):
        """Clean up old changes to maintain history limits"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM state_changes WHERE session_id = ?",
                (session_id,)
            )
            count = cursor.fetchone()[0]
            
            if count > self.max_history_per_session:
                # Delete oldest changes
                excess = count - self.max_history_per_session
                conn.execute("""
                    DELETE FROM state_changes 
                    WHERE session_id = ? 
                    AND id IN (
                        SELECT id FROM state_changes 
                        WHERE session_id = ?
                        ORDER BY timestamp ASC 
                        LIMIT ?
                    )
                """, (session_id, session_id, excess))
                conn.commit()
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using path"""
        if path == "/":
            return data
        
        keys = path.strip("/").split("/")
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set value in nested dictionary using path"""
        if path == "/":
            data.clear()
            if isinstance(value, dict):
                data.update(value)
            return
        
        keys = path.strip("/").split("/")
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _delete_nested_value(self, data: Dict[str, Any], path: str):
        """Delete value from nested dictionary using path"""
        if path == "/":
            data.clear()
            return
        
        keys = path.strip("/").split("/")
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                return
            current = current[key]
        
        current.pop(keys[-1], None)
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge source dictionary into target"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value