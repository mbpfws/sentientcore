"""Session Persistence Service for Build 2

This service implements the critical persistence layer that ensures conversation history,
monitoring logs, and system state survive browser refreshes and application restarts.
"""

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

from core.models import AppState, Message, LogEntry
from core.state.enhanced_state_manager import EnhancedStateManager, StatePersistenceMode


class SessionPersistenceService:
    """
    Manages persistent storage of user sessions, conversation history, and system state.
    Implements multiple storage backends for reliability and performance.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/sessions")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for session metadata
        self.db_path = self.storage_path / "sessions.db"
        self._init_database()
        
        # Enhanced state manager for advanced persistence
        self.state_manager = EnhancedStateManager(
            persistence_mode=StatePersistenceMode.HYBRID,
            storage_path=self.storage_path / "state_snapshots"
        )
        
        # In-memory cache for active sessions
        self._session_cache: Dict[str, AppState] = {}
        self._cache_lock = asyncio.Lock()
        
        print(f"Session Persistence Service initialized at {self.storage_path}")
    
    def _init_database(self):
        """Initialize SQLite database for session metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    state_snapshot_id TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    message_index INTEGER,
                    sender TEXT,
                    content TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_data BLOB,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    log_index INTEGER,
                    source TEXT,
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            conn.commit()
    
    async def save_session(self, session_id: str, state: AppState) -> bool:
        """
        Save complete session state with conversation history and logs.
        """
        try:
            async with self._cache_lock:
                # Update cache
                self._session_cache[session_id] = state.copy(deep=True)
            
            # Create state snapshot
            snapshot_id = await self.state_manager.create_snapshot(
                snapshot_id=f"session_{session_id}_{int(datetime.now().timestamp())}",
                metadata={"session_id": session_id, "type": "session_save"}
            )
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                # Update session metadata
                conn.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, last_accessed, message_count, state_snapshot_id, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    datetime.now(),
                    len(state.messages),
                    snapshot_id,
                    json.dumps({"language": state.language, "user_prompt": state.user_prompt})
                ))
                
                # Clear existing conversation history for this session
                conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
                
                # Save conversation history
                for idx, message in enumerate(state.messages):
                    conn.execute("""
                        INSERT INTO conversation_history 
                        (session_id, message_index, sender, content, image_data)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        idx,
                        message.sender,
                        message.content,
                        message.image
                    ))
                
                # Clear existing logs for this session
                conn.execute("DELETE FROM system_logs WHERE session_id = ?", (session_id,))
                
                # Save system logs
                for idx, log in enumerate(state.logs):
                    conn.execute("""
                        INSERT INTO system_logs 
                        (session_id, log_index, source, message)
                        VALUES (?, ?, ?, ?)
                    """, (
                        session_id,
                        idx,
                        log.source,
                        log.message
                    ))
                
                conn.commit()
            
            print(f"Session {session_id} saved successfully with {len(state.messages)} messages")
            return True
            
        except Exception as e:
            print(f"Error saving session {session_id}: {e}")
            return False
    
    async def load_session(self, session_id: str) -> Optional[AppState]:
        """
        Load complete session state from persistent storage.
        """
        try:
            # Check cache first
            async with self._cache_lock:
                if session_id in self._session_cache:
                    print(f"Loading session {session_id} from cache")
                    return self._session_cache[session_id].copy(deep=True)
            
            # Load from database
            with sqlite3.connect(self.db_path) as conn:
                # Get session metadata
                session_row = conn.execute(
                    "SELECT state_snapshot_id, metadata FROM sessions WHERE session_id = ?",
                    (session_id,)
                ).fetchone()
                
                if not session_row:
                    print(f"Session {session_id} not found")
                    return None
                
                snapshot_id, metadata_json = session_row
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Load conversation history
                message_rows = conn.execute("""
                    SELECT sender, content, image_data 
                    FROM conversation_history 
                    WHERE session_id = ? 
                    ORDER BY message_index
                """, (session_id,)).fetchall()
                
                messages = [
                    Message(
                        sender=row[0],
                        content=row[1],
                        image=row[2] if row[2] else None
                    )
                    for row in message_rows
                ]
                
                # Load system logs
                log_rows = conn.execute("""
                    SELECT source, message 
                    FROM system_logs 
                    WHERE session_id = ? 
                    ORDER BY log_index
                """, (session_id,)).fetchall()
                
                logs = [
                    LogEntry(source=row[0], message=row[1])
                    for row in log_rows
                ]
                
                # Create AppState
                state = AppState(
                    messages=messages,
                    logs=logs,
                    user_prompt=metadata.get("user_prompt", ""),
                    language=metadata.get("language", "en"),
                    conversation_history=[f"{msg.sender}: {msg.content}" for msg in messages]
                )
                
                # Update cache
                async with self._cache_lock:
                    self._session_cache[session_id] = state.copy(deep=True)
                
                # Update last accessed time
                conn.execute(
                    "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                    (datetime.now(), session_id)
                )
                conn.commit()
                
                print(f"Session {session_id} loaded successfully with {len(messages)} messages")
                return state
                
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    async def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List recent sessions with metadata.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("""
                    SELECT session_id, created_at, last_accessed, message_count
                    FROM sessions 
                    ORDER BY last_accessed DESC 
                    LIMIT ?
                """, (limit,)).fetchall()
                
                return [
                    {
                        "session_id": row[0],
                        "created_at": row[1],
                        "last_accessed": row[2],
                        "message_count": row[3]
                    }
                    for row in rows
                ]
        except Exception as e:
            print(f"Error listing sessions: {e}")
            return []
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all associated data.
        """
        try:
            # Remove from cache
            async with self._cache_lock:
                self._session_cache.pop(session_id, None)
            
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM conversation_history WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM system_logs WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                conn.commit()
            
            print(f"Session {session_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
    
    async def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Clean up sessions older than specified days.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get sessions to delete
                old_sessions = conn.execute(
                    "SELECT session_id FROM sessions WHERE last_accessed < ?",
                    (cutoff_date,)
                ).fetchall()
                
                # Delete old sessions
                for (session_id,) in old_sessions:
                    await self.delete_session(session_id)
                
                print(f"Cleaned up {len(old_sessions)} old sessions")
                return len(old_sessions)
                
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return 0
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored sessions.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                # Total sessions
                stats["total_sessions"] = conn.execute(
                    "SELECT COUNT(*) FROM sessions"
                ).fetchone()[0]
                
                # Total messages
                stats["total_messages"] = conn.execute(
                    "SELECT COUNT(*) FROM conversation_history"
                ).fetchone()[0]
                
                # Total logs
                stats["total_logs"] = conn.execute(
                    "SELECT COUNT(*) FROM system_logs"
                ).fetchone()[0]
                
                # Active sessions (accessed in last 24 hours)
                yesterday = datetime.now() - timedelta(days=1)
                stats["active_sessions"] = conn.execute(
                    "SELECT COUNT(*) FROM sessions WHERE last_accessed > ?",
                    (yesterday,)
                ).fetchone()[0]
                
                return stats
                
        except Exception as e:
            print(f"Error getting session stats: {e}")
            return {}


# Global instance
_session_persistence_service: Optional[SessionPersistenceService] = None


def get_session_persistence_service() -> SessionPersistenceService:
    """Get the global session persistence service instance."""
    global _session_persistence_service
    if _session_persistence_service is None:
        _session_persistence_service = SessionPersistenceService()
    return _session_persistence_service