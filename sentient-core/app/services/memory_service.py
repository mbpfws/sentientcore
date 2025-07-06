from typing import Dict, List, Optional, Any, Union
import json
import sqlite3
import asyncio
from datetime import datetime
from pathlib import Path
import uuid
from dataclasses import dataclass, asdict
from enum import Enum


class MemoryType(Enum):
    """Types of memory storage in the hierarchical system"""
    WORKING = "working"  # Layer 1: Active conversation context
    SHORT_TERM = "short_term"  # Layer 2: Recent interactions
    EPISODIC = "episodic"  # Layer 3: Conversation episodes
    SEMANTIC = "semantic"  # Layer 4: Knowledge and facts
    PROCEDURAL = "procedural"  # Layer 5: Learned patterns and procedures


@dataclass
class MemoryEntry:
    """Represents a single memory entry in the system"""
    id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    session_id: Optional[str] = None
    importance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'memory_type': self.memory_type.value,
            'content': json.dumps(self.content),
            'metadata': json.dumps(self.metadata),
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'importance_score': self.importance_score,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        return cls(
            id=data['id'],
            memory_type=MemoryType(data['memory_type']),
            content=json.loads(data['content']),
            metadata=json.loads(data['metadata']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            session_id=data.get('session_id'),
            importance_score=data.get('importance_score', 0.0),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None
        )


class MemoryService:
    """5-Layer Hierarchical Memory Management System"""
    
    def __init__(self, db_path: str = "app/memory_management.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._memory_cache: Dict[str, MemoryEntry] = {}
        
    def _init_database(self):
        """Initialize the SQLite database with memory tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    importance_score REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_id ON memory_entries(session_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance ON memory_entries(importance_score)
            """)
            
            conn.commit()
    
    async def store_memory(
        self, 
        memory_type: MemoryType, 
        content: Dict[str, Any], 
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        importance_score: float = 0.0
    ) -> str:
        """Store a new memory entry"""
        memory_id = str(uuid.uuid4())
        
        entry = MemoryEntry(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            timestamp=datetime.now(),
            session_id=session_id,
            importance_score=importance_score
        )
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO memory_entries 
                (id, memory_type, content, metadata, timestamp, session_id, importance_score, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id,
                entry.memory_type.value,
                json.dumps(entry.content),
                json.dumps(entry.metadata),
                entry.timestamp.isoformat(),
                entry.session_id,
                entry.importance_score,
                entry.access_count,
                entry.last_accessed.isoformat() if entry.last_accessed else None
            ))
            conn.commit()
        
        # Cache the entry
        self._memory_cache[memory_id] = entry
        
        return memory_id
    
    async def retrieve_memory(
        self, 
        memory_id: str, 
        update_access: bool = True
    ) -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry by ID"""
        # Check cache first
        if memory_id in self._memory_cache:
            entry = self._memory_cache[memory_id]
            if update_access:
                await self._update_access_stats(memory_id)
            return entry
        
        # Query database
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM memory_entries WHERE id = ?", 
                (memory_id,)
            )
            row = cursor.fetchone()
            
            if row:
                entry = MemoryEntry.from_dict(dict(row))
                self._memory_cache[memory_id] = entry
                
                if update_access:
                    await self._update_access_stats(memory_id)
                
                return entry
        
        return None
    
    async def query_memories(
        self,
        memory_type: Optional[MemoryType] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
        min_importance: float = 0.0,
        order_by: str = "timestamp",
        ascending: bool = False
    ) -> List[MemoryEntry]:
        """Query memories with various filters"""
        query = "SELECT * FROM memory_entries WHERE 1=1"
        params = []
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type.value)
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if min_importance > 0:
            query += " AND importance_score >= ?"
            params.append(min_importance)
        
        # Add ordering
        order_direction = "ASC" if ascending else "DESC"
        query += f" ORDER BY {order_by} {order_direction}"
        
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        
        memories = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            
            for row in cursor.fetchall():
                entry = MemoryEntry.from_dict(dict(row))
                memories.append(entry)
                # Cache the entry
                self._memory_cache[entry.id] = entry
        
        return memories
    
    async def update_memory(
        self, 
        memory_id: str, 
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance_score: Optional[float] = None
    ) -> bool:
        """Update an existing memory entry"""
        entry = await self.retrieve_memory(memory_id, update_access=False)
        if not entry:
            return False
        
        # Update fields
        if content is not None:
            entry.content = content
        if metadata is not None:
            entry.metadata = metadata
        if importance_score is not None:
            entry.importance_score = importance_score
        
        # Update in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE memory_entries 
                SET content = ?, metadata = ?, importance_score = ?
                WHERE id = ?
            """, (
                json.dumps(entry.content),
                json.dumps(entry.metadata),
                entry.importance_score,
                memory_id
            ))
            conn.commit()
        
        # Update cache
        self._memory_cache[memory_id] = entry
        
        return True
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM memory_entries WHERE id = ?", 
                (memory_id,)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                # Remove from cache
                self._memory_cache.pop(memory_id, None)
                return True
        
        return False
    
    async def _update_access_stats(self, memory_id: str):
        """Update access statistics for a memory entry"""
        now = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE memory_entries 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (now.isoformat(), memory_id))
            conn.commit()
        
        # Update cache
        if memory_id in self._memory_cache:
            entry = self._memory_cache[memory_id]
            entry.access_count += 1
            entry.last_accessed = now
    
    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get all context for a specific session"""
        memories = await self.query_memories(
            session_id=session_id,
            order_by="timestamp",
            ascending=True
        )
        
        context = {
            "session_id": session_id,
            "total_memories": len(memories),
            "memories_by_type": {},
            "timeline": []
        }
        
        for memory in memories:
            # Group by type
            mem_type = memory.memory_type.value
            if mem_type not in context["memories_by_type"]:
                context["memories_by_type"][mem_type] = []
            context["memories_by_type"][mem_type].append(memory.to_dict())
            
            # Add to timeline
            context["timeline"].append({
                "timestamp": memory.timestamp.isoformat(),
                "type": mem_type,
                "content_summary": str(memory.content)[:100] + "..." if len(str(memory.content)) > 100 else str(memory.content)
            })
        
        return context
    
    async def cleanup_old_memories(
        self, 
        memory_type: MemoryType, 
        max_age_days: int = 30,
        max_count: int = 1000
    ) -> int:
        """Clean up old memories based on age and count limits"""
        cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - max_age_days)
        
        deleted_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            # Delete by age
            cursor = conn.execute("""
                DELETE FROM memory_entries 
                WHERE memory_type = ? AND timestamp < ?
            """, (memory_type.value, cutoff_date.isoformat()))
            deleted_count += cursor.rowcount
            
            # Delete excess by count (keep most recent and most important)
            cursor = conn.execute("""
                DELETE FROM memory_entries 
                WHERE id IN (
                    SELECT id FROM memory_entries 
                    WHERE memory_type = ?
                    ORDER BY importance_score DESC, timestamp DESC
                    LIMIT -1 OFFSET ?
                )
            """, (memory_type.value, max_count))
            deleted_count += cursor.rowcount
            
            conn.commit()
        
        # Clear cache for deleted entries
        self._memory_cache.clear()
        
        return deleted_count