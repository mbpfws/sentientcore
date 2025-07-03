"""Memory Management Service for Multi-Agent System

This service provides a 4-layer hierarchical memory management system:
1. Knowledge Synthesis Layer - Research findings, technical reports
2. Conversation History Layer - User interactions, agent decisions
3. Codebase Knowledge Layer - Generated code, patterns, documentation
4. Stack Dependencies Layer - Technology choices, library documentation
"""

import asyncio
import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from contextlib import asynccontextmanager

# Import vector service for semantic search
from .vector_service import (
    EnhancedVectorService as VectorService, 
    Document, 
    SearchResult,
    SentenceTransformerProvider,
    ChromaVectorStore
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryLayer(Enum):
    """Memory layer enumeration"""
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    CONVERSATION_HISTORY = "conversation_history"
    CODEBASE_KNOWLEDGE = "codebase_knowledge"
    STACK_DEPENDENCIES = "stack_dependencies"

class MemoryType(Enum):
    """Memory type enumeration for different content types"""
    CONVERSATION = "conversation"
    CODE_SNIPPET = "code_snippet"
    DOCUMENTATION = "documentation"
    RESEARCH_FINDING = "research_finding"
    ARCHITECTURAL_DECISION = "architectural_decision"
    DEPENDENCY_INFO = "dependency_info"
    BEST_PRACTICE = "best_practice"
    ERROR_SOLUTION = "error_solution"

@dataclass
class MemoryEntry:
    """Memory entry data structure"""
    entry_id: str
    layer: MemoryLayer
    content: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    tags: List[str] = None
    relevance_score: float = 1.0
    access_count: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class KnowledgeLayer:
    """Knowledge Synthesis Layer - Research findings, technical reports"""
    
    def __init__(self, db_connection, vector_service: VectorService):
        self.db = db_connection
        self.vector_service = vector_service
        self.layer_name = "knowledge_synthesis"
    
    async def store_research_finding(self, content: str, source: str, 
                                   topic: str, metadata: Dict[str, Any] = None) -> str:
        """Store research finding with semantic indexing"""
        entry_id = self._generate_id(content)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "source": source,
            "topic": topic,
            "type": "research_finding",
            "layer": self.layer_name
        })
        
        # Store in database
        await self._store_entry(entry_id, content, metadata)
        
        # Create vector document for semantic search
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=content,
            metadata=metadata
        )
        
        # Add to vector store
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored research finding: {entry_id}")
        return entry_id
    
    async def store_technical_report(self, content: str, title: str,
                                   technology: str, metadata: Dict[str, Any] = None) -> str:
        """Store technical report with categorization"""
        entry_id = self._generate_id(content)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "title": title,
            "technology": technology,
            "type": "technical_report",
            "layer": self.layer_name
        })
        
        await self._store_entry(entry_id, content, metadata)
        
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=content,
            metadata=metadata
        )
        
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored technical report: {entry_id}")
        return entry_id
    
    async def _store_entry(self, entry_id: str, content: str, metadata: Dict[str, Any]):
        """Store entry in database"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memory_entries 
            (entry_id, layer, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry_id,
            self.layer_name,
            content,
            json.dumps(metadata),
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        self.db.commit()
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

class ConversationLayer:
    """Conversation History Layer - User interactions, agent decisions"""
    
    def __init__(self, db_connection, vector_service: VectorService):
        self.db = db_connection
        self.vector_service = vector_service
        self.layer_name = "conversation_history"
    
    async def store_user_interaction(self, user_input: str, agent_response: str,
                                   agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Store user interaction with context"""
        content = f"User: {user_input}\nAgent ({agent_id}): {agent_response}"
        entry_id = self._generate_id(content)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "agent_id": agent_id,
            "type": "user_interaction",
            "layer": self.layer_name,
            "user_input": user_input,
            "agent_response": agent_response
        })
        
        await self._store_entry(entry_id, content, metadata)
        
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=content,
            metadata=metadata
        )
        
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored user interaction: {entry_id}")
        return entry_id
    
    async def store_agent_decision(self, decision: str, reasoning: str,
                                 agent_id: str, metadata: Dict[str, Any] = None) -> str:
        """Store agent decision with reasoning"""
        content = f"Decision: {decision}\nReasoning: {reasoning}"
        entry_id = self._generate_id(content)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "agent_id": agent_id,
            "type": "agent_decision",
            "layer": self.layer_name,
            "decision": decision,
            "reasoning": reasoning
        })
        
        await self._store_entry(entry_id, content, metadata)
        
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=content,
            metadata=metadata
        )
        
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored agent decision: {entry_id}")
        return entry_id
    
    async def _store_entry(self, entry_id: str, content: str, metadata: Dict[str, Any]):
        """Store entry in database"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memory_entries 
            (entry_id, layer, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry_id,
            self.layer_name,
            content,
            json.dumps(metadata),
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        self.db.commit()
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

class CodebaseLayer:
    """Codebase Knowledge Layer - Generated code, patterns, documentation"""
    
    def __init__(self, db_connection, vector_service: VectorService):
        self.db = db_connection
        self.vector_service = vector_service
        self.layer_name = "codebase_knowledge"
    
    async def store_generated_code(self, code: str, language: str, purpose: str,
                                 metadata: Dict[str, Any] = None) -> str:
        """Store generated code with context"""
        entry_id = self._generate_id(code)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "language": language,
            "purpose": purpose,
            "type": "generated_code",
            "layer": self.layer_name
        })
        
        await self._store_entry(entry_id, code, metadata)
        
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=code,
            metadata=metadata
        )
        
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored generated code: {entry_id}")
        return entry_id
    
    async def store_code_pattern(self, pattern: str, description: str,
                               use_case: str, metadata: Dict[str, Any] = None) -> str:
        """Store code pattern with description"""
        content = f"Pattern: {pattern}\nDescription: {description}\nUse Case: {use_case}"
        entry_id = self._generate_id(content)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "pattern": pattern,
            "description": description,
            "use_case": use_case,
            "type": "code_pattern",
            "layer": self.layer_name
        })
        
        await self._store_entry(entry_id, content, metadata)
        
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=content,
            metadata=metadata
        )
        
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored code pattern: {entry_id}")
        return entry_id
    
    async def _store_entry(self, entry_id: str, content: str, metadata: Dict[str, Any]):
        """Store entry in database"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memory_entries 
            (entry_id, layer, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry_id,
            self.layer_name,
            content,
            json.dumps(metadata),
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        self.db.commit()
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

class StackLayer:
    """Stack Dependencies Layer - Technology choices, library documentation"""
    
    def __init__(self, db_connection, vector_service: VectorService):
        self.db = db_connection
        self.vector_service = vector_service
        self.layer_name = "stack_dependencies"
    
    async def store_technology_choice(self, technology: str, rationale: str,
                                    alternatives: List[str], metadata: Dict[str, Any] = None) -> str:
        """Store technology choice with rationale"""
        content = f"Technology: {technology}\nRationale: {rationale}\nAlternatives: {', '.join(alternatives)}"
        entry_id = self._generate_id(content)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "technology": technology,
            "rationale": rationale,
            "alternatives": alternatives,
            "type": "technology_choice",
            "layer": self.layer_name
        })
        
        await self._store_entry(entry_id, content, metadata)
        
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=content,
            metadata=metadata
        )
        
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored technology choice: {entry_id}")
        return entry_id
    
    async def store_library_documentation(self, library: str, documentation: str,
                                        version: str, metadata: Dict[str, Any] = None) -> str:
        """Store library documentation"""
        content = f"Library: {library} (v{version})\nDocumentation: {documentation}"
        entry_id = self._generate_id(content)
        
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "library": library,
            "version": version,
            "type": "library_documentation",
            "layer": self.layer_name
        })
        
        await self._store_entry(entry_id, content, metadata)
        
        doc = Document(
            id=f"{self.layer_name}_{entry_id}",
            content=content,
            metadata=metadata
        )
        
        await self.vector_service.add_document(doc)
        
        logger.info(f"Stored library documentation: {entry_id}")
        return entry_id
    
    async def _store_entry(self, entry_id: str, content: str, metadata: Dict[str, Any]):
        """Store entry in database"""
        cursor = self.db.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memory_entries 
            (entry_id, layer, content, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entry_id,
            self.layer_name,
            content,
            json.dumps(metadata),
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        self.db.commit()
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for content"""
        return hashlib.md5(content.encode()).hexdigest()[:16]

class MemoryService:
    """4-Layer Hierarchical Memory Management Service
    
    Provides comprehensive memory management across four specialized layers:
    1. Knowledge Synthesis - Research findings, technical reports
    2. Conversation History - User interactions, agent decisions
    3. Codebase Knowledge - Generated code, patterns, documentation
    4. Stack Dependencies - Technology choices, library documentation
    """
    
    def __init__(self, db_path: str = "memory_management.db", 
                 vector_db_path: str = "./memory_vectors"):
        """Initialize the Memory Management Service
        
        Args:
            db_path: Path to SQLite database file
            vector_db_path: Path to vector database directory
        """
        self.db_path = Path(db_path)
        self.vector_db_path = Path(vector_db_path)
        
        # Initialize database connection
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Initialize vector service components
        embedding_provider = SentenceTransformerProvider()
        vector_store = ChromaVectorStore(
            collection_name="memory_vectors",
            persist_directory=str(self.vector_db_path)
        )
        self.vector_service = VectorService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            enable_keyword_search=True
        )
        
        # Initialize layers
        self.layers = {
            MemoryLayer.KNOWLEDGE_SYNTHESIS: KnowledgeLayer(self.db, self.vector_service),
            MemoryLayer.CONVERSATION_HISTORY: ConversationLayer(self.db, self.vector_service),
            MemoryLayer.CODEBASE_KNOWLEDGE: CodebaseLayer(self.db, self.vector_service),
            MemoryLayer.STACK_DEPENDENCIES: StackLayer(self.db, self.vector_service)
        }
        
        # Performance tracking
        self.performance_stats = {
            "entries_stored": 0,
            "searches_performed": 0,
            "avg_search_time": 0.0,
            "cache_hits": 0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info(f"MemoryService initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        cursor = self.db.cursor()
        
        # Memory entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                entry_id TEXT PRIMARY KEY,
                layer TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                tags TEXT,
                relevance_score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_layer ON memory_entries(layer)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memory_entries(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_relevance ON memory_entries(relevance_score)")
        
        self.db.commit()
        logger.info("Memory database initialized")
    
    async def store_knowledge(self, layer: Union[MemoryLayer, str], data: Dict[str, Any], 
                            metadata: Dict[str, Any] = None) -> str:
        """Store information in specified memory layer
        
        Args:
            layer: Memory layer to store in
            data: Data to store (structure depends on layer)
            metadata: Additional metadata
            
        Returns:
            Entry ID of stored data
        """
        if isinstance(layer, str):
            layer = MemoryLayer(layer)
        
        layer_service = self.layers[layer]
        
        # Route to appropriate storage method based on layer and data type
        if layer == MemoryLayer.KNOWLEDGE_SYNTHESIS:
            if "research_finding" in data:
                return await layer_service.store_research_finding(
                    data["content"], data["source"], data["topic"], metadata
                )
            elif "technical_report" in data:
                return await layer_service.store_technical_report(
                    data["content"], data["title"], data["technology"], metadata
                )
        
        elif layer == MemoryLayer.CONVERSATION_HISTORY:
            if "user_interaction" in data:
                return await layer_service.store_user_interaction(
                    data["user_input"], data["agent_response"], data["agent_id"], metadata
                )
            elif "agent_decision" in data:
                return await layer_service.store_agent_decision(
                    data["decision"], data["reasoning"], data["agent_id"], metadata
                )
        
        elif layer == MemoryLayer.CODEBASE_KNOWLEDGE:
            if "generated_code" in data:
                return await layer_service.store_generated_code(
                    data["code"], data["language"], data["purpose"], metadata
                )
            elif "code_pattern" in data:
                return await layer_service.store_code_pattern(
                    data["pattern"], data["description"], data["use_case"], metadata
                )
        
        elif layer == MemoryLayer.STACK_DEPENDENCIES:
            if "technology_choice" in data:
                return await layer_service.store_technology_choice(
                    data["technology"], data["rationale"], data["alternatives"], metadata
                )
            elif "library_documentation" in data:
                return await layer_service.store_library_documentation(
                    data["library"], data["documentation"], data["version"], metadata
                )
        
        raise ValueError(f"Unsupported data type for layer {layer}")
    
    async def retrieve_relevant(self, query: str, layers: List[Union[MemoryLayer, str]] = None,
                              limit: int = 10, min_score: float = 0.5) -> List[SearchResult]:
        """Retrieve relevant information across specified layers
        
        Args:
            query: Search query
            layers: Layers to search in (default: all layers)
            limit: Maximum number of results
            min_score: Minimum relevance score
            
        Returns:
            List of search results
        """
        if layers is None:
            layers = list(MemoryLayer)
        
        # Convert string layers to enum
        layer_filters = []
        for layer in layers:
            if isinstance(layer, str):
                layer = MemoryLayer(layer)
            layer_filters.append(layer.value)
        
        # Perform semantic search
        results = await self.vector_service.search(
            query=query,
            k=limit,
            metadata_filter={"layer": {"$in": layer_filters}}
        )
        
        # Filter by minimum score
        filtered_results = [r for r in results if r.score >= min_score]
        
        # Update access counts
        for result in filtered_results:
            await self._update_access_count(result.document.id)
        
        self.performance_stats["searches_performed"] += 1
        
        logger.info(f"Retrieved {len(filtered_results)} relevant entries for query: {query[:50]}...")
        return filtered_results
    
    async def get_layer_statistics(self, layer: Union[MemoryLayer, str]) -> Dict[str, Any]:
        """Get statistics for a specific memory layer"""
        if isinstance(layer, str):
            layer = MemoryLayer(layer)
        
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_entries,
                AVG(relevance_score) as avg_relevance,
                SUM(access_count) as total_accesses,
                MIN(created_at) as oldest_entry,
                MAX(created_at) as newest_entry
            FROM memory_entries 
            WHERE layer = ?
        """, (layer.value,))
        
        result = cursor.fetchone()
        
        return {
            "layer": layer.value,
            "total_entries": result[0],
            "average_relevance": result[1] or 0.0,
            "total_accesses": result[2] or 0,
            "oldest_entry": result[3],
            "newest_entry": result[4]
        }
    
    async def cleanup_old_entries(self, days_old: int = 30, min_access_count: int = 1):
        """Clean up old, rarely accessed entries"""
        cutoff_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
        
        cursor = self.db.cursor()
        cursor.execute("""
            DELETE FROM memory_entries 
            WHERE created_at < ? AND access_count < ?
        """, (cutoff_date, min_access_count))
        
        deleted_count = cursor.rowcount
        self.db.commit()
        
        logger.info(f"Cleaned up {deleted_count} old memory entries")
        return deleted_count
    
    async def _update_access_count(self, entry_id: str):
        """Update access count for an entry"""
        # Extract actual entry ID from document ID
        if "_" in entry_id:
            actual_entry_id = entry_id.split("_", 1)[1]
        else:
            actual_entry_id = entry_id
        
        cursor = self.db.cursor()
        cursor.execute("""
            UPDATE memory_entries 
            SET access_count = access_count + 1, updated_at = ?
            WHERE entry_id = ?
        """, (datetime.utcnow().isoformat(), actual_entry_id))
        self.db.commit()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get memory service performance statistics"""
        return self.performance_stats.copy()
    
    async def start(self):
        """Start the memory service (async initialization)"""
        # Vector service doesn't need explicit start
        logger.info("MemoryService started")
    
    async def store_memory(self, layer: MemoryLayer, memory_type: MemoryType, 
                          content: str, metadata: Dict[str, Any] = None, 
                          tags: List[str] = None) -> str:
        """Store memory with specified layer and type
        
        Args:
            layer: Memory layer to store in
            memory_type: Type of memory content
            content: Content to store
            metadata: Additional metadata
            tags: Tags for categorization
            
        Returns:
            Entry ID of stored memory
        """
        if metadata is None:
            metadata = {}
        if tags is None:
            tags = []
            
        # Add memory type to metadata
        metadata["memory_type"] = memory_type.value
        # Convert tags list to comma-separated string for ChromaDB compatibility
        metadata["tags"] = ",".join(tags) if tags else ""
        
        # Route to appropriate layer based on memory type and layer
        layer_service = self.layers[layer]
        
        if layer == MemoryLayer.KNOWLEDGE_SYNTHESIS:
            if memory_type == MemoryType.RESEARCH_FINDING:
                return await layer_service.store_research_finding(
                    content, metadata.get("source", "unknown"), 
                    metadata.get("topic", "general"), metadata
                )
            else:
                return await layer_service.store_technical_report(
                    content, metadata.get("title", "Technical Report"), 
                    metadata.get("technology", "general"), metadata
                )
        
        elif layer == MemoryLayer.CONVERSATION_HISTORY:
            if memory_type == MemoryType.CONVERSATION:
                return await layer_service.store_user_interaction(
                    metadata.get("user_input", ""), content, 
                    metadata.get("agent_id", "unknown"), metadata
                )
            else:
                return await layer_service.store_agent_decision(
                    content, metadata.get("reasoning", ""), 
                    metadata.get("agent_id", "unknown"), metadata
                )
        
        elif layer == MemoryLayer.CODEBASE_KNOWLEDGE:
            if memory_type == MemoryType.CODE_SNIPPET:
                return await layer_service.store_generated_code(
                    content, metadata.get("language", "unknown"), 
                    metadata.get("purpose", "general"), metadata
                )
            else:
                return await layer_service.store_code_pattern(
                    content, metadata.get("description", ""), 
                    metadata.get("use_case", "general"), metadata
                )
        
        elif layer == MemoryLayer.STACK_DEPENDENCIES:
            if memory_type == MemoryType.DEPENDENCY_INFO:
                return await layer_service.store_library_documentation(
                    metadata.get("library", "unknown"), content, 
                    metadata.get("version", "latest"), metadata
                )
            else:
                return await layer_service.store_technology_choice(
                    metadata.get("technology", "unknown"), content, 
                    metadata.get("alternatives", []), metadata
                )
        
        raise ValueError(f"Unsupported memory type {memory_type} for layer {layer}")
    
    async def retrieve_memories(self, query: str, layer: MemoryLayer = None, 
                               memory_type: MemoryType = None, limit: int = 10, 
                               similarity_threshold: float = 0.7) -> List[Any]:
        """Retrieve memories based on query and filters
        
        Args:
            query: Search query
            layer: Optional layer filter
            memory_type: Optional memory type filter
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of memory entries
        """
        # Build metadata filter
        metadata_filter = {}
        if layer:
            metadata_filter["layer"] = layer.value
        if memory_type:
            metadata_filter["memory_type"] = memory_type.value
        
        # Perform search
        results = await self.vector_service.search(
            query=query,
            k=limit,
            metadata_filter=metadata_filter if metadata_filter else None
        )
        
        # Filter by similarity threshold and convert to memory objects
        memories = []
        for result in results:
            if result.score >= similarity_threshold:
                # Create memory object from search result
                memory = type('Memory', (), {
                    'id': result.document.id,
                    'content': result.document.content,
                    'layer': MemoryLayer(result.document.metadata.get('layer', 'knowledge_synthesis')),
                    'memory_type': MemoryType(result.document.metadata.get('memory_type', 'documentation')),
                    'metadata': result.document.metadata,
                    'tags': result.document.metadata.get('tags', '').split(',') if result.document.metadata.get('tags') else [],
                    'created_at': datetime.fromisoformat(result.document.metadata.get('created_at', datetime.utcnow().isoformat())),
                    'similarity_score': result.score
                })()
                memories.append(memory)
        
        self.performance_stats["searches_performed"] += 1
        return memories
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = {
            "total_entries": 0,
            "layers": {},
            "performance": self.get_performance_stats()
        }
        
        # Get statistics for each layer
        for layer in MemoryLayer:
            layer_stats = await self.get_layer_statistics(layer)
            stats["layers"][layer.value] = layer_stats
            stats["total_entries"] += layer_stats["total_entries"]
        
        return stats

    def close(self):
        """Close database connections"""
        if self.db:
            self.db.close()
        logger.info("MemoryService closed")