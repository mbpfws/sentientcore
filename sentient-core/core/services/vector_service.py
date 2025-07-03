import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
from collections import defaultdict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not available. Install with: pip install chromadb")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("SentenceTransformers not available. Install with: pip install sentence-transformers")


@dataclass
class Document:
    """Represents a document with content and metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class SearchResult:
    """Represents a search result with score and metadata."""
    document: Document
    score: float
    search_type: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using SentenceTransformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available")
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dimension = None

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = await asyncio.to_thread(self.model.encode, text)
        return embedding.tolist()

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = await asyncio.to_thread(self.model.encode, texts)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        if self._dimension is None:
            # Get dimension by encoding a sample text
            sample_embedding = self.model.encode("sample")
            self._dimension = len(sample_embedding)
        return self._dimension


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], k: int = 10, 
                    metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store."""
        pass

    @abstractmethod
    async def update_document(self, document: Document) -> None:
        """Update a document in the vector store."""
        pass


class ChromaVectorStore(VectorStore):
    """Vector store implementation using ChromaDB."""

    def __init__(self, collection_name: str = "default", persist_directory: str = "./chroma_db"):
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available")
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to ChromaDB."""
        if not documents:
            return
        
        ids = [doc.id for doc in documents]
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        metadatas = [doc.metadata for doc in documents]
        documents_content = [doc.content for doc in documents]
        
        if embeddings and len(embeddings) == len(documents):
            await asyncio.to_thread(
                self.collection.add,
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
        else:
            # Add without embeddings if not provided
            await asyncio.to_thread(
                self.collection.add,
                ids=ids,
                metadatas=metadatas,
                documents=documents_content
            )

    async def search(self, query_embedding: List[float], k: int = 10, 
                    metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents in ChromaDB."""
        where_clause = metadata_filter if metadata_filter else None
        
        results = await asyncio.to_thread(
            self.collection.query,
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause
        )
        
        search_results = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                document = Document(
                    id=doc_id,
                    content=results['documents'][0][i] if results['documents'] else "",
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    embedding=None  # ChromaDB doesn't return embeddings in query
                )
                
                score = 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                
                search_results.append(SearchResult(
                    document=document,
                    score=score,
                    search_type="semantic"
                ))
        
        return search_results

    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from ChromaDB."""
        if document_ids:
            await asyncio.to_thread(self.collection.delete, ids=document_ids)

    async def update_document(self, document: Document) -> None:
        """Update a document in ChromaDB."""
        # ChromaDB doesn't have direct update, so we delete and add
        await self.delete_documents([document.id])
        await self.add_documents([document])


class FAISSVectorStore(VectorStore):
    """Vector store implementation using FAISS for high-performance search."""

    def __init__(self, dimension: int, index_type: str = "IVFFlat", nlist: int = 100):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available")
        
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        
        # Initialize FAISS index
        if index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Storage for documents and metadata
        self.documents: Dict[int, Document] = {}
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.next_index = 0

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to FAISS index."""
        if not documents:
            return
        
        embeddings = []
        for doc in documents:
            if doc.embedding:
                # Store document
                self.documents[self.next_index] = doc
                self.id_to_index[doc.id] = self.next_index
                self.index_to_id[self.next_index] = doc.id
                
                embeddings.append(doc.embedding)
                self.next_index += 1
        
        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Train index if needed
            if not self.index.is_trained and hasattr(self.index, 'train'):
                self.index.train(embeddings_array)
            
            # Add embeddings to index
            await asyncio.to_thread(self.index.add, embeddings_array)

    async def search(self, query_embedding: List[float], k: int = 10, 
                    metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar documents in FAISS index."""
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS
        distances, indices = await asyncio.to_thread(self.index.search, query_array, k)
        
        search_results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.index_to_id:
                doc_id = self.index_to_id[idx]
                document = self.documents[idx]
                
                # Apply metadata filter if provided
                if metadata_filter:
                    if not self._matches_filter(document.metadata, metadata_filter):
                        continue
                
                # Convert distance to similarity score (higher is better)
                score = 1.0 / (1.0 + distances[0][i])
                
                search_results.append(SearchResult(
                    document=document,
                    score=score,
                    search_type="semantic"
                ))
        
        return search_results

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from FAISS store."""
        # FAISS doesn't support direct deletion, so we rebuild the index
        indices_to_remove = [self.id_to_index[doc_id] for doc_id in document_ids if doc_id in self.id_to_index]
        
        if indices_to_remove:
            # Remove from storage
            for doc_id in document_ids:
                if doc_id in self.id_to_index:
                    idx = self.id_to_index[doc_id]
                    del self.documents[idx]
                    del self.id_to_index[doc_id]
                    del self.index_to_id[idx]
            
            # Rebuild index (simplified approach)
            remaining_docs = list(self.documents.values())
            self._rebuild_index(remaining_docs)

    def _rebuild_index(self, documents: List[Document]):
        """Rebuild the FAISS index with remaining documents."""
        # Reset index
        if self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Reset mappings
        self.documents.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.next_index = 0
        
        # Re-add documents
        if documents:
            asyncio.create_task(self.add_documents(documents))

    async def update_document(self, document: Document) -> None:
        """Update a document in FAISS store."""
        # Delete and re-add
        await self.delete_documents([document.id])
        await self.add_documents([document])


class EnhancedVectorService:
    """
    Advanced vector service with multiple search capabilities:
    - Semantic search using embeddings
    - Hybrid search combining semantic and keyword search
    - Metadata filtering
    - Multiple vector store backends
    """

    def __init__(self, 
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 vector_store: Optional[VectorStore] = None,
                 enable_keyword_search: bool = True):
        
        # Initialize embedding provider
        if embedding_provider is None:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_provider = SentenceTransformerProvider()
            else:
                raise ImportError("No embedding provider available")
        else:
            self.embedding_provider = embedding_provider
        
        # Initialize vector store
        if vector_store is None:
            if CHROMADB_AVAILABLE:
                self.vector_store = ChromaVectorStore()
            elif FAISS_AVAILABLE:
                self.vector_store = FAISSVectorStore(self.embedding_provider.dimension)
            else:
                raise ImportError("No vector store backend available")
        else:
            self.vector_store = vector_store
        
        self.enable_keyword_search = enable_keyword_search
        
        # Simple keyword index for hybrid search
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)
        self.document_cache: Dict[str, Document] = {}
        
        # Usage tracking
        self.search_stats = defaultdict(int)
        self.performance_metrics = defaultdict(list)

    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector service."""
        if not documents:
            return
        
        # Generate embeddings if not provided
        docs_to_embed = [doc for doc in documents if doc.embedding is None]
        if docs_to_embed:
            texts = [doc.content for doc in docs_to_embed]
            embeddings = await self.embedding_provider.embed_batch(texts)
            
            for doc, embedding in zip(docs_to_embed, embeddings):
                doc.embedding = embedding
        
        # Add to vector store
        await self.vector_store.add_documents(documents)
        
        # Update caches
        for doc in documents:
            self.document_cache[doc.id] = doc
            
            # Build keyword index
            if self.enable_keyword_search:
                self._update_keyword_index(doc)

    def _update_keyword_index(self, document: Document):
        """Update the keyword index with document content."""
        # Simple tokenization (can be enhanced with proper NLP)
        words = document.content.lower().split()
        for word in words:
            # Remove punctuation and filter short words
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2:
                self.keyword_index[clean_word].append(document.id)

    async def semantic_search(self, query: str, k: int = 10, 
                             metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        import time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = await self.embedding_provider.embed_text(query)
        
        # Search in vector store
        results = await self.vector_store.search(query_embedding, k, metadata_filter)
        
        # Track performance
        search_time = time.time() - start_time
        self.search_stats['semantic'] += 1
        self.performance_metrics['semantic_search_time'].append(search_time)
        
        return results

    async def keyword_search(self, query: str, k: int = 10, 
                            metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform keyword-based search."""
        if not self.enable_keyword_search:
            return []
        
        import time
        start_time = time.time()
        
        # Simple keyword matching
        query_words = [word.lower().strip() for word in query.split()]
        doc_scores = defaultdict(float)
        
        for word in query_words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word in self.keyword_index:
                for doc_id in self.keyword_index[clean_word]:
                    doc_scores[doc_id] += 1.0
        
        # Sort by score and apply metadata filter
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for doc_id, score in sorted_docs:
            if doc_id in self.document_cache:
                document = self.document_cache[doc_id]
                
                # Apply metadata filter
                if metadata_filter:
                    if not self._matches_filter(document.metadata, metadata_filter):
                        continue
                
                results.append(SearchResult(
                    document=document,
                    score=score / len(query_words),  # Normalize score
                    search_type="keyword"
                ))
        
        # Track performance
        search_time = time.time() - start_time
        self.search_stats['keyword'] += 1
        self.performance_metrics['keyword_search_time'].append(search_time)
        
        return results

    async def hybrid_search(self, query: str, k: int = 10, 
                           semantic_weight: float = 0.7,
                           metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword search."""
        import time
        start_time = time.time()
        
        # Perform both searches
        semantic_results = await self.semantic_search(query, k * 2, metadata_filter)
        keyword_results = await self.keyword_search(query, k * 2, metadata_filter)
        
        # Combine results with weighted scores
        combined_scores = defaultdict(float)
        doc_map = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result.document.id
            combined_scores[doc_id] += result.score * semantic_weight
            doc_map[doc_id] = result.document
        
        # Add keyword results
        keyword_weight = 1.0 - semantic_weight
        for result in keyword_results:
            doc_id = result.document.id
            combined_scores[doc_id] += result.score * keyword_weight
            doc_map[doc_id] = result.document
        
        # Sort and return top k results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        hybrid_results = []
        for doc_id, score in sorted_results:
            hybrid_results.append(SearchResult(
                document=doc_map[doc_id],
                score=score,
                search_type="hybrid",
                metadata={"semantic_weight": semantic_weight, "keyword_weight": keyword_weight}
            ))
        
        # Track performance
        search_time = time.time() - start_time
        self.search_stats['hybrid'] += 1
        self.performance_metrics['hybrid_search_time'].append(search_time)
        
        return hybrid_results

    async def search(self, query: str, k: int = 10, 
                    metadata_filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """General search method that uses hybrid search by default."""
        return await self.hybrid_search(query, k, metadata_filter=metadata_filter)

    async def add_document(self, document: Document) -> None:
        """Add a single document to the vector service."""
        await self.add_documents([document])

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector service."""
        await self.vector_store.delete_documents(document_ids)
        
        # Update caches
        for doc_id in document_ids:
            if doc_id in self.document_cache:
                del self.document_cache[doc_id]
        
        # Update keyword index (simplified - rebuild)
        if self.enable_keyword_search:
            self._rebuild_keyword_index()

    def _rebuild_keyword_index(self):
        """Rebuild the keyword index from cached documents."""
        self.keyword_index.clear()
        for document in self.document_cache.values():
            self._update_keyword_index(document)

    async def update_document(self, document: Document) -> None:
        """Update a document in the vector service."""
        # Generate embedding if not provided
        if document.embedding is None:
            document.embedding = await self.embedding_provider.embed_text(document.content)
        
        document.updated_at = datetime.now()
        
        # Update in vector store
        await self.vector_store.update_document(document)
        
        # Update cache
        self.document_cache[document.id] = document
        
        # Update keyword index
        if self.enable_keyword_search:
            self._update_keyword_index(document)

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics and performance metrics."""
        stats = {
            'search_counts': dict(self.search_stats),
            'total_documents': len(self.document_cache),
            'total_searches': sum(self.search_stats.values()),
            'average_search_times': {}
        }
        
        # Calculate average search times
        for search_type, times in self.performance_metrics.items():
            if times:
                stats['average_search_times'][search_type] = sum(times) / len(times)
        
        return stats

    async def get_document(self, document_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self.document_cache.get(document_id)

    async def list_documents(self, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """List all documents, optionally filtered by metadata."""
        documents = list(self.document_cache.values())
        
        if metadata_filter:
            filtered_docs = []
            for doc in documents:
                if self._matches_filter(doc.metadata, metadata_filter):
                    filtered_docs.append(doc)
            return filtered_docs
        
        return documents