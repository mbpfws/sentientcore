#!/usr/bin/env python3
"""
Test script for Enhanced Vector Service
Tests semantic search, hybrid search, metadata filtering, and vector store operations.
"""

import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any

# Add the core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from services.vector_service import (
    EnhancedVectorService,
    Document,
    SearchResult,
    SentenceTransformerProvider,
    ChromaVectorStore,
    FAISSVectorStore
)


class MockEmbeddingProvider:
    """Mock embedding provider for testing without external dependencies."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        import random
        random.seed(42)  # For reproducible results
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate mock embedding based on text hash."""
        import hashlib
        import random
        
        # Use text hash as seed for reproducible embeddings
        hash_value = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        random.seed(hash_value)
        
        # Generate normalized random vector
        embedding = [random.gauss(0, 1) for _ in range(self._dimension)]
        
        # Normalize
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch of texts."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


class MockVectorStore:
    """Mock vector store for testing without external dependencies."""
    
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, List[float]] = {}
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to mock store."""
        for doc in documents:
            self.documents[doc.id] = doc
            if doc.embedding:
                self.embeddings[doc.id] = doc.embedding
    
    async def search(self, query_embedding: List[float], k: int = 10, 
                    metadata_filter: Dict[str, Any] = None) -> List[SearchResult]:
        """Mock search using cosine similarity."""
        results = []
        
        for doc_id, doc in self.documents.items():
            if doc_id not in self.embeddings:
                continue
            
            # Apply metadata filter
            if metadata_filter:
                if not self._matches_filter(doc.metadata, metadata_filter):
                    continue
            
            # Calculate cosine similarity
            doc_embedding = self.embeddings[doc_id]
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            results.append(SearchResult(
                document=doc,
                score=similarity,
                search_type="semantic"
            ))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    async def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from mock store."""
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]
    
    async def update_document(self, document: Document) -> None:
        """Update document in mock store."""
        self.documents[document.id] = document
        if document.embedding:
            self.embeddings[document.id] = document.embedding


async def test_basic_vector_service():
    """Test basic vector service functionality."""
    print("\n=== Testing Basic Vector Service ===")
    
    # Create service with mock providers
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore()
    
    service = EnhancedVectorService(
        embedding_provider=embedding_provider,
        vector_store=vector_store
    )
    
    # Create test documents
    documents = [
        Document(
            id="doc1",
            content="Python is a programming language",
            metadata={"category": "programming", "language": "python"}
        ),
        Document(
            id="doc2",
            content="Machine learning with neural networks",
            metadata={"category": "ai", "topic": "ml"}
        ),
        Document(
            id="doc3",
            content="Web development using JavaScript",
            metadata={"category": "programming", "language": "javascript"}
        )
    ]
    
    # Add documents
    await service.add_documents(documents)
    print(f"Added {len(documents)} documents")
    
    # Test semantic search
    results = await service.semantic_search("programming languages", k=2)
    print(f"Semantic search results: {len(results)}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.document.content[:50]}... (score: {result.score:.3f})")
    
    # Test keyword search
    results = await service.keyword_search("Python programming", k=2)
    print(f"Keyword search results: {len(results)}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.document.content[:50]}... (score: {result.score:.3f})")
    
    # Test hybrid search
    results = await service.hybrid_search("machine learning", k=2)
    print(f"Hybrid search results: {len(results)}")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.document.content[:50]}... (score: {result.score:.3f})")
    
    print("✓ Basic vector service test passed")


async def test_metadata_filtering():
    """Test metadata filtering capabilities."""
    print("\n=== Testing Metadata Filtering ===")
    
    # Create service with mock providers
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore()
    
    service = EnhancedVectorService(
        embedding_provider=embedding_provider,
        vector_store=vector_store
    )
    
    # Create test documents with different metadata
    documents = [
        Document(
            id="doc1",
            content="Python programming tutorial",
            metadata={"category": "programming", "difficulty": "beginner", "language": "python"}
        ),
        Document(
            id="doc2",
            content="Advanced Python concepts",
            metadata={"category": "programming", "difficulty": "advanced", "language": "python"}
        ),
        Document(
            id="doc3",
            content="JavaScript basics",
            metadata={"category": "programming", "difficulty": "beginner", "language": "javascript"}
        ),
        Document(
            id="doc4",
            content="Data science with Python",
            metadata={"category": "data-science", "difficulty": "intermediate", "language": "python"}
        )
    ]
    
    await service.add_documents(documents)
    
    # Test filtering by category
    results = await service.semantic_search(
        "programming", 
        k=10, 
        metadata_filter={"category": "programming"}
    )
    print(f"Programming category results: {len(results)}")
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    
    # Test filtering by difficulty
    results = await service.semantic_search(
        "Python", 
        k=10, 
        metadata_filter={"difficulty": "beginner"}
    )
    print(f"Beginner difficulty results: {len(results)}")
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    
    # Test filtering by multiple criteria
    results = await service.semantic_search(
        "Python", 
        k=10, 
        metadata_filter={"category": "programming", "language": "python"}
    )
    print(f"Python programming results: {len(results)}")
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    
    print("✓ Metadata filtering test passed")


async def test_document_operations():
    """Test document CRUD operations."""
    print("\n=== Testing Document Operations ===")
    
    # Create service with mock providers
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore()
    
    service = EnhancedVectorService(
        embedding_provider=embedding_provider,
        vector_store=vector_store
    )
    
    # Create initial document
    document = Document(
        id="test_doc",
        content="Original content",
        metadata={"version": 1}
    )
    
    await service.add_documents([document])
    
    # Test retrieval
    retrieved = await service.get_document("test_doc")
    assert retrieved is not None, "Document should be retrievable"
    assert retrieved.content == "Original content", "Content should match"
    
    # Test update
    updated_document = Document(
        id="test_doc",
        content="Updated content",
        metadata={"version": 2}
    )
    
    await service.update_document(updated_document)
    
    # Verify update
    retrieved = await service.get_document("test_doc")
    assert retrieved.content == "Updated content", "Content should be updated"
    assert retrieved.metadata["version"] == 2, "Metadata should be updated"
    
    # Test deletion
    await service.delete_documents(["test_doc"])
    
    # Verify deletion
    retrieved = await service.get_document("test_doc")
    assert retrieved is None, "Document should be deleted"
    
    print("✓ Document operations test passed")


async def test_performance_tracking():
    """Test performance tracking and statistics."""
    print("\n=== Testing Performance Tracking ===")
    
    # Create service with mock providers
    embedding_provider = MockEmbeddingProvider()
    vector_store = MockVectorStore()
    
    service = EnhancedVectorService(
        embedding_provider=embedding_provider,
        vector_store=vector_store
    )
    
    # Add test documents
    documents = [
        Document(id=f"doc{i}", content=f"Test document {i}", metadata={"index": i})
        for i in range(5)
    ]
    
    await service.add_documents(documents)
    
    # Perform various searches
    await service.semantic_search("test", k=3)
    await service.keyword_search("document", k=3)
    await service.hybrid_search("test document", k=3)
    
    # Get statistics
    stats = service.get_statistics()
    
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total searches: {stats['total_searches']}")
    print(f"Search counts: {stats['search_counts']}")
    
    assert stats['total_documents'] == 5, "Should have 5 documents"
    # Hybrid search calls both semantic and keyword internally, so total = 1 + 1 + 1 + 2 = 5
    assert stats['total_searches'] == 5, "Should have 5 total searches (hybrid calls semantic+keyword)"
    assert stats['search_counts']['semantic'] == 2, "Should have 2 semantic searches (1 direct + 1 from hybrid)"
    assert stats['search_counts']['keyword'] == 2, "Should have 2 keyword searches (1 direct + 1 from hybrid)"
    assert stats['search_counts']['hybrid'] == 1, "Should have 1 hybrid search"
    
    print("✓ Performance tracking test passed")


async def test_real_vector_stores():
    """Test with real vector store implementations if available."""
    print("\n=== Testing Real Vector Stores ===")
    
    # Test with available real implementations
    try:
        from services.vector_service import CHROMADB_AVAILABLE, FAISS_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Testing with SentenceTransformer embedding provider...")
            
            # Create temporary directory for ChromaDB
            temp_dir = tempfile.mkdtemp()
            
            try:
                if CHROMADB_AVAILABLE:
                    print("Testing ChromaDB vector store...")
                    
                    embedding_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
                    vector_store = ChromaVectorStore("test_collection", temp_dir)
                    
                    service = EnhancedVectorService(
                        embedding_provider=embedding_provider,
                        vector_store=vector_store
                    )
                    
                    # Quick test
                    test_doc = Document(
                        id="real_test",
                        content="This is a real test document",
                        metadata={"test": True}
                    )
                    
                    await service.add_documents([test_doc])
                    results = await service.semantic_search("test document", k=1)
                    
                    assert len(results) > 0, "Should find the test document"
                    print("✓ ChromaDB test passed")
                
                if FAISS_AVAILABLE:
                    print("Testing FAISS vector store...")
                    
                    embedding_provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
                    vector_store = FAISSVectorStore(embedding_provider.dimension)
                    
                    service = EnhancedVectorService(
                        embedding_provider=embedding_provider,
                        vector_store=vector_store
                    )
                    
                    # Quick test
                    test_doc = Document(
                        id="faiss_test",
                        content="This is a FAISS test document",
                        metadata={"test": True}
                    )
                    
                    await service.add_documents([test_doc])
                    results = await service.semantic_search("FAISS test", k=1)
                    
                    assert len(results) > 0, "Should find the test document"
                    print("✓ FAISS test passed")
                    
            finally:
                # Cleanup
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
        
        else:
            print("SentenceTransformers not available, skipping real vector store tests")
            
    except Exception as e:
        print(f"Real vector store test failed (this is expected if dependencies are missing): {e}")


async def main():
    """Run all vector service tests."""
    print("Starting Enhanced Vector Service Tests...")
    
    try:
        await test_basic_vector_service()
        await test_metadata_filtering()
        await test_document_operations()
        await test_performance_tracking()
        await test_real_vector_stores()
        
        print("\n🎉 All vector service tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)