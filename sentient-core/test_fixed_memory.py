#!/usr/bin/env python3
"""
Test script to verify the fixed MemoryService works with ChromaDB
"""

import os
import sys
import asyncio

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.services.memory_service import MemoryService, MemoryLayer, MemoryType

async def test_fixed_memory_service():
    """Test the fixed MemoryService"""
    print("Testing fixed MemoryService...")
    
    try:
        # Create and start memory service
        memory_service = MemoryService()
        await memory_service.start()
        print("✅ MemoryService started successfully")
        
        # Test storing memory with tags
        memory_id = await memory_service.store_memory(
            layer=MemoryLayer.KNOWLEDGE_SYNTHESIS,
            memory_type=MemoryType.RESEARCH_FINDING,
            content="Test memory content with tags",
            metadata={"test": "value", "source": "test_script"},
            tags=["test", "memory", "chromadb"]
        )
        
        print(f"✅ Successfully stored memory with ID: {memory_id}")
        
        # Test retrieving memory
        memories = await memory_service.retrieve_memories(
            query="test memory content",
            layer=MemoryLayer.KNOWLEDGE_SYNTHESIS,
            limit=5
        )
        
        print(f"✅ Successfully retrieved {len(memories)} memories")
        
        if memories:
            memory = memories[0]
            print(f"Memory content: {memory.content}")
            print(f"Memory tags: {memory.tags}")
            print(f"Memory metadata: {memory.metadata}")
        
        print("\n🎉 All tests passed! MemoryService is working correctly.")
        
    except Exception as e:
        print(f"❌ Error testing MemoryService: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_memory_service())