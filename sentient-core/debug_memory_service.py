#!/usr/bin/env python3
"""
Debug script to test MemoryService import and method availability
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=== MemoryService Debug Test ===")
print(f"Python path: {sys.path[:3]}...")
print(f"Current working directory: {os.getcwd()}")

try:
    print("\n1. Importing MemoryService...")
    from core.services.memory_service import MemoryService, MemoryLayer, MemoryType
    print("✓ Import successful")
    
    print("\n2. Creating MemoryService instance...")
    memory_service = MemoryService()
    print("✓ Instance created")
    
    print("\n3. Checking available methods...")
    methods = [method for method in dir(memory_service) if not method.startswith('_')]
    print(f"Available methods: {methods}")
    
    print("\n4. Checking specific methods...")
    has_store_memory = hasattr(memory_service, 'store_memory')
    has_retrieve_memories = hasattr(memory_service, 'retrieve_memories')
    has_get_memory_stats = hasattr(memory_service, 'get_memory_stats')
    
    print(f"Has store_memory: {has_store_memory}")
    print(f"Has retrieve_memories: {has_retrieve_memories}")
    print(f"Has get_memory_stats: {has_get_memory_stats}")
    
    if has_store_memory:
        print(f"store_memory method: {memory_service.store_memory}")
    
    if has_retrieve_memories:
        print(f"retrieve_memories method: {memory_service.retrieve_memories}")
        
    if has_get_memory_stats:
        print(f"get_memory_stats method: {memory_service.get_memory_stats}")
    
    print("\n5. Testing method signatures...")
    import inspect
    if has_store_memory:
        sig = inspect.signature(memory_service.store_memory)
        print(f"store_memory signature: {sig}")
    
    if has_retrieve_memories:
        sig = inspect.signature(memory_service.retrieve_memories)
        print(f"retrieve_memories signature: {sig}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug Test Complete ===")