#!/usr/bin/env python3
"""
Direct test of MemoryService import and method availability
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from core.services.memory_service import MemoryService, MemoryLayer, MemoryType
    print("✅ Successfully imported MemoryService")
    
    # Create instance
    memory_service = MemoryService()
    print("✅ Successfully created MemoryService instance")
    
    # Check if store_memory method exists
    if hasattr(memory_service, 'store_memory'):
        print("✅ store_memory method exists")
        print(f"Method signature: {memory_service.store_memory.__doc__}")
    else:
        print("❌ store_memory method does NOT exist")
        print(f"Available methods: {[method for method in dir(memory_service) if not method.startswith('_')]}")
    
    # Check class definition
    print(f"\nMemoryService class: {MemoryService}")
    print(f"MemoryService MRO: {MemoryService.__mro__}")
    print(f"MemoryService module: {MemoryService.__module__}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")