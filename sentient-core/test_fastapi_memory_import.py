#!/usr/bin/env python3
"""
Test script to check MemoryService import in FastAPI context
"""

import os
import sys

# Mimic the same path setup as in the FastAPI router
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "app", "api", "routers", "../../.."))
print(f"Project root from FastAPI context: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Python path:")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

try:
    from core.services.memory_service import MemoryService, MemoryLayer, MemoryType
    print(f"\n✅ Successfully imported MemoryService from FastAPI context")
    print(f"MemoryService class: {MemoryService}")
    print(f"MemoryService module: {MemoryService.__module__}")
    print(f"MemoryService file: {MemoryService.__module__.__file__ if hasattr(MemoryService.__module__, '__file__') else 'N/A'}")
    
    # Create instance and check for store_memory method
    memory_service = MemoryService()
    print(f"\n✅ Successfully created MemoryService instance")
    
    if hasattr(memory_service, 'store_memory'):
        print(f"✅ store_memory method exists")
        print(f"Method: {memory_service.store_memory}")
    else:
        print(f"❌ store_memory method does NOT exist")
        print(f"Available methods: {[method for method in dir(memory_service) if not method.startswith('_')]}")
        
except Exception as e:
    print(f"❌ Error importing MemoryService: {e}")
    import traceback
    traceback.print_exc()