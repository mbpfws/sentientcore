#!/usr/bin/env python3

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.services.memory_service import MemoryService
import inspect

def debug_memory_service():
    """Debug MemoryService class and instance"""
    print("=== MemoryService Class Debug ===")
    
    # Check class definition
    print(f"MemoryService class: {MemoryService}")
    print(f"MemoryService module: {MemoryService.__module__}")
    print(f"MemoryService file: {inspect.getfile(MemoryService)}")
    
    # List all class methods
    print("\n=== Class Methods ===")
    for name, method in inspect.getmembers(MemoryService, predicate=inspect.isfunction):
        print(f"  {name}: {method}")
    
    # Create instance
    print("\n=== Creating Instance ===")
    try:
        service = MemoryService()
        print(f"Instance created: {service}")
        print(f"Instance type: {type(service)}")
        
        # Check instance methods
        print("\n=== Instance Methods ===")
        for name in dir(service):
            if not name.startswith('_'):
                attr = getattr(service, name)
                print(f"  {name}: {type(attr)} - {attr}")
        
        # Specifically check store_memory
        print("\n=== store_memory Method Check ===")
        if hasattr(service, 'store_memory'):
            store_memory = getattr(service, 'store_memory')
            print(f"store_memory exists: {store_memory}")
            print(f"store_memory type: {type(store_memory)}")
            print(f"store_memory signature: {inspect.signature(store_memory)}")
        else:
            print("store_memory method NOT FOUND!")
            
    except Exception as e:
        print(f"Error creating instance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_memory_service()