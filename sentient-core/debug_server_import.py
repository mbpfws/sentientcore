#!/usr/bin/env python3
"""
Debug script to test the exact import path used by the FastAPI server
"""

import sys
import os
from pathlib import Path

# Print current working directory and Python path
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:5]}")
print(f"Script location: {__file__}")
print(f"Script directory: {os.path.dirname(__file__)}")

# Test the exact import path that the FastAPI app uses
print("\n=== Testing FastAPI App Import Path ===")

try:
    # Change to the app directory like the FastAPI server
    app_dir = os.path.join(os.path.dirname(__file__), "app")
    print(f"App directory: {app_dir}")
    
    # Add the parent directory to sys.path (like the server does)
    parent_dir = os.path.dirname(__file__)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added to sys.path: {parent_dir}")
    
    # Now try the import exactly as the FastAPI app does
    print("\nImporting from app.api.routers.core_services...")
    from app.api.routers.core_services import MemoryService
    
    print(f"MemoryService imported: {MemoryService}")
    print(f"MemoryService module: {MemoryService.__module__}")
    print(f"MemoryService file: {MemoryService.__module__.replace('.', '/')}.py")
    
    # Check methods
    print(f"\nHas store_memory: {hasattr(MemoryService, 'store_memory')}")
    print(f"Has get_memory_stats: {hasattr(MemoryService, 'get_memory_stats')}")
    
    if hasattr(MemoryService, 'store_memory'):
        print(f"store_memory type: {type(MemoryService.store_memory)}")
    
    if hasattr(MemoryService, 'get_memory_stats'):
        print(f"get_memory_stats type: {type(MemoryService.get_memory_stats)}")
    
    # Create an instance and test
    print("\nCreating instance...")
    instance = MemoryService()
    print(f"Instance: {instance}")
    print(f"Instance type: {type(instance)}")
    print(f"Instance has store_memory: {hasattr(instance, 'store_memory')}")
    print(f"Instance has get_memory_stats: {hasattr(instance, 'get_memory_stats')}")
    
    # List all methods
    methods = [attr for attr in dir(instance) if not attr.startswith('_') and callable(getattr(instance, attr))]
    print(f"\nAll public methods: {methods}")
    
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()

# Also test direct import
print("\n=== Testing Direct Import ===")
try:
    from core.services.memory_service import MemoryService as DirectMemoryService
    print(f"Direct MemoryService: {DirectMemoryService}")
    print(f"Direct has store_memory: {hasattr(DirectMemoryService, 'store_memory')}")
    print(f"Direct has get_memory_stats: {hasattr(DirectMemoryService, 'get_memory_stats')}")
    
    # Compare the two classes
    if 'MemoryService' in locals():
        print(f"\nClass comparison:")
        print(f"FastAPI == Direct: {MemoryService == DirectMemoryService}")
        print(f"FastAPI is Direct: {MemoryService is DirectMemoryService}")
        print(f"FastAPI id: {id(MemoryService)}")
        print(f"Direct id: {id(DirectMemoryService)}")
        
except Exception as e:
    print(f"Direct import failed: {e}")
    import traceback
    traceback.print_exc()