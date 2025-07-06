#!/usr/bin/env python3
"""
Debug script to check memory layers
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=== Debugging Memory Layers ===")

# Test MemoryLayer enum
try:
    from core.models import MemoryLayer
    print(f"\nMemoryLayer enum values:")
    for layer in MemoryLayer:
        print(f"  - {layer.name}: {layer.value}")
except Exception as e:
    print(f"❌ Failed to import MemoryLayer: {e}")

# Test MemoryService layers
try:
    from core.services.memory_service import MemoryService
    memory_service = MemoryService()
    
    print(f"\nMemoryService layers:")
    print(f"  - Type: {type(memory_service.layers)}")
    print(f"  - Keys: {list(memory_service.layers.keys())}")
    
    print(f"\nChecking layer presence:")
    for layer in MemoryLayer:
        is_present = layer in memory_service.layers
        print(f"  - {layer.name}: {is_present}")
        if not is_present:
            print(f"    ❌ Missing layer: {layer}")
            
except Exception as e:
    print(f"❌ Failed to test MemoryService: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug Complete ===")