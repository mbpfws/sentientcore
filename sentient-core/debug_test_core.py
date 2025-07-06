#!/usr/bin/env python3
"""
Debug script to test core functionality directly
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=== Testing Core System Components ===")

# Test 1: Memory Service
print("\n1. Testing MemoryService...")
try:
    from core.services.memory_service import MemoryService
    memory_service = MemoryService()
    print(f"✅ MemoryService initialized")
    print(f"   - Has db: {hasattr(memory_service, 'db')}")
    print(f"   - Has vector_service: {hasattr(memory_service, 'vector_service')}")
    print(f"   - Has layers: {hasattr(memory_service, 'layers')}")
    if hasattr(memory_service, 'layers'):
        print(f"   - Number of layers: {len(memory_service.layers)}")
except Exception as e:
    print(f"❌ MemoryService failed: {e}")

# Test 2: Enhanced LLM Service
print("\n2. Testing EnhancedLLMService...")
try:
    from core.services.enhanced_llm_service import EnhancedLLMService
    llm_service = EnhancedLLMService()
    print(f"✅ EnhancedLLMService initialized")
    print(f"   - Has providers: {hasattr(llm_service, 'providers')}")
    print(f"   - Has default_provider: {hasattr(llm_service, 'default_provider')}")
    if hasattr(llm_service, 'providers'):
        print(f"   - Number of providers: {len(llm_service.providers)}")
        print(f"   - Provider names: {list(llm_service.providers.keys())}")
except Exception as e:
    print(f"❌ EnhancedLLMService failed: {e}")

# Test 3: State Service
print("\n3. Testing StateService...")
try:
    from core.services.state_service import StateService
    state_service = StateService(db_path=":memory:")
    print(f"✅ StateService initialized")
    print(f"   - Has db_path: {hasattr(state_service, 'db_path')}")
    print(f"   - Has agent_states: {hasattr(state_service, 'agent_states')}")
except Exception as e:
    print(f"❌ StateService failed: {e}")

# Test 4: UltraOrchestrator
print("\n4. Testing UltraOrchestrator...")
try:
    from core.agents.ultra_orchestrator import UltraOrchestrator
    from core.services.enhanced_llm_service import EnhancedLLMService
    
    # Try with LLM service
    try:
        llm_service = EnhancedLLMService()
        orchestrator = UltraOrchestrator(llm_service)
        print(f"✅ UltraOrchestrator initialized with LLM service")
    except ValueError as e:
        if "No LLM providers available" in str(e):
            print(f"⚠️ UltraOrchestrator cannot initialize - no API keys available")
        else:
            raise e
except Exception as e:
    print(f"❌ UltraOrchestrator failed: {e}")

# Test 5: Memory Layer Structure
print("\n5. Testing Memory Layer Structure...")
try:
    layer1_path = Path("memory/layer1_research_docs")
    layer2_path = Path("memory/layer2_build_artifacts")
    
    print(f"   - Layer1 exists: {layer1_path.exists()}")
    print(f"   - Layer2 exists: {layer2_path.exists()}")
    
    if layer1_path.exists():
        layer1_files = list(layer1_path.glob("*.md"))
        print(f"   - Layer1 research docs: {len(layer1_files)}")
    
    if layer2_path.exists():
        layer2_files = list(layer2_path.glob("*.md"))
        print(f"   - Layer2 build artifacts: {len(layer2_files)}")
        
except Exception as e:
    print(f"❌ Memory layer structure test failed: {e}")

print("\n=== Core System Test Complete ===")