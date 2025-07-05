#!/usr/bin/env python3
"""
Test script to debug workflow initialization and UltraOrchestrator usage
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("Testing workflow initialization...")

try:
    print("\n1. Testing workflow graph import...")
    from core.graphs.sentient_workflow_graph import get_ultra_orchestrator, get_llm_service
    print("✅ Workflow graph imported successfully")
except Exception as e:
    print(f"❌ Workflow graph import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. Testing LLM service initialization...")
    llm_service = get_llm_service()
    print(f"✅ LLM Service initialized: {type(llm_service).__name__}")
except Exception as e:
    print(f"❌ LLM Service failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n3. Testing UltraOrchestrator initialization...")
    orchestrator = get_ultra_orchestrator()
    print(f"✅ UltraOrchestrator initialized: {type(orchestrator).__name__}")
except Exception as e:
    print(f"❌ UltraOrchestrator failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n4. Testing workflow compilation...")
    from core.graphs.sentient_workflow_graph import sentient_workflow_app
    print(f"✅ Workflow app compiled: {type(sentient_workflow_app).__name__}")
except Exception as e:
    print(f"❌ Workflow compilation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Workflow initialization test completed!")