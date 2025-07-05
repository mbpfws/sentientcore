#!/usr/bin/env python3
"""
Test workflow compilation to identify the exact issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("Testing workflow compilation step by step...")

try:
    print("\n1. Testing imports...")
    from core.graphs.sentient_workflow_graph import get_llm_service, get_ultra_orchestrator
    print("✅ Functions imported")
    
    print("\n2. Testing LLM service creation...")
    llm_service = get_llm_service()
    print(f"✅ LLM Service: {type(llm_service).__name__}")
    
    print("\n3. Testing UltraOrchestrator creation...")
    orchestrator = get_ultra_orchestrator()
    print(f"✅ UltraOrchestrator: {type(orchestrator).__name__}")
    
    print("\n4. Testing workflow graph import...")
    from core.graphs.sentient_workflow_graph import workflow
    print("✅ Workflow graph imported")
    
    print("\n5. Testing workflow compilation...")
    from core.graphs.sentient_workflow_graph import sentient_workflow_app
    print(f"✅ Workflow app: {type(sentient_workflow_app).__name__}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Workflow compilation test completed successfully!")