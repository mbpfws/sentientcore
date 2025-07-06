#!/usr/bin/env python3
"""
Minimal test to identify the exact import causing the hang
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("Testing minimal imports step by step...")

try:
    print("\n1. Testing basic imports...")
    import asyncio
    print("✅ asyncio imported")
    
    print("\n2. Testing core.models...")
    from core.models import AppState, Message
    print("✅ core.models imported")
    
    print("\n3. Testing get_llm_service import only...")
    from core.graphs.sentient_workflow_graph import get_llm_service
    print("✅ get_llm_service imported")
    
    print("\n4. Testing get_ultra_orchestrator import only...")
    from core.graphs.sentient_workflow_graph import get_ultra_orchestrator
    print("✅ get_ultra_orchestrator imported")
    
    print("\n5. Testing workflow import only...")
    from core.graphs.sentient_workflow_graph import workflow
    print("✅ workflow imported")
    
    print("\n6. Testing get_sentient_workflow_app import only...")
    from core.graphs.sentient_workflow_graph import get_sentient_workflow_app
    print("✅ get_sentient_workflow_app imported")
    
    print("\n7. Testing get_compiled_workflow import only...")
    from core.graphs.sentient_workflow_graph import get_compiled_workflow
    print("✅ get_compiled_workflow imported")
    
    print("\n8. Testing sentient_workflow_app import only...")
    from core.graphs.sentient_workflow_graph import sentient_workflow_app
    print("✅ sentient_workflow_app imported")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All imports completed successfully!")
print("Now testing function calls...")

try:
    print("\n9. Testing get_llm_service() call...")
    llm_service = get_llm_service()
    print(f"✅ LLM Service created: {type(llm_service).__name__}")
    
except Exception as e:
    print(f"❌ Error in get_llm_service(): {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Test completed successfully!")