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
    
    print("\n5. Testing LazyWorkflowApp...")
    from core.graphs.sentient_workflow_graph import sentient_workflow_app
    print(f"✅ Workflow app: {type(sentient_workflow_app).__name__}")
    
    print("\n6. Testing direct workflow compilation...")
    from core.graphs.sentient_workflow_graph import get_sentient_workflow_app
    print("Calling get_sentient_workflow_app() directly...")
    direct_app = get_sentient_workflow_app()
    print(f"✅ Direct app: {type(direct_app).__name__}")
    
    print("\n7. Testing get_compiled_workflow...")
    from core.graphs.sentient_workflow_graph import get_compiled_workflow
    print("Calling get_compiled_workflow()...")
    compiled_app = get_compiled_workflow()
    print(f"✅ Compiled app: {type(compiled_app).__name__}")
    
    print("\n8. Testing actual workflow execution...")
    from core.models import AppState, Message
    import asyncio
    
    async def test_workflow():
        print("Creating test state...")
        test_state = AppState()
        test_state.messages.append(Message(sender="user", content="Hello, test message"))
        
        print("Invoking workflow...")
        result = await compiled_app.ainvoke(test_state)
        print(f"✅ Workflow executed! Result type: {type(result).__name__}")
        print(f"Messages in result: {len(result.messages)}")
        return result
    
    print("Running async workflow test...")
    result = asyncio.run(test_workflow())
    print(f"✅ Async test completed with {len(result.messages)} messages")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Workflow compilation test completed successfully!")