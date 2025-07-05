#!/usr/bin/env python3
"""
Test individual imports to isolate the problematic import
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("Testing individual imports...")

try:
    print("\n1. Testing core.models import...")
    from core.models import AppState, Message, LogEntry
    print("✅ core.models imported")
    
    print("\n2. Testing langgraph import...")
    from langgraph.graph import StateGraph, END
    print("✅ langgraph imported")
    
    print("\n3. Testing UltraOrchestrator import...")
    from core.agents.ultra_orchestrator import UltraOrchestrator
    print("✅ UltraOrchestrator imported")
    
    print("\n4. Testing EnhancedLLMService import...")
    from core.services.enhanced_llm_service_main import EnhancedLLMService
    print("✅ EnhancedLLMService imported")
    
    print("\n5. Testing SessionPersistenceService import...")
    from core.services.session_persistence_service import SessionPersistenceService
    print("✅ SessionPersistenceService imported")
    
    print("\n6. Testing StateGraph creation...")
    workflow = StateGraph(AppState)
    print("✅ StateGraph created")
    
    print("\n7. Testing workflow node addition...")
    def dummy_node(state: AppState) -> AppState:
        return state
    workflow.add_node("test", dummy_node)
    workflow.set_entry_point("test")
    workflow.add_edge("test", END)
    print("✅ Workflow nodes added")
    
    print("\n8. Testing workflow compilation...")
    compiled_workflow = workflow.compile()
    print("✅ Workflow compiled")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Individual imports test completed successfully!")