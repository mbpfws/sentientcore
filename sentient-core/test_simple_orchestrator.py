#!/usr/bin/env python3
"""
Simple test to isolate UltraOrchestrator initialization issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("Testing simple UltraOrchestrator initialization...")

try:
    print("\n1. Importing EnhancedLLMService...")
    from core.services.enhanced_llm_service_main import EnhancedLLMService
    print("✅ EnhancedLLMService imported")
    
    print("\n2. Creating EnhancedLLMService instance...")
    llm_service = EnhancedLLMService()
    print("✅ EnhancedLLMService created")
    
    print("\n3. Importing UltraOrchestrator...")
    from core.agents.ultra_orchestrator import UltraOrchestrator
    print("✅ UltraOrchestrator imported")
    
    print("\n4. Creating UltraOrchestrator instance...")
    orchestrator = UltraOrchestrator(llm_service)
    print("✅ UltraOrchestrator created successfully!")
    print(f"   Type: {type(orchestrator).__name__}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 Simple UltraOrchestrator test completed successfully!")