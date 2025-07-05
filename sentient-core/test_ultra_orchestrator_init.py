#!/usr/bin/env python3
"""
Test script to debug UltraOrchestrator initialization issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("Testing UltraOrchestrator initialization...")
print(f"GROQ_API_KEY set: {'GROQ_API_KEY' in os.environ and os.environ['GROQ_API_KEY'] != ''}")

try:
    print("\n1. Testing EnhancedLLMService initialization...")
    from core.services.enhanced_llm_service_main import EnhancedLLMService
    llm_service = EnhancedLLMService()
    print("✅ EnhancedLLMService initialized successfully")
except Exception as e:
    print(f"❌ EnhancedLLMService failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. Testing Build2ResearchAgent initialization...")
    from core.agents.build2_research_agent import Build2ResearchAgent
    research_agent = Build2ResearchAgent()
    print("✅ Build2ResearchAgent initialized successfully")
except Exception as e:
    print(f"❌ Build2ResearchAgent failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. Testing UltraOrchestrator initialization...")
    from core.agents.ultra_orchestrator import UltraOrchestrator
    orchestrator = UltraOrchestrator(llm_service)
    print("✅ UltraOrchestrator initialized successfully")
except Exception as e:
    print(f"❌ UltraOrchestrator failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All components initialized successfully!")
print("The issue might be in the get_ultra_orchestrator() function or during FastAPI startup.")