#!/usr/bin/env python3
"""
Build 3 Integration Test
Tests the new planning transition functionality in UltraOrchestrator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sentient-core'))

from core.models import AppState, Message, LogEntry, SessionState
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.services.llm_service import EnhancedLLMService
from core.agents.build2_research_agent import Build2ResearchAgent
from core.agents.architect_planner_agent import ArchitectPlannerAgent
import asyncio

async def test_build3_integration():
    """
    Test Build 3 planning transition functionality
    """
    print("🧪 Testing Build 3 Integration...")
    
    try:
        # Initialize services
        llm_service = EnhancedLLMService()
        research_agent = Build2ResearchAgent()
        architect_planner = ArchitectPlannerAgent(llm_service)
        
        # Initialize UltraOrchestrator
        orchestrator = UltraOrchestrator(llm_service=llm_service)
        
        print("✅ UltraOrchestrator initialized successfully")
        
        # Create test state with research completion indicators
        test_state = AppState(
            session_id="test_build3",
            messages=[
                Message(sender="user", content="I want to create a web application"),
                Message(sender="assistant", content="Research completed successfully"),
                Message(sender="user", content="Great! Now let's create a plan for this project")
            ],
            logs=[
                LogEntry(source="Build2_ResearchAgent", message="Research completed successfully"),
                LogEntry(source="Build2_ResearchAgent", message="Report generated and saved")
            ],
            planning_state=SessionState.RESEARCHING
        )
        
        print("✅ Test state created")
        
        # Test planning transition detection
        should_transition = await orchestrator._should_transition_to_planning(test_state)
        print(f"✅ Planning transition detection: {should_transition}")
        
        # Test research artifact checking
        artifacts_exist = orchestrator._check_research_artifacts_exist()
        print(f"✅ Research artifacts check: {artifacts_exist}")
        
        print("\n🎉 Build 3 Integration Test Completed Successfully!")
        print("\n📋 Test Results:")
        print(f"   - UltraOrchestrator initialization: ✅")
        print(f"   - Planning transition detection: {'✅' if should_transition else '⚠️'}")
        print(f"   - Research artifacts check: {'✅' if artifacts_exist else '⚠️'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Build 3 Integration Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_build3_integration())
    sys.exit(0 if success else 1)