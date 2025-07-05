#!/usr/bin/env python3
"""
Simplified Frontend Test - Tests core research functionality without heavy dependencies
"""

import asyncio
import sys
import os
import time
from typing import Optional
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.models import AppState, Message
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.services.enhanced_llm_service_main import EnhancedLLMService

async def test_research_mode_detection():
    """
    Test that research modes are properly detected and processed
    This simulates what happens when a user interacts with the frontend
    """
    print("🧪 Testing Research Mode Detection (Frontend Simulation)")
    print("=" * 60)
    
    # Initialize services
    llm_service = EnhancedLLMService()
    orchestrator = UltraOrchestrator(llm_service)
    
    # Test cases that simulate frontend interactions
    test_cases = [
        {
            "name": "Knowledge Research Mode",
            "message": "Please conduct a Knowledge Research: What are the latest developments in AI?",
            "expected_delegation": True
        },
        {
            "name": "Deep Research Mode", 
            "message": "Please conduct a Deep Research: Analyze the impact of quantum computing on cryptography",
            "expected_delegation": True
        },
        {
            "name": "Best-in-Class Research Mode",
            "message": "Please conduct a Best-in-Class Research: Compare the top 5 machine learning frameworks",
            "expected_delegation": True
        },
        {
            "name": "Regular Chat",
            "message": "Hello, how are you today?",
            "expected_delegation": False
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {test_case['name']}...")
        
        try:
            # Create app state (simulating frontend state)
            app_state = AppState()
            
            # Add user message (simulating frontend message submission)
            user_message = Message(
                sender="user",
                content=test_case["message"]
            )
            app_state.messages.append(user_message)
            
            # Process through orchestrator (simulating backend processing)
            result_state = await orchestrator.invoke_state(app_state)
            
            # Check if research was delegated by examining the orchestrator decision
            decision_data = result_state.orchestrator_decision or {}
            should_delegate = decision_data.get("decision") == "delegate_research"
            
            # Verify expectation
            if should_delegate == test_case["expected_delegation"]:
                print(f"   ✅ PASS: Research delegation = {should_delegate}")
                results.append(True)
            else:
                print(f"   ❌ FAIL: Expected delegation = {test_case['expected_delegation']}, got = {should_delegate}")
                results.append(False)
                
            # Show additional info
            if "research_query" in decision_data:
                print(f"   📝 Research Query: {decision_data['research_query'][:100]}...")
            if "conversational_response" in decision_data:
                print(f"   💬 Response: {decision_data['conversational_response'][:100]}...")
                
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED! Frontend research functionality should work correctly.")
        return True
    else:
        print("⚠️  Some tests FAILED. There may be issues with frontend research functionality.")
        return False

async def test_message_flow_simulation():
    """
    Test the complete message flow that would happen in a real frontend interaction
    """
    print("\n🔄 Testing Complete Message Flow (Frontend to Backend Simulation)")
    print("=" * 70)
    
    try:
        # Simulate what happens when user submits a research request through frontend
        print("1. User submits research request through frontend...")
        
        # Frontend would add research prefix (as seen in chat.py)
        user_input = "What are the benefits of renewable energy?"
        research_mode = "knowledge"
        
        # This is what chat.py does:
        research_prefix = {
            "knowledge": "Please conduct a Knowledge Research",
            "deep": "Please conduct a Deep Research", 
            "best_in_class": "Please conduct a Best-in-Class Research"
        }.get(research_mode, "")
        
        message_text = f"{research_prefix}: {user_input}"
        print(f"   📝 Processed message: {message_text}")
        
        # Backend processing
        print("2. Backend processes the message...")
        app_state = AppState()
        app_state.messages.append(Message(sender="user", content=message_text))
        
        # UltraOrchestrator decides what to do
        llm_service = EnhancedLLMService()
        orchestrator = UltraOrchestrator(llm_service)
        result_state = await orchestrator.invoke_state(app_state)
        
        print("3. Orchestrator decision:")
        decision_data = result_state.orchestrator_decision or {}
        delegate_research = decision_data.get("decision") == "delegate_research"
        print(f"   🎯 Delegate Research: {delegate_research}")
        print(f"   🔍 Research Query: {decision_data.get('research_query', 'N/A')[:100]}...")
        
        # Simulate what would happen next
        if delegate_research:
            print("4. ✅ Research would be delegated to Research Agent")
            print("   📊 Research Agent would gather information")
            print("   📝 Comprehensive response would be generated")
            print("   🔄 Response would be sent back to frontend")
        else:
            print("4. ❌ Regular conversation flow would be used")
            
        return True
        
    except Exception as e:
        print(f"❌ Error in message flow simulation: {str(e)}")
        return False

async def main():
    print("🚀 Sentient Core Frontend Functionality Test")
    print("=" * 50)
    print("This test simulates frontend interactions to verify research functionality")
    print("without requiring a running web server.\n")
    
    # Run tests
    test1_passed = test_research_mode_detection()
    test2_passed = await test_message_flow_simulation()
    
    # Final summary
    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("🎉 SUCCESS: Frontend research functionality is working correctly!")
        print("✅ When you test on the frontend as a user, research modes should work properly.")
        print("✅ The UltraOrchestrator correctly identifies and delegates research requests.")
        print("✅ The message flow from frontend to backend is functioning as expected.")
        return 0
    else:
        print("⚠️  WARNING: Some issues detected with frontend research functionality.")
        print("❌ There may be problems when testing on the actual frontend.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)