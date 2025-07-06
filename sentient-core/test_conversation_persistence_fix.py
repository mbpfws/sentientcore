#!/usr/bin/env python3
"""
Test script to verify that conversation persistence and cumulative understanding is working correctly.
This tests the fix for the UltraOrchestrator accumulative understanding issue.
"""

import asyncio
import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import AppState, Message, LogEntry
from core.services.session_persistence_service import SessionPersistenceService

async def test_conversation_persistence():
    """
    Test that conversation history and cumulative understanding persists across sessions.
    """
    print("=== Testing Conversation Persistence Fix ===")
    
    # Initialize persistence service
    try:
        persistence_service = SessionPersistenceService()
        print("✓ SessionPersistenceService initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize persistence service: {e}")
        return False
    
    session_id = "test_persistence_session_002"
    
    # Test 1: Create initial state with conversation history
    print("\n--- Test 1: Save State with Conversation History ---")
    
    # Simulate what UltraOrchestrator would create
    conversation_history = [
        "Initial context: User wants to build an IELTS chatbot",
        "Building on previous context: User specified Google Gemini 2.5 Flash",
        "Enhanced summary: Comprehensive IELTS tutoring system with voice capabilities"
    ]
    
    state1 = AppState(
        session_id=session_id,
        messages=[
            Message(sender="user", content="I want to build a chatbot for IELTS English tutoring"),
            Message(sender="assistant", content="I understand you want to create an IELTS tutoring chatbot. Could you tell me more about the specific features you'd like?"),
            Message(sender="user", content="The chatbot should use Google Gemini 2.5 Flash with voice capabilities")
        ],
        conversation_history=conversation_history,
        tasks=[],
        logs=[
            LogEntry(source="UltraOrchestrator", message="Context progression: 3 cumulative understanding points tracked")
        ]
    )
    
    print(f"Original conversation history: {len(state1.conversation_history)} entries")
    for i, entry in enumerate(state1.conversation_history):
        print(f"  {i+1}: {entry}")
    
    # Save session
    save_success = await persistence_service.save_session(session_id, state1)
    print(f"Session save: {'✓ Success' if save_success else '✗ Failed'}")
    
    # Test 2: Load session and verify conversation history is preserved
    print("\n--- Test 2: Load Session and Verify Conversation History ---")
    loaded_state = await persistence_service.load_session(session_id)
    
    if loaded_state:
        print(f"✓ Session loaded successfully")
        print(f"Loaded conversation history: {len(loaded_state.conversation_history)} entries")
        for i, entry in enumerate(loaded_state.conversation_history):
            print(f"  {i+1}: {entry}")
        
        # Verify conversation history matches
        if loaded_state.conversation_history == conversation_history:
            print("✓ Conversation history preserved correctly")
            
            # Test 3: Simulate adding more conversation history
            print("\n--- Test 3: Add More Conversation History ---")
            loaded_state.conversation_history.append("Building on previous context: User confirmed voice interaction requirements")
            loaded_state.messages.append(
                Message(sender="user", content="Yes, I want voice input and audio output for speaking practice")
            )
            
            # Save updated state
            save_success2 = await persistence_service.save_session(session_id, loaded_state)
            print(f"Updated session save: {'✓ Success' if save_success2 else '✗ Failed'}")
            
            # Load again to verify accumulation
            final_state = await persistence_service.load_session(session_id)
            if final_state:
                print(f"Final conversation history: {len(final_state.conversation_history)} entries")
                for i, entry in enumerate(final_state.conversation_history):
                    print(f"  {i+1}: {entry}")
                
                if len(final_state.conversation_history) == 4:
                    print("✓ Conversation history accumulation working correctly")
                    return True
                else:
                    print(f"✗ Expected 4 entries, got {len(final_state.conversation_history)}")
                    return False
            else:
                print("✗ Failed to load final state")
                return False
        else:
            print("✗ Conversation history not preserved correctly")
            print(f"Expected: {conversation_history}")
            print(f"Got: {loaded_state.conversation_history}")
            return False
    else:
        print("✗ Failed to load session")
        return False

async def main():
    """
    Main test function
    """
    try:
        success = await test_conversation_persistence()
        if success:
            print("\n🎉 Conversation persistence fix is working correctly!")
            print("\nThe UltraOrchestrator should now maintain cumulative understanding across sessions.")
        else:
            print("\n❌ Conversation persistence fix needs more work")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())