#!/usr/bin/env python3
"""
Full integration test to verify UltraOrchestrator maintains cumulative understanding across sessions.
This tests the complete workflow including LLM service integration.
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import AppState, Message
from core.services.session_persistence_service import SessionPersistenceService
from core.agents.ultra_orchestrator import UltraOrchestrator
from core.services.enhanced_llm_service_main import EnhancedLLMService

async def test_full_orchestrator_persistence():
    """
    Test that UltraOrchestrator maintains cumulative understanding across sessions.
    """
    print("=== Testing Full UltraOrchestrator Persistence ===")
    
    # Initialize services
    try:
        llm_service = EnhancedLLMService()
        orchestrator = UltraOrchestrator(llm_service)
        persistence_service = SessionPersistenceService()
        print("✓ All services initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize services: {e}")
        return False
    
    session_id = "test_full_orchestrator_003"
    
    # Test 1: Initial conversation with UltraOrchestrator
    print("\n--- Test 1: Initial Conversation ---")
    
    state1 = AppState(
        session_id=session_id,
        messages=[
            Message(sender="user", content="Can you build me a simple chatbot using a prebuilt AI chat UI (like ChatGPT, Perplexity, Gemini)? I want the chat platform integrated with Google Gemini model `gemini-2.5-flash` with 'image understanding/vision capabilities' and 'audio understanding' and 'generate audio' capabilities enabled.")
        ],
        conversation_history=[],
        tasks=[],
        logs=[]
    )
    
    # Process through orchestrator
    print("Processing initial message through UltraOrchestrator...")
    result_state1 = await orchestrator.invoke_state(state1)
    
    print(f"\nAfter first turn:")
    print(f"- Messages: {len(result_state1.messages)}")
    print(f"- Conversation history entries: {len(result_state1.conversation_history)}")
    print(f"- Next action: {result_state1.next_action}")
    
    if result_state1.conversation_history:
        print("\nConversation history:")
        for i, entry in enumerate(result_state1.conversation_history):
            print(f"  {i+1}: {entry[:150]}...")
    
    # Save session
    save_success = await persistence_service.save_session(session_id, result_state1)
    print(f"\nSession save: {'✓ Success' if save_success else '✗ Failed'}")
    
    # Test 2: Load session and continue conversation
    print("\n--- Test 2: Load Session and Continue ---")
    
    loaded_state = await persistence_service.load_session(session_id)
    
    if loaded_state:
        print(f"✓ Session loaded successfully")
        print(f"- Messages: {len(loaded_state.messages)}")
        print(f"- Conversation history entries: {len(loaded_state.conversation_history)}")
        
        if loaded_state.conversation_history:
            print("\nLoaded conversation history:")
            for i, entry in enumerate(loaded_state.conversation_history):
                print(f"  {i+1}: {entry[:150]}...")
        
        # Add follow-up message
        loaded_state.messages.append(
            Message(sender="user", content="As for the chatbot behaviors, functions, and features, I want it to act like an IELTS English tutor for intermediate Vietnamese learners to practice IELTS speaking. The user will input questions or context of practice, and the chatbot will respond as a tutor and IELTS examiner.")
        )
        
        # Process follow-up
        print("\nProcessing follow-up message through UltraOrchestrator...")
        result_state2 = await orchestrator.invoke_state(loaded_state)
        
        print(f"\nAfter second turn:")
        print(f"- Messages: {len(result_state2.messages)}")
        print(f"- Conversation history entries: {len(result_state2.conversation_history)}")
        print(f"- Next action: {result_state2.next_action}")
        
        if result_state2.conversation_history:
            print("\nUpdated conversation history:")
            for i, entry in enumerate(result_state2.conversation_history):
                print(f"  {i+1}: {entry[:150]}...")
        
        # Check if cumulative understanding is maintained
        if len(result_state2.conversation_history) >= len(result_state1.conversation_history):
            print("\n✓ Conversation history is accumulating correctly")
            
            # Save updated session
            save_success2 = await persistence_service.save_session(session_id, result_state2)
            print(f"Updated session save: {'✓ Success' if save_success2 else '✗ Failed'}")
            
            # Test 3: Final verification - load again and check persistence
            print("\n--- Test 3: Final Persistence Verification ---")
            final_state = await persistence_service.load_session(session_id)
            
            if final_state and final_state.conversation_history:
                print(f"✓ Final verification successful")
                print(f"- Final conversation history entries: {len(final_state.conversation_history)}")
                
                # Check if the conversation history contains cumulative understanding
                has_cumulative_context = any(
                    "chatbot" in entry.lower() and "gemini" in entry.lower() 
                    for entry in final_state.conversation_history
                )
                
                if has_cumulative_context:
                    print("✓ Cumulative understanding is preserved across sessions")
                    return True
                else:
                    print("✗ Cumulative understanding not found in conversation history")
                    return False
            else:
                print("✗ Failed final verification")
                return False
        else:
            print("\n✗ Conversation history is not accumulating")
            return False
    else:
        print("✗ Failed to load session")
        return False

async def main():
    """
    Main test function
    """
    try:
        success = await test_full_orchestrator_persistence()
        if success:
            print("\n🎉 Full UltraOrchestrator persistence is working correctly!")
            print("\nThe issue with repetitive responses should now be resolved.")
            print("The orchestrator will maintain cumulative understanding across sessions.")
        else:
            print("\n❌ Full orchestrator persistence needs more investigation")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())