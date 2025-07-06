#!/usr/bin/env python3
"""
Test script to reproduce and fix the conversation_history persistence issue.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import AppState, Message, LogEntry
from core.services.session_persistence_service import SessionPersistenceService
from core.graphs.sentient_workflow_graph import get_ultra_orchestrator, get_session_persistence, load_session_if_exists

async def test_conversation_history_persistence():
    """
    Test that conversation_history persists across session loads.
    """
    print("=== Testing Conversation History Persistence ===")
    
    # Initialize services using workflow functions
    session_persistence = get_session_persistence()
    orchestrator = get_ultra_orchestrator()
    
    session_id = "test_conv_history_123"
    
    # Step 1: Create initial state with a message
    print("\n1. Creating initial state...")
    initial_state = AppState(
        messages=[Message(sender="user", content="Hello, I need help with a project")],
        user_prompt="Hello, I need help with a project",
        conversation_history=[],
        session_id=session_id
    )
    
    print(f"Initial conversation_history: {initial_state.conversation_history}")
    
    # Step 2: Process through UltraOrchestrator (should add to conversation_history)
    print("\n2. Processing through UltraOrchestrator...")
    processed_state = await orchestrator.invoke_state(initial_state)
    
    print(f"After orchestrator - conversation_history: {processed_state.conversation_history}")
    print(f"Conversation history length: {len(processed_state.conversation_history)}")
    
    # Step 3: Save the state
    print("\n3. Saving state...")
    await session_persistence.save_session(session_id, processed_state)
    print("State saved successfully")
    
    # Step 4: Load the state using the workflow's load function
    print("\n4. Loading state using workflow function...")
    loaded_state = await load_session_if_exists(session_id)
    
    print(f"Loaded conversation_history: {loaded_state.conversation_history}")
    print(f"Loaded conversation history length: {len(loaded_state.conversation_history)}")
    
    # Step 5: Process another message through the loaded state
    print("\n5. Adding another message and processing...")
    loaded_state.messages.append(Message(sender="user", content="Can you help me build a web app?"))
    loaded_state.user_prompt = "Can you help me build a web app?"
    
    print(f"Before second processing - conversation_history: {loaded_state.conversation_history}")
    
    # Process through orchestrator again
    final_state = await orchestrator.invoke_state(loaded_state)
    
    print(f"After second processing - conversation_history: {final_state.conversation_history}")
    print(f"Final conversation history length: {len(final_state.conversation_history)}")
    
    # Step 6: Verify the issue
    print("\n=== ANALYSIS ===")
    if len(final_state.conversation_history) >= 2:
        print("✓ SUCCESS: Conversation history is persisting correctly")
        for i, entry in enumerate(final_state.conversation_history):
            print(f"  [{i}]: {entry}")
    else:
        print("✗ ISSUE: Conversation history is not persisting correctly")
        print(f"Expected at least 2 entries, got {len(final_state.conversation_history)}")
        
        # Debug: Check what's in the database
        print("\nChecking database directly...")
        db_state = await session_persistence.load_session(session_id)
        if db_state:
            print(f"Database conversation_history: {db_state.conversation_history}")
        else:
            print("No state found in database")
    
    return final_state

if __name__ == "__main__":
    asyncio.run(test_conversation_history_persistence())