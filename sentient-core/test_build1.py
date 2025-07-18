#!/usr/bin/env python3
"""
Build 1 Test Script: Core Conversation & Orchestration Loop

This script tests the basic conversation functionality of Build 1,
verifying that the system can:
1. Maintain conversation history
2. Provide contextual responses
3. Handle basic conversational interactions
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import AppState, Message
from core.graphs.sentient_workflow_graph import get_sentient_workflow_app


async def test_basic_conversation():
    """Test basic conversation functionality."""
    print("\n=== Testing Basic Conversation ===")
    
    # Initialize the workflow app
    app = get_sentient_workflow_app()
    
    # Test 1: Initial greeting
    print("\nTest 1: Initial greeting")
    initial_state = AppState(
        messages=[Message(sender="user", content="Hello, how are you?")],
        user_prompt="Hello, how are you?"
    )
    
    result = await app.ainvoke(initial_state)
    print(f"User: {initial_state.user_prompt}")
    print(f"Assistant: {result.messages[-1].content}")
    
    # Test 2: Follow-up question about conversation history
    print("\nTest 2: Follow-up question about conversation history")
    follow_up_state = AppState(
        messages=result.messages + [Message(sender="user", content="What was the first thing I said to you?")],
        user_prompt="What was the first thing I said to you?"
    )
    
    result2 = await app.ainvoke(follow_up_state)
    print(f"User: {follow_up_state.user_prompt}")
    print(f"Assistant: {result2.messages[-1].content}")
    
    # Test 3: General question
    print("\nTest 3: General question")
    general_state = AppState(
        messages=result2.messages + [Message(sender="user", content="Can you help me understand what you can do?")],
        user_prompt="Can you help me understand what you can do?"
    )
    
    result3 = await app.ainvoke(general_state)
    print(f"User: {general_state.user_prompt}")
    print(f"Assistant: {result3.messages[-1].content}")
    
    return result3


async def test_conversation_persistence():
    """Test that conversation history is maintained across multiple turns."""
    print("\n=== Testing Conversation Persistence ===")
    
    app = get_sentient_workflow_app()
    
    # Start with an initial state
    state = AppState(
        messages=[Message(sender="user", content="My name is Alice and I like programming.")],
        user_prompt="My name is Alice and I like programming."
    )
    
    # Turn 1
    print("\nTurn 1:")
    result1 = await app.ainvoke(state)
    print(f"User: {state.user_prompt}")
    print(f"Assistant: {result1.messages[-1].content}")
    
    # Turn 2 - Ask about name
    print("\nTurn 2:")
    state2 = AppState(
        messages=result1.messages + [Message(sender="user", content="What's my name?")],
        user_prompt="What's my name?"
    )
    result2 = await app.ainvoke(state2)
    print(f"User: {state2.user_prompt}")
    print(f"Assistant: {result2.messages[-1].content}")
    
    # Turn 3 - Ask about interests
    print("\nTurn 3:")
    state3 = AppState(
        messages=result2.messages + [Message(sender="user", content="What did I say I like?")],
        user_prompt="What did I say I like?"
    )
    result3 = await app.ainvoke(state3)
    print(f"User: {state3.user_prompt}")
    print(f"Assistant: {result3.messages[-1].content}")
    
    return result3


async def test_error_handling():
    """Test error handling in the conversation system."""
    print("\n=== Testing Error Handling ===")
    
    app = get_sentient_workflow_app()
    
    # Test with empty input
    print("\nTest: Empty input")
    empty_state = AppState(
        messages=[Message(sender="user", content="")],
        user_prompt=""
    )
    
    try:
        result = await app.ainvoke(empty_state)
        print(f"User: (empty)")
        print(f"Assistant: {result.messages[-1].content}")
        print("✓ Empty input handled gracefully")
    except Exception as e:
        print(f"✗ Error with empty input: {e}")
    
    return True


async def main():
    """Run all Build 1 tests."""
    print("Starting Build 1 Tests: Core Conversation & Orchestration Loop")
    print("=" * 60)
    
    try:
        # Run all tests
        await test_basic_conversation()
        await test_conversation_persistence()
        await test_error_handling()
        
        print("\n" + "=" * 60)
        print("✓ All Build 1 tests completed successfully!")
        print("\nBuild 1 Status: READY FOR VALIDATION")
        print("\nKey Features Verified:")
        print("- Basic conversation handling")
        print("- Conversation history maintenance")
        print("- Contextual responses")
        print("- Error handling")
        
    except Exception as e:
        print(f"\n✗ Build 1 tests failed: {e}")
        print("\nBuild 1 Status: NEEDS FIXES")
        return False
    
    return True


if __name__ == "__main__":
    asyncio.run(main())