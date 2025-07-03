#!/usr/bin/env python3
"""
Test script for the Enhanced LLM Service with OpenAI provider and tracking capabilities.
"""

import asyncio
import os
from dotenv import load_dotenv
from core.services.llm_service import EnhancedLLMService

# Load environment variables
load_dotenv()

async def test_enhanced_llm_service():
    """Test the enhanced LLM service with multiple providers and tracking."""
    
    print("=== Testing Enhanced LLM Service ===")
    
    # Initialize the service
    llm_service = EnhancedLLMService()
    
    # Test basic functionality
    print("\n1. Testing basic generation...")
    try:
        response = await llm_service.invoke(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello in a friendly way.",
            model="llama3-8b-8192"  # Groq model
        )
        print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"Error in basic generation: {e}")
    
    # Test OpenAI model (if API key is available)
    print("\n2. Testing OpenAI model...")
    if os.getenv("OPENAI_API_KEY"):
        try:
            response = await llm_service.invoke(
                system_prompt="You are a helpful assistant.",
                user_prompt="Explain what AI is in one sentence.",
                model="gpt-4o-mini"
            )
            print(f"OpenAI Response: {response[:100]}...")
        except Exception as e:
            print(f"Error with OpenAI: {e}")
    else:
        print("OpenAI API key not found, skipping OpenAI test.")
    
    # Test fallback mechanism
    print("\n3. Testing fallback with invalid model...")
    try:
        response = await llm_service.invoke(
            system_prompt="You are a helpful assistant.",
            user_prompt="This should fallback to a working provider.",
            model="invalid-model-name"
        )
        print(f"Fallback Response: {response[:100]}...")
    except Exception as e:
        print(f"Error in fallback test: {e}")
    
    # Test streaming
    print("\n4. Testing streaming...")
    try:
        print("Streaming response: ", end="")
        async for chunk in await llm_service.invoke(
            system_prompt="You are a helpful assistant.",
            user_prompt="Count from 1 to 5.",
            model="llama3-8b-8192",
            stream=True
        ):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error in streaming test: {e}")
    
    # Display usage statistics
    print("\n5. Usage Statistics:")
    stats = llm_service.get_usage_statistics()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total errors: {stats['total_errors']}")
    print(f"Usage by provider: {stats['usage_by_provider']}")
    print(f"Errors by provider: {stats['errors_by_provider']}")
    print(f"Average response times: {stats['average_response_times']}")
    print(f"Recent requests tracked: {stats['recent_requests_count']}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_enhanced_llm_service())