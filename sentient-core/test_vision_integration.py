#!/usr/bin/env python3
"""
Integration test to demonstrate vision capabilities with Groq prioritization
in the existing Sentient-Core chat system.
"""

import asyncio
import base64
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.models import Message, AppState
from core.services.enhanced_llm_service_main import get_enhanced_llm_service


def create_sample_image() -> bytes:
    """
    Create a simple test image for vision testing.
    This creates a small PNG with some basic content.
    """
    # A simple 8x8 red square PNG in base64
    red_square_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz"
        "AAAB2AAAAdgB+lymcgAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAAdSURB"
        "VBiVY/z//z8DAwMDEwMDw38GBgYGJgYGBgYAFgAH/A8ECQAAAABJRU5ErkJggg=="
    )
    return base64.b64decode(red_square_base64)


async def test_vision_with_message_model():
    """
    Test vision capabilities using the existing Message and AppState models.
    """
    print("🔍 Testing Vision Integration with Message Model...")
    
    # Create a test image
    test_image = create_sample_image()
    
    # Create a message with image (using existing Message model)
    message = Message(
        sender="user",
        content="What do you see in this image? Please describe the colors and shapes.",
        image=test_image  # The Message model already supports image bytes!
    )
    
    print(f"📝 Created message with image: {len(test_image)} bytes")
    print(f"💬 Message content: {message.content}")
    
    # Get the enhanced LLM service
    llm_service = get_enhanced_llm_service()
    
    try:
        # Test the vision request using the generate method
        response = await llm_service.generate(
            prompt=message.content,
            image_bytes=message.image,
            context_id="vision_test_context"
        )
        
        print(f"🤖 Vision Response: {response}")
        print("✅ Vision integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Vision integration test failed: {e}")
        return False


async def test_appstate_with_vision():
    """
    Test vision capabilities with the AppState model.
    """
    print("\n🏗️  Testing Vision with AppState Model...")
    
    # Create test image
    test_image = create_sample_image()
    
    # Create an AppState with image
    app_state = AppState(
        user_prompt="Analyze the provided image",
        image=test_image  # AppState also supports image!
    )
    
    # Add a message to the conversation history
    vision_message = Message(
        sender="user",
        content="Please analyze this image and tell me what you observe.",
        image=test_image
    )
    
    app_state.messages.append(vision_message)
    
    print(f"📊 AppState created with image: {len(app_state.image)} bytes")
    print(f"💭 User prompt: {app_state.user_prompt}")
    print(f"📚 Messages length: {len(app_state.messages)}")
    
    # Get LLM service and test
    llm_service = get_enhanced_llm_service()
    
    try:
        # Use the image from AppState
        response = await llm_service.generate(
            prompt=vision_message.content,
            image_bytes=app_state.image,
            context_id="appstate_vision_test"
        )
        
        # Add response to conversation history
        response_message = Message(
            sender="assistant",
            content=response
        )
        app_state.messages.append(response_message)
        
        print(f"🤖 Assistant Response: {response}")
        print(f"📚 Updated messages length: {len(app_state.messages)}")
        print("✅ AppState vision integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ AppState vision integration test failed: {e}")
        return False


async def test_groq_model_selection():
    """
    Test that Groq models are properly selected for vision tasks.
    """
    print("\n🎯 Testing Groq Model Selection for Vision...")
    
    llm_service = get_enhanced_llm_service()
    
    # Test optimal model selection for vision
    from core.services.enhanced_llm_service import ModelCapability
    
    try:
        optimal_model, optimal_provider = llm_service._get_optimal_model([ModelCapability.VISION])
        print(f"🤖 Selected model: {optimal_model}")
        print(f"🏢 Selected provider: {optimal_provider}")
        
        # Verify it's a Groq model
        if optimal_provider == 'groq':
            print("✅ Groq provider correctly selected for vision tasks!")
            
            # Check if it's one of our new vision models
            if 'llama-4-scout' in optimal_model or 'llama-4-maverick' in optimal_model:
                print(f"✅ New Groq vision model selected: {optimal_model}")
            else:
                print(f"ℹ️  Other Groq model selected: {optimal_model}")
                
            return True
        else:
            print(f"⚠️  Expected Groq, but got {optimal_provider}")
            return False
            
    except Exception as e:
        print(f"❌ Model selection test failed: {e}")
        return False


async def main():
    """
    Main test function.
    """
    print("🚀 Starting Vision Integration Tests for Sentient-Core\n")
    
    # Run all tests
    test1_passed = await test_vision_with_message_model()
    test2_passed = await test_appstate_with_vision()
    test3_passed = await test_groq_model_selection()
    
    print("\n📋 Integration Test Summary:")
    print(f"   Message Model Vision: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   AppState Vision: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"   Groq Model Selection: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("\n🎉 All integration tests passed!")
        print("✅ Sentient-Core is ready for vision-enabled conversations with Groq prioritization!")
    else:
        print("\n⚠️  Some integration tests failed. Please check the configuration.")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)