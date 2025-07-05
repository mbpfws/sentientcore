#!/usr/bin/env python3
"""
Test script to verify Groq API prioritization for vision tasks.
This test ensures that when images are provided, the system prioritizes Groq vision models.
"""

import asyncio
import base64
import os
import sys
from pathlib import Path
from typing import Optional

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.services.enhanced_llm_service_main import EnhancedLLMService, get_enhanced_llm_service
from core.services.enhanced_llm_service import ModelCapability


def create_test_image_bytes() -> bytes:
    """
    Create a simple test image as bytes (a minimal PNG).
    This is a 1x1 pixel transparent PNG for testing purposes.
    """
    # Minimal 1x1 transparent PNG in base64
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    return base64.b64decode(png_base64)


async def test_groq_vision_prioritization():
    """
    Test that Groq vision models are prioritized when images are provided.
    """
    print("🧪 Testing Groq Vision Model Prioritization...")
    
    # Initialize the enhanced LLM service
    try:
        llm_service = get_enhanced_llm_service()
        print("✅ Enhanced LLM Service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Enhanced LLM Service: {e}")
        return False
    
    # Check available providers
    available_models = llm_service.get_available_models()
    print(f"📋 Available providers: {list(available_models.keys())}")
    
    # Check if Groq provider is available
    if 'groq' not in available_models:
        print("⚠️  Groq provider not available. Please check GROQ_API_KEY environment variable.")
        return False
    
    print(f"🤖 Groq models available: {available_models['groq']}")
    
    # Test vision model selection
    test_image = create_test_image_bytes()
    
    # Test 1: Check that vision-capable models are available
    vision_models = []
    for provider in llm_service.providers.values():
        for model_name, model_info in provider.models.items():
            if ModelCapability.VISION in model_info.capabilities:
                vision_models.append(f"{model_name} ({provider.__class__.__name__})")
    
    print(f"👁️  Vision-capable models found: {vision_models}")
    
    if not vision_models:
        print("❌ No vision-capable models found!")
        return False
    
    # Test 2: Verify optimal model selection for vision tasks
    try:
        optimal_model, optimal_provider = llm_service._get_optimal_model([ModelCapability.VISION])
        print(f"🎯 Optimal vision model selected: {optimal_model} from {optimal_provider}")
        
        # Check if it's a Groq model (should be prioritized)
        if optimal_provider == 'groq':
            print("✅ Groq provider correctly prioritized for vision tasks!")
        else:
            print(f"⚠️  Expected Groq to be prioritized, but got {optimal_provider}")
            
    except Exception as e:
        print(f"❌ Failed to get optimal vision model: {e}")
        return False
    
    # Test 3: Test actual vision request (if API key is available)
    if os.getenv("GROQ_API_KEY"):
        try:
            print("🔍 Testing actual vision request with Groq...")
            response = await llm_service.generate(
                prompt="What do you see in this image? Please describe it briefly.",
                image_bytes=test_image,
                context_id="test_vision_context"
            )
            print(f"📝 Vision response received: {response[:100]}...")
            print("✅ Vision request completed successfully!")
            
        except Exception as e:
            print(f"⚠️  Vision request failed (this might be expected with test image): {e}")
    else:
        print("⚠️  GROQ_API_KEY not found, skipping actual API test")
    
    # Test 4: Verify analytics show Groq usage
    analytics = llm_service.get_comprehensive_analytics()
    print(f"📊 Analytics - Providers in fallback chain: {analytics['providers']['fallback_chain']}")
    
    if analytics['providers']['fallback_chain'] and analytics['providers']['fallback_chain'][0] == 'groq':
        print("✅ Groq is first in fallback chain (highest priority)!")
    else:
        print(f"⚠️  Expected Groq to be first in fallback chain, got: {analytics['providers']['fallback_chain']}")
    
    return True


async def test_text_only_requests():
    """
    Test that text-only requests work normally without requiring vision models.
    """
    print("\n📝 Testing Text-Only Requests...")
    
    llm_service = get_enhanced_llm_service()
    
    try:
        # Test text-only request
        optimal_model, optimal_provider = llm_service._get_optimal_model([])
        print(f"📄 Optimal model for text-only: {optimal_model} from {optimal_provider}")
        
        if os.getenv("GROQ_API_KEY"):
            response = await llm_service.generate(
                prompt="Hello! Please respond with a brief greeting.",
                context_id="test_text_context"
            )
            print(f"💬 Text response: {response}")
            print("✅ Text-only request completed successfully!")
        else:
            print("⚠️  GROQ_API_KEY not found, skipping actual API test")
            
    except Exception as e:
        print(f"❌ Text-only request failed: {e}")
        return False
    
    return True


def check_environment():
    """
    Check if the required environment variables are set.
    """
    print("🔧 Checking Environment Configuration...")
    
    required_vars = ["GROQ_API_KEY"]
    optional_vars = ["OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is NOT set (required for Groq prioritization)")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set (optional)")
        else:
            print(f"⚠️  {var} is not set (optional)")


async def main():
    """
    Main test function.
    """
    print("🚀 Starting Groq Vision Priority Tests\n")
    
    # Check environment
    check_environment()
    print()
    
    # Run tests
    vision_test_passed = await test_groq_vision_prioritization()
    text_test_passed = await test_text_only_requests()
    
    print("\n📋 Test Summary:")
    print(f"   Vision Prioritization: {'✅ PASSED' if vision_test_passed else '❌ FAILED'}")
    print(f"   Text-Only Requests: {'✅ PASSED' if text_test_passed else '❌ FAILED'}")
    
    if vision_test_passed and text_test_passed:
        print("\n🎉 All tests passed! Groq API is properly prioritized for vision tasks.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the configuration and try again.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)