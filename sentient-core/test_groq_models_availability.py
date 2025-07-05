#!/usr/bin/env python3
"""
Test script to verify which Groq models are actually available and working.
"""

import asyncio
import base64
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from groq import Groq


def create_sample_image() -> bytes:
    """
    Create a simple test image for vision testing.
    """
    # A simple 8x8 red square PNG in base64
    red_square_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz"
        "AAAB2AAAAdgB+lymcgAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAAdSURB"
        "VBiVY/z//z8DAwMDEwMDw38GBgYGJgYGBgYAFgAH/A8ECQAAAABJRU5ErkJggg=="
    )
    return base64.b64decode(red_square_base64)


async def test_groq_model_availability():
    """
    Test which Groq models are actually available.
    """
    print("🔍 Testing Groq Model Availability...")
    
    # Check if API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found")
        return False
    
    client = Groq(api_key=api_key)
    
    # Test models to check
    test_models = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    print("\n📋 Testing Text-Only Requests:")
    working_text_models = []
    
    for model in test_models:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": "Hello! Just say 'Hi' back."}],
                max_tokens=10
            )
            print(f"✅ {model}: {response.choices[0].message.content.strip()}")
            working_text_models.append(model)
        except Exception as e:
            print(f"❌ {model}: {str(e)}")
    
    print("\n🖼️  Testing Vision Requests:")
    working_vision_models = []
    
    # Create test image
    test_image = create_sample_image()
    base64_image = base64.b64encode(test_image).decode("utf-8")
    
    # Test vision with working text models
    for model in working_text_models:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }],
                max_tokens=50
            )
            print(f"✅ {model}: Vision supported - {response.choices[0].message.content.strip()[:50]}...")
            working_vision_models.append(model)
        except Exception as e:
            print(f"❌ {model}: Vision failed - {str(e)}")
    
    print("\n📊 Summary:")
    print(f"   Working text models: {len(working_text_models)}")
    print(f"   Working vision models: {len(working_vision_models)}")
    
    if working_vision_models:
        print(f"\n🎉 Vision-capable Groq models found: {working_vision_models}")
        return working_vision_models
    else:
        print("\n⚠️  No vision-capable Groq models found")
        return []


async def test_groq_models_list():
    """
    Try to get the list of available models from Groq API.
    """
    print("\n🔍 Fetching Available Models from Groq API...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found")
        return
    
    client = Groq(api_key=api_key)
    
    try:
        models = await asyncio.to_thread(client.models.list)
        print(f"\n📋 Available Groq Models ({len(models.data)} total):")
        for model in models.data:
            print(f"   - {model.id}")
    except Exception as e:
        print(f"❌ Failed to fetch models: {e}")


async def main():
    """
    Main test function.
    """
    print("🚀 Starting Groq Model Availability Tests\n")
    
    # Test model availability
    working_vision_models = await test_groq_model_availability()
    
    # Try to get official model list
    await test_groq_models_list()
    
    if working_vision_models:
        print(f"\n✅ Recommendation: Use these models for vision: {working_vision_models}")
        return True
    else:
        print("\n⚠️  Recommendation: Use Gemini or OpenAI for vision tasks until Groq vision models are available")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)