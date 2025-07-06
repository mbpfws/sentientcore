#!/usr/bin/env python3
"""
Test script to isolate which LLM provider is causing the hang during initialization.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting provider initialization test...")

try:
    print("1. Testing basic imports...")
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Basic imports successful")
    
    print("2. Testing Groq import...")
    from groq import Groq
    print("✓ Groq import successful")
    
    print("3. Testing OpenAI import...")
    from openai import OpenAI
    print("✓ OpenAI import successful")
    
    print("4. Testing EnhancedGroqProvider import...")
    from core.services.enhanced_llm_service import EnhancedGroqProvider
    print("✓ EnhancedGroqProvider import successful")
    
    print("5. Testing EnhancedGroqProvider initialization...")
    if os.getenv("GROQ_API_KEY"):
        groq_provider = EnhancedGroqProvider()
        print("✓ EnhancedGroqProvider initialized successfully")
    else:
        print("⚠ GROQ_API_KEY not set, skipping Groq provider initialization")
    
    print("6. Testing OpenAIProvider import...")
    from core.services.enhanced_llm_service import OpenAIProvider
    print("✓ OpenAIProvider import successful")
    
    print("7. Testing OpenAIProvider initialization...")
    if os.getenv("OPENAI_API_KEY"):
        openai_provider = OpenAIProvider()
        print("✓ OpenAIProvider initialized successfully")
    else:
        print("⚠ OPENAI_API_KEY not set, skipping OpenAI provider initialization")
    
    print("8. Testing EnhancedLLMService import...")
    from core.services.enhanced_llm_service import EnhancedLLMService
    print("✓ EnhancedLLMService import successful")
    
    print("9. Testing EnhancedLLMService initialization...")
    enhanced_service = EnhancedLLMService()
    print("✓ EnhancedLLMService initialized successfully")
    
    print("\n🎉 All provider tests completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during provider initialization: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    sys.exit(1)