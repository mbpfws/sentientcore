#!/usr/bin/env python3
"""
Test script to check EnhancedLLMService initialization
"""

from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, '.')

print("Environment variables after load_dotenv():")
print(f"GROQ_API_KEY: {'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET'}")
print(f"OPENAI_API_KEY: {'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET'}")
print()

print("Testing EnhancedLLMService initialization...")
try:
    from core.services.enhanced_llm_service import EnhancedLLMService
    service = EnhancedLLMService()
    print("✓ EnhancedLLMService initialized successfully")
    print(f"Available providers: {list(service.providers.keys()) if hasattr(service, 'providers') else 'Unknown'}")
except Exception as e:
    print(f"✗ Failed to initialize EnhancedLLMService: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")