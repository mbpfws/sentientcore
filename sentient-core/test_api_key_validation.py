#!/usr/bin/env python3
"""
API Key Validation Test
Tests all configured API keys to identify which ones are invalid.
"""

import os
import asyncio
from groq import Groq
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_api():
    """Test Groq API key validity"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return False, "GROQ_API_KEY not found in environment"
        
        print(f"Testing Groq API key: {api_key[:10]}...{api_key[-4:]}")
        
        client = Groq(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": "Say 'API key is valid'"}
            ],
            max_tokens=10
        )
        
        return True, f"Success: {response.choices[0].message.content}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_openai_api():
    """Test OpenAI API key validity"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not found in environment"
        
        print(f"Testing OpenAI API key: {api_key[:10]}...{api_key[-4:]}")
        
        client = OpenAI(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API key is valid'"}
            ],
            max_tokens=10
        )
        
        return True, f"Success: {response.choices[0].message.content}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_google_api():
    """Test Google/Gemini API key validity"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return False, "GOOGLE_API_KEY/GEMINI_API_KEY not found in environment"
        
        print(f"Testing Google API key: {api_key[:10]}...{api_key[-4:]}")
        
        genai.configure(api_key=api_key)
        
        # Test with a simple generation
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'API key is valid'")
        
        return True, f"Success: {response.text}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Test all API keys"""
    print("=== API Key Validation Test ===")
    print()
    
    # Test Groq API
    print("🔍 Testing Groq API...")
    groq_valid, groq_msg = test_groq_api()
    print(f"{'✅' if groq_valid else '❌'} Groq API: {groq_msg}")
    print()
    
    # Test OpenAI API
    print("🔍 Testing OpenAI API...")
    openai_valid, openai_msg = test_openai_api()
    print(f"{'✅' if openai_valid else '❌'} OpenAI API: {openai_msg}")
    print()
    
    # Test Google API
    print("🔍 Testing Google/Gemini API...")
    google_valid, google_msg = test_google_api()
    print(f"{'✅' if google_valid else '❌'} Google API: {google_msg}")
    print()
    
    # Summary
    valid_apis = sum([groq_valid, openai_valid, google_valid])
    print(f"📊 Summary: {valid_apis}/3 API keys are valid")
    
    if valid_apis == 0:
        print("⚠️  No valid API keys found! Please check your .env file.")
    elif valid_apis < 3:
        print("⚠️  Some API keys are invalid. Please update them in your .env file.")
    else:
        print("✅ All API keys are valid!")

if __name__ == "__main__":
    main()