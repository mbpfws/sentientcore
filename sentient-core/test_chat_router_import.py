#!/usr/bin/env python3
"""
Test script to test importing the actual chat router module.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Starting chat router module import test...")

try:
    print("1. Loading environment variables...")
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment variables loaded")
    
    print("2. Testing FastAPI import...")
    from fastapi import FastAPI
    print("✓ FastAPI imported")
    
    print("3. Creating FastAPI app...")
    app = FastAPI()
    print("✓ FastAPI app created")
    
    print("4. Testing chat router import...")
    from app.api.routers import chat
    print("✓ Chat router imported successfully")
    
    print("5. Testing router inclusion...")
    app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
    print("✓ Chat router included successfully")
    
    print("\n🎉 Chat router import and inclusion completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during chat router import: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    sys.exit(1)