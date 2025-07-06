#!/usr/bin/env python3
"""
Test script to test FastAPI app startup without uvicorn
"""

import os
import sys
from dotenv import load_dotenv

print("Starting FastAPI app startup test...")

try:
    print("1. Loading environment variables...")
    load_dotenv()
    print("✓ Environment variables loaded")
    
    print("2. Setting up Python path...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print("✓ Python path configured")
    
    print("3. Importing the FastAPI app...")
    from app.api.app import app
    print("✓ FastAPI app imported successfully")
    
    print("4. Testing app configuration...")
    print(f"✓ App title: {app.title}")
    print(f"✓ App version: {app.version}")
    print(f"✓ Number of routes: {len(app.routes)}")
    
    print("5. Testing route enumeration...")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            print(f"  - {route.path} [{', '.join(route.methods)}]")
    
    print("\n🎉 FastAPI app startup test completed successfully!")
    print("The hang is likely occurring during uvicorn startup or reload functionality.")
    
except Exception as e:
    print(f"❌ Error during FastAPI app startup: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    sys.exit(1)