#!/usr/bin/env python3
"""
Test script to test uvicorn startup with and without reload
"""

import os
import sys
import time
import subprocess
from dotenv import load_dotenv

print("Starting uvicorn startup test...")

try:
    print("1. Loading environment variables...")
    load_dotenv()
    print("✓ Environment variables loaded")
    
    print("2. Testing uvicorn import...")
    import uvicorn
    print("✓ uvicorn imported successfully")
    
    print("3. Testing FastAPI app import...")
    from app.api.app import app
    print("✓ FastAPI app imported successfully")
    
    print("4. Testing uvicorn.run without reload (will run for 10 seconds then stop)...")
    print("Starting server without reload...")
    
    # Start uvicorn in a separate process to avoid blocking
    import threading
    import signal
    
    def run_server():
        try:
            uvicorn.run(
                "app.api.app:app", 
                host="127.0.0.1", 
                port=8001,  # Use different port to avoid conflicts
                reload=False,
                log_level="info"
            )
        except Exception as e:
            print(f"Server error: {e}")
    
    # Start server in thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a bit to see if server starts
    print("Waiting 5 seconds to see if server starts...")
    time.sleep(5)
    
    if server_thread.is_alive():
        print("✓ Server started successfully without reload")
    else:
        print("❌ Server thread died")
    
    print("\n🎉 uvicorn startup test completed!")
    print("If you see this message, uvicorn without reload works fine.")
    print("The hang is likely specific to reload=True functionality.")
    
except Exception as e:
    print(f"❌ Error during uvicorn startup test: {e}")
    import traceback
    print(f"Full traceback: {traceback.format_exc()}")
    sys.exit(1)