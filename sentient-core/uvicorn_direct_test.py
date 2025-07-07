#!/usr/bin/env python3
"""
Direct Uvicorn Test
Tests uvicorn.run() directly to isolate the issue
"""

import uvicorn
from fastapi import FastAPI
import asyncio
import sys
import threading
import time
import requests
from datetime import datetime

# Create a simple FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello from direct uvicorn test", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

def test_uvicorn_run_blocking():
    """Test uvicorn.run() in blocking mode"""
    print("\n=== Testing uvicorn.run() blocking mode ===")
    
    try:
        print("Starting uvicorn server (blocking)...")
        # This should block and run the server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8779,
            log_level="info",
            access_log=True
        )
        print("Server stopped")
    except Exception as e:
        print(f"✗ Uvicorn blocking test failed: {e}")
        import traceback
        traceback.print_exc()

def test_uvicorn_server_class():
    """Test uvicorn Server class directly"""
    print("\n=== Testing uvicorn Server class ===")
    
    try:
        # Set ProactorEventLoop for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Create config
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8780,
            log_level="info",
            access_log=True
        )
        
        # Create server
        server = uvicorn.Server(config)
        
        print("Starting uvicorn server (Server class)...")
        
        # Run server
        asyncio.run(server.serve())
        
        print("Server stopped")
        
    except Exception as e:
        print(f"✗ Uvicorn Server class test failed: {e}")
        import traceback
        traceback.print_exc()

def test_uvicorn_with_different_configs():
    """Test uvicorn with different configurations"""
    print("\n=== Testing uvicorn with different configurations ===")
    
    configs = [
        {
            "name": "Default config",
            "kwargs": {
                "host": "127.0.0.1",
                "port": 8781,
                "log_level": "info"
            }
        },
        {
            "name": "No access log",
            "kwargs": {
                "host": "127.0.0.1",
                "port": 8782,
                "log_level": "error",
                "access_log": False
            }
        },
        {
            "name": "Different loop",
            "kwargs": {
                "host": "127.0.0.1",
                "port": 8783,
                "log_level": "error",
                "access_log": False,
                "loop": "asyncio"
            }
        }
    ]
    
    for config in configs:
        try:
            print(f"\nTesting: {config['name']}")
            
            # Create a thread to run the server
            def run_server():
                try:
                    uvicorn.run(app, **config['kwargs'])
                except Exception as e:
                    print(f"Server thread error: {e}")
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Try to connect
            try:
                response = requests.get(f"http://127.0.0.1:{config['kwargs']['port']}/health", timeout=5)
                print(f"✓ {config['name']}: Server responded with {response.status_code}")
                print(f"  Response: {response.json()}")
            except Exception as e:
                print(f"✗ {config['name']}: Failed to connect - {e}")
            
            # Give it a moment then continue
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ {config['name']}: Configuration test failed - {e}")

def test_minimal_uvicorn():
    """Test the most minimal uvicorn setup possible"""
    print("\n=== Testing minimal uvicorn setup ===")
    
    # Create the simplest possible FastAPI app
    minimal_app = FastAPI()
    
    @minimal_app.get("/")
    def simple_root():
        return "OK"
    
    try:
        print("Starting minimal uvicorn server...")
        
        # Use a thread to avoid blocking
        def run_minimal():
            try:
                uvicorn.run(
                    minimal_app,
                    host="127.0.0.1",
                    port=8784,
                    log_level="critical",  # Minimal logging
                    access_log=False
                )
            except Exception as e:
                print(f"Minimal server error: {e}")
                import traceback
                traceback.print_exc()
        
        server_thread = threading.Thread(target=run_minimal, daemon=True)
        server_thread.start()
        
        # Wait for startup
        time.sleep(3)
        
        # Test connection
        try:
            response = requests.get("http://127.0.0.1:8784/", timeout=5)
            print(f"✓ Minimal server: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"✗ Minimal server connection failed: {e}")
        
        time.sleep(1)
        
    except Exception as e:
        print(f"✗ Minimal uvicorn test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("Direct Uvicorn Test")
    print("=" * 50)
    
    # Test different approaches
    test_minimal_uvicorn()
    test_uvicorn_with_different_configs()
    
    print("\n=== Non-blocking tests complete ===")
    print("Now testing blocking mode (this will run indefinitely until interrupted)...")
    
    # Test blocking mode last (since it will block)
    try:
        test_uvicorn_run_blocking()
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
    except Exception as e:
        print(f"Blocking test failed: {e}")

if __name__ == "__main__":
    main()