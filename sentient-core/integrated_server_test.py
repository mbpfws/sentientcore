#!/usr/bin/env python3
"""
Integrated Server Test
Starts a server and tests it in the same process
"""

import asyncio
import threading
import time
import requests
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager

# Global server reference
server = None
server_thread = None

def create_app():
    """Create FastAPI application"""
    app = FastAPI(title="Integrated Test Server")
    
    @app.get("/")
    def read_root():
        return {"message": "Server is running!", "status": "ok"}
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "timestamp": time.time()}
    
    @app.get("/test")
    def test_endpoint():
        return {"test": "success", "data": [1, 2, 3, 4, 5]}
    
    return app

def run_server_in_thread(port=8005):
    """Run server in a separate thread"""
    global server
    
    app = create_app()
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    
    # Run server
    print(f"Starting server on http://127.0.0.1:{port}")
    try:
        asyncio.run(server.serve())
    except Exception as e:
        print(f"Server error: {e}")
        import traceback
        traceback.print_exc()

def test_endpoints(port=8005, max_retries=10):
    """Test server endpoints"""
    base_url = f"http://127.0.0.1:{port}"
    endpoints = [
        ("/", "root"),
        ("/health", "health"),
        ("/test", "test")
    ]
    
    print(f"\nTesting endpoints on {base_url}")
    
    # Wait for server to start
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"Server ready after {attempt + 1} attempts")
                break
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1}: Server not ready, waiting...")
                time.sleep(1)
            else:
                print("Server failed to start after maximum retries")
                return False
    
    # Test all endpoints
    success_count = 0
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✓ {name.upper()}: {response.json()}")
                success_count += 1
            else:
                print(f"✗ {name.upper()}: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"✗ {name.upper()}: {e}")
    
    print(f"\nResults: {success_count}/{len(endpoints)} endpoints successful")
    return success_count == len(endpoints)

def main():
    """Main test function"""
    global server_thread
    
    print("=" * 60)
    print("INTEGRATED SERVER TEST")
    print("=" * 60)
    
    port = 8005
    
    # Start server in background thread
    server_thread = threading.Thread(
        target=run_server_in_thread,
        args=(port,),
        daemon=True
    )
    
    print("Starting server thread...")
    server_thread.start()
    
    # Give server time to start
    time.sleep(3)
    
    # Test endpoints
    success = test_endpoints(port)
    
    if success:
        print("\n🎉 All tests passed! Server is working correctly.")
    else:
        print("\n❌ Some tests failed. Server has issues.")
    
    # Keep server running for manual testing
    print(f"\nServer running at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if server:
            server.should_exit = True

if __name__ == "__main__":
    main()