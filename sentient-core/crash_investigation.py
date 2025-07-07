#!/usr/bin/env python3
"""
Investigate what's causing uvicorn servers to crash when receiving HTTP requests.
"""

import asyncio
import logging
import threading
import time
import requests
import signal
import sys
from fastapi import FastAPI
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag to track server status
server_running = False
server_thread = None

app = FastAPI(title="Crash Investigation Server")

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"status": "ok", "message": "Server is responding"}

@app.get("/health")
async def health():
    logger.info("Health endpoint called")
    return {"status": "healthy", "message": "Health check successful"}

def signal_handler(signum, frame):
    """Handle signals gracefully"""
    logger.info(f"Received signal {signum}")
    global server_running
    server_running = False
    sys.exit(0)

def run_server():
    """Run the uvicorn server"""
    global server_running
    logger.info("Starting uvicorn server in thread...")
    
    try:
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8006,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        server_running = True
        logger.info("Server thread: About to start uvicorn")
        server.run()
        logger.info("Server thread: Uvicorn has stopped")
    except Exception as e:
        logger.error(f"Server thread error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        server_running = False
        logger.info("Server thread: Exiting")

def test_requests():
    """Test making requests to the server"""
    global server_running
    
    # Wait for server to start
    logger.info("Waiting for server to start...")
    for i in range(10):
        if server_running:
            break
        time.sleep(1)
        logger.info(f"Waiting... {i+1}/10")
    
    if not server_running:
        logger.error("Server failed to start within timeout")
        return
    
    # Give server a bit more time to be ready
    time.sleep(2)
    logger.info("Server should be ready, testing requests...")
    
    try:
        # Test 1: Simple GET request
        logger.info("Test 1: Making GET request to /")
        response = requests.get('http://127.0.0.1:8006/', timeout=5)
        logger.info(f"Test 1 SUCCESS: Status {response.status_code}, Response: {response.text}")
        
        # Wait a bit
        time.sleep(1)
        
        # Test 2: Health check
        logger.info("Test 2: Making GET request to /health")
        response = requests.get('http://127.0.0.1:8006/health', timeout=5)
        logger.info(f"Test 2 SUCCESS: Status {response.status_code}, Response: {response.text}")
        
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        logger.error("Server appears to have crashed or become unreachable")
    except Exception as e:
        logger.error(f"Request error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Check server status
    logger.info(f"After requests, server_running = {server_running}")
    if server_thread and server_thread.is_alive():
        logger.info("Server thread is still alive")
    else:
        logger.info("Server thread has died")

def main():
    global server_thread
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting crash investigation...")
    
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=False)
    server_thread.start()
    
    # Test requests
    test_requests()
    
    # Keep monitoring
    logger.info("Monitoring server for 10 seconds...")
    for i in range(10):
        time.sleep(1)
        if server_thread.is_alive():
            logger.info(f"Second {i+1}: Server thread still alive")
        else:
            logger.info(f"Second {i+1}: Server thread has died")
            break
    
    logger.info("Investigation complete")

if __name__ == "__main__":
    main()