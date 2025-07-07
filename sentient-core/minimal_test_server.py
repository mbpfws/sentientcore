#!/usr/bin/env python3
"""Minimal test server for debugging connection issues."""

import sys
import time
from datetime import datetime

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    print(f"[{datetime.now()}] FastAPI and uvicorn imported successfully")
except ImportError as e:
    print(f"[{datetime.now()}] Import error: {e}")
    sys.exit(1)

print(f"[{datetime.now()}] Creating FastAPI app...")
app = FastAPI(title="Minimal Test Server")

print(f"[{datetime.now()}] Adding CORS middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    print(f"[{datetime.now()}] Root endpoint accessed")
    return {"message": "Minimal server running", "timestamp": datetime.now().isoformat()}

@app.get("/health")
def health():
    print(f"[{datetime.now()}] Health endpoint accessed")
    return {"status": "healthy", "port": 8009, "timestamp": datetime.now().isoformat()}

@app.get("/api/chat")
def get_chat():
    print(f"[{datetime.now()}] GET chat endpoint accessed")
    return {"message": "Chat GET working", "timestamp": datetime.now().isoformat()}

@app.post("/api/chat")
def post_chat(data: dict = None):
    print(f"[{datetime.now()}] POST chat endpoint accessed")
    return {"message": "Chat POST working", "data": data, "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
def status():
    print(f"[{datetime.now()}] Status endpoint accessed")
    return {"status": "operational", "timestamp": datetime.now().isoformat()}

@app.get("/api/test")
def test():
    print(f"[{datetime.now()}] Test endpoint accessed")
    return {"test": "success", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    try:
        print(f"[{datetime.now()}] Starting minimal server on port 8009...")
        print(f"[{datetime.now()}] Python version: {sys.version}")
        print(f"[{datetime.now()}] Platform: {sys.platform}")
        
        # Try to bind to the port first
        import socket
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            test_socket.bind(('127.0.0.1', 8009))
            print(f"[{datetime.now()}] Port 8009 is available")
            test_socket.close()
        except Exception as e:
            print(f"[{datetime.now()}] Port 8009 binding failed: {e}")
            test_socket.close()
            sys.exit(1)
        
        print(f"[{datetime.now()}] Starting uvicorn server...")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8009,
            log_level="debug"
        )
    except KeyboardInterrupt:
        print(f"[{datetime.now()}] Server interrupted by user")
    except Exception as e:
        print(f"[{datetime.now()}] Server error: {e}")
        import traceback
        print(f"[{datetime.now()}] Traceback: {traceback.format_exc()}")
    finally:
        print(f"[{datetime.now()}] Server shutdown complete")