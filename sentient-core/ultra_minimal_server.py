#!/usr/bin/env python3
"""
Ultra minimal FastAPI server - bypasses all service initialization
"""

import sys
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

print("=== Ultra Minimal Server Starting ===")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries

# Create the most basic FastAPI app possible
app = FastAPI(
    title="Ultra Minimal Test Server",
    description="Simplest possible FastAPI server for testing",
    version="1.0.0"
)

# Add basic CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "success",
        "message": "Ultra minimal server is running!",
        "server": "FastAPI",
        "port": 8004
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Server is operational",
        "checks": {
            "fastapi": "ok",
            "python": "ok",
            "server": "running"
        }
    }

@app.get("/api/chat/message/json")
async def chat_endpoint():
    """Simple chat endpoint for testing"""
    return {
        "status": "success",
        "message": "Chat endpoint is working",
        "response": "Hello from ultra minimal server!",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.get("/test")
async def test_endpoint():
    """Additional test endpoint"""
    return {
        "test": "passed",
        "server_type": "ultra_minimal",
        "endpoints": ["/", "/health", "/api/chat/message/json", "/test"]
    }

if __name__ == "__main__":
    print("Starting ultra minimal server on port 8004...")
    try:
        # Use the most basic uvicorn configuration
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8004,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"Server failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)