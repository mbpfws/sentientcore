#!/usr/bin/env python3
"""
Simple Uvicorn Test - Test if uvicorn can start a basic server
"""

import uvicorn
from fastapi import FastAPI
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("Creating FastAPI app...")
app = FastAPI(title="Simple Test")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting uvicorn server...")
    print(f"Python version: {sys.version}")
    
    try:
        # Use the most basic uvicorn configuration
        uvicorn.run(
            app,  # Pass the app object directly
            host="127.0.0.1",
            port=8001,  # Use a different port
            log_level="debug"
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)