#!/usr/bin/env python3
"""
Test server on port 8001
"""

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server running on port 8001"}

@app.get("/health")
async def health():
    return {"status": "ok", "port": 8001}

if __name__ == "__main__":
    print("Starting server on port 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)