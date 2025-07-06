#!/usr/bin/env python3
"""
Minimal FastAPI server for testing
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Minimal Test Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Minimal server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is operational"}

@app.get("/api/test")
async def test_endpoint():
    return {"test": "success", "message": "API endpoint working"}

if __name__ == "__main__":
    print("Starting minimal server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)