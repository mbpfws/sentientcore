#!/usr/bin/env python3
"""
Simplest possible FastAPI server test without lifespan.
"""

import logging
from fastapi import FastAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the simplest possible FastAPI app
app = FastAPI(title="Simple Test Server")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Simple server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Simple health check"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting simple uvicorn server...")
    uvicorn.run(
        app,  # Pass the app object directly
        host="127.0.0.1",
        port=8003,
        log_level="info"
    )