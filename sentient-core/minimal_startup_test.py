#!/usr/bin/env python3
"""
Minimal startup test to isolate server crash issues.
"""

import asyncio
import logging
import traceback
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def minimal_lifespan(app: FastAPI):
    """Minimal lifespan manager for testing"""
    logger.info("Starting minimal server...")
    try:
        # Minimal startup - just log
        logger.info("Minimal startup complete")
        yield
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        logger.info("Shutting down minimal server...")

# Create minimal FastAPI app
app = FastAPI(
    title="Minimal Test Server",
    lifespan=minimal_lifespan
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Minimal server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Minimal health check"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting minimal uvicorn server...")
    try:
        uvicorn.run(
            "minimal_startup_test:app",
            host="127.0.0.1",
            port=8002,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())