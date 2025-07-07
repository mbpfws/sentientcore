#!/usr/bin/env python3
"""Robust test server with comprehensive logging and error handling."""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('robust_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Robust Test Server",
    description="A stable backend server for testing frontend-backend connectivity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.now()
    logger.info(f"Incoming request: {request.method} {request.url}")
    
    try:
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Request completed: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.method} {request.url} - Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(e)}
        )

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    logger.info("Root endpoint accessed")
    return {
        "message": "Robust Test Server is running!",
        "timestamp": datetime.now().isoformat(),
        "server": "robust_test_server",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "server": "robust_test_server",
        "uptime": "running",
        "port": 8008
    }

@app.get("/api/chat")
async def get_chat() -> Dict[str, Any]:
    """Get chat endpoint."""
    logger.info("GET chat endpoint accessed")
    return {
        "message": "Chat endpoint is working",
        "method": "GET",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def post_chat(data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Post chat endpoint."""
    logger.info(f"POST chat endpoint accessed with data: {data}")
    return {
        "message": "Chat message received",
        "method": "POST",
        "received_data": data,
        "timestamp": datetime.now().isoformat(),
        "response": "This is a test response from the robust server"
    }

@app.get("/api/status")
async def status() -> Dict[str, Any]:
    """Status endpoint."""
    logger.info("Status endpoint accessed")
    return {
        "status": "operational",
        "server": "robust_test_server",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/",
            "/health",
            "/api/chat",
            "/api/status",
            "/api/test"
        ]
    }

@app.get("/api/test")
async def test() -> Dict[str, Any]:
    """Test endpoint."""
    logger.info("Test endpoint accessed")
    return {
        "test": "success",
        "message": "Test endpoint is working correctly",
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

def main():
    """Main function to run the server."""
    try:
        logger.info("Starting Robust Test Server...")
        logger.info("Server will be available at: http://127.0.0.1:8008")
        logger.info("Available endpoints:")
        logger.info("  - GET  /")
        logger.info("  - GET  /health")
        logger.info("  - GET  /api/chat")
        logger.info("  - POST /api/chat")
        logger.info("  - GET  /api/status")
        logger.info("  - GET  /api/test")
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8008,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Robust Test Server shutdown complete")

if __name__ == "__main__":
    main()