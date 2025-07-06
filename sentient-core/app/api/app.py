from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import services
from app.services.service_factory import get_service_factory, initialize_services, cleanup_services, ServiceConfig
from core.models import AppState, Message, EnhancedTask, TaskStatus, LogEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Sentient Core API...")
    
    try:
        # Initialize all services
        success = await initialize_services()
        if not success:
            logger.error("Failed to initialize services")
            raise RuntimeError("Service initialization failed")
        
        logger.info("All services initialized successfully")
        
        # Store service factory in app state for access in routes
        app.state.service_factory = get_service_factory()
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Sentient Core API...")
        try:
            await cleanup_services()
            logger.info("Services cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


app = FastAPI(
    title="Sentient Core API",
    description="API for the Sentient Core Multi-Agent RAG System",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from .routers import agents, workflows, chat, core_services, api_endpoints, interactive_workflows, research, monitoring, implementation

# Register routers with /api prefix
app.include_router(agents.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(core_services.router, prefix="/api")
app.include_router(api_endpoints.router, prefix="/api")
app.include_router(interactive_workflows.router, prefix="/api")
app.include_router(research.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")
app.include_router(implementation.router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "ok", "message": "Sentient Core API is running"}

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint with service status"""
    try:
        service_factory = request.app.state.service_factory
        health_status = await service_factory.health_check()
        
        return {
            "status": "healthy" if health_status["overall"] == "healthy" else "degraded",
            "message": "Sentient Core API is operational",
            "services": health_status
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }

@app.get("/api/services/status")
async def get_services_status(request: Request):
    """Get detailed service status"""
    try:
        service_factory = request.app.state.service_factory
        return service_factory.get_service_status()
    except Exception as e:
        logger.error(f"Service status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/services/stats")
async def get_services_stats(request: Request):
    """Get service statistics"""
    try:
        service_factory = request.app.state.service_factory
        return service_factory.get_service_stats()
    except Exception as e:
        logger.error(f"Service stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dependency to get service factory
def get_services(request: Request):
    """Dependency to get service factory"""
    if not hasattr(request.app.state, 'service_factory'):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return request.app.state.service_factory
