from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

class NormalizeSlashMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.scope['path'] != '/' and request.scope['path'].endswith('/'):
            request.scope['path'] = request.scope['path'][:-1]
        response = await call_next(request)
        return response

from fastapi.responses import JSONResponse
import os
import sys
import logging
import traceback
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

# Import enhanced core systems
from app.core.error_handling import error_handler, SentientCoreError, ErrorContext
from app.core.health_monitor import health_monitor, ComponentType, HealthStatus
from app.core.testing_framework import TestRunner, create_health_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with enhanced monitoring"""
    # Startup
    logger.info("Starting Sentient Core API...")
    
    try:
        # Start health monitoring
        await health_monitor.start_monitoring()
        logger.info("Health monitoring started")
        
        # Initialize all services
        success = await initialize_services()
        if not success:
            error_context = ErrorContext(
                operation="service_initialization",
                component="startup",
                details={"phase": "initialization"}
            )
            await error_handler.handle_error(
                RuntimeError("Service initialization failed"),
                error_context
            )
            raise RuntimeError("Service initialization failed")
        
        logger.info("All services initialized successfully")
        
        # Store service factory and enhanced systems in app state
        app.state.service_factory = get_service_factory()
        app.state.error_handler = error_handler
        app.state.health_monitor = health_monitor
        
        # Register services with health monitor
        service_factory = get_service_factory()
        if hasattr(service_factory, 'memory_service') and service_factory.memory_service:
            health_monitor.register_component("memory_service", ComponentType.SERVICE)
        if hasattr(service_factory, 'llm_service') and service_factory.llm_service:
            health_monitor.register_component("llm_service", ComponentType.SERVICE)
        if hasattr(service_factory, 'workflow_orchestrator') and service_factory.workflow_orchestrator:
            health_monitor.register_component("workflow_orchestrator", ComponentType.SERVICE)
        if hasattr(service_factory, 'research_service') and service_factory.research_service:
            health_monitor.register_component("research_service", ComponentType.SERVICE)
        if hasattr(service_factory, 'agent_service') and service_factory.agent_service:
            health_monitor.register_component("agent_service", ComponentType.SERVICE)
        
        logger.info("Enhanced monitoring systems initialized")
        
        yield
        
    except Exception as e:
        error_context = ErrorContext(
            operation="application_startup",
            component="lifespan",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )
        await error_handler.handle_error(e, error_context)
        logger.error(f"Startup error: {e}")
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down Sentient Core API...")
        try:
            # Stop health monitoring
            await health_monitor.stop_monitoring()
            logger.info("Health monitoring stopped")
            
            # Cleanup services
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
app.add_middleware(NormalizeSlashMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Global exception handlers
@app.exception_handler(SentientCoreError)
async def sentient_core_exception_handler(request: Request, exc: SentientCoreError):
    """Handle custom Sentient Core exceptions"""
    error_context = ErrorContext(
        operation="api_request",
        component="exception_handler",
        details={
            "endpoint": str(request.url),
            "method": request.method,
            "error_type": exc.__class__.__name__
        }
    )
    
    if hasattr(request.app.state, 'error_handler'):
        await request.app.state.error_handler.handle_error(exc, error_context)
    
    return JSONResponse(
        status_code=exc.status_code if hasattr(exc, 'status_code') else 500,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc),
            "details": exc.details if hasattr(exc, 'details') else None
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    error_context = ErrorContext(
        operation="api_request",
        component="global_exception_handler",
        details={
            "endpoint": str(request.url),
            "method": request.method,
            "error_type": exc.__class__.__name__,
            "traceback": traceback.format_exc()
        }
    )
    
    if hasattr(request.app.state, 'error_handler'):
        await request.app.state.error_handler.handle_error(exc, error_context)
    
    logger.error(f"Unhandled exception in {request.method} {request.url}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": str(exc) if app.debug else None
        }
    )

# Import routers
from .routers import agents, workflows, chat, unified_router, interactive_workflows, research, monitoring, implementation, sse_events, workflow_websockets

# Register routers with /api prefix
app.include_router(agents.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(unified_router.router, prefix="/api")
app.include_router(interactive_workflows.router, prefix="/api")
app.include_router(research.router, prefix="/api")
app.include_router(monitoring.router, prefix="/api")
app.include_router(implementation.router, prefix="/api")
app.include_router(sse_events.router, prefix="/api")
app.include_router(workflow_websockets.router, prefix="/api")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "ok", "message": "Sentient Core API is running"}

@app.get("/health")
async def health_check(request: Request):
    """Enhanced health check endpoint with comprehensive monitoring"""
    try:
        # Get health monitor if available
        if hasattr(request.app.state, 'health_monitor'):
            health_monitor_instance = request.app.state.health_monitor
            system_health = health_monitor_instance.get_current_health()
            
            return {
                "status": system_health.status.value,
                "message": "Sentient Core API health check",
                "timestamp": system_health.timestamp.isoformat(),
                "system_metrics": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "status": metric.status.value
                    }
                    for metric in system_health.system_metrics
                ],
                "components": {
                    name: {
                        "status": comp.status.value,
                        "last_check": comp.last_check.isoformat() if comp.last_check else None,
                        "error_message": comp.error_message
                    }
                    for name, comp in system_health.components.items()
                }
            }
        else:
            # Fallback to service factory health check
            service_factory = request.app.state.service_factory
            health_status = await service_factory.health_check()
            
            return {
                "status": "healthy" if health_status["overall"] == "healthy" else "degraded",
                "message": "Sentient Core API is operational",
                "services": health_status
            }
    except Exception as e:
        error_context = ErrorContext(
            operation="health_check",
            component="health_endpoint",
            details={"error": str(e)}
        )
        
        if hasattr(request.app.state, 'error_handler'):
            await request.app.state.error_handler.handle_error(e, error_context)
        
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}",
            "timestamp": None
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

@app.get("/api/health/detailed")
async def get_detailed_health(request: Request):
    """Get detailed health information including history"""
    try:
        if hasattr(request.app.state, 'health_monitor'):
            health_monitor_instance = request.app.state.health_monitor
            current_health = health_monitor_instance.get_current_health()
            health_history = health_monitor_instance.get_health_history(hours=1)
            
            return {
                "current": {
                    "status": current_health.status.value,
                    "timestamp": current_health.timestamp.isoformat(),
                    "system_metrics": current_health.system_metrics,
                    "components": {
                        name: {
                            "status": comp.status.value,
                            "last_check": comp.last_check.isoformat() if comp.last_check else None,
                            "error_message": comp.error_message,
                            "metrics": comp.metrics
                        }
                        for name, comp in current_health.components.items()
                    }
                },
                "history": [
                    {
                        "status": h.status.value,
                        "timestamp": h.timestamp.isoformat(),
                        "system_metrics": h.system_metrics
                    }
                    for h in health_history
                ]
            }
        else:
            raise HTTPException(status_code=503, detail="Health monitoring not available")
    except Exception as e:
        logger.error(f"Detailed health error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors/summary")
async def get_error_summary(request: Request):
    """Get error statistics and recent errors"""
    try:
        if hasattr(request.app.state, 'error_handler'):
            error_handler_instance = request.app.state.error_handler
            stats = error_handler_instance.get_error_stats()
            recent_errors = error_handler_instance.get_recent_errors(limit=10)
            
            return {
                "statistics": stats,
                "recent_errors": [
                    {
                        "timestamp": error.timestamp.isoformat(),
                        "severity": error.severity.value,
                        "category": error.category.value,
                        "operation": error.context.operation,
                        "component": error.context.component,
                        "message": str(error.exception)
                    }
                    for error in recent_errors
                ]
            }
        else:
            raise HTTPException(status_code=503, detail="Error handling not available")
    except Exception as e:
        logger.error(f"Error summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/test")
async def run_system_tests(request: Request):
    """Run comprehensive system tests"""
    try:
        service_factory = request.app.state.service_factory
        test_runner = TestRunner()
        
        # Create health tests for all services
        if hasattr(service_factory, 'memory_service') and service_factory.memory_service:
            test_runner.add_test(create_service_health_test(
                "memory_service", 
                service_factory.memory_service
            ))
        
        if hasattr(service_factory, 'llm_service') and service_factory.llm_service:
            test_runner.add_test(create_service_health_test(
                "llm_service", 
                service_factory.llm_service
            ))
        
        if hasattr(service_factory, 'workflow_orchestrator') and service_factory.workflow_orchestrator:
            test_runner.add_test(create_service_health_test(
                "workflow_orchestrator", 
                service_factory.workflow_orchestrator
            ))
        
        # Run tests
        results = await test_runner.run_all_tests()
        
        return {
            "test_results": {
                "total_tests": len(results),
                "passed": len([r for r in results if r.status.value == "passed"]),
                "failed": len([r for r in results if r.status.value == "failed"]),
                "results": [
                    {
                        "test_name": r.test_name,
                        "status": r.status.value,
                        "duration": r.duration,
                        "message": r.message,
                        "error": r.error
                    }
                    for r in results
                ]
            }
        }
    except Exception as e:
        logger.error(f"System test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dependency to get service factory
def get_services(request: Request):
    """Dependency to get service factory"""
    if not hasattr(request.app.state, 'service_factory'):
        raise HTTPException(status_code=503, detail="Services not initialized")
    return request.app.state.service_factory
