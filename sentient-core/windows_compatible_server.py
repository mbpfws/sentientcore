#!/usr/bin/env python3
"""
Windows Compatible FastAPI Server
Addresses Windows-specific networking and process isolation issues
"""

import os
import sys
import signal
import socket
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager

# Global server state
server_running = True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global server_running
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    server_running = False
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

def check_port_binding(host, port):
    """Check if we can bind to the specified host and port"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            print(f"✓ Successfully bound to {host}:{port}")
            return True
    except OSError as e:
        print(f"✗ Failed to bind to {host}:{port} - {e}")
        return False

def create_app():
    """Create FastAPI application with Windows-specific configurations"""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        print("🚀 Application starting up...")
        yield
        # Shutdown
        print("🛑 Application shutting down...")
    
    app = FastAPI(
        title="Windows Compatible Server",
        description="FastAPI server optimized for Windows networking",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware with permissive settings for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for local testing
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def read_root():
        """Root endpoint"""
        return {
            "message": "Windows Compatible Server is running!",
            "status": "ok",
            "timestamp": time.time(),
            "platform": sys.platform,
            "python_version": sys.version.split()[0]
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time(),  # Simplified uptime
            "platform": sys.platform
        }
    
    @app.get("/api/chat/message/json")
    async def chat_endpoint():
        """Chat endpoint for compatibility"""
        return {
            "response": "Hello from Windows Compatible Server!",
            "status": "success",
            "timestamp": time.time(),
            "model": "test-model"
        }
    
    @app.post("/api/chat/message/json")
    async def chat_post_endpoint(request: dict = None):
        """POST chat endpoint"""
        return {
            "response": f"Received: {request}",
            "status": "success",
            "timestamp": time.time(),
            "echo": request or {}
        }
    
    @app.get("/test")
    async def test_endpoint():
        """Test endpoint with sample data"""
        return {
            "test": "success",
            "data": list(range(1, 11)),
            "timestamp": time.time(),
            "server_info": {
                "platform": sys.platform,
                "python": sys.version.split()[0],
                "pid": os.getpid()
            }
        }
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler"""
        print(f"Global exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "timestamp": time.time()
            }
        )
    
    return app

def main():
    """Main server function with Windows-specific optimizations"""
    print("=" * 60)
    print("WINDOWS COMPATIBLE FASTAPI SERVER")
    print("=" * 60)
    
    # Configuration
    host = "0.0.0.0"  # Bind to all interfaces
    port = 8006
    
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Process ID: {os.getpid()}")
    print()
    
    # Check port binding capability
    print("Checking port binding...")
    if not check_port_binding(host, port):
        print(f"Cannot bind to {host}:{port}, trying localhost...")
        host = "127.0.0.1"
        if not check_port_binding(host, port):
            print("Failed to bind to any interface. Exiting.")
            return
    
    print()
    
    # Create app
    app = create_app()
    
    # Uvicorn configuration optimized for Windows
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False,  # Disable reload for stability
        workers=1,     # Single worker for Windows compatibility
        loop="asyncio",  # Explicit event loop
        http="h11",    # Use h11 HTTP implementation
        ws="websockets",  # WebSocket implementation
        lifespan="on",  # Enable lifespan events
        use_colors=True,
        server_header=True,
        date_header=True
    )
    
    print(f"Starting server on {host}:{port}")
    print(f"Access URLs:")
    if host == "0.0.0.0":
        print(f"  - Local: http://127.0.0.1:{port}")
        print(f"  - Network: http://localhost:{port}")
    else:
        print(f"  - Local: http://{host}:{port}")
    print()
    
    # Start server
    try:
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Server shutdown complete")

if __name__ == "__main__":
    main()