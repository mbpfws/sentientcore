from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.models import AppState, Message, EnhancedTask, TaskStatus, LogEntry

app = FastAPI(
    title="Sentient Core API - Debug",
    description="Minimal API for debugging",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Basic app created successfully")

# Try importing routers one by one
try:
    print("Importing agents router...")
    from app.api.routers import agents
    app.include_router(agents.router, prefix="/api")
    print("✅ Agents router imported successfully")
except Exception as e:
    print(f"❌ Failed to import agents router: {e}")

try:
    print("Importing workflows router...")
    from app.api.routers import workflows
    app.include_router(workflows.router, prefix="/api")
    print("✅ Workflows router imported successfully")
except Exception as e:
    print(f"❌ Failed to import workflows router: {e}")

try:
    print("Importing chat router...")
    from app.api.routers import chat
    app.include_router(chat.router, prefix="/api")
    print("✅ Chat router imported successfully")
except Exception as e:
    print(f"❌ Failed to import chat router: {e}")

try:
    print("Importing core_services router...")
    from app.api.routers import core_services
    app.include_router(core_services.router, prefix="/api")
    print("✅ Core services router imported successfully")
except Exception as e:
    print(f"❌ Failed to import core_services router: {e}")

try:
    print("Importing api_endpoints router...")
    from app.api.routers import api_endpoints
    app.include_router(api_endpoints.router, prefix="/api")
    print("✅ API endpoints router imported successfully")
except Exception as e:
    print(f"❌ Failed to import api_endpoints router: {e}")

try:
    print("Importing interactive_workflows router...")
    from app.api.routers import interactive_workflows
    app.include_router(interactive_workflows.router, prefix="/api")
    print("✅ Interactive workflows router imported successfully")
except Exception as e:
    print(f"❌ Failed to import interactive_workflows router: {e}")

try:
    print("Importing research router...")
    from app.api.routers import research
    app.include_router(research.router, prefix="/api")
    print("✅ Research router imported successfully")
except Exception as e:
    print(f"❌ Failed to import research router: {e}")

try:
    print("Importing monitoring router...")
    from app.api.routers import monitoring
    app.include_router(monitoring.router, prefix="/api")
    print("✅ Monitoring router imported successfully")
except Exception as e:
    print(f"❌ Failed to import monitoring router: {e}")

try:
    print("Importing implementation router...")
    from app.api.routers import implementation
    app.include_router(implementation.router, prefix="/api")
    print("✅ Implementation router imported successfully")
except Exception as e:
    print(f"❌ Failed to import implementation router: {e}")

print("All router imports completed")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"status": "ok", "message": "Debug Sentient Core API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Debug Sentient Core API is operational"}

print("App setup completed successfully")

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)