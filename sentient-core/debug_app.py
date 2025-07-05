from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Starting debug app...")

app = FastAPI(
    title="Sentient Core Debug API",
    description="Debug version of Sentient Core API",
    version="0.1.0"
)

print("FastAPI app created")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("CORS middleware added")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Sentient Core Debug API is running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentient-core-debug"}

print("Routes defined")

# Try importing core models gradually
try:
    print("Importing core models...")
    from core.models import AppState, Message, EnhancedTask, TaskStatus, LogEntry
    print("Core models imported successfully")
except Exception as e:
    print(f"Error importing core models: {e}")

# Try importing routers one at a time
try:
    print("Importing agents router...")
    from app.api.routers import agents
    app.include_router(agents.router, prefix="/api")
    print("Agents router imported successfully")
except Exception as e:
    print(f"Error importing agents router: {e}")

try:
    print("Importing workflows router...")
    from app.api.routers import workflows
    app.include_router(workflows.router, prefix="/api")
    print("Workflows router imported successfully")
except Exception as e:
    print(f"Error importing workflows router: {e}")

try:
    print("Importing chat router...")
    from app.api.routers import chat
    app.include_router(chat.router, prefix="/api")
    print("Chat router imported successfully")
except Exception as e:
    print(f"Error importing chat router: {e}")

try:
    print("Importing core_services router...")
    from app.api.routers import core_services
    app.include_router(core_services.router, prefix="/api")
    print("Core services router imported successfully")
except Exception as e:
    print(f"Error importing core_services router: {e}")

print("Debug app setup complete")
print("All routers loaded successfully!")