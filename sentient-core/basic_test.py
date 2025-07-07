#!/usr/bin/env python3
"""
Basic Python Test - Check if basic imports work
"""

import sys
import os
from datetime import datetime

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Current time: {datetime.now()}")

# Test basic imports
try:
    import fastapi
    print(f"✓ FastAPI version: {fastapi.__version__}")
except ImportError as e:
    print(f"✗ FastAPI import failed: {e}")

try:
    import uvicorn
    print(f"✓ Uvicorn imported successfully")
except ImportError as e:
    print(f"✗ Uvicorn import failed: {e}")

try:
    import pydantic
    print(f"✓ Pydantic version: {pydantic.__version__}")
except ImportError as e:
    print(f"✗ Pydantic import failed: {e}")

# Test if we can create a basic FastAPI app
try:
    from fastapi import FastAPI
    app = FastAPI()
    print("✓ FastAPI app created successfully")
except Exception as e:
    print(f"✗ FastAPI app creation failed: {e}")

print("\nBasic test completed.")