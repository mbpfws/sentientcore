#!/usr/bin/env python3
"""
Uvicorn debug test with extensive error handling and logging.
"""

import logging
import traceback
import sys
import os
from fastapi import FastAPI

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('uvicorn_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Create the simplest possible FastAPI app
app = FastAPI(title="Uvicorn Debug Test")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Uvicorn debug server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Uvicorn debug health check"}

def main():
    logger.info("Starting uvicorn debug test...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
    try:
        import uvicorn
        logger.info(f"Uvicorn version: {uvicorn.__version__}")
        logger.info(f"Uvicorn location: {uvicorn.__file__}")
        
        import fastapi
        logger.info(f"FastAPI version: {fastapi.__version__}")
        logger.info(f"FastAPI location: {fastapi.__file__}")
        
        # Try to run uvicorn with extensive error handling
        logger.info("Attempting to start uvicorn server...")
        
        # Use a different approach - create config manually
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8005,
            log_level="debug",
            access_log=True,
            use_colors=False
        )
        
        server = uvicorn.Server(config)
        logger.info("Uvicorn server created successfully")
        
        # Try to run the server
        logger.info("Starting server...")
        server.run()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(traceback.format_exc())
        
        # Try to get more details about the error
        if hasattr(e, '__cause__') and e.__cause__:
            logger.error(f"Caused by: {e.__cause__}")
        if hasattr(e, '__context__') and e.__context__:
            logger.error(f"Context: {e.__context__}")
    
    logger.info("Uvicorn debug test completed")

if __name__ == "__main__":
    main()