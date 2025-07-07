#!/usr/bin/env python3
"""
Debug script to test the exact same initialization flow as main app
"""

import asyncio
import logging
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_main_app_flow():
    """Test the exact same flow as the main app"""
    try:
        logger.info("Starting main app flow test...")
        
        # Import services exactly like main app
        from app.services.service_factory import get_service_factory, initialize_services, cleanup_services
        
        logger.info("Imported service factory successfully")
        
        # Initialize services exactly like the lifespan function
        logger.info("Calling initialize_services()...")
        success = await initialize_services()
        
        if not success:
            logger.error("initialize_services() returned False")
            return False
        
        logger.info("initialize_services() returned True")
        
        # Get service factory like the app does
        service_factory = get_service_factory()
        logger.info(f"Got service factory: {service_factory}")
        
        # Test health check
        logger.info("Testing health check...")
        health_status = await service_factory.health_check()
        logger.info(f"Health status: {health_status}")
        
        # Test service status
        logger.info("Testing service status...")
        service_status = service_factory.get_service_status()
        logger.info(f"Service status: {service_status}")
        
        logger.info("✅ Main app flow test completed successfully!")
        
        # Cleanup
        await cleanup_services()
        logger.info("Services cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Main app flow test failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Try to cleanup on error
        try:
            from app.services.service_factory import cleanup_services
            await cleanup_services()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")
        
        return False

if __name__ == "__main__":
    result = asyncio.run(test_main_app_flow())
    if result:
        print("\n🎉 Main app flow test PASSED!")
    else:
        print("\n💥 Main app flow test FAILED!")
        sys.exit(1)