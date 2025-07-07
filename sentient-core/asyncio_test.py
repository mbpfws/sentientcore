#!/usr/bin/env python3
"""
Asyncio Event Loop Test
Tests if the issue is with asyncio event loops on Windows
"""

import asyncio
import aiohttp
from aiohttp import web
import sys
import time
from datetime import datetime

async def hello_handler(request):
    """Simple hello handler"""
    return web.Response(text="Hello from asyncio server!")

async def health_handler(request):
    """Health check handler"""
    return web.json_response({"status": "healthy", "timestamp": datetime.now().isoformat()})

async def test_asyncio_server():
    """Test basic asyncio server functionality"""
    print("Testing asyncio server...")
    
    try:
        # Create aiohttp application
        app = web.Application()
        app.router.add_get('/', hello_handler)
        app.router.add_get('/health', health_handler)
        
        # Create server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '127.0.0.1', 8777)
        await site.start()
        
        print("✓ Asyncio server started on http://127.0.0.1:8777")
        
        # Test the server
        await asyncio.sleep(1)  # Give server time to start
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://127.0.0.1:8777/health') as response:
                    data = await response.json()
                    print(f"✓ Server response: {data}")
        except Exception as e:
            print(f"✗ Failed to connect to server: {e}")
        
        # Keep server running for a moment
        await asyncio.sleep(2)
        
        # Cleanup
        await runner.cleanup()
        print("✓ Server stopped cleanly")
        
    except Exception as e:
        print(f"✗ Asyncio server test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_uvicorn_import():
    """Test if uvicorn can be imported and basic functionality works"""
    print("\nTesting uvicorn import...")
    
    try:
        import uvicorn
        print(f"✓ Uvicorn imported successfully: {uvicorn.__version__}")
        
        # Test FastAPI import
        from fastapi import FastAPI
        print(f"✓ FastAPI imported successfully")
        
        # Create a simple FastAPI app
        app = FastAPI()
        
        @app.get("/")
        async def root():
            return {"message": "Hello World"}
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        print("✓ FastAPI app created successfully")
        
        # Test if we can create uvicorn config
        config = uvicorn.Config(
            app=app,
            host="127.0.0.1",
            port=8778,
            log_level="info"
        )
        print("✓ Uvicorn config created successfully")
        
        # Try to create server instance
        server = uvicorn.Server(config)
        print("✓ Uvicorn server instance created successfully")
        
    except Exception as e:
        print(f"✗ Uvicorn/FastAPI test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_event_loop_policies():
    """Test different event loop policies on Windows"""
    print("\nTesting event loop policies...")
    
    try:
        # Get current policy
        current_policy = asyncio.get_event_loop_policy()
        print(f"Current event loop policy: {type(current_policy).__name__}")
        
        # Test if we're on Windows
        if sys.platform == 'win32':
            print("Running on Windows - testing ProactorEventLoop")
            
            # Try ProactorEventLoop (recommended for Windows)
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print("✓ ProactorEventLoop policy set successfully")
                
                # Test basic async operation
                await asyncio.sleep(0.1)
                print("✓ Basic async operation works with ProactorEventLoop")
                
            except Exception as e:
                print(f"✗ ProactorEventLoop test failed: {e}")
            
            # Try SelectorEventLoop
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print("✓ SelectorEventLoop policy set successfully")
                
                # Test basic async operation
                await asyncio.sleep(0.1)
                print("✓ Basic async operation works with SelectorEventLoop")
                
            except Exception as e:
                print(f"✗ SelectorEventLoop test failed: {e}")
        
    except Exception as e:
        print(f"✗ Event loop policy test failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main test function"""
    print("Asyncio and Uvicorn Diagnostic Tool")
    print("=" * 50)
    
    await test_event_loop_policies()
    await test_uvicorn_import()
    await test_asyncio_server()
    
    print("\n=== Asyncio Diagnostic Complete ===")

if __name__ == "__main__":
    # Set ProactorEventLoop for Windows if available
    if sys.platform == 'win32':
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except:
            pass
    
    asyncio.run(main())