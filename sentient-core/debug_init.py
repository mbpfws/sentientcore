#!/usr/bin/env python3
"""
Debug script to test workflow initialization
"""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.graphs.sentient_workflow_graph import get_sentient_workflow_app

async def test_initialization():
    """Test the workflow initialization and see debug output"""
    print("Starting workflow initialization test...")
    
    try:
        # Get the workflow app
        app = get_sentient_workflow_app()
        print(f"Workflow app created: {type(app)}")
        
        # Test a simple conversation
        result = await app.ainvoke({
            "messages": [{"role": "user", "content": "Hello, can you help me?"}]
        })
        
        print(f"Response: {result}")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_initialization())