#!/usr/bin/env python3
"""
Test script for Build2ResearchAgent using Groq's compound-beta agentic tooling.
This script tests the updated research agent that uses compound-beta and compound-beta-mini models
with built-in agentic capabilities instead of third-party tools.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.agents.build2_research_agent import Build2ResearchAgent
from core.models import AppState, LogEntry

async def test_compound_beta_research():
    """
    Test the Build2ResearchAgent with compound-beta agentic tooling.
    """
    print("=== Testing Build2ResearchAgent with Compound-Beta Agentic Tooling ===")
    
    # Check if GROQ_API_KEY is available
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set")
        return False
    
    try:
        # Initialize the research agent
        print("\n1. Initializing Build2ResearchAgent...")
        agent = Build2ResearchAgent()
        print(f"   - Research model: {agent.research_model}")
        print(f"   - Synthesis model: {agent.synthesis_model}")
        print("   - Agent initialized successfully")
        
        # Create initial state
        from core.models import Message
        state = AppState(
            messages=[],
            logs=[LogEntry(
                source="TestScript",
                message="Starting compound-beta research agent test"
            )],
            session_id="test_session"
        )
        
        # Test query about current AI developments
        test_query = "What are the latest developments in AI agent frameworks and agentic tooling in 2024?"
        print(f"\n2. Testing research query: '{test_query}'")
        
        # Conduct research using compound-beta
        print("\n3. Conducting research with compound-beta agentic tooling...")
        result_state = await agent.conduct_research(test_query, state, "test_session")
        
        # Check results
        print("\n4. Analyzing results...")
        
        # Print logs
        print("\n   Research Logs:")
        for i, log in enumerate(result_state.logs, 1):
            print(f"   {i}. [{log.source}]: {log.message}")
        
        # Check if research artifacts were created
        artifacts_dir = Path("research_artifacts")
        if artifacts_dir.exists():
            print(f"\n   Research artifacts directory exists: {artifacts_dir}")
            
            # List artifact files
            artifact_files = list(artifacts_dir.glob("*"))
            print(f"   Found {len(artifact_files)} artifact files:")
            for file in artifact_files:
                print(f"     - {file.name} ({file.stat().st_size} bytes)")
                
                # Show content preview for markdown files
                if file.suffix == '.md':
                    try:
                        content = file.read_text(encoding='utf-8')
                        preview = content[:500] + "..." if len(content) > 500 else content
                        print(f"       Preview: {preview}")
                    except Exception as e:
                        print(f"       Could not read file: {e}")
        else:
            print("   No research artifacts directory found")
        
        # Test the invoke method as well
        print("\n5. Testing invoke method...")
        invoke_result = await agent.invoke(test_query)
        print(f"   Invoke result type: {type(invoke_result)}")
        print(f"   Invoke result keys: {invoke_result.keys() if hasattr(invoke_result, 'keys') else 'N/A'}")
        
        print("\n=== Test completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"\nERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_model_availability():
    """
    Test if the compound-beta models are available.
    """
    print("\n=== Testing Model Availability ===")
    
    try:
        from groq import Groq
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Test compound-beta
        print("\n1. Testing compound-beta model...")
        response = client.chat.completions.create(
            model="compound-beta",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with web search capabilities."},
                {"role": "user", "content": "Hello, can you search for information about Python programming?"}
            ],
            max_tokens=100
        )
        print(f"   compound-beta response: {response.choices[0].message.content[:100]}...")
        
        # Test compound-beta-mini
        print("\n2. Testing compound-beta-mini model...")
        response = client.chat.completions.create(
            model="compound-beta-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Summarize the benefits of using AI agents."}
            ],
            max_tokens=100
        )
        print(f"   compound-beta-mini response: {response.choices[0].message.content[:100]}...")
        
        print("\n   Both models are available and working!")
        return True
        
    except Exception as e:
        print(f"\nERROR testing models: {str(e)}")
        return False

async def main():
    """
    Main test function.
    """
    print("Starting compound-beta research agent tests...")
    
    # Test model availability first
    models_ok = await test_model_availability()
    if not models_ok:
        print("\nModel availability test failed. Exiting.")
        return
    
    # Test the research agent
    research_ok = await test_compound_beta_research()
    
    if research_ok:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    asyncio.run(main())