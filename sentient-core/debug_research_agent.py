#!/usr/bin/env python3
"""
Debug Research Agent - Check if research agent is working properly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.agents.build2_research_agent import get_build2_research_agent
from core.models import AppState, Message

async def debug_research_agent():
    """Debug the research agent directly."""
    print("🔍 Debugging Research Agent")
    print("=" * 50)
    
    try:
        # Get research agent instance
        research_agent = get_build2_research_agent()
        print(f"✅ Research agent instance created")
        print(f"📁 Research docs path: {research_agent.research_docs_path}")
        print(f"📊 Research docs path exists: {research_agent.research_docs_path.exists()}")
        
        # Check directory permissions
        try:
            test_file = research_agent.research_docs_path / "test_write.txt"
            with open(test_file, 'w') as f:
                f.write("test")
            test_file.unlink()  # Delete test file
            print(f"✅ Directory is writable")
        except Exception as e:
            print(f"❌ Directory write test failed: {e}")
            return False
        
        # Test research agent directly
        print(f"\n🧪 Testing research agent directly...")
        
        query = "What are the best practices for API documentation?"
        session_id = "debug-test-session"
        
        print(f"Query: {query}")
        print(f"Session ID: {session_id}")
        
        # Create initial state
        state = AppState(
            messages=[Message(sender="user", content=query)],
            logs=[],
            session_id=session_id
        )
        
        print(f"\n⏳ Calling research agent...")
        
        # Call research agent
        result_state = await research_agent.conduct_research(query, state, session_id)
        
        print(f"\n📊 Research completed")
        print(f"Messages count: {len(result_state.messages)}")
        print(f"Logs count: {len(result_state.logs)}")
        
        # Print logs
        print(f"\n📝 Research logs:")
        for i, log in enumerate(result_state.logs):
            print(f"  {i+1}. [{log.source}] {log.message}")
        
        # Check for artifacts
        print(f"\n📄 Checking for artifacts...")
        artifacts = list(research_agent.research_docs_path.glob("*.md")) + list(research_agent.research_docs_path.glob("*.pdf"))
        
        if artifacts:
            print(f"✅ Found {len(artifacts)} artifacts:")
            for artifact in artifacts:
                size = artifact.stat().st_size
                print(f"  📄 {artifact.name} ({size} bytes)")
                
                # Show content preview for markdown files
                if artifact.suffix == '.md' and size > 0:
                    try:
                        with open(artifact, 'r', encoding='utf-8') as f:
                            content = f.read(300)
                            print(f"     Preview: {content[:200]}...")
                    except Exception as e:
                        print(f"     Error reading: {e}")
        else:
            print(f"❌ No artifacts found")
            
        # Test invoke method (the one used by orchestrator)
        print(f"\n🔄 Testing invoke method...")
        invoke_result = await research_agent.invoke(query, session_id)
        
        print(f"Invoke result status: {invoke_result.get('status')}")
        print(f"Invoke result message: {invoke_result.get('message')}")
        
        if invoke_result.get('logs'):
            print(f"Invoke logs count: {len(invoke_result['logs'])}")
            for log in invoke_result['logs'][:3]:  # Show first 3
                print(f"  - {log}")
        
        return True
        
    except Exception as e:
        print(f"💥 Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_research_agent())
    if success:
        print("\n✅ Debug completed successfully")
    else:
        print("\n❌ Debug failed")
        exit(1)