#!/usr/bin/env python3
"""
Direct test of Build 2 research functionality without web server
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing Build 2 Research Functionality (Direct)")
print("=" * 50)

try:
    # Import our components
    print("1. Importing components...")
    from core.models import AppState, Message
    from core.agents.ultra_orchestrator import UltraOrchestrator
    from core.services.enhanced_llm_service_main import EnhancedLLMService
    print("✓ Successfully imported all components")
    
    # Initialize services
    print("\n2. Initializing services...")
    llm_service = EnhancedLLMService()
    orchestrator = UltraOrchestrator(llm_service)
    print("✓ Successfully initialized UltraOrchestrator")
    
    # Create app state
    print("\n3. Creating app state...")
    app_state = AppState()
    print("✓ Successfully created app state")
    
    # Test different research modes
    test_cases = [
        ("knowledge", "What is machine learning?"),
        ("deep", "Explain quantum computing in detail"),
        ("best-in-class", "What are the latest developments in AI?")
    ]
    
    async def test_research_mode(mode, query):
        print(f"\n4. Testing {mode} research mode...")
        try:
            # Add research mode prefix like the chat router does
            research_prefix = {
                "knowledge": "Please conduct a Knowledge Research",
                "deep": "Please conduct a Deep Research",
                "best-in-class": "Please conduct a Best-in-Class Research"
            }.get(mode, "")
            
            message_text = f"{research_prefix}: {query}" if research_prefix else query
            
            # Create message and add to app state
            message = Message(
                sender="user",
                content=message_text
            )
            
            # Add message to app state
            app_state.messages.append(message)
            
            # Process with orchestrator
            result = await orchestrator.invoke_state(app_state)
            
            print(f"✓ {mode} mode processed successfully")
            print(f"  Response type: {type(result)}")
            if hasattr(result, 'content'):
                print(f"  Response length: {len(str(result.content))} characters")
            
            # Check for research delegation keywords
            result_str = str(result).lower()
            research_keywords = ['research', 'delegate', 'agent', 'search', 'investigate']
            found_keywords = [kw for kw in research_keywords if kw in result_str]
            
            if found_keywords:
                print(f"  ✓ Research delegation detected: {found_keywords}")
            else:
                print(f"  ⚠ No clear research delegation detected")
                
            return True
            
        except Exception as e:
            print(f"  ✗ Error in {mode} mode: {str(e)}")
            return False
    
    # Run tests
    async def run_all_tests():
        results = []
        for mode, query in test_cases:
            success = await test_research_mode(mode, query)
            results.append((mode, success))
        
        print("\n" + "=" * 50)
        print("Test Results Summary:")
        for mode, success in results:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  {mode}: {status}")
        
        total_passed = sum(1 for _, success in results if success)
        print(f"\nOverall: {total_passed}/{len(results)} tests passed")
        
        if total_passed == len(results):
            print("🎉 All Build 2 research functionality tests PASSED!")
        else:
            print("⚠ Some tests failed - Build 2 needs attention")
    
    # Run the tests
    asyncio.run(run_all_tests())
    
except Exception as e:
    print(f"\n✗ Critical error during testing: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nDirect testing completed.")