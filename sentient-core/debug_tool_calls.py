#!/usr/bin/env python3
"""
Debug Tool Calls - Check the exact structure of tool_calls causing 400 error
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.services.enhanced_llm_service import EnhancedLLMService
from core.agents.build2_research_agent import Build2ResearchAgent

async def debug_tool_calls():
    """Debug the exact tool_calls structure causing issues."""
    print("🔍 Debugging Tool Calls Structure")
    print("=" * 50)
    
    try:
        # Initialize LLM service
        llm_service = EnhancedLLMService()
        print(f"✅ LLM service initialized")
        
        # Get available models
        models = llm_service.get_available_models()
        print(f"📊 Available models: {models}")
        
        # Test simple tool call
        messages = [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "Search for information about Python best practices"}
        ]
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        print(f"\n🧪 Testing direct tool call...")
        print(f"Messages: {json.dumps(messages, indent=2)}")
        print(f"Tools: {json.dumps(tools, indent=2)}")
        
        # Test with EnhancedLLMService
        try:
            response = await llm_service.generate_with_tools(
                messages=messages,
                tools=tools,
                provider="groq",
                model="llama-3.3-70b-versatile"
            )
            
            print(f"\n✅ Tool call successful!")
            print(f"Content: {response.get('content')}")
            print(f"Tool calls: {response.get('tool_calls')}")
            print(f"Finish reason: {response.get('finish_reason')}")
            
            # Test adding tool_calls to messages
            if response.get('tool_calls'):
                print(f"\n🔄 Testing message with tool_calls...")
                
                test_messages = messages + [
                    {
                        "role": "assistant",
                        "content": response.get('content') or "",
                        "tool_calls": response.get('tool_calls')
                    }
                ]
                
                print(f"Test messages structure:")
                for i, msg in enumerate(test_messages):
                    print(f"  Message {i}: {json.dumps(msg, indent=4, default=str)}")
                
                # Try another call with this structure
                try:
                    response2 = await llm_service.generate_with_tools(
                        messages=test_messages,
                        tools=tools,
                        provider="groq",
                        model="llama-3.3-70b-versatile"
                    )
                    print(f"\n✅ Second call with tool_calls successful!")
                except Exception as e:
                    print(f"\n❌ Second call failed: {e}")
                    print(f"Error type: {type(e)}")
                    
                    # Try without tool_calls
                    clean_messages = []
                    for msg in test_messages:
                        clean_msg = {"role": msg["role"], "content": msg.get("content", "")}
                        if msg["role"] != "tool" and clean_msg["content"].strip():
                            clean_messages.append(clean_msg)
                    
                    print(f"\n🧹 Trying with cleaned messages...")
                    try:
                        response3 = await llm_service.generate(
                            messages=clean_messages,
                            provider="groq",
                            model="llama-3.3-70b-versatile"
                        )
                        print(f"✅ Clean messages call successful!")
                        print(f"Response: {response3[:200]}...")
                    except Exception as e2:
                        print(f"❌ Clean messages call also failed: {e2}")
            
        except Exception as e:
            print(f"❌ Tool call failed: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"💥 Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_tool_calls())
    if success:
        print("\n✅ Debug completed successfully")
    else:
        print("\n❌ Debug failed")
        exit(1)