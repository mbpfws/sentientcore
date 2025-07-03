"""
Test script for the Intelligent Orchestrator and RAG workflow
"""

import sys
import os
from dotenv import load_dotenv

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import AppState, Message, EnhancedTask, TaskStatus
from core.agents.intelligent_orchestrator import IntelligentOrchestrator
from core.services.llm_service import EnhancedLLMService
from graphs.intelligent_rag_graph import intelligent_workflow
import json

def test_intelligent_orchestrator():
    """Test the intelligent orchestrator with various input scenarios."""
    
    print("Testing Intelligent Orchestrator...")
    print("=" * 60)
    
    # Initialize components
    llm_service = EnhancedLLMService()
    orchestrator = IntelligentOrchestrator(llm_service)
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Vague Request",
            "input": "I want to build something cool",
            "expected_decision": "request_clarification"
        },
        {
            "name": "Off-topic Request", 
            "input": "What's the weather today?",
            "expected_decision": "redirect_conversation"
        },
        {
            "name": "Clear Development Request",
            "input": "I want to create a Vietnamese IELTS writing practice app with AI feedback features",
            "expected_decision": "create_task_breakdown"
        },
        {
            "name": "Research Request",
            "input": "Can you research the best frameworks for building AI-powered educational apps?",
            "expected_decision": "create_task_breakdown"
        },
        {
            "name": "Vietnamese Input",
            "input": "Tôi muốn tạo một ứng dụng học tiếng Anh cho trẻ em",
            "expected_decision": "create_task_breakdown"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. Testing: {scenario['name']}")
        print(f"Input: {scenario['input']}")
        print("-" * 40)
        
        # Create test state
        state = AppState(
            messages=[Message(sender="user", content=scenario['input'])],
            tasks=[],
            logs=[]
        )
        
        try:
            # Test orchestrator
            result = orchestrator.analyze_and_orchestrate(state)
            
            # Extract decision
            decision_data = result.get("orchestrator_decision", {})
            decision = decision_data.get("decision", "unknown")
            message = decision_data.get("message", "No message")
            
            print(f"Decision: {decision}")
            print(f"Response: {message}")
            
            # Check if decision matches expected
            if decision == scenario["expected_decision"]:
                print(f"PASS: Decision matches expected ({scenario['expected_decision']})")
            else:
                print(f"FAIL: Expected {scenario['expected_decision']}, got {decision}")
            
            # Show created tasks
            if "task_breakdown" in decision_data and decision_data["task_breakdown"]:
                print(f"Tasks created: {len(decision_data['task_breakdown'])}")
                for enhanced_task in decision_data["task_breakdown"]:
                    print(f"   - {enhanced_task.get('title', 'Untitled')} ({enhanced_task.get('agent_type', 'unknown')})")
            
            # Show follow-up questions if any
            if "follow_up_questions" in decision_data and decision_data["follow_up_questions"]:
                print(f"Follow-up questions: {len(decision_data['follow_up_questions'])}")
                for q in decision_data["follow_up_questions"]:
                    print(f"   - {q}")
                    
        except Exception as e:
            print(f"ERROR: {str(e)}")
        
        print("-" * 40)

def test_full_workflow():
    """Test the complete intelligent RAG workflow."""
    
    print("\nTesting Complete Intelligent RAG Workflow...")
    print("=" * 60)
    
    # Test with a clear request that should create tasks
    test_input = "I want to create a personal finance tracking app with AI-powered insights and budget recommendations"
    
    state = AppState(
        messages=[Message(sender="user", content=test_input)],
        tasks=[],
        logs=[]
    )
    
    print(f"Input: {test_input}")
    print("-" * 40)
    
    try:
        # Run the full workflow
        result = intelligent_workflow.app.invoke(state.model_dump())
        
        # Convert result back to AppState for analysis
        updated_state = AppState(**result)
        
        print(f"✅ Workflow completed successfully")
        print(f"📬 Messages: {len(updated_state.messages)}")
        print(f"📋 Tasks: {len(updated_state.tasks)}")
        print(f"📊 Logs: {len(updated_state.logs)}")
        
        # Show latest assistant message
        if updated_state.messages:
            latest_message = updated_state.messages[-1]
            if latest_message.sender == "assistant":
                print(f"💬 Latest response: {latest_message.content[:200]}...")
        
        # Show created tasks
        if updated_state.tasks:
            print("\n📋 Created Tasks:")
            for enhanced_task in updated_state.tasks:
                status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(enhanced_task.status, "❓")
                print(f"   {status_emoji} {enhanced_task.description[:100]}... ({enhanced_task.agent})")
        
        # Show recent logs
        if updated_state.logs:
            print("\n📊 Recent Logs:")
            for log_entry in updated_state.logs[-3:]:
                print(f"   [{log_entry.source}] {log_entry.message}")
                
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_language_detection():
    """Test language detection and multilingual responses."""
    
    print("\nTesting Language Detection...")
    print("=" * 60)
    
    llm_service = EnhancedLLMService()
    orchestrator = IntelligentOrchestrator(llm_service)
    
    language_tests = [
        {
            "language": "Vietnamese",
            "input": "Tôi cần tạo một website bán hàng online",
            "expected_lang": "vi"
        },
        {
            "language": "English", 
            "input": "I need to build an e-commerce website",
            "expected_lang": "en"
        }
    ]
    
    for test in language_tests:
        print(f"\nTesting {test['language']}:")
        print(f"Input: {test['input']}")
        
        state = AppState(
            messages=[Message(sender="user", content=test['input'])],
            tasks=[],
            logs=[]
        )
        
        try:
            result = orchestrator.analyze_and_orchestrate(state)
            decision_data = result.get("orchestrator_decision", {})
            detected_lang = decision_data.get("language_detected", "unknown")
            
            print(f"Detected language: {detected_lang}")
            
            if detected_lang == test["expected_lang"]:
                print("✅ PASS: Language detection correct")
            else:
                print(f"❌ FAIL: Expected {test['expected_lang']}, got {detected_lang}")
                
        except Exception as e:
            print(f"ERROR: {str(e)}")

if __name__ == "__main__":
    print("=== Intelligent Orchestrator Test Suite ===")
    print("=" * 60)
    
    # Run all tests
    test_intelligent_orchestrator()
    test_language_detection()
    
    # Test full workflow
    success = test_full_workflow()
    
    print("\n" + "=" * 60)
    if success:
        print("All tests completed! The Intelligent Orchestrator is ready.")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)