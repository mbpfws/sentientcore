"""
Quick test to verify the app input error is fixed
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import AppState, Message
from graphs.intelligent_rag_graph import intelligent_workflow

def test_app_input():
    """Test the same input that was causing the error."""
    
    print("Testing: 'I wanna build an app'")
    print("=" * 40)
    
    # Create test state with the problematic input
    state = AppState(
        messages=[Message(sender="user", content="I wanna build an app")],
        tasks=[],
        logs=[]
    )
    
    try:
        # Run the workflow
        result = intelligent_workflow.workflow.invoke(state.model_dump())
        
        print("SUCCESS: No routing error!")
        
        # Check the result
        if isinstance(result, dict):
            next_action = result.get("next_action", "unknown")
            print(f"Next action: {next_action}")
            
            # Check for assistant messages
            messages = result.get("messages", [])
            if messages:
                latest_msg = messages[-1]
                if isinstance(latest_msg, dict) and latest_msg.get("sender") == "assistant":
                    print(f"Assistant response: {latest_msg.get('content', 'No content')[:100]}...")
                    
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing App Input Fix")
    print("=" * 40)
    
    success = test_app_input()
    
    if success:
        print("\nThe fix works! The app should now handle 'I wanna build an app' properly.")
    else:
        print("\nThe error still exists. Please check the implementation.")