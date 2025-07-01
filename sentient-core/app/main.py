import sys
import os

# Add the project's root directory to the Python path
# This MUST be at the top of the file, before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
from core.models import AppState, Message, Task, TaskStatus
from graphs.orchestration_graph import app as orchestration_app

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="Sentient-Core: Walk-Me-Through",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Sentient-Core 🤖")
    st.write("Your AI-driven application development assistant. Start by saying hello!")

    # Initialize session state for the app state
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()

    # Define the two-column layout
    control_column, chat_column = st.columns([0.4, 0.6])

    with control_column:
        st.header("Control Panel")
        
        # Add a button to run the next pending task
        has_pending_tasks = any(task.status == TaskStatus.PENDING for task in st.session_state.app_state.tasks)
        if has_pending_tasks:
            if st.button("Run Next Task", type="primary"):
                with st.spinner("🤖 Agent at work..."):
                    updated_state_dict = orchestration_app.invoke(st.session_state.app_state)
                    st.session_state.app_state = AppState(**updated_state_dict)
                st.rerun()

        st.subheader("Task List")
        if not st.session_state.app_state.tasks:
            st.info("Tasks will appear here once a plan is generated.")
        else:
            for task in st.session_state.app_state.tasks:
                if task.status == TaskStatus.COMPLETED:
                    icon = "✅"
                    with st.expander(f"{icon} ({task.agent}) {task.description}", expanded=False):
                        st.markdown(task.result or "_No result recorded._")
                elif task.status == TaskStatus.IN_PROGRESS:
                    icon = "⏳"
                    st.write(f"{icon} ({task.agent}) {task.description}")
                else: # PENDING
                    icon = "📝"
                    st.write(f"{icon} ({task.agent}) {task.description}")


    with chat_column:
        st.header("Conversation")

        # Display chat messages
        for msg in st.session_state.app_state.messages:
            with st.chat_message(msg.sender):
                if msg.sender == "user" and msg.image:
                    st.image(msg.image, width=200)
                st.write(msg.content)

        # User input area
        uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
        
        if prompt := st.chat_input("What would you like to build?"):
            image_bytes = None
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()

            # Update state with new prompt and potential image
            st.session_state.app_state.user_prompt = prompt
            st.session_state.app_state.image = image_bytes

            # Append a new message to the history to be displayed
            new_user_message = Message(sender="user", content=prompt)
            if image_bytes:
                # Add a reference to the image in the message object for display
                # Note: This is for display only. The actual bytes are in app_state.image
                new_user_message.image = image_bytes # type: ignore
            st.session_state.app_state.messages.append(new_user_message)

            with st.spinner("Assistant is thinking..."):
                updated_state_dict = orchestration_app.invoke(st.session_state.app_state)
                st.session_state.app_state = AppState(**updated_state_dict)
            
            st.rerun()

if __name__ == "__main__":
    main()
