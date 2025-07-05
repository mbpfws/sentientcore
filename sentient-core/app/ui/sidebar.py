# sentient-core/app/ui/sidebar.py

import streamlit as st
from core.models import AppState, TaskStatus, LogEntry

def render_sidebar():
    """Render the sidebar UI for the Multi-Agent RAG system."""
    with st.sidebar:
        st.header("🤖 Multi-Agent RAG System")

        # --- Workflow Mode ---
        mode_options = ["intelligent", "multi_agent", "legacy"]
        current_index = mode_options.index(st.session_state.workflow_mode) if st.session_state.workflow_mode in mode_options else 0

        st.session_state.workflow_mode = st.selectbox(
            "Workflow Mode",
            options=mode_options,
            index=current_index,
            help="Select workflow: Intelligent (NLU), Multi-Agent RAG, or Legacy"
        )

        st.divider()

        # --- Display Options ---
        st.subheader("📊 Display Options")
        st.session_state.show_step_details = st.toggle(
            "Show Step Details",
            value=st.session_state.show_step_details,
            help="Show step-by-step reasoning and execution logs"
        )

        st.session_state.show_raw_output = st.toggle(
            "Show Raw Output",
            value=st.session_state.show_raw_output,
            help="Show raw agent outputs alongside friendly responses"
        )

        st.divider()

        # --- Progress ---
        st.subheader("🔄 Workflow Progress")
        if st.session_state.app_state.tasks:
            completed = sum(1 for t in st.session_state.app_state.tasks if t.status == TaskStatus.COMPLETED)
            total = len(st.session_state.app_state.tasks)
            st.progress(completed / total, text=f"Tasks: {completed}/{total}")

            if st.session_state.app_state.logs:
                st.info(f"Current: {st.session_state.app_state.logs[-1].source}")

        st.divider()

        # --- Logs ---
        st.subheader("📝 Action Logs")
        if st.session_state.app_state.logs:
            with st.expander("View Logs", expanded=False):
                for log in st.session_state.app_state.logs[-10:]:
                    st.text(f"[{log.source}] {log.message}")
        else:
            st.text("No logs yet")

        st.divider()

        # --- Session Controls ---
        if st.button("🔄 Reset Session", type="secondary"):
            st.session_state.app_state = AppState()
            st.rerun()

        st.header("🧰 Control Panel")
        if st.button("Clear Session State", key="clear_session"):
            st.session_state.app_state = AppState()
            st.session_state.research_mode = None
            st.session_state.show_raw_searches = False
            st.session_state.search_streams = []
            st.rerun()

        # --- Task List ---
        st.header("📌 Tasks")
        if not st.session_state.app_state.tasks:
            st.info("No tasks created yet.")
        else:
            for i, task in enumerate(st.session_state.app_state.tasks):
                col1, col2 = st.columns([4, 1])
                with col1:
                    icon = {
                        TaskStatus.COMPLETED: "✅",
                        TaskStatus.IN_PROGRESS: "⏳",
                        TaskStatus.PENDING: "📝"
                    }.get(task.status, "❔")
                    st.write(f"{icon} ({task.agent}) {task.description}")
                with col2:
                    if task.status == TaskStatus.PENDING:
                        if st.button("Run", key=f"run_task_{task.id}_{i}"):
                            st.session_state.app_state.task_to_run_id = task.id
                            st.rerun()

        # --- Developer Logs ---
        st.header("💻 Developer Logs")
        if st.toggle("Show Developer Logs", value=False, key="dev_logs_toggle"):
            if st.session_state.app_state.logs:
                log_container = st.container(height=300)
                for log in reversed(st.session_state.app_state.logs):
                    log_container.info(f"[{log.source}] {log.message}")
            else:
                st.info("No log entries yet.")
