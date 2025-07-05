import os
import sys
import os
from dotenv import load_dotenv

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

# This MUST be the first thing in the file, before any other imports
# to ensure that the 'core' and 'graphs' modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import base64
import streamlit as st
from fpdf import FPDF
from core.models import AppState, Message, EnhancedTask, TaskStatus, LogEntry
from core.orchestration import (
    initialize_workflow_orchestrator,
    get_workflow_orchestrator,
    shutdown_workflow_orchestrator
)
from core.services.state_service import StateService
from core.services.llm_service import EnhancedLLMService
import time
import json
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_pdf(text: str) -> bytes:
    """Creates a PDF file from a string of text using a Unicode font."""
    try:
        pdf = FPDF()
        pdf.add_page()
        font_path = "dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf"
        if not os.path.exists(font_path):
            st.error(f"Font file not found at {font_path}. Please ensure it's in the correct location.")
            return b""
        
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
        
        # Handle Unicode text properly
        # Replace problematic characters that can't be encoded
        safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
        pdf.multi_cell(0, 10, safe_text)
        return bytes(pdf.output(dest='S'))
    except Exception as e:
        st.error(f"PDF generation error: {str(e)}")
        return b""

async def initialize_services():
    """Initialize core services asynchronously."""
    try:
        # Initialize state service
        if "state_service" not in st.session_state:
            st.session_state.state_service = StateService()
            await st.session_state.state_service.initialize()
        
        # Initialize LLM service
        if "llm_service" not in st.session_state:
            st.session_state.llm_service = EnhancedLLMService()
            await st.session_state.llm_service.initialize()
        
        # Initialize workflow orchestrator
        if "workflow_orchestrator" not in st.session_state:
            st.session_state.workflow_orchestrator = await initialize_workflow_orchestrator(
                st.session_state.state_service,
                st.session_state.llm_service
            )
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        st.error(f"Failed to initialize system services: {e}")
        return False

def initialize_session_state():
    """Initialize all session state variables."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    if "workflow_mode" not in st.session_state:
        st.session_state.workflow_mode = "intelligent"  # "intelligent", "multi_agent", or "legacy"
    if "show_step_details" not in st.session_state:
        st.session_state.show_step_details = True
    if "show_raw_output" not in st.session_state:
        st.session_state.show_raw_output = False
    if "research_mode" not in st.session_state:
        st.session_state.research_mode = None
    if "search_streams" not in st.session_state:
        st.session_state.search_streams = []
    if "show_raw_searches" not in st.session_state:
        st.session_state.show_raw_searches = False
    if "services_initialized" not in st.session_state:
        st.session_state.services_initialized = False

def render_sidebar():
    """Render the sidebar with controls and logs."""
    with st.sidebar:
        st.header("🤖 Multi-Agent RAG System")
        
        # Workflow mode selection
        mode_options = ["intelligent", "multi_agent", "end_to_end", "legacy"]
        current_index = mode_options.index(st.session_state.workflow_mode) if st.session_state.workflow_mode in mode_options else 0
        
        st.session_state.workflow_mode = st.selectbox(
            "Workflow Mode", 
            mode_options,
            index=current_index,
            help="Select workflow: Intelligent (natural language), Multi-Agent RAG, End-to-End (full pipeline), or Legacy mode"
        )
        
        # End-to-end workflow type selection
        if st.session_state.workflow_mode == "end_to_end":
            if "end_to_end_type" not in st.session_state:
                st.session_state.end_to_end_type = "full"
            
            st.session_state.end_to_end_type = st.selectbox(
                "End-to-End Type",
                ["full", "research_only", "development_only"],
                index=["full", "research_only", "development_only"].index(st.session_state.end_to_end_type),
                help="Full: Complete pipeline, Research: Research + Knowledge only, Development: Architecture + Development only"
            )
        
        st.divider()
        
        # Display controls
        st.subheader("📊 Display Options")
        st.session_state.show_step_details = st.toggle(
            "Show Step Details", 
            value=st.session_state.show_step_details,
            help="Show detailed step-by-step workflow progress"
        )
        
        st.session_state.show_raw_output = st.toggle(
            "Show Raw Output", 
            value=st.session_state.show_raw_output,
            help="Display raw agent outputs alongside natural language responses"
        )
        
        st.divider()
        
        # Workflow progress
        st.subheader("🔄 Workflow Progress")
        if st.session_state.app_state.tasks:
            completed_tasks = sum(1 for task in st.session_state.app_state.tasks if task.status == TaskStatus.COMPLETED)
            total_tasks = len(st.session_state.app_state.tasks)
            progress = completed_tasks / total_tasks if total_tasks > 0 else 0
            
            st.progress(progress, text=f"Tasks: {completed_tasks}/{total_tasks}")
            
            # Show current phase
            if st.session_state.app_state.logs:
                latest_log = st.session_state.app_state.logs[-1]
                st.info(f"Current: {latest_log.source}")
        
        st.divider()
        
        # Action logs
        st.subheader("📝 Action Logs")
        if st.session_state.app_state.logs:
            with st.expander("View Logs", expanded=False):
                for log in st.session_state.app_state.logs[-10:]:  # Show last 10 logs
                    st.text(f"[{log.source}] {log.message}")
        else:
            st.text("No logs yet")
        
        st.divider()
        
        # Reset button
        if st.button("🔄 Reset Session", type="secondary"):
            # Cleanup services first
            cleanup_services()
            
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # Reinitialize basic session state
            initialize_session_state()
            st.rerun()
        st.header("Control Panel")

        if st.button("Clear Session State", key="clear_session"):
            st.session_state.app_state = AppState()
            st.session_state.research_mode = None
            st.session_state.show_raw_searches = False
            st.session_state.search_streams = []
            st.rerun()

        st.header("Tasks")
        if not st.session_state.app_state.tasks:
            st.info("No tasks created yet.")
        else:
            for i, task in enumerate(st.session_state.app_state.tasks):
                col1, col2 = st.columns([4, 1])
                with col1:
                    icon = "✅" if task.status == TaskStatus.COMPLETED else "⏳" if task.status == TaskStatus.IN_PROGRESS else "📝"
                    st.write(f"{icon} ({task.agent}) {task.description}")
                with col2:
                    if task.status == TaskStatus.PENDING:
                        if st.button("Run", key=f"run_task_{task.id}_{i}"):
                            st.session_state.app_state.task_to_run_id = task.id
                            st.rerun()

        st.header("Developer Logs")
        show_logs = st.toggle("Show Developer Logs", value=False, key="dev_logs_toggle")
        if show_logs:
            if st.session_state.app_state.logs:
                log_container = st.container(height=300)
                for log in reversed(st.session_state.app_state.logs):
                    log_container.info(f"[{log.source}] {log.message}")
            else:
                st.info("No log entries yet.")

def render_research_modes():
    """Render research mode selection buttons."""
    st.markdown("### 🔍 Research Modes")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Knowledge Research", 
                    help="Multi-source search with consolidation", 
                    use_container_width=True,
                    key="knowledge_mode_btn"):
            st.session_state.research_mode = "knowledge"
    
    with col2:
        if st.button("🧠 Deep Research", 
                    help="In-depth analysis with citations and reasoning", 
                    use_container_width=True,
                    key="deep_mode_btn"):
            st.session_state.research_mode = "deep"
    
    with col3:
        if st.button("🏆 Best-in-Class", 
                    help="Comparative analysis to find optimal solutions", 
                    use_container_width=True,
                    key="best_mode_btn"):
            st.session_state.research_mode = "best_in_class"
    
    # Display selected mode
    if st.session_state.research_mode:
        mode_descriptions = {
            "knowledge": "📚 **Knowledge Research**: Multi-source keyword search with comprehensive consolidation",
            "deep": "🧠 **Deep Research**: In-depth analysis with sophisticated reasoning and citations",
            "best_in_class": "🏆 **Best-in-Class Research**: Comparative analysis to identify optimal solutions"
        }
        st.info(mode_descriptions.get(st.session_state.research_mode, ""))

def render_chat_messages():
    """Render all chat messages."""
    for msg in st.session_state.app_state.messages:
        with st.chat_message(msg.sender):
            if msg.image:
                st.image(msg.image, width=250)
            st.write(msg.content)

def render_task_results():
    """Render completed task results with download options."""
    for i, task in enumerate(st.session_state.app_state.tasks):
        if task.status == TaskStatus.COMPLETED and task.result:
            with st.chat_message("assistant"):
                with st.expander(f"Result for: {task.description}", expanded=True):
                    st.markdown(task.result)
                    
                    # Add download buttons
                    b_col1, b_col2 = st.columns(2)
                    with b_col1:
                        try:
                            pdf_data = create_pdf(task.result)
                            st.download_button(
                                label="Download as PDF",
                                data=pdf_data,
                                file_name=f"task_{task.id}_result.pdf",
                                mime="application/pdf",
                                key=f"pdf_download_{task.id}_{i}"
                            )
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")
                    with b_col2:
                        st.download_button(
                            label="Download as Markdown",
                            data=task.result.encode('utf-8'),
                            file_name=f"task_{task.id}_result.md",
                            mime="text/markdown",
                            key=f"md_download_{task.id}_{i}"
                        )

                    # Display follow-up questions
                    if task.follow_up_questions and len(task.follow_up_questions) > 0:
                        st.markdown("---")
                        st.markdown("#### Continual Search Suggestions:")
                        for j, question in enumerate(task.follow_up_questions):
                            if st.button(question, key=f"follow_up_{task.id}_{i}_{j}"):
                                st.session_state.new_prompt = question
                                st.rerun()
            
            # Mark task as "seen" to prevent re-displaying results
            task.status = TaskStatus.DONE

def render_raw_search_toggle():
    """Render the raw search toggle and display search instances."""
    # Only show toggle if we have search streams
    if st.session_state.search_streams:
        st.session_state.show_raw_searches = st.toggle(
            "🔍 Show Raw Search Instances", 
            value=st.session_state.show_raw_searches,
            help="Toggle to see individual search queries and their raw results",
            key="raw_search_toggle_main"
        )
        
        if st.session_state.show_raw_searches:
            with st.expander("🔍 Raw Search Instances", expanded=True):
                for i, search_data in enumerate(st.session_state.search_streams):
                    st.markdown(f"**Search {i+1}:** `{search_data.get('query', 'Unknown')}`")
                    
                    if search_data.get('status') == 'in_progress':
                        st.markdown("🔄 *Searching...*")
                        if search_data.get('accumulated'):
                            st.markdown(search_data['accumulated'])
                    elif search_data.get('status') == 'completed':
                        st.markdown("✅ *Completed*")
                        if search_data.get('result'):
                            st.markdown(search_data['result'])
                    
                    if i < len(st.session_state.search_streams) - 1:
                        st.markdown("---")

async def handle_user_input():
    """Handle user input and process it through the orchestration graph."""
    # Check for new prompts from chat input or follow-up clicks
    prompt = st.chat_input("What do you want to build today?")
    if "new_prompt" in st.session_state and st.session_state.new_prompt:
        prompt = st.session_state.new_prompt
        st.session_state.new_prompt = None

    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key="image_upload")

    if prompt:
        # Clear previous search streams for new query
        st.session_state.search_streams = []
        
        image_data = None
        if uploaded_file:
            image_data = uploaded_file.getvalue()
        
        # Add research mode to the prompt if selected
        if st.session_state.research_mode:
            research_mode_text = {
                "knowledge": "Please conduct a Knowledge Research",
                "deep": "Please conduct a Deep Research",
                "best_in_class": "Please conduct a Best-in-Class Research"
            }
            mode_instruction = research_mode_text.get(st.session_state.research_mode, "")
            prompt = f"{mode_instruction}: {prompt}"
        
        # Ensure services are initialized
        if not st.session_state.services_initialized:
            with st.spinner("Initializing system services..."):
                success = await initialize_services()
                if not success:
                    return
                st.session_state.services_initialized = True
        
        # Add user message to state
        st.session_state.app_state.messages.append(Message(sender="user", content=prompt, image=image_data))
        st.session_state.app_state.image = image_data
        
        # Process through orchestration graph
        with st.spinner("Agent is thinking..."):
            try:
                # Get the workflow orchestrator
                orchestrator = get_workflow_orchestrator()
                if not orchestrator:
                    st.error("Workflow orchestrator not available")
                    return
                
                # Process through orchestrator
                result = await orchestrator.execute_workflow(
                    user_input=prompt,
                    workflow_mode=st.session_state.workflow_mode,
                    research_mode=st.session_state.research_mode,
                    image_data=image_data
                )
                
                # Add response message
                response_content = result.get("response", "Request processed successfully.")
                st.session_state.app_state.messages.append(
                    Message(sender="assistant", content=response_content)
                )
                
                # Add tasks if any were created
                if "tasks" in result:
                    for task_data in result["tasks"]:
                        task = EnhancedTask(
                            id=task_data.get("id", f"task_{len(st.session_state.app_state.tasks) + 1}"),
                            description=task_data.get("description", "Task"),
                            status=TaskStatus(task_data.get("status", "pending")),
                            agent=task_data.get("agent", "orchestrator"),
                            result=task_data.get("result")
                        )
                        st.session_state.app_state.tasks.append(task)
                
                # Add log entries if any
                if "logs" in result:
                    for log_data in result["logs"]:
                        log_entry = LogEntry(
                            source=log_data.get("source", "orchestrator"),
                            message=log_data.get("message", "")
                        )
                        st.session_state.app_state.logs.append(log_entry)
                        
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                st.error(f"Error processing request: {e}")
                st.session_state.app_state.logs.append(
                    LogEntry(source="UI", message=f"Error: {e}")
                )

        # Clear research mode selection after use
        st.session_state.research_mode = None
        st.rerun()

def handle_pending_tasks():
    """Handle execution of pending tasks."""
    if st.session_state.app_state.task_to_run_id:
        with st.spinner("Agent is working on the task..."):
            try:
                # TODO: Implement task execution logic
                st.session_state.app_state.logs.append(
                    LogEntry(source="UI", message="Executing task...")
                )
            except Exception as e:
                st.error(f"Error executing task: {e}")
                st.session_state.app_state.logs.append(
                    LogEntry(source="UI", message=f"Task execution error: {e}")
                )
        st.session_state.app_state.task_to_run_id = None
        st.rerun()

async def handle_workflow_execution(user_input: str, uploaded_image: bytes = None):
    """Execute the appropriate workflow based on user selection."""
    try:
        # Ensure services are initialized
        if not st.session_state.services_initialized:
            with st.spinner("Initializing system services..."):
                success = await initialize_services()
                if not success:
                    return
                st.session_state.services_initialized = True
        
        # Add user message to state
        st.session_state.app_state.messages.append(
            Message(sender="user", content=user_input, image=uploaded_image)
        )
        
        # Set image in state if provided
        if uploaded_image:
            st.session_state.app_state.image = uploaded_image
        
        # Get the workflow orchestrator
        orchestrator = get_workflow_orchestrator()
        if not orchestrator:
            st.error("Workflow orchestrator not available")
            return
        
        # Process through orchestrator
        with st.spinner(f"Processing with {st.session_state.workflow_mode} workflow..."):
            if st.session_state.workflow_mode == "end_to_end":
                # Use the new end-to-end workflow
                workflow_type = getattr(st.session_state, 'end_to_end_type', 'full')
                session_id = f"session_{int(time.time())}"
                
                result = await orchestrator.execute_end_to_end_workflow(
                    user_input=user_input,
                    session_id=session_id,
                    workflow_type=workflow_type
                )
            else:
                # Use existing workflow modes
                result = await orchestrator.execute_workflow(
                    user_input=user_input,
                    workflow_mode=st.session_state.workflow_mode,
                    research_mode=st.session_state.research_mode,
                    image_data=uploaded_image
                )
        
        # Handle response based on workflow mode
        if st.session_state.workflow_mode == "end_to_end":
            # Handle end-to-end workflow results
            if result.get("success", False):
                # Add messages from the workflow
                if "messages" in result:
                    for msg_data in result["messages"]:
                        st.session_state.app_state.messages.append(
                            Message(
                                sender=msg_data.get("sender", "assistant"),
                                content=msg_data.get("content", "Workflow completed."),
                                image=msg_data.get("image")
                            )
                        )
                
                # Add execution results as logs
                if "execution_results" in result:
                    for exec_result in result["execution_results"]:
                        log_entry = LogEntry(
                            source=f"Graph: {exec_result.get('graph_type', 'Unknown')}",
                            message=f"Status: {exec_result.get('status', 'Unknown')} ({exec_result.get('duration_seconds', 0):.2f}s)"
                        )
                        st.session_state.app_state.logs.append(log_entry)
                
                # Store artifacts in session state for later access
                if "artifacts" in result:
                    if "workflow_artifacts" not in st.session_state:
                        st.session_state.workflow_artifacts = {}
                    st.session_state.workflow_artifacts.update(result["artifacts"])
                
                # Add workflow logs
                if "logs" in result:
                    for log_data in result["logs"]:
                        log_entry = LogEntry(
                            source=log_data.get("source", "End-to-End Workflow"),
                            message=log_data.get("message", "")
                        )
                        st.session_state.app_state.logs.append(log_entry)
            else:
                # Handle workflow error
                error_msg = result.get("error", "End-to-end workflow failed")
                st.session_state.app_state.messages.append(
                    Message(sender="assistant", content=f"❌ **Workflow Error**: {error_msg}")
                )
        else:
            # Handle traditional workflow results
            response_content = result.get("response", "Request processed successfully.")
            st.session_state.app_state.messages.append(
                Message(sender="assistant", content=response_content)
            )
            
            # Add tasks if any were created
            if "tasks" in result:
                for task_data in result["tasks"]:
                    task = EnhancedTask(
                        id=task_data.get("id", f"task_{len(st.session_state.app_state.tasks) + 1}"),
                        description=task_data.get("description", "Task"),
                        status=TaskStatus(task_data.get("status", "pending")),
                        agent=task_data.get("agent", "orchestrator"),
                        result=task_data.get("result")
                    )
                    st.session_state.app_state.tasks.append(task)
            
            # Add log entries if any
            if "logs" in result:
                for log_data in result["logs"]:
                    log_entry = LogEntry(
                        source=log_data.get("source", "orchestrator"),
                        message=log_data.get("message", "")
                    )
                    st.session_state.app_state.logs.append(log_entry)
        
        # Clear image after processing
        st.session_state.app_state.image = None
        
    except Exception as e:
        error_msg = f"Workflow execution error: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.session_state.app_state.logs.append(
            LogEntry(source="UI", message=error_msg)
        )

def render_executable_tasks():
    """Render executable task buttons in sequence order."""
    if st.session_state.app_state.tasks:
        # Filter tasks that are ready for execution (pending status)
        pending_tasks = [
            task for task in st.session_state.app_state.tasks 
            if task.status == TaskStatus.PENDING
        ]
        
        if pending_tasks:
            st.markdown("### 🎯 Executable Tasks")
            st.markdown("Click on tasks below to execute them in sequence:")
            
            # Sort by sequence if available
            pending_tasks.sort(key=lambda x: getattr(x, 'sequence', 1))
            
            for i, task in enumerate(pending_tasks):
                # Create columns for task layout
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Task description
                    sequence_num = getattr(task, 'sequence', i + 1)
                    agent_type = getattr(task, 'agent', 'unknown')
                    agent_emoji = {
                        'research': '🔬',
                        'architecture': '🏗️', 
                        'design': '🎨',
                        'builder': '🚀'
                    }.get(agent_type, '⚙️')
                    
                    st.markdown(f"**{sequence_num}. {agent_emoji} {task.description}**")
                    
                    # Show dependencies if any
                    dependencies = getattr(task, 'dependencies', [])
                    if dependencies:
                        st.caption(f"Dependencies: {', '.join(dependencies)}")
                
                with col2:
                    # Agent type badge
                    st.markdown(f"`{agent_type.upper()}`")
                
                with col3:
                    # Execute button
                    if st.button(
                        "▶️ Execute", 
                        key=f"execute_task_{task.id}",
                        type="primary",
                        help=f"Execute this {agent_type} task"
                    ):
                        # Execute the task
                        try:
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(execute_single_task(task.id))
                            loop.close()
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error executing task: {e}")
                            st.error(f"Error executing task: {e}")
                
                st.divider()

async def execute_single_task(task_id: str):
    """Execute a single task by ID."""
    # Find the task
    task = None
    for t in st.session_state.app_state.tasks:
        if t.id == task_id:
            task = t
            break
    
    if task:
        try:
            # Ensure services are initialized
            if not st.session_state.services_initialized:
                st.error("Services not initialized. Please refresh the page.")
                return
            
            # Mark task as in progress
            task.status = TaskStatus.IN_PROGRESS
            
            # Add a message about task execution
            st.session_state.app_state.messages.append(
                Message(
                    sender="assistant",
                    content=f"⚡ Executing task: {task.description}"
                )
            )
            
            # Get the workflow orchestrator
            orchestrator = get_workflow_orchestrator()
            if not orchestrator:
                st.error("Workflow orchestrator not available")
                task.status = TaskStatus.FAILED
                return
            
            # Execute task through orchestrator
            with st.spinner(f"Executing {task.description}..."):
                result = await orchestrator.execute_task(
                    task_id=task.id,
                    task_description=task.description,
                    agent_source=getattr(task, 'agent', 'unknown')
                )
                
                if result.get("success", False):
                    task.result = result.get("result", "Task completed successfully")
                    task.status = TaskStatus.COMPLETED
                    st.success(f"Task completed: {task.description}")
                else:
                    task.result = result.get("error", "Task execution failed")
                    task.status = TaskStatus.FAILED
                    st.error(f"Task failed: {task.result}")
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            task.status = TaskStatus.FAILED
            task.result = f"Task failed: {str(e)}"
            st.error(f"Task failed: {e}")

def render_workflow_artifacts():
    """Render workflow artifacts from end-to-end execution."""
    if hasattr(st.session_state, 'workflow_artifacts') and st.session_state.workflow_artifacts:
        st.markdown("### 📦 Workflow Artifacts")
        
        # Create tabs for different artifact types
        artifact_types = list(st.session_state.workflow_artifacts.keys())
        if artifact_types:
            tabs = st.tabs([f"📊 {artifact_type.replace('_', ' ').title()}" for artifact_type in artifact_types])
            
            for i, (artifact_type, artifacts) in enumerate(st.session_state.workflow_artifacts.items()):
                with tabs[i]:
                    if isinstance(artifacts, dict):
                        # Display structured artifacts
                        for key, value in artifacts.items():
                            if isinstance(value, (str, int, float, bool)):
                                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                            elif isinstance(value, dict):
                                with st.expander(f"📋 {key.replace('_', ' ').title()}", expanded=False):
                                    st.json(value)
                            elif isinstance(value, list):
                                with st.expander(f"📝 {key.replace('_', ' ').title()}", expanded=False):
                                    for item in value:
                                        if isinstance(item, str):
                                            st.markdown(f"• {item}")
                                        else:
                                            st.json(item)
                    else:
                        # Display simple artifacts
                        st.markdown(str(artifacts))
                    
                    # Add download button for artifacts
                    if st.button(f"📥 Download {artifact_type.replace('_', ' ').title()} Artifacts", key=f"download_{artifact_type}"):
                        artifact_json = json.dumps(artifacts, indent=2, default=str)
                        st.download_button(
                            label=f"Download {artifact_type}.json",
                            data=artifact_json.encode('utf-8'),
                            file_name=f"{artifact_type}_artifacts.json",
                            mime="application/json",
                            key=f"download_btn_{artifact_type}"
                        )

def render_workflow_status():
    """Render current workflow status and phase information."""
    if st.session_state.workflow_mode == "end_to_end":
        st.markdown("### 🔄 End-to-End Workflow Status")
        
        # Show execution summary if available
        if st.session_state.app_state.logs:
            latest_log = st.session_state.app_state.logs[-1]
            st.info(f"**Latest Update**: {latest_log.source} - {latest_log.message}")
            
            # Show execution timeline
            if len(st.session_state.app_state.logs) > 1:
                with st.expander("Execution Timeline", expanded=False):
                    for log in st.session_state.app_state.logs[-10:]:
                        st.text(f"✓ {log.source}: {log.message}")
        
        # Render artifacts
        render_workflow_artifacts()
        
    elif st.session_state.workflow_mode in ["intelligent", "multi_agent"] and st.session_state.show_step_details:
        st.markdown("### 🔄 Multi-Agent Workflow Status")
        
        # Show current phase based on latest logs
        if st.session_state.app_state.logs:
            latest_log = st.session_state.app_state.logs[-1]
            
            # Map agent sources to workflow phases
            phase_mapping = {
                "Orchestrator": "🎯 Phase 1: Understanding & Refinement",
                "Monitor": "🛡️ Phase 1: Validation & Compliance",
                "IntentClassifier": "🧠 Phase 1: Intent Analysis",
                "ResearchCoordinator": "🔬 Phase 2a: Research & Knowledge Synthesis",
                "KnowledgeSynthesizer": "📚 Phase 2a: Knowledge Consolidation",
                "ArchitecturePlanner": "🏗️ Phase 2b: Architecture & Planning",
                "DesignGenerator": "🎨 Phase 2c: Design & UX Generation",
                "ActionPlanner": "📋 Phase 2d: Final Planning",
                "Builder": "🚀 Phase 3: Solution Development"
            }
            
            current_phase = phase_mapping.get(latest_log.source, f"🤖 {latest_log.source}")
            st.info(f"**Current Phase**: {current_phase}")
            
            # Show recent steps
            if len(st.session_state.app_state.logs) > 1:
                with st.expander("Recent Steps", expanded=False):
                    for log in st.session_state.app_state.logs[-5:]:
                        phase = phase_mapping.get(log.source, log.source)
                        st.text(f"✓ {phase}: {log.message}")

def main():
    """Main application entry point with proper multi-agent RAG workflow."""
    st.set_page_config(
        page_title="Sentient Core - Multi-Agent RAG System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize services if not already done
    if not st.session_state.services_initialized:
        try:
            # Run async initialization in a sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(initialize_services())
            loop.close()
            
            if success:
                st.session_state.services_initialized = True
            else:
                st.error("Failed to initialize system services. Please refresh the page.")
                return
        except Exception as e:
            logger.error(f"Error initializing services: {e}")
            st.error(f"Error initializing services: {e}")
            return
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("🤖 Sentient Core - Multi-Agent RAG System")
    st.markdown("**Foundational Multi-Agent Workflow**: Sophisticated AI-driven solution generation with step-by-step guidance")
    
    # Workflow status (for multi-agent mode)
    if st.session_state.workflow_mode == "multi_agent":
        render_workflow_status()
    
    # Executable tasks (for intelligent workflow)
    if st.session_state.workflow_mode == "intelligent":
        render_executable_tasks()
    
    st.markdown("---")
    
    # Display chat messages
    render_chat_messages()
    
    # Display task results with proper encoding
    render_task_results()
    
    # Raw output toggle (for debugging)
    if st.session_state.show_raw_output and st.session_state.app_state.logs:
        with st.expander("🔧 Raw Agent Output", expanded=False):
            for log in st.session_state.app_state.logs[-10:]:
                st.code(f"[{log.source}] {log.message}", language="text")
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload an image (optional)", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to provide visual context for your request"
    )
    
    uploaded_image = None
    if uploaded_file:
        uploaded_image = uploaded_file.read()
        st.image(uploaded_image, width=200, caption="Uploaded image")
    
    # Chat input
    if prompt := st.chat_input("Describe what you want to build or research..."):
        try:
            # Run async workflow execution in a sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handle_workflow_execution(prompt, uploaded_image))
            loop.close()
            st.rerun()
        except Exception as e:
            logger.error(f"Error handling user input: {e}")
            st.error(f"Error processing your request: {e}")
    
    # Handle follow-up questions from session state
    if hasattr(st.session_state, 'new_prompt') and st.session_state.new_prompt:
        try:
            # Run async workflow execution in a sync context
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(handle_workflow_execution(st.session_state.new_prompt))
            loop.close()
            st.session_state.new_prompt = None
            st.rerun()
        except Exception as e:
            logger.error(f"Error handling follow-up prompt: {e}")
            st.error(f"Error processing follow-up request: {e}")

def cleanup_services():
    """Cleanup function to properly shutdown services."""
    try:
        if "workflow_orchestrator" in st.session_state:
            # Shutdown workflow orchestrator
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(shutdown_workflow_orchestrator())
            loop.close()
            
        # Clear service references
        for key in ["state_service", "llm_service", "workflow_orchestrator"]:
            if key in st.session_state:
                del st.session_state[key]
                
        st.session_state.services_initialized = False
        logger.info("Services cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        cleanup_services()
    except Exception as e:
        logger.error(f"Application error: {e}")
        cleanup_services()
        raise