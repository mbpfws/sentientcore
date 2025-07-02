"""
Ultra Orchestrator Agent with Assistant
The primary conductor of the multi-agent RAG system with comprehensive capabilities:
- Conversation management across languages and modalities
- Task assignment and workflow orchestration
- Anti-exploitation measures and session management
- Specialized agent creation for edge cases
- 4-layer memory management integration
"""

from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta
import uuid
import json
import re
from enum import Enum

from core.enhanced_models import (
    SystemState, SessionContext, SessionState, EnhancedTask, TaskStatus, AgentType,
    MemoryEntry, MemoryLayer, ConversationTurn, TaskDependencyGraph, SystemConfiguration
)
from core.models import AppState, Message, LogEntry
from core.services.llm_service import LLMService


class ConversationPhase(str, Enum):
    INITIAL_CONTACT = "initial_contact"
    CLARIFICATION = "clarification"
    RESEARCH_PLANNING = "research_planning"
    ARCHITECTURE_PLANNING = "architecture_planning"
    TASK_BREAKDOWN = "task_breakdown"
    EXECUTION_MONITORING = "execution_monitoring"
    USER_FEEDBACK = "user_feedback"
    COMPLETION = "completion"


class UltraOrchestrator:
    """
    Ultra Orchestrator: The master conductor of the multi-agent RAG system.
    Handles conversation flow, task orchestration, and system coordination.
    """
    
    def __init__(self, llm_service: LLMService, system_config: SystemConfiguration):
        self.llm_service = llm_service
        self.config = system_config
        self.assistant_id = str(uuid.uuid4())
        
        # Memory and state management
        self.system_state: Optional[SystemState] = None
        self.task_dependency_graph = TaskDependencyGraph()
        
        # Anti-exploitation tracking
        self.banned_ips: Dict[str, datetime] = {}
        self.session_warnings: Dict[str, int] = {}
        
    def initialize_session(self, user_id: str, user_ip: str = "unknown") -> SystemState:
        """Initialize a new system session with comprehensive state management."""
        
        # Check if IP is banned
        if self._is_ip_banned(user_ip):
            raise Exception(f"IP {user_ip} is temporarily banned due to exploitation attempts")
        
        # Create new session context
        session = SessionContext(
            user_id=user_id,
            state=SessionState.ACTIVE,
            detected_language="en"  # Will be detected from first message
        )
        
        # Initialize system state
        self.system_state = SystemState(
            session=session,
            tasks=[],
            agents=[],
            knowledge_synthesis=[],
            build_conversation=[],
            codebase_knowledge=[],
            stack_dependencies=[],
            active_graphs=[],
            logs=[],
            metrics={"session_start": datetime.now().isoformat()}
        )
        
        self._log("UltraOrchestrator", f"New session initialized for user {user_id}")
        return self.system_state
    
    def process_user_input(
        self, 
        user_input: str, 
        attachments: Optional[List[str]] = None,
        audio_data: Optional[bytes] = None,
        image_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Process user input with comprehensive analysis and decision making.
        Supports text, image, audio, and document inputs.
        """
        if not self.system_state:
            raise Exception("Session not initialized. Call initialize_session first.")
        
        # Anti-exploitation check
        self._check_exploitation_attempts(user_input)
        
        # Detect language and update session
        detected_language = self._detect_language(user_input)
        self.system_state.session.detected_language = detected_language
        
        # Add conversation turn
        turn = ConversationTurn(
            sender="user",
            content=user_input,
            language=detected_language,
            attachments=attachments or [],
            metadata={
                "has_audio": audio_data is not None,
                "has_image": image_data is not None,
                "turn_number": len(self.system_state.session.conversation_turns) + 1
            }
        )
        self.system_state.session.conversation_turns.append(turn)
        self.system_state.session.total_turns += 1
        self.system_state.session.last_activity = datetime.now()
        
        # Analyze input and make orchestration decisions
        orchestration_decision = self._analyze_and_orchestrate(
            user_input, image_data, audio_data, attachments
        )
        
        # Update conversation phase
        self._update_conversation_phase(orchestration_decision)
        
        # Execute orchestration decision
        response = self._execute_orchestration_decision(orchestration_decision)
        
        # Add assistant response to conversation
        assistant_turn = ConversationTurn(
            sender="ultra_orchestrator",
            content=response.get("message", ""),
            language=detected_language,
            agent_id=self.assistant_id,
            metadata=orchestration_decision
        )
        self.system_state.session.conversation_turns.append(assistant_turn)
        
        return response
    
    def _analyze_and_orchestrate(
        self, 
        user_input: str, 
        image_data: Optional[bytes] = None,
        audio_data: Optional[bytes] = None,
        attachments: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis and orchestration decision making.
        Uses advanced models with multimodal capabilities.
        """
        
        # Choose appropriate model based on input modality
        primary_model = self._select_optimal_model(image_data, audio_data)
        
        # Build comprehensive context
        conversation_context = self._build_conversation_context()
        system_context = self._build_system_context()
        
        # Advanced orchestration prompt
        orchestration_prompt = self._get_orchestration_system_prompt()
        
        # Construct input for LLM
        full_context = f"""
{orchestration_prompt}

CONVERSATION CONTEXT:
{conversation_context}

SYSTEM CONTEXT:
{system_context}

CURRENT USER INPUT: {user_input}
DETECTED LANGUAGE: {self.system_state.session.detected_language}
CONVERSATION PHASE: {self._get_current_phase()}
"""
        
        # Get orchestration decision with fallback
        decision_response = self._get_orchestration_decision(
            full_context, primary_model, image_data
        )
        
        return decision_response
    
    def _get_orchestration_system_prompt(self) -> str:
        """Comprehensive system prompt for Ultra Orchestrator."""
        return """
You are the Ultra Orchestrator, the master conductor of a sophisticated multi-agent RAG system. You are the PRIMARY decision-maker with the following core capabilities:

CORE INTELLIGENCE:
1. **CONVERSATION MASTERY**: 
   - Communicate fluently in any language (English, Vietnamese, Chinese, etc.)
   - Handle multimodal inputs (text, images, audio, documents)
   - Guide users from vague requests to detailed, actionable specifications
   - Maintain conversation flow across multiple sessions

2. **TASK ORCHESTRATION**:
   - Assign tasks to specialized agents (Research, Architecture, Frontend, Backend, Coding)
   - Ensure sequential task execution with proper dependencies
   - Create new specialized agents for edge cases
   - Monitor and coordinate all agent activities

3. **ANTI-EXPLOITATION MEASURES**:
   - Detect off-topic conversations and redirect professionally
   - Track conversation efficiency and warn after 7 unproductive turns
   - Identify potential system abuse and implement protective measures
   - Maintain session integrity and user engagement quality

4. **WORKFLOW INTELLIGENCE**:
   - Decide when to search for additional information
   - Determine when sufficient detail exists for task breakdown
   - Choose appropriate graphs and sub-graphs for execution
   - Balance autonomy with user involvement

5. **MEMORY INTEGRATION**:
   - Utilize 4-layer memory system (Knowledge, Conversation, Codebase, Dependencies)
   - Maintain context across complex multi-turn conversations
   - Synthesize information from multiple sources and agents

RESPONSE FORMAT - Return JSON:
{
    "decision_type": "continue_conversation|create_tasks|search_required|redirect_off_topic|request_clarification|assign_specialized_agent",
    "message": "Your response in the user's detected language",
    "conversation_phase": "initial_contact|clarification|research_planning|architecture_planning|task_breakdown|execution_monitoring|user_feedback|completion",
    "confidence_score": 0.95,
    "language_detected": "en|vi|zh|fr|es|etc",
    "requires_search": true|false,
    "search_queries": ["query 1", "query 2"] if search needed,
    "task_assignments": [
        {
            "agent_type": "research|architecture|frontend|backend|coding|specialized",
            "task_title": "Brief title",
            "task_description": "Detailed description",
            "priority": 1-5,
            "estimated_duration": 30,
            "dependencies": ["task_id_1", "task_id_2"],
            "requires_user_confirmation": true|false
        }
    ],
    "specialized_agent_creation": {
        "needed": true|false,
        "agent_name": "Custom Agent Name",
        "capabilities": ["capability1", "capability2"],
        "purpose": "Why this agent is needed"
    },
    "memory_updates": {
        "knowledge_synthesis": "Information to store in knowledge layer",
        "build_conversation": "Conversation context to preserve",
        "codebase_knowledge": "Code-related information",
        "stack_dependencies": "Technology stack information"
    },
    "user_warning": {
        "needed": true|false,
        "message": "Warning message if needed",
        "severity": "low|medium|high"
    },
    "next_actions": ["action1", "action2"],
    "reasoning": "Brief explanation of your decision-making process"
}

INTELLIGENCE PRINCIPLES:
- Be proactive in gathering requirements but avoid overwhelming users
- Balance technical sophistication with user-friendly communication  
- Ensure every decision advances toward a concrete, actionable outcome
- Maintain professional tone while being conversational and helpful
- Protect system resources while maximizing user value
"""

    def _select_optimal_model(self, image_data: Optional[bytes], audio_data: Optional[bytes]) -> str:
        """Select optimal model based on input modality and current system load."""
        
        # Multimodal inputs require advanced models
        if image_data or audio_data:
            return self.config.primary_models.get("ultra_orchestrator", "gemini-2.5-flash")
        
        # Complex reasoning tasks
        if self._get_current_phase() in [ConversationPhase.ARCHITECTURE_PLANNING, ConversationPhase.TASK_BREAKDOWN]:
            return self.config.primary_models.get("ultra_orchestrator", "gemini-2.5-flash")
        
        # Standard conversation
        return "llama-3.3-70b-versatile"
    
    def _build_conversation_context(self) -> str:
        """Build comprehensive conversation context."""
        if not self.system_state or not self.system_state.session.conversation_turns:
            return "No previous conversation history."
        
        context_parts = []
        
        # Recent conversation turns (last 10)
        recent_turns = self.system_state.session.conversation_turns[-10:]
        context_parts.append("RECENT CONVERSATION:")
        
        for i, turn in enumerate(recent_turns, 1):
            role = turn.sender.upper()
            content_preview = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            context_parts.append(f"{i}. [{role}]: {content_preview}")
        
        # Session metadata
        context_parts.append(f"\nSESSION INFO:")
        context_parts.append(f"- Total turns: {self.system_state.session.total_turns}")
        context_parts.append(f"- Current state: {self.system_state.session.state}")
        context_parts.append(f"- User expertise level: {self.system_state.session.user_expertise_level or 'unknown'}")
        context_parts.append(f"- Industry context: {self.system_state.session.industry_context or 'unknown'}")
        context_parts.append(f"- Goal clarity score: {self.system_state.session.goal_clarity_score}")
        
        return "\n".join(context_parts)
    
    def _build_system_context(self) -> str:
        """Build comprehensive system context."""
        if not self.system_state:
            return "No system state available."
        
        context_parts = []
        
        # Active tasks
        active_tasks = [t for t in self.system_state.tasks if t.status != TaskStatus.COMPLETED]
        context_parts.append(f"ACTIVE TASKS: {len(active_tasks)}")
        for task in active_tasks[:5]:  # Show up to 5 active tasks
            context_parts.append(f"- {task.title} ({task.status}) - {task.agent_type}")
        
        # Available agents
        context_parts.append(f"\nAVAILABLE AGENTS: {len(self.system_state.agents)}")
        for agent in self.system_state.agents[:5]:
            context_parts.append(f"- {agent.name} ({agent.agent_type}) - {agent.status}")
        
        # Memory layers status
        context_parts.append(f"\nMEMORY LAYERS:")
        context_parts.append(f"- Knowledge synthesis: {len(self.system_state.knowledge_synthesis)} entries")
        context_parts.append(f"- Build conversation: {len(self.system_state.build_conversation)} entries")
        context_parts.append(f"- Codebase knowledge: {len(self.system_state.codebase_knowledge)} entries")
        context_parts.append(f"- Stack dependencies: {len(self.system_state.stack_dependencies)} entries")
        
        # Recent logs
        if self.system_state.logs:
            context_parts.append(f"\nRECENT SYSTEM ACTIVITY:")
            for log in self.system_state.logs[-3:]:
                context_parts.append(f"- [{log.get('source', 'Unknown')}]: {log.get('message', '')}")
        
        return "\n".join(context_parts)
    
    def _get_orchestration_decision(
        self, 
        context: str, 
        model: str, 
        image_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Get orchestration decision from LLM with fallback handling."""
        
        fallback_models = self.config.fallback_models.get("groq", []) + self.config.fallback_models.get("google", [])
        models_to_try = [model] + [m for m in fallback_models if m != model]
        
        for current_model in models_to_try:
            try:
                self._log("UltraOrchestrator", f"Attempting decision with model: {current_model}")
                
                response = self.llm_service.generate_response(
                    model_name=current_model,
                    prompt=context,
                    image_bytes=image_data,
                    stream=False
                )
                
                # Parse JSON response
                cleaned_response = self._clean_json_response(str(response))
                decision_data = json.loads(cleaned_response)
                
                self._log("UltraOrchestrator", f"Successfully got decision using {current_model}")
                return decision_data
                
            except Exception as e:
                self._log("UltraOrchestrator", f"Error with model {current_model}: {str(e)}")
                if "rate_limit" in str(e).lower() and current_model == model:
                    continue
                elif current_model == models_to_try[-1]:  # Last model failed
                    break
                else:
                    continue
        
        # Fallback response if all models fail
        return self._get_fallback_decision()
    
    def _get_fallback_decision(self) -> Dict[str, Any]:
        """Fallback decision when all models fail."""
        return {
            "decision_type": "continue_conversation",
            "message": "I'm experiencing some technical difficulties. Could you please rephrase your request?",
            "conversation_phase": "clarification",
            "confidence_score": 0.1,
            "language_detected": self.system_state.session.detected_language,
            "requires_search": False,
            "search_queries": [],
            "task_assignments": [],
            "specialized_agent_creation": {"needed": False},
            "memory_updates": {},
            "user_warning": {"needed": False},
            "next_actions": ["await_user_input"],
            "reasoning": "System fallback due to model failures"
        }
    
    def _execute_orchestration_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the orchestration decision and update system state."""
        
        decision_type = decision.get("decision_type", "continue_conversation")
        
        # Update memory layers if specified
        self._update_memory_layers(decision.get("memory_updates", {}))
        
        # Handle task assignments
        if decision.get("task_assignments"):
            self._create_and_assign_tasks(decision["task_assignments"])
        
        # Handle specialized agent creation
        if decision.get("specialized_agent_creation", {}).get("needed"):
            self._create_specialized_agent(decision["specialized_agent_creation"])
        
        # Execute search if required
        if decision.get("requires_search"):
            self._execute_search_queries(decision.get("search_queries", []))
        
        # Handle user warnings
        if decision.get("user_warning", {}).get("needed"):
            self._handle_user_warning(decision["user_warning"])
        
        # Update system metrics
        self._update_system_metrics(decision)
        
        self._log("UltraOrchestrator", f"Executed decision: {decision_type}")
        
        return {
            "message": decision.get("message", ""),
            "decision_type": decision_type,
            "conversation_phase": decision.get("conversation_phase"),
            "next_actions": decision.get("next_actions", []),
            "system_state": self.system_state.model_dump() if self.system_state else None
        }
    
    def _clean_json_response(self, response: str) -> str:
        """Clean and extract JSON from LLM response."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response.strip()
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection (can be enhanced with proper language detection)."""
        # Basic language detection based on character sets and common words
        if re.search(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text):
            return "vi"  # Vietnamese
        elif re.search(r'[你我他她它们的是在有不了人这中国时候]', text):
            return "zh"  # Chinese
        elif re.search(r'[がのにをはでとから日本語ひらがなカタカナ]', text):
            return "ja"  # Japanese
        else:
            return "en"  # Default to English
    
    def _check_exploitation_attempts(self, user_input: str):
        """Check for exploitation attempts and implement protective measures."""
        session_id = self.system_state.session.session_id
        
        # Check conversation efficiency
        if self.system_state.session.total_turns >= self.config.warning_threshold:
            if self.system_state.session.goal_clarity_score < 0.3:  # Low progress
                warning_count = self.session_warnings.get(session_id, 0) + 1
                self.session_warnings[session_id] = warning_count
                
                if warning_count >= 2:
                    # Ban IP for specified duration
                    self._ban_ip(self.system_state.session.user_id)
                    raise Exception("Session terminated due to unproductive usage patterns")
                else:
                    self.system_state.session.warning_count = warning_count
    
    def _ban_ip(self, user_ip: str):
        """Ban an IP address for specified duration."""
        ban_until = datetime.now() + timedelta(hours=self.config.ban_duration_hours)
        self.banned_ips[user_ip] = ban_until
        self._log("UltraOrchestrator", f"IP {user_ip} banned until {ban_until}")
    
    def _is_ip_banned(self, user_ip: str) -> bool:
        """Check if an IP is currently banned."""
        if user_ip in self.banned_ips:
            ban_until = self.banned_ips[user_ip]
            if datetime.now() < ban_until:
                return True
            else:
                # Remove expired ban
                del self.banned_ips[user_ip]
        return False
    
    def _log(self, source: str, message: str):
        """Add log entry to system state."""
        if self.system_state:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "message": message,
                "session_id": self.system_state.session.session_id
            }
            self.system_state.logs.append(log_entry)
        
        print(f"[{source}] {message}")
    
    def _get_current_phase(self) -> ConversationPhase:
        """Determine current conversation phase."""
        if not self.system_state or not self.system_state.session.conversation_turns:
            return ConversationPhase.INITIAL_CONTACT
        
        # Simple phase detection based on conversation progress
        turn_count = len(self.system_state.session.conversation_turns)
        active_tasks = len([t for t in self.system_state.tasks if t.status != TaskStatus.COMPLETED])
        
        if turn_count <= 2:
            return ConversationPhase.INITIAL_CONTACT
        elif self.system_state.session.goal_clarity_score < 0.5:
            return ConversationPhase.CLARIFICATION
        elif active_tasks == 0:
            return ConversationPhase.RESEARCH_PLANNING
        elif any(t.agent_type == AgentType.RESEARCH_AGENT for t in self.system_state.tasks):
            return ConversationPhase.RESEARCH_PLANNING
        elif any(t.agent_type == AgentType.ARCHITECT_PLANNER for t in self.system_state.tasks):
            return ConversationPhase.ARCHITECTURE_PLANNING
        elif active_tasks > 0:
            return ConversationPhase.EXECUTION_MONITORING
        else:
            return ConversationPhase.COMPLETION
    
    def _update_conversation_phase(self, decision: Dict[str, Any]):
        """Update conversation phase based on orchestration decision."""
        new_phase = decision.get("conversation_phase")
        if new_phase and self.system_state:
            # Update goal clarity score based on phase progression
            phase_scores = {
                "initial_contact": 0.1,
                "clarification": 0.3,
                "research_planning": 0.5,
                "architecture_planning": 0.7,
                "task_breakdown": 0.8,
                "execution_monitoring": 0.9,
                "completion": 1.0
            }
            self.system_state.session.goal_clarity_score = phase_scores.get(new_phase, 0.5)
    
    def _update_memory_layers(self, memory_updates: Dict[str, str]):
        """Update the 4-layer memory system."""
        if not self.system_state or not memory_updates:
            return
        
        timestamp = datetime.now()
        
        for layer_name, content in memory_updates.items():
            if not content:
                continue
                
            memory_entry = MemoryEntry(
                layer=MemoryLayer(layer_name),
                content=content,
                timestamp=timestamp,
                metadata={
                    "session_id": self.system_state.session.session_id,
                    "conversation_turn": len(self.system_state.session.conversation_turns)
                }
            )
            
            # Add to appropriate memory layer
            if layer_name == "knowledge_synthesis":
                self.system_state.knowledge_synthesis.append(memory_entry)
            elif layer_name == "build_conversation":
                self.system_state.build_conversation.append(memory_entry)
            elif layer_name == "codebase_knowledge":
                self.system_state.codebase_knowledge.append(memory_entry)
            elif layer_name == "stack_dependencies":
                self.system_state.stack_dependencies.append(memory_entry)
    
    def _create_and_assign_tasks(self, task_assignments: List[Dict[str, Any]]):
        """Create and assign tasks to the dependency graph."""
        if not self.system_state:
            return
        
        for task_data in task_assignments:
            enhanced_task = EnhancedTask(
                title=task_data.get("task_title", "Untitled Task"),
                description=task_data.get("task_description", ""),
                agent_type=AgentType(task_data.get("agent_type", "specialized_agent")),
                priority=task_data.get("priority", 3),
                estimated_duration=task_data.get("estimated_duration", 30),
                dependencies=task_data.get("dependencies", []),
                context={
                    "session_id": self.system_state.session.session_id,
                    "created_by": "ultra_orchestrator",
                    "requires_user_confirmation": task_data.get("requires_user_confirmation", True)
                }
            )
            
            # Add to system state and dependency graph
            self.system_state.tasks.append(enhanced_task)
            self.task_dependency_graph.add_task(enhanced_task)
            
            self._log("UltraOrchestrator", f"Created task: {enhanced_task.title} ({enhanced_task.agent_type})")
    
    def _create_specialized_agent(self, agent_spec: Dict[str, Any]):
        """Create a new specialized agent for edge cases."""
        # This would be implemented to dynamically create new agent types
        # For now, log the request for specialized agent creation
        self._log("UltraOrchestrator", f"Specialized agent creation requested: {agent_spec.get('agent_name', 'Unknown')}")
    
    def _execute_search_queries(self, search_queries: List[str]):
        """Execute search queries using available search services."""
        # This would integrate with Tavily/Exa search APIs
        self._log("UltraOrchestrator", f"Search requested for {len(search_queries)} queries")
    
    def _handle_user_warning(self, warning_data: Dict[str, Any]):
        """Handle user warnings for exploitation attempts."""
        warning_message = warning_data.get("message", "")
        severity = warning_data.get("severity", "low")
        
        if self.system_state:
            self.system_state.warning_log.append({
                "timestamp": datetime.now().isoformat(),
                "message": warning_message,
                "severity": severity,
                "session_id": self.system_state.session.session_id
            })
        
        self._log("UltraOrchestrator", f"User warning issued: {severity} - {warning_message}")
    
    def _update_system_metrics(self, decision: Dict[str, Any]):
        """Update system performance metrics."""
        if not self.system_state:
            return
        
        # Update decision metrics
        if "metrics" not in self.system_state.metrics:
            self.system_state.metrics["decisions"] = []
        
        self.system_state.metrics["decisions"].append({
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision.get("decision_type"),
            "confidence_score": decision.get("confidence_score", 0.5),
            "conversation_phase": decision.get("conversation_phase")
        })
    
    def get_next_executable_task(self) -> Optional[EnhancedTask]:
        """Get the next task that can be executed based on dependencies."""
        next_task_id = self.task_dependency_graph.get_next_executable_task()
        if next_task_id and self.system_state:
            for task in self.system_state.tasks:
                if task.id == next_task_id:
                    return task
        return None
    
    def reset_session(self, user_id: str) -> SystemState:
        """Reset the current session and start fresh."""
        self._log("UltraOrchestrator", f"Session reset requested for user {user_id}")
        
        # Clear session warnings for this user
        if self.system_state:
            session_id = self.system_state.session.session_id
            if session_id in self.session_warnings:
                del self.session_warnings[session_id]
        
        # Initialize new session
        return self.initialize_session(user_id) 