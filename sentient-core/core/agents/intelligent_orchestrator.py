"""
Intelligent Orchestrator Agent for Multi-Agent RAG System
Implements sophisticated natural language understanding and workflow orchestration
"""

from core.models import AppState, Message, Task, TaskStatus, LogEntry
from core.services.llm_service import LLMService
import json
import re
import uuid
from typing import Dict, Any, List
from datetime import datetime

class IntelligentOrchestrator:
    """
    Intelligent Orchestrator that makes decisions based on natural language understanding.
    Manages conversation flow, task breakdown, and agent coordination.
    """
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.conversation_memory = []
        self.information_gathered = {}
        
    def analyze_and_orchestrate(self, state: AppState) -> Dict[str, Any]:
        """
        Main orchestration method that intelligently analyzes input and makes decisions.
        """
        # Track conversation history
        if state.messages:
            latest_message = state.messages[-1]
            self.conversation_memory.append(latest_message.content)
        
        # Use intelligent model for decision making
        primary_model = "gemini-2.5-flash" if state.image else "llama-3.3-70b-versatile"
        fallback_model = "gemini-2.5-flash"
        
        system_prompt = self._get_intelligent_system_prompt()
        conversation_context = self._build_conversation_context(state)
        
        # Get intelligent decision from LLM
        decision_response = self._get_intelligent_decision(
            system_prompt, conversation_context, primary_model, fallback_model, state.image
        )
        
        # Process the decision and update state
        return self._process_decision(decision_response, state)
    
    def _get_intelligent_system_prompt(self) -> str:
        """Returns the sophisticated system prompt for intelligent orchestration."""
        return """
You are an Intelligent Orchestrator Agent for a sophisticated multi-agent RAG system. You are the PRIMARY DECISION-MAKER and conductor of all workflows.

YOUR CORE INTELLIGENCE:

1. **NATURAL LANGUAGE UNDERSTANDING**: 
   - Analyze user input in ANY language (English, Vietnamese, Chinese, etc.)
   - Understand intent, context, and technical requirements from natural conversation
   - Detect language and respond appropriately in the same language

2. **CONVERSATION INTELLIGENCE**:
   - Determine conversation stage: initial_contact, gathering_details, clarifying_requirements, ready_for_execution
   - Identify when requests are: too_vague, off_topic, sufficiently_detailed, or ready_for_tasks
   - Engage in natural back-and-forth until you have comprehensive understanding

3. **WORKFLOW ORCHESTRATION**:
   - Make intelligent routing decisions: continue_conversation, request_clarification, redirect_off_topic, create_task_breakdown
   - Only create executable task lists when you have sufficient detail
   - Break down complex requests into logical, sequential tasks for specialized agents

4. **TASK INTELLIGENCE**:
   - Create task lists only when information is comprehensive enough for execution
   - Tasks must be in logical sequence with proper dependencies
   - Each task should be specific, actionable, and assigned to the right agent type

RESPOND IN JSON FORMAT:
{
    "decision": "continue_conversation|create_task_breakdown|redirect_conversation|request_clarification",
    "message": "Your natural response in user's language - be conversational and helpful",
    "conversation_stage": "initial_contact|gathering_details|clarifying_requirements|ready_for_execution|off_topic",
    "information_status": "insufficient|gathering|sufficient|comprehensive",
    "language_detected": "en|vi|zh|etc",
    "reasoning": "Brief explanation of your decision-making process",
    "follow_up_questions": ["Specific questions to gather missing details"],
    "task_breakdown": [
        {
            "id": "task_001", 
            "title": "Brief task title",
            "description": "Detailed executable description",
            "agent_type": "research|architecture|design|builder",
            "sequence": 1,
            "dependencies": [],
            "estimated_duration": "short|medium|long",
            "user_confirmation_required": true|false
        }
    ],
    "ready_for_execution": false|true
}

INTELLIGENCE RULES:
1. **NO RUSHED DECISIONS**: Don't create tasks until you truly understand what needs to be built
2. **NATURAL CONVERSATION**: Engage like a helpful expert, not a rigid system
3. **INTELLIGENT QUESTIONING**: Ask specific, targeted questions that help clarify requirements
4. **OFF-TOPIC DETECTION**: Politely redirect non-development conversations back to legitimate software/research needs
5. **MULTILINGUAL AWARENESS**: Detect language and maintain cultural context
6. **PROGRESSIVE INFORMATION GATHERING**: Build understanding through multiple conversation turns

EXAMPLES:

**Vague Input**: "I want to build something cool"
→ Decision: request_clarification
→ Message: "I'd love to help you build something amazing! To get started, could you tell me what area interests you? For example, are you thinking about a mobile app, web application, data analysis tool, or something else entirely?"

**Off-topic**: "What's the weather today?"
→ Decision: redirect_conversation  
→ Message: "I'm specialized in helping with software development and technical projects. Is there an application or system you'd like to build or research?"

**Sufficient Detail**: "I want to create a Vietnamese IELTS writing practice app with AI feedback"
→ Decision: create_task_breakdown
→ Creates comprehensive task sequence for research, architecture, design, and building phases

Your intelligence should shine through natural conversation and smart workflow orchestration.
"""

    def _build_conversation_context(self, state: AppState) -> str:
        """Builds comprehensive conversation context for intelligent analysis."""
        context_parts = []
        
        # Add conversation history
        if len(self.conversation_memory) > 1:
            context_parts.append("CONVERSATION HISTORY:")
            for i, msg in enumerate(self.conversation_memory[-5:], 1):  # Last 5 messages
                context_parts.append(f"{i}. {msg}")
        
        # Add current user input
        if state.messages:
            latest_message = state.messages[-1]
            context_parts.append(f"\nCURRENT USER INPUT: {latest_message.content}")
            
            # Add image context if present
            if latest_message.image or state.image:
                context_parts.append("NOTE: User has provided an image for visual context")
        
        # Add any gathered information
        if self.information_gathered:
            context_parts.append(f"\nINFORMATION GATHERED SO FAR: {json.dumps(self.information_gathered, indent=2)}")
        
        # Add existing tasks context
        if state.tasks:
            context_parts.append(f"\nEXISTING TASKS: {len(state.tasks)} tasks already created")
            for task in state.tasks[-3:]:  # Show last 3 tasks
                context_parts.append(f"- {task.description} (Status: {task.status})")
        
        return "\n".join(context_parts)
    
    def _get_intelligent_decision(self, system_prompt: str, context: str, primary_model: str, fallback_model: str, image: bytes = None) -> Dict[str, Any]:
        """Gets intelligent decision from LLM with fallback mechanism."""
        
        for model in [primary_model, fallback_model]:
            try:
                print(f"[IntelligentOrchestrator] Using model: {model}")
                
                full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context}\n\nProvide your intelligent analysis and decision:"
                
                response = self.llm_service.generate_response(
                    model_name=model,
                    prompt=full_prompt,
                    image_bytes=image,
                    stream=False
                )
                
                # Parse JSON response
                response_str = str(response)
                cleaned_response = self._clean_json_response(response_str)
                
                try:
                    decision_data = json.loads(cleaned_response)
                    print(f"[IntelligentOrchestrator] Successfully got decision using {model}")
                    return decision_data
                except json.JSONDecodeError as e:
                    print(f"[IntelligentOrchestrator] JSON decode error: {e}")
                    print(f"[IntelligentOrchestrator] Raw response: {response_str[:500]}...")
                    continue
                    
            except Exception as e:
                print(f"[IntelligentOrchestrator] Error with model {model}: {e}")
                if "rate_limit" in str(e).lower() and model == primary_model:
                    print(f"[IntelligentOrchestrator] Rate limit hit, trying fallback...")
                    continue
                else:
                    continue
        
        # Fallback response if all models fail
        return {
            "decision": "continue_conversation",
            "message": "I'm experiencing some technical difficulties. Could you please rephrase your request?",
            "conversation_stage": "gathering_details",
            "information_status": "insufficient",
            "language_detected": "en",
            "reasoning": "Model error fallback",
            "follow_up_questions": [],
            "task_breakdown": [],
            "ready_for_execution": False
        }
    
    def _clean_json_response(self, response: str) -> str:
        """Cleans LLM response to extract valid JSON."""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response.strip()
    
    def _process_decision(self, decision_data: Dict[str, Any], state: AppState) -> Dict[str, Any]:
        """Processes the intelligent decision and updates application state."""
        
        # Log the decision
        reasoning = decision_data.get("reasoning", "No reasoning provided")
        self._log(state, f"Decision: {decision_data.get('decision')} | {reasoning}")
        
        # Add assistant response to messages
        message = decision_data.get("message", "I'm processing your request...")
        state.messages.append(Message(sender="assistant", content=message))
        
        # Update information gathered
        if decision_data.get("information_status") in ["gathering", "sufficient", "comprehensive"]:
            # Extract key information from conversation
            self._extract_information(decision_data, state)
        
        # Handle different decision types
        decision = decision_data.get("decision", "continue_conversation")
        
        if decision == "create_task_breakdown" and decision_data.get("ready_for_execution"):
            # Create executable tasks
            self._create_executable_tasks(decision_data, state)
            
        elif decision == "request_clarification":
            # Add follow-up questions to help guide user
            follow_ups = decision_data.get("follow_up_questions", [])
            if follow_ups:
                self._log(state, f"Requesting clarification with {len(follow_ups)} follow-up questions")
        
        # Store decision data for UI
        result = state.model_dump()
        result["orchestrator_decision"] = decision_data
        result["next_action"] = decision
        
        return result
    
    def _extract_information(self, decision_data: Dict[str, Any], state: AppState):
        """Extracts and stores information from the conversation."""
        # Simple information extraction based on conversation
        if state.messages:
            latest_content = state.messages[-2].content if len(state.messages) > 1 else ""
            
            # Extract key information (this could be enhanced with NLP)
            if "app" in latest_content.lower():
                self.information_gathered["project_type"] = "mobile_app"
            elif "website" in latest_content.lower() or "web" in latest_content.lower():
                self.information_gathered["project_type"] = "web_application"
            elif "research" in latest_content.lower():
                self.information_gathered["project_type"] = "research_project"
            
            # Store language detected
            self.information_gathered["language"] = decision_data.get("language_detected", "en")
            
            # Store conversation stage
            self.information_gathered["stage"] = decision_data.get("conversation_stage", "initial_contact")
    
    def _create_executable_tasks(self, decision_data: Dict[str, Any], state: AppState):
        """Creates executable tasks based on intelligent breakdown."""
        task_breakdown = decision_data.get("task_breakdown", [])
        
        for task_data in task_breakdown:
            task = Task(
                id=task_data.get("id", str(uuid.uuid4())),
                description=f"{task_data.get('title', 'Task')}: {task_data.get('description', '')}",
                agent=task_data.get("agent_type", "research"),
                status=TaskStatus.PENDING
            )
            
            # Add sequence and dependency information
            task.sequence = task_data.get("sequence", 1)
            task.dependencies = task_data.get("dependencies", [])
            task.estimated_duration = task_data.get("estimated_duration", "medium")
            task.user_confirmation_required = task_data.get("user_confirmation_required", True)
            
            state.tasks.append(task)
        
        self._log(state, f"Created {len(task_breakdown)} executable tasks in sequence")
    
    def _log(self, state: AppState, message: str):
        """Adds a log entry to the state."""
        log_entry = LogEntry(
            source="IntelligentOrchestrator",
            message=message,
            timestamp=datetime.now().isoformat()
        )
        state.logs.append(log_entry)
        try:
            print(f"[IntelligentOrchestrator] {message}")
        except UnicodeEncodeError:
            # Fallback for Unicode encoding issues
            print(f"[IntelligentOrchestrator] {message.encode('ascii', 'ignore').decode('ascii')}")