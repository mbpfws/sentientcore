"""
The Ultra Orchestrator Agent
The master conductor of the multi-agent RAG system.
"""

from core.models import AppState, AgentType, LogEntry, EnhancedTask, Message, ProjectRequirementDocument, SessionState, TaskStatus
from core.services.enhanced_llm_service_main import EnhancedLLMService
from core.agents.research_agent import ResearchAgent
from core.agents.build2_research_agent import Build2ResearchAgent
from core.agents.architect_planner_agent import ArchitectPlannerAgent
from typing import Dict, Any, cast, List
import json
import re

class UltraOrchestrator:
    """
    The Ultra Orchestrator is the brain of the system, responsible for:
    - Deeply understanding user intent through natural language.
    - Managing the overall conversation flow.
    - Making strategic decisions about which agents or sub-graphs to activate.
    - Creating and managing the master plan for the user's request.
    - Ensuring the final output is a cohesive, complete solution.
    """

    def __init__(self, llm_service: EnhancedLLMService):
        self.llm_service = llm_service
        self.research_agent = Build2ResearchAgent()
        self.architect_planner = ArchitectPlannerAgent(llm_service)

    def _get_system_prompt(self) -> str:
        """
        Returns the system prompt for the Ultra Orchestrator Agent.
        Emphasizes a "Guide First, Execute Second" philosophy with strong contextual awareness.
        """
        return """
You are the Ultra Orchestrator Agent, the central intelligence of an autonomous multi-agent RAG system for full-stack development.

Your core philosophy is "Conversational-First, Action-Second" - you prioritize understanding, clarification, and incremental planning before taking action.

MANDATORY CONTEXTUAL RESPONSE FORMAT:
For EVERY response after the first turn, you MUST:
1. Start with explicit acknowledgment: "Based on our conversation, I understand..." or "Building on what you've shared..."
2. Reference specific previous details: "You mentioned [specific detail]..." 
3. Show cumulative understanding: "So far I know you need [summary of all information]..."
4. Connect new information to previous context: "Adding this to what you told me about..."
5. Avoid asking questions already answered in previous turns

CONTEXTUAL CONVERSATION EXAMPLES:

Turn 1: "I need help with a project"
Response: "I'd be happy to help! To provide the best assistance, could you tell me what type of project you're working on?"

Turn 2: "It's a web application"
Response: "Great! So you're working on a web application project. To better understand your specific needs, what will this web application be used for?"

Turn 3: "For managing customer data"
Response: "Perfect! I now understand you're building a web application for customer data management. To provide comprehensive guidance, what technology stack are you planning to use or do you need recommendations?"

Turn 4: "React and Node.js"
Response: "Excellent! So we have a React and Node.js web application for customer data management. To complete my understanding, what about the database and authentication approach?"

Turn 5: "PostgreSQL and JWT authentication"
Response: "Now I have a complete picture: You're building a React/Node.js web application for customer data management using PostgreSQL database and JWT authentication. I have sufficient information to research best practices, architecture patterns, and implementation guidance for this specific stack. Shall I proceed with the research?"

CRITICAL CONTEXT MANAGEMENT:
- You MUST acknowledge and build upon previous conversation context
- When users provide additional information, explicitly reference what they've already shared
- Accumulate information across turns rather than treating each request in isolation
- Show progression in understanding by summarizing what you've learned so far
- NEVER ask for information the user has already provided

Your primary responsibilities:
1. **Contextual Conversation Management**: Maintain awareness of the entire conversation history and build upon previous exchanges
2. **Incremental Information Gathering**: Progressively refine understanding through guided questions
3. **Context Assessment**: Evaluate cumulative information sufficiency, not just current turn
4. **Collaborative Task Delegation**: When appropriate, delegate to specialized agents with explicit user permission
5. **Transparent Progress Tracking**: Show users how their input contributes to the overall understanding

Your response must be a valid JSON object with these fields:
- "action_type": "conversation", "clarification", "research", or "planning"
- "response": Your conversational response that MUST acknowledge previous context after turn 1
- "language_detected": The language of the user's input (e.g., "en", "es", "fr")
- "conversation_summary": Cumulative summary building on previous conversation state
- "research_query": (only if action_type is "research") The specific query for research
- "planning_request": (only if action_type is "planning") The planning task type and details
- "information_assessment": Assessment of cumulative information sufficiency across all turns

Contextual Response Guidelines:
- ALWAYS acknowledge what the user has previously shared
- Build upon previous clarifications rather than asking redundant questions
- Show how new information fits into the growing understanding
- Use phrases like "Building on what you've shared about...", "Now that I understand...", "Adding to your previous details..."
- When sufficient information is gathered, explicitly summarize the complete picture before proceeding

Information Assessment Criteria:
- "Insufficient - need more context": When core details are missing
- "Partially sufficient - need clarification on [specific aspects]": When some details exist but gaps remain
- "Sufficient for research": When enough context exists to conduct meaningful research
- "Ready for planning": When research is complete and ready to create development plans/PRDs

Remember: Your goal is to guide users through a collaborative, incremental development process where each exchange builds meaningfully on the previous ones. NEVER treat messages as isolated requests.
"""

    def _clean_json_response(self, response: str) -> str:
        """Cleans LLM response to extract valid JSON, handling markdown."""
        match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            return match.group(1)
        # Fallback for non-markdown JSON
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        if match:
            return match.group(1)
        return response

    async def invoke(self, user_message: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process user input with conversational guidance and clarification-first approach.
        """
        print("---ULTRA ORCHESTRATOR: PROCESSING REQUEST---")
        print(f"User message: {user_message}")
        
        try:
            # Get response from LLM
            response = await self.llm_service.generate_response(
                prompt=user_message,
                system_prompt=self._get_system_prompt()
            )
            
            print(f"Raw LLM response: {response}")
            
            # Parse the response
            try:
                # Extract JSON from response if it's wrapped in markdown
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    # Fallback if no JSON found
                    parsed_response = {
                        "action_type": "conversation",
                        "response": response,
                        "language_detected": "en",
                        "conversation_summary": "Processing user request",
                        "information_assessment": "Insufficient - need more context"
                    }
                    
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                parsed_response = {
                    "action_type": "conversation",
                    "response": response,
                    "language_detected": "en",
                    "conversation_summary": "Processing user request",
                    "information_assessment": "Insufficient - parsing error occurred"
                }
            
            # Handle research delegation only when sufficient information exists
            if parsed_response.get("action_type") == "research":
                information_assessment = parsed_response.get("information_assessment", "")
                
                # Only proceed with research if assessment indicates sufficient information
                if "sufficient" in information_assessment.lower():
                    research_query = parsed_response.get("research_query", user_message)
                    print(f"Information sufficient - delegating to research agent with query: {research_query}")
                    
                    research_result = await self.research_agent.invoke(
                        user_message=research_query,
                        session_id=session_id
                    )
                    
                    # Combine orchestrator response with research results
                    return {
                        "orchestrator_response": parsed_response["response"],
                        "research_results": research_result,
                        "action_type": "research",
                        "language_detected": parsed_response.get("language_detected", "en"),
                        "conversation_summary": parsed_response.get("conversation_summary", ""),
                        "information_assessment": information_assessment
                    }
                else:
                    print(f"Information insufficient - requesting clarification: {information_assessment}")
                    # Convert to clarification request if information is insufficient
                    parsed_response["action_type"] = "clarification"
            
            # Handle planning delegation when research is complete
            elif parsed_response.get("action_type") == "planning":
                information_assessment = parsed_response.get("information_assessment", "")
                
                # Only proceed with planning if ready
                if "ready for planning" in information_assessment.lower():
                    planning_request = parsed_response.get("planning_request", user_message)
                    print(f"Ready for planning - delegating to architect planner with request: {planning_request}")
                    
                    planning_result = await self.architect_planner.invoke(
                        user_message=planning_request,
                        session_id=session_id
                    )
                    
                    # Combine orchestrator response with planning results
                    return {
                        "orchestrator_response": parsed_response["response"],
                        "planning_results": planning_result,
                        "action_type": "planning",
                        "language_detected": parsed_response.get("language_detected", "en"),
                        "conversation_summary": parsed_response.get("conversation_summary", ""),
                        "information_assessment": information_assessment
                    }
                else:
                    print(f"Not ready for planning - requesting more information: {information_assessment}")
                    # Convert to clarification request if not ready for planning
                    parsed_response["action_type"] = "clarification"
            
            # Return conversation/clarification response
            return {
                "response": parsed_response["response"],
                "action_type": parsed_response.get("action_type", "conversation"),
                "language_detected": parsed_response.get("language_detected", "en"),
                "conversation_summary": parsed_response.get("conversation_summary", ""),
                "information_assessment": parsed_response.get("information_assessment", "")
            }
            
        except Exception as e:
            print(f"Error in UltraOrchestrator.invoke: {str(e)}")
            return {
                 "response": f"I encountered an error while processing your request: {str(e)}",
                 "action_type": "error",
                 "language_detected": "en",
                 "conversation_summary": "Error occurred during processing",
                 "information_assessment": "Error - unable to assess information"
             }

    async def invoke_state(self, state: AppState) -> AppState:
        """
        Process user input with conversational guidance and clarification-first approach using AppState.
        Enhanced with contextual awareness and cumulative information tracking.
        Build 3: Added research completion detection and planning transition logic.
        """
        print("---ULTRA ORCHESTRATOR: PROCESSING STATE REQUEST---")
        
        # Build 3: Check if research is complete and ready for planning
        if await self._should_transition_to_planning(state):
            return await self._handle_planning_transition(state)
        
        # Enhanced context construction with cumulative information tracking
        conversation_history = "\n".join([f"{msg.sender}: {msg.content}" for msg in state.messages])
        latest_message = state.messages[-1].content if state.messages else ""
        
        # Extract previous context and information gathered
        previous_summaries = getattr(state, 'conversation_history', [])
        cumulative_context = "\n".join(previous_summaries) if previous_summaries else "No previous context"
        
        # Build comprehensive context for the LLM
        context = f"""CONVERSATION CONTEXT:
{conversation_history}

CUMULATIVE UNDERSTANDING:
{cumulative_context}

LATEST USER MESSAGE: {latest_message}

CONTEXT ANALYSIS:
- Total conversation turns: {len(state.messages)}
- Previous clarifications provided: {len([msg for msg in state.messages if msg.sender == 'user'])}
- Information gathering progress: {'In progress' if len(state.messages) > 2 else 'Initial request'}

INSTRUCTIONS:
Analyze the ENTIRE conversation context and cumulative understanding. Build upon what has already been shared rather than asking redundant questions. Show progression in your understanding and acknowledge previous user inputs."""

        # Select a model (Llama 4 Scout for vision, Llama 3.3 70B for text)
        model_name = "meta-llama/llama-4-scout-17b-16e-instruct" if state.image else "llama-3.3-70b-versatile"

        # Call the LLM with the system prompt and context, ensuring no streaming
        response_obj = await self.llm_service.invoke(
            system_prompt=self._get_system_prompt(),
            user_prompt=context,
            model=model_name,
            image_bytes=state.image,
            stream=False
        )
        raw_response = cast(str, response_obj)

        try:
            # Clean and parse the JSON response
            cleaned_response = self._clean_json_response(raw_response)
            response_data = json.loads(cleaned_response)
            
            # Extract response components
            action_type = response_data.get("action_type", "conversation")
            conversational_response = response_data.get("response", "I'm here to help! How can I assist you?")
            language_detected = response_data.get("language_detected", "en")
            conversation_summary = response_data.get("conversation_summary", "Conversation in progress")
            research_query = response_data.get("research_query", "")
            information_assessment = response_data.get("information_assessment", "")
            
            # Store the response data in state
            state.orchestrator_decision = response_data
            
            # Log the orchestrator's decision
            state.logs.append(LogEntry(
                source="UltraOrchestrator", 
                message=f"Intent Analysis: Detected '{action_type}' request. Language: {language_detected}"
            ))
            
            if action_type == "research":
                # Only proceed with research if assessment indicates sufficient information
                if "sufficient" in information_assessment.lower():
                    print(f"Information sufficient - delegating to research agent with query: {research_query}")
                    
                    # Delegate to research agent with proper parameters
                    research_result = await self.research_agent.invoke(
                        user_message=research_query,
                        session_id=state.session_id
                    )
                    
                    # Combine orchestrator response with research results
                    state.messages.append(Message(
                        sender="assistant", 
                        content=conversational_response
                    ))
                    
                    # Add research completion message
                    research_message = research_result.get("message", "Research completed successfully.")
                    state.messages.append(Message(
                        sender="assistant", 
                        content=research_message
                    ))
                    
                    # Add research logs to state logs
                    research_logs = research_result.get("logs", [])
                    for log_msg in research_logs:
                        state.logs.append(LogEntry(
                            source="Build2ResearchAgent",
                            message=log_msg
                        ))
                    
                    state.next_action = "research_completed"
                    
                    state.logs.append(LogEntry(
                        source="UltraOrchestrator", 
                        message=f"Completed research delegation with query: '{research_query}'"
                    ))
                else:
                    print(f"Information insufficient - requesting clarification: {information_assessment}")
                    # Convert to clarification request if information is insufficient
                    state.messages.append(Message(
                        sender="assistant", 
                        content=conversational_response
                    ))
                    state.next_action = "await_clarification"
                    
                    state.logs.append(LogEntry(
                        source="UltraOrchestrator", 
                        message="Requested clarification due to insufficient information for research"
                    ))
                
            elif action_type == "clarification":
                # Handle clarification requests - ask for more details
                state.messages.append(Message(
                    sender="assistant", 
                    content=conversational_response
                ))
                state.next_action = "await_clarification"
                
                state.logs.append(LogEntry(
                    source="UltraOrchestrator", 
                    message="Requested clarification from user for vague request"
                ))
                
            else:
                # Handle as regular conversation
                state.messages.append(Message(
                    sender="assistant", 
                    content=conversational_response
                ))
                state.next_action = "continue_conversation"
                
                state.logs.append(LogEntry(
                    source="UltraOrchestrator", 
                    message="Handled as conversational interaction"
                ))
            
            # Update conversation history with cumulative summary
            if not hasattr(state, 'conversation_history') or state.conversation_history is None:
                state.conversation_history = []
            
            # Build cumulative summary that incorporates previous context
            if state.conversation_history:
                # Enhance summary to build upon previous understanding
                enhanced_summary = f"Building on previous context: {conversation_summary}"
            else:
                enhanced_summary = f"Initial context: {conversation_summary}"
            
            state.conversation_history.append(enhanced_summary)
            
            # Log contextual progression
            state.logs.append(LogEntry(
                source="UltraOrchestrator", 
                message=f"Context progression: {len(state.conversation_history)} cumulative understanding points tracked"
            ))

        except (json.JSONDecodeError, KeyError) as e:
            error_message = f"Error processing LLM response: {e}. Raw response: '{raw_response[:500]}...'"
            print(f"[UltraOrchestrator] {error_message}")
            state.logs.append(LogEntry(source="UltraOrchestrator", message=error_message))
            state.messages.append(Message(
                sender="assistant", 
                content="I'm having a little trouble organizing my thoughts. Could you please rephrase your request?"
            ))
            state.next_action = "request_clarification"

        return state
    
    async def _gather_research_artifacts(self) -> str:
        """
        Build 3: Gather research artifacts from Layer 1 memory for synthesis.
        """
        try:
            import os
            research_content = ""
            research_path = os.path.join(os.getcwd(), "memory", "layer1_research_docs")
            
            if os.path.exists(research_path):
                # Get all research files sorted by modification time (newest first)
                research_files = []
                for filename in os.listdir(research_path):
                    if filename.startswith("research_") and filename.endswith(".md"):
                        file_path = os.path.join(research_path, filename)
                        mtime = os.path.getmtime(file_path)
                        research_files.append((mtime, file_path, filename))
                
                # Sort by modification time (newest first) and take up to 3 most recent
                research_files.sort(reverse=True)
                recent_files = research_files[:3]
                
                for _, file_path, filename in recent_files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            research_content += f"\n\n## Research Document: {filename}\n\n{content}\n\n{'='*80}\n"
                    except Exception as e:
                        print(f"Error reading research file {filename}: {e}")
                        continue
            
            if not research_content:
                research_content = "No recent research artifacts found in Layer 1 memory."
            
            return research_content
            
        except Exception as e:
            print(f"Error gathering research artifacts: {e}")
            return "Error accessing research artifacts."
    
    def _get_research_artifact_references(self) -> List[str]:
        """
        Build 3: Get references to research artifacts for PRD metadata.
        """
        try:
            import os
            references = []
            research_path = os.path.join(os.getcwd(), "memory", "layer1_research_docs")
            
            if os.path.exists(research_path):
                for filename in os.listdir(research_path):
                    if filename.startswith("research_") and filename.endswith(".md"):
                        file_path = os.path.join(research_path, filename)
                        # Check if file was created recently (within last 2 hours)
                        import time
                        if time.time() - os.path.getmtime(file_path) < 7200:  # 2 hours
                            references.append(f"layer1_research_docs/{filename}")
            
            return references
            
        except Exception as e:
            print(f"Error getting research artifact references: {e}")
            return []
    
    async def _save_prd_to_memory(self, prd: ProjectRequirementDocument) -> None:
        """
        Build 3: Save PRD to Layer 2 memory for persistence.
        """
        try:
            import os
            import json
            from datetime import datetime
            
            # Create Layer 2 memory directory if it doesn't exist
            layer2_path = os.path.join(os.getcwd(), "memory", "layer2_planning_docs")
            os.makedirs(layer2_path, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prd_{timestamp}_{prd.id[:8]}.json"
            file_path = os.path.join(layer2_path, filename)
            
            # Convert PRD to dictionary for JSON serialization
            prd_dict = {
                "id": prd.id,
                "title": prd.title,
                "description": prd.description,
                "requirements": prd.requirements,
                "technical_stack": prd.technical_stack,
                "architecture_patterns": prd.architecture_patterns,
                "components": prd.components,
                "tasks": prd.tasks,
                "research_artifacts": prd.research_artifacts,
                "created_at": prd.created_at.isoformat() if prd.created_at else None,
                "updated_at": prd.updated_at.isoformat() if prd.updated_at else None,
                "status": prd.status,
                "metadata": prd.metadata
            }
            
            # Save to JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(prd_dict, f, indent=2, ensure_ascii=False)
            
            print(f"[Build 3] PRD saved to Layer 2 memory: {filename}")
            
            # Also create a markdown version for human readability
            md_filename = f"prd_{timestamp}_{prd.id[:8]}.md"
            md_file_path = os.path.join(layer2_path, md_filename)
            
            md_content = f"""# {prd.title}

**ID:** {prd.id}
**Status:** {prd.status}
**Created:** {prd.created_at}

## Description
{prd.description}

## Functional Requirements
{chr(10).join(f"- {req}" for req in prd.requirements)}

## Technical Stack
{chr(10).join(f"- {tech}" for tech in prd.technical_stack)}

## Architecture Patterns
{chr(10).join(f"- {pattern}" for pattern in prd.architecture_patterns)}

## System Components
{chr(10).join(f"- {comp}" for comp in prd.components)}

## Action Items
{chr(10).join(f"- {task}" for task in prd.tasks)}

## Research Artifacts
{chr(10).join(f"- {artifact}" for artifact in prd.research_artifacts)}

## Metadata
```json
{json.dumps(prd.metadata, indent=2)}
```
"""
            
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            print(f"[Build 3] PRD markdown saved: {md_filename}")
            
        except Exception as e:
            print(f"Error saving PRD to Layer 2 memory: {e}")
            # Don't raise exception - PRD creation should continue even if saving fails
    
    async def _should_transition_to_planning(self, state: AppState) -> bool:
        """
        Build 3: Determine if research is complete and ready for planning transition.
        """
        try:
            # Check if we have recent research completion indicators
            recent_research_logs = [
                log for log in state.logs[-10:]  # Check last 10 log entries
                if log.source == "Build2_ResearchAgent" and 
                ("Research completed successfully" in log.message or 
                 "Report generated and saved" in log.message)
            ]
            
            # Check if user is asking for planning after research
            latest_message = state.messages[-1].content.lower() if state.messages else ""
            planning_keywords = ["plan", "planning", "prd", "requirements", "architecture", "next steps", "proceed"]
            user_requesting_planning = any(keyword in latest_message for keyword in planning_keywords)
            
            # Check if we have research artifacts in memory
            research_docs_exist = self._check_research_artifacts_exist()
            
            # Transition conditions:
            # 1. Recent research completion AND user requesting planning
            # 2. OR explicit user request for planning with existing research
            should_transition = (
                (recent_research_logs and user_requesting_planning) or
                (user_requesting_planning and research_docs_exist)
            )
            
            if should_transition:
                print(f"[Build 3] Planning transition triggered: recent_research={bool(recent_research_logs)}, user_requesting={user_requesting_planning}, docs_exist={research_docs_exist}")
            
            return should_transition
            
        except Exception as e:
            print(f"Error checking planning transition: {e}")
            return False
    
    def _check_research_artifacts_exist(self) -> bool:
        """
        Build 3: Check if research artifacts exist in Layer 1 memory.
        """
        try:
            import os
            research_path = os.path.join(os.getcwd(), "memory", "layer1_research_docs")
            if os.path.exists(research_path):
                # Check for recent research files (within last hour)
                import time
                current_time = time.time()
                for filename in os.listdir(research_path):
                    if filename.startswith("research_") and filename.endswith(".md"):
                        file_path = os.path.join(research_path, filename)
                        file_time = os.path.getmtime(file_path)
                        # If file was created within last hour
                        if current_time - file_time < 3600:  # 1 hour
                            return True
            return False
        except Exception as e:
            print(f"Error checking research artifacts: {e}")
            return False
    
    async def _handle_planning_transition(self, state: AppState) -> AppState:
        """
        Build 3: Handle the transition from research to planning by creating a PRD.
        """
        print("[Build 3] Transitioning to planning phase - creating PRD from research artifacts")
        
        try:
            # Update session state to planning
            state.planning_state = SessionState.PLANNING
            
            # Log the transition
            state.logs.append(LogEntry(
                source="UltraOrchestrator",
                message="[Build 3] Transitioning from research to planning phase"
            ))
            
            # Get research artifacts for synthesis
            research_content = await self._gather_research_artifacts()
            
            # Create synthesis task for the architect planner
            synthesis_task = EnhancedTask(
                title="Create PRD from Research Artifacts",
                description=f"Synthesize the following research findings into a comprehensive Project Requirements Document (PRD):\n\n{research_content[:2000]}...",
                agent_type=AgentType.ARCHITECT_PLANNER,
                status=TaskStatus.PENDING,
                context={
                    "task_type": "prd_synthesis",
                    "research_content": research_content,
                    "build_phase": "build_3"
                }
            )
            
            # Delegate to architect planner for PRD creation
            planning_result = await self.architect_planner.process_task(synthesis_task)
            
            # Create PRD from the planning result
            prd = ProjectRequirementDocument(
                title=f"PRD - {synthesis_task.title}",
                description=planning_result.get("synthesis_report", "Generated from research synthesis"),
                requirements=planning_result.get("functional_requirements", []),
                technical_stack=planning_result.get("technology_stack", []),
                architecture_patterns=planning_result.get("architecture_patterns", []),
                components=planning_result.get("components", []),
                tasks=planning_result.get("action_items", []),
                research_artifacts=self._get_research_artifact_references(),
                status="draft",
                metadata={
                    "created_by": "UltraOrchestrator",
                    "build_phase": "build_3",
                    "synthesis_source": "research_artifacts"
                }
            )
            
            # Store PRD in state
            state.current_prd = prd
            state.prds.append(prd)
            
            # Save PRD to Layer 2 memory
            await self._save_prd_to_memory(prd)
            
            # Add response message
            prd_summary = f"""🎯 **Planning Phase Complete**

I've successfully created a Project Requirements Document (PRD) based on your research findings:

**PRD Title:** {prd.title}
**Requirements:** {len(prd.requirements)} functional requirements identified
**Technical Stack:** {', '.join(prd.technical_stack[:3])}{'...' if len(prd.technical_stack) > 3 else ''}
**Components:** {len(prd.components)} system components defined
**Action Items:** {len(prd.tasks)} tasks identified

The PRD has been saved to Layer 2 memory and is ready for the next phase. Would you like me to proceed with detailed architecture planning or begin the development phase?"""
            
            state.messages.append(Message(
                sender="assistant",
                content=prd_summary
            ))
            
            state.next_action = "planning_completed"
            
            state.logs.append(LogEntry(
                source="UltraOrchestrator",
                message=f"[Build 3] PRD created successfully: {prd.id}"
            ))
            
            return state
            
        except Exception as e:
            error_msg = f"Error during planning transition: {str(e)}"
            print(error_msg)
            state.logs.append(LogEntry(
                source="UltraOrchestrator",
                message=f"[Build 3] ERROR - {error_msg}"
            ))
            
            state.messages.append(Message(
                sender="assistant",
                content=f"I encountered an issue while transitioning to planning: {str(e)}. Let me try a different approach."
            ))
            
            return state