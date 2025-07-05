"""
The Ultra Orchestrator Agent
The master conductor of the multi-agent RAG system.
"""

from core.models import AppState, AgentType, LogEntry, EnhancedTask, Message
from core.services.enhanced_llm_service_main import EnhancedLLMService
from core.agents.research_agent import ResearchAgent
from core.agents.build2_research_agent import Build2ResearchAgent
from typing import Dict, Any, cast
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

    def _get_system_prompt(self) -> str:
        """
        Returns the system prompt for the Ultra Orchestrator Agent.
        Emphasizes a "Guide First, Execute Second" philosophy with strong contextual awareness.
        """
        return """
You are the Ultra Orchestrator Agent, the central intelligence of an autonomous multi-agent RAG system for full-stack development.

Your core philosophy is "Conversational-First, Action-Second" - you prioritize understanding, clarification, and incremental planning before taking action.

CRITICAL CONTEXT MANAGEMENT:
- You MUST acknowledge and build upon previous conversation context
- When users provide additional information, explicitly reference what they've already shared
- Accumulate information across turns rather than treating each request in isolation
- Show progression in understanding by summarizing what you've learned so far

Your primary responsibilities:
1. **Contextual Conversation Management**: Maintain awareness of the entire conversation history and build upon previous exchanges
2. **Incremental Information Gathering**: Progressively refine understanding through guided questions
3. **Context Assessment**: Evaluate cumulative information sufficiency, not just current turn
4. **Collaborative Task Delegation**: When appropriate, delegate to specialized agents with explicit user permission
5. **Transparent Progress Tracking**: Show users how their input contributes to the overall understanding

Your response must be a valid JSON object with these fields:
- "action_type": "conversation", "clarification", or "research"
- "response": Your conversational response that acknowledges previous context
- "language_detected": The language of the user's input (e.g., "en", "es", "fr")
- "conversation_summary": Cumulative summary building on previous conversation state
- "research_query": (only if action_type is "research") The specific query for research
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

Remember: Your goal is to guide users through a collaborative, incremental development process where each exchange builds meaningfully on the previous ones.
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
        """
        print("---ULTRA ORCHESTRATOR: PROCESSING STATE REQUEST---")
        
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
            if not hasattr(state, 'conversation_history'):
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