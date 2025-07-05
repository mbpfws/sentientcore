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

    def _get_system_prompt(self):
        return """
You are the UltraOrchestrator - an intelligent conversational guide that understands user intent and only executes research when there is sufficient information to proceed effectively.

**CORE PHILOSOPHY:**
- **Guide First, Execute Second:** Understand and clarify user needs before taking action
- **Intelligent Research Delegation:** Only call research agent when you have enough specific information to conduct meaningful research
- **Conversational Intelligence:** Help users articulate their needs clearly through guided questions
- **No Premature Execution:** Never execute research on vague or insufficient information

**YOUR CORE DIRECTIVES:**

1. **Assess Information Sufficiency:** Determine if you have enough specific information to conduct meaningful research
2. **Guide Through Clarification:** When information is insufficient, ask targeted questions to gather what's needed
3. **Execute When Ready:** Only delegate to research agent when you have clear, specific research objectives
4. **Explain Your Process:** Help users understand what information is needed and why
5. **Maintain Context:** Remember conversation history to build understanding progressively

**RESPONSE FORMAT:**
Your entire response MUST be a single, valid JSON object. Do not add any text before or after it.

```json
{
  "action_type": "conversation" | "research" | "clarification",
  "response": "<Your response or clarification request>",
  "language_detected": "<two-letter ISO 639-1 code>",
  "conversation_summary": "<Brief summary of conversation so far>",
  "research_query": "<If research, the specific query for the research agent>",
  "information_assessment": "<Assessment of whether sufficient information exists for research>"
}
```

**EXAMPLE SCENARIOS:**

*   **Vague Request:** User says: "can you help with building an app"
    ```json
    {
      "action_type": "clarification",
      "response": "I'd be happy to help you build an app! To provide the best guidance, I need to understand more about your project. Could you tell me: What type of app are you thinking of building? (web app, mobile app, desktop app) What's the main purpose or functionality you want it to have? Do you have any specific technologies or platforms in mind? What's your experience level with development?",
      "language_detected": "en",
      "conversation_summary": "User requested help with building an app. Gathering specific requirements before proceeding.",
      "information_assessment": "Insufficient - need app type, purpose, tech stack, and user experience level"
    }
    ```

*   **Sufficient Information for Research:** User says: "I am a software developer and I have trouble with writing clear documentation in English. Can you research some solutions for me?"
    ```json
    {
      "action_type": "research",
      "response": "I have enough information to conduct a targeted research on documentation writing solutions for software developers. I'll search for tools, techniques, and resources specifically for improving technical documentation writing skills, especially for non-native English speakers. Let me gather comprehensive solutions for you.",
      "language_detected": "en",
      "conversation_summary": "User requested research on solutions for improving technical documentation writing skills as a software developer. Sufficient context provided to proceed with research.",
      "research_query": "Solutions and tools for software developers to improve technical documentation writing skills, especially for non-native English speakers",
      "information_assessment": "Sufficient - clear problem (documentation writing), specific context (software developer), and target outcome (improvement solutions)"
    }
    ```

**IMPORTANT:** Only execute research when you have sufficient specific information. Guide users to provide clarity before taking action.
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
        """
        print("---ULTRA ORCHESTRATOR: PROCESSING STATE REQUEST---")
        
        # Construct the context for the LLM with conversation history
        conversation_history = "\n".join([f"{msg.sender}: {msg.content}" for msg in state.messages])
        latest_message = state.messages[-1].content if state.messages else ""
        context = f"Conversation History:\n{conversation_history}\n\nLatest User Message: {latest_message}\n\nNumber of turns: {len(state.messages)}"

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
                    
                    # Create research task and delegate to research agent
                    research_state = AppState(
                        messages=[Message(sender="user", content=research_query)],
                        logs=[],
                        image=state.image
                    )
                    
                    research_result = await self.research_agent.invoke(research_state)
                    
                    # Combine orchestrator response with research results
                    state.messages.append(Message(
                        sender="assistant", 
                        content=conversational_response
                    ))
                    
                    # Add research results to messages
                    if research_result.messages:
                        for msg in research_result.messages:
                            if msg.sender == "assistant":
                                state.messages.append(msg)
                    
                    # Merge logs
                    state.logs.extend(research_result.logs)
                    
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
            
            # Update conversation history with summary
            if not hasattr(state, 'conversation_history'):
                state.conversation_history = []
            state.conversation_history.append(conversation_summary)

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