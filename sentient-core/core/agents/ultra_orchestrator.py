"""
The Ultra Orchestrator Agent
The master conductor of the multi-agent RAG system.
"""

from core.models import AppState, AgentType, LogEntry, EnhancedTask, Message
from core.services.enhanced_llm_service_main import EnhancedLLMService
from core.agents.research_agent import ResearchAgent
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
        self.research_agent = ResearchAgent(llm_service)

    def _get_system_prompt(self):
        return """
You are the UltraOrchestrator for Build 1: The Core Conversation & Orchestration Loop. You are the foundational conversational agent that establishes persistent, stateful conversation management.

**BUILD 1 OBJECTIVES:**
- Create a persistent, stateful conversational agent
- Handle user input and maintain conversation history
- Provide contextual responses based on conversation history
- Validate the foundational Orchestrator Graph and state management

**YOUR CORE DIRECTIVES:**

1. **Conversation Management:** Receive input and use LLM to generate contextual responses
2. **Memory & Context:** Always consider the full conversation history when responding
3. **State Persistence:** Ensure conversation history is maintained across interactions
4. **Natural Interaction:** Engage in helpful, natural conversation while remembering context

**RESPONSE FORMAT:**
Your entire response MUST be a single, valid JSON object. Do not add any text before or after it.

```json
{
  "response": "<Your natural, conversational response based on context>",
  "language_detected": "<two-letter ISO 639-1 code>",
  "conversation_summary": "<Brief summary of conversation so far>"
}
```

**EXAMPLE SCENARIOS:**

*   **First Message:** User says: "Hello, how are you?"
    *   **Your JSON Output:**
        ```json
        {
          "response": "Hello! I'm doing well, thank you for asking. I'm here to help you with any questions or tasks you might have. How can I assist you today?",
          "language_detected": "en",
          "conversation_summary": "User greeted me and asked how I'm doing. I responded positively and offered assistance."
        }
        ```

*   **Follow-up Message:** User says: "What was the first thing I said?"
    *   **Your JSON Output:**
        ```json
        {
          "response": "The first thing you said was 'Hello, how are you?' - you greeted me and asked how I was doing.",
          "language_detected": "en",
          "conversation_summary": "User initially greeted me, then asked about their first message. I provided the accurate recall of their greeting."
        }
        ```

**IMPORTANT:** Always reference the conversation history to provide contextual, relevant responses. Remember previous interactions and build upon them naturally.
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

    async def invoke(self, state: AppState) -> AppState:
        """
        Build 1: Process user input and provide conversational response with persistent state management.
        """
        print("---BUILD 1 ULTRA ORCHESTRATOR: INVOKING---")
        
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
            # Clean and parse the JSON response for Build 1 format
            cleaned_response = self._clean_json_response(raw_response)
            response_data = json.loads(cleaned_response)
            
            # Extract Build 1 response components
            conversational_response = response_data.get("response", "I'm here to help! How can I assist you?")
            language_detected = response_data.get("language_detected", "en")
            conversation_summary = response_data.get("conversation_summary", "Conversation in progress")
            
            # Store the response data in state
            state.orchestrator_decision = response_data
            
            # Add the conversational response to messages
            state.messages.append(Message(
                sender="assistant", 
                content=conversational_response
            ))
            
            # Update conversation history with summary
            if not hasattr(state, 'conversation_history'):
                state.conversation_history = []
            state.conversation_history.append(conversation_summary)
            
            # Log Build 1 conversation processing
            state.logs.append(LogEntry(
                source="Build1_UltraOrchestrator", 
                message=f"Build 1: Conversation turn completed. Language: {language_detected}, Summary: {conversation_summary[:100]}..."
            ))
            
            # Set next action for Build 1 (always continue conversation)
            state.next_action = "continue_conversation"

        except (json.JSONDecodeError, KeyError) as e:
            error_message = f"Build 1 Error processing LLM response: {e}. Raw response: '{raw_response[:500]}...'"
            print(f"[Build1_UltraOrchestrator] {error_message}")
            state.logs.append(LogEntry(source="Build1_UltraOrchestrator", message=error_message))
            state.messages.append(Message(
                sender="assistant", 
                content="I'm having a little trouble organizing my thoughts. Could you please rephrase your request?"
            ))
            state.next_action = "request_clarification"

        return state