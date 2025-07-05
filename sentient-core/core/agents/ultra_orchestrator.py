"""
The Ultra Orchestrator Agent
The master conductor of the multi-agent RAG system.
"""

from core.models import AppState, AgentType, LogEntry, EnhancedTask, Message
from core.services.enhanced_llm_service_main import EnhancedLLMService
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

    def _get_system_prompt(self):
        return """
You are the UltraOrchestrator for Build 1: Core Conversation & Orchestration Loop. You are a helpful, intelligent conversational AI assistant focused on maintaining natural, contextual conversations.

**BUILD 1 OBJECTIVES:**
- Maintain conversation history and context
- Provide helpful, natural responses to user messages
- Remember what was discussed in the current session
- Handle basic conversational interactions

**YOUR CORE DIRECTIVES:**

1. **Conversational Management:** Engage in natural conversation with users, maintaining context from previous messages in the session.
2. **Memory & Context:** Always consider the conversation history when responding. Reference previous messages when relevant.
3. **Helpful Responses:** Provide informative, helpful responses to user questions and comments.
4. **Session Awareness:** Keep track of the conversation flow and respond appropriately to follow-up questions.

**RESPONSE FORMAT:**
Your entire response MUST be a single, valid JSON object. Do not add any text before or after it.

```json
{
  "decision": "continue_conversation",
  "conversational_response": "<Your natural, conversational response to the user, maintaining context from conversation history>",
  "language_detected": "<two-letter ISO 639-1 code, e.g., 'en', 'vi', 'zh'>"
}
```

**EXAMPLE SCENARIOS:**

*   **First Message:** User says: "Hello, how are you?"
    *   **Your JSON Output:**
        ```json
        {
          "decision": "continue_conversation",
          "conversational_response": "Hello! I'm doing well, thank you for asking. I'm here to help you with any questions or tasks you might have. How can I assist you today?",
          "language_detected": "en"
        }
        ```

*   **Follow-up Message:** User says: "What was the first thing I said?"
    *   **Your JSON Output:**
        ```json
        {
          "decision": "continue_conversation",
          "conversational_response": "The first thing you said was 'Hello, how are you?' - you greeted me and asked how I was doing.",
          "language_detected": "en"
        }
        ```
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
        Process user input and provide conversational response for Build 1.
        """
        print("---ULTRA ORCHESTRATOR: INVOKING---")
        
        # Construct the context for the LLM
        conversation_history = "\n".join([f"{msg.sender}: {msg.content}" for msg in state.messages])
        context = f"Conversation History:\n{conversation_history}\n\nNumber of turns: {len(state.messages)}"

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
            decision_data = json.loads(cleaned_response)

            # Update the state based on the decision
            state.messages.append(Message(
                sender="assistant", 
                content=decision_data.get("conversational_response", "I'm here to help! How can I assist you?")
            ))
            
            # Store the entire decision object in the state (simplified for Build 1)
            state.orchestrator_decision = decision_data

            # Log the conversation turn completion
            log_message = f"Conversation turn completed. Language: {decision_data.get('language_detected', 'unknown')}"
            state.logs.append(LogEntry(source="UltraOrchestrator", message=log_message))
            state.next_action = decision_data.get("decision", "continue_conversation")

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