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
You are the UltraOrchestrator for Build 2: Introducing the First Specialized Agent & Tool Use. You are an intelligent orchestrator that can handle conversations and delegate research tasks to specialized agents.

**BUILD 2 OBJECTIVES:**
- Maintain conversation history and context
- Detect when user requests require research or external information
- Delegate research tasks to the Research Agent
- Provide helpful, natural responses to user messages
- Handle basic conversational interactions

**YOUR CORE DIRECTIVES:**

1. **Task Recognition:** Analyze user messages to determine if they require:
   - Simple conversation (continue_conversation)
   - Research/information gathering (delegate_research)

2. **Research Detection:** Look for keywords and patterns that indicate research needs:
   - "research", "find information", "what is", "tell me about", "look up"
   - Questions about facts, current events, comparisons
   - Requests for recommendations or lists
   - Any message prefixed with research mode indicators

3. **Conversational Management:** For non-research requests, engage naturally while maintaining context.

4. **Memory & Context:** Always consider conversation history when responding.

**RESPONSE FORMAT:**
Your entire response MUST be a single, valid JSON object. Do not add any text before or after it.

For RESEARCH requests:
```json
{
  "decision": "delegate_research",
  "research_query": "<Extract the core research question from user input>",
  "research_mode": "<knowledge|deep|best_in_class based on complexity>",
  "language_detected": "<two-letter ISO 639-1 code>"
}
```

For CONVERSATION requests:
```json
{
  "decision": "continue_conversation",
  "conversational_response": "<Your natural, conversational response>",
  "language_detected": "<two-letter ISO 639-1 code>"
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
        Process user input and provide conversational response or delegate to research agent for Build 2.
        """
        print("---ULTRA ORCHESTRATOR: INVOKING---")
        
        # Construct the context for the LLM
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
            decision_data = json.loads(cleaned_response)
            decision = decision_data.get("decision", "continue_conversation")

            # Store the decision in state
            state.orchestrator_decision = decision_data

            if decision == "delegate_research":
                # Extract research parameters
                research_query = decision_data.get("research_query", latest_message)
                research_mode = decision_data.get("research_mode", "knowledge")
                
                # Log research delegation
                state.logs.append(LogEntry(
                    source="UltraOrchestrator", 
                    message=f"Delegating research task: '{research_query}' (mode: {research_mode})"
                ))
                
                # Delegate to research agent
                try:
                    research_result = await self.research_agent.conduct_research(
                        query=research_query,
                        research_mode=research_mode
                    )
                    
                    # Add research result to messages
                    state.messages.append(Message(
                        sender="assistant",
                        content=research_result
                    ))
                    
                    state.logs.append(LogEntry(
                        source="UltraOrchestrator", 
                        message="Research task completed successfully"
                    ))
                    
                except Exception as research_error:
                    error_msg = f"Research failed: {str(research_error)}"
                    state.logs.append(LogEntry(source="UltraOrchestrator", message=error_msg))
                    state.messages.append(Message(
                        sender="assistant",
                        content="I encountered an issue while researching that topic. Could you please try rephrasing your request?"
                    ))
            
            else:  # continue_conversation
                # Handle as regular conversation
                conversational_response = decision_data.get(
                    "conversational_response", 
                    "I'm here to help! How can I assist you?"
                )
                state.messages.append(Message(
                    sender="assistant", 
                    content=conversational_response
                ))
                
                state.logs.append(LogEntry(
                    source="UltraOrchestrator", 
                    message="Conversation turn completed"
                ))

            # Set next action and language
            state.next_action = decision
            language = decision_data.get('language_detected', 'unknown')
            state.logs.append(LogEntry(
                source="UltraOrchestrator", 
                message=f"Language detected: {language}"
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