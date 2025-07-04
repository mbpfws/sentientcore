"""
The Ultra Orchestrator Agent
The master conductor of the multi-agent RAG system.
"""

from core.models import AppState, AgentType, LogEntry, EnhancedTask, Message
from core.services.llm_service import EnhancedLLMService
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
You are the UltraOrchestrator, the master conductor of a multi-agent AI development platform. You are deeply intelligent, multilingual, and an expert in software architecture and project management. Your primary goal is to collaborate with a user to transform their vague ideas into fully-realized, actionable development plans.

**YOUR CORE DIRECTIVES:**

1.  **Analyze & Understand:** Scrutinize the user's prompt (in any language) and any provided images. Your first step is ALWAYS to understand, not to act.
2.  **Converse & Clarify:**
    *   If a request is vague (e.g., "make an app"), you MUST ask targeted, guiding questions. Do NOT just ask "what do you want?". Suggest possibilities: "That sounds like a great project. To get started, could you tell me its primary purpose? For instance, is it for e-commerce, social media, productivity, or something else?"
    *   If a request is off-topic (e.g., "what's the weather?"), you MUST politely redirect the user back to software development or research tasks.
3.  **Manage Session State:**
    *   You must track the conversation. If a conversation is unproductive after 7 turns, issue a warning.
    *   Your primary job is to gather enough information to create a plan.
4.  **Decide & Delegate:** Based on your analysis, you will make a single, critical decision on the next action to take.

**JSON OUTPUT FORMAT:**
Your entire response MUST be a single, valid JSON object. Do not add any text before or after it.

```json
{
  "decision": "request_clarification | create_plan | redirect_off_topic | issue_warning",
  "conversational_response": "<Your natural, conversational response to the user, in their language. This is where you ask questions or provide updates.>",
  "language_detected": "<two-letter ISO 639-1 code, e.g., 'en', 'vi', 'zh'>",
  "information_status": "insufficient | gathering | sufficient",
  "plan": [
    {
      "description": "<A clear and concise task description for another agent>",
      "agent_type": "<'RESEARCH_AGENT' or 'ARCHITECT_PLANNER' or 'FRONTEND_DEVELOPER' etc.>",
      "sequence": 1,
      "dependencies": []
    }
  ]
}
```

**FIELD DEFINITIONS & RULES:**

*   `decision`: Your single most important output.
    *   `request_clarification`: Use this when the user's request is vague and missing core details. `conversational_response` MUST contain clarifying questions. The `plan` array MUST be empty.
    *   `create_plan`: **Use this ONLY when `information_status` is 'sufficient'.**
    *   `redirect_off_topic`: Use this for non-development related queries. The `plan` array MUST be empty.
    *   `issue_warning`: Use this if the conversation is unproductive. The `plan` array MUST be empty.
*   `conversational_response`: Your message to the user.
*   `language_detected`: The detected language of the user's prompt.
*   `information_status`: Your assessment of the current state of knowledge.
    *   **A request is 'sufficient' if it describes the application type (e.g., web app, mobile app), its core purpose (e.g., fitness tracking, e-commerce), AND at least two key features (e.g., user accounts, dashboards, payment processing). If these conditions are met, you MUST set this to 'sufficient' and `decision` to 'create_plan'.**
*   `plan`: A list of tasks for other agents. This MUST be empty unless your `decision` is `create_plan`.

**EXAMPLE SCENARIOS:**

*   **Vague Request:** User says: "I need help with my project."
    *   **Your JSON Output:**
        ```json
        {
          "decision": "request_clarification",
          "conversational_response": "I can certainly help with your project. To get started, could you describe what the project is about and what you'd like to achieve?",
          "language_detected": "en",
          "information_status": "insufficient",
          "plan": []
        }
        ```

*   **Sufficient Request:** User says: "I need to build a web app for tracking personal fitness goals. It should have user accounts, a dashboard to log workouts, and progress charts."
    *   **Your JSON Output:**
        ```json
        {
          "decision": "create_plan",
          "conversational_response": "That's a fantastic and well-defined project! I'll create an initial plan. First, our Research Agent will investigate existing fitness tracking apps for best-in-class features. Then, our Architect Planner will design the database schema and API structure.",
          "language_detected": "en",
          "information_status": "sufficient",
          "plan": [
            {
              "description": "Investigate and analyze the top 5 fitness tracking web apps, focusing on UI/UX, core features, and technology stacks.",
              "agent_type": "RESEARCH_AGENT",
              "sequence": 1,
              "dependencies": []
            },
            {
              "description": "Design the database schema for user accounts, workout logs, and goals. Define the primary API endpoints.",
              "agent_type": "ARCHITECT_PLANNER",
              "sequence": 2,
              "dependencies": ["<task_id_of_sequence_1>"]
            }
          ]
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

    def invoke(self, state: AppState) -> Dict[str, Any]:
        """
        The main entry point for the Ultra Orchestrator's decision-making process.
        """
        print("---ULTRA ORCHESTRATOR: INVOKING---")
        
        # Construct the context for the LLM
        conversation_history = "\n".join([f"{msg.sender}: {msg.content}" for msg in state.messages])
        context = f"Conversation History:\n{conversation_history}\n\nNumber of turns: {len(state.messages)}"

        # Select a model (Gemini for vision, Llama for text)
        model_name = "gemini-2.5-flash" if state.image else "llama-3.3-70b-versatile"

        # Call the LLM with the system prompt and context, ensuring no streaming
        response_obj = self.llm_service.invoke(
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
            state.messages.append(Message(sender="assistant", content=decision_data["conversational_response"]))
            
            # Store the entire decision object in the state for other agents to use
            state.orchestrator_decision = decision_data

            if decision_data.get("decision") == "create_plan" and decision_data.get("plan"):
                # Convert agent_type to lowercase to match the Enum
                for task_data in decision_data["plan"]:
                    if 'agent_type' in task_data:
                        task_data['agent_type'] = task_data['agent_type'].lower()
                
                new_tasks = [EnhancedTask(**task_data) for task_data in decision_data["plan"]]
                state.tasks.extend(new_tasks)
                log_message = f"Created a new plan with {len(new_tasks)} tasks."
            else:
                log_message = f"Decision: '{decision_data.get('decision')}'. No new tasks created."

            state.logs.append(LogEntry(source="UltraOrchestrator", message=log_message))
            state.next_action = decision_data.get("decision")

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