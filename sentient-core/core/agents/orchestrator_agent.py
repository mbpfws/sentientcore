from langgraph.graph import StateGraph, END
from core.models import Task, AppState, Message
from core.services.llm_service import LLMService, GroqModel
import json

class OrchestratorAgent:
    """
    The orchestrator agent that manages the overall workflow, plans tasks,
    and coordinates with other agents.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def invoke(self, state: AppState) -> AppState:
        """
        The entry point for the orchestrator agent.
        It's responsible for planning and responding to the user.
        """
        print("---ORCHESTRATOR AGENT---")

        # Dynamically select the model based on whether an image is present
        model: GroqModel = (
            "meta-llama/llama-4-scout-17b-16e-instruct"
            if state.image
            else "llama-3.3-70b-versatile"
        )
        print(f"Using model: {model}")

        # Add the user's prompt to the message history
        state.messages.append(Message(sender="user", content=state.user_prompt, image=state.image))

        system_prompt = """
You are a world-class AI orchestrator for a software development platform.
Your primary role is to communicate with the user, understand their request (which may include images),
and break it down into a series of actionable tasks for other specialized AI agents.

**RESPONSE LANGUAGE:**
First, detect the user's language. You MUST respond in the same language.

**IMAGE ANALYSIS:**
If the user provides an image, analyze it carefully. The user will likely ask to build an application
based on the visual content of the image. Your response should reflect that you have seen and
understood the image.

**CORE INSTRUCTIONS:**
1.  Engage the user in a helpful, conversational manner.
2.  If the user's request is vague, ask clarifying questions.
3.  If the request is clear, create a high-level plan of tasks.
4.  Each task must be assigned to a specific agent (e.g., 'research', 'design', 'code').
5.  You MUST ALWAYS return your response as a single, valid JSON object. Do NOT add any text
    before or after the JSON object.

**JSON OUTPUT FORMAT:**
Your entire output MUST be a JSON object with the following structure:
{
  "response": "<Your conversational response to the user, in their language>",
  "language": "<The detected language code, e.g., 'en', 'es', 'vi'>",
  "tasks": [
    {
      "description": "<A clear and concise description of the task>",
      "agent": "<The name of the agent assigned to this task>"
    }
  ]
}

- The "response" field is mandatory.
- The "language" field is mandatory.
- The "tasks" field is optional. Only include it if the user's request is clear enough to create a plan.

Example for a clear request in English:
{
  "response": "Great! I'll start by researching modern e-commerce architectures.",
  "language": "en",
  "tasks": [
    {
      "description": "Research modern e-commerce platform architectures and technology stacks.",
      "agent": "research"
    }
  ]
}

Example for a vague request in Vietnamese:
{
  "response": "Chào bạn, tôi có thể giúp gì cho bạn hôm nay?",
  "language": "vi",
  "tasks": []
}

Now, begin.
"""

        try:
            # Invoke the LLM service
            response_json_str = self.llm_service.invoke(
                system_prompt=system_prompt,
                user_prompt=state.user_prompt,
                image_bytes=state.image,
                model=model,
            )

            # Parse the JSON response
            response_data = json.loads(response_json_str)

            # Update state with the response and tasks
            state.messages.append(
                Message(sender="assistant", content=response_data["response"])
            )
            state.language = response_data.get("language", "en") # Update language
            
            if response_data.get("tasks"):
                new_tasks = [Task(**task_data) for task_data in response_data["tasks"]]
                state.tasks.extend(new_tasks)
                print(f"Added {len(new_tasks)} new tasks.")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error: Could not decode LLM response: {response_json_str}")
            state.messages.append(
                Message(
                    sender="assistant",
                    content="I'm having a little trouble thinking straight right now. Could you please rephrase that?",
                )
            )

        # Clear the user prompt and image to prevent reprocessing
        state.user_prompt = ""
        state.image = None

        print(
            f"Ending workflow. Current state: {len(state.messages)} messages, {len(state.tasks)} tasks."
        )
        return state
