from langgraph.graph import StateGraph, END
from core.models import Task, AppState, Message, LogEntry
from core.services.llm_service import LLMService
import json
import re
from typing import cast

class OrchestratorAgent:
    """
    The orchestrator agent that manages the overall workflow, plans tasks,
    and coordinates with other agents.
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def invoke(self, state: AppState) -> dict:
        """
        The entry point for the orchestrator agent. It's responsible for planning.
        This method is designed to be a "planning step" that can be called
        multiple times until a satisfactory plan is created.
        """
        print("---ORCHESTRATOR AGENT---")

        # Primary model selection: Use Gemini for multimodal, otherwise use Groq.
        primary_model = "gemini-2.5-flash" if state.image else "llama-3.3-70b-versatile"
        fallback_model = "gemini-2.5-flash"
        
        print(f"Primary model: {primary_model}, Fallback: {fallback_model}")

        system_prompt = """
You are a Senior AI Solutions Architect. Your goal is to help a user turn their idea into a fully-specified software project. You are multilingual and must communicate in the user's language.

**Your Process:**
1.  **Analyze the Request:** Carefully analyze the user's prompt and any provided images.
2.  **Clarify if Necessary:** If the request is vague (e.g., "make me a cool app"), you MUST ask specific, guiding questions to understand their goals. Do not just ask "what do you want?". Instead, suggest possibilities. For example: "That sounds interesting! To get started, could you tell me what the app's main purpose will be? For example, is it for e-commerce, social networking, education, or something else?"
3.  **Formulate a Plan:** Once you have a clear understanding, create a high-level project plan. This plan should consist of logical steps (tasks) assigned to specialized agents ('research' or 'design').
4.  **Converse and Plan:** Your response MUST be a single JSON object. This object contains your conversational reply and, if ready, the plan.

**JSON OUTPUT FORMAT:**
You MUST return a single JSON object. Do NOT add any text before or after it.

{
  "response": "<Your conversational response to the user, in their language. Ask clarifying questions here if needed.>",
  "plan": [
    {
      "description": "<A clear and concise description of the task for another agent>",
      "agent": "<'research' or 'design'>"
    }
  ]
}

**- `plan`:** If the user's request is still too vague to create a plan, return an empty list `[]`.
**- `response`:** This is your way of talking to the user. Use it to guide the conversation.

**Example: Vague Request**
User says: "I want to make an app for my business."
Your JSON output:
{
  "response": "I can certainly help with that! To start, could you tell me a bit about your business and what you'd like the app to do for your customers or employees?",
  "plan": []
}

**Example: Clearer Request**
User says: "tôi muốn tạo một app giúp dạy viết tiếng Anh IELTS" (I want to create an app to help teach IELTS English writing)
Your JSON output:
{
  "response": "That's a great idea! An IELTS writing assistant could be very helpful. I'll start by creating a plan. First, I'll have our research agent investigate the key features of successful IELTS preparation apps. Then, I'll have our design agent create some initial wireframes for the main user interface.",
  "plan": [
    {
      "description": "Investigate and analyze the top 3 existing IELTS writing assistant applications, focusing on their core features, user feedback, and monetization strategies.",
      "agent": "research"
    },
    {
        "description": "Create initial low-fidelity wireframes for the main user workflow: user registration, essay submission, and feedback view.",
        "agent": "design"
    }
  ]
}
"""
        response_json_str = ""
        model_used = None
        
        # Try primary model first, fallback to secondary if rate limited
        for model in [primary_model, fallback_model]:
            try:
                print(f"Attempting to use model: {model}")
                user_query = state.messages[-1].content if state.messages else ""
                full_prompt = f"{system_prompt}\n\nUser query: {user_query}"
                
                response = self.llm_service.generate_response(
                    model_name=model,
                    prompt=full_prompt,
                    image_bytes=state.image,
                    stream=False,
                )
                # Since stream=False, the response should be a string. Cast it for the linter.
                response_json_str = cast(str, response)
                model_used = model
                print(f"Successfully used model: {model}")
                break
            except Exception as e:
                error_msg = str(e)
                print(f"Error with model {model}: {error_msg}")
                
                # Check if it's a rate limit error
                if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                    print(f"Rate limit hit for {model}, trying fallback...")
                    if model == fallback_model:
                        # If fallback also fails, raise the error to be caught by outer try-catch
                        raise Exception(f"Both primary and fallback models failed. Last error: {str(e)}")
                    continue
                else:
                    # For non-rate-limit errors, try fallback once
                    if model == primary_model:
                        print(f"Non-rate-limit error with {model}, trying fallback...")
                        continue
                    else:
                        # If fallback also fails with non-rate-limit error, raise error
                        raise e
        
        try:

            # --- Final, robust helper to find and parse the last JSON block ---
            def extract_json_from_string(s: str) -> str:
                """Finds all JSON markdown blocks and returns the last one."""
                # Regex to find all ```json ... ``` blocks
                json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', s, re.DOTALL)
                if json_blocks:
                    # Return the last JSON block found
                    return json_blocks[-1]
                
                # Fallback for cases where there are no markdown fences
                match = re.search(r'(\{.*\})', s, re.DOTALL)
                if match:
                    return match.group(1)

                return s # Return original string if no JSON is found

            cleaned_json_str = extract_json_from_string(response_json_str)
            response_data = json.loads(cleaned_json_str)

            # The response is now the *only* thing that goes into the message history
            state.messages.append(
                Message(sender="assistant", content=response_data["response"])
            )
            
            new_tasks = []
            if response_data.get("plan"):
                new_tasks = [Task(**task_data) for task_data in response_data["plan"]]
                state.tasks.extend(new_tasks)
            
            log_msg = f"Added {len(new_tasks)} new tasks."
            state.logs.append(LogEntry(source="OrchestratorAgent", message=log_msg))
            print(log_msg)

        except (json.JSONDecodeError, KeyError) as e:
            # Enhanced error logging
            cleaned_str_for_log = "Could not clean."
            try:
                cleaned_str_for_log = extract_json_from_string(response_json_str)
            except:
                pass
            error_msg = f"Error: Could not decode LLM response from {model_used}. Raw: '{response_json_str}'. Cleaned attempt: '{cleaned_str_for_log}'"
            state.logs.append(LogEntry(source="OrchestratorAgent", message=error_msg))
            print(error_msg)
            state.messages.append(
                Message(
                    sender="assistant",
                    content="I'm having a little trouble thinking straight. Could you please rephrase?",
                )
            )
        except Exception as e:
            # Handle model failure errors
            error_msg = f"[OrchestratorAgent] Error: {str(e)}"
            state.logs.append(LogEntry(source="OrchestratorAgent", message=error_msg))
            print(error_msg)
            state.messages.append(
                Message(
                    sender="assistant",
                    content="I'm experiencing technical difficulties. Please try again in a moment.",
                )
            )

        # DO NOT clear the image here. The UI will manage the state.
        # state.image = None

        print(
            f"Ending workflow. Current state: {len(state.messages)} messages, {len(state.tasks)} tasks."
        )
        return state.model_dump() # Return the updated state as a dictionary
