from groq import Groq
from core.config import settings
from typing import Literal, List, Dict, Any
import base64

# Define specific model types for clarity and type safety
GroqModel = Literal[
    "llama-3.3-70b-versatile", 
    "compound-beta", 
    "meta-llama/llama-4-scout-17b-16e-instruct"
]

class LLMService:
    """
    A service to interact with Groq's Large Language Models (LLMs).
    Provides a unified interface for invoking different models, including multimodal vision models.
    """

    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        image_bytes: bytes | None = None,
        model: GroqModel = "llama-3.3-70b-versatile",
    ) -> str:
        """
        Invokes the specified Groq LLM with a system prompt, user prompt, and optional image.
        """
        # The groq library expects a list of dictionaries.
        # We use List[Dict[str, Any]] for type hinting flexibility.
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        
        user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            # Automatically switch to the vision-capable model if an image is provided
            model = "meta-llama/llama-4-scout-17b-16e-instruct"

        messages.append({"role": "user", "content": user_content})

        chat_completion = self.groq_client.chat.completions.create(
            messages=messages, # type: ignore
            model=model,
        )
        return chat_completion.choices[0].message.content or ""
