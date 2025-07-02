import os
import base64
import os
from openai import OpenAI
from groq import Groq
from typing import Optional, List, Dict, Union, Iterator

# This service is built using the OpenAI Python library for both Groq and Gemini.
# The Gemini client uses the OpenAI compatibility layer as suggested.
# Reference: https://ai.google.dev/gemini-api/docs/openai

class LLMService:
    """
    A unified service to interact with Groq and Google Gemini models
    using their OpenAI-compatible APIs.
    """
    def __init__(self):
        """Initializes the clients for Groq and Gemini."""
        groq_api_key = os.getenv("GROQ_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Client for Groq API (which is OpenAI-compatible)
        self.groq_client = Groq(api_key=groq_api_key)

        # Client for Gemini API using the OpenAI compatibility layer
        self.gemini_client = OpenAI(
            api_key=gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def generate_response(
        self,
        model_name: str,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        stream: bool = False,
    ) -> Union[str, Iterator]:
        """
        Generates a response from the specified LLM.
        Includes automatic fallback from Groq to Gemini on rate limits.

        Args:
            model_name: The name of the model to use (e.g., "gemini-1.5-flash-latest", "llama3-8b-8192").
            prompt: The text prompt.
            image_bytes: Optional bytes of the image for multimodal input.
            stream: Whether to stream the response.

        Returns:
            The generated response as a string or a stream iterator.
        """
        # Define primary and fallback models
        primary_model = model_name
        fallback_model = "gemini-2.5-flash" if "gemini" not in model_name.lower() else model_name
        
        # Try primary model first, fallback to Gemini if rate limited
        for model_to_try in [primary_model, fallback_model]:
            if "gemini" in model_to_try.lower():
                print(f"---LLMService: Attempting Gemini model: {model_to_try}---")
                client = self.gemini_client
                messages = self._construct_multimodal_messages(prompt, image_bytes)
                model_to_use = model_to_try
            else:
                print(f"---LLMService: Attempting Groq model: {model_to_try}---")
                if image_bytes:
                    print("Warning: Groq models do not support image input. The image will be ignored.")
                client = self.groq_client
                messages = [{"role": "user", "content": prompt}]
                model_to_use = model_to_try

            try:
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    stream=stream,
                )

                if stream:
                    return response
                else:
                    return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                print(f"Error calling LLM API for model {model_to_try}: {error_msg}")
                
                # Check if it's a rate limit error
                if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                    print(f"Rate limit hit for {model_to_try}, trying fallback...")
                    if model_to_try == fallback_model:
                        # If fallback also fails, raise the error
                        raise Exception(f"Both primary and fallback models failed. Last error: {str(e)}")
                    continue
                else:
                    # For non-rate-limit errors, try fallback once
                    if model_to_try == primary_model and primary_model != fallback_model:
                        print(f"Non-rate-limit error with {model_to_try}, trying fallback...")
                        continue
                    else:
                        # If fallback also fails with non-rate-limit error, raise error
                        raise e
        
        # This should never be reached, but just in case
        raise Exception("All models failed to generate response")

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        image_bytes: Optional[bytes] = None,
        stream: bool = False,
    ) -> Union[str, Iterator]:
        """
        Alternative interface method for compatibility with existing code.
        Combines system and user prompts into a single conversation.
        Includes automatic fallback from Groq to Gemini on rate limits.
        """
        # Combine system and user prompts
        combined_prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        
        # Define primary and fallback models
        primary_model = model
        fallback_model = "gemini-2.5-flash" if "gemini" not in model.lower() else model
        
        # Try primary model first, fallback to Gemini if rate limited
        for model_to_try in [primary_model, fallback_model]:
            try:
                print(f"LLMService.invoke: Attempting model: {model_to_try}")
                result = self.generate_response(
                    model_name=model_to_try,
                    prompt=combined_prompt,
                    image_bytes=image_bytes,
                    stream=stream
                )
                print(f"LLMService.invoke: Successfully used model: {model_to_try}")
                return result
            except Exception as e:
                error_msg = str(e)
                print(f"LLMService.invoke: Error with model {model_to_try}: {error_msg}")
                
                # Check if it's a rate limit error
                if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                    print(f"Rate limit hit for {model_to_try}, trying fallback...")
                    if model_to_try == fallback_model:
                        # If fallback also fails, raise the error
                        raise Exception(f"Both primary and fallback models failed. Last error: {str(e)}")
                    continue
                else:
                    # For non-rate-limit errors, try fallback once
                    if model_to_try == primary_model and primary_model != fallback_model:
                        print(f"Non-rate-limit error with {model_to_try}, trying fallback...")
                        continue
                    else:
                        # If fallback also fails with non-rate-limit error, raise error
                        raise e
        
        # This should never be reached, but just in case
        raise Exception("All models failed to generate response")

    def _construct_multimodal_messages(self, prompt: str, image_bytes: Optional[bytes]) -> List[Dict[str, any]]:
        """Constructs the message payload for a multimodal request in OpenAI format."""
        if not image_bytes:
            return [{"role": "user", "content": prompt}]

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        # This is the standard OpenAI format for vision, which the Gemini compatibility layer supports.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ] 