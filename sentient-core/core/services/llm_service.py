import os
import asyncio
import base64
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional, Union, Awaitable
from collections import defaultdict, deque
from datetime import datetime

from groq import Groq, RateLimitError as GroqRateLimitError
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, client, model_list):
        self.client = client
        self.model_list = model_list

    @abstractmethod
    async def generate(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate a non-streaming response."""
        pass

    @abstractmethod
    async def generate_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        # This is a placeholder for the async generator
        yield ""

class GroqProvider(LLMProvider):
    """LLM provider for Groq models."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        super().__init__(Groq(api_key=api_key), ["llama3-8b-8192", "mixtral-8x7b-32768"])

    async def generate(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    async def generate_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        stream = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, stream=True, **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class OpenAIProvider(LLMProvider):
    """LLM provider for OpenAI models."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        super().__init__(OpenAI(api_key=api_key), ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])

    async def generate(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    async def generate_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        stream = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, stream=True, **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class GeminiProvider(LLMProvider):
    """LLM provider for Google Gemini models via OpenAI compatibility layer."""

    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set.")
        client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        super().__init__(client, ["gemini-1.5-flash-latest"])

    async def generate(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    async def generate_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        stream = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, stream=True, **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class EnhancedLLMService:
    """
    A robust, multi-provider LLM service with streaming and fallback capabilities.
    Includes basic tracking for usage statistics and provider performance.
    """

    def __init__(self):
        self.providers = {}
        self.fallback_chain = []
        
        # Initialize providers only if their API keys are available
        if os.getenv("GROQ_API_KEY"):
            try:
                self.providers['groq'] = GroqProvider()
                self.fallback_chain.append('groq')
            except Exception as e:
                print(f"Failed to initialize Groq provider: {e}")
        
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.providers['openai'] = OpenAIProvider()
                self.fallback_chain.append('openai')
            except Exception as e:
                print(f"Failed to initialize OpenAI provider: {e}")
        
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            try:
                self.providers['gemini'] = GeminiProvider()
                self.fallback_chain.append('gemini')
            except Exception as e:
                print(f"Failed to initialize Gemini provider: {e}")
        
        if not self.providers:
            raise ValueError("No LLM providers could be initialized. Please check your API keys.")
        
        print(f"Initialized LLM service with providers: {list(self.providers.keys())}")
        
        # Tracking system
        self.usage_stats = defaultdict(int)  # Track usage per provider
        self.error_stats = defaultdict(int)  # Track errors per provider
        self.response_times = defaultdict(list)  # Track response times per provider
        self.recent_requests = deque(maxlen=100)  # Keep last 100 requests for analysis

    def _get_provider_for_model(self, model: str) -> str:
        """Finds the provider that supports the given model, defaulting to first available."""
        for name, provider in self.providers.items():
            if model in provider.model_list:
                return name
        
        # Fallback logic based on model name patterns (only if provider is available)
        if 'gemini' in model.lower() and 'gemini' in self.providers:
            return 'gemini'
        elif ('gpt' in model.lower() or 'openai' in model.lower()) and 'openai' in self.providers:
            return 'openai'
        elif 'groq' in self.providers:
            return 'groq'
        else:
            # Return the first available provider
            return list(self.providers.keys())[0]

    def _construct_multimodal_messages(self, prompt: str, image_bytes: Optional[bytes]) -> List[Dict[str, Any]]:
        """Constructs the message payload for a multimodal request."""
        if not image_bytes:
            return [{"role": "user", "content": prompt}]
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        image_bytes: Optional[bytes] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[Awaitable[str], AsyncGenerator[str, None]]:
        """Primary method to interact with the LLM service."""
        prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        messages = self._construct_multimodal_messages(prompt, image_bytes)
        if stream:
            return self.stream_response(model, messages, **kwargs)
        else:
            return self.generate_with_fallback(model, messages, **kwargs)
    
    def invoke_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        image_bytes: Optional[bytes] = None,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """Synchronous wrapper for the invoke method."""
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        self.invoke(system_prompt, user_prompt, model, image_bytes, stream, **kwargs)
                    )
                    return future.result()
            else:
                # If no loop is running, we can use asyncio.run
                return asyncio.run(
                    self.invoke(system_prompt, user_prompt, model, image_bytes, stream, **kwargs)
                )
        except RuntimeError:
            # Fallback: create a new event loop
            return asyncio.run(
                self.invoke(system_prompt, user_prompt, model, image_bytes, stream, **kwargs)
            )

    async def generate_with_fallback(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generates a response with fallback and tracks usage statistics."""
        provider_name = self._get_provider_for_model(model)
        start_index = self.fallback_chain.index(provider_name)
        
        for i in range(start_index, len(self.fallback_chain)):
            current_provider_name = self.fallback_chain[i]
            provider = self.providers[current_provider_name]
            current_model = model if current_provider_name == provider_name else provider.model_list[0]
            
            start_time = time.time()
            try:
                print(f"Attempting generation with {current_provider_name} using model {current_model}...")
                result = await provider.generate(current_model, messages, **kwargs)
                
                # Track successful usage
                response_time = time.time() - start_time
                self._track_success(current_provider_name, current_model, response_time)
                
                return result
            except (GroqRateLimitError, OpenAIRateLimitError) as e:
                self._track_error(current_provider_name, "rate_limit", str(e))
                print(f"Provider {current_provider_name} failed with rate limit: {e}")
                if i == len(self.fallback_chain) - 1: raise
            except Exception as e:
                self._track_error(current_provider_name, "general_error", str(e))
                print(f"Provider {current_provider_name} failed with error: {e}")
                if i == len(self.fallback_chain) - 1: raise
        
        raise Exception("All LLM providers failed.")

    def _track_success(self, provider_name: str, model: str, response_time: float):
        """Track successful API call."""
        self.usage_stats[provider_name] += 1
        self.response_times[provider_name].append(response_time)
        
        # Keep only last 50 response times per provider for memory efficiency
        if len(self.response_times[provider_name]) > 50:
            self.response_times[provider_name] = self.response_times[provider_name][-50:]
        
        self.recent_requests.append({
            'timestamp': datetime.now(),
            'provider': provider_name,
            'model': model,
            'response_time': response_time,
            'status': 'success'
        })

    def _track_error(self, provider_name: str, error_type: str, error_message: str):
        """Track failed API call."""
        self.error_stats[f"{provider_name}_{error_type}"] += 1
        
        self.recent_requests.append({
            'timestamp': datetime.now(),
            'provider': provider_name,
            'error_type': error_type,
            'error_message': error_message[:100],  # Truncate long error messages
            'status': 'error'
        })

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        stats = {
            'usage_by_provider': dict(self.usage_stats),
            'errors_by_provider': dict(self.error_stats),
            'average_response_times': {},
            'total_requests': sum(self.usage_stats.values()),
            'total_errors': sum(self.error_stats.values()),
            'recent_requests_count': len(self.recent_requests)
        }
        
        # Calculate average response times
        for provider, times in self.response_times.items():
            if times:
                stats['average_response_times'][provider] = sum(times) / len(times)
        
        return stats

    async def stream_response(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        """Streams a response with fallback and tracks usage statistics."""
        provider_name = self._get_provider_for_model(model)
        start_index = self.fallback_chain.index(provider_name)
        
        for i in range(start_index, len(self.fallback_chain)):
            current_provider_name = self.fallback_chain[i]
            provider = self.providers[current_provider_name]
            current_model = model if current_provider_name == provider_name else provider.model_list[0]
            
            start_time = time.time()
            try:
                print(f"Attempting stream with {current_provider_name} using model {current_model}...")
                chunk_count = 0
                async for chunk in provider.generate_stream(current_model, messages, **kwargs):
                    chunk_count += 1
                    yield chunk
                
                # Track successful streaming
                response_time = time.time() - start_time
                self._track_success(current_provider_name, current_model, response_time)
                return
                
            except (GroqRateLimitError, OpenAIRateLimitError) as e:
                self._track_error(current_provider_name, "rate_limit", str(e))
                print(f"Provider {current_provider_name} stream failed with rate limit: {e}")
                if i == len(self.fallback_chain) - 1: 
                    yield f"Error: All providers failed. Last error: {e}"
            except Exception as e:
                self._track_error(current_provider_name, "general_error", str(e))
                print(f"Provider {current_provider_name} stream failed with error: {e}")
                if i == len(self.fallback_chain) - 1: 
                    yield f"Error: All providers failed. Last error: {e}"