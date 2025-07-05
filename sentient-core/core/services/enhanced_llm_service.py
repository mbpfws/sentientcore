import os
import asyncio
import base64
import time
import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional, Union, Awaitable, Callable
from collections import defaultdict, deque
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from groq import Groq, RateLimitError as GroqRateLimitError
from openai import OpenAI, RateLimitError as OpenAIRateLimitError
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables for API keys
load_dotenv()

class ModelCapability(Enum):
    """Enum for different model capabilities."""
    TEXT_GENERATION = "text_generation"
    STRUCTURED_OUTPUT = "structured_output"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    REASONING = "reasoning"
    CODING = "coding"

@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    provider: str
    capabilities: List[ModelCapability]
    context_length: int
    max_output_tokens: Optional[int] = None
    supports_streaming: bool = True
    supports_tools: bool = False
    reasoning_model: bool = False

class StructuredOutputSchema(BaseModel):
    """Base class for structured output schemas."""
    pass

class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, client, models: List[ModelInfo]):
        self.client = client
        self.models = {model.name: model for model in models}
        self.model_list = [model.name for model in models]

    @abstractmethod
    async def generate(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate a non-streaming response."""
        pass

    @abstractmethod
    async def generate_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        yield ""

    @abstractmethod
    async def generate_structured(self, model: str, messages: List[Dict[str, Any]], 
                                schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate a structured response."""
        pass

    @abstractmethod
    async def generate_with_tools(self, model: str, messages: List[Dict[str, Any]], 
                                tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate a response with tool calling capabilities."""
        pass

class EnhancedGroqProvider(LLMProvider):
    """Enhanced LLM provider for Groq models with advanced features."""

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        
        # Define latest Groq models with their capabilities
        models = [
            # Vision-capable models (prioritized for multimodal tasks)
            ModelInfo(
                name="meta-llama/llama-4-scout-17b-16e-instruct",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.VISION, 
                            ModelCapability.FUNCTION_CALLING, ModelCapability.STRUCTURED_OUTPUT],
                context_length=131072,
                max_output_tokens=8192,
                supports_tools=True
            ),
            ModelInfo(
                name="meta-llama/llama-4-maverick-17b-128e-instruct",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.VISION, 
                            ModelCapability.FUNCTION_CALLING, ModelCapability.STRUCTURED_OUTPUT],
                context_length=131072,
                max_output_tokens=8192,
                supports_tools=True
            ),
            # Text-only models
            ModelInfo(
                name="llama-3.3-70b-versatile",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT, 
                            ModelCapability.FUNCTION_CALLING, ModelCapability.REASONING],
                context_length=131072,
                max_output_tokens=32768,
                supports_tools=True,
                reasoning_model=True
            ),
            ModelInfo(
                name="llama-3.1-70b-versatile",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT, 
                            ModelCapability.FUNCTION_CALLING],
                context_length=131072,
                max_output_tokens=32768,
                supports_tools=True
            ),
            ModelInfo(
                name="llama-3.1-8b-instant",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT],
                context_length=131072,
                max_output_tokens=32768
            ),
            ModelInfo(
                name="mixtral-8x7b-32768",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODING],
                context_length=32768,
                max_output_tokens=32768
            ),
            ModelInfo(
                name="gemma2-9b-it",
                provider="groq",
                capabilities=[ModelCapability.TEXT_GENERATION],
                context_length=8192,
                max_output_tokens=8192
            )
        ]
        
        super().__init__(Groq(api_key=api_key), models)

    async def generate(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate a non-streaming response."""
        response = await asyncio.to_thread(
            self.client.chat.completions.create, 
            model=model, 
            messages=messages, 
            **kwargs
        )
        return response.choices[0].message.content

    async def generate_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        stream = await asyncio.to_thread(
            self.client.chat.completions.create, 
            model=model, 
            messages=messages, 
            stream=True, 
            **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_structured(self, model: str, messages: List[Dict[str, Any]], 
                                schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate a structured response using JSON schema."""
        if model not in self.models or ModelCapability.STRUCTURED_OUTPUT not in self.models[model].capabilities:
            raise ValueError(f"Model {model} does not support structured output")
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": schema
                }
            },
            **kwargs
        )
        
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse structured response: {e}")

    async def generate_with_tools(self, model: str, messages: List[Dict[str, Any]], 
                                tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Generate a response with tool calling capabilities."""
        if model not in self.models or not self.models[model].supports_tools:
            raise ValueError(f"Model {model} does not support tool calling")
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **kwargs
        )
        
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
            "finish_reason": response.choices[0].finish_reason
        }

class OpenAIProvider(LLMProvider):
    """Enhanced OpenAI provider."""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        models = [
            ModelInfo(
                name="gpt-4o",
                provider="openai",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.VISION, 
                            ModelCapability.FUNCTION_CALLING, ModelCapability.STRUCTURED_OUTPUT],
                context_length=128000,
                max_output_tokens=16384,
                supports_tools=True
            ),
            ModelInfo(
                name="gpt-4o-mini",
                provider="openai",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.VISION, 
                            ModelCapability.FUNCTION_CALLING, ModelCapability.STRUCTURED_OUTPUT],
                context_length=128000,
                max_output_tokens=16384,
                supports_tools=True
            ),
            ModelInfo(
                name="o1-preview",
                provider="openai",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.REASONING],
                context_length=128000,
                max_output_tokens=32768,
                supports_streaming=False,
                reasoning_model=True
            )
        ]
        
        super().__init__(OpenAI(api_key=api_key), models)

    async def generate(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, **kwargs
        )
        return response.choices[0].message.content

    async def generate_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator[str, None]:
        if model in self.models and not self.models[model].supports_streaming:
            # For non-streaming models, yield the complete response
            result = await self.generate(model, messages, **kwargs)
            yield result
            return
            
        stream = await asyncio.to_thread(
            self.client.chat.completions.create, model=model, messages=messages, stream=True, **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate_structured(self, model: str, messages: List[Dict[str, Any]], 
                                schema: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if model not in self.models or ModelCapability.STRUCTURED_OUTPUT not in self.models[model].capabilities:
            raise ValueError(f"Model {model} does not support structured output")
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": schema
                }
            },
            **kwargs
        )
        
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse structured response: {e}")

    async def generate_with_tools(self, model: str, messages: List[Dict[str, Any]], 
                                tools: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        if model not in self.models or not self.models[model].supports_tools:
            raise ValueError(f"Model {model} does not support tool calling")
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            **kwargs
        )
        
        return {
            "content": response.choices[0].message.content,
            "tool_calls": response.choices[0].message.tool_calls,
            "finish_reason": response.choices[0].finish_reason
        }

# GeminiProvider removed - using only Groq API as requested

class EnhancedLLMService:
    """Enhanced LLM service that manages multiple providers and intelligent model selection."""
    
    def __init__(self):
        self.providers = {}
        self.default_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers based on API keys."""
        try:
            groq_provider = EnhancedGroqProvider()
            self.providers["groq"] = groq_provider
            self.default_provider = "groq"
        except ValueError:
            pass
        
        try:
            openai_provider = OpenAIProvider()
            self.providers["openai"] = openai_provider
            if not self.default_provider:
                self.default_provider = "openai"
        except ValueError:
            pass
        
        if not self.providers:
            raise ValueError("No LLM providers available. Please set GROQ_API_KEY or OPENAI_API_KEY.")
    
    def get_best_model_for_capability(self, capability: ModelCapability, provider: Optional[str] = None) -> tuple[str, str]:
        """Get the best model for a specific capability.
        
        Returns:
            tuple: (provider_name, model_name)
        """
        target_providers = [provider] if provider and provider in self.providers else list(self.providers.keys())
        
        best_model = None
        best_provider = None
        best_score = -1
        
        for provider_name in target_providers:
            provider_obj = self.providers[provider_name]
            for model_name, model_info in provider_obj.models.items():
                if capability in model_info.capabilities:
                    # Score based on context length and capabilities
                    score = model_info.context_length
                    if model_info.reasoning_model:
                        score += 50000  # Bonus for reasoning models
                    if model_info.supports_tools:
                        score += 10000  # Bonus for tool support
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                        best_provider = provider_name
        
        if not best_model:
            raise ValueError(f"No model found for capability: {capability}")
        
        return best_provider, best_model
    
    async def generate(self, messages: List[Dict[str, Any]], capability: Optional[ModelCapability] = None, 
                      provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> str:
        """Generate a response using the best available model."""
        if model and provider:
            target_provider = provider
            target_model = model
        elif capability:
            target_provider, target_model = self.get_best_model_for_capability(capability, provider)
        else:
            target_provider = provider or self.default_provider
            target_model = model or list(self.providers[target_provider].models.keys())[0]
        
        return await self.providers[target_provider].generate(target_model, messages, **kwargs)
    
    async def generate_stream(self, messages: List[Dict[str, Any]], capability: Optional[ModelCapability] = None,
                             provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Generate a streaming response."""
        if model and provider:
            target_provider = provider
            target_model = model
        elif capability:
            target_provider, target_model = self.get_best_model_for_capability(capability, provider)
        else:
            target_provider = provider or self.default_provider
            target_model = model or list(self.providers[target_provider].models.keys())[0]
        
        async for chunk in self.providers[target_provider].generate_stream(target_model, messages, **kwargs):
            yield chunk
    
    async def generate_structured(self, messages: List[Dict[str, Any]], schema: Dict[str, Any],
                                 provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate a structured response."""
        if model and provider:
            target_provider = provider
            target_model = model
        else:
            target_provider, target_model = self.get_best_model_for_capability(ModelCapability.STRUCTURED_OUTPUT, provider)
        
        return await self.providers[target_provider].generate_structured(target_model, messages, schema, **kwargs)
    
    async def generate_with_tools(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                                 provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate a response with tool calling."""
        if model and provider:
            target_provider = provider
            target_model = model
        else:
            target_provider, target_model = self.get_best_model_for_capability(ModelCapability.FUNCTION_CALLING, provider)
        
        return await self.providers[target_provider].generate_with_tools(target_model, messages, tools, **kwargs)
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models by provider."""
        return {provider_name: list(provider.models.keys()) for provider_name, provider in self.providers.items()}
    
    def get_model_info(self, provider: str, model: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        if provider in self.providers and model in self.providers[provider].models:
            return self.providers[provider].models[model]
        return None