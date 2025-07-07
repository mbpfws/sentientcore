from typing import Dict, List, Optional, Any, AsyncGenerator, Union, Callable
import asyncio
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

# LLM Provider imports
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
except ImportError:
    OpenAI = None
    ChatOpenAI = None


class LLMProvider(Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    GOOGLE = "google"
    OPENAI = "openai"
    LOCAL = "local"


class MessageRole(Enum):
    """Message roles in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


@dataclass
class ChatMessage:
    """Represents a chat message"""
    role: MessageRole
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata or {},
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            metadata=data.get("metadata"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
        )


@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    retry_attempts: int = 3
    custom_params: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from LLM service"""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    processing_time: Optional[float] = None


class EnhancedLLMService:
    """Enhanced LLM service with multiple provider support and advanced features"""
    
    def __init__(self, default_config: Optional[LLMConfig] = None):
        self.logger = logging.getLogger(__name__)
        
        # Provider configurations
        self._configs: Dict[str, LLMConfig] = {}
        self._clients: Dict[str, Any] = {}
        
        # Default configuration
        if default_config:
            self.add_provider_config("default", default_config)
        else:
            self._setup_default_config()
        
        # Conversation management
        self._conversations: Dict[str, List[ChatMessage]] = {}
        self._conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "average_response_time": 0.0
        }
        
        # Response caching
        self._response_cache: Dict[str, LLMResponse] = {}
        self._cache_enabled = True
        self._max_cache_size = 1000
    
    def _setup_default_config(self):
        """Setup default configuration based on available environment variables"""
        # Try Groq first
        if os.getenv("GROQ_API_KEY") and Groq:
            config = LLMConfig(
                provider=LLMProvider.GROQ,
                model="mixtral-8x7b-32768",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.7,
                max_tokens=4000
            )
            self.add_provider_config("default", config)
        
        # Try Google Gemini
        elif os.getenv("GOOGLE_API_KEY") and genai:
            config = LLMConfig(
                provider=LLMProvider.GOOGLE,
                model="gemini-pro",
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7,
                max_tokens=4000
            )
            self.add_provider_config("default", config)
        
        # Try OpenAI
        elif os.getenv("OPENAI_API_KEY") and ChatOpenAI:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7,
                max_tokens=4000
            )
            self.add_provider_config("default", config)
        
        else:
            self.logger.warning("No LLM provider configured. Please set up API keys.")
    
    def add_provider_config(self, name: str, config: LLMConfig):
        """Add a provider configuration"""
        self._configs[name] = config
        self._initialize_client(name, config)
    


    def _initialize_client(self, name: str, config: LLMConfig):
        """Initialize client for a provider"""
        try:
            if config.provider == LLMProvider.GROQ and Groq:
                self._clients[name] = Groq(api_key=config.api_key)
            
            elif config.provider == LLMProvider.GOOGLE and genai:
                genai.configure(api_key=config.api_key)
                self._clients[name] = genai.GenerativeModel(config.model)
            
            elif config.provider == LLMProvider.OPENAI and ChatOpenAI:
                self._clients[name] = ChatOpenAI(
                    openai_api_key=config.api_key,
                    model_name=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
            
            self.logger.info(f"Initialized {config.provider.value} client: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {config.provider.value} client {name}: {e}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        provider_name: str = "default",
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate chat completion"""
        if provider_name not in self._configs:
            raise ValueError(f"Provider {provider_name} not configured")
        
        config = self._configs[provider_name]
        client = self._clients.get(provider_name)
        
        if not client:
            raise RuntimeError(f"Client for {provider_name} not initialized")
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(messages, config, kwargs)
            if self._cache_enabled and not stream and cache_key in self._response_cache:
                self.logger.debug(f"Cache hit for request: {cache_key[:50]}...")
                return self._response_cache[cache_key]
            
            # Generate response based on provider
            if config.provider == LLMProvider.GROQ:
                response = await self._groq_completion(client, config, messages, stream, **kwargs)
            elif config.provider == LLMProvider.GOOGLE:
                response = await self._google_completion(client, config, messages, stream, **kwargs)
            elif config.provider == LLMProvider.OPENAI:
                response = await self._openai_completion(client, config, messages, stream, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
            
            if not stream:
                return response.content
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_stats(True, processing_time)
            
            # Cache non-streaming responses
            if not stream and isinstance(response, LLMResponse):
                response.processing_time = processing_time
                if self._cache_enabled:
                    self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            self._update_stats(False, (datetime.now() - start_time).total_seconds())
            self.logger.error(f"Chat completion failed: {e}")
            raise
    
    async def _groq_completion(
        self,
        client: Any,
        config: LLMConfig,
        messages: List[ChatMessage],
        stream: bool,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Groq completion"""
        formatted_messages = messages
        
        params = {
            "model": config.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "top_p": kwargs.get("top_p", config.top_p),
            "stream": stream
        }
        
        if stream:
            return self._groq_stream_completion(client, params)
        else:
            response = client.chat.completions.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=config.provider,
                model=config.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None,
                timestamp=datetime.now()
            )
    
    async def _groq_stream_completion(self, client: Any, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Handle Groq streaming completion"""
        try:
            stream = client.chat.completions.create(**params)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            self.logger.error(f"Groq streaming error: {e}")
            yield f"Error: {str(e)}"
    
    async def _google_completion(
        self,
        client: Any,
        config: LLMConfig,
        messages: List[ChatMessage],
        stream: bool,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle Google Gemini completion"""
        # Convert messages to Gemini format
        prompt = "\n".join([f"{msg.role.value}: {msg.content}" for msg in messages])
        
        generation_config = {
            "temperature": kwargs.get("temperature", config.temperature),
            "max_output_tokens": kwargs.get("max_tokens", config.max_tokens),
            "top_p": kwargs.get("top_p", config.top_p)
        }
        
        if stream:
            return self._google_stream_completion(client, prompt, generation_config)
        else:
            response = client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return LLMResponse(
                content=response.text,
                provider=config.provider,
                model=config.model,
                usage={
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                } if hasattr(response, 'usage_metadata') else None,
                timestamp=datetime.now()
            )
    
    async def _google_stream_completion(
        self, 
        client: Any, 
        prompt: str, 
        generation_config: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Handle Google streaming completion"""
        try:
            response = client.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self.logger.error(f"Google streaming error: {e}")
            yield f"Error: {str(e)}"
    
    async def _openai_completion(
        self,
        client: Any,
        config: LLMConfig,
        messages: List[ChatMessage],
        stream: bool,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Handle OpenAI completion"""
        formatted_messages = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]
        
        if stream:
            return self._openai_stream_completion(client, formatted_messages, **kwargs)
        else:
            response = await client.agenerate([formatted_messages])
            
            return LLMResponse(
                content=response.generations[0][0].text,
                provider=config.provider,
                model=config.model,
                timestamp=datetime.now()
            )
    
    async def _openai_stream_completion(
        self, 
        client: Any, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Handle OpenAI streaming completion"""
        try:
            async for chunk in client.astream(messages):
                if chunk:
                    yield chunk
        except Exception as e:
            self.logger.error(f"OpenAI streaming error: {e}")
            yield f"Error: {str(e)}"
    
    async def start_conversation(
        self,
        conversation_id: str,
        system_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new conversation"""
        self._conversations[conversation_id] = []
        self._conversation_metadata[conversation_id] = metadata or {}
        
        if system_message:
            self._conversations[conversation_id].append(
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=system_message,
                    timestamp=datetime.now()
                )
            )
        
        self.logger.info(f"Started conversation: {conversation_id}")
        return conversation_id
    
    async def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add message to conversation"""
        if conversation_id not in self._conversations:
            await self.start_conversation(conversation_id)
        
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata,
            timestamp=datetime.now()
        )
        
        self._conversations[conversation_id].append(message)
        return message
    
    async def get_conversation(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get conversation messages"""
        if conversation_id not in self._conversations:
            return []
        
        messages = self._conversations[conversation_id]
        if limit:
            return messages[-limit:]
        return messages
    
    async def continue_conversation(
        self,
        conversation_id: str,
        user_message: str,
        provider_name: str = "default",
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Continue an existing conversation"""
        # Add user message
        await self.add_message(conversation_id, MessageRole.USER, user_message)
        
        # Get conversation history
        messages = await self.get_conversation(conversation_id)
        
        # Generate response
        response = await self.chat_completion(
            messages=messages,
            provider_name=provider_name,
            stream=stream,
            **kwargs
        )
        
        # Add assistant response to conversation (if not streaming)
        if not stream and isinstance(response, LLMResponse):
            await self.add_message(
                conversation_id,
                MessageRole.ASSISTANT,
                response.content
            )
        
        return response
    
    def _generate_cache_key(
        self,
        messages: List[ChatMessage],
        config: LLMConfig,
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for request"""
        key_data = {
            "messages": [msg.to_dict() for msg in messages],
            "provider": config.provider.value,
            "model": config.model,
            "temperature": kwargs.get("temperature", config.temperature),
            "max_tokens": kwargs.get("max_tokens", config.max_tokens)
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _cache_response(self, key: str, response: LLMResponse):
        """Cache a response"""
        if len(self._response_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[key] = response
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update service statistics"""
        self._stats["total_requests"] += 1
        
        if success:
            self._stats["successful_requests"] += 1
        else:
            self._stats["failed_requests"] += 1
        
        # Update average response time
        total_time = self._stats["average_response_time"] * (self._stats["total_requests"] - 1)
        self._stats["average_response_time"] = (total_time + processing_time) / self._stats["total_requests"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self._stats,
            "configured_providers": list(self._configs.keys()),
            "active_conversations": len(self._conversations),
            "cache_size": len(self._response_cache),
            "cache_enabled": self._cache_enabled
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self._response_cache.clear()
        self.logger.info("Response cache cleared")
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable response caching"""
        self._cache_enabled = enabled
        self.logger.info(f"Response caching {'enabled' if enabled else 'disabled'}")
    
    async def cleanup_conversations(self, max_age_hours: int = 24):
        """Clean up old conversations"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        conversations_to_remove = []
        for conv_id, messages in self._conversations.items():
            if messages and messages[-1].timestamp:
                if messages[-1].timestamp.timestamp() < cutoff_time:
                    conversations_to_remove.append(conv_id)
        
        for conv_id in conversations_to_remove:
            del self._conversations[conv_id]
            self._conversation_metadata.pop(conv_id, None)
        
        self.logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
        return len(conversations_to_remove)