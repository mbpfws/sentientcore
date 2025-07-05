import os
import asyncio
import base64
import time
import json
from typing import AsyncGenerator, List, Dict, Any, Optional, Union, Awaitable, Callable
from collections import defaultdict, deque
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .enhanced_llm_service import (
    EnhancedGroqProvider, OpenAIProvider, 
    ModelCapability, ModelInfo, StructuredOutputSchema
)

class AgenticTool:
    """Represents an agentic tool that can be called by the LLM."""
    
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], 
                 function: Callable, async_function: bool = False):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.async_function = async_function
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with given arguments."""
        if self.async_function:
            return await self.function(**arguments)
        else:
            return self.function(**arguments)

class ConversationContext:
    """Manages conversation context and memory."""
    
    def __init__(self, max_messages: int = 50):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        self.metadata: Dict[str, Any] = {}
        self.tools_used: List[str] = []
        self.structured_outputs: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation context."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            message["metadata"] = metadata
        
        self.messages.append(message)
        
        # Trim messages if exceeding max_messages
        if len(self.messages) > self.max_messages:
            # Keep system message if it exists
            system_messages = [msg for msg in self.messages if msg["role"] == "system"]
            other_messages = [msg for msg in self.messages if msg["role"] != "system"]
            
            # Keep the most recent messages
            recent_messages = other_messages[-(self.max_messages - len(system_messages)):]
            self.messages = system_messages + recent_messages
    
    def get_messages_for_api(self) -> List[Dict[str, Any]]:
        """Get messages formatted for API calls."""
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def add_tool_usage(self, tool_name: str):
        """Track tool usage."""
        self.tools_used.append(tool_name)
    
    def add_structured_output(self, output: Dict[str, Any]):
        """Track structured outputs."""
        self.structured_outputs.append({
            "output": output,
            "timestamp": datetime.now().isoformat()
        })

class EnhancedLLMService:
    """
    Advanced multi-provider LLM service with enhanced capabilities:
    - Structured output generation
    - Agentic tool calling
    - Advanced streaming
    - Conversation context management
    - Intelligent model selection
    - Comprehensive analytics
    """

    def __init__(self):
        self.providers = {}
        self.fallback_chain = []
        self.tools: Dict[str, AgenticTool] = {}
        self.contexts: Dict[str, ConversationContext] = {}
        
        # Initialize providers
        self._initialize_providers()
        
        # Analytics and tracking
        self.usage_stats = defaultdict(int)
        self.error_stats = defaultdict(int)
        self.response_times = defaultdict(list)
        self.recent_requests = deque(maxlen=200)
        self.model_performance = defaultdict(lambda: {"success_rate": 0, "avg_response_time": 0})
        
        print(f"Enhanced LLM service initialized with providers: {list(self.providers.keys())}")
        print(f"Available models: {self.get_available_models()}")

    def _initialize_providers(self):
        """Initialize all available providers."""
        # Initialize Groq provider (priority)
        if os.getenv("GROQ_API_KEY"):
            try:
                self.providers['groq'] = EnhancedGroqProvider()
                self.fallback_chain.append('groq')
                print("✓ Groq provider initialized with enhanced features")
            except Exception as e:
                print(f"✗ Failed to initialize Groq provider: {e}")
        
        # Initialize OpenAI provider
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.providers['openai'] = OpenAIProvider()
                self.fallback_chain.append('openai')
                print("✓ OpenAI provider initialized")
            except Exception as e:
                print(f"✗ Failed to initialize OpenAI provider: {e}")
        
        # Gemini provider removed - using only Groq API as requested
        
        if not self.providers:
            raise ValueError("No LLM providers could be initialized. Please check your API keys.")

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models by provider."""
        return {provider_name: provider.model_list for provider_name, provider in self.providers.items()}

    def get_model_capabilities(self, model: str) -> List[ModelCapability]:
        """Get capabilities for a specific model."""
        for provider in self.providers.values():
            if model in provider.models:
                return provider.models[model].capabilities
        return []

    def register_tool(self, tool: AgenticTool):
        """Register an agentic tool."""
        self.tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")

    def get_or_create_context(self, context_id: str) -> ConversationContext:
        """Get or create a conversation context."""
        if context_id not in self.contexts:
            self.contexts[context_id] = ConversationContext()
        return self.contexts[context_id]

    def _get_optimal_model(self, required_capabilities: List[ModelCapability], 
                          prefer_reasoning: bool = False) -> tuple[str, str]:
        """Get the optimal model based on required capabilities."""
        best_model = None
        best_provider = None
        best_score = -1
        
        for provider_name in self.fallback_chain:
            provider = self.providers[provider_name]
            for model_name, model_info in provider.models.items():
                score = 0
                
                # Check if model has all required capabilities
                if all(cap in model_info.capabilities for cap in required_capabilities):
                    score += 10
                    
                    # Bonus for reasoning models if preferred
                    if prefer_reasoning and model_info.reasoning_model:
                        score += 5
                    
                    # Bonus for larger context length
                    score += min(model_info.context_length / 10000, 5)
                    
                    # Bonus based on historical performance
                    perf_key = f"{provider_name}_{model_name}"
                    if perf_key in self.model_performance:
                        score += self.model_performance[perf_key]["success_rate"] * 3
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                        best_provider = provider_name
        
        if best_model is None:
            # Fallback to first available model
            first_provider = self.fallback_chain[0]
            first_model = self.providers[first_provider].model_list[0]
            return first_model, first_provider
        
        return best_model, best_provider

    def _construct_multimodal_messages(self, prompt: str, image_bytes: Optional[bytes] = None,
                                     context: Optional[ConversationContext] = None) -> List[Dict[str, Any]]:
        """Construct messages for multimodal requests."""
        messages = []
        
        # Add context messages if available
        if context:
            messages.extend(context.get_messages_for_api())
        
        # Construct current message
        if not image_bytes:
            messages.append({"role": "user", "content": prompt})
        else:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            })
        
        return messages

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        context_id: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        **kwargs
    ) -> str:
        """Generate a response, with special prioritization for Groq vision models when image is provided."""
        # If image is provided, prioritize Groq vision models
        if image_bytes and model is None:
            model, provider_name = self._get_optimal_model([ModelCapability.VISION])
        elif model is None:
            # For text-only requests, use general model selection
            model, provider_name = self._get_optimal_model([])
        else:
            provider_name = self._get_provider_for_model(model)
        
        # Get or create context
        context = self.get_or_create_context(context_id) if context_id else None
        
        # Construct messages with image support
        messages = self._construct_multimodal_messages(prompt, image_bytes, context)
        
        # Add user message to context
        if context:
            context.add_message("user", prompt)
        
        start_time = time.time()
        try:
            provider = self.providers[provider_name]
            result = await provider.generate(model, messages, **kwargs)
            
            # Track success
            response_time = time.time() - start_time
            self._track_success(provider_name, model, response_time)
            
            # Add to context
            if context:
                context.add_message("assistant", result)
            
            return result
            
        except Exception as e:
            self._track_error(provider_name, "generation_error", str(e))
            raise

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        context_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a structured response using JSON schema."""
        # Get optimal model for structured output
        if model is None:
            model, provider_name = self._get_optimal_model([ModelCapability.STRUCTURED_OUTPUT])
        else:
            provider_name = self._get_provider_for_model(model)
        
        # Get or create context
        context = self.get_or_create_context(context_id) if context_id else None
        
        # Construct messages
        messages = self._construct_multimodal_messages(prompt, context=context)
        
        # Add user message to context
        if context:
            context.add_message("user", prompt)
        
        start_time = time.time()
        try:
            provider = self.providers[provider_name]
            result = await provider.generate_structured(model, messages, schema, **kwargs)
            
            # Track success
            response_time = time.time() - start_time
            self._track_success(provider_name, model, response_time)
            
            # Add to context
            if context:
                context.add_message("assistant", json.dumps(result))
                context.add_structured_output(result)
            
            return result
            
        except Exception as e:
            self._track_error(provider_name, "structured_generation_error", str(e))
            raise

    async def generate_with_tools(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        model: Optional[str] = None,
        context_id: Optional[str] = None,
        auto_execute_tools: bool = True,
        max_tool_iterations: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response with tool calling capabilities."""
        # Get optimal model for tool calling
        if model is None:
            model, provider_name = self._get_optimal_model([ModelCapability.FUNCTION_CALLING])
        else:
            provider_name = self._get_provider_for_model(model)
        
        # Get or create context
        context = self.get_or_create_context(context_id) if context_id else None
        
        # Prepare tools
        available_tools = []
        if tools:
            for tool_name in tools:
                if tool_name in self.tools:
                    available_tools.append(self.tools[tool_name].to_openai_format())
        else:
            # Use all available tools
            available_tools = [tool.to_openai_format() for tool in self.tools.values()]
        
        # Construct initial messages
        messages = self._construct_multimodal_messages(prompt, context=context)
        
        # Add user message to context
        if context:
            context.add_message("user", prompt)
        
        iteration = 0
        final_response = None
        
        while iteration < max_tool_iterations:
            start_time = time.time()
            try:
                provider = self.providers[provider_name]
                result = await provider.generate_with_tools(model, messages, available_tools, **kwargs)
                
                response_time = time.time() - start_time
                self._track_success(provider_name, model, response_time)
                
                # Check if tools were called
                if result.get("tool_calls") and auto_execute_tools:
                    # Execute tools
                    tool_results = []
                    for tool_call in result["tool_calls"]:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        if tool_name in self.tools:
                            try:
                                tool_result = await self.tools[tool_name].execute(tool_args)
                                tool_results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": json.dumps(tool_result)
                                })
                                
                                # Track tool usage
                                if context:
                                    context.add_tool_usage(tool_name)
                                    
                            except Exception as e:
                                tool_results.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": f"Error executing tool: {str(e)}"
                                })
                    
                    # Add assistant message and tool results to conversation
                    messages.append({
                        "role": "assistant",
                        "content": result["content"],
                        "tool_calls": result["tool_calls"]
                    })
                    messages.extend(tool_results)
                    
                    # Add to context
                    if context:
                        context.add_message("assistant", result["content"], 
                                           {"tool_calls": [tc.function.name for tc in result["tool_calls"]]})
                    
                    iteration += 1
                    continue
                else:
                    # No more tools to call or auto-execution disabled
                    final_response = result
                    
                    # Add to context
                    if context:
                        context.add_message("assistant", result["content"])
                    
                    break
                    
            except Exception as e:
                self._track_error(provider_name, "tool_calling_error", str(e))
                raise
        
        return final_response or {"content": "Maximum tool iterations reached", "tool_calls": None}

    def _get_provider_for_model(self, model: str) -> str:
        """Find the provider that supports the given model."""
        for name, provider in self.providers.items():
            if model in provider.model_list:
                return name
        
        # Fallback logic - prioritize Groq as requested
        if ('gpt' in model.lower() or 'openai' in model.lower()) and 'openai' in self.providers:
            return 'openai'
        elif 'groq' in self.providers:
            return 'groq'
        else:
            return list(self.providers.keys())[0]

    def _track_success(self, provider_name: str, model: str, response_time: float):
        """Track successful API call."""
        self.usage_stats[provider_name] += 1
        self.response_times[provider_name].append(response_time)
        
        # Update model performance
        perf_key = f"{provider_name}_{model}"
        if perf_key not in self.model_performance:
            self.model_performance[perf_key] = {"success_count": 0, "total_count": 0, "total_time": 0}
        
        self.model_performance[perf_key]["success_count"] += 1
        self.model_performance[perf_key]["total_count"] += 1
        self.model_performance[perf_key]["total_time"] += response_time
        
        # Calculate success rate and average response time
        perf = self.model_performance[perf_key]
        perf["success_rate"] = perf["success_count"] / perf["total_count"]
        perf["avg_response_time"] = perf["total_time"] / perf["total_count"]
        
        # Keep only last 50 response times per provider
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
            'error_message': error_message[:100],
            'status': 'error'
        })

    def get_comprehensive_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics and performance metrics."""
        analytics = {
            'providers': {
                'available': list(self.providers.keys()),
                'fallback_chain': self.fallback_chain
            },
            'usage_statistics': {
                'total_requests': sum(self.usage_stats.values()),
                'requests_by_provider': dict(self.usage_stats),
                'total_errors': sum(self.error_stats.values()),
                'errors_by_type': dict(self.error_stats)
            },
            'performance_metrics': {
                'model_performance': dict(self.model_performance),
                'average_response_times': {}
            },
            'tools': {
                'registered_tools': list(self.tools.keys()),
                'total_tools': len(self.tools)
            },
            'contexts': {
                'active_contexts': len(self.contexts),
                'context_ids': list(self.contexts.keys())
            },
            'capabilities': {
                'models_by_capability': self._get_models_by_capability()
            }
        }
        
        # Calculate average response times
        for provider, times in self.response_times.items():
            if times:
                analytics['performance_metrics']['average_response_times'][provider] = sum(times) / len(times)
        
        return analytics

    def _get_models_by_capability(self) -> Dict[str, List[str]]:
        """Get models grouped by their capabilities."""
        capability_map = defaultdict(list)
        
        for provider in self.providers.values():
            for model_name, model_info in provider.models.items():
                for capability in model_info.capabilities:
                    capability_map[capability.value].append(model_name)
        
        return dict(capability_map)

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        image_bytes: Optional[bytes] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[Awaitable[str], AsyncGenerator[str, None]]:
        """Primary method to interact with the LLM service - compatibility method."""
        prompt = f"{system_prompt}\n\nUser: {user_prompt}"
        
        if stream:
            # For streaming, we need to return an async generator
            return self._stream_invoke(prompt, model, image_bytes, **kwargs)
        else:
            # For non-streaming, return an awaitable
            return self.generate(prompt, model, image_bytes=image_bytes, **kwargs)
    
    async def _stream_invoke(self, prompt: str, model: str, image_bytes: Optional[bytes] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Internal method for streaming invoke."""
        # Get provider for the model
        provider_name = self._get_provider_for_model(model)
        
        # Construct messages
        messages = self._construct_multimodal_messages(prompt, image_bytes)
        
        try:
            provider = self.providers[provider_name]
            async for chunk in provider.generate_stream(model, messages, **kwargs):
                yield chunk
        except Exception as e:
            self._track_error(provider_name, "streaming_error", str(e))
            yield f"Error: {str(e)}"
    
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

# Global instance
_enhanced_llm_service = None

def get_enhanced_llm_service() -> EnhancedLLMService:
    """Get the global enhanced LLM service instance."""
    global _enhanced_llm_service
    if _enhanced_llm_service is None:
        _enhanced_llm_service = EnhancedLLMService()
    return _enhanced_llm_service

def initialize_enhanced_llm_service() -> EnhancedLLMService:
    """Initialize and return the enhanced LLM service."""
    global _enhanced_llm_service
    _enhanced_llm_service = EnhancedLLMService()
    return _enhanced_llm_service