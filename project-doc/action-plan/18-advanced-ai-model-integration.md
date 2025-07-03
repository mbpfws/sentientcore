# 18 - Advanced AI Model Integration

## Overview

The Advanced AI Model Integration framework provides sophisticated AI model management, multi-model orchestration, intelligent model selection, and advanced AI capabilities. This includes support for multiple AI providers, model performance optimization, context-aware model switching, and specialized AI agent capabilities.

## Current State Analysis

### AI Model Requirements
- Multi-provider AI model support (OpenAI, Anthropic, Google, local models)
- Intelligent model selection based on task requirements
- Model performance monitoring and optimization
- Context-aware model switching
- Specialized AI capabilities (code generation, analysis, reasoning)
- Model cost optimization and usage tracking
- Fallback mechanisms for model failures
- Custom model fine-tuning support

### Integration Points
- Agent framework AI capabilities
- External service integrations
- Performance monitoring system
- Cost tracking and optimization
- Security and authentication
- Workflow orchestration

## Implementation Tasks

### Task 18.1: AI Model Management Engine

**File**: `core/ai/model_manager.py`

**AI Model Management Engine**:
```python
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import openai
import anthropic
from google.cloud import aiplatform
import tiktoken
from collections import defaultdict, deque

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"

class ModelCapability(Enum):
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    CONVERSATION = "conversation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    IMAGE_ANALYSIS = "image_analysis"

class ModelTier(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"

@dataclass
class ModelConfig:
    name: str
    provider: ModelProvider
    model_id: str
    capabilities: List[ModelCapability]
    tier: ModelTier
    max_tokens: int
    cost_per_token: float
    cost_per_request: float = 0.0
    context_window: int = 4096
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 10000
    quality_score: float = 0.8
    speed_score: float = 0.8
    reliability_score: float = 0.9
    enabled: bool = True
    fallback_models: List[str] = field(default_factory=list)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelUsageStats:
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_used: Optional[datetime] = None
    error_rate: float = 0.0
    performance_score: float = 1.0
    usage_trend: List[float] = field(default_factory=list)

@dataclass
class ModelRequest:
    task_type: ModelCapability
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
    functions: List[Dict[str, Any]] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    priority: str = "normal"  # low, normal, high, critical
    cost_limit: Optional[float] = None
    quality_requirement: float = 0.7
    speed_requirement: float = 0.5
    preferred_providers: List[ModelProvider] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)

@dataclass
class ModelResponse:
    content: str
    model_used: str
    provider: ModelProvider
    tokens_used: int
    cost: float
    response_time: float
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    function_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "completed"
    error: Optional[str] = None

class BaseModelProvider(ABC):
    """Base class for AI model providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.rate_limiter = None
        
    @abstractmethod
    async def initialize(self):
        """Initialize the provider"""
        pass
    
    @abstractmethod
    async def generate_response(self, request: ModelRequest, model_config: ModelConfig) -> ModelResponse:
        """Generate response using the model"""
        pass
    
    @abstractmethod
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        pass
    
    @abstractmethod
    async def validate_model(self, model_id: str) -> bool:
        """Validate if model is available and accessible"""
        pass

class OpenAIProvider(BaseModelProvider):
    """OpenAI model provider"""
    
    async def initialize(self):
        self.client = openai.AsyncOpenAI(
            api_key=self.config.get("api_key"),
            organization=self.config.get("organization")
        )
    
    async def generate_response(self, request: ModelRequest, model_config: ModelConfig) -> ModelResponse:
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [
                {"role": "user", "content": request.prompt}
            ]
            
            # Add context if available
            if request.context.get("system_prompt"):
                messages.insert(0, {"role": "system", "content": request.context["system_prompt"]})
            
            # Add images if supported
            if request.images and model_config.supports_vision:
                content = [{"type": "text", "text": request.prompt}]
                for image in request.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image}
                    })
                messages[-1]["content"] = content
            
            # Prepare parameters
            params = {
                "model": model_config.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens or model_config.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "frequency_penalty": request.frequency_penalty,
                "presence_penalty": request.presence_penalty,
                "stream": request.stream
            }
            
            if request.stop_sequences:
                params["stop"] = request.stop_sequences
            
            if request.functions and model_config.supports_functions:
                params["functions"] = request.functions
                params["function_call"] = "auto"
            
            # Make API call
            response = await self.client.chat.completions.create(**params)
            
            # Process response
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens
            cost = self._calculate_cost(tokens_used, model_config)
            response_time = time.time() - start_time
            
            # Extract function calls if any
            function_calls = []
            if hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
                function_calls.append({
                    "name": response.choices[0].message.function_call.name,
                    "arguments": response.choices[0].message.function_call.arguments
                })
            
            return ModelResponse(
                content=content,
                model_used=model_config.name,
                provider=ModelProvider.OPENAI,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time,
                quality_score=model_config.quality_score,
                function_calls=function_calls,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "model_id": model_config.model_id,
                    "usage": response.usage.model_dump()
                }
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_config.name,
                provider=ModelProvider.OPENAI,
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def _calculate_cost(self, tokens: int, model_config: ModelConfig) -> float:
        return tokens * model_config.cost_per_token + model_config.cost_per_request
    
    async def get_available_models(self) -> List[str]:
        try:
            models = await self.client.models.list()
            return [model.id for model in models.data]
        except Exception:
            return []
    
    async def validate_model(self, model_id: str) -> bool:
        available_models = await self.get_available_models()
        return model_id in available_models

class AnthropicProvider(BaseModelProvider):
    """Anthropic model provider"""
    
    async def initialize(self):
        self.client = anthropic.AsyncAnthropic(
            api_key=self.config.get("api_key")
        )
    
    async def generate_response(self, request: ModelRequest, model_config: ModelConfig) -> ModelResponse:
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = [
                {"role": "user", "content": request.prompt}
            ]
            
            # Prepare parameters
            params = {
                "model": model_config.model_id,
                "messages": messages,
                "max_tokens": request.max_tokens or model_config.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stream": request.stream
            }
            
            if request.context.get("system_prompt"):
                params["system"] = request.context["system_prompt"]
            
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences
            
            # Make API call
            response = await self.client.messages.create(**params)
            
            # Process response
            content = response.content[0].text if response.content else ""
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost = self._calculate_cost(tokens_used, model_config)
            response_time = time.time() - start_time
            
            return ModelResponse(
                content=content,
                model_used=model_config.name,
                provider=ModelProvider.ANTHROPIC,
                tokens_used=tokens_used,
                cost=cost,
                response_time=response_time,
                quality_score=model_config.quality_score,
                finish_reason=response.stop_reason or "completed",
                metadata={
                    "model_id": model_config.model_id,
                    "usage": response.usage.model_dump()
                }
            )
            
        except Exception as e:
            return ModelResponse(
                content="",
                model_used=model_config.name,
                provider=ModelProvider.ANTHROPIC,
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                quality_score=0.0,
                error=str(e)
            )
    
    def _calculate_cost(self, tokens: int, model_config: ModelConfig) -> float:
        return tokens * model_config.cost_per_token + model_config.cost_per_request
    
    async def get_available_models(self) -> List[str]:
        # Anthropic doesn't have a models endpoint, return known models
        return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    
    async def validate_model(self, model_id: str) -> bool:
        available_models = await self.get_available_models()
        return model_id in available_models

class ModelSelectionEngine:
    """Intelligent model selection engine"""
    
    def __init__(self):
        self.selection_history: List[Dict[str, Any]] = []
        self.performance_weights = {
            "quality": 0.4,
            "speed": 0.3,
            "cost": 0.2,
            "reliability": 0.1
        }
    
    def select_model(self, request: ModelRequest, available_models: List[ModelConfig], 
                    usage_stats: Dict[str, ModelUsageStats]) -> Optional[ModelConfig]:
        """Select the best model for the request"""
        
        # Filter models by capability
        capable_models = [
            model for model in available_models
            if request.task_type in model.capabilities and model.enabled
        ]
        
        if not capable_models:
            return None
        
        # Apply filters
        filtered_models = self._apply_filters(capable_models, request)
        
        if not filtered_models:
            return None
        
        # Score models
        scored_models = []
        for model in filtered_models:
            score = self._calculate_model_score(model, request, usage_stats.get(model.name))
            scored_models.append((model, score))
        
        # Sort by score (descending)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Record selection
        selected_model = scored_models[0][0]
        self.selection_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "request_type": request.task_type.value,
            "selected_model": selected_model.name,
            "score": scored_models[0][1],
            "alternatives": [(m.name, s) for m, s in scored_models[1:5]]  # Top 5 alternatives
        })
        
        return selected_model
    
    def _apply_filters(self, models: List[ModelConfig], request: ModelRequest) -> List[ModelConfig]:
        """Apply request filters to models"""
        filtered = models
        
        # Filter by preferred providers
        if request.preferred_providers:
            filtered = [m for m in filtered if m.provider in request.preferred_providers]
        
        # Filter by excluded models
        if request.excluded_models:
            filtered = [m for m in filtered if m.name not in request.excluded_models]
        
        # Filter by quality requirement
        filtered = [m for m in filtered if m.quality_score >= request.quality_requirement]
        
        # Filter by speed requirement
        filtered = [m for m in filtered if m.speed_score >= request.speed_requirement]
        
        # Filter by cost limit
        if request.cost_limit:
            max_tokens = request.max_tokens or 1000
            estimated_cost = max_tokens * m.cost_per_token + m.cost_per_request
            filtered = [m for m in filtered if estimated_cost <= request.cost_limit]
        
        # Filter by context window
        if request.context.get("context_length"):
            required_context = request.context["context_length"]
            filtered = [m for m in filtered if m.context_window >= required_context]
        
        # Filter by special requirements
        if request.functions:
            filtered = [m for m in filtered if m.supports_functions]
        
        if request.images:
            filtered = [m for m in filtered if m.supports_vision]
        
        if request.stream:
            filtered = [m for m in filtered if m.supports_streaming]
        
        return filtered
    
    def _calculate_model_score(self, model: ModelConfig, request: ModelRequest, 
                             stats: Optional[ModelUsageStats]) -> float:
        """Calculate score for model selection"""
        # Base scores from model configuration
        quality_score = model.quality_score
        speed_score = model.speed_score
        reliability_score = model.reliability_score
        
        # Cost score (inverse of cost, normalized)
        max_tokens = request.max_tokens or 1000
        estimated_cost = max_tokens * model.cost_per_token + model.cost_per_request
        cost_score = max(0, 1 - (estimated_cost / 10))  # Assuming $10 is expensive
        
        # Adjust scores based on usage statistics
        if stats:
            # Adjust reliability based on error rate
            reliability_score *= (1 - stats.error_rate)
            
            # Adjust speed based on average response time
            if stats.average_response_time > 0:
                # Normalize response time (assuming 30s is slow)
                time_factor = max(0, 1 - (stats.average_response_time / 30))
                speed_score *= time_factor
            
            # Boost score for recently successful models
            if stats.last_used and stats.successful_requests > 0:
                recency_hours = (datetime.utcnow() - stats.last_used).total_seconds() / 3600
                if recency_hours < 24:  # Recent usage bonus
                    reliability_score *= 1.1
        
        # Priority adjustments
        priority_multiplier = {
            "low": 0.8,
            "normal": 1.0,
            "high": 1.2,
            "critical": 1.5
        }.get(request.priority, 1.0)
        
        # Calculate weighted score
        total_score = (
            quality_score * self.performance_weights["quality"] +
            speed_score * self.performance_weights["speed"] +
            cost_score * self.performance_weights["cost"] +
            reliability_score * self.performance_weights["reliability"]
        ) * priority_multiplier
        
        return min(1.0, max(0.0, total_score))
    
    def update_weights(self, weights: Dict[str, float]):
        """Update performance weights for model selection"""
        total = sum(weights.values())
        if total > 0:
            self.performance_weights = {k: v/total for k, v in weights.items()}

class AIModelManager:
    """Main AI model management system"""
    
    def __init__(self, state_service, memory_service, performance_engine):
        self.state_service = state_service
        self.memory_service = memory_service
        self.performance_engine = performance_engine
        
        # Model configurations
        self.model_configs: Dict[str, ModelConfig] = {}
        self.providers: Dict[ModelProvider, BaseModelProvider] = {}
        
        # Usage tracking
        self.usage_stats: Dict[str, ModelUsageStats] = {}
        self.request_queue: deque = deque(maxlen=1000)
        
        # Model selection
        self.selection_engine = ModelSelectionEngine()
        
        # Caching
        self.response_cache: Dict[str, ModelResponse] = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialize the AI model manager"""
        await self._load_model_configurations()
        await self._initialize_providers()
        await self._load_usage_statistics()
        
    async def _load_model_configurations(self):
        """Load model configurations"""
        # Default model configurations
        default_configs = [
            ModelConfig(
                name="gpt-4-turbo",
                provider=ModelProvider.OPENAI,
                model_id="gpt-4-turbo-preview",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION, 
                            ModelCapability.ANALYSIS, ModelCapability.REASONING],
                tier=ModelTier.PREMIUM,
                max_tokens=4096,
                cost_per_token=0.00003,
                context_window=128000,
                supports_functions=True,
                supports_vision=True,
                quality_score=0.95,
                speed_score=0.7,
                reliability_score=0.95
            ),
            ModelConfig(
                name="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                model_id="gpt-3.5-turbo",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CONVERSATION],
                tier=ModelTier.STANDARD,
                max_tokens=4096,
                cost_per_token=0.000002,
                context_window=16385,
                supports_functions=True,
                quality_score=0.8,
                speed_score=0.9,
                reliability_score=0.9
            ),
            ModelConfig(
                name="claude-3-opus",
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-3-opus-20240229",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION,
                            ModelCapability.ANALYSIS, ModelCapability.REASONING],
                tier=ModelTier.PREMIUM,
                max_tokens=4096,
                cost_per_token=0.000075,
                context_window=200000,
                quality_score=0.98,
                speed_score=0.6,
                reliability_score=0.95
            ),
            ModelConfig(
                name="claude-3-sonnet",
                provider=ModelProvider.ANTHROPIC,
                model_id="claude-3-sonnet-20240229",
                capabilities=[ModelCapability.TEXT_GENERATION, ModelCapability.CODE_GENERATION,
                            ModelCapability.ANALYSIS],
                tier=ModelTier.STANDARD,
                max_tokens=4096,
                cost_per_token=0.000015,
                context_window=200000,
                quality_score=0.9,
                speed_score=0.8,
                reliability_score=0.95
            )
        ]
        
        for config in default_configs:
            self.model_configs[config.name] = config
            self.usage_stats[config.name] = ModelUsageStats(model_name=config.name)
    
    async def _initialize_providers(self):
        """Initialize AI model providers"""
        # Initialize OpenAI provider
        openai_config = {
            "api_key": "your-openai-api-key",  # Should come from environment
            "organization": "your-org-id"
        }
        self.providers[ModelProvider.OPENAI] = OpenAIProvider(openai_config)
        await self.providers[ModelProvider.OPENAI].initialize()
        
        # Initialize Anthropic provider
        anthropic_config = {
            "api_key": "your-anthropic-api-key"  # Should come from environment
        }
        self.providers[ModelProvider.ANTHROPIC] = AnthropicProvider(anthropic_config)
        await self.providers[ModelProvider.ANTHROPIC].initialize()
    
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using the best available model"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                if self._is_cache_valid(cached_response):
                    return cached_response
            
            # Select best model
            selected_model = self.selection_engine.select_model(
                request, 
                list(self.model_configs.values()),
                self.usage_stats
            )
            
            if not selected_model:
                raise Exception("No suitable model found for request")
            
            # Check rate limits
            if not await self._check_rate_limit(selected_model):
                # Try fallback models
                for fallback_name in selected_model.fallback_models:
                    fallback_model = self.model_configs.get(fallback_name)
                    if fallback_model and await self._check_rate_limit(fallback_model):
                        selected_model = fallback_model
                        break
                else:
                    raise Exception("Rate limit exceeded for all available models")
            
            # Get provider
            provider = self.providers.get(selected_model.provider)
            if not provider:
                raise Exception(f"Provider {selected_model.provider} not available")
            
            # Generate response
            response = await provider.generate_response(request, selected_model)
            
            # Update usage statistics
            await self._update_usage_stats(selected_model.name, response, time.time() - start_time)
            
            # Cache response if successful
            if not response.error:
                self.response_cache[cache_key] = response
            
            # Record performance metrics
            await self.performance_engine.record_metric(
                f"ai.model.{selected_model.name}.response_time",
                response.response_time,
                "timer",
                {"provider": selected_model.provider.value},
                "seconds"
            )
            
            await self.performance_engine.record_metric(
                f"ai.model.{selected_model.name}.tokens_used",
                response.tokens_used,
                "counter",
                {"provider": selected_model.provider.value},
                "tokens"
            )
            
            await self.performance_engine.record_metric(
                f"ai.model.{selected_model.name}.cost",
                response.cost,
                "counter",
                {"provider": selected_model.provider.value},
                "dollars"
            )
            
            return response
            
        except Exception as e:
            # Return error response
            return ModelResponse(
                content="",
                model_used="unknown",
                provider=ModelProvider.OPENAI,  # Default
                tokens_used=0,
                cost=0.0,
                response_time=time.time() - start_time,
                quality_score=0.0,
                error=str(e)
            )
    
    async def _update_usage_stats(self, model_name: str, response: ModelResponse, request_time: float):
        """Update usage statistics for a model"""
        stats = self.usage_stats[model_name]
        
        stats.total_requests += 1
        stats.last_used = datetime.utcnow()
        
        if response.error:
            stats.failed_requests += 1
        else:
            stats.successful_requests += 1
            stats.total_tokens += response.tokens_used
            stats.total_cost += response.cost
        
        # Update averages
        stats.error_rate = stats.failed_requests / stats.total_requests
        
        if stats.successful_requests > 0:
            # Update average response time
            total_time = stats.average_response_time * (stats.successful_requests - 1) + request_time
            stats.average_response_time = total_time / stats.successful_requests
            
            # Calculate performance score
            stats.performance_score = self._calculate_performance_score(stats)
        
        # Update usage trend
        stats.usage_trend.append(stats.total_requests)
        if len(stats.usage_trend) > 100:  # Keep last 100 data points
            stats.usage_trend.pop(0)
        
        # Store in state service
        await self.state_service.update_model_stats(model_name, {
            "total_requests": stats.total_requests,
            "success_rate": 1 - stats.error_rate,
            "average_response_time": stats.average_response_time,
            "total_cost": stats.total_cost,
            "performance_score": stats.performance_score
        })
    
    def _calculate_performance_score(self, stats: ModelUsageStats) -> float:
        """Calculate performance score for a model"""
        success_rate = 1 - stats.error_rate
        speed_score = max(0, 1 - (stats.average_response_time / 30))  # 30s baseline
        
        return (success_rate * 0.6 + speed_score * 0.4)
    
    def _generate_cache_key(self, request: ModelRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        
        # Create hash from request parameters
        cache_data = {
            "prompt": request.prompt,
            "task_type": request.task_type.value,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "context": request.context
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _is_cache_valid(self, response: ModelResponse) -> bool:
        """Check if cached response is still valid"""
        # Simple TTL-based validation
        # In practice, you might want more sophisticated cache validation
        return True  # Simplified for this example
    
    async def _check_rate_limit(self, model_config: ModelConfig) -> bool:
        """Check if model is within rate limits"""
        # Simplified rate limiting - in practice, implement proper rate limiting
        return True
    
    async def get_model_stats(self) -> Dict[str, Any]:
        """Get comprehensive model statistics"""
        stats_summary = {
            "total_models": len(self.model_configs),
            "active_models": len([m for m in self.model_configs.values() if m.enabled]),
            "total_requests": sum(s.total_requests for s in self.usage_stats.values()),
            "total_cost": sum(s.total_cost for s in self.usage_stats.values()),
            "average_success_rate": sum(1 - s.error_rate for s in self.usage_stats.values()) / len(self.usage_stats),
            "model_performance": {}
        }
        
        for name, stats in self.usage_stats.items():
            stats_summary["model_performance"][name] = {
                "requests": stats.total_requests,
                "success_rate": 1 - stats.error_rate,
                "average_response_time": stats.average_response_time,
                "total_cost": stats.total_cost,
                "performance_score": stats.performance_score,
                "last_used": stats.last_used.isoformat() if stats.last_used else None
            }
        
        return stats_summary
    
    async def add_model_config(self, config: ModelConfig):
        """Add new model configuration"""
        self.model_configs[config.name] = config
        self.usage_stats[config.name] = ModelUsageStats(model_name=config.name)
        
        # Validate model with provider
        provider = self.providers.get(config.provider)
        if provider:
            is_valid = await provider.validate_model(config.model_id)
            if not is_valid:
                config.enabled = False
    
    async def update_model_config(self, name: str, updates: Dict[str, Any]):
        """Update model configuration"""
        if name in self.model_configs:
            config = self.model_configs[name]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    async def disable_model(self, name: str):
        """Disable a model"""
        if name in self.model_configs:
            self.model_configs[name].enabled = False
    
    async def enable_model(self, name: str):
        """Enable a model"""
        if name in self.model_configs:
            self.model_configs[name].enabled = True
```

### Task 18.2: AI Capabilities Dashboard

**File**: `frontend/components/ai/ai-dashboard.tsx`

**AI Capabilities Dashboard**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Brain, Cpu, DollarSign, Clock, CheckCircle, XCircle, TrendingUp, Settings, Zap } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';

interface ModelConfig {
  name: string;
  provider: string;
  model_id: string;
  capabilities: string[];
  tier: string;
  max_tokens: number;
  cost_per_token: number;
  context_window: number;
  quality_score: number;
  speed_score: number;
  reliability_score: number;
  enabled: boolean;
}

interface ModelStats {
  requests: number;
  success_rate: number;
  average_response_time: number;
  total_cost: number;
  performance_score: number;
  last_used: string | null;
}

interface AIStats {
  total_models: number;
  active_models: number;
  total_requests: number;
  total_cost: number;
  average_success_rate: number;
  model_performance: Record<string, ModelStats>;
}

interface ModelRequest {
  task_type: string;
  prompt: string;
  max_tokens?: number;
  temperature: number;
  priority: string;
  cost_limit?: number;
  quality_requirement: number;
  speed_requirement: number;
}

export const AIDashboard: React.FC = () => {
  const [stats, setStats] = useState<AIStats | null>(null);
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [testRequest, setTestRequest] = useState<ModelRequest>({
    task_type: 'text_generation',
    prompt: '',
    temperature: 0.7,
    priority: 'normal',
    quality_requirement: 0.7,
    speed_requirement: 0.5
  });
  const [testResponse, setTestResponse] = useState<any>(null);
  const [testLoading, setTestLoading] = useState(false);

  useEffect(() => {
    fetchAIData();
    const interval = setInterval(fetchAIData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchAIData = async () => {
    try {
      // Fetch AI statistics
      const statsResponse = await fetch('/api/ai/stats');
      if (!statsResponse.ok) throw new Error('Failed to fetch AI stats');
      const statsData = await statsResponse.json();
      setStats(statsData.stats);

      // Fetch model configurations
      const modelsResponse = await fetch('/api/ai/models');
      if (!modelsResponse.ok) throw new Error('Failed to fetch models');
      const modelsData = await modelsResponse.json();
      setModels(modelsData.models);

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleTestRequest = async () => {
    if (!testRequest.prompt.trim()) {
      alert('Please enter a prompt');
      return;
    }

    setTestLoading(true);
    try {
      const response = await fetch('/api/ai/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(testRequest)
      });

      if (!response.ok) throw new Error('Failed to generate response');
      const data = await response.json();
      setTestResponse(data.response);
    } catch (err) {
      setTestResponse({ error: err instanceof Error ? err.message : 'Unknown error' });
    } finally {
      setTestLoading(false);
    }
  };

  const toggleModel = async (modelName: string, enabled: boolean) => {
    try {
      const response = await fetch(`/api/ai/models/${modelName}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ enabled })
      });

      if (!response.ok) throw new Error('Failed to update model');
      await fetchAIData();
    } catch (err) {
      alert(`Error updating model: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const getProviderColor = (provider: string) => {
    const colors = {
      openai: '#10B981',
      anthropic: '#F59E0B',
      google: '#3B82F6',
      local: '#8B5CF6',
      azure: '#06B6D4'
    };
    return colors[provider.toLowerCase()] || '#6B7280';
  };

  const getTierBadgeColor = (tier: string) => {
    const colors = {
      basic: 'bg-gray-500',
      standard: 'bg-blue-500',
      premium: 'bg-purple-500',
      enterprise: 'bg-gold-500'
    };
    return colors[tier.toLowerCase()] || 'bg-gray-500';
  };

  const formatCost = (cost: number) => {
    if (cost < 0.01) return `$${(cost * 1000).toFixed(2)}k`;
    return `$${cost.toFixed(4)}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2">Loading AI data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <XCircle className="h-4 w-4" />
        <AlertDescription>Error loading AI data: {error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold flex items-center">
          <Brain className="w-8 h-8 mr-3" />
          AI Model Dashboard
        </h1>
        <div className="flex items-center space-x-2">
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm">
                <Zap className="w-4 h-4 mr-2" />
                Test Model
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>Test AI Model</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="task_type">Task Type</Label>
                  <Select
                    value={testRequest.task_type}
                    onValueChange={(value) => setTestRequest({...testRequest, task_type: value})}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="text_generation">Text Generation</SelectItem>
                      <SelectItem value="code_generation">Code Generation</SelectItem>
                      <SelectItem value="analysis">Analysis</SelectItem>
                      <SelectItem value="reasoning">Reasoning</SelectItem>
                      <SelectItem value="conversation">Conversation</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div>
                  <Label htmlFor="prompt">Prompt</Label>
                  <textarea
                    id="prompt"
                    className="w-full p-3 border rounded-md h-32"
                    placeholder="Enter your prompt here..."
                    value={testRequest.prompt}
                    onChange={(e) => setTestRequest({...testRequest, prompt: e.target.value})}
                  />
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="temperature">Temperature</Label>
                    <Input
                      id="temperature"
                      type="number"
                      min="0"
                      max="2"
                      step="0.1"
                      value={testRequest.temperature}
                      onChange={(e) => setTestRequest({...testRequest, temperature: parseFloat(e.target.value)})}
                    />
                  </div>
                  
                  <div>
                    <Label htmlFor="priority">Priority</Label>
                    <Select
                      value={testRequest.priority}
                      onValueChange={(value) => setTestRequest({...testRequest, priority: value})}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="low">Low</SelectItem>
                        <SelectItem value="normal">Normal</SelectItem>
                        <SelectItem value="high">High</SelectItem>
                        <SelectItem value="critical">Critical</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                
                <Button 
                  onClick={handleTestRequest} 
                  disabled={testLoading || !testRequest.prompt.trim()}
                  className="w-full"
                >
                  {testLoading ? 'Generating...' : 'Generate Response'}
                </Button>
                
                {testResponse && (
                  <div className="mt-4">
                    <Label>Response</Label>
                    <div className="p-4 border rounded-md bg-gray-50 max-h-64 overflow-y-auto">
                      {testResponse.error ? (
                        <div className="text-red-600">
                          Error: {testResponse.error}
                        </div>
                      ) : (
                        <div>
                          <div className="mb-2 text-sm text-gray-600">
                            Model: {testResponse.model_used} | 
                            Tokens: {testResponse.tokens_used} | 
                            Cost: {formatCost(testResponse.cost)} | 
                            Time: {testResponse.response_time.toFixed(2)}s
                          </div>
                          <div className="whitespace-pre-wrap">
                            {testResponse.content}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </DialogContent>
          </Dialog>
          
          <Button onClick={fetchAIData} variant="outline" size="sm">
            Refresh
          </Button>
        </div>
      </div>

      {/* Overview Stats */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Models</CardTitle>
              <Brain className="w-4 h-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_models}</div>
              <div className="text-xs text-gray-600">
                {stats.active_models} active
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
              <Cpu className="w-4 h-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.total_requests.toLocaleString()}</div>
              <div className="text-xs text-gray-600">
                Success: {(stats.average_success_rate * 100).toFixed(1)}%
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Cost</CardTitle>
              <DollarSign className="w-4 h-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${stats.total_cost.toFixed(2)}</div>
              <div className="text-xs text-gray-600">
                This month
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Avg Response</CardTitle>
              <Clock className="w-4 h-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {Object.values(stats.model_performance).length > 0 ? 
                  (Object.values(stats.model_performance).reduce((sum, model) => sum + model.average_response_time, 0) / Object.values(stats.model_performance).length).toFixed(1) : '0.0'
                }s
              </div>
              <div className="text-xs text-gray-600">
                Average time
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Performance</CardTitle>
              <TrendingUp className="w-4 h-4" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {Object.values(stats.model_performance).length > 0 ? 
                  (Object.values(stats.model_performance).reduce((sum, model) => sum + model.performance_score, 0) / Object.values(stats.model_performance).length * 100).toFixed(0) : '0'
                }%
              </div>
              <div className="text-xs text-gray-600">
                Overall score
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="models" className="space-y-4">
        <TabsList>
          <TabsTrigger value="models">Model Management</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="usage">Usage Analytics</TabsTrigger>
          <TabsTrigger value="costs">Cost Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="models" className="space-y-4">
          {/* Model Management */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {models.map((model) => {
              const modelStats = stats?.model_performance[model.name];
              return (
                <Card key={model.name} className={`${!model.enabled ? 'opacity-60' : ''}`}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{model.name}</CardTitle>
                      <div className="flex items-center space-x-2">
                        <Badge className={getTierBadgeColor(model.tier)}>
                          {model.tier.toUpperCase()}
                        </Badge>
                        <Switch
                          checked={model.enabled}
                          onCheckedChange={(checked) => toggleModel(model.name, checked)}
                        />
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div 
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: getProviderColor(model.provider) }}
                      />
                      <span className="text-sm text-gray-600 capitalize">{model.provider}</span>
                      <span className="text-xs text-gray-500">({model.model_id})</span>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {/* Capabilities */}
                      <div>
                        <div className="text-sm font-medium mb-1">Capabilities</div>
                        <div className="flex flex-wrap gap-1">
                          {model.capabilities.slice(0, 3).map((cap) => (
                            <Badge key={cap} variant="outline" className="text-xs">
                              {cap.replace('_', ' ')}
                            </Badge>
                          ))}
                          {model.capabilities.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{model.capabilities.length - 3}
                            </Badge>
                          )}
                        </div>
                      </div>
                      
                      {/* Performance Scores */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>Quality</span>
                          <div className="flex items-center space-x-2">
                            <Progress value={model.quality_score * 100} className="w-16" />
                            <span className="text-xs">{(model.quality_score * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between text-sm">
                          <span>Speed</span>
                          <div className="flex items-center space-x-2">
                            <Progress value={model.speed_score * 100} className="w-16" />
                            <span className="text-xs">{(model.speed_score * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                        
                        <div className="flex items-center justify-between text-sm">
                          <span>Reliability</span>
                          <div className="flex items-center space-x-2">
                            <Progress value={model.reliability_score * 100} className="w-16" />
                            <span className="text-xs">{(model.reliability_score * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Usage Stats */}
                      {modelStats && (
                        <div className="pt-2 border-t">
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                              <span className="text-gray-600">Requests:</span>
                              <span className="ml-1 font-semibold">{modelStats.requests}</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Success:</span>
                              <span className="ml-1 font-semibold">{(modelStats.success_rate * 100).toFixed(1)}%</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Avg Time:</span>
                              <span className="ml-1 font-semibold">{modelStats.average_response_time.toFixed(1)}s</span>
                            </div>
                            <div>
                              <span className="text-gray-600">Cost:</span>
                              <span className="ml-1 font-semibold">{formatCost(modelStats.total_cost)}</span>
                            </div>
                          </div>
                        </div>
                      )}
                      
                      {/* Technical Details */}
                      <div className="pt-2 border-t text-xs text-gray-600">
                        <div>Context: {model.context_window.toLocaleString()} tokens</div>
                        <div>Cost: {formatCost(model.cost_per_token)}/token</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          {/* Performance Charts */}
          {stats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Model Performance Scores</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={Object.entries(stats.model_performance).map(([name, data]) => ({
                        name: name.replace(/^\w+\-/, ''),
                        performance: data.performance_score * 100,
                        success_rate: data.success_rate * 100
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="performance" fill="#3B82F6" name="Performance Score" />
                        <Bar dataKey="success_rate" fill="#10B981" name="Success Rate" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Response Time Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={Object.entries(stats.model_performance).map(([name, data]) => ({
                        name: name.replace(/^\w+\-/, ''),
                        response_time: data.average_response_time
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="response_time" fill="#F59E0B" name="Avg Response Time (s)" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="usage" className="space-y-4">
          {/* Usage Analytics */}
          {stats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Request Distribution</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={Object.entries(stats.model_performance).map(([name, data]) => ({
                            name: name.replace(/^\w+\-/, ''),
                            value: data.requests,
                            fill: getProviderColor(models.find(m => m.name === name)?.provider || 'unknown')
                          }))}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {Object.entries(stats.model_performance).map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={getProviderColor(models.find(m => m.name === entry[0])?.provider || 'unknown')} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Model Usage Trends</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(stats.model_performance).map(([name, data]) => (
                      <div key={name} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">{name.replace(/^\w+\-/, '')}</span>
                          <span className="text-sm text-gray-600">{data.requests} requests</span>
                        </div>
                        <Progress value={(data.requests / stats.total_requests) * 100} className="h-2" />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="costs" className="space-y-4">
          {/* Cost Analysis */}
          {stats && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Cost by Model</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={Object.entries(stats.model_performance).map(([name, data]) => ({
                        name: name.replace(/^\w+\-/, ''),
                        cost: data.total_cost
                      }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                        <YAxis />
                        <Tooltip formatter={(value) => [`$${value.toFixed(4)}`, 'Cost']} />
                        <Bar dataKey="cost" fill="#EF4444" name="Total Cost" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader>
                  <CardTitle>Cost Efficiency</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(stats.model_performance)
                      .sort(([,a], [,b]) => (a.total_cost / a.requests) - (b.total_cost / b.requests))
                      .map(([name, data]) => {
                        const costPerRequest = data.requests > 0 ? data.total_cost / data.requests : 0;
                        return (
                          <div key={name} className="flex items-center justify-between p-3 border rounded">
                            <div>
                              <div className="font-medium">{name.replace(/^\w+\-/, '')}</div>
                              <div className="text-sm text-gray-600">
                                {data.requests} requests
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="font-semibold">{formatCost(costPerRequest)}</div>
                              <div className="text-sm text-gray-600">per request</div>
                            </div>
                          </div>
                        );
                      })
                    }
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};
```