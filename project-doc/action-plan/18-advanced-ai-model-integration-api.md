# Advanced AI Model Integration - API Implementation

## Task 18.3: AI Model API Endpoints

**File**: `app/api/ai.py`

**AI Model API Endpoints**:
```python
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from core.ai.model_manager import AIModelManager, ModelRequest, ModelConfig, ModelCapability, ModelProvider, ModelTier
from core.security.middleware import get_current_user, require_permission
from core.services.state_service import StateService
from core.services.memory_service import MemoryService
from core.performance.monitoring_engine import PerformanceMonitoringEngine

router = APIRouter(prefix="/ai", tags=["ai"])

# Request/Response Models
class GenerateRequest(BaseModel):
    task_type: str = Field(..., description="Type of task (text_generation, code_generation, etc.)")
    prompt: str = Field(..., description="Input prompt")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Top-p sampling")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")
    stream: bool = Field(False, description="Stream response")
    functions: List[Dict[str, Any]] = Field(default_factory=list, description="Function definitions")
    images: List[str] = Field(default_factory=list, description="Image URLs")
    priority: str = Field("normal", description="Request priority")
    cost_limit: Optional[float] = Field(None, description="Maximum cost limit")
    quality_requirement: float = Field(0.7, ge=0.0, le=1.0, description="Minimum quality requirement")
    speed_requirement: float = Field(0.5, ge=0.0, le=1.0, description="Minimum speed requirement")
    preferred_providers: List[str] = Field(default_factory=list, description="Preferred providers")
    excluded_models: List[str] = Field(default_factory=list, description="Excluded models")

class GenerateResponse(BaseModel):
    content: str
    model_used: str
    provider: str
    tokens_used: int
    cost: float
    response_time: float
    quality_score: float
    metadata: Dict[str, Any]
    function_calls: List[Dict[str, Any]]
    finish_reason: str
    error: Optional[str]

class ModelConfigRequest(BaseModel):
    name: str
    provider: str
    model_id: str
    capabilities: List[str]
    tier: str
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
    fallback_models: List[str] = Field(default_factory=list)
    custom_parameters: Dict[str, Any] = Field(default_factory=dict)

class ModelUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    quality_score: Optional[float] = None
    speed_score: Optional[float] = None
    reliability_score: Optional[float] = None
    cost_per_token: Optional[float] = None
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    fallback_models: Optional[List[str]] = None
    custom_parameters: Optional[Dict[str, Any]] = None

# Dependency injection
async def get_ai_manager() -> AIModelManager:
    state_service = StateService()
    memory_service = MemoryService()
    performance_engine = PerformanceMonitoringEngine(state_service)
    
    manager = AIModelManager(state_service, memory_service, performance_engine)
    await manager.initialize()
    return manager

@router.post("/generate", response_model=GenerateResponse)
async def generate_response(
    request: GenerateRequest,
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Generate AI response using the best available model"""
    try:
        # Convert request to ModelRequest
        model_request = ModelRequest(
            task_type=ModelCapability(request.task_type),
            prompt=request.prompt,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop_sequences=request.stop_sequences,
            stream=request.stream,
            functions=request.functions,
            images=request.images,
            priority=request.priority,
            cost_limit=request.cost_limit,
            quality_requirement=request.quality_requirement,
            speed_requirement=request.speed_requirement,
            preferred_providers=[ModelProvider(p) for p in request.preferred_providers],
            excluded_models=request.excluded_models
        )
        
        # Generate response
        response = await ai_manager.generate_response(model_request)
        
        return GenerateResponse(
            content=response.content,
            model_used=response.model_used,
            provider=response.provider.value,
            tokens_used=response.tokens_used,
            cost=response.cost,
            response_time=response.response_time,
            quality_score=response.quality_score,
            metadata=response.metadata,
            function_calls=response.function_calls,
            finish_reason=response.finish_reason,
            error=response.error
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@router.get("/stats")
async def get_ai_stats(
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Get AI model statistics"""
    try:
        stats = await ai_manager.get_model_stats()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get AI stats: {str(e)}")

@router.get("/models")
async def get_models(
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Get all model configurations"""
    try:
        models = []
        for config in ai_manager.model_configs.values():
            models.append({
                "name": config.name,
                "provider": config.provider.value,
                "model_id": config.model_id,
                "capabilities": [cap.value for cap in config.capabilities],
                "tier": config.tier.value,
                "max_tokens": config.max_tokens,
                "cost_per_token": config.cost_per_token,
                "cost_per_request": config.cost_per_request,
                "context_window": config.context_window,
                "supports_streaming": config.supports_streaming,
                "supports_functions": config.supports_functions,
                "supports_vision": config.supports_vision,
                "quality_score": config.quality_score,
                "speed_score": config.speed_score,
                "reliability_score": config.reliability_score,
                "enabled": config.enabled,
                "fallback_models": config.fallback_models
            })
        
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

@router.post("/models", dependencies=[Depends(require_permission("ai.manage"))])
async def add_model(
    request: ModelConfigRequest,
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Add new model configuration"""
    try:
        config = ModelConfig(
            name=request.name,
            provider=ModelProvider(request.provider),
            model_id=request.model_id,
            capabilities=[ModelCapability(cap) for cap in request.capabilities],
            tier=ModelTier(request.tier),
            max_tokens=request.max_tokens,
            cost_per_token=request.cost_per_token,
            cost_per_request=request.cost_per_request,
            context_window=request.context_window,
            supports_streaming=request.supports_streaming,
            supports_functions=request.supports_functions,
            supports_vision=request.supports_vision,
            rate_limit_rpm=request.rate_limit_rpm,
            rate_limit_tpm=request.rate_limit_tpm,
            quality_score=request.quality_score,
            speed_score=request.speed_score,
            reliability_score=request.reliability_score,
            enabled=request.enabled,
            fallback_models=request.fallback_models,
            custom_parameters=request.custom_parameters
        )
        
        await ai_manager.add_model_config(config)
        return {"message": "Model added successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add model: {str(e)}")

@router.patch("/models/{model_name}", dependencies=[Depends(require_permission("ai.manage"))])
async def update_model(
    model_name: str,
    request: ModelUpdateRequest,
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Update model configuration"""
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        await ai_manager.update_model_config(model_name, updates)
        return {"message": "Model updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model: {str(e)}")

@router.delete("/models/{model_name}", dependencies=[Depends(require_permission("ai.manage"))])
async def disable_model(
    model_name: str,
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Disable a model"""
    try:
        await ai_manager.disable_model(model_name)
        return {"message": "Model disabled successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disable model: {str(e)}")

@router.post("/models/{model_name}/enable", dependencies=[Depends(require_permission("ai.manage"))])
async def enable_model(
    model_name: str,
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Enable a model"""
    try:
        await ai_manager.enable_model(model_name)
        return {"message": "Model enabled successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enable model: {str(e)}")

@router.get("/models/{model_name}/stats")
async def get_model_stats(
    model_name: str,
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Get statistics for a specific model"""
    try:
        if model_name not in ai_manager.usage_stats:
            raise HTTPException(status_code=404, detail="Model not found")
        
        stats = ai_manager.usage_stats[model_name]
        return {
            "model_name": stats.model_name,
            "total_requests": stats.total_requests,
            "successful_requests": stats.successful_requests,
            "failed_requests": stats.failed_requests,
            "success_rate": 1 - stats.error_rate,
            "total_tokens": stats.total_tokens,
            "total_cost": stats.total_cost,
            "average_response_time": stats.average_response_time,
            "last_used": stats.last_used.isoformat() if stats.last_used else None,
            "performance_score": stats.performance_score,
            "usage_trend": stats.usage_trend
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model stats: {str(e)}")

@router.post("/test")
async def test_model(
    request: GenerateRequest,
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Test AI model with a simple request"""
    try:
        # Override some settings for testing
        test_request = ModelRequest(
            task_type=ModelCapability(request.task_type),
            prompt=request.prompt,
            context=request.context,
            max_tokens=min(request.max_tokens or 100, 100),  # Limit for testing
            temperature=request.temperature,
            priority="low",  # Low priority for testing
            quality_requirement=0.5,  # Lower requirement for testing
            speed_requirement=0.3
        )
        
        response = await ai_manager.generate_response(test_request)
        
        return {
            "success": not bool(response.error),
            "model_used": response.model_used,
            "provider": response.provider.value,
            "response_time": response.response_time,
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "content_preview": response.content[:200] + "..." if len(response.content) > 200 else response.content,
            "error": response.error
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@router.get("/providers")
async def get_providers(
    ai_manager: AIModelManager = Depends(get_ai_manager),
    current_user = Depends(get_current_user)
):
    """Get available AI providers"""
    try:
        providers = []
        for provider_enum, provider_instance in ai_manager.providers.items():
            available_models = await provider_instance.get_available_models()
            providers.append({
                "name": provider_enum.value,
                "available_models": available_models,
                "status": "active" if provider_instance.client else "inactive"
            })
        
        return {"providers": providers}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get providers: {str(e)}")

@router.get("/capabilities")
async def get_capabilities(
    current_user = Depends(get_current_user)
):
    """Get available AI capabilities"""
    capabilities = [
        {
            "name": cap.value,
            "description": {
                "text_generation": "Generate human-like text content",
                "code_generation": "Generate and complete code",
                "analysis": "Analyze and interpret data or content",
                "reasoning": "Perform logical reasoning and problem solving",
                "conversation": "Engage in natural conversations",
                "summarization": "Summarize long content",
                "translation": "Translate between languages",
                "embedding": "Generate text embeddings",
                "image_generation": "Generate images from text",
                "image_analysis": "Analyze and describe images"
            }.get(cap.value, "AI capability")
        }
        for cap in ModelCapability
    ]
    
    return {"capabilities": capabilities}
```

## Unit Testing

### Test Files

1. **Backend Tests**: `tests/unit/test_ai_model_manager.py`
2. **Frontend Tests**: `tests/frontend/ai-dashboard.test.tsx`
3. **Integration Tests**: `tests/integration/test_ai_integration.py`

### Test Scenarios

**Backend Testing**:
- Model selection algorithm accuracy
- Provider integration functionality
- Usage statistics tracking
- Rate limiting and fallback mechanisms
- Cost calculation accuracy
- Performance monitoring integration

**Frontend Testing**:
- AI dashboard rendering and data display
- Model management interface functionality
- Real-time statistics updates
- Test request interface
- Model configuration forms
- Performance charts and analytics

**Integration Testing**:
- End-to-end AI request processing
- Multi-provider failover scenarios
- Performance monitoring integration
- Cost tracking accuracy
- Real-time dashboard updates

## Human Testing Scenarios

1. **Model Performance Comparison**: Test different models with the same prompt and compare results
2. **Cost Optimization**: Configure cost limits and verify model selection
3. **Capability Testing**: Test different AI capabilities (code generation, analysis, etc.)
4. **Provider Failover**: Disable primary models and test fallback mechanisms
5. **Dashboard Monitoring**: Monitor real-time performance and usage statistics

## Validation Criteria

### Backend Validation
- [ ] AI model manager initializes successfully
- [ ] Model selection algorithm works correctly
- [ ] All AI providers integrate properly
- [ ] Usage statistics are tracked accurately
- [ ] Performance monitoring is integrated
- [ ] Cost calculations are correct
- [ ] Rate limiting and fallbacks work
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] AI dashboard loads and displays data
- [ ] Model management interface works
- [ ] Real-time updates function properly
- [ ] Test interface generates responses
- [ ] Performance charts render correctly
- [ ] Cost analysis displays accurately
- [ ] Model configuration forms work

### Integration Validation
- [ ] End-to-end AI requests work
- [ ] Multi-provider scenarios function
- [ ] Performance monitoring integration
- [ ] Real-time dashboard updates
- [ ] Cost tracking accuracy
- [ ] Security and authentication
- [ ] Error handling and recovery

## Next Steps

After completing the Advanced AI Model Integration, proceed to:
**19-intelligent-code-analysis-framework.md**

This will implement intelligent code analysis capabilities, including code quality assessment, security vulnerability detection, performance optimization suggestions, and automated code review features.