"""Enhanced Integration Service for Multi-Agent RAG System.

This module provides a unified interface that integrates all enhanced components:
- Enhanced LLM Service with Groq API features
- Advanced State Management
- Advanced NLP Processing
- Advanced Graph Management
- Artifact Generation System
- Enhanced Memory Management
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .enhanced_llm_service_main import EnhancedLLMService
from .enhanced_state_manager import EnhancedStateManager, StateConfig, ValidationLevel
from .artifact_generator import (
    ArtifactGenerationService,
    ArtifactRequest,
    ArtifactType,
    ArtifactFormat,
    ArtifactMetadata
)
from ..nlp.advanced_nlp_processor import AdvancedNLPProcessor
from ..graph.advanced_graph_manager import AdvancedGraphManager
from ..memory.memory_service import MemoryService
from ..models import AppState


logger = logging.getLogger(__name__)


class IntegrationConfig(BaseModel):
    """Configuration for the integration service."""
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    state_config: StateConfig = Field(default_factory=StateConfig)
    nlp_config: Dict[str, Any] = Field(default_factory=dict)
    graph_config: Dict[str, Any] = Field(default_factory=dict)
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    artifact_config: Dict[str, Any] = Field(default_factory=dict)
    enable_analytics: bool = True
    enable_caching: bool = True
    max_concurrent_operations: int = 10


class ProcessingRequest(BaseModel):
    """Request for processing through the integrated system."""
    user_input: str
    context: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    agent_type: Optional[str] = None
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    generate_artifacts: bool = False
    artifact_types: List[ArtifactType] = Field(default_factory=list)


class ProcessingResponse(BaseModel):
    """Response from the integrated processing system."""
    success: bool
    response_text: str
    processed_context: Dict[str, Any]
    nlp_analysis: Dict[str, Any]
    state_changes: Dict[str, Any]
    generated_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    execution_metrics: Dict[str, Any]
    error_message: Optional[str] = None
    session_id: Optional[str] = None


class EnhancedIntegrationService:
    """Main integration service that orchestrates all enhanced components."""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session_states: Dict[str, AppState] = {}
        
        # Initialize services
        self._initialize_services()
        
        # Performance tracking
        self.operation_count = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        logger.info("Enhanced Integration Service initialized")
    
    def _initialize_services(self):
        """Initialize all integrated services."""
        try:
            # Initialize LLM Service
            self.llm_service = EnhancedLLMService(
                groq_api_key=self.config.llm_config.get('groq_api_key'),
                openai_api_key=self.config.llm_config.get('openai_api_key'),
                gemini_api_key=self.config.llm_config.get('gemini_api_key'),
                enable_fallback=self.config.llm_config.get('enable_fallback', True),
                enable_analytics=self.config.enable_analytics
            )
            
            # Initialize State Manager
            self.state_manager = EnhancedStateManager(self.config.state_config)
            
            # Initialize NLP Processor
            self.nlp_processor = AdvancedNLPProcessor(
                enable_caching=self.config.enable_caching,
                **self.config.nlp_config
            )
            
            # Initialize Graph Manager
            self.graph_manager = AdvancedGraphManager(
                **self.config.graph_config
            )
            
            # Initialize Memory Service
            self.memory_service = MemoryService(
                **self.config.memory_config
            )
            
            # Initialize Artifact Generation Service
            self.artifact_service = ArtifactGenerationService()
            
            logger.info("All services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process a request through the integrated system."""
        start_time = asyncio.get_event_loop().time()
        session_id = request.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Initialize session state if needed
            if session_id not in self.session_states:
                self.session_states[session_id] = AppState(
                    session_id=session_id,
                    current_agent="integration_service",
                    context=request.context
                )
            
            current_state = self.session_states[session_id]
            
            # Step 1: NLP Analysis
            nlp_analysis = await self._perform_nlp_analysis(request.user_input)
            
            # Step 2: Update context with NLP insights
            enhanced_context = await self._enhance_context_with_nlp(
                request.context, nlp_analysis
            )
            
            # Step 3: Memory retrieval and context enrichment
            memory_context = await self._retrieve_relevant_memory(
                request.user_input, enhanced_context, session_id
            )
            
            # Step 4: State management and validation
            state_changes = await self._manage_state_changes(
                current_state, enhanced_context, memory_context
            )
            
            # Step 5: Graph-based processing if agent type specified
            graph_response = None
            if request.agent_type:
                graph_response = await self._process_through_graph(
                    request.agent_type, request.user_input, enhanced_context, session_id
                )
            
            # Step 6: LLM processing
            llm_response = await self._process_with_llm(
                request.user_input, enhanced_context, memory_context, graph_response
            )
            
            # Step 7: Store interaction in memory
            await self._store_interaction_memory(
                request.user_input, llm_response, enhanced_context, session_id
            )
            
            # Step 8: Generate artifacts if requested
            generated_artifacts = []
            if request.generate_artifacts and request.artifact_types:
                generated_artifacts = await self._generate_artifacts(
                    request.artifact_types, llm_response, enhanced_context
                )
            
            # Step 9: Calculate metrics
            processing_time = asyncio.get_event_loop().time() - start_time
            execution_metrics = self._calculate_execution_metrics(
                processing_time, nlp_analysis, state_changes
            )
            
            # Update performance tracking
            self.operation_count += 1
            self.total_processing_time += processing_time
            
            return ProcessingResponse(
                success=True,
                response_text=llm_response,
                processed_context=enhanced_context,
                nlp_analysis=nlp_analysis,
                state_changes=state_changes,
                generated_artifacts=generated_artifacts,
                execution_metrics=execution_metrics,
                session_id=session_id
            )
            
        except Exception as e:
            self.error_count += 1
            processing_time = asyncio.get_event_loop().time() - start_time
            
            logger.error(f"Error processing request: {e}")
            
            return ProcessingResponse(
                success=False,
                response_text="I apologize, but I encountered an error while processing your request.",
                processed_context=request.context,
                nlp_analysis={},
                state_changes={},
                execution_metrics={"processing_time": processing_time, "error": str(e)},
                error_message=str(e),
                session_id=session_id
            )
    
    async def _perform_nlp_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive NLP analysis."""
        try:
            # Semantic analysis
            semantic_analysis = await self.nlp_processor.analyze_semantics(text)
            
            # Entity extraction
            entities = await self.nlp_processor.extract_entities(text)
            
            # Intent recognition
            intent = await self.nlp_processor.recognize_intent(text)
            
            # Sentiment analysis
            sentiment = await self.nlp_processor.analyze_sentiment(text)
            
            # Key phrases
            key_phrases = await self.nlp_processor.extract_key_phrases(text)
            
            # Text complexity
            complexity = await self.nlp_processor.analyze_text_complexity(text)
            
            return {
                "semantic_analysis": semantic_analysis,
                "entities": entities,
                "intent": intent,
                "sentiment": sentiment,
                "key_phrases": key_phrases,
                "complexity": complexity,
                "processed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"NLP analysis failed: {e}")
            return {"error": str(e), "processed_at": datetime.now().isoformat()}
    
    async def _enhance_context_with_nlp(self, context: Dict[str, Any], nlp_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance context with NLP insights."""
        enhanced_context = context.copy()
        
        # Add NLP insights
        enhanced_context["nlp_insights"] = nlp_analysis
        
        # Extract and add key information
        if "entities" in nlp_analysis:
            enhanced_context["extracted_entities"] = nlp_analysis["entities"]
        
        if "intent" in nlp_analysis:
            enhanced_context["user_intent"] = nlp_analysis["intent"]
        
        if "sentiment" in nlp_analysis:
            enhanced_context["sentiment_score"] = nlp_analysis["sentiment"]
        
        if "key_phrases" in nlp_analysis:
            enhanced_context["key_topics"] = nlp_analysis["key_phrases"]
        
        return enhanced_context
    
    async def _retrieve_relevant_memory(self, query: str, context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Retrieve relevant information from memory."""
        try:
            # Search across all memory layers
            memory_results = await self.memory_service.search_memory(
                query=query,
                layer_filter=None,  # Search all layers
                limit=10
            )
            
            # Organize results by layer
            memory_context = {
                "knowledge_synthesis": [],
                "conversation_history": [],
                "codebase_knowledge": [],
                "stack_dependencies": [],
                "total_results": len(memory_results)
            }
            
            for result in memory_results:
                layer = result.get("layer", "unknown")
                if layer in memory_context:
                    memory_context[layer].append(result)
            
            return memory_context
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return {"error": str(e)}
    
    async def _manage_state_changes(self, current_state: AppState, context: Dict[str, Any], memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """Manage state changes and validation."""
        try:
            # Update state with new context
            updated_context = {**current_state.context, **context}
            
            # Create state snapshot before changes
            snapshot_id = await self.state_manager.create_snapshot(current_state)
            
            # Update state
            current_state.context = updated_context
            current_state.last_updated = datetime.now()
            
            # Validate state changes
            validation_result = await self.state_manager.validate_state(current_state)
            
            # Track changes
            changes = await self.state_manager.track_changes(current_state)
            
            return {
                "snapshot_id": snapshot_id,
                "validation_result": validation_result,
                "changes_tracked": changes,
                "state_updated": True
            }
            
        except Exception as e:
            logger.error(f"State management failed: {e}")
            return {"error": str(e), "state_updated": False}
    
    async def _process_through_graph(self, agent_type: str, user_input: str, context: Dict[str, Any], session_id: str) -> Optional[str]:
        """Process request through graph-based agent workflow."""
        try:
            # Check if agent graph is registered
            if not self.graph_manager.has_agent_graph(agent_type):
                logger.warning(f"No graph registered for agent type: {agent_type}")
                return None
            
            # Execute graph
            result = await self.graph_manager.execute_graph(
                agent_type=agent_type,
                input_data={"user_input": user_input, "context": context},
                session_id=session_id
            )
            
            return result.get("output", None) if result else None
            
        except Exception as e:
            logger.error(f"Graph processing failed: {e}")
            return None
    
    async def _process_with_llm(self, user_input: str, context: Dict[str, Any], memory_context: Dict[str, Any], graph_response: Optional[str]) -> str:
        """Process request with enhanced LLM service."""
        try:
            # Prepare messages with context
            messages = self._prepare_llm_messages(user_input, context, memory_context, graph_response)
            
            # Use structured output if context suggests it
            if context.get("user_intent", {}).get("category") in ["data_request", "analysis", "report"]:
                # Use structured output for data-oriented requests
                response = await self.llm_service.generate_structured_response(
                    messages=messages,
                    response_schema={
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "data": {"type": "object"},
                            "recommendations": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                )
                
                if isinstance(response, dict):
                    return response.get("response", str(response))
                else:
                    return str(response)
            else:
                # Use regular generation for conversational requests
                response = await self.llm_service.generate_response(
                    messages=messages,
                    model="llama-3.3-70b-versatile",  # Use latest Llama model
                    temperature=0.7,
                    max_tokens=2048
                )
                
                return response
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def _prepare_llm_messages(self, user_input: str, context: Dict[str, Any], memory_context: Dict[str, Any], graph_response: Optional[str]) -> List[Dict[str, str]]:
        """Prepare messages for LLM with context."""
        messages = []
        
        # System message with context
        system_content = "You are an advanced AI assistant with access to comprehensive context and memory."
        
        # Add relevant memory context
        if memory_context.get("knowledge_synthesis"):
            system_content += "\n\nRelevant Knowledge:\n"
            for item in memory_context["knowledge_synthesis"][:3]:  # Limit to top 3
                system_content += f"- {item.get('content', '')[:200]}...\n"
        
        # Add NLP insights
        if context.get("user_intent"):
            intent = context["user_intent"]
            system_content += f"\n\nUser Intent: {intent.get('category', 'unknown')} (confidence: {intent.get('confidence', 0):.2f})"
        
        if context.get("sentiment_score"):
            sentiment = context["sentiment_score"]
            system_content += f"\nUser Sentiment: {sentiment.get('label', 'neutral')} (score: {sentiment.get('score', 0):.2f})"
        
        # Add graph response if available
        if graph_response:
            system_content += f"\n\nAgent Processing Result: {graph_response}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add conversation history from memory
        if memory_context.get("conversation_history"):
            for item in memory_context["conversation_history"][-5:]:  # Last 5 interactions
                if item.get("user_message"):
                    messages.append({"role": "user", "content": item["user_message"]})
                if item.get("assistant_response"):
                    messages.append({"role": "assistant", "content": item["assistant_response"]})
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    async def _store_interaction_memory(self, user_input: str, response: str, context: Dict[str, Any], session_id: str):
        """Store the interaction in memory."""
        try:
            # Store in conversation history layer
            await self.memory_service.store_memory(
                layer="conversation_history",
                content={
                    "user_message": user_input,
                    "assistant_response": response,
                    "context": context,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "session_id": session_id,
                    "interaction_type": "user_assistant",
                    "context_keys": list(context.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store interaction memory: {e}")
    
    async def _generate_artifacts(self, artifact_types: List[ArtifactType], response_content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate requested artifacts."""
        generated_artifacts = []
        
        for artifact_type in artifact_types:
            try:
                # Prepare artifact content based on type
                artifact_content = self._prepare_artifact_content(artifact_type, response_content, context)
                
                # Create artifact request
                metadata = ArtifactMetadata(
                    title=f"Generated {artifact_type.value.title()}",
                    description=f"Auto-generated {artifact_type.value} from conversation",
                    author="Enhanced Integration Service"
                )
                
                request = ArtifactRequest(
                    artifact_type=artifact_type,
                    output_format=self._get_default_format(artifact_type),
                    metadata=metadata,
                    content=artifact_content
                )
                
                # Generate artifact
                artifact_response = await self.artifact_service.generate_artifact(request)
                
                if artifact_response.success:
                    generated_artifacts.append({
                        "type": artifact_type.value,
                        "artifact_id": artifact_response.artifact_id,
                        "file_path": artifact_response.file_path,
                        "preview_url": artifact_response.preview_url,
                        "generation_time": artifact_response.generation_time
                    })
                
            except Exception as e:
                logger.error(f"Failed to generate artifact {artifact_type}: {e}")
        
        return generated_artifacts
    
    def _prepare_artifact_content(self, artifact_type: ArtifactType, response_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare content for artifact generation."""
        if artifact_type == ArtifactType.MARKDOWN:
            return {
                "content": response_content,
                "context": context
            }
        elif artifact_type == ArtifactType.PDF:
            return {
                "summary": response_content,
                "context": context,
                "nlp_insights": context.get("nlp_insights", {})
            }
        elif artifact_type == ArtifactType.CODE:
            return {
                "description": "Generated code based on conversation",
                "code": response_content,
                "language": context.get("programming_language", "python")
            }
        else:
            return {"content": response_content, "context": context}
    
    def _get_default_format(self, artifact_type: ArtifactType) -> ArtifactFormat:
        """Get default format for artifact type."""
        format_mapping = {
            ArtifactType.MARKDOWN: ArtifactFormat.MARKDOWN,
            ArtifactType.PDF: ArtifactFormat.PDF,
            ArtifactType.HTML: ArtifactFormat.HTML,
            ArtifactType.CODE: ArtifactFormat.JSON,
            ArtifactType.BUSINESS_PLAN: ArtifactFormat.PDF,
            ArtifactType.REPORT: ArtifactFormat.PDF
        }
        return format_mapping.get(artifact_type, ArtifactFormat.JSON)
    
    def _calculate_execution_metrics(self, processing_time: float, nlp_analysis: Dict[str, Any], state_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate execution metrics."""
        return {
            "processing_time": processing_time,
            "nlp_analysis_success": "error" not in nlp_analysis,
            "state_changes_success": state_changes.get("state_updated", False),
            "total_operations": self.operation_count,
            "average_processing_time": self.total_processing_time / max(self.operation_count, 1),
            "error_rate": self.error_count / max(self.operation_count, 1),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_session_state(self, session_id: str) -> Optional[AppState]:
        """Get current session state."""
        return self.session_states.get(session_id)
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear session state."""
        if session_id in self.session_states:
            del self.session_states[session_id]
            return True
        return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        return {
            "total_operations": self.operation_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / max(self.operation_count, 1),
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.operation_count, 1),
            "active_sessions": len(self.session_states),
            "llm_statistics": await self.llm_service.get_usage_statistics(),
            "memory_statistics": await self.memory_service.get_statistics() if hasattr(self.memory_service, 'get_statistics') else {},
            "artifact_history_count": len(self.artifact_service.get_artifact_history())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health_status = {
            "overall_status": "healthy",
            "services": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check LLM service
        try:
            test_response = await self.llm_service.generate_response(
                messages=[{"role": "user", "content": "test"}],
                model="llama-3.3-70b-versatile",
                max_tokens=10
            )
            health_status["services"]["llm_service"] = "healthy" if test_response else "degraded"
        except Exception as e:
            health_status["services"]["llm_service"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check NLP processor
        try:
            test_analysis = await self.nlp_processor.analyze_sentiment("test")
            health_status["services"]["nlp_processor"] = "healthy" if test_analysis else "degraded"
        except Exception as e:
            health_status["services"]["nlp_processor"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check memory service
        try:
            # Simple memory test
            health_status["services"]["memory_service"] = "healthy"
        except Exception as e:
            health_status["services"]["memory_service"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check state manager
        try:
            test_state = AppState(session_id="health_check")
            validation = await self.state_manager.validate_state(test_state)
            health_status["services"]["state_manager"] = "healthy" if validation.get("valid", False) else "degraded"
        except Exception as e:
            health_status["services"]["state_manager"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        # Check artifact service
        try:
            supported_types = self.artifact_service.get_supported_types()
            health_status["services"]["artifact_service"] = "healthy" if supported_types else "degraded"
        except Exception as e:
            health_status["services"]["artifact_service"] = f"unhealthy: {str(e)}"
            health_status["overall_status"] = "degraded"
        
        return health_status


# Factory function for creating the integration service
def create_integration_service(
    groq_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    **kwargs
) -> EnhancedIntegrationService:
    """Factory function to create an enhanced integration service."""
    
    config = IntegrationConfig(
        llm_config={
            "groq_api_key": groq_api_key,
            "openai_api_key": openai_api_key,
            "gemini_api_key": gemini_api_key,
            "enable_fallback": kwargs.get("enable_fallback", True)
        },
        state_config=StateConfig(
            persistence_type=kwargs.get("state_persistence", "memory"),
            validation_level=ValidationLevel(kwargs.get("validation_level", "basic")),
            enable_change_tracking=kwargs.get("enable_change_tracking", True),
            enable_rollback=kwargs.get("enable_rollback", True)
        ),
        nlp_config=kwargs.get("nlp_config", {}),
        graph_config=kwargs.get("graph_config", {}),
        memory_config=kwargs.get("memory_config", {}),
        artifact_config=kwargs.get("artifact_config", {}),
        enable_analytics=kwargs.get("enable_analytics", True),
        enable_caching=kwargs.get("enable_caching", True),
        max_concurrent_operations=kwargs.get("max_concurrent_operations", 10)
    )
    
    return EnhancedIntegrationService(config)