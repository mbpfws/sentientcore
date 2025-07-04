"""Comprehensive test suite for the Enhanced Multi-Agent RAG System.

This module provides end-to-end testing for all enhanced components:
- Enhanced LLM Service with Groq API features
- Advanced State Management
- Advanced NLP Processing
- Advanced Graph Management
- Artifact Generation System
- Enhanced Memory Management
- Integration Service
"""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

import pytest

# Import the enhanced components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.services.enhanced_llm_service_main import EnhancedLLMService
from core.services.enhanced_state_manager import EnhancedStateManager, StateConfig, ValidationLevel
from core.services.artifact_generator import (
    ArtifactGenerationService,
    ArtifactRequest,
    ArtifactType,
    ArtifactFormat,
    ArtifactMetadata
)
from core.nlp.advanced_nlp_processor import AdvancedNLPProcessor
from core.graph.advanced_graph_manager import AdvancedGraphManager
from core.services.enhanced_integration_service import (
    EnhancedIntegrationService,
    IntegrationConfig,
    ProcessingRequest,
    create_integration_service
)
from core.models import AppState


class TestEnhancedLLMService(unittest.TestCase):
    """Test cases for Enhanced LLM Service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_service = EnhancedLLMService(
            groq_api_key="test_key",
            enable_fallback=True,
            enable_analytics=True
        )
    
    @patch('core.services.enhanced_llm_service_main.Groq')
    async def test_groq_provider_initialization(self, mock_groq):
        """Test Groq provider initialization."""
        # Mock Groq client
        mock_client = Mock()
        mock_groq.return_value = mock_client
        
        # Test provider creation
        provider = self.llm_service.providers['groq']
        self.assertIsNotNone(provider)
        self.assertEqual(provider.name, 'groq')
    
    @patch('core.services.enhanced_llm_service_main.Groq')
    async def test_structured_output_generation(self, mock_groq):
        """Test structured output generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"response": "test", "confidence": 0.9}'
        
        mock_client = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_groq.return_value = mock_client
        
        # Test structured output
        schema = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "confidence": {"type": "number"}
            }
        }
        
        result = await self.llm_service.generate_structured_response(
            messages=[{"role": "user", "content": "test"}],
            response_schema=schema
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn("response", result)
        self.assertIn("confidence", result)
    
    async def test_model_selection_logic(self):
        """Test intelligent model selection."""
        # Test different request types
        test_cases = [
            ("simple question", "llama-3.1-8b-instant"),
            ("complex analysis with multiple steps", "llama-3.3-70b-versatile"),
            ("code generation task", "llama-3.3-70b-versatile")
        ]
        
        for request, expected_model in test_cases:
            selected_model = self.llm_service._select_optimal_model(request)
            self.assertIn(expected_model.split('-')[0], selected_model)
    
    async def test_fallback_mechanism(self):
        """Test provider fallback mechanism."""
        with patch.object(self.llm_service.providers['groq'], 'generate_text', side_effect=Exception("API Error")):
            with patch.object(self.llm_service.providers['openai'], 'generate_text', return_value="fallback response"):
                response = await self.llm_service.generate_with_fallback(
                    messages=[{"role": "user", "content": "test"}]
                )
                self.assertEqual(response, "fallback response")


class TestEnhancedStateManager(unittest.TestCase):
    """Test cases for Enhanced State Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = StateConfig(
            persistence_type="memory",
            validation_level=ValidationLevel.BASIC,
            enable_change_tracking=True,
            enable_rollback=True
        )
        self.state_manager = EnhancedStateManager(self.config)
    
    async def test_state_persistence(self):
        """Test state persistence mechanisms."""
        # Create test state
        test_state = AppState(
            session_id="test_session",
            current_agent="test_agent",
            context={"test_key": "test_value"}
        )
        
        # Test memory persistence
        await self.state_manager.persist_state(test_state)
        retrieved_state = await self.state_manager.load_state("test_session")
        
        self.assertEqual(retrieved_state.session_id, test_state.session_id)
        self.assertEqual(retrieved_state.context, test_state.context)
    
    async def test_state_validation(self):
        """Test state validation."""
        # Valid state
        valid_state = AppState(
            session_id="valid_session",
            current_agent="test_agent"
        )
        
        validation_result = await self.state_manager.validate_state(valid_state)
        self.assertTrue(validation_result["valid"])
        
        # Invalid state (missing session_id)
        invalid_state = AppState(current_agent="test_agent")
        invalid_state.session_id = None
        
        validation_result = await self.state_manager.validate_state(invalid_state)
        self.assertFalse(validation_result["valid"])
    
    async def test_change_tracking_and_rollback(self):
        """Test change tracking and rollback functionality."""
        # Create initial state
        initial_state = AppState(
            session_id="rollback_test",
            context={"counter": 0}
        )
        
        # Create snapshot
        snapshot_id = await self.state_manager.create_snapshot(initial_state)
        
        # Modify state
        initial_state.context["counter"] = 5
        await self.state_manager.track_changes(initial_state)
        
        # Rollback to snapshot
        rolled_back_state = await self.state_manager.rollback_to_snapshot(snapshot_id)
        
        self.assertEqual(rolled_back_state.context["counter"], 0)


class TestAdvancedNLPProcessor(unittest.TestCase):
    """Test cases for Advanced NLP Processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nlp_processor = AdvancedNLPProcessor(enable_caching=True)
    
    @patch('spacy.load')
    async def test_entity_extraction(self, mock_spacy_load):
        """Test entity extraction functionality."""
        # Mock spaCy model
        mock_doc = Mock()
        mock_entity = Mock()
        mock_entity.text = "Apple"
        mock_entity.label_ = "ORG"
        mock_entity.start_char = 0
        mock_entity.end_char = 5
        mock_doc.ents = [mock_entity]
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        # Test entity extraction
        entities = await self.nlp_processor.extract_entities("Apple is a technology company")
        
        self.assertIsInstance(entities, list)
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["text"], "Apple")
        self.assertEqual(entities[0]["label"], "ORG")
    
    @patch('transformers.pipeline')
    async def test_sentiment_analysis(self, mock_pipeline):
        """Test sentiment analysis."""
        # Mock transformer pipeline
        mock_classifier = Mock()
        mock_classifier.return_value = [{"label": "POSITIVE", "score": 0.9}]
        mock_pipeline.return_value = mock_classifier
        
        # Test sentiment analysis
        sentiment = await self.nlp_processor.analyze_sentiment("I love this product!")
        
        self.assertIsInstance(sentiment, dict)
        self.assertEqual(sentiment["label"], "POSITIVE")
        self.assertEqual(sentiment["score"], 0.9)
    
    async def test_intent_recognition(self):
        """Test intent recognition."""
        test_cases = [
            ("What is the weather today?", "question"),
            ("Please book a flight", "request"),
            ("I want to cancel my order", "request"),
            ("Thank you for your help", "acknowledgment")
        ]
        
        for text, expected_category in test_cases:
            intent = await self.nlp_processor.recognize_intent(text)
            self.assertIsInstance(intent, dict)
            self.assertIn("category", intent)
            self.assertIn("confidence", intent)
    
    async def test_semantic_similarity(self):
        """Test semantic similarity calculation."""
        text1 = "The cat sat on the mat"
        text2 = "A feline rested on the rug"
        text3 = "The weather is nice today"
        
        # Similar texts should have high similarity
        similarity_high = await self.nlp_processor.calculate_semantic_similarity(text1, text2)
        self.assertGreater(similarity_high, 0.5)
        
        # Dissimilar texts should have low similarity
        similarity_low = await self.nlp_processor.calculate_semantic_similarity(text1, text3)
        self.assertLess(similarity_low, 0.5)


class TestArtifactGenerationService(unittest.TestCase):
    """Test cases for Artifact Generation Service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.artifact_service = ArtifactGenerationService()
    
    async def test_pdf_generation(self):
        """Test PDF artifact generation."""
        metadata = ArtifactMetadata(
            title="Test Report",
            description="A test PDF report",
            author="Test Suite"
        )
        
        request = ArtifactRequest(
            artifact_type=ArtifactType.PDF,
            output_format=ArtifactFormat.PDF,
            metadata=metadata,
            content={
                "summary": "This is a test report",
                "sections": [
                    {"title": "Introduction", "content": "Test content"},
                    {"title": "Conclusion", "content": "Test conclusion"}
                ]
            }
        )
        
        response = await self.artifact_service.generate_artifact(request)
        
        self.assertTrue(response.success)
        self.assertIsNotNone(response.artifact_id)
        self.assertTrue(response.file_path.endswith('.pdf'))
    
    async def test_markdown_generation(self):
        """Test Markdown artifact generation."""
        metadata = ArtifactMetadata(
            title="Test Documentation",
            description="Test markdown documentation",
            author="Test Suite"
        )
        
        request = ArtifactRequest(
            artifact_type=ArtifactType.MARKDOWN,
            output_format=ArtifactFormat.MARKDOWN,
            metadata=metadata,
            content={
                "content": "# Test Document\n\nThis is test content.",
                "context": {"project": "test_project"}
            }
        )
        
        response = await self.artifact_service.generate_artifact(request)
        
        self.assertTrue(response.success)
        self.assertIsNotNone(response.artifact_id)
        self.assertTrue(response.file_path.endswith('.md'))
    
    async def test_code_generation(self):
        """Test code artifact generation."""
        metadata = ArtifactMetadata(
            title="Test Code",
            description="Generated test code",
            author="Test Suite"
        )
        
        request = ArtifactRequest(
            artifact_type=ArtifactType.CODE,
            output_format=ArtifactFormat.JSON,
            metadata=metadata,
            content={
                "description": "A simple Python function",
                "code": "def hello_world():\n    return 'Hello, World!'",
                "language": "python"
            }
        )
        
        response = await self.artifact_service.generate_artifact(request)
        
        self.assertTrue(response.success)
        self.assertIsNotNone(response.artifact_id)
    
    def test_supported_types_and_formats(self):
        """Test supported artifact types and formats."""
        supported_types = self.artifact_service.get_supported_types()
        supported_formats = self.artifact_service.get_supported_formats()
        
        self.assertIn(ArtifactType.PDF, supported_types)
        self.assertIn(ArtifactType.MARKDOWN, supported_types)
        self.assertIn(ArtifactType.CODE, supported_types)
        
        self.assertIn(ArtifactFormat.PDF, supported_formats)
        self.assertIn(ArtifactFormat.MARKDOWN, supported_formats)
        self.assertIn(ArtifactFormat.JSON, supported_formats)


class TestAdvancedGraphManager(unittest.TestCase):
    """Test cases for Advanced Graph Manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.graph_manager = AdvancedGraphManager()
    
    async def test_graph_registration(self):
        """Test agent graph registration."""
        # Mock LangGraph instance
        mock_graph = Mock()
        
        # Register graph
        self.graph_manager.register_agent_graph("test_agent", mock_graph)
        
        # Verify registration
        self.assertTrue(self.graph_manager.has_agent_graph("test_agent"))
        self.assertEqual(self.graph_manager.agent_graphs["test_agent"], mock_graph)
    
    async def test_dynamic_routing(self):
        """Test dynamic routing functionality."""
        # Test intent-based routing
        routing_result = await self.graph_manager.router.route_by_intent(
            "I need help with coding",
            available_agents=["coding_agent", "general_agent"]
        )
        
        self.assertIsInstance(routing_result, dict)
        self.assertIn("selected_agent", routing_result)
        self.assertIn("confidence", routing_result)
    
    async def test_state_synchronization(self):
        """Test state synchronization between graph and state manager."""
        # Create test state
        test_state = AppState(
            session_id="sync_test",
            context={"test_data": "value"}
        )
        
        # Test synchronization
        sync_result = await self.graph_manager.state_manager.sync_with_external_state(test_state)
        
        self.assertTrue(sync_result["success"])
    
    async def test_graph_execution_modes(self):
        """Test different graph execution modes."""
        # Mock graph execution
        with patch.object(self.graph_manager, '_execute_graph_sync', return_value={"output": "sync_result"}):
            sync_result = await self.graph_manager.execute_graph(
                agent_type="test_agent",
                input_data={"input": "test"},
                execution_mode="sync"
            )
            self.assertEqual(sync_result["output"], "sync_result")
        
        with patch.object(self.graph_manager, '_execute_graph_async', return_value={"output": "async_result"}):
            async_result = await self.graph_manager.execute_graph(
                agent_type="test_agent",
                input_data={"input": "test"},
                execution_mode="async"
            )
            self.assertEqual(async_result["output"], "async_result")


class TestEnhancedIntegrationService(unittest.TestCase):
    """Test cases for Enhanced Integration Service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration_service = create_integration_service(
            groq_api_key="test_key",
            enable_analytics=True,
            enable_caching=True
        )
    
    async def test_end_to_end_processing(self):
        """Test end-to-end request processing."""
        # Create test request
        request = ProcessingRequest(
            user_input="What is the weather like today?",
            context={"location": "New York"},
            session_id="test_session",
            generate_artifacts=False
        )
        
        # Mock external dependencies
        with patch.object(self.integration_service.nlp_processor, 'analyze_semantics', return_value={"topics": ["weather"]}):
            with patch.object(self.integration_service.nlp_processor, 'extract_entities', return_value=[{"text": "today", "label": "DATE"}]):
                with patch.object(self.integration_service.nlp_processor, 'recognize_intent', return_value={"category": "question", "confidence": 0.9}):
                    with patch.object(self.integration_service.llm_service, 'generate_response', return_value="The weather is sunny today."):
                        
                        response = await self.integration_service.process_request(request)
                        
                        self.assertTrue(response.success)
                        self.assertEqual(response.response_text, "The weather is sunny today.")
                        self.assertIn("nlp_analysis", response.nlp_analysis)
                        self.assertEqual(response.session_id, "test_session")
    
    async def test_artifact_generation_integration(self):
        """Test artifact generation integration."""
        request = ProcessingRequest(
            user_input="Generate a report about AI trends",
            context={"topic": "AI trends"},
            generate_artifacts=True,
            artifact_types=[ArtifactType.PDF, ArtifactType.MARKDOWN]
        )
        
        # Mock dependencies
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value="AI trends report content"):
            with patch.object(self.integration_service.artifact_service, 'generate_artifact') as mock_generate:
                mock_generate.return_value = Mock(
                    success=True,
                    artifact_id="test_id",
                    file_path="/path/to/artifact.pdf",
                    preview_url="http://preview.url",
                    generation_time=1.5
                )
                
                response = await self.integration_service.process_request(request)
                
                self.assertTrue(response.success)
                self.assertEqual(len(response.generated_artifacts), 2)  # PDF and Markdown
                self.assertEqual(mock_generate.call_count, 2)
    
    async def test_session_management(self):
        """Test session state management."""
        session_id = "session_management_test"
        
        # Create initial request
        request1 = ProcessingRequest(
            user_input="Hello",
            session_id=session_id,
            context={"user_name": "Alice"}
        )
        
        # Process first request
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value="Hello Alice!"):
            response1 = await self.integration_service.process_request(request1)
        
        # Verify session state was created
        session_state = await self.integration_service.get_session_state(session_id)
        self.assertIsNotNone(session_state)
        self.assertEqual(session_state.session_id, session_id)
        
        # Create follow-up request
        request2 = ProcessingRequest(
            user_input="What's my name?",
            session_id=session_id
        )
        
        # Process second request
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value="Your name is Alice."):
            response2 = await self.integration_service.process_request(request2)
        
        # Verify context continuity
        self.assertTrue(response2.success)
        self.assertIn("user_name", response2.processed_context)
    
    async def test_performance_metrics(self):
        """Test performance metrics collection."""
        # Process a few requests to generate metrics
        for i in range(3):
            request = ProcessingRequest(
                user_input=f"Test request {i}",
                session_id=f"metrics_test_{i}"
            )
            
            with patch.object(self.integration_service.llm_service, 'generate_response', return_value=f"Response {i}"):
                await self.integration_service.process_request(request)
        
        # Get performance metrics
        metrics = await self.integration_service.get_performance_metrics()
        
        self.assertGreaterEqual(metrics["total_operations"], 3)
        self.assertGreater(metrics["total_processing_time"], 0)
        self.assertGreaterEqual(metrics["average_processing_time"], 0)
        self.assertEqual(metrics["error_rate"], 0)  # No errors in successful tests
    
    async def test_health_check(self):
        """Test system health check."""
        # Mock successful health checks for all services
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value="test"):
            with patch.object(self.integration_service.nlp_processor, 'analyze_sentiment', return_value={"label": "neutral"}):
                
                health_status = await self.integration_service.health_check()
                
                self.assertEqual(health_status["overall_status"], "healthy")
                self.assertIn("services", health_status)
                self.assertIn("llm_service", health_status["services"])
                self.assertIn("nlp_processor", health_status["services"])
    
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        request = ProcessingRequest(
            user_input="Test error handling",
            session_id="error_test"
        )
        
        # Simulate LLM service failure
        with patch.object(self.integration_service.llm_service, 'generate_response', side_effect=Exception("LLM Error")):
            response = await self.integration_service.process_request(request)
            
            self.assertFalse(response.success)
            self.assertIsNotNone(response.error_message)
            self.assertIn("error", response.response_text.lower())


class TestIntegrationPerformance(unittest.TestCase):
    """Performance tests for the integration system."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.integration_service = create_integration_service(
            groq_api_key="test_key",
            enable_analytics=True,
            max_concurrent_operations=20
        )
    
    async def test_concurrent_request_processing(self):
        """Test concurrent request processing performance."""
        # Create multiple concurrent requests
        requests = [
            ProcessingRequest(
                user_input=f"Concurrent test request {i}",
                session_id=f"concurrent_session_{i}"
            )
            for i in range(10)
        ]
        
        # Mock LLM responses
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value="Concurrent response"):
            
            start_time = asyncio.get_event_loop().time()
            
            # Process requests concurrently
            responses = await asyncio.gather(*[
                self.integration_service.process_request(request)
                for request in requests
            ])
            
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            # Verify all requests were processed successfully
            self.assertEqual(len(responses), 10)
            for response in responses:
                self.assertTrue(response.success)
            
            # Performance should be reasonable (less than 10 seconds for 10 concurrent requests)
            self.assertLess(processing_time, 10.0)
    
    async def test_memory_usage_stability(self):
        """Test memory usage stability under load."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process many requests to test memory stability
        for batch in range(5):
            requests = [
                ProcessingRequest(
                    user_input=f"Memory test batch {batch} request {i}",
                    session_id=f"memory_test_{batch}_{i}"
                )
                for i in range(20)
            ]
            
            with patch.object(self.integration_service.llm_service, 'generate_response', return_value="Memory test response"):
                await asyncio.gather(*[
                    self.integration_service.process_request(request)
                    for request in requests
                ])
            
            # Force garbage collection
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        self.assertLess(memory_increase, 100 * 1024 * 1024)


class TestEndToEndScenarios(unittest.TestCase):
    """End-to-end scenario tests."""
    
    def setUp(self):
        """Set up end-to-end test fixtures."""
        self.integration_service = create_integration_service(
            groq_api_key="test_key",
            enable_analytics=True,
            enable_caching=True
        )
    
    async def test_research_and_report_generation_scenario(self):
        """Test complete research and report generation scenario."""
        # Step 1: Research request
        research_request = ProcessingRequest(
            user_input="Research the latest trends in artificial intelligence",
            context={"research_depth": "comprehensive"},
            session_id="research_scenario",
            generate_artifacts=True,
            artifact_types=[ArtifactType.REPORT, ArtifactType.MARKDOWN]
        )
        
        # Mock research response
        research_response_text = """
        Based on current research, key AI trends include:
        1. Large Language Models (LLMs) advancement
        2. Multimodal AI systems
        3. AI safety and alignment
        4. Edge AI deployment
        5. AI democratization through no-code platforms
        """
        
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value=research_response_text):
            with patch.object(self.integration_service.artifact_service, 'generate_artifact') as mock_generate:
                mock_generate.return_value = Mock(
                    success=True,
                    artifact_id="research_report_123",
                    file_path="/path/to/ai_trends_report.pdf",
                    preview_url="http://preview.url/report",
                    generation_time=2.5
                )
                
                research_response = await self.integration_service.process_request(research_request)
        
        # Verify research processing
        self.assertTrue(research_response.success)
        self.assertIn("AI trends", research_response.response_text)
        self.assertEqual(len(research_response.generated_artifacts), 2)
        
        # Step 2: Follow-up question
        followup_request = ProcessingRequest(
            user_input="Can you elaborate on the AI safety trend?",
            session_id="research_scenario"  # Same session
        )
        
        followup_response_text = """
        AI safety focuses on ensuring AI systems behave as intended and remain beneficial.
        Key areas include alignment research, robustness testing, and interpretability.
        """
        
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value=followup_response_text):
            followup_response = await self.integration_service.process_request(followup_request)
        
        # Verify follow-up processing with context continuity
        self.assertTrue(followup_response.success)
        self.assertIn("AI safety", followup_response.response_text)
        self.assertEqual(followup_response.session_id, "research_scenario")
    
    async def test_code_generation_and_documentation_scenario(self):
        """Test code generation and documentation scenario."""
        # Step 1: Code generation request
        code_request = ProcessingRequest(
            user_input="Create a Python function to calculate fibonacci numbers with memoization",
            context={"programming_language": "python", "optimization": "memoization"},
            session_id="coding_scenario",
            generate_artifacts=True,
            artifact_types=[ArtifactType.CODE]
        )
        
        code_response_text = """
        Here's a Python function with memoization for fibonacci calculation:
        
        ```python
        def fibonacci_memo(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 1:
                return n
            memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)
            return memo[n]
        ```
        """
        
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value=code_response_text):
            with patch.object(self.integration_service.artifact_service, 'generate_artifact') as mock_generate:
                mock_generate.return_value = Mock(
                    success=True,
                    artifact_id="fibonacci_code_456",
                    file_path="/path/to/fibonacci.py",
                    preview_url="http://preview.url/code",
                    generation_time=1.2
                )
                
                code_response = await self.integration_service.process_request(code_request)
        
        # Verify code generation
        self.assertTrue(code_response.success)
        self.assertIn("fibonacci", code_response.response_text.lower())
        self.assertIn("memoization", code_response.response_text.lower())
        self.assertEqual(len(code_response.generated_artifacts), 1)
        
        # Step 2: Documentation request
        doc_request = ProcessingRequest(
            user_input="Generate documentation for the fibonacci function",
            session_id="coding_scenario",  # Same session
            generate_artifacts=True,
            artifact_types=[ArtifactType.MARKDOWN]
        )
        
        doc_response_text = """
        # Fibonacci Function Documentation
        
        ## Overview
        The `fibonacci_memo` function calculates fibonacci numbers using memoization for optimization.
        
        ## Parameters
        - `n`: The position in the fibonacci sequence
        - `memo`: Dictionary for memoization (default: {})
        
        ## Returns
        The fibonacci number at position n
        
        ## Time Complexity
        O(n) with memoization
        """
        
        with patch.object(self.integration_service.llm_service, 'generate_response', return_value=doc_response_text):
            with patch.object(self.integration_service.artifact_service, 'generate_artifact') as mock_generate:
                mock_generate.return_value = Mock(
                    success=True,
                    artifact_id="fibonacci_docs_789",
                    file_path="/path/to/fibonacci_docs.md",
                    preview_url="http://preview.url/docs",
                    generation_time=0.8
                )
                
                doc_response = await self.integration_service.process_request(doc_request)
        
        # Verify documentation generation
        self.assertTrue(doc_response.success)
        self.assertIn("documentation", doc_response.response_text.lower())
        self.assertEqual(len(doc_response.generated_artifacts), 1)


if __name__ == '__main__':
    # Run tests
    unittest.main()