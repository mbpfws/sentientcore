"""Backend Development Graph for Multi-Agent RAG System.

This module implements a specialized LangGraph workflow for backend development tasks,
including API design, database schema creation, authentication implementation,
and deployment configuration.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph import StateGraph, END
from langgraph.graph import Graph
from datetime import datetime
import json

from ..services.llm_service import EnhancedLLMService
from ..services.memory_service import MemoryService
from ..services.state_service import StateService


class BackendDevelopmentState(BaseModel):
    """State model for backend development workflow."""
    
    # Core workflow state
    current_phase: Literal[
        "requirements_analysis",
        "architecture_design",
        "technology_selection",
        "database_design",
        "api_design",
        "authentication_setup",
        "core_implementation",
        "integration_development",
        "testing_implementation",
        "security_hardening",
        "performance_optimization",
        "deployment_preparation",
        "documentation",
        "completed",
        "error"
    ] = "requirements_analysis"
    
    # Input and context
    project_requirements: Dict[str, Any] = Field(default_factory=dict)
    functional_requirements: Dict[str, Any] = Field(default_factory=dict)
    non_functional_requirements: Dict[str, Any] = Field(default_factory=dict)
    integration_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    # Architecture and design
    system_architecture: Dict[str, Any] = Field(default_factory=dict)
    service_architecture: Dict[str, Any] = Field(default_factory=dict)
    data_flow_design: Dict[str, Any] = Field(default_factory=dict)
    
    # Technology stack
    selected_framework: Optional[str] = None
    database_technology: Optional[str] = None
    authentication_method: Optional[str] = None
    deployment_platform: Optional[str] = None
    additional_services: List[str] = Field(default_factory=list)
    
    # Database design
    database_schema: Dict[str, Any] = Field(default_factory=dict)
    entity_relationships: Dict[str, Any] = Field(default_factory=dict)
    migration_scripts: List[Dict[str, str]] = Field(default_factory=list)
    
    # API design
    api_specification: Dict[str, Any] = Field(default_factory=dict)
    endpoint_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    data_models: Dict[str, Any] = Field(default_factory=dict)
    
    # Authentication and security
    authentication_config: Dict[str, Any] = Field(default_factory=dict)
    authorization_rules: Dict[str, Any] = Field(default_factory=dict)
    security_measures: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Implementation artifacts
    core_modules: Dict[str, str] = Field(default_factory=dict)
    service_implementations: Dict[str, str] = Field(default_factory=dict)
    integration_code: Dict[str, str] = Field(default_factory=dict)
    
    # Testing
    test_specifications: Dict[str, Any] = Field(default_factory=dict)
    test_implementations: Dict[str, str] = Field(default_factory=dict)
    test_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance and optimization
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    optimization_strategies: List[Dict[str, Any]] = Field(default_factory=list)
    caching_strategy: Dict[str, Any] = Field(default_factory=dict)
    
    # Deployment
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    infrastructure_code: Dict[str, str] = Field(default_factory=dict)
    ci_cd_pipeline: Dict[str, Any] = Field(default_factory=dict)
    
    # Documentation
    api_documentation: Dict[str, str] = Field(default_factory=dict)
    deployment_guide: Dict[str, str] = Field(default_factory=dict)
    maintenance_guide: Dict[str, str] = Field(default_factory=dict)
    
    # Progress tracking
    completed_phases: List[str] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 3
    
    # Output and results
    final_artifacts: Dict[str, Any] = Field(default_factory=dict)
    deployment_package: Dict[str, Any] = Field(default_factory=dict)
    
    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None


class BackendDevelopmentGraphBuilder:
    """Builder class for creating the backend development graph."""
    
    def __init__(
        self,
        llm_service: EnhancedLLMService,
        memory_service: MemoryService,
        state_service: StateService
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.state_service = state_service
    
    def analyze_requirements(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Analyze backend requirements and constraints."""
        try:
            context = {
                "project_requirements": state.project_requirements,
                "functional_requirements": state.functional_requirements,
                "non_functional_requirements": state.non_functional_requirements
            }
            
            prompt = f"""
            Analyze the backend development requirements:
            
            Context: {json.dumps(context, indent=2)}
            
            Provide comprehensive analysis including:
            1. Core functional requirements breakdown
            2. Performance and scalability requirements
            3. Security and compliance needs
            4. Integration requirements
            5. Data storage and processing needs
            6. API requirements and specifications
            7. Authentication and authorization needs
            8. Deployment and infrastructure requirements
            
            Format your response as structured JSON with detailed analysis.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            analysis = json.loads(response.content)
            state.project_requirements.update(analysis.get("requirements", {}))
            state.integration_requirements = analysis.get("integration_requirements", {})
            
            state.completed_phases.append("requirements_analysis")
            state.current_phase = "architecture_design"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="requirements_analysis",
                content=analysis,
                metadata={"phase": "requirements_analysis", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "requirements_analysis",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def design_architecture(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Design the overall system architecture."""
        try:
            context = {
                "requirements": state.project_requirements,
                "integration_requirements": state.integration_requirements
            }
            
            prompt = f"""
            Design a comprehensive backend architecture:
            
            Context: {json.dumps(context, indent=2)}
            
            Create architecture design including:
            1. High-level system architecture
            2. Service decomposition and boundaries
            3. Data flow and communication patterns
            4. Scalability and performance considerations
            5. Security architecture
            6. Integration patterns
            7. Deployment architecture
            8. Monitoring and observability design
            
            Provide detailed architectural specifications and diagrams.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            architecture = json.loads(response.content)
            state.system_architecture = architecture.get("system_architecture", {})
            state.service_architecture = architecture.get("service_architecture", {})
            state.data_flow_design = architecture.get("data_flow", {})
            
            state.completed_phases.append("architecture_design")
            state.current_phase = "technology_selection"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="architecture_design",
                content=architecture,
                metadata={"phase": "architecture_design", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "architecture_design",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def select_technology_stack(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Select appropriate technology stack for backend development."""
        try:
            context = {
                "architecture": state.system_architecture,
                "requirements": state.project_requirements,
                "service_architecture": state.service_architecture
            }
            
            prompt = f"""
            Select optimal technology stack for the backend:
            
            Context: {json.dumps(context, indent=2)}
            
            Recommend technologies for:
            1. Backend framework (FastAPI, Django, Express.js, Spring Boot, etc.)
            2. Database technology (PostgreSQL, MongoDB, Redis, etc.)
            3. Authentication and authorization
            4. Message queues and event streaming
            5. Caching solutions
            6. API documentation tools
            7. Testing frameworks
            8. Deployment and containerization
            9. Monitoring and logging
            10. CI/CD tools
            
            Include justification for each choice and integration considerations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            tech_stack = json.loads(response.content)
            state.selected_framework = tech_stack.get("framework")
            state.database_technology = tech_stack.get("database")
            state.authentication_method = tech_stack.get("authentication")
            state.deployment_platform = tech_stack.get("deployment")
            state.additional_services = tech_stack.get("additional_services", [])
            
            state.completed_phases.append("technology_selection")
            state.current_phase = "database_design"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="technology_selection",
                content=tech_stack,
                metadata={"phase": "technology_selection", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "technology_selection",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def design_database(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Design database schema and data models."""
        try:
            context = {
                "requirements": state.project_requirements,
                "architecture": state.system_architecture,
                "database_tech": state.database_technology
            }
            
            prompt = f"""
            Design comprehensive database schema:
            
            Context: {json.dumps(context, indent=2)}
            
            Create database design including:
            1. Entity-relationship model
            2. Table schemas with constraints
            3. Indexes and performance optimizations
            4. Data migration strategies
            5. Backup and recovery procedures
            6. Scaling considerations
            7. Security and access control
            8. Data validation rules
            
            Provide SQL scripts and migration files.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            database_design = json.loads(response.content)
            state.database_schema = database_design.get("schema", {})
            state.entity_relationships = database_design.get("relationships", {})
            state.migration_scripts = database_design.get("migrations", [])
            
            state.completed_phases.append("database_design")
            state.current_phase = "api_design"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="database_design",
                content=database_design,
                metadata={"phase": "database_design", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "database_design",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def design_api(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Design API endpoints and specifications."""
        try:
            context = {
                "requirements": state.project_requirements,
                "database_schema": state.database_schema,
                "framework": state.selected_framework
            }
            
            prompt = f"""
            Design comprehensive API specification:
            
            Context: {json.dumps(context, indent=2)}
            
            Create API design including:
            1. RESTful endpoint definitions
            2. Request/response schemas
            3. Authentication and authorization
            4. Error handling and status codes
            5. Rate limiting and throttling
            6. API versioning strategy
            7. Documentation specifications
            8. Testing scenarios
            
            Provide OpenAPI/Swagger specifications.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            api_design = json.loads(response.content)
            state.api_specification = api_design.get("specification", {})
            state.endpoint_definitions = api_design.get("endpoints", [])
            state.data_models = api_design.get("models", {})
            
            state.completed_phases.append("api_design")
            state.current_phase = "authentication_setup"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="api_design",
                content=api_design,
                metadata={"phase": "api_design", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "api_design",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def setup_authentication(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Setup authentication and authorization systems."""
        try:
            context = {
                "auth_method": state.authentication_method,
                "api_spec": state.api_specification,
                "requirements": state.project_requirements
            }
            
            prompt = f"""
            Design authentication and authorization system:
            
            Context: {json.dumps(context, indent=2)}
            
            Create auth system including:
            1. Authentication mechanisms (JWT, OAuth, etc.)
            2. User management and registration
            3. Role-based access control (RBAC)
            4. Permission management
            5. Session management
            6. Security best practices
            7. Multi-factor authentication
            8. Password policies and recovery
            
            Provide implementation specifications and code.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            auth_design = json.loads(response.content)
            state.authentication_config = auth_design.get("config", {})
            state.authorization_rules = auth_design.get("authorization", {})
            state.security_measures = auth_design.get("security_measures", [])
            
            state.completed_phases.append("authentication_setup")
            state.current_phase = "core_implementation"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="authentication_setup",
                content=auth_design,
                metadata={"phase": "authentication_setup", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "authentication_setup",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def implement_core_modules(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Implement core backend modules and services."""
        try:
            context = {
                "framework": state.selected_framework,
                "api_spec": state.api_specification,
                "database_schema": state.database_schema,
                "auth_config": state.authentication_config
            }
            
            prompt = f"""
            Implement core backend modules:
            
            Context: {json.dumps(context, indent=2)}
            
            Create implementation including:
            1. Application structure and configuration
            2. Database models and ORM setup
            3. API route handlers
            4. Business logic services
            5. Data access layer
            6. Middleware and utilities
            7. Error handling and logging
            8. Configuration management
            
            Provide complete code files and structure.
            Format as structured JSON with file contents.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            implementation = json.loads(response.content)
            state.core_modules = implementation.get("core_modules", {})
            state.service_implementations = implementation.get("services", {})
            
            state.completed_phases.append("core_implementation")
            state.current_phase = "integration_development"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="core_implementation",
                content=implementation,
                metadata={"phase": "core_implementation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "core_implementation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def develop_integrations(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Develop external integrations and third-party services."""
        try:
            context = {
                "integration_requirements": state.integration_requirements,
                "core_modules": state.core_modules,
                "additional_services": state.additional_services
            }
            
            prompt = f"""
            Develop external integrations:
            
            Context: {json.dumps(context, indent=2)}
            
            Create integrations including:
            1. Third-party API integrations
            2. Message queue implementations
            3. External service connectors
            4. Webhook handlers
            5. Event processing systems
            6. Data synchronization
            7. Monitoring and alerting
            8. Backup and recovery
            
            Provide integration code and configurations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            integrations = json.loads(response.content)
            state.integration_code = integrations.get("integration_code", {})
            
            state.completed_phases.append("integration_development")
            state.current_phase = "testing_implementation"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="integration_development",
                content=integrations,
                metadata={"phase": "integration_development", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "integration_development",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def implement_testing(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Implement comprehensive testing suite."""
        try:
            context = {
                "core_modules": state.core_modules,
                "api_spec": state.api_specification,
                "integration_code": state.integration_code
            }
            
            prompt = f"""
            Implement comprehensive testing suite:
            
            Context: {json.dumps(context, indent=2)}
            
            Create testing implementation including:
            1. Unit tests for all modules
            2. Integration tests for APIs
            3. Database testing strategies
            4. Authentication testing
            5. Performance testing
            6. Security testing
            7. Load testing scenarios
            8. End-to-end testing
            
            Provide test code and configurations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            testing = json.loads(response.content)
            state.test_specifications = testing.get("specifications", {})
            state.test_implementations = testing.get("implementations", {})
            state.test_results = testing.get("results", [])
            
            state.completed_phases.append("testing_implementation")
            state.current_phase = "security_hardening"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="testing_implementation",
                content=testing,
                metadata={"phase": "testing_implementation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "testing_implementation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def harden_security(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Implement security hardening measures."""
        try:
            context = {
                "security_measures": state.security_measures,
                "auth_config": state.authentication_config,
                "api_spec": state.api_specification
            }
            
            prompt = f"""
            Implement security hardening:
            
            Context: {json.dumps(context, indent=2)}
            
            Apply security measures including:
            1. Input validation and sanitization
            2. SQL injection prevention
            3. XSS protection
            4. CSRF protection
            5. Rate limiting and DDoS protection
            6. Encryption and data protection
            7. Security headers
            8. Vulnerability scanning
            
            Provide security configurations and code.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            security = json.loads(response.content)
            state.security_measures.extend(security.get("additional_measures", []))
            
            state.completed_phases.append("security_hardening")
            state.current_phase = "performance_optimization"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="security_hardening",
                content=security,
                metadata={"phase": "security_hardening", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "security_hardening",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def optimize_performance(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Implement performance optimizations."""
        try:
            context = {
                "core_modules": state.core_modules,
                "database_schema": state.database_schema,
                "api_spec": state.api_specification
            }
            
            prompt = f"""
            Implement performance optimizations:
            
            Context: {json.dumps(context, indent=2)}
            
            Apply optimizations including:
            1. Database query optimization
            2. Caching strategies
            3. Connection pooling
            4. Async processing
            5. Memory optimization
            6. Response compression
            7. CDN integration
            8. Monitoring and profiling
            
            Provide optimization code and configurations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            optimization = json.loads(response.content)
            state.performance_metrics = optimization.get("metrics", {})
            state.optimization_strategies = optimization.get("strategies", [])
            state.caching_strategy = optimization.get("caching", {})
            
            state.completed_phases.append("performance_optimization")
            state.current_phase = "deployment_preparation"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="performance_optimization",
                content=optimization,
                metadata={"phase": "performance_optimization", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "performance_optimization",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def prepare_deployment(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Prepare deployment configurations and infrastructure."""
        try:
            context = {
                "deployment_platform": state.deployment_platform,
                "core_modules": state.core_modules,
                "database_schema": state.database_schema
            }
            
            prompt = f"""
            Prepare deployment configuration:
            
            Context: {json.dumps(context, indent=2)}
            
            Create deployment setup including:
            1. Containerization (Docker)
            2. Orchestration (Kubernetes)
            3. Infrastructure as Code
            4. CI/CD pipeline
            5. Environment configurations
            6. Monitoring and logging
            7. Backup and recovery
            8. Scaling strategies
            
            Provide deployment files and scripts.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            deployment = json.loads(response.content)
            state.deployment_config = deployment.get("config", {})
            state.infrastructure_code = deployment.get("infrastructure", {})
            state.ci_cd_pipeline = deployment.get("pipeline", {})
            
            state.completed_phases.append("deployment_preparation")
            state.current_phase = "documentation"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="deployment_preparation",
                content=deployment,
                metadata={"phase": "deployment_preparation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "deployment_preparation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def create_documentation(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Create comprehensive documentation."""
        try:
            context = {
                "api_spec": state.api_specification,
                "deployment_config": state.deployment_config,
                "core_modules": state.core_modules
            }
            
            prompt = f"""
            Create comprehensive documentation:
            
            Context: {json.dumps(context, indent=2)}
            
            Generate documentation including:
            1. API documentation
            2. Deployment guide
            3. Developer setup guide
            4. Architecture documentation
            5. Database schema documentation
            6. Security guidelines
            7. Troubleshooting guide
            8. Maintenance procedures
            
            Provide complete documentation files.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            documentation = json.loads(response.content)
            state.api_documentation = documentation.get("api_docs", {})
            state.deployment_guide = documentation.get("deployment_guide", {})
            state.maintenance_guide = documentation.get("maintenance_guide", {})
            
            # Prepare final artifacts
            state.final_artifacts = {
                "core_modules": state.core_modules,
                "service_implementations": state.service_implementations,
                "integration_code": state.integration_code,
                "test_implementations": state.test_implementations,
                "deployment_config": state.deployment_config,
                "infrastructure_code": state.infrastructure_code,
                "documentation": documentation
            }
            
            state.deployment_package = {
                "deployment_config": state.deployment_config,
                "infrastructure_code": state.infrastructure_code,
                "ci_cd_pipeline": state.ci_cd_pipeline,
                "deployment_guide": state.deployment_guide
            }
            
            state.completed_phases.append("documentation")
            state.current_phase = "completed"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="documentation",
                content=documentation,
                metadata={"phase": "documentation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "documentation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def handle_error(self, state: BackendDevelopmentState) -> BackendDevelopmentState:
        """Handle errors and attempt recovery."""
        if state.errors:
            latest_error = state.errors[-1]
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "backend_dev",
                message_type="error",
                content=latest_error,
                metadata={"phase": "error_handling", "timestamp": datetime.now().isoformat()}
            )
            
            error_phase = latest_error.get("phase")
            if error_phase and error_phase in state.completed_phases:
                state.completed_phases.remove(error_phase)
                state.current_phase = error_phase
            else:
                state.current_phase = "completed"
        
        return state
    
    def should_continue(self, state: BackendDevelopmentState) -> str:
        """Determine the next node based on current state."""
        if state.current_phase == "error":
            return "handle_error"
        elif state.current_phase == "completed":
            return END
        elif state.current_phase == "requirements_analysis":
            return "analyze_requirements"
        elif state.current_phase == "architecture_design":
            return "design_architecture"
        elif state.current_phase == "technology_selection":
            return "select_technology_stack"
        elif state.current_phase == "database_design":
            return "design_database"
        elif state.current_phase == "api_design":
            return "design_api"
        elif state.current_phase == "authentication_setup":
            return "setup_authentication"
        elif state.current_phase == "core_implementation":
            return "implement_core_modules"
        elif state.current_phase == "integration_development":
            return "develop_integrations"
        elif state.current_phase == "testing_implementation":
            return "implement_testing"
        elif state.current_phase == "security_hardening":
            return "harden_security"
        elif state.current_phase == "performance_optimization":
            return "optimize_performance"
        elif state.current_phase == "deployment_preparation":
            return "prepare_deployment"
        elif state.current_phase == "documentation":
            return "create_documentation"
        else:
            return END
    
    def build_graph(self) -> Graph:
        """Build and return the backend development graph."""
        workflow = StateGraph(BackendDevelopmentState)
        
        # Add nodes
        workflow.add_node("analyze_requirements", self.analyze_requirements)
        workflow.add_node("design_architecture", self.design_architecture)
        workflow.add_node("select_technology_stack", self.select_technology_stack)
        workflow.add_node("design_database", self.design_database)
        workflow.add_node("design_api", self.design_api)
        workflow.add_node("setup_authentication", self.setup_authentication)
        workflow.add_node("implement_core_modules", self.implement_core_modules)
        workflow.add_node("develop_integrations", self.develop_integrations)
        workflow.add_node("implement_testing", self.implement_testing)
        workflow.add_node("harden_security", self.harden_security)
        workflow.add_node("optimize_performance", self.optimize_performance)
        workflow.add_node("prepare_deployment", self.prepare_deployment)
        workflow.add_node("create_documentation", self.create_documentation)
        workflow.add_node("handle_error", self.handle_error)
        
        # Set entry point
        workflow.set_entry_point("analyze_requirements")
        
        # Add conditional edges
        for node in [
            "analyze_requirements", "design_architecture", "select_technology_stack",
            "design_database", "design_api", "setup_authentication",
            "implement_core_modules", "develop_integrations", "implement_testing",
            "harden_security", "optimize_performance", "prepare_deployment",
            "create_documentation", "handle_error"
        ]:
            workflow.add_conditional_edges(node, self.should_continue)
        
        return workflow.compile()


def create_backend_development_graph(
    llm_service: EnhancedLLMService,
    memory_service: MemoryService,
    state_service: StateService
) -> Graph:
    """Factory function to create a backend development graph."""
    builder = BackendDevelopmentGraphBuilder(llm_service, memory_service, state_service)
    return builder.build_graph()