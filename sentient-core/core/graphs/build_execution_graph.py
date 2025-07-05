"""Build Execution Graph for Multi-Agent RAG System.

This module implements a specialized LangGraph workflow for build execution tasks,
including compilation, testing, packaging, deployment, and monitoring.
"""

from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.graph import StateGraph
from datetime import datetime
import json

from ..services.llm_service import EnhancedLLMService
from ..services.memory_service import MemoryService
from ..services.state_service import StateService


class BuildExecutionState(BaseModel):
    """State model for build execution workflow."""
    
    # Core workflow state
    current_phase: Literal[
        "environment_setup",
        "dependency_resolution",
        "code_compilation",
        "unit_testing",
        "integration_testing",
        "quality_assurance",
        "security_scanning",
        "performance_testing",
        "packaging",
        "deployment_preparation",
        "deployment_execution",
        "post_deployment_validation",
        "monitoring_setup",
        "completed",
        "error"
    ] = "environment_setup"
    
    # Input artifacts
    source_code: Dict[str, str] = Field(default_factory=dict)
    frontend_artifacts: Dict[str, Any] = Field(default_factory=dict)
    backend_artifacts: Dict[str, Any] = Field(default_factory=dict)
    architecture_specs: Dict[str, Any] = Field(default_factory=dict)
    deployment_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Environment configuration
    build_environment: Dict[str, Any] = Field(default_factory=dict)
    runtime_environment: Dict[str, Any] = Field(default_factory=dict)
    deployment_environment: Dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies and packages
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    resolved_dependencies: Dict[str, Any] = Field(default_factory=dict)
    dependency_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Build artifacts
    compiled_artifacts: Dict[str, Any] = Field(default_factory=dict)
    build_logs: List[Dict[str, Any]] = Field(default_factory=list)
    build_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Testing results
    unit_test_results: Dict[str, Any] = Field(default_factory=dict)
    integration_test_results: Dict[str, Any] = Field(default_factory=dict)
    performance_test_results: Dict[str, Any] = Field(default_factory=dict)
    security_scan_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality metrics
    code_quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    coverage_reports: Dict[str, Any] = Field(default_factory=dict)
    static_analysis_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Packaging
    package_specifications: Dict[str, Any] = Field(default_factory=dict)
    packaged_artifacts: Dict[str, Any] = Field(default_factory=dict)
    container_images: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Deployment
    deployment_plan: Dict[str, Any] = Field(default_factory=dict)
    deployment_status: Dict[str, Any] = Field(default_factory=dict)
    deployment_logs: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Monitoring and validation
    health_checks: Dict[str, Any] = Field(default_factory=dict)
    monitoring_config: Dict[str, Any] = Field(default_factory=dict)
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Progress tracking
    completed_phases: List[str] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 3
    
    # Output and results
    final_build_artifacts: Dict[str, Any] = Field(default_factory=dict)
    deployment_endpoints: List[Dict[str, str]] = Field(default_factory=list)
    
    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None


class BuildExecutionGraphBuilder:
    """Builder class for creating the build execution graph."""
    
    def __init__(
        self,
        llm_service: EnhancedLLMService,
        memory_service: MemoryService,
        state_service: StateService
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.state_service = state_service
    
    def setup_environment(self, state: BuildExecutionState) -> BuildExecutionState:
        """Setup build and runtime environments."""
        try:
            context = {
                "architecture_specs": state.architecture_specs,
                "deployment_config": state.deployment_config,
                "frontend_artifacts": state.frontend_artifacts,
                "backend_artifacts": state.backend_artifacts
            }
            
            prompt = f"""
            Setup comprehensive build environment:
            
            Context: {json.dumps(context, indent=2)}
            
            Configure environment including:
            1. Build environment setup (Node.js, Python, Docker, etc.)
            2. Runtime environment configuration
            3. Development tools and utilities
            4. Environment variables and secrets
            5. Build system configuration
            6. CI/CD environment setup
            7. Testing environment preparation
            8. Deployment environment configuration
            
            Provide detailed environment specifications and setup scripts.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            environment = json.loads(response.content)
            state.build_environment = environment.get("build_environment", {})
            state.runtime_environment = environment.get("runtime_environment", {})
            state.deployment_environment = environment.get("deployment_environment", {})
            
            state.completed_phases.append("environment_setup")
            state.current_phase = "dependency_resolution"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="environment_setup",
                content=environment,
                metadata={"phase": "environment_setup", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "environment_setup",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def resolve_dependencies(self, state: BuildExecutionState) -> BuildExecutionState:
        """Resolve and manage project dependencies."""
        try:
            context = {
                "frontend_artifacts": state.frontend_artifacts,
                "backend_artifacts": state.backend_artifacts,
                "build_environment": state.build_environment
            }
            
            prompt = f"""
            Resolve project dependencies:
            
            Context: {json.dumps(context, indent=2)}
            
            Handle dependencies including:
            1. Frontend dependencies (npm, yarn packages)
            2. Backend dependencies (pip, requirements)
            3. System dependencies and libraries
            4. Development dependencies
            5. Testing dependencies
            6. Build tool dependencies
            7. Dependency version management
            8. Conflict resolution strategies
            
            Provide dependency resolution plan and lock files.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            dependencies = json.loads(response.content)
            state.dependencies = dependencies.get("dependencies", {})
            state.resolved_dependencies = dependencies.get("resolved", {})
            state.dependency_conflicts = dependencies.get("conflicts", [])
            
            state.completed_phases.append("dependency_resolution")
            state.current_phase = "code_compilation"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="dependency_resolution",
                content=dependencies,
                metadata={"phase": "dependency_resolution", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "dependency_resolution",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def compile_code(self, state: BuildExecutionState) -> BuildExecutionState:
        """Compile and build the source code."""
        try:
            context = {
                "source_code": state.source_code,
                "resolved_dependencies": state.resolved_dependencies,
                "build_environment": state.build_environment
            }
            
            prompt = f"""
            Compile and build source code:
            
            Context: {json.dumps(context, indent=2)}
            
            Execute build process including:
            1. Frontend compilation (TypeScript, React, etc.)
            2. Backend compilation (Python bytecode, etc.)
            3. Asset optimization and bundling
            4. Static analysis and linting
            5. Code minification and compression
            6. Source map generation
            7. Build artifact organization
            8. Build verification and validation
            
            Provide build scripts and compiled artifacts.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            compilation = json.loads(response.content)
            state.compiled_artifacts = compilation.get("artifacts", {})
            state.build_logs = compilation.get("logs", [])
            state.build_metrics = compilation.get("metrics", {})
            
            state.completed_phases.append("code_compilation")
            state.current_phase = "unit_testing"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="code_compilation",
                content=compilation,
                metadata={"phase": "code_compilation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "code_compilation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def run_unit_tests(self, state: BuildExecutionState) -> BuildExecutionState:
        """Execute unit tests and generate reports."""
        try:
            context = {
                "compiled_artifacts": state.compiled_artifacts,
                "frontend_artifacts": state.frontend_artifacts,
                "backend_artifacts": state.backend_artifacts
            }
            
            prompt = f"""
            Execute comprehensive unit testing:
            
            Context: {json.dumps(context, indent=2)}
            
            Run unit tests including:
            1. Frontend unit tests (Jest, React Testing Library)
            2. Backend unit tests (pytest, unittest)
            3. Component testing
            4. Service testing
            5. Utility function testing
            6. Mock and stub testing
            7. Coverage analysis
            8. Test report generation
            
            Provide test execution results and coverage reports.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            testing = json.loads(response.content)
            state.unit_test_results = testing.get("results", {})
            state.coverage_reports = testing.get("coverage", {})
            
            state.completed_phases.append("unit_testing")
            state.current_phase = "integration_testing"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="unit_testing",
                content=testing,
                metadata={"phase": "unit_testing", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "unit_testing",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def run_integration_tests(self, state: BuildExecutionState) -> BuildExecutionState:
        """Execute integration tests."""
        try:
            context = {
                "compiled_artifacts": state.compiled_artifacts,
                "unit_test_results": state.unit_test_results,
                "deployment_config": state.deployment_config
            }
            
            prompt = f"""
            Execute integration testing:
            
            Context: {json.dumps(context, indent=2)}
            
            Run integration tests including:
            1. API integration tests
            2. Database integration tests
            3. Frontend-backend integration
            4. Third-party service integration
            5. End-to-end workflow testing
            6. Cross-browser testing
            7. Mobile responsiveness testing
            8. Performance integration testing
            
            Provide integration test results and analysis.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            integration = json.loads(response.content)
            state.integration_test_results = integration.get("results", {})
            
            state.completed_phases.append("integration_testing")
            state.current_phase = "quality_assurance"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="integration_testing",
                content=integration,
                metadata={"phase": "integration_testing", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "integration_testing",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def perform_quality_assurance(self, state: BuildExecutionState) -> BuildExecutionState:
        """Perform quality assurance checks."""
        try:
            context = {
                "compiled_artifacts": state.compiled_artifacts,
                "unit_test_results": state.unit_test_results,
                "integration_test_results": state.integration_test_results
            }
            
            prompt = f"""
            Perform quality assurance analysis:
            
            Context: {json.dumps(context, indent=2)}
            
            Execute QA processes including:
            1. Code quality analysis
            2. Static code analysis
            3. Complexity analysis
            4. Documentation quality check
            5. Accessibility testing
            6. Usability testing
            7. Compliance verification
            8. Best practices validation
            
            Provide quality metrics and recommendations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            quality = json.loads(response.content)
            state.code_quality_metrics = quality.get("metrics", {})
            state.static_analysis_results = quality.get("static_analysis", {})
            
            state.completed_phases.append("quality_assurance")
            state.current_phase = "security_scanning"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="quality_assurance",
                content=quality,
                metadata={"phase": "quality_assurance", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "quality_assurance",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def scan_security(self, state: BuildExecutionState) -> BuildExecutionState:
        """Perform security scanning and vulnerability assessment."""
        try:
            context = {
                "compiled_artifacts": state.compiled_artifacts,
                "resolved_dependencies": state.resolved_dependencies,
                "deployment_config": state.deployment_config
            }
            
            prompt = f"""
            Perform security scanning:
            
            Context: {json.dumps(context, indent=2)}
            
            Execute security analysis including:
            1. Dependency vulnerability scanning
            2. Static application security testing (SAST)
            3. Dynamic application security testing (DAST)
            4. Container security scanning
            5. Infrastructure security assessment
            6. Secrets detection
            7. License compliance checking
            8. Security best practices validation
            
            Provide security scan results and remediation recommendations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            security = json.loads(response.content)
            state.security_scan_results = security.get("results", {})
            
            state.completed_phases.append("security_scanning")
            state.current_phase = "performance_testing"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="security_scanning",
                content=security,
                metadata={"phase": "security_scanning", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "security_scanning",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def test_performance(self, state: BuildExecutionState) -> BuildExecutionState:
        """Execute performance testing."""
        try:
            context = {
                "compiled_artifacts": state.compiled_artifacts,
                "integration_test_results": state.integration_test_results,
                "deployment_config": state.deployment_config
            }
            
            prompt = f"""
            Execute performance testing:
            
            Context: {json.dumps(context, indent=2)}
            
            Run performance tests including:
            1. Load testing
            2. Stress testing
            3. Volume testing
            4. Endurance testing
            5. Frontend performance testing
            6. API performance testing
            7. Database performance testing
            8. Resource utilization analysis
            
            Provide performance test results and optimization recommendations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            performance = json.loads(response.content)
            state.performance_test_results = performance.get("results", {})
            
            state.completed_phases.append("performance_testing")
            state.current_phase = "packaging"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="performance_testing",
                content=performance,
                metadata={"phase": "performance_testing", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "performance_testing",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def package_artifacts(self, state: BuildExecutionState) -> BuildExecutionState:
        """Package build artifacts for deployment."""
        try:
            context = {
                "compiled_artifacts": state.compiled_artifacts,
                "deployment_config": state.deployment_config,
                "build_environment": state.build_environment
            }
            
            prompt = f"""
            Package build artifacts:
            
            Context: {json.dumps(context, indent=2)}
            
            Create deployment packages including:
            1. Docker container images
            2. Application bundles
            3. Static asset packages
            4. Configuration packages
            5. Database migration packages
            6. Documentation packages
            7. Deployment scripts
            8. Monitoring configurations
            
            Provide packaging specifications and artifacts.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            packaging = json.loads(response.content)
            state.package_specifications = packaging.get("specifications", {})
            state.packaged_artifacts = packaging.get("artifacts", {})
            state.container_images = packaging.get("containers", [])
            
            state.completed_phases.append("packaging")
            state.current_phase = "deployment_preparation"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="packaging",
                content=packaging,
                metadata={"phase": "packaging", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "packaging",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def prepare_deployment(self, state: BuildExecutionState) -> BuildExecutionState:
        """Prepare deployment infrastructure and configuration."""
        try:
            context = {
                "packaged_artifacts": state.packaged_artifacts,
                "deployment_config": state.deployment_config,
                "deployment_environment": state.deployment_environment
            }
            
            prompt = f"""
            Prepare deployment infrastructure:
            
            Context: {json.dumps(context, indent=2)}
            
            Setup deployment including:
            1. Infrastructure provisioning
            2. Environment configuration
            3. Network and security setup
            4. Database setup and migration
            5. Load balancer configuration
            6. SSL/TLS certificate setup
            7. Monitoring and logging setup
            8. Backup and recovery setup
            
            Provide deployment plan and infrastructure code.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            deployment_prep = json.loads(response.content)
            state.deployment_plan = deployment_prep.get("plan", {})
            
            state.completed_phases.append("deployment_preparation")
            state.current_phase = "deployment_execution"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="deployment_preparation",
                content=deployment_prep,
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
    
    def execute_deployment(self, state: BuildExecutionState) -> BuildExecutionState:
        """Execute the deployment process."""
        try:
            context = {
                "deployment_plan": state.deployment_plan,
                "packaged_artifacts": state.packaged_artifacts,
                "container_images": state.container_images
            }
            
            prompt = f"""
            Execute deployment process:
            
            Context: {json.dumps(context, indent=2)}
            
            Deploy application including:
            1. Container deployment
            2. Application deployment
            3. Database migration execution
            4. Configuration deployment
            5. Service startup and verification
            6. Health check validation
            7. Traffic routing setup
            8. Rollback preparation
            
            Provide deployment execution results and endpoints.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            deployment = json.loads(response.content)
            state.deployment_status = deployment.get("status", {})
            state.deployment_logs = deployment.get("logs", [])
            state.deployment_endpoints = deployment.get("endpoints", [])
            
            state.completed_phases.append("deployment_execution")
            state.current_phase = "post_deployment_validation"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="deployment_execution",
                content=deployment,
                metadata={"phase": "deployment_execution", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "deployment_execution",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def validate_deployment(self, state: BuildExecutionState) -> BuildExecutionState:
        """Validate post-deployment functionality."""
        try:
            context = {
                "deployment_status": state.deployment_status,
                "deployment_endpoints": state.deployment_endpoints,
                "integration_test_results": state.integration_test_results
            }
            
            prompt = f"""
            Validate deployment functionality:
            
            Context: {json.dumps(context, indent=2)}
            
            Perform validation including:
            1. Endpoint availability testing
            2. Functional validation
            3. Performance validation
            4. Security validation
            5. Data integrity checks
            6. User acceptance testing
            7. Smoke testing
            8. Regression testing
            
            Provide validation results and recommendations.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            validation = json.loads(response.content)
            state.validation_results = validation.get("results", {})
            state.health_checks = validation.get("health_checks", {})
            
            state.completed_phases.append("post_deployment_validation")
            state.current_phase = "monitoring_setup"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="post_deployment_validation",
                content=validation,
                metadata={"phase": "post_deployment_validation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "post_deployment_validation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def setup_monitoring(self, state: BuildExecutionState) -> BuildExecutionState:
        """Setup monitoring and alerting systems."""
        try:
            context = {
                "deployment_endpoints": state.deployment_endpoints,
                "health_checks": state.health_checks,
                "performance_test_results": state.performance_test_results
            }
            
            prompt = f"""
            Setup monitoring and alerting:
            
            Context: {json.dumps(context, indent=2)}
            
            Configure monitoring including:
            1. Application performance monitoring
            2. Infrastructure monitoring
            3. Log aggregation and analysis
            4. Error tracking and alerting
            5. User experience monitoring
            6. Security monitoring
            7. Business metrics tracking
            8. Automated alerting rules
            
            Provide monitoring configuration and dashboards.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            monitoring = json.loads(response.content)
            state.monitoring_config = monitoring.get("config", {})
            
            # Prepare final build artifacts
            state.final_build_artifacts = {
                "compiled_artifacts": state.compiled_artifacts,
                "packaged_artifacts": state.packaged_artifacts,
                "container_images": state.container_images,
                "deployment_endpoints": state.deployment_endpoints,
                "monitoring_config": state.monitoring_config,
                "validation_results": state.validation_results,
                "test_results": {
                    "unit_tests": state.unit_test_results,
                    "integration_tests": state.integration_test_results,
                    "performance_tests": state.performance_test_results,
                    "security_scans": state.security_scan_results
                }
            }
            
            state.completed_phases.append("monitoring_setup")
            state.current_phase = "completed"
            state.updated_at = datetime.now()
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
                message_type="monitoring_setup",
                content=monitoring,
                metadata={"phase": "monitoring_setup", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "monitoring_setup",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def handle_error(self, state: BuildExecutionState) -> BuildExecutionState:
        """Handle errors and attempt recovery."""
        if state.errors:
            latest_error = state.errors[-1]
            
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "build_exec",
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
    
    def should_continue(self, state: BuildExecutionState) -> str:
        """Determine the next node based on current state."""
        if state.current_phase == "error":
            return "handle_error"
        elif state.current_phase == "completed":
            return END
        elif state.current_phase == "environment_setup":
            return "setup_environment"
        elif state.current_phase == "dependency_resolution":
            return "resolve_dependencies"
        elif state.current_phase == "code_compilation":
            return "compile_code"
        elif state.current_phase == "unit_testing":
            return "run_unit_tests"
        elif state.current_phase == "integration_testing":
            return "run_integration_tests"
        elif state.current_phase == "quality_assurance":
            return "perform_quality_assurance"
        elif state.current_phase == "security_scanning":
            return "scan_security"
        elif state.current_phase == "performance_testing":
            return "test_performance"
        elif state.current_phase == "packaging":
            return "package_artifacts"
        elif state.current_phase == "deployment_preparation":
            return "prepare_deployment"
        elif state.current_phase == "deployment_execution":
            return "execute_deployment"
        elif state.current_phase == "post_deployment_validation":
            return "validate_deployment"
        elif state.current_phase == "monitoring_setup":
            return "setup_monitoring"
        else:
            return END
    
    def build_graph(self) -> Any:
        """Build and return the build execution graph."""
        workflow = StateGraph(BuildExecutionState)
        
        # Add nodes
        workflow.add_node("setup_environment", self.setup_environment)
        workflow.add_node("resolve_dependencies", self.resolve_dependencies)
        workflow.add_node("compile_code", self.compile_code)
        workflow.add_node("run_unit_tests", self.run_unit_tests)
        workflow.add_node("run_integration_tests", self.run_integration_tests)
        workflow.add_node("perform_quality_assurance", self.perform_quality_assurance)
        workflow.add_node("scan_security", self.scan_security)
        workflow.add_node("test_performance", self.test_performance)
        workflow.add_node("package_artifacts", self.package_artifacts)
        workflow.add_node("prepare_deployment", self.prepare_deployment)
        workflow.add_node("execute_deployment", self.execute_deployment)
        workflow.add_node("validate_deployment", self.validate_deployment)
        workflow.add_node("setup_monitoring", self.setup_monitoring)
        workflow.add_node("handle_error", self.handle_error)
        
        # Set entry point
        workflow.set_entry_point("setup_environment")
        
        # Add conditional edges
        for node in [
            "setup_environment", "resolve_dependencies", "compile_code",
            "run_unit_tests", "run_integration_tests", "perform_quality_assurance",
            "scan_security", "test_performance", "package_artifacts",
            "prepare_deployment", "execute_deployment", "validate_deployment",
            "setup_monitoring", "handle_error"
        ]:
            workflow.add_conditional_edges(node, self.should_continue)
        
        return workflow.compile()


def create_build_execution_graph(
    llm_service: EnhancedLLMService,
    memory_service: MemoryService,
    state_service: StateService
) -> Any:
    """Factory function to create a build execution graph."""
    builder = BuildExecutionGraphBuilder(llm_service, memory_service, state_service)
    return builder.build_graph()