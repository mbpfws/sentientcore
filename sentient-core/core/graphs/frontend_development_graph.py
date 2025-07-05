"""Frontend Development Graph for Multi-Agent RAG System.

This module implements a specialized LangGraph workflow for frontend development tasks,
including mockup generation, wireframe creation, prototype development, and full
frontend implementation with user feedback integration.
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


class FrontendDevelopmentState(BaseModel):
    """State model for frontend development workflow."""
    
    # Core workflow state
    current_phase: Literal[
        "requirements_analysis",
        "technology_selection", 
        "wireframe_design",
        "mockup_creation",
        "prototype_development",
        "component_implementation",
        "integration_testing",
        "user_feedback",
        "refinement",
        "finalization",
        "completed",
        "error"
    ] = "requirements_analysis"
    
    # Input and context
    project_requirements: Dict[str, Any] = Field(default_factory=dict)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    technical_constraints: Dict[str, Any] = Field(default_factory=dict)
    
    # Technology stack
    selected_framework: Optional[str] = None
    ui_library: Optional[str] = None
    styling_approach: Optional[str] = None
    build_tools: List[str] = Field(default_factory=list)
    
    # Design artifacts
    wireframes: List[Dict[str, Any]] = Field(default_factory=list)
    mockups: List[Dict[str, Any]] = Field(default_factory=list)
    design_system: Dict[str, Any] = Field(default_factory=dict)
    
    # Development artifacts
    component_specifications: List[Dict[str, Any]] = Field(default_factory=list)
    prototype_code: Dict[str, str] = Field(default_factory=dict)
    implementation_plan: Dict[str, Any] = Field(default_factory=dict)
    
    # Testing and feedback
    test_results: List[Dict[str, Any]] = Field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = Field(default_factory=list)
    refinement_requests: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Progress tracking
    completed_phases: List[str] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 3
    
    # Output and results
    final_artifacts: Dict[str, Any] = Field(default_factory=dict)
    deployment_instructions: Dict[str, Any] = Field(default_factory=dict)
    documentation: Dict[str, str] = Field(default_factory=dict)
    
    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None


class FrontendDevelopmentGraphBuilder:
    """Builder class for creating the frontend development graph."""
    
    def __init__(
        self,
        llm_service: EnhancedLLMService,
        memory_service: MemoryService,
        state_service: StateService
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.state_service = state_service
    
    def analyze_requirements(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Analyze project requirements and user preferences."""
        try:
            # Prepare context for LLM
            context = {
                "project_requirements": state.project_requirements,
                "user_preferences": state.user_preferences,
                "technical_constraints": state.technical_constraints
            }
            
            # Generate requirements analysis
            prompt = f"""
            Analyze the following frontend development requirements:
            
            Project Requirements: {json.dumps(context['project_requirements'], indent=2)}
            User Preferences: {json.dumps(context['user_preferences'], indent=2)}
            Technical Constraints: {json.dumps(context['technical_constraints'], indent=2)}
            
            Provide a comprehensive analysis including:
            1. Key functional requirements
            2. User experience priorities
            3. Technical feasibility assessment
            4. Recommended approach
            5. Potential challenges and solutions
            
            Format your response as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            # Parse and store analysis
            analysis = json.loads(response.content)
            state.project_requirements.update(analysis)
            state.completed_phases.append("requirements_analysis")
            state.current_phase = "technology_selection"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="analysis",
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
    
    def select_technology_stack(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Select appropriate technology stack for the frontend."""
        try:
            # Prepare context for technology selection
            context = {
                "requirements": state.project_requirements,
                "constraints": state.technical_constraints,
                "preferences": state.user_preferences
            }
            
            prompt = f"""
            Based on the analyzed requirements, recommend the best technology stack:
            
            Context: {json.dumps(context, indent=2)}
            
            Provide recommendations for:
            1. Frontend framework (React, Vue, Angular, Svelte, etc.)
            2. UI component library (Material-UI, Ant Design, Chakra UI, etc.)
            3. Styling approach (CSS-in-JS, Tailwind, SCSS, etc.)
            4. Build tools and bundlers
            5. Testing frameworks
            6. Deployment strategy
            
            Include justification for each choice and alternative options.
            Format as structured JSON.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            # Parse and store technology selections
            tech_stack = json.loads(response.content)
            state.selected_framework = tech_stack.get("framework")
            state.ui_library = tech_stack.get("ui_library")
            state.styling_approach = tech_stack.get("styling")
            state.build_tools = tech_stack.get("build_tools", [])
            
            state.completed_phases.append("technology_selection")
            state.current_phase = "wireframe_design"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
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
    
    def create_wireframes(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Create wireframes for the frontend application."""
        try:
            context = {
                "requirements": state.project_requirements,
                "framework": state.selected_framework,
                "ui_library": state.ui_library
            }
            
            prompt = f"""
            Create detailed wireframes for the frontend application:
            
            Context: {json.dumps(context, indent=2)}
            
            Generate wireframes including:
            1. Page layouts and navigation structure
            2. Component hierarchy and placement
            3. User interaction flows
            4. Responsive design considerations
            5. Accessibility features
            
            For each wireframe, provide:
            - Page/component name
            - Layout description
            - Key elements and their positions
            - Interaction patterns
            - Notes for implementation
            
            Format as structured JSON with wireframe specifications.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.4
            )
            
            # Parse and store wireframes
            wireframes = json.loads(response.content)
            state.wireframes = wireframes.get("wireframes", [])
            
            state.completed_phases.append("wireframe_design")
            state.current_phase = "mockup_creation"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="wireframes",
                content=wireframes,
                metadata={"phase": "wireframe_design", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "wireframe_design",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def create_mockups(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Create high-fidelity mockups based on wireframes."""
        try:
            context = {
                "wireframes": state.wireframes,
                "framework": state.selected_framework,
                "ui_library": state.ui_library,
                "styling": state.styling_approach
            }
            
            prompt = f"""
            Create high-fidelity mockups based on the wireframes:
            
            Context: {json.dumps(context, indent=2)}
            
            Generate detailed mockups including:
            1. Visual design specifications
            2. Color schemes and typography
            3. Component styling details
            4. Interactive element designs
            5. Responsive breakpoint designs
            
            For each mockup, provide:
            - Visual description
            - CSS/styling specifications
            - Component library usage
            - Asset requirements
            - Implementation notes
            
            Format as structured JSON with mockup specifications.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.4
            )
            
            # Parse and store mockups
            mockups = json.loads(response.content)
            state.mockups = mockups.get("mockups", [])
            state.design_system = mockups.get("design_system", {})
            
            state.completed_phases.append("mockup_creation")
            state.current_phase = "prototype_development"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="mockups",
                content=mockups,
                metadata={"phase": "mockup_creation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "mockup_creation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def develop_prototype(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Develop a working prototype based on mockups."""
        try:
            context = {
                "mockups": state.mockups,
                "design_system": state.design_system,
                "framework": state.selected_framework,
                "ui_library": state.ui_library,
                "styling": state.styling_approach
            }
            
            prompt = f"""
            Generate prototype code based on the mockups and design system:
            
            Context: {json.dumps(context, indent=2)}
            
            Create a working prototype including:
            1. Main application structure
            2. Key components implementation
            3. Basic routing and navigation
            4. Styling and theming
            5. Sample data and interactions
            
            Provide:
            - Complete file structure
            - Component code files
            - Styling files
            - Configuration files
            - Package.json dependencies
            - README with setup instructions
            
            Format as structured JSON with file contents.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            # Parse and store prototype code
            prototype = json.loads(response.content)
            state.prototype_code = prototype.get("files", {})
            state.implementation_plan = prototype.get("implementation_plan", {})
            
            state.completed_phases.append("prototype_development")
            state.current_phase = "component_implementation"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="prototype",
                content=prototype,
                metadata={"phase": "prototype_development", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "prototype_development",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def implement_components(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Implement detailed components and features."""
        try:
            context = {
                "prototype_code": state.prototype_code,
                "implementation_plan": state.implementation_plan,
                "requirements": state.project_requirements
            }
            
            prompt = f"""
            Implement detailed components and features for the frontend:
            
            Context: {json.dumps(context, indent=2)}
            
            Create comprehensive implementation including:
            1. Advanced component features
            2. State management integration
            3. API integration patterns
            4. Error handling and validation
            5. Performance optimizations
            
            Provide:
            - Enhanced component specifications
            - State management setup
            - API service implementations
            - Utility functions and helpers
            - Testing specifications
            
            Format as structured JSON with implementation details.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            # Parse and store component implementations
            implementation = json.loads(response.content)
            state.component_specifications = implementation.get("components", [])
            
            state.completed_phases.append("component_implementation")
            state.current_phase = "integration_testing"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="implementation",
                content=implementation,
                metadata={"phase": "component_implementation", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "component_implementation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def perform_integration_testing(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Perform integration testing of the frontend components."""
        try:
            context = {
                "components": state.component_specifications,
                "prototype_code": state.prototype_code,
                "framework": state.selected_framework
            }
            
            prompt = f"""
            Create comprehensive integration tests for the frontend:
            
            Context: {json.dumps(context, indent=2)}
            
            Generate testing strategy including:
            1. Unit tests for components
            2. Integration tests for workflows
            3. End-to-end testing scenarios
            4. Performance testing guidelines
            5. Accessibility testing checklist
            
            Provide:
            - Test specifications
            - Test implementation code
            - Testing configuration
            - CI/CD integration
            - Quality metrics
            
            Format as structured JSON with testing details.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            # Parse and store test results
            testing = json.loads(response.content)
            state.test_results = testing.get("test_results", [])
            
            state.completed_phases.append("integration_testing")
            state.current_phase = "user_feedback"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="testing",
                content=testing,
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
    
    def process_user_feedback(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Process user feedback and determine next steps."""
        try:
            # Check if we have user feedback to process
            if not state.user_feedback:
                # No feedback yet, wait for user input
                state.current_phase = "user_feedback"
                return state
            
            # Process the latest feedback
            latest_feedback = state.user_feedback[-1]
            
            context = {
                "feedback": latest_feedback,
                "current_implementation": state.component_specifications,
                "iteration": state.current_iteration
            }
            
            prompt = f"""
            Analyze user feedback and determine refinement actions:
            
            Context: {json.dumps(context, indent=2)}
            
            Based on the feedback, determine:
            1. Required changes and their priority
            2. Impact assessment on existing implementation
            3. Refinement strategy
            4. Timeline and effort estimation
            5. Whether additional iterations are needed
            
            Provide structured response with action plan.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            # Parse feedback analysis
            analysis = json.loads(response.content)
            state.refinement_requests.append(analysis)
            
            # Determine next phase based on feedback
            if analysis.get("requires_refinement", False) and state.current_iteration < state.max_iterations:
                state.current_phase = "refinement"
                state.current_iteration += 1
            else:
                state.current_phase = "finalization"
            
            state.completed_phases.append("user_feedback")
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="feedback_analysis",
                content=analysis,
                metadata={"phase": "user_feedback", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "user_feedback",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def refine_implementation(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Refine implementation based on user feedback."""
        try:
            if not state.refinement_requests:
                state.current_phase = "finalization"
                return state
            
            latest_refinement = state.refinement_requests[-1]
            
            context = {
                "refinement_request": latest_refinement,
                "current_implementation": state.component_specifications,
                "prototype_code": state.prototype_code
            }
            
            prompt = f"""
            Implement refinements based on the analysis:
            
            Context: {json.dumps(context, indent=2)}
            
            Apply the requested changes including:
            1. Code modifications
            2. Design updates
            3. Feature enhancements
            4. Bug fixes
            5. Performance improvements
            
            Provide updated implementation with change documentation.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            
            # Parse and apply refinements
            refinements = json.loads(response.content)
            
            # Update implementation with refinements
            if "updated_components" in refinements:
                state.component_specifications = refinements["updated_components"]
            if "updated_code" in refinements:
                state.prototype_code.update(refinements["updated_code"])
            
            state.completed_phases.append("refinement")
            state.current_phase = "integration_testing"  # Re-test after refinements
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="refinements",
                content=refinements,
                metadata={"phase": "refinement", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "refinement",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def finalize_implementation(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Finalize the frontend implementation and prepare deliverables."""
        try:
            context = {
                "components": state.component_specifications,
                "prototype_code": state.prototype_code,
                "test_results": state.test_results,
                "design_system": state.design_system
            }
            
            prompt = f"""
            Finalize the frontend implementation and create deliverables:
            
            Context: {json.dumps(context, indent=2)}
            
            Create final deliverables including:
            1. Production-ready code
            2. Deployment instructions
            3. Documentation
            4. Style guide
            5. Maintenance guidelines
            
            Provide comprehensive final package.
            """
            
            response = self.llm_service.generate_response(
                prompt=prompt,
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            
            # Parse and store final artifacts
            final_package = json.loads(response.content)
            state.final_artifacts = final_package.get("artifacts", {})
            state.deployment_instructions = final_package.get("deployment", {})
            state.documentation = final_package.get("documentation", {})
            
            state.completed_phases.append("finalization")
            state.current_phase = "completed"
            state.updated_at = datetime.now()
            
            # Store in memory
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="final_artifacts",
                content=final_package,
                metadata={"phase": "finalization", "timestamp": state.updated_at.isoformat()}
            )
            
        except Exception as e:
            state.errors.append({
                "phase": "finalization",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            state.current_phase = "error"
        
        return state
    
    def handle_error(self, state: FrontendDevelopmentState) -> FrontendDevelopmentState:
        """Handle errors and attempt recovery."""
        if state.errors:
            latest_error = state.errors[-1]
            
            # Log error for monitoring
            self.memory_service.store_conversation_memory(
                session_id=state.session_id or "frontend_dev",
                message_type="error",
                content=latest_error,
                metadata={"phase": "error_handling", "timestamp": datetime.now().isoformat()}
            )
            
            # Attempt to recover based on error type
            error_phase = latest_error.get("phase")
            if error_phase and error_phase in state.completed_phases:
                # Remove the failed phase and retry
                state.completed_phases.remove(error_phase)
                state.current_phase = error_phase
            else:
                # Critical error, end workflow
                state.current_phase = "completed"
        
        return state
    
    def should_continue(self, state: FrontendDevelopmentState) -> str:
        """Determine the next node based on current state."""
        if state.current_phase == "error":
            return "handle_error"
        elif state.current_phase == "completed":
            return END
        elif state.current_phase == "requirements_analysis":
            return "analyze_requirements"
        elif state.current_phase == "technology_selection":
            return "select_technology_stack"
        elif state.current_phase == "wireframe_design":
            return "create_wireframes"
        elif state.current_phase == "mockup_creation":
            return "create_mockups"
        elif state.current_phase == "prototype_development":
            return "develop_prototype"
        elif state.current_phase == "component_implementation":
            return "implement_components"
        elif state.current_phase == "integration_testing":
            return "perform_integration_testing"
        elif state.current_phase == "user_feedback":
            return "process_user_feedback"
        elif state.current_phase == "refinement":
            return "refine_implementation"
        elif state.current_phase == "finalization":
            return "finalize_implementation"
        else:
            return END
    
    def build_graph(self) -> Graph:
        """Build and return the frontend development graph."""
        # Create the state graph
        workflow = StateGraph(FrontendDevelopmentState)
        
        # Add nodes
        workflow.add_node("analyze_requirements", self.analyze_requirements)
        workflow.add_node("select_technology_stack", self.select_technology_stack)
        workflow.add_node("create_wireframes", self.create_wireframes)
        workflow.add_node("create_mockups", self.create_mockups)
        workflow.add_node("develop_prototype", self.develop_prototype)
        workflow.add_node("implement_components", self.implement_components)
        workflow.add_node("perform_integration_testing", self.perform_integration_testing)
        workflow.add_node("process_user_feedback", self.process_user_feedback)
        workflow.add_node("refine_implementation", self.refine_implementation)
        workflow.add_node("finalize_implementation", self.finalize_implementation)
        workflow.add_node("handle_error", self.handle_error)
        
        # Set entry point
        workflow.set_entry_point("analyze_requirements")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_requirements",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "select_technology_stack",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "create_wireframes",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "create_mockups",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "develop_prototype",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "implement_components",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "perform_integration_testing",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "process_user_feedback",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "refine_implementation",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "finalize_implementation",
            self.should_continue
        )
        workflow.add_conditional_edges(
            "handle_error",
            self.should_continue
        )
        
        # Compile and return the graph
        return workflow.compile()


def create_frontend_development_graph(
    llm_service: EnhancedLLMService,
    memory_service: MemoryService,
    state_service: StateService
) -> Graph:
    """Factory function to create a frontend development graph."""
    builder = FrontendDevelopmentGraphBuilder(llm_service, memory_service, state_service)
    return builder.build_graph()