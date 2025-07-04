"""Project Architecture Graph Implementation

This module implements the project architecture graph that defines the flow for
generating high-level architectural designs and specifications.
"""

import logging
from typing import Dict, Any, List, Optional, Literal
from langgraph import StateGraph, END
from langgraph.graph import CompiledGraph
from pydantic import BaseModel, Field

from ..models import AppState, TaskStatus, AgentType
from ..agents.architect_planner import ArchitectPlanner
from ..agents.research_agent import ResearchAgent
from ..services.enhanced_llm_service import EnhancedLLMService
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ArchitecturalComponent(BaseModel):
    """Model for architectural components."""
    name: str
    type: Literal["frontend", "backend", "database", "api", "service", "infrastructure"]
    description: str
    dependencies: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    complexity_score: float = 0.0
    estimated_effort: str = ""

class TechnicalSpecification(BaseModel):
    """Model for technical specifications."""
    component: str
    specification_type: Literal["api", "database", "interface", "security", "performance"]
    details: Dict[str, Any] = Field(default_factory=dict)
    requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)

class ProjectArchitectureState(BaseModel):
    """State model for the project architecture graph."""
    
    # Input requirements
    project_requirements: Dict[str, Any] = Field(default_factory=dict)
    user_constraints: List[str] = Field(default_factory=list)
    target_platforms: List[str] = Field(default_factory=list)
    
    # Architecture phase
    architecture_phase: Literal[
        "requirements_analysis",
        "technology_research",
        "architecture_design",
        "component_specification",
        "integration_planning",
        "validation",
        "documentation"
    ] = "requirements_analysis"
    
    # Research and analysis
    technology_research: Dict[str, Any] = Field(default_factory=dict)
    best_practices: List[Dict[str, Any]] = Field(default_factory=list)
    architecture_patterns: List[str] = Field(default_factory=list)
    
    # Architecture design
    high_level_architecture: Dict[str, Any] = Field(default_factory=dict)
    system_components: List[ArchitecturalComponent] = Field(default_factory=list)
    data_flow: Dict[str, Any] = Field(default_factory=dict)
    
    # Technical specifications
    technical_specifications: List[TechnicalSpecification] = Field(default_factory=list)
    api_specifications: Dict[str, Any] = Field(default_factory=dict)
    database_schema: Dict[str, Any] = Field(default_factory=dict)
    
    # Planning outputs
    high_level_plan: Dict[str, Any] = Field(default_factory=dict)
    prd_document: Dict[str, Any] = Field(default_factory=dict)
    action_plan: Dict[str, Any] = Field(default_factory=dict)
    
    # Validation and feedback
    architecture_validation: Dict[str, Any] = Field(default_factory=dict)
    identified_risks: List[Dict[str, Any]] = Field(default_factory=list)
    mitigation_strategies: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Progress tracking
    completed_phases: List[str] = Field(default_factory=list)
    current_deliverables: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Error handling
    validation_errors: List[str] = Field(default_factory=list)
    architecture_warnings: List[str] = Field(default_factory=list)

class ProjectArchitectureGraphBuilder:
    """Builder class for creating the project architecture graph."""
    
    def __init__(self, llm_service: EnhancedLLMService):
        self.llm_service = llm_service
        self.architect_planner = ArchitectPlanner(llm_service)
        self.research_agent = ResearchAgent(llm_service)
        
    def create_graph(self) -> CompiledGraph:
        """Create and compile the project architecture graph."""
        
        # Create the state graph
        workflow = StateGraph(ProjectArchitectureState)
        
        # Add nodes
        workflow.add_node("analyze_requirements", self._analyze_requirements_node)
        workflow.add_node("research_technologies", self._research_technologies_node)
        workflow.add_node("design_architecture", self._design_architecture_node)
        workflow.add_node("specify_components", self._specify_components_node)
        workflow.add_node("plan_integration", self._plan_integration_node)
        workflow.add_node("validate_architecture", self._validate_architecture_node)
        workflow.add_node("generate_documentation", self._generate_documentation_node)
        workflow.add_node("refine_architecture", self._refine_architecture_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_requirements")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_requirements",
            self._route_from_requirements_analysis,
            {
                "research": "research_technologies",
                "design": "design_architecture",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "research_technologies",
            self._route_from_technology_research,
            {
                "design": "design_architecture",
                "continue_research": "research_technologies",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "design_architecture",
            self._route_from_architecture_design,
            {
                "specify": "specify_components",
                "research_more": "research_technologies",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "specify_components",
            self._route_from_component_specification,
            {
                "plan_integration": "plan_integration",
                "refine_design": "design_architecture",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "plan_integration",
            self._route_from_integration_planning,
            {
                "validate": "validate_architecture",
                "refine_components": "specify_components",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "validate_architecture",
            self._route_from_validation,
            {
                "document": "generate_documentation",
                "refine": "refine_architecture",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "refine_architecture",
            self._route_from_refinement,
            {
                "design": "design_architecture",
                "specify": "specify_components",
                "validate": "validate_architecture",
                "error": END
            }
        )
        
        workflow.add_edge("generate_documentation", END)
        
        return workflow.compile()
    
    async def _analyze_requirements_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Analyze project requirements and constraints."""
        try:
            logger.info("Starting requirements analysis")
            
            state.architecture_phase = "requirements_analysis"
            
            # Analyze functional requirements
            functional_analysis = await self.architect_planner.analyze_functional_requirements(
                state.project_requirements
            )
            
            # Analyze non-functional requirements
            nonfunctional_analysis = await self.architect_planner.analyze_nonfunctional_requirements(
                state.project_requirements,
                state.user_constraints
            )
            
            # Determine target platforms and constraints
            platform_analysis = await self.architect_planner.analyze_platform_requirements(
                state.target_platforms,
                state.project_requirements
            )
            
            # Update state with analysis results
            state.project_requirements.update({
                "functional_analysis": functional_analysis,
                "nonfunctional_analysis": nonfunctional_analysis,
                "platform_analysis": platform_analysis
            })
            
            state.completed_phases.append("requirements_analysis")
            
            logger.info("Requirements analysis completed")
            
        except Exception as e:
            logger.error(f"Error in requirements analysis: {str(e)}")
            state.validation_errors.append(f"Requirements analysis failed: {str(e)}")
            
        return state
    
    async def _research_technologies_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Research appropriate technologies and architecture patterns."""
        try:
            logger.info("Starting technology research")
            
            state.architecture_phase = "technology_research"
            
            # Research technology stack options
            tech_research = await self.research_agent.research_technology_stack(
                state.project_requirements,
                state.target_platforms
            )
            
            # Research architecture patterns
            pattern_research = await self.research_agent.research_architecture_patterns(
                state.project_requirements["functional_analysis"]
            )
            
            # Research best practices
            best_practices = await self.research_agent.research_best_practices(
                tech_research["recommended_technologies"],
                state.project_requirements["nonfunctional_analysis"]
            )
            
            # Update state with research results
            state.technology_research = tech_research
            state.architecture_patterns = pattern_research["recommended_patterns"]
            state.best_practices = best_practices
            
            state.completed_phases.append("technology_research")
            
            logger.info("Technology research completed")
            
        except Exception as e:
            logger.error(f"Error in technology research: {str(e)}")
            state.validation_errors.append(f"Technology research failed: {str(e)}")
            
        return state
    
    async def _design_architecture_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Design the high-level system architecture."""
        try:
            logger.info("Starting architecture design")
            
            state.architecture_phase = "architecture_design"
            
            # Design high-level architecture
            architecture_design = await self.architect_planner.design_system_architecture(
                state.project_requirements,
                state.technology_research,
                state.architecture_patterns
            )
            
            # Define system components
            components = await self.architect_planner.define_system_components(
                architecture_design,
                state.technology_research["recommended_technologies"]
            )
            
            # Design data flow
            data_flow = await self.architect_planner.design_data_flow(
                components,
                state.project_requirements["functional_analysis"]
            )
            
            # Update state with design results
            state.high_level_architecture = architecture_design
            state.system_components = [
                ArchitecturalComponent(**comp) for comp in components
            ]
            state.data_flow = data_flow
            
            state.completed_phases.append("architecture_design")
            
            logger.info("Architecture design completed")
            
        except Exception as e:
            logger.error(f"Error in architecture design: {str(e)}")
            state.validation_errors.append(f"Architecture design failed: {str(e)}")
            
        return state
    
    async def _specify_components_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Create detailed specifications for system components."""
        try:
            logger.info("Starting component specification")
            
            state.architecture_phase = "component_specification"
            
            # Generate API specifications
            api_specs = await self.architect_planner.generate_api_specifications(
                state.system_components,
                state.data_flow
            )
            
            # Generate database schema
            db_schema = await self.architect_planner.generate_database_schema(
                state.system_components,
                state.project_requirements["functional_analysis"]
            )
            
            # Generate technical specifications
            tech_specs = await self.architect_planner.generate_technical_specifications(
                state.system_components,
                state.best_practices
            )
            
            # Update state with specifications
            state.api_specifications = api_specs
            state.database_schema = db_schema
            state.technical_specifications = [
                TechnicalSpecification(**spec) for spec in tech_specs
            ]
            
            state.completed_phases.append("component_specification")
            
            logger.info("Component specification completed")
            
        except Exception as e:
            logger.error(f"Error in component specification: {str(e)}")
            state.validation_errors.append(f"Component specification failed: {str(e)}")
            
        return state
    
    async def _plan_integration_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Plan integration between system components."""
        try:
            logger.info("Starting integration planning")
            
            state.architecture_phase = "integration_planning"
            
            # Plan component integration
            integration_plan = await self.architect_planner.plan_component_integration(
                state.system_components,
                state.api_specifications,
                state.data_flow
            )
            
            # Identify integration challenges
            integration_challenges = await self.architect_planner.identify_integration_challenges(
                state.system_components,
                state.technical_specifications
            )
            
            # Plan deployment strategy
            deployment_plan = await self.architect_planner.plan_deployment_strategy(
                state.system_components,
                state.target_platforms,
                state.project_requirements["nonfunctional_analysis"]
            )
            
            # Update state with integration planning
            state.high_level_architecture.update({
                "integration_plan": integration_plan,
                "integration_challenges": integration_challenges,
                "deployment_plan": deployment_plan
            })
            
            state.completed_phases.append("integration_planning")
            
            logger.info("Integration planning completed")
            
        except Exception as e:
            logger.error(f"Error in integration planning: {str(e)}")
            state.validation_errors.append(f"Integration planning failed: {str(e)}")
            
        return state
    
    async def _validate_architecture_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Validate the architecture design and identify risks."""
        try:
            logger.info("Starting architecture validation")
            
            state.architecture_phase = "validation"
            
            # Validate architecture completeness
            completeness_validation = await self.architect_planner.validate_architecture_completeness(
                state.high_level_architecture,
                state.system_components,
                state.project_requirements
            )
            
            # Validate technical feasibility
            feasibility_validation = await self.architect_planner.validate_technical_feasibility(
                state.system_components,
                state.technical_specifications,
                state.technology_research
            )
            
            # Identify risks and mitigation strategies
            risk_analysis = await self.architect_planner.analyze_architecture_risks(
                state.high_level_architecture,
                state.system_components,
                state.project_requirements
            )
            
            # Update state with validation results
            state.architecture_validation = {
                "completeness": completeness_validation,
                "feasibility": feasibility_validation,
                "overall_score": (completeness_validation["score"] + feasibility_validation["score"]) / 2
            }
            
            state.identified_risks = risk_analysis["risks"]
            state.mitigation_strategies = risk_analysis["mitigation_strategies"]
            
            state.completed_phases.append("validation")
            
            logger.info("Architecture validation completed")
            
        except Exception as e:
            logger.error(f"Error in architecture validation: {str(e)}")
            state.validation_errors.append(f"Architecture validation failed: {str(e)}")
            
        return state
    
    async def _generate_documentation_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Generate comprehensive architecture documentation."""
        try:
            logger.info("Starting documentation generation")
            
            state.architecture_phase = "documentation"
            
            # Generate high-level plan
            high_level_plan = await self.architect_planner.generate_high_level_plan(
                state.high_level_architecture,
                state.system_components,
                state.project_requirements
            )
            
            # Generate PRD document
            prd_document = await self.architect_planner.generate_prd_document(
                state.project_requirements,
                state.high_level_architecture,
                state.technical_specifications
            )
            
            # Generate action plan
            action_plan = await self.architect_planner.generate_action_plan(
                state.system_components,
                state.high_level_architecture["integration_plan"],
                state.mitigation_strategies
            )
            
            # Update state with documentation
            state.high_level_plan = high_level_plan
            state.prd_document = prd_document
            state.action_plan = action_plan
            
            # Generate deliverables
            state.current_deliverables = [
                {
                    "type": "high_level_plan",
                    "content": high_level_plan,
                    "format": "markdown"
                },
                {
                    "type": "prd_document",
                    "content": prd_document,
                    "format": "markdown"
                },
                {
                    "type": "action_plan",
                    "content": action_plan,
                    "format": "markdown"
                },
                {
                    "type": "technical_specifications",
                    "content": [spec.dict() for spec in state.technical_specifications],
                    "format": "json"
                },
                {
                    "type": "api_specifications",
                    "content": state.api_specifications,
                    "format": "json"
                },
                {
                    "type": "database_schema",
                    "content": state.database_schema,
                    "format": "json"
                }
            ]
            
            state.completed_phases.append("documentation")
            
            logger.info("Documentation generation completed")
            
        except Exception as e:
            logger.error(f"Error in documentation generation: {str(e)}")
            state.validation_errors.append(f"Documentation generation failed: {str(e)}")
            
        return state
    
    async def _refine_architecture_node(self, state: ProjectArchitectureState) -> ProjectArchitectureState:
        """Refine architecture based on validation feedback."""
        try:
            logger.info("Starting architecture refinement")
            
            # Analyze validation issues
            refinement_plan = await self.architect_planner.create_refinement_plan(
                state.architecture_validation,
                state.identified_risks,
                state.validation_errors
            )
            
            # Apply refinements
            if refinement_plan["refine_design"]:
                state.architecture_phase = "architecture_design"
            elif refinement_plan["refine_components"]:
                state.architecture_phase = "component_specification"
            else:
                state.architecture_phase = "validation"
            
            # Clear previous errors if refinement addresses them
            if refinement_plan["addresses_errors"]:
                state.validation_errors = []
                state.architecture_warnings = []
            
            logger.info(f"Architecture refinement planned. Next phase: {state.architecture_phase}")
            
        except Exception as e:
            logger.error(f"Error in architecture refinement: {str(e)}")
            state.validation_errors.append(f"Architecture refinement failed: {str(e)}")
            
        return state
    
    def _route_from_requirements_analysis(self, state: ProjectArchitectureState) -> str:
        """Route from requirements analysis."""
        if state.validation_errors:
            return "error"
        elif not state.technology_research:
            return "research"
        else:
            return "design"
    
    def _route_from_technology_research(self, state: ProjectArchitectureState) -> str:
        """Route from technology research."""
        if state.validation_errors:
            return "error"
        elif not state.technology_research.get("research_complete", False):
            return "continue_research"
        else:
            return "design"
    
    def _route_from_architecture_design(self, state: ProjectArchitectureState) -> str:
        """Route from architecture design."""
        if state.validation_errors:
            return "error"
        elif not state.high_level_architecture or not state.system_components:
            return "research_more"
        else:
            return "specify"
    
    def _route_from_component_specification(self, state: ProjectArchitectureState) -> str:
        """Route from component specification."""
        if state.validation_errors:
            return "error"
        elif not state.technical_specifications or not state.api_specifications:
            return "refine_design"
        else:
            return "plan_integration"
    
    def _route_from_integration_planning(self, state: ProjectArchitectureState) -> str:
        """Route from integration planning."""
        if state.validation_errors:
            return "error"
        elif not state.high_level_architecture.get("integration_plan"):
            return "refine_components"
        else:
            return "validate"
    
    def _route_from_validation(self, state: ProjectArchitectureState) -> str:
        """Route from validation."""
        if state.validation_errors:
            return "error"
        elif (state.architecture_validation.get("overall_score", 0) < 0.7 or 
              len(state.identified_risks) > 5):
            return "refine"
        else:
            return "document"
    
    def _route_from_refinement(self, state: ProjectArchitectureState) -> str:
        """Route from refinement."""
        if state.validation_errors:
            return "error"
        elif state.architecture_phase == "architecture_design":
            return "design"
        elif state.architecture_phase == "component_specification":
            return "specify"
        else:
            return "validate"

# Factory function for creating project architecture graph
def create_project_architecture_graph(llm_service: EnhancedLLMService) -> CompiledGraph:
    """Create and return a compiled project architecture graph."""
    builder = ProjectArchitectureGraphBuilder(llm_service)
    return builder.create_graph()