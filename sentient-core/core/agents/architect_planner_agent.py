"""
The Architect Planner Agent
Synthesizes research and conversations into actionable plans.
"""

from core.models import AppState, EnhancedTask, AgentType, LogEntry
from core.services.llm_service import EnhancedLLMService
from core.agents.base_agent import BaseAgent, AgentCapability, ActivityType
from typing import Dict, Any, Optional, List
import json
import asyncio

class ArchitectPlannerAgent(BaseAgent):
    """
    The Architect Planner Agent is responsible for:
    - Synthesizing research findings and user conversations.
    - Creating high-level project plans.
    - Generating Product Requirements Documents (PRDs).
    - Defining the technical architecture and specifications.
    - Breaking down the high-level plan into detailed, buildable tasks.
    """

    def __init__(self, llm_service: EnhancedLLMService, agent_id: str = "architect_planner_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Architect Planner Agent",
            capabilities=[AgentCapability.PLANNING, AgentCapability.ANALYSIS],
            description="Specialized agent for project planning, architecture design, and requirement analysis"
        )
        self.llm_service = llm_service
        self.planning_methodologies = ['agile', 'waterfall', 'lean', 'design_thinking']
        self.architecture_patterns = ['microservices', 'monolith', 'serverless', 'event_driven', 'layered']
        self.documentation_types = ['prd', 'technical_spec', 'user_stories', 'acceptance_criteria']

    def can_handle_task(self, task: EnhancedTask) -> bool:
        """
        Determines if this agent can handle the given task.
        """
        planning_keywords = [
            'plan', 'planning', 'architecture', 'design', 'requirements', 'prd',
            'specification', 'blueprint', 'roadmap', 'strategy', 'analysis',
            'breakdown', 'structure', 'organize', 'synthesize', 'document'
        ]
        task_description = task.description.lower()
        return any(keyword in task_description for keyword in planning_keywords)
    
    async def process_task(self, task: EnhancedTask, state: Optional[AppState] = None) -> Dict:
        """
        Processes an architecture and planning task.
        """
        try:
            self.log_activity(ActivityType.TASK_STARTED, f"Processing planning task: {task.description}")
            
            # Determine the type of planning task
            task_type = self._determine_task_type(task.description)
            
            result = {}
            
            if task_type == 'prd':
                result = await self._handle_prd_task(task)
            elif task_type == 'architecture':
                result = await self._handle_architecture_task(task)
            elif task_type == 'breakdown':
                result = await self._handle_breakdown_task(task)
            elif task_type == 'requirements':
                result = await self._handle_requirements_task(task)
            elif task_type == 'synthesis':
                result = await self._handle_synthesis_task(task)
            elif task_type == 'roadmap':
                result = await self._handle_roadmap_task(task)
            else:
                result = await self._handle_general_planning_task(task)
            
            self.log_activity(ActivityType.TASK_COMPLETED, f"Planning task completed: {task.description}")
            
            return {
                "status": "completed",
                "task_type": task_type,
                "result": result
            }
            
        except Exception as e:
            self.handle_error(e, f"Error processing planning task: {task.description}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _determine_task_type(self, description: str) -> str:
        """
        Determines the type of planning task based on description.
        """
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['prd', 'product requirements', 'requirements document']):
            return 'prd'
        elif any(word in description_lower for word in ['architecture', 'system design', 'technical design']):
            return 'architecture'
        elif any(word in description_lower for word in ['breakdown', 'tasks', 'decompose', 'split']):
            return 'breakdown'
        elif any(word in description_lower for word in ['requirements', 'specs', 'specification']):
            return 'requirements'
        elif any(word in description_lower for word in ['synthesize', 'combine', 'merge', 'consolidate']):
            return 'synthesis'
        elif any(word in description_lower for word in ['roadmap', 'timeline', 'schedule', 'milestones']):
            return 'roadmap'
        else:
            return 'general'
    
    async def _handle_prd_task(self, task: EnhancedTask) -> Dict:
        """
        Handles Product Requirements Document creation tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Creating Product Requirements Document")
        
        try:
            prd_prompt = f"""
            Create a comprehensive Product Requirements Document (PRD) for: {task.description}
            
            Include the following sections:
            1. Executive Summary
            2. Problem Statement
            3. Goals and Objectives
            4. User Stories and Use Cases
            5. Functional Requirements
            6. Non-Functional Requirements
            7. Technical Constraints
            8. Success Metrics
            9. Timeline and Milestones
            10. Risk Assessment
            
            Provide detailed, actionable content for each section.
            """
            
            response = await self.llm_service.generate_response(
                prompt=prd_prompt,
                model="compound-beta"
            )
            
            return {
                "prd_document": response,
                "document_type": "Product Requirements Document",
                "sections": self._extract_prd_sections(response),
                "requirements_count": self._count_requirements(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in PRD task: {e}")
            raise
    
    async def _handle_architecture_task(self, task: EnhancedTask) -> Dict:
        """
        Handles system architecture design tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Designing system architecture")
        
        try:
            arch_prompt = f"""
            Design a comprehensive system architecture for: {task.description}
            
            Include:
            1. High-level architecture overview
            2. System components and their responsibilities
            3. Data flow diagrams
            4. Technology stack recommendations
            5. Scalability considerations
            6. Security architecture
            7. Integration points
            8. Deployment architecture
            9. Performance considerations
            10. Monitoring and observability
            
            Provide detailed architectural documentation with rationale for design decisions.
            """
            
            response = await self.llm_service.generate_response(
                prompt=arch_prompt,
                model="compound-beta"
            )
            
            return {
                "architecture_design": response,
                "architecture_pattern": self._identify_architecture_pattern(response),
                "components": self._extract_components(response),
                "technology_stack": self._extract_technology_stack(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in architecture task: {e}")
            raise
    
    async def _handle_breakdown_task(self, task: EnhancedTask) -> Dict:
        """
        Handles task breakdown and decomposition.
        """
        self.log_activity(ActivityType.PROCESSING, "Breaking down project into tasks")
        
        try:
            breakdown_prompt = f"""
            Break down the following project into detailed, actionable tasks: {task.description}
            
            For each task, provide:
            1. Task name and description
            2. Estimated effort (hours/days)
            3. Dependencies
            4. Required skills/expertise
            5. Acceptance criteria
            6. Priority level
            7. Risk factors
            
            Organize tasks into logical phases or sprints.
            Ensure tasks are specific, measurable, and achievable.
            """
            
            response = await self.llm_service.generate_response(
                prompt=breakdown_prompt,
                model="compound-beta"
            )
            
            return {
                "task_breakdown": response,
                "total_tasks": self._count_tasks(response),
                "phases": self._extract_phases(response),
                "estimated_effort": self._calculate_total_effort(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in breakdown task: {e}")
            raise
    
    async def _handle_requirements_task(self, task: EnhancedTask) -> Dict:
        """
        Handles requirements analysis and specification.
        """
        self.log_activity(ActivityType.PROCESSING, "Analyzing and documenting requirements")
        
        try:
            requirements_prompt = f"""
            Analyze and document detailed requirements for: {task.description}
            
            Include:
            1. Functional Requirements
               - User interactions
               - System behaviors
               - Business rules
            2. Non-Functional Requirements
               - Performance requirements
               - Security requirements
               - Usability requirements
               - Reliability requirements
            3. Technical Requirements
               - Platform requirements
               - Integration requirements
               - Data requirements
            4. Constraints and Assumptions
            5. Acceptance Criteria
            
            Use clear, testable language for all requirements.
            """
            
            response = await self.llm_service.generate_response(
                prompt=requirements_prompt,
                model="compound-beta"
            )
            
            return {
                "requirements_document": response,
                "functional_requirements": self._extract_functional_requirements(response),
                "non_functional_requirements": self._extract_non_functional_requirements(response),
                "constraints": self._extract_constraints(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in requirements task: {e}")
            raise
    
    async def _handle_synthesis_task(self, task: EnhancedTask) -> Dict:
        """
        Handles synthesis of research and information.
        """
        self.log_activity(ActivityType.PROCESSING, "Synthesizing research and information")
        
        try:
            synthesis_prompt = f"""
            Synthesize the following information into a coherent plan: {task.description}
            
            Provide:
            1. Key findings summary
            2. Patterns and insights
            3. Recommendations
            4. Action items
            5. Next steps
            6. Risk assessment
            7. Success factors
            
            Ensure the synthesis is actionable and provides clear direction.
            """
            
            response = await self.llm_service.generate_response(
                prompt=synthesis_prompt,
                model="compound-beta"
            )
            
            return {
                "synthesis_report": response,
                "key_insights": self._extract_insights(response),
                "recommendations": self._extract_recommendations(response),
                "action_items": self._extract_action_items(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in synthesis task: {e}")
            raise
    
    async def _handle_roadmap_task(self, task: EnhancedTask) -> Dict:
        """
        Handles roadmap and timeline creation.
        """
        self.log_activity(ActivityType.PROCESSING, "Creating project roadmap")
        
        try:
            roadmap_prompt = f"""
            Create a detailed project roadmap for: {task.description}
            
            Include:
            1. Project phases and milestones
            2. Timeline with dates
            3. Dependencies between phases
            4. Resource requirements
            5. Risk mitigation strategies
            6. Success criteria for each phase
            7. Review and checkpoint schedule
            
            Provide realistic timelines with buffer for unexpected challenges.
            """
            
            response = await self.llm_service.generate_response(
                prompt=roadmap_prompt,
                model="compound-beta"
            )
            
            return {
                "roadmap_document": response,
                "milestones": self._extract_milestones(response),
                "timeline": self._extract_timeline(response),
                "dependencies": self._extract_dependencies(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in roadmap task: {e}")
            raise
    
    async def _handle_general_planning_task(self, task: EnhancedTask) -> Dict:
        """
        Handles general planning tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Processing general planning task")
        
        try:
            general_prompt = f"""
            Create a comprehensive plan for: {task.description}
            
            Provide:
            1. Situation analysis
            2. Goals and objectives
            3. Strategy and approach
            4. Implementation plan
            5. Resource requirements
            6. Timeline
            7. Risk assessment
            8. Success metrics
            9. Monitoring and evaluation
            
            Ensure the plan is detailed, actionable, and realistic.
            """
            
            response = await self.llm_service.generate_response(
                prompt=general_prompt,
                model="compound-beta"
            )
            
            return {
                "planning_document": response,
                "objectives": self._extract_objectives(response),
                "strategies": self._extract_strategies(response),
                "success_metrics": self._extract_success_metrics(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in general planning task: {e}")
            raise
    
    def _extract_prd_sections(self, document: str) -> List[str]:
        """
        Extracts PRD sections from the document.
        """
        sections = []
        import re
        
        # Look for numbered sections or headers
        section_patterns = [
            r'\d+\.\s+([A-Z][^\n]+)',
            r'##\s+([A-Z][^\n]+)',
            r'\*\*([A-Z][^\*]+)\*\*'
        ]
        
        for pattern in section_patterns:
            matches = re.findall(pattern, document)
            sections.extend(matches)
        
        return list(set(sections))
    
    def _count_requirements(self, document: str) -> int:
        """
        Counts the number of requirements in the document.
        """
        import re
        
        # Look for requirement patterns
        requirement_patterns = [
            r'REQ-\d+',
            r'Requirement \d+',
            r'The system shall',
            r'The system must',
            r'The application should'
        ]
        
        count = 0
        for pattern in requirement_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            count += len(matches)
        
        return count
    
    def _identify_architecture_pattern(self, document: str) -> str:
        """
        Identifies the architecture pattern from the document.
        """
        document_lower = document.lower()
        
        for pattern in self.architecture_patterns:
            if pattern in document_lower:
                return pattern
        return 'layered'  # Default
    
    def _extract_components(self, document: str) -> List[str]:
        """
        Extracts system components from the architecture document.
        """
        components = []
        document_lower = document.lower()
        
        component_keywords = [
            'service', 'component', 'module', 'layer', 'gateway',
            'controller', 'repository', 'middleware', 'proxy', 'cache'
        ]
        
        for keyword in component_keywords:
            if keyword in document_lower:
                components.append(keyword.title())
        
        return list(set(components))
    
    def _extract_technology_stack(self, document: str) -> List[str]:
        """
        Extracts technology stack from the document.
        """
        technologies = []
        document_lower = document.lower()
        
        tech_keywords = [
            'react', 'vue', 'angular', 'python', 'java', 'node.js',
            'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes',
            'aws', 'azure', 'gcp', 'fastapi', 'django', 'flask'
        ]
        
        for tech in tech_keywords:
            if tech in document_lower:
                technologies.append(tech)
        
        return technologies
    
    def _count_tasks(self, document: str) -> int:
        """
        Counts the number of tasks in the breakdown.
        """
        import re
        
        # Look for task patterns
        task_patterns = [
            r'Task \d+',
            r'\d+\.\s+[A-Z]',
            r'-\s+[A-Z]',
            r'\*\s+[A-Z]'
        ]
        
        count = 0
        for pattern in task_patterns:
            matches = re.findall(pattern, document)
            count += len(matches)
        
        return count
    
    def _extract_phases(self, document: str) -> List[str]:
        """
        Extracts project phases from the document.
        """
        phases = []
        document_lower = document.lower()
        
        phase_keywords = [
            'phase', 'sprint', 'iteration', 'milestone',
            'stage', 'step', 'cycle'
        ]
        
        for keyword in phase_keywords:
            if keyword in document_lower:
                phases.append(keyword.title())
        
        return list(set(phases))
    
    def _calculate_total_effort(self, document: str) -> str:
        """
        Calculates total effort from the document.
        """
        import re
        
        # Look for effort estimates
        effort_patterns = [
            r'(\d+)\s+hours?',
            r'(\d+)\s+days?',
            r'(\d+)\s+weeks?'
        ]
        
        total_hours = 0
        
        for pattern in effort_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            for match in matches:
                hours = int(match)
                if 'day' in pattern:
                    hours *= 8
                elif 'week' in pattern:
                    hours *= 40
                total_hours += hours
        
        if total_hours > 0:
            days = total_hours / 8
            weeks = days / 5
            return f"{total_hours} hours ({days:.1f} days, {weeks:.1f} weeks)"
        
        return "Effort not specified"
    
    def _extract_functional_requirements(self, document: str) -> List[str]:
        """
        Extracts functional requirements from the document.
        """
        requirements = []
        import re
        
        # Look for functional requirement patterns
        patterns = [
            r'The system shall ([^.]+)',
            r'The application must ([^.]+)',
            r'FR-\d+: ([^\n]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            requirements.extend(matches)
        
        return requirements
    
    def _extract_non_functional_requirements(self, document: str) -> List[str]:
        """
        Extracts non-functional requirements from the document.
        """
        requirements = []
        document_lower = document.lower()
        
        nfr_keywords = {
            'performance': 'Performance Requirements',
            'security': 'Security Requirements',
            'usability': 'Usability Requirements',
            'reliability': 'Reliability Requirements',
            'scalability': 'Scalability Requirements'
        }
        
        for keyword, requirement in nfr_keywords.items():
            if keyword in document_lower:
                requirements.append(requirement)
        
        return requirements
    
    def _extract_constraints(self, document: str) -> List[str]:
        """
        Extracts constraints from the document.
        """
        constraints = []
        document_lower = document.lower()
        
        constraint_keywords = [
            'budget', 'timeline', 'technology', 'resource',
            'compliance', 'regulation', 'platform'
        ]
        
        for keyword in constraint_keywords:
            if keyword in document_lower:
                constraints.append(f"{keyword.title()} Constraint")
        
        return list(set(constraints))
    
    def _extract_insights(self, document: str) -> List[str]:
        """
        Extracts key insights from the synthesis.
        """
        insights = []
        import re
        
        # Look for insight patterns
        insight_patterns = [
            r'Key insight: ([^\n]+)',
            r'Important finding: ([^\n]+)',
            r'Critical observation: ([^\n]+)'
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            insights.extend(matches)
        
        return insights
    
    def _extract_recommendations(self, document: str) -> List[str]:
        """
        Extracts recommendations from the document.
        """
        recommendations = []
        import re
        
        # Look for recommendation patterns
        rec_patterns = [
            r'Recommendation: ([^\n]+)',
            r'We recommend ([^\n]+)',
            r'It is suggested ([^\n]+)'
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            recommendations.extend(matches)
        
        return recommendations
    
    def _extract_action_items(self, document: str) -> List[str]:
        """
        Extracts action items from the document.
        """
        actions = []
        import re
        
        # Look for action item patterns
        action_patterns = [
            r'Action: ([^\n]+)',
            r'Next step: ([^\n]+)',
            r'TODO: ([^\n]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            actions.extend(matches)
        
        return actions
    
    def _extract_milestones(self, document: str) -> List[str]:
        """
        Extracts milestones from the roadmap.
        """
        milestones = []
        import re
        
        # Look for milestone patterns
        milestone_patterns = [
            r'Milestone: ([^\n]+)',
            r'M\d+: ([^\n]+)',
            r'Phase \d+ completion: ([^\n]+)'
        ]
        
        for pattern in milestone_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            milestones.extend(matches)
        
        return milestones
    
    def _extract_timeline(self, document: str) -> str:
        """
        Extracts timeline information from the document.
        """
        import re
        
        # Look for timeline patterns
        timeline_patterns = [
            r'Timeline: ([^\n]+)',
            r'Duration: ([^\n]+)',
            r'Expected completion: ([^\n]+)'
        ]
        
        for pattern in timeline_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            if matches:
                return matches[0]
        
        return "Timeline not specified"
    
    def _extract_dependencies(self, document: str) -> List[str]:
        """
        Extracts dependencies from the document.
        """
        dependencies = []
        import re
        
        # Look for dependency patterns
        dep_patterns = [
            r'Depends on: ([^\n]+)',
            r'Prerequisite: ([^\n]+)',
            r'Requires: ([^\n]+)'
        ]
        
        for pattern in dep_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            dependencies.extend(matches)
        
        return dependencies
    
    def _extract_objectives(self, document: str) -> List[str]:
        """
        Extracts objectives from the planning document.
        """
        objectives = []
        import re
        
        # Look for objective patterns
        obj_patterns = [
            r'Objective: ([^\n]+)',
            r'Goal: ([^\n]+)',
            r'Aim: ([^\n]+)'
        ]
        
        for pattern in obj_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            objectives.extend(matches)
        
        return objectives
    
    def _extract_strategies(self, document: str) -> List[str]:
        """
        Extracts strategies from the document.
        """
        strategies = []
        import re
        
        # Look for strategy patterns
        strategy_patterns = [
            r'Strategy: ([^\n]+)',
            r'Approach: ([^\n]+)',
            r'Method: ([^\n]+)'
        ]
        
        for pattern in strategy_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            strategies.extend(matches)
        
        return strategies
    
    def _extract_success_metrics(self, document: str) -> List[str]:
        """
        Extracts success metrics from the document.
        """
        metrics = []
        import re
        
        # Look for metric patterns
        metric_patterns = [
            r'Metric: ([^\n]+)',
            r'KPI: ([^\n]+)',
            r'Success criteria: ([^\n]+)'
        ]
        
        for pattern in metric_patterns:
            matches = re.findall(pattern, document, re.IGNORECASE)
            metrics.extend(matches)
        
        return metrics

    def invoke(self, state: AppState) -> AppState:
        """
        Legacy method for backward compatibility.
        The main entry point for the architect's planning process.
        """
        print("---ARCHITECT PLANNER AGENT---")
        
        self.log_activity(ActivityType.PROCESSING, "Legacy invoke method called")
        
        # Placeholder logic
        state.logs.append(LogEntry(
            source="ArchitectPlannerAgent",
            message="Architect planning logic not yet implemented."
        ))
        
        return state