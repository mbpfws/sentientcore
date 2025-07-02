# 06 - Architect Planner Agent Enhancement

## Overview

The Architect Planner Agent serves as the system design and architecture planning hub for the multi-agent system. It analyzes requirements, creates comprehensive system architectures, generates technical specifications, and provides detailed implementation roadmaps. This phase transforms the existing basic architect planner into a sophisticated system design and planning agent.

## Current State Analysis

### Existing File
- `core/agents/architect_planner_agent.py` - Basic architecture planning functionality

### Enhancement Requirements
- Advanced system architecture design
- Technology stack analysis and recommendations
- Database schema design
- API specification generation
- Security architecture planning
- Scalability and performance planning
- Integration with research findings
- Visual architecture diagram generation

## Implementation Tasks

### Task 6.1: Enhanced Architect Planner Agent

**File**: `core/agents/architect_planner_agent.py` (Complete Rewrite)

**Architect Planner Agent Implementation**:
```python
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
from enum import Enum
import json
import yaml

from .base_agent import BaseAgent, AgentStatus
from ..services.llm_service import LLMService
from ..services.memory_service import MemoryService
from ..models import ArchitectureTask, SystemArchitecture, TechStackRecommendation

class ArchitectureType(Enum):
    MICROSERVICES = "microservices"
    MONOLITHIC = "monolithic"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"
    CLEAN_ARCHITECTURE = "clean_architecture"
    DOMAIN_DRIVEN = "domain_driven"

class ProjectScale(Enum):
    SMALL = "small"  # 1-3 developers, simple requirements
    MEDIUM = "medium"  # 4-10 developers, moderate complexity
    LARGE = "large"  # 10+ developers, complex requirements
    ENTERPRISE = "enterprise"  # Large scale, high complexity

class ArchitectPlannerAgent(BaseAgent):
    def __init__(self, agent_id: str = "architect_planner_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Architect Planner Agent",
            description="Advanced system architecture design and planning agent"
        )
        self.capabilities = [
            "system_architecture_design",
            "technology_stack_analysis",
            "database_schema_design",
            "api_specification_generation",
            "security_architecture_planning",
            "scalability_planning",
            "performance_optimization",
            "integration_planning",
            "deployment_architecture",
            "documentation_generation"
        ]
        
        self.architecture_templates = {}
        self.tech_stack_knowledge = {}
        self.design_patterns = {}
        self.active_planning_sessions = {}
        
    async def initialize(self, llm_service: LLMService, memory_service: MemoryService):
        """Initialize architect planner agent"""
        self.llm_service = llm_service
        self.memory_service = memory_service
        await self._load_architecture_templates()
        await self._load_tech_stack_knowledge()
        await self._load_design_patterns()
        await self.update_status(AgentStatus.IDLE)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process architecture planning task"""
        try:
            await self.update_status(AgentStatus.THINKING)
            
            # Parse architecture task
            arch_task = self._parse_architecture_task(task)
            
            # Create planning session
            session_id = f"arch_plan_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.active_planning_sessions[session_id] = {
                'task': arch_task,
                'start_time': datetime.utcnow(),
                'status': 'active',
                'artifacts': []
            }
            
            await self.update_status(AgentStatus.WORKING)
            
            # Execute architecture planning workflow
            planning_result = await self._execute_architecture_planning(session_id, arch_task)
            
            # Generate comprehensive architecture documentation
            documentation = await self._generate_architecture_documentation(planning_result)
            
            # Store architecture artifacts
            await self._store_architecture_artifacts(session_id, planning_result, documentation)
            
            await self.update_status(AgentStatus.COMPLETED)
            
            return {
                'session_id': session_id,
                'architecture_type': planning_result.get('architecture_type'),
                'system_architecture': planning_result.get('system_architecture'),
                'tech_stack': planning_result.get('tech_stack'),
                'database_design': planning_result.get('database_design'),
                'api_specifications': planning_result.get('api_specifications'),
                'security_architecture': planning_result.get('security_architecture'),
                'deployment_architecture': planning_result.get('deployment_architecture'),
                'implementation_roadmap': planning_result.get('implementation_roadmap'),
                'documentation': documentation,
                'confidence_score': planning_result.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            await self.update_status(AgentStatus.ERROR, str(e))
            raise
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Determine if agent can handle architecture planning task"""
        return task.get('type') in [
            'architecture_planning',
            'system_design',
            'tech_stack_analysis',
            'database_design',
            'api_design',
            'security_planning',
            'deployment_planning'
        ]
    
    def _parse_architecture_task(self, task: Dict[str, Any]) -> 'ArchitectureTask':
        """Parse incoming task into structured architecture task"""
        return ArchitectureTask(
            project_name=task.get('project_name', 'Unnamed Project'),
            description=task.get('description', ''),
            requirements=task.get('requirements', []),
            constraints=task.get('constraints', []),
            scale=ProjectScale(task.get('scale', 'medium')),
            target_users=task.get('target_users', 100),
            performance_requirements=task.get('performance_requirements', {}),
            security_requirements=task.get('security_requirements', []),
            integration_requirements=task.get('integration_requirements', []),
            technology_preferences=task.get('technology_preferences', {}),
            budget_constraints=task.get('budget_constraints', {}),
            timeline=task.get('timeline', {}),
            existing_systems=task.get('existing_systems', []),
            research_context=task.get('research_context', {})
        )
    
    async def _execute_architecture_planning(self, session_id: str, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        """Execute comprehensive architecture planning workflow"""
        planning_result = {}
        
        # Step 1: Requirements Analysis
        await self.log_activity(f"Analyzing requirements for {arch_task.project_name}")
        requirements_analysis = await self._analyze_requirements(arch_task)
        planning_result['requirements_analysis'] = requirements_analysis
        
        # Step 2: Architecture Type Selection
        await self.log_activity("Selecting optimal architecture type")
        architecture_type = await self._select_architecture_type(arch_task, requirements_analysis)
        planning_result['architecture_type'] = architecture_type
        
        # Step 3: Technology Stack Analysis
        await self.log_activity("Analyzing and recommending technology stack")
        tech_stack = await self._analyze_technology_stack(arch_task, architecture_type)
        planning_result['tech_stack'] = tech_stack
        
        # Step 4: System Architecture Design
        await self.log_activity("Designing system architecture")
        system_architecture = await self._design_system_architecture(arch_task, architecture_type, tech_stack)
        planning_result['system_architecture'] = system_architecture
        
        # Step 5: Database Design
        await self.log_activity("Designing database architecture")
        database_design = await self._design_database_architecture(arch_task, system_architecture)
        planning_result['database_design'] = database_design
        
        # Step 6: API Specifications
        await self.log_activity("Generating API specifications")
        api_specifications = await self._generate_api_specifications(arch_task, system_architecture)
        planning_result['api_specifications'] = api_specifications
        
        # Step 7: Security Architecture
        await self.log_activity("Planning security architecture")
        security_architecture = await self._plan_security_architecture(arch_task, system_architecture)
        planning_result['security_architecture'] = security_architecture
        
        # Step 8: Deployment Architecture
        await self.log_activity("Designing deployment architecture")
        deployment_architecture = await self._design_deployment_architecture(arch_task, system_architecture, tech_stack)
        planning_result['deployment_architecture'] = deployment_architecture
        
        # Step 9: Implementation Roadmap
        await self.log_activity("Creating implementation roadmap")
        implementation_roadmap = await self._create_implementation_roadmap(arch_task, planning_result)
        planning_result['implementation_roadmap'] = implementation_roadmap
        
        # Step 10: Quality Assessment
        await self.log_activity("Assessing architecture quality")
        quality_assessment = await self._assess_architecture_quality(planning_result)
        planning_result['quality_assessment'] = quality_assessment
        planning_result['confidence_score'] = quality_assessment.get('overall_score', 0.0)
        
        return planning_result
    
    async def _analyze_requirements(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        """Analyze and categorize project requirements"""
        prompt = f"""
        Analyze the following project requirements and categorize them:
        
        Project: {arch_task.project_name}
        Description: {arch_task.description}
        Requirements: {arch_task.requirements}
        Constraints: {arch_task.constraints}
        Scale: {arch_task.scale.value}
        Target Users: {arch_task.target_users}
        Performance Requirements: {arch_task.performance_requirements}
        Security Requirements: {arch_task.security_requirements}
        Integration Requirements: {arch_task.integration_requirements}
        
        Categorize and analyze these requirements into:
        1. Functional Requirements (what the system should do)
        2. Non-Functional Requirements (performance, security, scalability)
        3. Technical Constraints (technology limitations, existing systems)
        4. Business Constraints (budget, timeline, resources)
        5. Quality Attributes (reliability, maintainability, usability)
        6. Integration Requirements (external systems, APIs, data sources)
        
        For each category, provide:
        - Priority level (High, Medium, Low)
        - Complexity assessment (Simple, Moderate, Complex)
        - Impact on architecture decisions
        - Potential risks or challenges
        
        Return as JSON with this structure:
        {{
            "functional_requirements": [
                {{"requirement": "...", "priority": "High", "complexity": "Moderate", "impact": "...", "risks": [...]}}
            ],
            "non_functional_requirements": [...],
            "technical_constraints": [...],
            "business_constraints": [...],
            "quality_attributes": [...],
            "integration_requirements": [...],
            "analysis_summary": "...",
            "key_challenges": [...],
            "architecture_drivers": [...]
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_requirements_analysis(arch_task)
    
    async def _select_architecture_type(self, arch_task: 'ArchitectureTask', requirements_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal architecture type based on requirements"""
        prompt = f"""
        Based on the following project characteristics and requirements analysis, recommend the most suitable architecture type:
        
        Project Scale: {arch_task.scale.value}
        Target Users: {arch_task.target_users}
        Requirements Analysis: {json.dumps(requirements_analysis, indent=2)}
        Technology Preferences: {arch_task.technology_preferences}
        
        Consider these architecture types:
        1. Microservices - Good for large scale, team independence, technology diversity
        2. Monolithic - Good for small-medium scale, simple deployment, rapid development
        3. Serverless - Good for event-driven, auto-scaling, cost optimization
        4. Event-Driven - Good for real-time processing, loose coupling, scalability
        5. Layered - Good for traditional applications, clear separation of concerns
        6. Hexagonal - Good for testability, external system integration
        7. Clean Architecture - Good for maintainability, business logic isolation
        8. Domain-Driven - Good for complex business domains, large teams
        
        Provide recommendation with:
        - Primary architecture type
        - Secondary/hybrid options if applicable
        - Detailed justification
        - Trade-offs and considerations
        - Implementation complexity
        - Scalability implications
        - Team structure requirements
        
        Return as JSON:
        {{
            "primary_architecture": "microservices",
            "secondary_options": ["event_driven"],
            "justification": "...",
            "trade_offs": {{
                "advantages": [...],
                "disadvantages": [...],
                "risks": [...]
            }},
            "implementation_complexity": "high",
            "scalability_rating": 9,
            "team_requirements": "...",
            "recommended_patterns": [...]
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.3
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_architecture_type(arch_task)
    
    async def _analyze_technology_stack(self, arch_task: 'ArchitectureTask', architecture_type: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and recommend comprehensive technology stack"""
        prompt = f"""
        Recommend a comprehensive technology stack for the following project:
        
        Project: {arch_task.project_name}
        Scale: {arch_task.scale.value}
        Architecture Type: {architecture_type.get('primary_architecture')}
        Technology Preferences: {arch_task.technology_preferences}
        Performance Requirements: {arch_task.performance_requirements}
        Integration Requirements: {arch_task.integration_requirements}
        
        Provide recommendations for:
        1. Frontend Technologies (frameworks, libraries, tools)
        2. Backend Technologies (languages, frameworks, runtime)
        3. Database Technologies (primary, caching, search)
        4. Infrastructure (cloud, containers, orchestration)
        5. DevOps Tools (CI/CD, monitoring, logging)
        6. Security Tools (authentication, authorization, encryption)
        7. Communication (APIs, messaging, protocols)
        8. Development Tools (IDEs, testing, documentation)
        
        For each technology, provide:
        - Specific recommendation with version
        - Justification for selection
        - Alternative options
        - Learning curve assessment
        - Community support rating
        - Long-term viability
        
        Return as JSON:
        {{
            "frontend": {{
                "framework": {{"name": "React", "version": "18.x", "justification": "...", "alternatives": [...]}},
                "ui_library": {{...}},
                "state_management": {{...}},
                "build_tools": {{...}}
            }},
            "backend": {{
                "language": {{...}},
                "framework": {{...}},
                "runtime": {{...}}
            }},
            "database": {{
                "primary": {{...}},
                "caching": {{...}},
                "search": {{...}}
            }},
            "infrastructure": {{
                "cloud_provider": {{...}},
                "containers": {{...}},
                "orchestration": {{...}}
            }},
            "devops": {{
                "ci_cd": {{...}},
                "monitoring": {{...}},
                "logging": {{...}}
            }},
            "security": {{
                "authentication": {{...}},
                "authorization": {{...}},
                "encryption": {{...}}
            }},
            "communication": {{
                "api_style": {{...}},
                "messaging": {{...}},
                "protocols": {{...}}
            }},
            "development": {{
                "ide": {{...}},
                "testing": {{...}},
                "documentation": {{...}}
            }},
            "stack_summary": "...",
            "total_complexity": "medium",
            "estimated_learning_time": "3-6 months",
            "technology_risks": [...]
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_tech_stack(arch_task)
    
    async def _design_system_architecture(self, arch_task: 'ArchitectureTask', architecture_type: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Design detailed system architecture"""
        prompt = f"""
        Design a detailed system architecture for:
        
        Project: {arch_task.project_name}
        Architecture Type: {architecture_type.get('primary_architecture')}
        Technology Stack: {json.dumps(tech_stack, indent=2)}
        Requirements: {arch_task.requirements}
        Scale: {arch_task.scale.value}
        
        Design the system architecture with:
        1. System Components (services, modules, layers)
        2. Component Interactions (data flow, communication patterns)
        3. Data Architecture (data models, storage, flow)
        4. Integration Points (external systems, APIs, events)
        5. Scalability Design (horizontal/vertical scaling strategies)
        6. Performance Optimization (caching, load balancing, CDN)
        7. Fault Tolerance (redundancy, failover, circuit breakers)
        8. Monitoring and Observability (metrics, logs, traces)
        
        For each component, specify:
        - Purpose and responsibilities
        - Technology implementation
        - Scaling characteristics
        - Dependencies
        - Performance considerations
        - Security requirements
        
        Return as JSON:
        {{
            "system_overview": "...",
            "components": [
                {{
                    "name": "User Service",
                    "type": "microservice",
                    "purpose": "...",
                    "technology": "...",
                    "responsibilities": [...],
                    "dependencies": [...],
                    "scaling": "...",
                    "performance_requirements": {{...}},
                    "security_requirements": [...]
                }}
            ],
            "data_flow": [
                {{
                    "from": "Frontend",
                    "to": "API Gateway",
                    "protocol": "HTTPS",
                    "data_type": "JSON",
                    "frequency": "real-time"
                }}
            ],
            "integration_points": [...],
            "scalability_strategy": {{...}},
            "performance_strategy": {{...}},
            "fault_tolerance": {{...}},
            "monitoring_strategy": {{...}},
            "architecture_diagram": "..."
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_system_architecture(arch_task)
    
    async def _design_database_architecture(self, arch_task: 'ArchitectureTask', system_architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Design comprehensive database architecture"""
        prompt = f"""
        Design a comprehensive database architecture for:
        
        Project: {arch_task.project_name}
        System Components: {[comp['name'] for comp in system_architecture.get('components', [])]}
        Scale: {arch_task.scale.value}
        Performance Requirements: {arch_task.performance_requirements}
        
        Design database architecture including:
        1. Database Selection (primary, secondary, specialized)
        2. Data Models (entities, relationships, schemas)
        3. Data Partitioning (sharding, federation strategies)
        4. Caching Strategy (levels, technologies, patterns)
        5. Data Consistency (ACID, eventual consistency, patterns)
        6. Backup and Recovery (strategies, RPO/RTO targets)
        7. Data Security (encryption, access control, auditing)
        8. Performance Optimization (indexing, query optimization)
        
        Return as JSON:
        {{
            "database_overview": "...",
            "primary_database": {{
                "type": "PostgreSQL",
                "version": "15.x",
                "justification": "...",
                "configuration": {{...}}
            }},
            "secondary_databases": [...],
            "data_models": [
                {{
                    "entity": "User",
                    "attributes": [...],
                    "relationships": [...],
                    "indexes": [...],
                    "constraints": [...]
                }}
            ],
            "partitioning_strategy": {{...}},
            "caching_strategy": {{...}},
            "consistency_model": "...",
            "backup_strategy": {{...}},
            "security_measures": [...],
            "performance_optimizations": [...],
            "migration_strategy": "..."
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_database_architecture(arch_task)
    
    async def _generate_api_specifications(self, arch_task: 'ArchitectureTask', system_architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive API specifications"""
        prompt = f"""
        Generate comprehensive API specifications for:
        
        Project: {arch_task.project_name}
        System Components: {[comp['name'] for comp in system_architecture.get('components', [])]}
        Requirements: {arch_task.requirements}
        
        Design API specifications including:
        1. API Architecture (REST, GraphQL, gRPC, WebSocket)
        2. Endpoint Design (resources, operations, parameters)
        3. Data Schemas (request/response models, validation)
        4. Authentication & Authorization (methods, scopes, tokens)
        5. Error Handling (error codes, messages, recovery)
        6. Rate Limiting (strategies, quotas, throttling)
        7. Versioning Strategy (approach, backward compatibility)
        8. Documentation (OpenAPI, examples, SDKs)
        
        Return as JSON:
        {{
            "api_overview": "...",
            "api_style": "REST",
            "base_url": "https://api.example.com/v1",
            "authentication": {{
                "type": "JWT",
                "implementation": "...",
                "scopes": [...]
            }},
            "endpoints": [
                {{
                    "path": "/users",
                    "method": "GET",
                    "purpose": "List users",
                    "parameters": [...],
                    "request_schema": {{...}},
                    "response_schema": {{...}},
                    "error_responses": [...],
                    "rate_limit": "100/hour"
                }}
            ],
            "data_schemas": {{...}},
            "error_handling": {{...}},
            "rate_limiting": {{...}},
            "versioning": {{...}},
            "documentation_strategy": "...",
            "testing_strategy": "..."
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_api_specifications(arch_task)
    
    async def _plan_security_architecture(self, arch_task: 'ArchitectureTask', system_architecture: Dict[str, Any]) -> Dict[str, Any]:
        """Plan comprehensive security architecture"""
        prompt = f"""
        Plan a comprehensive security architecture for:
        
        Project: {arch_task.project_name}
        Security Requirements: {arch_task.security_requirements}
        System Components: {[comp['name'] for comp in system_architecture.get('components', [])]}
        Scale: {arch_task.scale.value}
        
        Design security architecture covering:
        1. Authentication Strategy (methods, providers, MFA)
        2. Authorization Model (RBAC, ABAC, permissions)
        3. Data Protection (encryption, tokenization, masking)
        4. Network Security (firewalls, VPNs, segmentation)
        5. Application Security (input validation, OWASP top 10)
        6. Infrastructure Security (container, cloud, secrets)
        7. Monitoring & Incident Response (SIEM, alerting, procedures)
        8. Compliance (regulations, standards, auditing)
        
        Return as JSON:
        {{
            "security_overview": "...",
            "authentication": {{
                "primary_method": "OAuth 2.0 + OIDC",
                "providers": [...],
                "mfa_strategy": "...",
                "session_management": "..."
            }},
            "authorization": {{
                "model": "RBAC",
                "roles": [...],
                "permissions": [...],
                "implementation": "..."
            }},
            "data_protection": {{
                "encryption_at_rest": "...",
                "encryption_in_transit": "...",
                "key_management": "...",
                "data_classification": [...]
            }},
            "network_security": {{...}},
            "application_security": {{...}},
            "infrastructure_security": {{...}},
            "monitoring": {{...}},
            "compliance": {{...}},
            "security_controls": [...],
            "threat_model": [...],
            "security_testing": "..."
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_security_architecture(arch_task)
    
    async def _design_deployment_architecture(self, arch_task: 'ArchitectureTask', system_architecture: Dict[str, Any], tech_stack: Dict[str, Any]) -> Dict[str, Any]:
        """Design deployment and infrastructure architecture"""
        prompt = f"""
        Design deployment architecture for:
        
        Project: {arch_task.project_name}
        Technology Stack: {json.dumps(tech_stack.get('infrastructure', {}), indent=2)}
        System Components: {[comp['name'] for comp in system_architecture.get('components', [])]}
        Scale: {arch_task.scale.value}
        
        Design deployment architecture including:
        1. Infrastructure Strategy (cloud, on-premise, hybrid)
        2. Containerization (Docker, orchestration, registries)
        3. Environment Strategy (dev, staging, production)
        4. CI/CD Pipeline (build, test, deploy, rollback)
        5. Scaling Strategy (auto-scaling, load balancing)
        6. Monitoring & Logging (infrastructure, application, business)
        7. Disaster Recovery (backup, failover, geographic distribution)
        8. Cost Optimization (resource sizing, reserved instances, spot instances)
        
        Return as JSON:
        {{
            "deployment_overview": "...",
            "infrastructure": {{
                "provider": "AWS",
                "regions": [...],
                "availability_zones": [...],
                "networking": {{...}}
            }},
            "containerization": {{
                "strategy": "Docker + Kubernetes",
                "orchestration": "...",
                "registry": "...",
                "configuration": {{...}}
            }},
            "environments": [
                {{
                    "name": "production",
                    "purpose": "...",
                    "resources": {{...}},
                    "configuration": {{...}}
                }}
            ],
            "ci_cd": {{
                "pipeline_stages": [...],
                "tools": {{...}},
                "deployment_strategy": "...",
                "rollback_strategy": "..."
            }},
            "scaling": {{...}},
            "monitoring": {{...}},
            "disaster_recovery": {{...}},
            "cost_optimization": {{...}},
            "infrastructure_as_code": "..."
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_deployment_architecture(arch_task)
    
    async def _create_implementation_roadmap(self, arch_task: 'ArchitectureTask', planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed implementation roadmap"""
        prompt = f"""
        Create a detailed implementation roadmap for:
        
        Project: {arch_task.project_name}
        Timeline: {arch_task.timeline}
        Architecture Type: {planning_result.get('architecture_type', {}).get('primary_architecture')}
        System Components: {len(planning_result.get('system_architecture', {}).get('components', []))} components
        Technology Stack Complexity: {planning_result.get('tech_stack', {}).get('total_complexity', 'medium')}
        
        Create implementation roadmap with:
        1. Phase Breakdown (logical implementation phases)
        2. Task Dependencies (critical path, parallel tasks)
        3. Resource Requirements (team size, skills, tools)
        4. Risk Assessment (technical, timeline, resource risks)
        5. Milestone Definition (deliverables, success criteria)
        6. Testing Strategy (unit, integration, system, acceptance)
        7. Deployment Strategy (rollout plan, go-live criteria)
        8. Success Metrics (KPIs, performance targets, business metrics)
        
        Return as JSON:
        {{
            "roadmap_overview": "...",
            "total_duration": "6-9 months",
            "phases": [
                {{
                    "phase": "Foundation",
                    "duration": "4-6 weeks",
                    "objectives": [...],
                    "deliverables": [...],
                    "tasks": [
                        {{
                            "task": "Setup development environment",
                            "duration": "1 week",
                            "dependencies": [],
                            "resources": [...],
                            "risks": [...]
                        }}
                    ],
                    "success_criteria": [...],
                    "risks": [...]
                }}
            ],
            "critical_path": [...],
            "resource_requirements": {{...}},
            "risk_mitigation": {{...}},
            "testing_strategy": {{...}},
            "deployment_strategy": {{...}},
            "success_metrics": {{...}},
            "contingency_plans": [...]
        }}
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return self._get_default_implementation_roadmap(arch_task)
    
    async def _assess_architecture_quality(self, planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall architecture quality and provide recommendations"""
        # Quality assessment criteria
        criteria = {
            'scalability': 0.0,
            'maintainability': 0.0,
            'security': 0.0,
            'performance': 0.0,
            'reliability': 0.0,
            'testability': 0.0,
            'deployability': 0.0,
            'cost_effectiveness': 0.0
        }
        
        # Simple scoring based on architecture completeness
        if planning_result.get('system_architecture'):
            criteria['scalability'] += 0.3
            criteria['maintainability'] += 0.3
        
        if planning_result.get('security_architecture'):
            criteria['security'] += 0.4
        
        if planning_result.get('deployment_architecture'):
            criteria['deployability'] += 0.4
            criteria['reliability'] += 0.3
        
        if planning_result.get('tech_stack'):
            criteria['performance'] += 0.3
            criteria['cost_effectiveness'] += 0.3
        
        if planning_result.get('api_specifications'):
            criteria['testability'] += 0.3
        
        # Calculate overall score
        overall_score = sum(criteria.values()) / len(criteria)
        
        return {
            'criteria_scores': criteria,
            'overall_score': min(overall_score, 1.0),
            'quality_rating': self._get_quality_rating(overall_score),
            'recommendations': self._get_quality_recommendations(criteria),
            'strengths': self._identify_strengths(criteria),
            'areas_for_improvement': self._identify_improvements(criteria)
        }
    
    async def _generate_architecture_documentation(self, planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive architecture documentation"""
        documentation = {
            'architecture_overview': self._generate_overview_doc(planning_result),
            'system_design': self._generate_system_design_doc(planning_result),
            'api_documentation': self._generate_api_doc(planning_result),
            'deployment_guide': self._generate_deployment_doc(planning_result),
            'security_guide': self._generate_security_doc(planning_result),
            'implementation_guide': self._generate_implementation_doc(planning_result),
            'diagrams': self._generate_architecture_diagrams(planning_result)
        }
        
        return documentation
    
    async def _store_architecture_artifacts(self, session_id: str, planning_result: Dict[str, Any], documentation: Dict[str, Any]):
        """Store architecture artifacts in memory service"""
        # Store main planning result
        await self.memory_service.store_knowledge(
            'architecture_plan',
            planning_result,
            {
                'session_id': session_id,
                'project_name': planning_result.get('project_name'),
                'architecture_type': planning_result.get('architecture_type', {}).get('primary_architecture'),
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Store documentation
        await self.memory_service.store_knowledge(
            'architecture_documentation',
            documentation,
            {
                'session_id': session_id,
                'document_type': 'comprehensive_architecture_docs',
                'timestamp': datetime.utcnow().isoformat()
            }
        )
    
    # Helper methods for default responses
    def _get_default_requirements_analysis(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'functional_requirements': [],
            'non_functional_requirements': [],
            'technical_constraints': [],
            'business_constraints': [],
            'quality_attributes': [],
            'integration_requirements': [],
            'analysis_summary': 'Basic requirements analysis completed',
            'key_challenges': ['Requirements need further clarification'],
            'architecture_drivers': ['Scalability', 'Maintainability']
        }
    
    def _get_default_architecture_type(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        if arch_task.scale == ProjectScale.SMALL:
            return {
                'primary_architecture': 'monolithic',
                'justification': 'Monolithic architecture suitable for small scale projects',
                'implementation_complexity': 'low'
            }
        else:
            return {
                'primary_architecture': 'microservices',
                'justification': 'Microservices architecture suitable for larger scale projects',
                'implementation_complexity': 'high'
            }
    
    def _get_default_tech_stack(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'frontend': {
                'framework': {'name': 'React', 'version': '18.x'},
                'ui_library': {'name': 'Material-UI', 'version': '5.x'}
            },
            'backend': {
                'language': {'name': 'Python', 'version': '3.11+'},
                'framework': {'name': 'FastAPI', 'version': '0.100+'}
            },
            'database': {
                'primary': {'name': 'PostgreSQL', 'version': '15.x'}
            },
            'stack_summary': 'Modern full-stack web application',
            'total_complexity': 'medium'
        }
    
    def _get_default_system_architecture(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'system_overview': 'Basic system architecture',
            'components': [
                {
                    'name': 'Frontend',
                    'type': 'web_application',
                    'purpose': 'User interface',
                    'technology': 'React'
                },
                {
                    'name': 'Backend API',
                    'type': 'rest_api',
                    'purpose': 'Business logic and data access',
                    'technology': 'FastAPI'
                },
                {
                    'name': 'Database',
                    'type': 'relational_database',
                    'purpose': 'Data storage',
                    'technology': 'PostgreSQL'
                }
            ],
            'data_flow': [],
            'integration_points': []
        }
    
    def _get_default_database_architecture(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'database_overview': 'Basic database architecture',
            'primary_database': {
                'type': 'PostgreSQL',
                'version': '15.x',
                'justification': 'Reliable relational database'
            },
            'data_models': [],
            'caching_strategy': {'type': 'Redis', 'purpose': 'Session and query caching'}
        }
    
    def _get_default_api_specifications(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'api_overview': 'RESTful API design',
            'api_style': 'REST',
            'authentication': {'type': 'JWT'},
            'endpoints': [],
            'versioning': {'strategy': 'URL versioning'}
        }
    
    def _get_default_security_architecture(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'security_overview': 'Basic security architecture',
            'authentication': {'primary_method': 'JWT'},
            'authorization': {'model': 'RBAC'},
            'data_protection': {'encryption_at_rest': 'AES-256'},
            'security_controls': ['Input validation', 'HTTPS enforcement']
        }
    
    def _get_default_deployment_architecture(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'deployment_overview': 'Cloud-based deployment',
            'infrastructure': {'provider': 'AWS'},
            'containerization': {'strategy': 'Docker'},
            'environments': ['development', 'staging', 'production']
        }
    
    def _get_default_implementation_roadmap(self, arch_task: 'ArchitectureTask') -> Dict[str, Any]:
        return {
            'roadmap_overview': 'Phased implementation approach',
            'total_duration': '3-6 months',
            'phases': [
                {
                    'phase': 'Foundation',
                    'duration': '2-4 weeks',
                    'objectives': ['Setup development environment', 'Basic architecture implementation']
                },
                {
                    'phase': 'Core Development',
                    'duration': '6-12 weeks',
                    'objectives': ['Implement core features', 'API development']
                },
                {
                    'phase': 'Integration & Testing',
                    'duration': '2-4 weeks',
                    'objectives': ['System integration', 'Testing and validation']
                }
            ]
        }
    
    def _get_quality_rating(self, score: float) -> str:
        if score >= 0.8:
            return 'Excellent'
        elif score >= 0.6:
            return 'Good'
        elif score >= 0.4:
            return 'Fair'
        else:
            return 'Needs Improvement'
    
    def _get_quality_recommendations(self, criteria: Dict[str, float]) -> List[str]:
        recommendations = []
        
        for criterion, score in criteria.items():
            if score < 0.5:
                recommendations.append(f"Improve {criterion} aspects of the architecture")
        
        return recommendations
    
    def _identify_strengths(self, criteria: Dict[str, float]) -> List[str]:
        return [criterion for criterion, score in criteria.items() if score >= 0.7]
    
    def _identify_improvements(self, criteria: Dict[str, float]) -> List[str]:
        return [criterion for criterion, score in criteria.items() if score < 0.5]
    
    # Documentation generation methods
    def _generate_overview_doc(self, planning_result: Dict[str, Any]) -> str:
        return "# Architecture Overview\n\nComprehensive system architecture documentation."
    
    def _generate_system_design_doc(self, planning_result: Dict[str, Any]) -> str:
        return "# System Design\n\nDetailed system design documentation."
    
    def _generate_api_doc(self, planning_result: Dict[str, Any]) -> str:
        return "# API Documentation\n\nComprehensive API documentation."
    
    def _generate_deployment_doc(self, planning_result: Dict[str, Any]) -> str:
        return "# Deployment Guide\n\nStep-by-step deployment instructions."
    
    def _generate_security_doc(self, planning_result: Dict[str, Any]) -> str:
        return "# Security Guide\n\nSecurity implementation guidelines."
    
    def _generate_implementation_doc(self, planning_result: Dict[str, Any]) -> str:
        return "# Implementation Guide\n\nDetailed implementation instructions."
    
    def _generate_architecture_diagrams(self, planning_result: Dict[str, Any]) -> Dict[str, str]:
        return {
            'system_overview': 'System overview diagram (Mermaid/PlantUML)',
            'component_diagram': 'Component interaction diagram',
            'deployment_diagram': 'Deployment architecture diagram',
            'data_flow_diagram': 'Data flow diagram'
        }
    
    async def get_architecture_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent architecture planning history"""
        history = await self.memory_service.retrieve_knowledge(
            'architecture_plan',
            {},
            limit=limit
        )
        return history
    
    async def get_architecture_templates(self) -> Dict[str, Any]:
        """Get available architecture templates"""
        return self.architecture_templates
    
    async def _load_architecture_templates(self):
        """Load architecture templates"""
        self.architecture_templates = {
            'microservices': 'Microservices architecture template',
            'monolithic': 'Monolithic architecture template',
            'serverless': 'Serverless architecture template'
        }
    
    async def _load_tech_stack_knowledge(self):
        """Load technology stack knowledge"""
        self.tech_stack_knowledge = {
            'frontend': ['React', 'Vue', 'Angular', 'Svelte'],
            'backend': ['FastAPI', 'Django', 'Express', 'Spring Boot'],
            'database': ['PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch']
        }
    
    async def _load_design_patterns(self):
        """Load design patterns knowledge"""
        self.design_patterns = {
            'creational': ['Factory', 'Builder', 'Singleton'],
            'structural': ['Adapter', 'Decorator', 'Facade'],
            'behavioral': ['Observer', 'Strategy', 'Command']
        }
```

### Task 6.2: Architecture Data Models

**File**: `core/models.py` (Enhancement)

**Architecture Models**:
```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ArchitectureTask(BaseModel):
    project_name: str
    description: str
    requirements: List[str] = []
    constraints: List[str] = []
    scale: str = "medium"  # small, medium, large, enterprise
    target_users: int = 100
    performance_requirements: Dict[str, Any] = {}
    security_requirements: List[str] = []
    integration_requirements: List[str] = []
    technology_preferences: Dict[str, Any] = {}
    budget_constraints: Dict[str, Any] = {}
    timeline: Dict[str, Any] = {}
    existing_systems: List[str] = []
    research_context: Dict[str, Any] = {}

class SystemArchitecture(BaseModel):
    architecture_type: str
    components: List[Dict[str, Any]]
    data_flow: List[Dict[str, Any]]
    integration_points: List[Dict[str, Any]]
    scalability_strategy: Dict[str, Any]
    performance_strategy: Dict[str, Any]
    fault_tolerance: Dict[str, Any]
    monitoring_strategy: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class TechStackRecommendation(BaseModel):
    category: str  # frontend, backend, database, infrastructure
    technology: str
    version: str
    justification: str
    alternatives: List[str] = []
    learning_curve: str = "medium"  # low, medium, high
    community_support: int = 5  # 1-10 scale
    long_term_viability: str = "good"  # poor, fair, good, excellent

class ImplementationPhase(BaseModel):
    phase_name: str
    duration: str
    objectives: List[str]
    deliverables: List[str]
    tasks: List[Dict[str, Any]]
    dependencies: List[str] = []
    resources_required: Dict[str, Any] = {}
    risks: List[str] = []
    success_criteria: List[str] = []

class ArchitecturePlan(BaseModel):
    session_id: str
    project_name: str
    architecture_type: str
    system_architecture: SystemArchitecture
    tech_stack: Dict[str, TechStackRecommendation]
    database_design: Dict[str, Any]
    api_specifications: Dict[str, Any]
    security_architecture: Dict[str, Any]
    deployment_architecture: Dict[str, Any]
    implementation_roadmap: List[ImplementationPhase]
    quality_assessment: Dict[str, Any]
    documentation: Dict[str, Any]
    confidence_score: float
    created_at: datetime
    updated_at: datetime
```

### Task 6.3: Backend API Integration

**File**: `app/api/architecture.py`

**Architecture API Endpoints**:
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/api/architecture", tags=["architecture"])

@router.post("/plan")
async def create_architecture_plan(architecture_request: Dict[str, Any], background_tasks: BackgroundTasks):
    """Create comprehensive architecture plan"""
    try:
        architect_agent = await get_architect_agent()
        
        # Start architecture planning in background
        background_tasks.add_task(architect_agent.process_task, architecture_request)
        
        return {
            "status": "architecture_planning_initiated",
            "message": "Architecture planning started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}")
async def get_architecture_session(session_id: str):
    """Get architecture planning session results"""
    try:
        architect_agent = await get_architect_agent()
        session = architect_agent.active_planning_sessions.get(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_architecture_history(limit: int = 10):
    """Get architecture planning history"""
    try:
        architect_agent = await get_architect_agent()
        history = await architect_agent.get_architecture_history(limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_architecture_templates():
    """Get available architecture templates"""
    try:
        architect_agent = await get_architect_agent()
        templates = await architect_agent.get_architecture_templates()
        return templates
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-requirements")
async def analyze_requirements(requirements: Dict[str, Any]):
    """Analyze project requirements"""
    try:
        architect_agent = await get_architect_agent()
        
        # Create minimal task for requirements analysis
        task = {
            'type': 'requirements_analysis',
            **requirements
        }
        
        analysis = await architect_agent._analyze_requirements(
            architect_agent._parse_architecture_task(task)
        )
        
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommend-tech-stack")
async def recommend_tech_stack(project_info: Dict[str, Any]):
    """Get technology stack recommendations"""
    try:
        architect_agent = await get_architect_agent()
        
        # Create task for tech stack analysis
        task = {
            'type': 'tech_stack_analysis',
            **project_info
        }
        
        arch_task = architect_agent._parse_architecture_task(task)
        architecture_type = {'primary_architecture': project_info.get('architecture_type', 'microservices')}
        
        tech_stack = await architect_agent._analyze_technology_stack(arch_task, architecture_type)
        
        return tech_stack
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_architect_agent():
    """Get Architect Planner Agent instance"""
    # Implementation to get agent from registry
    pass
```

### Task 6.4: Frontend Architecture Interface

**File**: `frontend/components/architecture-planner.tsx`

**Architecture Planner Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from './ui/accordion';

interface ArchitecturePlan {
  session_id: string;
  architecture_type: string;
  system_architecture: any;
  tech_stack: any;
  implementation_roadmap: any;
  confidence_score: number;
}

interface ArchitecturePlannerProps {
  onPlanComplete: (plan: ArchitecturePlan) => void;
}

export const ArchitecturePlanner: React.FC<ArchitecturePlannerProps> = ({ onPlanComplete }) => {
  const [projectName, setProjectName] = useState('');
  const [description, setDescription] = useState('');
  const [requirements, setRequirements] = useState<string[]>(['']);
  const [scale, setScale] = useState('medium');
  const [targetUsers, setTargetUsers] = useState(100);
  const [isPlanning, setIsPlanning] = useState(false);
  const [currentPlan, setCurrentPlan] = useState<ArchitecturePlan | null>(null);
  const [planningHistory, setPlanningHistory] = useState<any[]>([]);
  const [templates, setTemplates] = useState<any>({});
  
  useEffect(() => {
    fetchPlanningHistory();
    fetchTemplates();
  }, []);
  
  const fetchPlanningHistory = async () => {
    try {
      const response = await fetch('/api/architecture/history');
      const history = await response.json();
      setPlanningHistory(history);
    } catch (error) {
      console.error('Failed to fetch planning history:', error);
    }
  };
  
  const fetchTemplates = async () => {
    try {
      const response = await fetch('/api/architecture/templates');
      const templates = await response.json();
      setTemplates(templates);
    } catch (error) {
      console.error('Failed to fetch templates:', error);
    }
  };
  
  const addRequirement = () => {
    setRequirements([...requirements, '']);
  };
  
  const updateRequirement = (index: number, value: string) => {
    const updated = [...requirements];
    updated[index] = value;
    setRequirements(updated);
  };
  
  const removeRequirement = (index: number) => {
    setRequirements(requirements.filter((_, i) => i !== index));
  };
  
  const startPlanning = async () => {
    if (!projectName.trim() || !description.trim()) return;
    
    setIsPlanning(true);
    setCurrentPlan(null);
    
    try {
      const planningRequest = {
        type: 'architecture_planning',
        project_name: projectName,
        description: description,
        requirements: requirements.filter(req => req.trim()),
        scale: scale,
        target_users: targetUsers,
        performance_requirements: {},
        security_requirements: [],
        integration_requirements: [],
        technology_preferences: {},
        timeline: {}
      };
      
      const response = await fetch('/api/architecture/plan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(planningRequest)
      });
      
      if (response.ok) {
        // Poll for results
        pollForResults();
      }
    } catch (error) {
      console.error('Failed to start planning:', error);
      setIsPlanning(false);
    }
  };
  
  const pollForResults = async () => {
    // Implementation would poll the backend for planning completion
    // For now, simulate with timeout
    setTimeout(async () => {
      try {
        // Simulate getting results
        const mockPlan: ArchitecturePlan = {
          session_id: 'arch_session_123',
          architecture_type: 'microservices',
          system_architecture: {
            components: [
              { name: 'API Gateway', type: 'service' },
              { name: 'User Service', type: 'microservice' },
              { name: 'Database', type: 'storage' }
            ]
          },
          tech_stack: {
            frontend: { framework: { name: 'React', version: '18.x' } },
            backend: { framework: { name: 'FastAPI', version: '0.100+' } }
          },
          implementation_roadmap: {
            phases: [
              { phase: 'Foundation', duration: '4-6 weeks' },
              { phase: 'Core Development', duration: '8-12 weeks' }
            ]
          },
          confidence_score: 0.85
        };
        
        setCurrentPlan(mockPlan);
        setIsPlanning(false);
        onPlanComplete(mockPlan);
        fetchPlanningHistory();
      } catch (error) {
        console.error('Failed to get planning results:', error);
        setIsPlanning(false);
      }
    }, 3000);
  };
  
  const renderSystemArchitecture = (architecture: any) => {
    if (!architecture?.components) return null;
    
    return (
      <div className="space-y-4">
        <h4 className="font-semibold">System Components</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {architecture.components.map((component: any, index: number) => (
            <Card key={index} className="p-4">
              <div className="font-medium">{component.name}</div>
              <Badge variant="secondary">{component.type}</Badge>
              {component.purpose && (
                <p className="text-sm text-gray-600 mt-2">{component.purpose}</p>
              )}
            </Card>
          ))}
        </div>
      </div>
    );
  };
  
  const renderTechStack = (techStack: any) => {
    if (!techStack) return null;
    
    return (
      <div className="space-y-4">
        {Object.entries(techStack).map(([category, tech]: [string, any]) => (
          <div key={category} className="border rounded-lg p-4">
            <h4 className="font-semibold capitalize mb-2">{category}</h4>
            <div className="space-y-2">
              {Object.entries(tech).map(([key, value]: [string, any]) => (
                <div key={key} className="flex justify-between items-center">
                  <span className="text-sm font-medium">{key.replace('_', ' ')}</span>
                  <Badge variant="outline">
                    {value.name} {value.version}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  };
  
  const renderImplementationRoadmap = (roadmap: any) => {
    if (!roadmap?.phases) return null;
    
    return (
      <div className="space-y-4">
        <h4 className="font-semibold">Implementation Phases</h4>
        <Accordion type="single" collapsible>
          {roadmap.phases.map((phase: any, index: number) => (
            <AccordionItem key={index} value={`phase-${index}`}>
              <AccordionTrigger>
                <div className="flex justify-between items-center w-full">
                  <span>{phase.phase}</span>
                  <Badge variant="secondary">{phase.duration}</Badge>
                </div>
              </AccordionTrigger>
              <AccordionContent>
                {phase.objectives && (
                  <div className="space-y-2">
                    <h5 className="font-medium">Objectives:</h5>
                    <ul className="list-disc list-inside text-sm">
                      {phase.objectives.map((objective: string, idx: number) => (
                        <li key={idx}>{objective}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    );
  };
  
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Architecture Planner</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Project Name</label>
                <Input
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder="Enter project name"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Project Scale</label>
                <Select value={scale} onValueChange={setScale}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="small">Small (1-3 developers)</SelectItem>
                    <SelectItem value="medium">Medium (4-10 developers)</SelectItem>
                    <SelectItem value="large">Large (10+ developers)</SelectItem>
                    <SelectItem value="enterprise">Enterprise</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Project Description</label>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe your project requirements and goals"
                rows={3}
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Requirements</label>
              <div className="space-y-2">
                {requirements.map((req, index) => (
                  <div key={index} className="flex gap-2">
                    <Input
                      value={req}
                      onChange={(e) => updateRequirement(index, e.target.value)}
                      placeholder="Enter requirement"
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => removeRequirement(index)}
                      disabled={requirements.length === 1}
                    >
                      Remove
                    </Button>
                  </div>
                ))}
                <Button variant="outline" onClick={addRequirement}>
                  Add Requirement
                </Button>
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">Target Users</label>
              <Input
                type="number"
                value={targetUsers}
                onChange={(e) => setTargetUsers(parseInt(e.target.value) || 100)}
                placeholder="Expected number of users"
              />
            </div>
            
            <Button
              onClick={startPlanning}
              disabled={isPlanning || !projectName.trim() || !description.trim()}
              className="w-full"
            >
              {isPlanning ? 'Planning Architecture...' : 'Start Architecture Planning'}
            </Button>
          </div>
        </CardContent>
      </Card>
      
      {currentPlan && (
        <Card>
          <CardHeader>
            <CardTitle className="flex justify-between items-center">
              Architecture Plan Results
              <Badge variant="secondary">
                Confidence: {Math.round(currentPlan.confidence_score * 100)}%
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="overview">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="architecture">Architecture</TabsTrigger>
                <TabsTrigger value="tech-stack">Tech Stack</TabsTrigger>
                <TabsTrigger value="roadmap">Roadmap</TabsTrigger>
              </TabsList>
              
              <TabsContent value="overview" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <Card className="p-4">
                    <div className="text-sm text-gray-600">Architecture Type</div>
                    <div className="font-semibold capitalize">
                      {currentPlan.architecture_type}
                    </div>
                  </Card>
                  <Card className="p-4">
                    <div className="text-sm text-gray-600">Components</div>
                    <div className="font-semibold">
                      {currentPlan.system_architecture?.components?.length || 0}
                    </div>
                  </Card>
                  <Card className="p-4">
                    <div className="text-sm text-gray-600">Confidence Score</div>
                    <div className="font-semibold">
                      {Math.round(currentPlan.confidence_score * 100)}%
                    </div>
                  </Card>
                </div>
              </TabsContent>
              
              <TabsContent value="architecture">
                {renderSystemArchitecture(currentPlan.system_architecture)}
              </TabsContent>
              
              <TabsContent value="tech-stack">
                {renderTechStack(currentPlan.tech_stack)}
              </TabsContent>
              
              <TabsContent value="roadmap">
                {renderImplementationRoadmap(currentPlan.implementation_roadmap)}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
      
      {planningHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Planning History</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {planningHistory.slice(0, 5).map((plan, index) => (
                <div key={index} className="flex justify-between items-center p-3 border rounded">
                  <div>
                    <div className="font-medium">{plan.project_name}</div>
                    <div className="text-sm text-gray-600">{plan.architecture_type}</div>
                  </div>
                  <Badge variant="outline">
                    {new Date(plan.created_at).toLocaleDateString()}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
```

## Testing Strategy

### Unit Tests

**File**: `tests/test_architect_planner_agent.py`

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from core.agents.architect_planner_agent import ArchitectPlannerAgent
from core.models import ArchitectureTask

@pytest.fixture
def architect_agent():
    return ArchitectPlannerAgent()

@pytest.fixture
def mock_llm_service():
    service = Mock()
    service.generate_response = AsyncMock()
    return service

@pytest.fixture
def mock_memory_service():
    service = Mock()
    service.store_knowledge = AsyncMock()
    service.retrieve_knowledge = AsyncMock()
    return service

@pytest.mark.asyncio
async def test_architect_agent_initialization(architect_agent, mock_llm_service, mock_memory_service):
    await architect_agent.initialize(mock_llm_service, mock_memory_service)
    assert architect_agent.llm_service == mock_llm_service
    assert architect_agent.memory_service == mock_memory_service

@pytest.mark.asyncio
async def test_can_handle_architecture_tasks(architect_agent):
    architecture_task = {'type': 'architecture_planning'}
    system_design_task = {'type': 'system_design'}
    invalid_task = {'type': 'invalid_task'}
    
    assert await architect_agent.can_handle_task(architecture_task)
    assert await architect_agent.can_handle_task(system_design_task)
    assert not await architect_agent.can_handle_task(invalid_task)

@pytest.mark.asyncio
async def test_requirements_analysis(architect_agent, mock_llm_service, mock_memory_service):
    await architect_agent.initialize(mock_llm_service, mock_memory_service)
    
    # Mock LLM response
    mock_llm_service.generate_response.return_value = '''
    {
        "functional_requirements": [
            {"requirement": "User authentication", "priority": "High", "complexity": "Moderate"}
        ],
        "non_functional_requirements": [],
        "analysis_summary": "Basic requirements identified"
    }
    '''
    
    arch_task = ArchitectureTask(
        project_name="Test Project",
        description="Test description",
        requirements=["User authentication", "Data storage"]
    )
    
    result = await architect_agent._analyze_requirements(arch_task)
    
    assert 'functional_requirements' in result
    assert len(result['functional_requirements']) > 0
    mock_llm_service.generate_response.assert_called_once()

@pytest.mark.asyncio
async def test_architecture_type_selection(architect_agent, mock_llm_service, mock_memory_service):
    await architect_agent.initialize(mock_llm_service, mock_memory_service)
    
    # Mock LLM response
    mock_llm_service.generate_response.return_value = '''
    {
        "primary_architecture": "microservices",
        "justification": "Suitable for scalable applications",
        "implementation_complexity": "high"
    }
    '''
    
    arch_task = ArchitectureTask(
        project_name="Test Project",
        scale="large",
        target_users=10000
    )
    
    requirements_analysis = {'key_challenges': ['Scalability']}
    
    result = await architect_agent._select_architecture_type(arch_task, requirements_analysis)
    
    assert result['primary_architecture'] == 'microservices'
    mock_llm_service.generate_response.assert_called_once()

@pytest.mark.asyncio
async def test_technology_stack_analysis(architect_agent, mock_llm_service, mock_memory_service):
    await architect_agent.initialize(mock_llm_service, mock_memory_service)
    
    # Mock LLM response
    mock_llm_service.generate_response.return_value = '''
    {
        "frontend": {
            "framework": {"name": "React", "version": "18.x", "justification": "Popular and well-supported"}
        },
        "backend": {
            "framework": {"name": "FastAPI", "version": "0.100+", "justification": "High performance Python framework"}
        },
        "stack_summary": "Modern full-stack solution"
    }
    '''
    
    arch_task = ArchitectureTask(
        project_name="Test Project",
        technology_preferences={"backend_language": "Python"}
    )
    
    architecture_type = {'primary_architecture': 'microservices'}
    
    result = await architect_agent._analyze_technology_stack(arch_task, architecture_type)
    
    assert 'frontend' in result
    assert 'backend' in result
    mock_llm_service.generate_response.assert_called_once()

@pytest.mark.asyncio
async def test_process_architecture_task(architect_agent, mock_llm_service, mock_memory_service):
    await architect_agent.initialize(mock_llm_service, mock_memory_service)
    
    # Mock all LLM responses
    mock_responses = [
        '{"functional_requirements": [], "analysis_summary": "Complete"}',  # requirements analysis
        '{"primary_architecture": "microservices", "justification": "Scalable"}',  # architecture type
        '{"frontend": {"framework": {"name": "React"}}, "stack_summary": "Modern"}',  # tech stack
        '{"system_overview": "Complete", "components": []}',  # system architecture
        '{"database_overview": "PostgreSQL setup", "primary_database": {}}',  # database design
        '{"api_overview": "REST API", "endpoints": []}',  # API specs
        '{"security_overview": "JWT auth", "authentication": {}}',  # security
        '{"deployment_overview": "Cloud deployment", "infrastructure": {}}',  # deployment
        '{"roadmap_overview": "Phased approach", "phases": []}'  # roadmap
    ]
    
    mock_llm_service.generate_response.side_effect = mock_responses
    
    task = {
        'type': 'architecture_planning',
        'project_name': 'Test Project',
        'description': 'Test project description',
        'requirements': ['Feature 1', 'Feature 2'],
        'scale': 'medium'
    }
    
    result = await architect_agent.process_task(task)
    
    assert 'session_id' in result
    assert 'architecture_type' in result
    assert 'confidence_score' in result
    assert result['confidence_score'] >= 0.0
    
    # Verify memory storage was called
    mock_memory_service.store_knowledge.assert_called()
```

### Integration Tests

**File**: `tests/test_architecture_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_architecture_plan():
    architecture_request = {
        "type": "architecture_planning",
        "project_name": "Test Project",
        "description": "Test project for architecture planning",
        "requirements": ["User authentication", "Data storage"],
        "scale": "medium",
        "target_users": 1000
    }
    
    response = client.post("/api/architecture/plan", json=architecture_request)
    assert response.status_code == 200
    assert "architecture_planning_initiated" in response.json()["status"]

def test_get_architecture_templates():
    response = client.get("/api/architecture/templates")
    assert response.status_code == 200
    templates = response.json()
    assert isinstance(templates, dict)

def test_analyze_requirements():
    requirements = {
        "project_name": "Test Project",
        "description": "Test description",
        "requirements": ["Authentication", "Data storage"],
        "scale": "medium"
    }
    
    response = client.post("/api/architecture/analyze-requirements", json=requirements)
    assert response.status_code == 200
    analysis = response.json()
    assert "functional_requirements" in analysis or "analysis_summary" in analysis

def test_recommend_tech_stack():
    project_info = {
        "project_name": "Test Project",
        "scale": "medium",
        "architecture_type": "microservices",
        "technology_preferences": {"backend_language": "Python"}
    }
    
    response = client.post("/api/architecture/recommend-tech-stack", json=project_info)
    assert response.status_code == 200
    tech_stack = response.json()
    assert "frontend" in tech_stack or "backend" in tech_stack or "stack_summary" in tech_stack

def test_get_architecture_history():
    response = client.get("/api/architecture/history")
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)
```

## Validation Criteria

### Backend Validation
- [ ] Architect Planner Agent processes architecture planning tasks
- [ ] Requirements analysis generates structured output
- [ ] Architecture type selection provides justified recommendations
- [ ] Technology stack analysis covers all major categories
- [ ] System architecture design includes components and interactions
- [ ] Database architecture includes schema and optimization strategies
- [ ] API specifications follow REST/GraphQL best practices
- [ ] Security architecture addresses authentication, authorization, and data protection
- [ ] Deployment architecture includes infrastructure and CI/CD
- [ ] Implementation roadmap provides phased approach with timelines
- [ ] Quality assessment evaluates architecture across multiple criteria
- [ ] Architecture artifacts are stored in memory service
- [ ] All API endpoints return appropriate responses
- [ ] Error handling works correctly

### Frontend Validation
- [ ] Architecture planner interface allows project configuration
- [ ] Requirements can be added, edited, and removed dynamically
- [ ] Planning process shows progress and status
- [ ] Architecture results display in organized tabs
- [ ] System architecture shows components and relationships
- [ ] Technology stack recommendations are clearly presented
- [ ] Implementation roadmap shows phases and timelines
- [ ] Planning history is accessible and searchable
- [ ] Confidence scores are displayed appropriately
- [ ] Interface is responsive and user-friendly

### Integration Validation
- [ ] Frontend successfully communicates with architecture API
- [ ] Planning requests are processed asynchronously
- [ ] Results are retrieved and displayed correctly
- [ ] Error states are handled gracefully
- [ ] Planning history is synchronized between frontend and backend
- [ ] Architecture templates are loaded and accessible
- [ ] Real-time updates work during planning process

## Human Testing Scenarios

### Scenario 1: Small Project Architecture Planning
1. Create architecture plan for small e-commerce website
2. Specify requirements: user registration, product catalog, shopping cart
3. Set scale to "small" with 100 target users
4. Verify monolithic architecture is recommended
5. Check technology stack includes appropriate choices for small scale
6. Review implementation roadmap for realistic timelines

### Scenario 2: Enterprise System Architecture
1. Plan architecture for large enterprise CRM system
2. Include complex requirements: multi-tenancy, real-time analytics, integrations
3. Set scale to "enterprise" with 50,000+ users
4. Verify microservices architecture is recommended
5. Check security architecture includes enterprise-grade measures
6. Review deployment architecture for high availability

### Scenario 3: API-First Architecture
1. Design architecture for API platform
2. Focus on API design, developer experience, and scalability
3. Verify API specifications are comprehensive
4. Check technology stack emphasizes API performance
5. Review documentation generation capabilities

### Scenario 4: Real-time Application Architecture
1. Plan architecture for real-time collaboration tool
2. Include requirements for WebSocket connections, real-time sync
3. Verify event-driven architecture components
4. Check technology stack includes real-time capabilities
5. Review scalability strategy for concurrent users

## Next Steps

After completing this phase:
1. **File**: `07-frontend-developer-agent-enhancement.md` - Enhance Frontend Developer Agent
2. Focus on advanced frontend development capabilities
3. Integration with design systems and component libraries
4. Automated testing and deployment for frontend applications

## Dependencies

- Completion of `05-research-agent-enhancement.md`
- Enhanced agent framework from `03-agent-framework-enhancement.md`
- Core services from `02-core-services-enhancement.md`
- LLM and Memory services integration
- Frontend component library setup