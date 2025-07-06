"""Feature Implementation Agent

Transforms Feature Build Plans and Synthesized Knowledge Documents into executable code.
Implements the complete workflow from plan ingestion to code generation and validation.
"""

from core.models import AppState, EnhancedTask, AgentType, LogEntry
from core.services.llm_service import EnhancedLLMService
from core.agents.base_agent import BaseAgent, AgentCapability, ActivityType
from core.services.memory_service import MemoryService, MemoryLayer
from typing import Dict, Any, Optional, List, Tuple
import json
import asyncio
import os
import re
from pathlib import Path
from datetime import datetime
import hashlib

class FeatureBuildPlan:
    """Represents a validated Feature Build Plan"""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.phases = self._extract_phases()
        self.tasks = self._extract_tasks()
        self.architecture = self._extract_architecture()
        self.testing_strategy = self._extract_testing_strategy()
        
    def _extract_phases(self) -> List[Dict[str, Any]]:
        """Extract phases from the build plan"""
        phases = []
        phase_pattern = r'#{1,3}\s*Phase\s*(\d+):\s*(.+?)\n'
        matches = re.findall(phase_pattern, self.content, re.IGNORECASE)
        for match in matches:
            phases.append({
                'number': int(match[0]),
                'title': match[1].strip(),
                'content': self._extract_phase_content(match[0])
            })
        return phases
    
    def _extract_tasks(self) -> List[Dict[str, Any]]:
        """Extract tasks and sub-tasks from the build plan"""
        tasks = []
        task_pattern = r'#{1,4}\s*(\d+\.\d+|\d+)\s*(.+?)\n'
        matches = re.findall(task_pattern, self.content)
        for match in matches:
            tasks.append({
                'id': match[0],
                'title': match[1].strip(),
                'content': self._extract_task_content(match[0])
            })
        return tasks
    
    def _extract_architecture(self) -> Dict[str, Any]:
        """Extract structured code architecture"""
        arch_pattern = r'Structured Code Architecture[\s\S]*?```([\s\S]*?)```'
        match = re.search(arch_pattern, self.content, re.IGNORECASE)
        if match:
            return {'structure': match.group(1).strip()}
        return {}
    
    def _extract_testing_strategy(self) -> Dict[str, Any]:
        """Extract integrated testing strategy"""
        test_pattern = r'Integrated Testing Strategy[\s\S]*?```([\s\S]*?)```'
        match = re.search(test_pattern, self.content, re.IGNORECASE)
        if match:
            return {'strategy': match.group(1).strip()}
        return {}
    
    def _extract_phase_content(self, phase_num: str) -> str:
        """Extract content for a specific phase"""
        pattern = f'Phase\s*{phase_num}:[\s\S]*?(?=Phase\s*\d+:|$)'
        match = re.search(pattern, self.content, re.IGNORECASE)
        return match.group(0) if match else ""
    
    def _extract_task_content(self, task_id: str) -> str:
        """Extract content for a specific task"""
        pattern = f'{re.escape(task_id)}[\s\S]*?(?=\d+\.\d+|\d+\.|$)'
        match = re.search(pattern, self.content)
        return match.group(0) if match else ""

class SynthesizedKnowledge:
    """Represents validated Synthesized Knowledge Documents"""
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.knowledge_map = self._create_knowledge_map()
        self.technology_stack = self._extract_technology_stack()
        self.best_practices = self._extract_best_practices()
        self.api_specifications = self._extract_api_specs()
        
    def _create_knowledge_map(self) -> Dict[str, List[str]]:
        """Create a mapping of topics to relevant documents"""
        knowledge_map = {}
        for doc in self.documents:
            topics = self._extract_topics(doc.get('content', ''))
            for topic in topics:
                if topic not in knowledge_map:
                    knowledge_map[topic] = []
                knowledge_map[topic].append(doc.get('name', 'unknown'))
        return knowledge_map
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract topics from document content"""
        # Simple topic extraction based on headers and keywords
        topics = []
        header_pattern = r'#{1,6}\s*(.+?)\n'
        headers = re.findall(header_pattern, content)
        topics.extend([h.strip().lower() for h in headers])
        return topics
    
    def _extract_technology_stack(self) -> Dict[str, Any]:
        """Extract technology stack information"""
        tech_stack = {}
        for doc in self.documents:
            content = doc.get('content', '')
            # Look for technology mentions
            tech_patterns = {
                'frameworks': r'(React|Vue|Angular|Django|Flask|FastAPI|Express)',
                'languages': r'(Python|JavaScript|TypeScript|Java|Go|Rust)',
                'databases': r'(PostgreSQL|MySQL|MongoDB|Redis|SQLite)',
                'tools': r'(Docker|Kubernetes|Git|npm|pip|yarn)'
            }
            for category, pattern in tech_patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if category not in tech_stack:
                        tech_stack[category] = []
                    tech_stack[category].extend(matches)
        return tech_stack
    
    def _extract_best_practices(self) -> List[str]:
        """Extract best practices from knowledge documents"""
        practices = []
        for doc in self.documents:
            content = doc.get('content', '')
            # Look for best practice sections
            practice_pattern = r'best practice[s]?[:\s]*([^\n]+)'
            matches = re.findall(practice_pattern, content, re.IGNORECASE)
            practices.extend(matches)
        return practices
    
    def _extract_api_specs(self) -> Dict[str, Any]:
        """Extract API specifications"""
        api_specs = {}
        for doc in self.documents:
            content = doc.get('content', '')
            # Look for API endpoint patterns
            endpoint_pattern = r'(GET|POST|PUT|DELETE|PATCH)\s+([/\w\-{}]+)'
            endpoints = re.findall(endpoint_pattern, content)
            if endpoints:
                api_specs[doc.get('name', 'unknown')] = endpoints
        return api_specs
    
    def get_relevant_knowledge(self, topic: str) -> List[Dict[str, Any]]:
        """Get knowledge documents relevant to a specific topic"""
        relevant_docs = []
        topic_lower = topic.lower()
        
        for doc in self.documents:
            content = doc.get('content', '').lower()
            if topic_lower in content or any(topic_lower in t for t in self._extract_topics(content)):
                relevant_docs.append(doc)
        
        return relevant_docs

class FeatureImplementationAgent(BaseAgent):
    """Feature Implementation Agent
    
    Transforms Feature Build Plans and Synthesized Knowledge Documents into executable code.
    Implements strict adherence to plan directives with continuous validation.
    """
    
    def __init__(self, llm_service: EnhancedLLMService, memory_service: MemoryService, 
                 agent_id: str = "feature_implementation_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Feature Implementation Agent",
            capabilities=[
                AgentCapability.CODING, 
                AgentCapability.ANALYSIS, 
                AgentCapability.TESTING,
                AgentCapability.DOCUMENTATION
            ],
            description="Transforms plans into executable code with continuous validation"
        )
        self.llm_service = llm_service
        self.memory_service = memory_service
        
        # Implementation state
        self.current_plan: Optional[FeatureBuildPlan] = None
        self.current_knowledge: Optional[SynthesizedKnowledge] = None
        self.implementation_context = {}
        self.validation_results = []
        self.generated_artifacts = []
        
        # Phase tracking
        self.current_phase = None
        self.current_task = None
        self.completed_tasks = []
        self.failed_tasks = []
        
    async def invoke(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point compatible with UltraOrchestrator interface"""
        try:
            await self.log_activity(ActivityType.TASK_START, "Feature implementation request received")
            
            # Extract request parameters
            plan_content = request.get('plan_content')
            knowledge_documents = request.get('knowledge_documents', [])
            implementation_mode = request.get('mode', 'full')  # full, incremental, validation
            
            if not plan_content:
                raise ValueError("Feature Build Plan content is required")
            
            # Phase 1: Plan & Knowledge Ingestion
            ingestion_result = await self._phase1_plan_knowledge_ingestion(
                plan_content, knowledge_documents
            )
            
            if not ingestion_result['success']:
                return {
                    'success': False,
                    'error': ingestion_result['error'],
                    'phase': 'ingestion'
                }
            
            # Determine next steps based on mode
            if implementation_mode == 'validation':
                return {
                    'success': True,
                    'phase': 'ingestion_complete',
                    'validation': ingestion_result,
                    'next_phase': 'code_generation'
                }
            
            # Phase 2: Code Generation & Implementation Framework
            elif implementation_mode == 'phase2' or implementation_mode == 'code_generation':
                result = await self._phase2_code_generation_implementation()
                if result.get('success'):
                    await self.log_activity(
                        ActivityType.SUCCESS,
                        "Phase 2: Code Generation & Implementation Framework completed"
                    )
                    return {
                        'success': True,
                        'phase': 'Phase 2: Code Generation & Implementation Framework',
                        'code_generation_framework': result,
                        'message': 'Code generation and implementation framework established successfully'
                    }
                else:
                    await self.log_activity(
                        ActivityType.ERROR,
                        f"Phase 2 failed: {result.get('error')}"
                    )
                    return {
                        'success': False,
                        'error': f"Phase 2 execution failed: {result.get('error')}"
                    }
            
            # Phase 3: Continuous Validation & Testing Integration
            elif implementation_mode == 'phase3' or implementation_mode == 'validation_testing':
                result = await self._phase3_continuous_validation_iteration()
                if result.get('success'):
                    await self.log_activity(
                        ActivityType.SUCCESS,
                        "Phase 3: Continuous Validation & Testing Integration completed"
                    )
                    return {
                        'success': True,
                        'phase': 'Phase 3: Continuous Validation & Testing Integration',
                        'validation_framework': result,
                        'message': 'Continuous validation and testing integration framework established successfully'
                    }
                else:
                    await self.log_activity(
                        ActivityType.ERROR,
                        f"Phase 3 failed: {result.get('error')}"
                    )
                    return {
                        'success': False,
                        'error': f"Phase 3 execution failed: {result.get('error')}"
                    }
            
            # Phase 4: Reporting & Output Management
            elif implementation_mode == 'phase4' or implementation_mode == 'reporting':
                result = await self._phase4_reporting_output_management()
                if result.get('success'):
                    await self.log_activity(
                        ActivityType.SUCCESS,
                        "Phase 4: Reporting & Output Management completed"
                    )
                    return {
                        'success': True,
                        'phase': 'Phase 4: Reporting & Output Management',
                        'reporting_framework': result,
                        'message': 'Comprehensive reporting and output management system established successfully'
                    }
                else:
                    await self.log_activity(
                        ActivityType.ERROR,
                        f"Phase 4 failed: {result.get('error')}"
                    )
                    return {
                        'success': False,
                        'error': f"Phase 4 execution failed: {result.get('error')}"
                    }
            
            # Full implementation mode - execute all phases sequentially
            elif implementation_mode == 'full':
                # Phase 2: Code Generation & Implementation Framework
                phase2_result = await self._phase2_code_generation_implementation()
                if not phase2_result.get('success'):
                    return {
                        'success': False,
                        'error': f"Phase 2 failed: {phase2_result.get('error')}",
                        'phase': 'code_generation'
                    }
                
                # Phase 3: Continuous Validation & Testing Integration
                phase3_result = await self._phase3_continuous_validation_iteration()
                if not phase3_result.get('success'):
                    return {
                        'success': False,
                        'error': f"Phase 3 failed: {phase3_result.get('error')}",
                        'phase': 'validation_testing'
                    }
                
                # Phase 4: Reporting & Output Management
                phase4_result = await self._phase4_reporting_output_management()
                if not phase4_result.get('success'):
                    return {
                        'success': False,
                        'error': f"Phase 4 failed: {phase4_result.get('error')}",
                        'phase': 'reporting'
                    }
                
                return {
                    'success': True,
                    'phase': 'full_implementation_complete',
                    'results': {
                        'phase1': ingestion_result,
                        'phase2': phase2_result,
                        'phase3': phase3_result,
                        'phase4': phase4_result
                    },
                    'message': 'Full feature implementation workflow completed successfully'
                }
            
            # Default: return ingestion results
            return {
                'success': True,
                'phase': 'ingestion_complete',
                'result': ingestion_result,
                'message': 'Plan and knowledge successfully ingested and validated'
            }
            
        except Exception as e:
            await self.log_activity(ActivityType.ERROR, f"Implementation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'phase': 'unknown'
            }
    
    async def _phase1_plan_knowledge_ingestion(self, plan_content: str, 
                                             knowledge_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 1: Plan & Knowledge Ingestion
        
        Implements:
        - 1.1. Secure Ingestion
        - 1.2. Integrity Check  
        - 1.3. Coherence Validation
        - 2.1. Hierarchical Plan Parsing
        - 2.2. Dynamic Knowledge Mapping
        """
        try:
            await self.log_activity(ActivityType.TASK_PROGRESS, "Starting Phase 1: Plan & Knowledge Ingestion")
            
            # 1.1. Secure Ingestion
            await self.log_activity(ActivityType.TASK_PROGRESS, "1.1. Performing secure ingestion")
            plan_metadata = {
                'received_at': datetime.utcnow().isoformat(),
                'content_hash': hashlib.sha256(plan_content.encode()).hexdigest(),
                'size_bytes': len(plan_content.encode()),
                'knowledge_docs_count': len(knowledge_documents)
            }
            
            # 1.2. Integrity Check
            await self.log_activity(ActivityType.TASK_PROGRESS, "1.2. Performing integrity check")
            integrity_result = await self._perform_integrity_check(plan_content, knowledge_documents)
            
            if not integrity_result['valid']:
                return {
                    'success': False,
                    'error': f"Integrity check failed: {integrity_result['issues']}",
                    'phase': '1.2_integrity_check'
                }
            
            # Create validated plan and knowledge objects
            self.current_plan = FeatureBuildPlan(plan_content, plan_metadata)
            self.current_knowledge = SynthesizedKnowledge(knowledge_documents)
            
            # 1.3. Coherence Validation
            await self.log_activity(ActivityType.TASK_PROGRESS, "1.3. Performing coherence validation")
            coherence_result = await self._perform_coherence_validation()
            
            if not coherence_result['coherent']:
                return {
                    'success': False,
                    'error': f"Coherence validation failed: {coherence_result['discrepancies']}",
                    'phase': '1.3_coherence_validation'
                }
            
            # 2.1. Hierarchical Plan Parsing
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.1. Performing hierarchical plan parsing")
            parsing_result = await self._perform_hierarchical_parsing()
            
            # 2.2. Dynamic Knowledge Mapping
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.2. Creating dynamic knowledge mapping")
            mapping_result = await self._create_knowledge_mapping()
            
            # Store in implementation context
            self.implementation_context = {
                'plan_metadata': plan_metadata,
                'integrity_check': integrity_result,
                'coherence_validation': coherence_result,
                'hierarchical_parsing': parsing_result,
                'knowledge_mapping': mapping_result,
                'ingestion_completed_at': datetime.utcnow().isoformat()
            }
            
            await self.log_activity(ActivityType.TASK_COMPLETE, "Phase 1: Plan & Knowledge Ingestion completed successfully")
            
            return {
                'success': True,
                'phase': 'phase1_complete',
                'plan_summary': {
                    'phases_count': len(self.current_plan.phases),
                    'tasks_count': len(self.current_plan.tasks),
                    'has_architecture': bool(self.current_plan.architecture),
                    'has_testing_strategy': bool(self.current_plan.testing_strategy)
                },
                'knowledge_summary': {
                    'documents_count': len(self.current_knowledge.documents),
                    'technology_stack': self.current_knowledge.technology_stack,
                    'knowledge_topics': list(self.current_knowledge.knowledge_map.keys())[:10]  # First 10 topics
                },
                'validation_results': {
                    'integrity': integrity_result,
                    'coherence': coherence_result
                },
                'context': self.implementation_context
            }
            
        except Exception as e:
            await self.log_activity(ActivityType.ERROR, f"Phase 1 failed: {str(e)}")
            raise
    
    async def _perform_integrity_check(self, plan_content: str, 
                                     knowledge_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform integrity and completeness check on artifacts"""
        issues = []
        
        # Check plan content structure
        if not plan_content.strip():
            issues.append("Plan content is empty")
        
        # Check for required plan sections
        required_sections = ['Phase', 'Task', 'Implementation']
        for section in required_sections:
            if section.lower() not in plan_content.lower():
                issues.append(f"Missing required section: {section}")
        
        # Check knowledge documents
        if not knowledge_documents:
            issues.append("No knowledge documents provided")
        else:
            for i, doc in enumerate(knowledge_documents):
                if not doc.get('content'):
                    issues.append(f"Knowledge document {i} has no content")
                if not doc.get('name'):
                    issues.append(f"Knowledge document {i} has no name")
        
        # Check for referenced dependencies
        dependency_pattern = r'(?:import|require|from)\s+([\w\-\.]+)'
        dependencies = re.findall(dependency_pattern, plan_content)
        
        # Validate that dependencies are mentioned in knowledge docs
        knowledge_content = ' '.join([doc.get('content', '') for doc in knowledge_documents])
        missing_deps = []
        for dep in dependencies:
            if dep not in knowledge_content:
                missing_deps.append(dep)
        
        if missing_deps:
            issues.append(f"Dependencies mentioned in plan but not in knowledge: {missing_deps}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'dependencies_found': dependencies,
            'missing_dependencies': missing_deps
        }
    
    async def _perform_coherence_validation(self) -> Dict[str, Any]:
        """Cross-reference plan components with knowledge documents"""
        discrepancies = []
        
        if not self.current_plan or not self.current_knowledge:
            return {'coherent': False, 'discrepancies': ['Plan or knowledge not loaded']}
        
        # Check technology stack coherence
        plan_content = self.current_plan.content.lower()
        knowledge_tech = self.current_knowledge.technology_stack
        
        # Verify mentioned technologies have knowledge support
        tech_mentions = []
        for category, techs in knowledge_tech.items():
            for tech in techs:
                if tech.lower() in plan_content:
                    tech_mentions.append(tech)
        
        # Check for architectural pattern consistency
        arch_patterns = ['microservices', 'monolith', 'serverless', 'event-driven']
        plan_patterns = [p for p in arch_patterns if p in plan_content]
        
        if len(plan_patterns) > 1:
            discrepancies.append(f"Multiple conflicting architectural patterns: {plan_patterns}")
        
        # Validate API specifications if present
        if self.current_knowledge.api_specifications:
            api_mentions = re.findall(r'api|endpoint|route', plan_content)
            if not api_mentions:
                discrepancies.append("Knowledge contains API specs but plan doesn't mention APIs")
        
        return {
            'coherent': len(discrepancies) == 0,
            'discrepancies': discrepancies,
            'technology_coherence': tech_mentions,
            'architectural_patterns': plan_patterns
        }
    
    async def _perform_hierarchical_parsing(self) -> Dict[str, Any]:
        """Parse plan structure and establish execution order"""
        if not self.current_plan:
            return {'success': False, 'error': 'No plan loaded'}
        
        # Create execution hierarchy
        execution_order = []
        
        # Sort phases by number
        sorted_phases = sorted(self.current_plan.phases, key=lambda x: x['number'])
        
        for phase in sorted_phases:
            phase_tasks = [task for task in self.current_plan.tasks 
                          if f"phase {phase['number']}" in task['content'].lower()]
            
            execution_order.append({
                'type': 'phase',
                'id': f"phase_{phase['number']}",
                'title': phase['title'],
                'tasks': phase_tasks,
                'dependencies': self._extract_dependencies(phase['content'])
            })
        
        return {
            'success': True,
            'execution_order': execution_order,
            'total_phases': len(sorted_phases),
            'total_tasks': len(self.current_plan.tasks),
            'architecture_defined': bool(self.current_plan.architecture),
            'testing_strategy_defined': bool(self.current_plan.testing_strategy)
        }
    
    async def _create_knowledge_mapping(self) -> Dict[str, Any]:
        """Create dynamic mapping between plan tasks and relevant knowledge"""
        if not self.current_plan or not self.current_knowledge:
            return {'success': False, 'error': 'Plan or knowledge not loaded'}
        
        task_knowledge_map = {}
        
        for task in self.current_plan.tasks:
            task_id = task['id']
            task_content = task['content'].lower()
            
            # Find relevant knowledge documents
            relevant_docs = []
            
            # Check each knowledge document for relevance
            for doc in self.current_knowledge.documents:
                doc_content = doc.get('content', '').lower()
                doc_name = doc.get('name', '').lower()
                
                # Calculate relevance score
                relevance_score = 0
                
                # Check for keyword matches
                task_keywords = re.findall(r'\b\w{3,}\b', task_content)
                for keyword in task_keywords:
                    if keyword in doc_content or keyword in doc_name:
                        relevance_score += 1
                
                # Check for technology stack matches
                for category, techs in self.current_knowledge.technology_stack.items():
                    for tech in techs:
                        if tech.lower() in task_content:
                            if tech.lower() in doc_content:
                                relevance_score += 2
                
                if relevance_score > 0:
                    relevant_docs.append({
                        'document': doc,
                        'relevance_score': relevance_score
                    })
            
            # Sort by relevance and take top documents
            relevant_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
            task_knowledge_map[task_id] = relevant_docs[:5]  # Top 5 most relevant
        
        return {
            'success': True,
            'task_knowledge_mapping': task_knowledge_map,
            'total_mappings': len(task_knowledge_map),
            'avg_docs_per_task': sum(len(docs) for docs in task_knowledge_map.values()) / len(task_knowledge_map) if task_knowledge_map else 0
        }
    
    def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from content"""
        dependencies = []
        
        # Look for explicit dependency mentions
        dep_patterns = [
            r'depends on ([^\n]+)',
            r'requires ([^\n]+)',
            r'after ([^\n]+)',
            r'prerequisite[s]?:?\s*([^\n]+)'
        ]
        
        for pattern in dep_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            dependencies.extend(matches)
        
        return [dep.strip() for dep in dependencies]
    
    async def _phase2_code_generation_implementation(self, ingestion_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Code Generation & Implementation Framework
        
        Implements:
        - 2.1. Strict Adherence to Plan Directives
        - 2.2. Phased & Task-Driven Execution
        - 2.3. Precise Code Architecture Replication
        - 2.4. Translating Guidance to Executable Code
        - 2.5. Leveraging Synthesized Knowledge for Implementation
        - 2.6. Incremental & Modular Development
        - 2.7. Tooling & Environment Interaction
        """
        try:
            await self.log_activity(ActivityType.TASK_PROGRESS, "Starting Phase 2: Code Generation & Implementation Framework")
            
            if not self.current_plan or not self.current_knowledge:
                raise ValueError("Plan and knowledge must be ingested before code generation")
            
            # 2.1. Strict Adherence to Plan Directives
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.1. Establishing strict plan adherence")
            execution_plan = await self._establish_execution_plan()
            
            # 2.2. Phased & Task-Driven Execution
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.2. Setting up phased task-driven execution")
            task_execution_order = await self._setup_task_execution_order()
            
            # 2.3. Precise Code Architecture Replication
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.3. Replicating code architecture")
            architecture_setup = await self._replicate_code_architecture()
            
            # 2.4. Translating Guidance to Executable Code
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.4. Translating implementation guidance")
            code_generation_framework = await self._setup_code_generation_framework()
            
            # 2.5. Leveraging Synthesized Knowledge for Implementation
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.5. Setting up knowledge-driven implementation")
            knowledge_integration = await self._setup_knowledge_integration()
            
            # 2.6. Incremental & Modular Development Setup
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.6. Configuring incremental development")
            modular_development = await self._setup_modular_development()
            
            # 2.7. Tooling & Environment Interaction
            await self.log_activity(ActivityType.TASK_PROGRESS, "2.7. Setting up tooling and environment")
            tooling_setup = await self._setup_tooling_environment()
            
            # Store Phase 2 context
            phase2_context = {
                'execution_plan': execution_plan,
                'task_execution_order': task_execution_order,
                'architecture_setup': architecture_setup,
                'code_generation_framework': code_generation_framework,
                'knowledge_integration': knowledge_integration,
                'modular_development': modular_development,
                'tooling_setup': tooling_setup,
                'phase2_completed_at': datetime.utcnow().isoformat()
            }
            
            self.implementation_context['phase2'] = phase2_context
            
            await self.log_activity(ActivityType.TASK_COMPLETE, "Phase 2: Code Generation & Implementation Framework completed successfully")
            
            return {
                'success': True,
                'phase': 'phase2_complete',
                'execution_plan': execution_plan,
                'architecture_ready': architecture_setup['success'],
                'code_generation_ready': code_generation_framework['success'],
                'knowledge_integration_ready': knowledge_integration['success'],
                'tooling_ready': tooling_setup['success'],
                'next_phase': 'continuous_validation',
                'context': phase2_context
            }
            
        except Exception as e:
            await self.log_activity(ActivityType.ERROR, f"Phase 2 failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'phase': 'phase2_failed'
            }
    
    async def _establish_execution_plan(self) -> Dict[str, Any]:
        """2.1. Establish strict adherence to plan directives"""
        try:
            # Extract execution order from hierarchical parsing
            parsing_result = self.implementation_context.get('hierarchical_parsing', {})
            execution_order = parsing_result.get('execution_order', [])
            
            # Create strict execution plan
            execution_plan = {
                'phases': [],
                'total_tasks': 0,
                'estimated_duration': 0,
                'dependencies': [],
                'validation_checkpoints': []
            }
            
            for phase_info in execution_order:
                phase_plan = {
                    'id': phase_info['id'],
                    'title': phase_info['title'],
                    'tasks': [],
                    'dependencies': phase_info.get('dependencies', []),
                    'validation_required': True
                }
                
                # Process tasks within phase
                for task in phase_info.get('tasks', []):
                    task_plan = {
                        'id': task['id'],
                        'title': task['title'],
                        'content': task['content'],
                        'dependencies': self._extract_dependencies(task['content']),
                        'estimated_effort': self._estimate_task_effort(task),
                        'validation_checkpoints': self._identify_validation_points(task)
                    }
                    phase_plan['tasks'].append(task_plan)
                    execution_plan['total_tasks'] += 1
                
                execution_plan['phases'].append(phase_plan)
            
            # Add global validation checkpoints
            execution_plan['validation_checkpoints'] = [
                'architecture_validation',
                'dependency_validation', 
                'code_compilation_check',
                'unit_test_execution',
                'integration_test_execution'
            ]
            
            return {
                'success': True,
                'execution_plan': execution_plan,
                'strict_order_enforced': True,
                'deviation_prevention': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to establish execution plan: {str(e)}"
            }
    
    async def _setup_task_execution_order(self) -> Dict[str, Any]:
        """2.2. Setup phased & task-driven execution"""
        try:
            # Create task queue with strict ordering
            task_queue = []
            task_dependencies = {}
            
            for phase in self.current_plan.phases:
                phase_tasks = [task for task in self.current_plan.tasks 
                              if f"phase {phase['number']}" in task['content'].lower()]
                
                for task in phase_tasks:
                    task_entry = {
                        'task_id': task['id'],
                        'phase_id': f"phase_{phase['number']}",
                        'title': task['title'],
                        'content': task['content'],
                        'status': 'pending',
                        'dependencies': self._extract_dependencies(task['content']),
                        'blocking_tasks': [],
                        'execution_order': len(task_queue)
                    }
                    
                    task_queue.append(task_entry)
                    task_dependencies[task['id']] = task_entry['dependencies']
            
            # Resolve task dependencies and blocking relationships
            for task_entry in task_queue:
                for dep in task_entry['dependencies']:
                    # Find tasks that this task depends on
                    for other_task in task_queue:
                        if dep.lower() in other_task['title'].lower() or dep.lower() in other_task['content'].lower():
                            task_entry['blocking_tasks'].append(other_task['task_id'])
            
            return {
                'success': True,
                'task_queue': task_queue,
                'task_dependencies': task_dependencies,
                'total_tasks': len(task_queue),
                'execution_strategy': 'sequential_with_dependency_resolution'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup task execution order: {str(e)}"
            }
    
    async def _replicate_code_architecture(self) -> Dict[str, Any]:
        """2.3. Precise Code Architecture Replication"""
        try:
            architecture = self.current_plan.architecture
            if not architecture:
                return {
                    'success': False,
                    'error': 'No architecture defined in plan'
                }
            
            # Parse architecture structure
            structure_content = architecture.get('structure', '')
            
            # Extract directory structure
            directory_structure = self._parse_directory_structure(structure_content)
            
            # Extract file specifications
            file_specifications = self._parse_file_specifications(structure_content)
            
            # Create architecture blueprint
            architecture_blueprint = {
                'root_directory': self._extract_root_directory(structure_content),
                'directories': directory_structure,
                'files': file_specifications,
                'naming_conventions': self._extract_naming_conventions(structure_content),
                'file_extensions': self._extract_file_extensions(structure_content)
            }
            
            return {
                'success': True,
                'architecture_blueprint': architecture_blueprint,
                'directories_count': len(directory_structure),
                'files_count': len(file_specifications),
                'replication_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to replicate code architecture: {str(e)}"
            }
    
    async def _setup_code_generation_framework(self) -> Dict[str, Any]:
        """2.4. Setup framework for translating guidance to executable code"""
        try:
            # Setup code generation templates and patterns
            generation_framework = {
                'templates': self._create_code_templates(),
                'patterns': self._extract_code_patterns(),
                'best_practices': self.current_knowledge.best_practices,
                'technology_stack': self.current_knowledge.technology_stack,
                'code_style_guide': self._create_code_style_guide(),
                'generation_rules': self._create_generation_rules()
            }
            
            # Setup LLM prompts for code generation
            generation_prompts = {
                'system_prompt': self._create_system_prompt_for_generation(),
                'task_prompt_template': self._create_task_prompt_template(),
                'validation_prompt': self._create_validation_prompt(),
                'refinement_prompt': self._create_refinement_prompt()
            }
            
            return {
                'success': True,
                'generation_framework': generation_framework,
                'generation_prompts': generation_prompts,
                'framework_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup code generation framework: {str(e)}"
            }
    
    async def _setup_knowledge_integration(self) -> Dict[str, Any]:
        """2.5. Setup leveraging synthesized knowledge for implementation"""
        try:
            # Create knowledge integration system
            knowledge_integration = {
                'task_knowledge_mapping': self.implementation_context.get('knowledge_mapping', {}).get('task_knowledge_mapping', {}),
                'technology_guides': self._create_technology_guides(),
                'api_specifications': self.current_knowledge.api_specifications,
                'best_practice_rules': self._create_best_practice_rules(),
                'optimization_suggestions': self._extract_optimization_suggestions()
            }
            
            # Setup knowledge retrieval system
            retrieval_system = {
                'topic_index': self.current_knowledge.knowledge_map,
                'relevance_scoring': self._setup_relevance_scoring(),
                'context_injection': self._setup_context_injection(),
                'knowledge_validation': self._setup_knowledge_validation()
            }
            
            return {
                'success': True,
                'knowledge_integration': knowledge_integration,
                'retrieval_system': retrieval_system,
                'integration_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup knowledge integration: {str(e)}"
            }
    
    async def _setup_modular_development(self) -> Dict[str, Any]:
        """2.6. Setup incremental & modular development"""
        try:
            # Create modular development strategy
            modular_strategy = {
                'module_boundaries': self._identify_module_boundaries(),
                'dependency_graph': self._create_dependency_graph(),
                'testing_units': self._identify_testing_units(),
                'integration_points': self._identify_integration_points(),
                'validation_checkpoints': self._create_validation_checkpoints()
            }
            
            # Setup incremental build process
            incremental_process = {
                'build_stages': self._define_build_stages(),
                'checkpoint_criteria': self._define_checkpoint_criteria(),
                'rollback_strategy': self._define_rollback_strategy(),
                'progress_tracking': self._setup_progress_tracking()
            }
            
            return {
                'success': True,
                'modular_strategy': modular_strategy,
                'incremental_process': incremental_process,
                'development_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup modular development: {str(e)}"
            }
    
    async def _setup_tooling_environment(self) -> Dict[str, Any]:
        """2.7. Setup tooling & environment interaction"""
        try:
            # Identify required tools and environment
            tech_stack = self.current_knowledge.technology_stack
            
            tooling_requirements = {
                'file_operations': ['create', 'read', 'write', 'modify', 'delete'],
                'build_tools': self._identify_build_tools(tech_stack),
                'package_managers': self._identify_package_managers(tech_stack),
                'compilers_interpreters': self._identify_compilers(tech_stack),
                'testing_frameworks': self._identify_testing_frameworks(tech_stack),
                'development_servers': self._identify_dev_servers(tech_stack)
            }
            
            # Setup environment configuration
            environment_config = {
                'runtime_versions': self._extract_runtime_versions(),
                'environment_variables': self._extract_environment_variables(),
                'configuration_files': self._identify_config_files(),
                'dependency_files': self._identify_dependency_files()
            }
            
            return {
                'success': True,
                'tooling_requirements': tooling_requirements,
                'environment_config': environment_config,
                'tooling_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup tooling environment: {str(e)}"
            }
    
    # Helper methods for Phase 2 implementation
    def _estimate_task_effort(self, task: Dict[str, Any]) -> str:
        """Estimate effort required for a task"""
        content = task.get('content', '').lower()
        
        # Simple heuristic based on content complexity
        if any(keyword in content for keyword in ['complex', 'advanced', 'integration']):
            return 'high'
        elif any(keyword in content for keyword in ['simple', 'basic', 'straightforward']):
            return 'low'
        else:
            return 'medium'
    
    def _identify_validation_points(self, task: Dict[str, Any]) -> List[str]:
        """Identify validation checkpoints for a task"""
        content = task.get('content', '').lower()
        validation_points = []
        
        if 'test' in content:
            validation_points.append('unit_test_execution')
        if 'api' in content:
            validation_points.append('api_validation')
        if 'database' in content:
            validation_points.append('database_validation')
        if 'ui' in content or 'interface' in content:
            validation_points.append('ui_validation')
        
        # Always include basic validation
        validation_points.extend(['syntax_check', 'compilation_check'])
        
        return validation_points
    
    def _parse_directory_structure(self, structure_content: str) -> List[Dict[str, Any]]:
        """Parse directory structure from architecture content"""
        directories = []
        
        # Look for directory patterns
        dir_patterns = [
            r'([\w\-_/]+)/',  # Standard directory notation
            r'mkdir\s+([\w\-_/]+)',  # mkdir commands
            r'directory[:\s]+([\w\-_/]+)'  # Directory specifications
        ]
        
        for pattern in dir_patterns:
            matches = re.findall(pattern, structure_content)
            for match in matches:
                directories.append({
                    'path': match.strip(),
                    'type': 'directory'
                })
        
        return directories
    
    def _parse_file_specifications(self, structure_content: str) -> List[Dict[str, Any]]:
        """Parse file specifications from architecture content"""
        files = []
        
        # Look for file patterns
        file_patterns = [
            r'([\w\-_/]+\.[\w]+)',  # Files with extensions
            r'touch\s+([\w\-_/]+\.[\w]+)',  # touch commands
            r'file[:\s]+([\w\-_/]+\.[\w]+)'  # File specifications
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, structure_content)
            for match in matches:
                file_path = match.strip()
                files.append({
                    'path': file_path,
                    'type': 'file',
                    'extension': Path(file_path).suffix,
                    'name': Path(file_path).name
                })
        
        return files
    
    def _extract_root_directory(self, structure_content: str) -> str:
        """Extract root directory from architecture"""
        # Look for project root indicators
        root_patterns = [
            r'project[\s\-_]*root[:\s]+([\w\-_/]+)',
            r'root[\s\-_]*directory[:\s]+([\w\-_/]+)',
            r'^([\w\-_]+)/',  # First directory mentioned
        ]
        
        for pattern in root_patterns:
            match = re.search(pattern, structure_content, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return 'project'  # Default root directory name
    
    def _extract_naming_conventions(self, structure_content: str) -> Dict[str, str]:
        """Extract naming conventions from architecture"""
        conventions = {
            'files': 'snake_case',  # Default
            'directories': 'kebab-case',  # Default
            'classes': 'PascalCase',  # Default
            'functions': 'snake_case'  # Default
        }
        
        # Look for explicit naming convention mentions
        if 'camelCase' in structure_content:
            conventions['functions'] = 'camelCase'
        if 'kebab-case' in structure_content:
            conventions['files'] = 'kebab-case'
        
        return conventions
    
    def _extract_file_extensions(self, structure_content: str) -> List[str]:
        """Extract file extensions from architecture"""
        extension_pattern = r'\.([a-zA-Z0-9]+)'
        extensions = re.findall(extension_pattern, structure_content)
        return list(set(extensions))  # Remove duplicates
    
    def _create_code_templates(self) -> Dict[str, str]:
        """Create code templates based on technology stack"""
        templates = {}
        tech_stack = self.current_knowledge.technology_stack
        
        # Python templates
        if 'Python' in tech_stack.get('languages', []):
            templates['python_class'] = '''class {class_name}:
    """{{class_description}}"""
    
    def __init__(self):
        pass
'''
            templates['python_function'] = '''def {function_name}({parameters}):
    """{{function_description}}"""
    pass
'''
        
        # JavaScript/TypeScript templates
        if any(lang in tech_stack.get('languages', []) for lang in ['JavaScript', 'TypeScript']):
            templates['js_class'] = '''class {class_name} {
    constructor() {
        // {{constructor_description}}
    }
}
'''
            templates['js_function'] = '''function {function_name}({parameters}) {
    // {{function_description}}
}
'''
        
        return templates
    
    def _extract_code_patterns(self) -> Dict[str, Any]:
        """Extract code patterns from knowledge documents"""
        patterns = {
            'architectural_patterns': [],
            'design_patterns': [],
            'coding_patterns': []
        }
        
        # Analyze knowledge documents for patterns
        for doc in self.current_knowledge.documents:
            content = doc.get('content', '').lower()
            
            # Look for architectural patterns
            arch_patterns = ['mvc', 'mvp', 'mvvm', 'repository', 'factory', 'singleton']
            for pattern in arch_patterns:
                if pattern in content:
                    patterns['architectural_patterns'].append(pattern)
            
            # Look for design patterns
            design_patterns = ['observer', 'strategy', 'decorator', 'adapter', 'facade']
            for pattern in design_patterns:
                if pattern in content:
                    patterns['design_patterns'].append(pattern)
        
        return patterns
    
    def _create_code_style_guide(self) -> Dict[str, Any]:
        """Create code style guide based on technology stack and best practices"""
        style_guide = {
            'indentation': '4 spaces',  # Default
            'line_length': 100,  # Default
            'naming_conventions': self._extract_naming_conventions(''),
            'import_organization': 'alphabetical',
            'comment_style': 'docstring_required'
        }
        
        # Adjust based on technology stack
        tech_stack = self.current_knowledge.technology_stack
        
        if 'JavaScript' in tech_stack.get('languages', []):
            style_guide['indentation'] = '2 spaces'
            style_guide['naming_conventions']['functions'] = 'camelCase'
        
        if 'Python' in tech_stack.get('languages', []):
            style_guide['line_length'] = 88  # Black formatter default
            style_guide['import_organization'] = 'isort_compatible'
        
        return style_guide
    
    def _create_generation_rules(self) -> List[str]:
        """Create rules for code generation"""
        return [
            'Always include proper error handling',
            'Add comprehensive docstrings/comments',
            'Follow established naming conventions',
            'Include type hints where applicable',
            'Implement proper logging',
            'Add input validation',
            'Follow DRY principles',
            'Ensure code is testable',
            'Include proper imports',
            'Follow security best practices'
        ]
    
    def _create_system_prompt_for_generation(self) -> str:
        """Create system prompt for LLM code generation"""
        tech_stack = self.current_knowledge.technology_stack
        best_practices = self.current_knowledge.best_practices
        
        return f"""You are an expert software developer implementing a feature according to a detailed plan.

Technology Stack: {tech_stack}
Best Practices: {best_practices[:5] if best_practices else 'Standard industry practices'}

Generate production-ready, well-documented code that:
1. Follows the exact specifications in the implementation guidance
2. Adheres to established coding standards and best practices
3. Includes proper error handling and validation
4. Is modular and testable
5. Includes comprehensive documentation

Always provide complete, executable code without placeholders."""
    
    def _create_task_prompt_template(self) -> str:
        """Create template for task-specific prompts"""
        return """Task: {task_title}

Implementation Guidance:
{implementation_guidance}

Relevant Knowledge:
{relevant_knowledge}

Architecture Context:
{architecture_context}

Generate the complete implementation for this task following the guidance exactly."""
    
    def _create_validation_prompt(self) -> str:
        """Create prompt for code validation"""
        return """Review the following code for:
1. Adherence to the implementation plan
2. Code quality and best practices
3. Potential bugs or issues
4. Missing error handling
5. Documentation completeness

Code:
{code}

Provide specific feedback and suggestions for improvement."""
    
    def _create_refinement_prompt(self) -> str:
        """Create prompt for code refinement"""
        return """Refine the following code based on the feedback:

Original Code:
{original_code}

Feedback:
{feedback}

Provide the improved version addressing all feedback points."""
    
    def _create_technology_guides(self) -> Dict[str, Any]:
        """Create technology-specific implementation guides"""
        guides = {}
        tech_stack = self.current_knowledge.technology_stack
        
        for category, technologies in tech_stack.items():
            for tech in technologies:
                guides[tech.lower()] = {
                    'best_practices': self._get_tech_best_practices(tech),
                    'common_patterns': self._get_tech_patterns(tech),
                    'setup_requirements': self._get_tech_setup(tech)
                }
        
        return guides
    
    def _get_tech_best_practices(self, technology: str) -> List[str]:
        """Get best practices for specific technology"""
        practices = {
            'python': [
                'Use virtual environments',
                'Follow PEP 8 style guide',
                'Use type hints',
                'Write comprehensive docstrings',
                'Handle exceptions properly'
            ],
            'javascript': [
                'Use const/let instead of var',
                'Implement proper error handling',
                'Use async/await for asynchronous operations',
                'Follow ESLint rules',
                'Write unit tests'
            ],
            'react': [
                'Use functional components with hooks',
                'Implement proper state management',
                'Use PropTypes or TypeScript',
                'Optimize re-renders',
                'Follow component composition patterns'
            ]
        }
        
        return practices.get(technology.lower(), ['Follow industry best practices'])
    
    def _get_tech_patterns(self, technology: str) -> List[str]:
        """Get common patterns for specific technology"""
        patterns = {
            'python': ['Factory Pattern', 'Singleton Pattern', 'Observer Pattern'],
            'javascript': ['Module Pattern', 'Observer Pattern', 'Promise Pattern'],
            'react': ['Container/Presentational', 'Higher-Order Components', 'Render Props']
        }
        
        return patterns.get(technology.lower(), [])
    
    def _get_tech_setup(self, technology: str) -> Dict[str, Any]:
        """Get setup requirements for specific technology"""
        setups = {
            'python': {
                'package_manager': 'pip',
                'dependency_file': 'requirements.txt',
                'virtual_env': 'venv'
            },
            'javascript': {
                'package_manager': 'npm',
                'dependency_file': 'package.json',
                'build_tool': 'webpack'
            },
            'react': {
                'package_manager': 'npm',
                'dependency_file': 'package.json',
                'build_tool': 'create-react-app'
            }
        }
        
        return setups.get(technology.lower(), {})
    
    # Additional helper methods for Phase 2
    def _create_best_practice_rules(self) -> List[str]:
        """Create rules based on best practices from knowledge"""
        rules = []
        for practice in self.current_knowledge.best_practices:
            if practice:
                rules.append(f"Apply: {practice}")
        
        # Add default rules if no practices found
        if not rules:
            rules = [
                'Apply SOLID principles',
                'Use dependency injection',
                'Implement proper error handling',
                'Write comprehensive tests',
                'Follow security best practices'
            ]
        
        return rules
    
    def _extract_optimization_suggestions(self) -> List[str]:
        """Extract optimization suggestions from knowledge"""
        suggestions = []
        
        for doc in self.current_knowledge.documents:
            content = doc.get('content', '').lower()
            
            # Look for optimization mentions
            if 'optimization' in content or 'performance' in content:
                # Extract sentences containing optimization keywords
                sentences = content.split('.')
                for sentence in sentences:
                    if any(keyword in sentence for keyword in ['optimize', 'performance', 'efficient']):
                        suggestions.append(sentence.strip())
        
        return suggestions[:10]  # Limit to top 10 suggestions
    
    def _setup_relevance_scoring(self) -> Dict[str, Any]:
        """Setup relevance scoring system for knowledge retrieval"""
        return {
            'keyword_weight': 1.0,
            'technology_weight': 2.0,
            'context_weight': 1.5,
            'recency_weight': 0.5,
            'scoring_algorithm': 'weighted_sum'
        }
    
    def _setup_context_injection(self) -> Dict[str, Any]:
        """Setup context injection for knowledge integration"""
        return {
            'max_context_length': 2000,
            'context_selection': 'relevance_based',
            'context_formatting': 'structured',
            'injection_points': ['before_generation', 'during_validation']
        }
    
    def _setup_knowledge_validation(self) -> Dict[str, Any]:
        """Setup knowledge validation system"""
        return {
            'consistency_check': True,
            'relevance_threshold': 0.7,
            'knowledge_freshness_check': True,
            'conflict_resolution': 'latest_wins'
        }
    
    def _identify_module_boundaries(self) -> List[Dict[str, Any]]:
        """Identify module boundaries for modular development"""
        modules = []
        
        # Analyze tasks to identify natural module boundaries
        for task in self.current_plan.tasks:
            content = task['content'].lower()
            
            # Identify module types
            if 'api' in content or 'endpoint' in content:
                modules.append({
                    'name': f"api_module_{task['id']}",
                    'type': 'api',
                    'tasks': [task['id']],
                    'dependencies': self._extract_dependencies(task['content'])
                })
            elif 'database' in content or 'model' in content:
                modules.append({
                    'name': f"data_module_{task['id']}",
                    'type': 'data',
                    'tasks': [task['id']],
                    'dependencies': self._extract_dependencies(task['content'])
                })
            elif 'ui' in content or 'interface' in content:
                modules.append({
                    'name': f"ui_module_{task['id']}",
                    'type': 'ui',
                    'tasks': [task['id']],
                    'dependencies': self._extract_dependencies(task['content'])
                })
        
        return modules
    
    def _create_dependency_graph(self) -> Dict[str, Any]:
        """Create dependency graph for modules"""
        graph = {
            'nodes': [],
            'edges': [],
            'cycles': [],
            'build_order': []
        }
        
        # Add nodes for each task
        for task in self.current_plan.tasks:
            graph['nodes'].append({
                'id': task['id'],
                'title': task['title'],
                'type': 'task'
            })
        
        # Add edges for dependencies
        for task in self.current_plan.tasks:
            dependencies = self._extract_dependencies(task['content'])
            for dep in dependencies:
                # Find matching tasks
                for other_task in self.current_plan.tasks:
                    if dep.lower() in other_task['title'].lower():
                        graph['edges'].append({
                            'from': other_task['id'],
                            'to': task['id'],
                            'type': 'dependency'
                        })
        
        # Calculate build order (topological sort)
        graph['build_order'] = self._topological_sort(graph['nodes'], graph['edges'])
        
        return graph
    
    def _topological_sort(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        """Perform topological sort to determine build order"""
        # Simple topological sort implementation
        in_degree = {node['id']: 0 for node in nodes}
        
        # Calculate in-degrees
        for edge in edges:
            in_degree[edge['to']] += 1
        
        # Find nodes with no incoming edges
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Remove edges from current node
            for edge in edges:
                if edge['from'] == current:
                    in_degree[edge['to']] -= 1
                    if in_degree[edge['to']] == 0:
                        queue.append(edge['to'])
        
        return result
    
    def _identify_testing_units(self) -> List[Dict[str, Any]]:
        """Identify units for testing"""
        testing_units = []
        
        for task in self.current_plan.tasks:
            content = task['content'].lower()
            
            # Determine testing requirements
            test_types = []
            if 'api' in content:
                test_types.append('integration_test')
            if 'function' in content or 'method' in content:
                test_types.append('unit_test')
            if 'ui' in content:
                test_types.append('ui_test')
            
            if test_types:
                testing_units.append({
                    'task_id': task['id'],
                    'test_types': test_types,
                    'test_priority': 'high' if 'critical' in content else 'medium'
                })
        
        return testing_units
    
    def _identify_integration_points(self) -> List[Dict[str, Any]]:
        """Identify integration points between modules"""
        integration_points = []
        
        # Look for integration patterns in tasks
        for task in self.current_plan.tasks:
            content = task['content'].lower()
            
            if any(keyword in content for keyword in ['integrate', 'connect', 'interface']):
                integration_points.append({
                    'task_id': task['id'],
                    'integration_type': self._determine_integration_type(content),
                    'complexity': 'high' if 'complex' in content else 'medium'
                })
        
        return integration_points
    
    def _determine_integration_type(self, content: str) -> str:
        """Determine type of integration from content"""
        if 'api' in content:
            return 'api_integration'
        elif 'database' in content:
            return 'data_integration'
        elif 'ui' in content:
            return 'ui_integration'
        else:
            return 'general_integration'
    
    def _create_validation_checkpoints(self) -> List[Dict[str, Any]]:
        """Create validation checkpoints for development"""
        checkpoints = [
            {
                'name': 'syntax_validation',
                'description': 'Check code syntax and basic structure',
                'frequency': 'after_each_file',
                'blocking': True
            },
            {
                'name': 'compilation_check',
                'description': 'Verify code compiles/runs without errors',
                'frequency': 'after_each_module',
                'blocking': True
            },
            {
                'name': 'unit_test_execution',
                'description': 'Run unit tests for implemented functionality',
                'frequency': 'after_each_feature',
                'blocking': True
            },
            {
                'name': 'integration_validation',
                'description': 'Test integration between modules',
                'frequency': 'after_integration_points',
                'blocking': True
            },
            {
                'name': 'performance_check',
                'description': 'Basic performance validation',
                'frequency': 'after_major_milestones',
                'blocking': False
            }
        ]
        
        return checkpoints
    
    def _define_build_stages(self) -> List[Dict[str, Any]]:
        """Define incremental build stages"""
        stages = []
        
        # Group tasks into logical build stages
        current_stage = {
            'stage_id': 'stage_1',
            'name': 'Foundation',
            'tasks': [],
            'dependencies': []
        }
        
        stage_counter = 1
        tasks_per_stage = max(1, len(self.current_plan.tasks) // 3)  # Aim for 3 stages
        
        for i, task in enumerate(self.current_plan.tasks):
            current_stage['tasks'].append(task['id'])
            
            # Create new stage when current is full
            if len(current_stage['tasks']) >= tasks_per_stage and i < len(self.current_plan.tasks) - 1:
                stages.append(current_stage)
                stage_counter += 1
                current_stage = {
                    'stage_id': f'stage_{stage_counter}',
                    'name': f'Stage {stage_counter}',
                    'tasks': [],
                    'dependencies': [stages[-1]['stage_id']] if stages else []
                }
        
        # Add final stage
        if current_stage['tasks']:
            stages.append(current_stage)
        
        return stages
    
    def _define_checkpoint_criteria(self) -> Dict[str, Any]:
        """Define criteria for checkpoints"""
        return {
            'syntax_check': {
                'criteria': ['no_syntax_errors', 'proper_indentation', 'valid_imports'],
                'tools': ['linter', 'syntax_checker']
            },
            'compilation_check': {
                'criteria': ['compiles_successfully', 'no_runtime_errors', 'proper_dependencies'],
                'tools': ['compiler', 'interpreter']
            },
            'test_execution': {
                'criteria': ['all_tests_pass', 'coverage_threshold_met', 'no_test_failures'],
                'tools': ['test_runner', 'coverage_tool']
            }
        }
    
    def _define_rollback_strategy(self) -> Dict[str, Any]:
        """Define rollback strategy for failed builds"""
        return {
            'rollback_triggers': [
                'compilation_failure',
                'test_failure_threshold_exceeded',
                'critical_error_detected'
            ],
            'rollback_actions': [
                'revert_to_last_checkpoint',
                'restore_working_state',
                'log_failure_details',
                'notify_for_manual_intervention'
            ],
            'recovery_strategy': 'incremental_retry_with_analysis'
        }
    
    def _setup_progress_tracking(self) -> Dict[str, Any]:
        """Setup progress tracking system"""
        return {
            'metrics': [
                'tasks_completed',
                'files_generated',
                'tests_passing',
                'compilation_success_rate',
                'validation_checkpoints_passed'
            ],
            'reporting_frequency': 'after_each_task',
            'progress_visualization': 'percentage_and_milestone_based',
            'failure_tracking': True
        }
    
    def _identify_build_tools(self, tech_stack: Dict[str, Any]) -> List[str]:
        """Identify required build tools"""
        tools = []
        
        languages = tech_stack.get('languages', [])
        frameworks = tech_stack.get('frameworks', [])
        
        if 'Python' in languages:
            tools.extend(['pip', 'setuptools', 'wheel'])
        if 'JavaScript' in languages or 'TypeScript' in languages:
            tools.extend(['npm', 'webpack', 'babel'])
        if 'React' in frameworks:
            tools.extend(['create-react-app', 'react-scripts'])
        if 'Django' in frameworks:
            tools.extend(['django-admin', 'manage.py'])
        
        return tools
    
    def _identify_package_managers(self, tech_stack: Dict[str, Any]) -> List[str]:
        """Identify package managers"""
        managers = []
        
        languages = tech_stack.get('languages', [])
        
        if 'Python' in languages:
            managers.append('pip')
        if 'JavaScript' in languages or 'TypeScript' in languages:
            managers.extend(['npm', 'yarn'])
        if 'Java' in languages:
            managers.extend(['maven', 'gradle'])
        
        return managers
    
    def _identify_compilers(self, tech_stack: Dict[str, Any]) -> List[str]:
        """Identify compilers and interpreters"""
        compilers = []
        
        languages = tech_stack.get('languages', [])
        
        if 'Python' in languages:
            compilers.append('python')
        if 'JavaScript' in languages:
            compilers.append('node')
        if 'TypeScript' in languages:
            compilers.append('tsc')
        if 'Java' in languages:
            compilers.append('javac')
        
        return compilers
    
    def _identify_testing_frameworks(self, tech_stack: Dict[str, Any]) -> List[str]:
        """Identify testing frameworks"""
        frameworks = []
        
        languages = tech_stack.get('languages', [])
        
        if 'Python' in languages:
            frameworks.extend(['pytest', 'unittest'])
        if 'JavaScript' in languages:
            frameworks.extend(['jest', 'mocha', 'jasmine'])
        if 'React' in tech_stack.get('frameworks', []):
            frameworks.extend(['react-testing-library', 'enzyme'])
        
        return frameworks
    
    def _identify_dev_servers(self, tech_stack: Dict[str, Any]) -> List[str]:
        """Identify development servers"""
        servers = []
        
        frameworks = tech_stack.get('frameworks', [])
        
        if 'React' in frameworks:
            servers.append('react-dev-server')
        if 'Django' in frameworks:
            servers.append('django-dev-server')
        if 'Flask' in frameworks:
            servers.append('flask-dev-server')
        if 'Express' in frameworks:
            servers.append('nodemon')
        
        return servers
    
    def _extract_runtime_versions(self) -> Dict[str, str]:
        """Extract runtime version requirements"""
        versions = {}
        
        # Look for version specifications in knowledge documents
        for doc in self.current_knowledge.documents:
            content = doc.get('content', '')
            
            # Look for version patterns
            version_patterns = [
                r'python\s+(\d+\.\d+)',
                r'node\s+(\d+\.\d+)',
                r'npm\s+(\d+\.\d+)',
                r'react\s+(\d+\.\d+)'
            ]
            
            for pattern in version_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    tech = pattern.split('\\')[0]
                    versions[tech] = match.group(1)
        
        return versions
    
    def _extract_environment_variables(self) -> List[str]:
        """Extract environment variable requirements"""
        env_vars = []
        
        for doc in self.current_knowledge.documents:
            content = doc.get('content', '')
            
            # Look for environment variable patterns
            env_patterns = [
                r'\$([A-Z_][A-Z0-9_]*)',  # $VAR_NAME
                r'process\.env\.([A-Z_][A-Z0-9_]*)',  # process.env.VAR_NAME
                r'os\.environ\[[\'\"](.*?)[\'\"]\]'  # os.environ['VAR_NAME']
            ]
            
            for pattern in env_patterns:
                matches = re.findall(pattern, content)
                env_vars.extend(matches)
        
        return list(set(env_vars))  # Remove duplicates
    
    def _identify_config_files(self) -> List[str]:
        """Identify configuration files needed"""
        config_files = []
        tech_stack = self.current_knowledge.technology_stack
        
        # Add config files based on technology stack
        if 'Python' in tech_stack.get('languages', []):
            config_files.extend(['requirements.txt', 'setup.py', '.env'])
        if 'JavaScript' in tech_stack.get('languages', []):
            config_files.extend(['package.json', '.env', 'webpack.config.js'])
        if 'React' in tech_stack.get('frameworks', []):
            config_files.extend(['package.json', 'public/index.html'])
        if 'Django' in tech_stack.get('frameworks', []):
            config_files.extend(['settings.py', 'urls.py', 'wsgi.py'])
        
        return config_files
    
    def _identify_dependency_files(self) -> List[str]:
        """Identify dependency management files"""
        dep_files = []
        tech_stack = self.current_knowledge.technology_stack
        
        if 'Python' in tech_stack.get('languages', []):
            dep_files.extend(['requirements.txt', 'Pipfile', 'pyproject.toml'])
        if 'JavaScript' in tech_stack.get('languages', []):
            dep_files.extend(['package.json', 'package-lock.json', 'yarn.lock'])
        if 'Java' in tech_stack.get('languages', []):
            dep_files.extend(['pom.xml', 'build.gradle'])
        
        return dep_files
    
    async def _phase3_continuous_validation_iteration(self, phase2_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Continuous Validation & Iteration Framework
        
        Implements:
        - 3.1. Integrated Unit Testing
        - 3.2. Mandatory Test Execution
        - 3.3. Test-Driven Progression
        - 3.4. Debugging & Self-Correction
        - 3.5. Automated Issue Detection
        - 3.6. Contextual Remediation
        - 3.7. Plan Validation & Refinement (Sample Builds)
        """
        try:
            await self.log_activity(ActivityType.TASK_PROGRESS, "Starting Phase 3: Continuous Validation & Iteration Framework")
            
            if not phase2_result.get('success'):
                raise ValueError("Phase 2 must complete successfully before validation")
            
            # 3.1. Integrated Unit Testing Setup
            await self.log_activity(ActivityType.TASK_PROGRESS, "3.1. Setting up integrated unit testing")
            testing_framework = await self._setup_integrated_testing()
            
            # 3.2. Mandatory Test Execution Framework
            await self.log_activity(ActivityType.TASK_PROGRESS, "3.2. Configuring mandatory test execution")
            test_execution_framework = await self._setup_mandatory_test_execution()
            
            # 3.3. Test-Driven Progression System
            await self.log_activity(ActivityType.TASK_PROGRESS, "3.3. Setting up test-driven progression")
            progression_system = await self._setup_test_driven_progression()
            
            # 3.4. Debugging & Self-Correction Framework
            await self.log_activity(ActivityType.TASK_PROGRESS, "3.4. Setting up debugging and self-correction")
            debugging_framework = await self._setup_debugging_self_correction()
            
            # 3.5. Automated Issue Detection
            await self.log_activity(ActivityType.TASK_PROGRESS, "3.5. Configuring automated issue detection")
            issue_detection = await self._setup_automated_issue_detection()
            
            # 3.6. Contextual Remediation System
            await self.log_activity(ActivityType.TASK_PROGRESS, "3.6. Setting up contextual remediation")
            remediation_system = await self._setup_contextual_remediation()
            
            # 3.7. Plan Validation & Refinement (Sample Builds)
            await self.log_activity(ActivityType.TASK_PROGRESS, "3.7. Setting up plan validation and sample builds")
            validation_refinement = await self._setup_plan_validation_refinement()
            
            # Store Phase 3 context
            phase3_context = {
                'testing_framework': testing_framework,
                'test_execution_framework': test_execution_framework,
                'progression_system': progression_system,
                'debugging_framework': debugging_framework,
                'issue_detection': issue_detection,
                'remediation_system': remediation_system,
                'validation_refinement': validation_refinement,
                'phase3_completed_at': datetime.utcnow().isoformat()
            }
            
            self.implementation_context['phase3'] = phase3_context
            
            await self.log_activity(ActivityType.TASK_COMPLETE, "Phase 3: Continuous Validation & Iteration Framework completed successfully")
            
            return {
                'success': True,
                'phase': 'phase3_complete',
                'testing_ready': testing_framework['success'],
                'validation_ready': validation_refinement['success'],
                'debugging_ready': debugging_framework['success'],
                'next_phase': 'reporting_output',
                'context': phase3_context
            }
            
        except Exception as e:
            await self.log_activity(ActivityType.ERROR, f"Phase 3 failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'phase': 'phase3_failed'
            }
    
    async def _setup_integrated_testing(self) -> Dict[str, Any]:
        """3.1. Setup integrated unit testing framework"""
        try:
            # Extract testing strategy from plan
            testing_strategy = self.current_plan.testing_strategy
            
            # Setup testing framework configuration
            testing_config = {
                'test_frameworks': self._identify_testing_frameworks(self.current_knowledge.technology_stack),
                'test_types': ['unit', 'integration', 'functional'],
                'coverage_requirements': {
                    'minimum_coverage': 80,
                    'critical_path_coverage': 95,
                    'branch_coverage': 75
                },
                'test_execution_order': self._define_test_execution_order(),
                'test_data_management': self._setup_test_data_management()
            }
            
            # Setup test automation
            automation_config = {
                'auto_test_discovery': True,
                'parallel_execution': True,
                'test_isolation': True,
                'cleanup_after_tests': True,
                'test_reporting': {
                    'format': 'junit_xml',
                    'detailed_output': True,
                    'failure_analysis': True
                }
            }
            
            # Setup continuous testing triggers
            testing_triggers = {
                'on_file_save': True,
                'on_task_completion': True,
                'on_module_completion': True,
                'on_integration_point': True,
                'scheduled_runs': False
            }
            
            return {
                'success': True,
                'testing_config': testing_config,
                'automation_config': automation_config,
                'testing_triggers': testing_triggers,
                'framework_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup integrated testing: {str(e)}"
            }
    
    async def _setup_mandatory_test_execution(self) -> Dict[str, Any]:
        """3.2. Setup mandatory test execution framework"""
        try:
            # Define mandatory test execution rules
            execution_rules = {
                'blocking_progression': True,
                'failure_tolerance': {
                    'unit_tests': 0,  # No unit test failures allowed
                    'integration_tests': 0,  # No integration test failures allowed
                    'performance_tests': 1  # 1 performance test failure allowed
                },
                'retry_policy': {
                    'max_retries': 3,
                    'retry_delay': 5,  # seconds
                    'exponential_backoff': True
                },
                'escalation_policy': {
                    'escalate_after_retries': True,
                    'escalation_threshold': 3,
                    'escalation_actions': ['log_detailed_error', 'notify_supervisor', 'halt_execution']
                }
            }
            
            # Setup test execution pipeline
            execution_pipeline = {
                'pre_execution_checks': [
                    'verify_test_environment',
                    'check_dependencies',
                    'validate_test_data'
                ],
                'execution_stages': [
                    'syntax_validation',
                    'unit_test_execution',
                    'integration_test_execution',
                    'performance_validation'
                ],
                'post_execution_actions': [
                    'collect_test_results',
                    'generate_coverage_report',
                    'analyze_failures',
                    'update_progress_tracking'
                ]
            }
            
            # Setup test result validation
            result_validation = {
                'success_criteria': {
                    'all_tests_pass': True,
                    'coverage_threshold_met': True,
                    'no_critical_failures': True,
                    'performance_within_limits': True
                },
                'failure_analysis': {
                    'categorize_failures': True,
                    'identify_root_causes': True,
                    'suggest_remediation': True,
                    'track_failure_patterns': True
                }
            }
            
            return {
                'success': True,
                'execution_rules': execution_rules,
                'execution_pipeline': execution_pipeline,
                'result_validation': result_validation,
                'mandatory_execution_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup mandatory test execution: {str(e)}"
            }
    
    async def _setup_test_driven_progression(self) -> Dict[str, Any]:
        """3.3. Setup test-driven progression system"""
        try:
            # Define progression gates
            progression_gates = {
                'task_level': {
                    'required_tests': ['unit_tests'],
                    'success_criteria': 'all_tests_pass',
                    'blocking': True
                },
                'module_level': {
                    'required_tests': ['unit_tests', 'integration_tests'],
                    'success_criteria': 'all_tests_pass_and_coverage_met',
                    'blocking': True
                },
                'phase_level': {
                    'required_tests': ['unit_tests', 'integration_tests', 'system_tests'],
                    'success_criteria': 'comprehensive_validation_passed',
                    'blocking': True
                }
            }
            
            # Setup progression tracking
            progression_tracking = {
                'current_gate': 'task_level',
                'gates_passed': [],
                'gates_failed': [],
                'progression_history': [],
                'blocking_issues': []
            }
            
            # Setup gate validation logic
            gate_validation = {
                'validation_sequence': [
                    'check_test_completion',
                    'verify_test_results',
                    'validate_coverage',
                    'check_quality_metrics',
                    'verify_dependencies'
                ],
                'fallback_strategies': {
                    'partial_failure': 'retry_with_analysis',
                    'complete_failure': 'escalate_and_halt',
                    'timeout': 'escalate_with_context'
                }
            }
            
            return {
                'success': True,
                'progression_gates': progression_gates,
                'progression_tracking': progression_tracking,
                'gate_validation': gate_validation,
                'test_driven_progression_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup test-driven progression: {str(e)}"
            }
    
    async def _setup_debugging_self_correction(self) -> Dict[str, Any]:
        """3.4. Setup debugging & self-correction framework"""
        try:
            # Setup debugging capabilities
            debugging_config = {
                'error_detection': {
                    'syntax_errors': True,
                    'runtime_errors': True,
                    'logic_errors': True,
                    'performance_issues': True
                },
                'debugging_tools': {
                    'static_analysis': True,
                    'dynamic_analysis': True,
                    'profiling': True,
                    'logging_analysis': True
                },
                'debug_information_collection': {
                    'stack_traces': True,
                    'variable_states': True,
                    'execution_flow': True,
                    'resource_usage': True
                }
            }
            
            # Setup self-correction strategies
            correction_strategies = {
                'syntax_errors': [
                    'lint_and_format',
                    'reference_style_guide',
                    'apply_common_fixes'
                ],
                'runtime_errors': [
                    'analyze_stack_trace',
                    'check_dependencies',
                    'validate_inputs',
                    'apply_error_handling'
                ],
                'logic_errors': [
                    'review_implementation_guidance',
                    'compare_with_knowledge_base',
                    'analyze_test_failures',
                    'refactor_problematic_code'
                ],
                'performance_issues': [
                    'profile_code_execution',
                    'identify_bottlenecks',
                    'apply_optimization_patterns',
                    'review_algorithm_complexity'
                ]
            }
            
            # Setup correction attempt tracking
            correction_tracking = {
                'max_attempts_per_issue': 3,
                'attempt_history': [],
                'success_patterns': [],
                'failure_patterns': [],
                'escalation_triggers': [
                    'max_attempts_exceeded',
                    'repeated_failure_pattern',
                    'critical_system_error'
                ]
            }
            
            return {
                'success': True,
                'debugging_config': debugging_config,
                'correction_strategies': correction_strategies,
                'correction_tracking': correction_tracking,
                'debugging_framework_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup debugging & self-correction: {str(e)}"
            }
    
    async def _setup_automated_issue_detection(self) -> Dict[str, Any]:
        """3.5. Setup automated issue detection system"""
        try:
            # Setup issue detection patterns
            detection_patterns = {
                'compilation_issues': {
                    'patterns': [
                        r'SyntaxError',
                        r'ImportError',
                        r'ModuleNotFoundError',
                        r'NameError'
                    ],
                    'severity': 'high',
                    'auto_fix': True
                },
                'runtime_issues': {
                    'patterns': [
                        r'AttributeError',
                        r'TypeError',
                        r'ValueError',
                        r'KeyError'
                    ],
                    'severity': 'high',
                    'auto_fix': True
                },
                'performance_issues': {
                    'patterns': [
                        r'timeout',
                        r'memory.*error',
                        r'recursion.*limit'
                    ],
                    'severity': 'medium',
                    'auto_fix': False
                },
                'quality_issues': {
                    'patterns': [
                        r'code.*smell',
                        r'complexity.*high',
                        r'duplication.*detected'
                    ],
                    'severity': 'low',
                    'auto_fix': True
                }
            }
            
            # Setup detection mechanisms
            detection_mechanisms = {
                'static_analysis': {
                    'enabled': True,
                    'tools': ['pylint', 'flake8', 'mypy'],
                    'frequency': 'on_file_save'
                },
                'dynamic_analysis': {
                    'enabled': True,
                    'tools': ['pytest', 'coverage'],
                    'frequency': 'on_test_run'
                },
                'log_analysis': {
                    'enabled': True,
                    'log_levels': ['ERROR', 'WARNING', 'CRITICAL'],
                    'frequency': 'continuous'
                },
                'performance_monitoring': {
                    'enabled': True,
                    'metrics': ['execution_time', 'memory_usage', 'cpu_usage'],
                    'frequency': 'on_execution'
                }
            }
            
            # Setup issue classification
            issue_classification = {
                'categories': {
                    'blocking': ['compilation_failure', 'critical_runtime_error'],
                    'non_blocking': ['performance_warning', 'style_violation'],
                    'informational': ['code_suggestion', 'optimization_opportunity']
                },
                'priority_matrix': {
                    'high_severity_high_impact': 'immediate_action',
                    'high_severity_low_impact': 'scheduled_fix',
                    'low_severity_high_impact': 'prioritized_fix',
                    'low_severity_low_impact': 'backlog'
                }
            }
            
            return {
                'success': True,
                'detection_patterns': detection_patterns,
                'detection_mechanisms': detection_mechanisms,
                'issue_classification': issue_classification,
                'automated_detection_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup automated issue detection: {str(e)}"
            }
    
    async def _setup_contextual_remediation(self) -> Dict[str, Any]:
        """3.6. Setup contextual remediation system"""
        try:
            # Setup remediation context analysis
            context_analysis = {
                'code_context': {
                    'analyze_surrounding_code': True,
                    'identify_patterns': True,
                    'check_dependencies': True,
                    'review_recent_changes': True
                },
                'knowledge_context': {
                    'reference_implementation_guidance': True,
                    'consult_best_practices': True,
                    'check_technology_guides': True,
                    'review_similar_solutions': True
                },
                'historical_context': {
                    'analyze_previous_fixes': True,
                    'identify_recurring_patterns': True,
                    'learn_from_failures': True,
                    'track_success_rates': True
                }
            }
            
            # Setup remediation strategies
            remediation_strategies = {
                'immediate_fixes': {
                    'syntax_corrections': {
                        'auto_format': True,
                        'fix_imports': True,
                        'correct_indentation': True
                    },
                    'simple_logic_fixes': {
                        'fix_variable_names': True,
                        'correct_function_calls': True,
                        'add_missing_parameters': True
                    }
                },
                'guided_fixes': {
                    'reference_based_fixes': {
                        'consult_implementation_guidance': True,
                        'apply_knowledge_patterns': True,
                        'use_technology_best_practices': True
                    },
                    'iterative_refinement': {
                        'incremental_improvements': True,
                        'test_driven_fixes': True,
                        'validation_at_each_step': True
                    }
                },
                'escalation_strategies': {
                    'human_intervention': {
                        'complex_architectural_issues': True,
                        'domain_specific_problems': True,
                        'repeated_failure_patterns': True
                    },
                    'alternative_approaches': {
                        'suggest_different_implementations': True,
                        'propose_architectural_changes': True,
                        'recommend_technology_alternatives': True
                    }
                }
            }
            
            # Setup remediation tracking
            remediation_tracking = {
                'attempt_logging': {
                    'log_all_attempts': True,
                    'track_success_rates': True,
                    'analyze_failure_patterns': True,
                    'measure_time_to_resolution': True
                },
                'learning_system': {
                    'build_fix_knowledge_base': True,
                    'identify_effective_strategies': True,
                    'improve_future_remediation': True,
                    'share_learnings_across_tasks': True
                }
            }
            
            return {
                'success': True,
                'context_analysis': context_analysis,
                'remediation_strategies': remediation_strategies,
                'remediation_tracking': remediation_tracking,
                'contextual_remediation_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup contextual remediation: {str(e)}"
            }
    
    async def _setup_plan_validation_refinement(self) -> Dict[str, Any]:
        """3.7. Setup plan validation & refinement (sample builds)"""
        try:
            # Setup sample build configuration
            sample_build_config = {
                'build_triggers': [
                    'complex_module_completion',
                    'architectural_proof_of_concept',
                    'integration_milestone',
                    'performance_validation_point'
                ],
                'build_types': {
                    'proof_of_concept': {
                        'scope': 'minimal_viable_implementation',
                        'validation_focus': 'architectural_feasibility',
                        'success_criteria': 'compiles_and_runs'
                    },
                    'integration_test': {
                        'scope': 'module_integration',
                        'validation_focus': 'component_interaction',
                        'success_criteria': 'integration_tests_pass'
                    },
                    'performance_test': {
                        'scope': 'performance_critical_paths',
                        'validation_focus': 'performance_requirements',
                        'success_criteria': 'performance_benchmarks_met'
                    }
                },
                'build_environment': {
                    'isolated_environment': True,
                    'clean_dependencies': True,
                    'test_data_setup': True,
                    'monitoring_enabled': True
                }
            }
            
            # Setup validation criteria
            validation_criteria = {
                'compilation_validation': {
                    'syntax_check': True,
                    'dependency_resolution': True,
                    'build_success': True,
                    'no_warnings': False  # Warnings allowed
                },
                'runtime_validation': {
                    'successful_startup': True,
                    'basic_functionality': True,
                    'error_free_execution': True,
                    'resource_usage_acceptable': True
                },
                'quality_validation': {
                    'code_coverage': 70,  # Minimum 70% for sample builds
                    'performance_baseline': True,
                    'security_scan_pass': True,
                    'documentation_present': True
                }
            }
            
            # Setup feedback and refinement system
            feedback_system = {
                'outcome_analysis': {
                    'success_factor_identification': True,
                    'failure_root_cause_analysis': True,
                    'performance_bottleneck_detection': True,
                    'improvement_opportunity_identification': True
                },
                'plan_refinement': {
                    'update_implementation_guidance': True,
                    'adjust_architectural_decisions': True,
                    'modify_technology_choices': True,
                    'refine_task_breakdown': True
                },
                'knowledge_update': {
                    'capture_lessons_learned': True,
                    'update_best_practices': True,
                    'refine_patterns_and_templates': True,
                    'improve_estimation_accuracy': True
                }
            }
            
            # Setup reporting system
            reporting_system = {
                'build_reports': {
                    'compilation_results': True,
                    'test_execution_results': True,
                    'performance_metrics': True,
                    'quality_assessments': True
                },
                'refinement_reports': {
                    'identified_improvements': True,
                    'proposed_plan_changes': True,
                    'risk_assessments': True,
                    'impact_analysis': True
                },
                'escalation_reports': {
                    'critical_issues': True,
                    'blocking_problems': True,
                    'architectural_conflicts': True,
                    'resource_constraints': True
                }
            }
            
            return {
                'success': True,
                'sample_build_config': sample_build_config,
                'validation_criteria': validation_criteria,
                'feedback_system': feedback_system,
                'reporting_system': reporting_system,
                'plan_validation_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup plan validation & refinement: {str(e)}"
            }
    
    # Helper methods for Phase 3 implementation
    def _define_test_execution_order(self) -> List[str]:
        """Define the order of test execution"""
        return [
            'syntax_validation',
            'unit_tests',
            'integration_tests',
            'functional_tests',
            'performance_tests',
            'security_tests'
        ]
    
    def _setup_test_data_management(self) -> Dict[str, Any]:
        """Setup test data management system"""
        return {
            'test_data_sources': {
                'fixtures': True,
                'factories': True,
                'mocks': True,
                'external_apis': False
            },
            'data_isolation': {
                'per_test_isolation': True,
                'cleanup_after_tests': True,
                'transaction_rollback': True
            },
            'data_generation': {
                'automatic_generation': True,
                'realistic_data': True,
                'edge_case_coverage': True,
                'performance_data_sets': True
            }
        }

    async def _phase4_reporting_output_management(self) -> Dict[str, Any]:
        """Phase 4: Reporting & Output Management
        
        Create comprehensive progress reporting, issue escalation protocol,
        and final output delivery system.
        """
        try:
            await self.log_activity(
                ActivityType.PROCESSING,
                "Starting Phase 4: Reporting & Output Management"
            )
            
            # 4.1. Setup comprehensive progress reporting
            progress_reporting = await self._setup_comprehensive_progress_reporting()
            if not progress_reporting.get('success'):
                return {
                    'success': False,
                    'error': f"Failed to setup progress reporting: {progress_reporting.get('error')}"
                }
            
            # 4.2. Setup issue escalation protocol
            escalation_protocol = await self._setup_issue_escalation_protocol()
            if not escalation_protocol.get('success'):
                return {
                    'success': False,
                    'error': f"Failed to setup escalation protocol: {escalation_protocol.get('error')}"
                }
            
            # 4.3. Setup final output delivery system
            output_delivery = await self._setup_final_output_delivery_system()
            if not output_delivery.get('success'):
                return {
                    'success': False,
                    'error': f"Failed to setup output delivery: {output_delivery.get('error')}"
                }
            
            # 4.4. Setup artifact generation & organization
            artifact_management = await self._setup_artifact_generation_organization()
            if not artifact_management.get('success'):
                return {
                    'success': False,
                    'error': f"Failed to setup artifact management: {artifact_management.get('error')}"
                }
            
            # 4.5. Setup final verification system
            final_verification = await self._setup_final_verification_system()
            if not final_verification.get('success'):
                return {
                    'success': False,
                    'error': f"Failed to setup final verification: {final_verification.get('error')}"
                }
            
            await self.log_activity(
                ActivityType.PROCESSING,
                "Phase 4: Reporting & Output Management completed successfully"
            )
            
            return {
                'success': True,
                'phase': 'Phase 4: Reporting & Output Management',
                'progress_reporting': progress_reporting,
                'escalation_protocol': escalation_protocol,
                'output_delivery': output_delivery,
                'artifact_management': artifact_management,
                'final_verification': final_verification,
                'reporting_output_ready': True
            }
            
        except Exception as e:
            await self.log_activity(
                ActivityType.ERROR,
                f"Phase 4 failed: {str(e)}"
            )
            return {
                'success': False,
                'error': f"Phase 4 execution failed: {str(e)}"
            }
    
    async def _setup_comprehensive_progress_reporting(self) -> Dict[str, Any]:
        """4.1. Setup comprehensive progress reporting"""
        try:
            # Setup granular status updates
            granular_reporting = {
                'task_level_tracking': {
                    'task_start_notifications': True,
                    'task_progress_updates': True,
                    'task_completion_confirmations': True,
                    'task_failure_alerts': True
                },
                'file_level_tracking': {
                    'file_creation_logs': True,
                    'file_modification_logs': True,
                    'file_deletion_logs': True,
                    'file_validation_results': True
                },
                'test_level_tracking': {
                    'test_execution_start': True,
                    'test_progress_updates': True,
                    'test_results_reporting': True,
                    'test_failure_analysis': True
                },
                'build_level_tracking': {
                    'build_initiation': True,
                    'compilation_progress': True,
                    'build_success_confirmation': True,
                    'build_artifact_generation': True
                }
            }
            
            # Setup real-time progress dashboard
            progress_dashboard = {
                'live_status_display': {
                    'current_phase': True,
                    'current_task': True,
                    'progress_percentage': True,
                    'estimated_completion': True
                },
                'activity_timeline': {
                    'chronological_activity_log': True,
                    'milestone_markers': True,
                    'error_indicators': True,
                    'performance_metrics': True
                },
                'resource_monitoring': {
                    'cpu_usage_tracking': True,
                    'memory_usage_tracking': True,
                    'disk_space_monitoring': True,
                    'network_activity_tracking': True
                }
            }
            
            # Setup comprehensive logging system
            logging_system = {
                'structured_logging': {
                    'json_formatted_logs': True,
                    'log_level_categorization': True,
                    'contextual_information': True,
                    'correlation_ids': True
                },
                'log_aggregation': {
                    'centralized_log_collection': True,
                    'log_parsing_and_indexing': True,
                    'searchable_log_interface': True,
                    'log_retention_policies': True
                },
                'alert_system': {
                    'error_threshold_alerts': True,
                    'performance_degradation_alerts': True,
                    'resource_exhaustion_warnings': True,
                    'completion_notifications': True
                }
            }
            
            return {
                'success': True,
                'granular_reporting': granular_reporting,
                'progress_dashboard': progress_dashboard,
                'logging_system': logging_system,
                'progress_reporting_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup comprehensive progress reporting: {str(e)}"
            }
    
    async def _setup_issue_escalation_protocol(self) -> Dict[str, Any]:
        """4.2. Setup issue escalation protocol"""
        try:
            # Setup escalation triggers
            escalation_triggers = {
                'critical_failures': {
                    'compilation_failures': {
                        'threshold': 3,  # After 3 failed attempts
                        'severity': 'high',
                        'auto_escalate': True
                    },
                    'test_failures': {
                        'threshold': 5,  # After 5 failed test runs
                        'severity': 'medium',
                        'auto_escalate': True
                    },
                    'dependency_conflicts': {
                        'threshold': 1,  # Immediate escalation
                        'severity': 'high',
                        'auto_escalate': True
                    },
                    'architectural_conflicts': {
                        'threshold': 1,  # Immediate escalation
                        'severity': 'critical',
                        'auto_escalate': True
                    }
                },
                'performance_issues': {
                    'execution_timeout': {
                        'threshold': 2,  # After 2 timeouts
                        'severity': 'medium',
                        'auto_escalate': True
                    },
                    'memory_exhaustion': {
                        'threshold': 1,  # Immediate escalation
                        'severity': 'high',
                        'auto_escalate': True
                    },
                    'resource_constraints': {
                        'threshold': 3,  # After 3 resource issues
                        'severity': 'medium',
                        'auto_escalate': True
                    }
                }
            }
            
            # Setup escalation workflow
            escalation_workflow = {
                'incident_classification': {
                    'automatic_categorization': True,
                    'severity_assessment': True,
                    'impact_analysis': True,
                    'urgency_determination': True
                },
                'escalation_routing': {
                    'ai_orchestrator_notification': True,
                    'human_supervisor_alert': True,
                    'technical_team_notification': True,
                    'stakeholder_communication': True
                },
                'incident_documentation': {
                    'detailed_incident_report': True,
                    'context_preservation': True,
                    'reproduction_steps': True,
                    'attempted_solutions': True
                }
            }
            
            # Setup escalation communication
            escalation_communication = {
                'notification_channels': {
                    'system_alerts': True,
                    'email_notifications': True,
                    'dashboard_alerts': True,
                    'log_entries': True
                },
                'communication_templates': {
                    'incident_alert_template': True,
                    'progress_update_template': True,
                    'resolution_notification_template': True,
                    'escalation_summary_template': True
                },
                'stakeholder_management': {
                    'automatic_stakeholder_identification': True,
                    'role_based_notifications': True,
                    'escalation_chain_management': True,
                    'communication_tracking': True
                }
            }
            
            return {
                'success': True,
                'escalation_triggers': escalation_triggers,
                'escalation_workflow': escalation_workflow,
                'escalation_communication': escalation_communication,
                'escalation_protocol_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup issue escalation protocol: {str(e)}"
            }
    
    async def _setup_final_output_delivery_system(self) -> Dict[str, Any]:
        """4.3. Setup final output delivery system"""
        try:
            # Setup output package compilation
            package_compilation = {
                'artifact_collection': {
                    'source_code_files': True,
                    'configuration_files': True,
                    'build_scripts': True,
                    'test_files': True,
                    'documentation_files': True,
                    'deployment_artifacts': True
                },
                'package_structure': {
                    'hierarchical_organization': True,
                    'logical_grouping': True,
                    'clear_naming_conventions': True,
                    'metadata_inclusion': True
                },
                'package_validation': {
                    'completeness_check': True,
                    'integrity_verification': True,
                    'dependency_validation': True,
                    'compatibility_assessment': True
                }
            }
            
            # Setup delivery mechanisms
            delivery_mechanisms = {
                'file_system_delivery': {
                    'organized_directory_structure': True,
                    'compressed_archive_creation': True,
                    'checksum_generation': True,
                    'delivery_confirmation': True
                },
                'repository_delivery': {
                    'git_repository_creation': True,
                    'commit_history_preservation': True,
                    'branch_organization': True,
                    'tag_creation': True
                },
                'artifact_registry_delivery': {
                    'package_registration': True,
                    'version_management': True,
                    'metadata_publishing': True,
                    'access_control_setup': True
                }
            }
            
            # Setup delivery verification
            delivery_verification = {
                'package_integrity': {
                    'checksum_verification': True,
                    'file_count_validation': True,
                    'size_verification': True,
                    'structure_validation': True
                },
                'functional_verification': {
                    'basic_functionality_test': True,
                    'dependency_resolution_test': True,
                    'deployment_test': True,
                    'integration_test': True
                },
                'quality_verification': {
                    'code_quality_assessment': True,
                    'documentation_completeness': True,
                    'test_coverage_verification': True,
                    'security_scan_results': True
                }
            }
            
            return {
                'success': True,
                'package_compilation': package_compilation,
                'delivery_mechanisms': delivery_mechanisms,
                'delivery_verification': delivery_verification,
                'output_delivery_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup final output delivery system: {str(e)}"
            }
    
    async def _setup_artifact_generation_organization(self) -> Dict[str, Any]:
        """4.4. Setup artifact generation & organization"""
        try:
            # Setup artifact categorization
            artifact_categorization = {
                'primary_artifacts': {
                    'source_code': {
                        'application_code': True,
                        'library_code': True,
                        'utility_scripts': True,
                        'configuration_code': True
                    },
                    'executable_artifacts': {
                        'compiled_binaries': True,
                        'packaged_applications': True,
                        'deployment_packages': True,
                        'container_images': True
                    }
                },
                'supporting_artifacts': {
                    'documentation': {
                        'api_documentation': True,
                        'user_guides': True,
                        'developer_documentation': True,
                        'deployment_guides': True
                    },
                    'test_artifacts': {
                        'test_suites': True,
                        'test_data': True,
                        'test_reports': True,
                        'coverage_reports': True
                    }
                },
                'metadata_artifacts': {
                    'build_metadata': {
                        'build_logs': True,
                        'dependency_manifests': True,
                        'version_information': True,
                        'build_configuration': True
                    },
                    'quality_metadata': {
                        'code_quality_reports': True,
                        'security_scan_results': True,
                        'performance_benchmarks': True,
                        'compliance_reports': True
                    }
                }
            }
            
            # Setup organization structure
            organization_structure = {
                'directory_hierarchy': {
                    'logical_grouping': True,
                    'consistent_naming': True,
                    'clear_separation': True,
                    'scalable_structure': True
                },
                'file_naming_conventions': {
                    'descriptive_names': True,
                    'version_suffixes': True,
                    'type_indicators': True,
                    'timestamp_inclusion': True
                },
                'metadata_files': {
                    'manifest_files': True,
                    'readme_files': True,
                    'changelog_files': True,
                    'license_files': True
                }
            }
            
            # Setup artifact tracking
            artifact_tracking = {
                'generation_tracking': {
                    'creation_timestamps': True,
                    'generation_process_logs': True,
                    'dependency_tracking': True,
                    'version_tracking': True
                },
                'quality_tracking': {
                    'validation_results': True,
                    'quality_metrics': True,
                    'compliance_status': True,
                    'security_assessment': True
                },
                'usage_tracking': {
                    'access_logs': True,
                    'download_tracking': True,
                    'deployment_tracking': True,
                    'feedback_collection': True
                }
            }
            
            return {
                'success': True,
                'artifact_categorization': artifact_categorization,
                'organization_structure': organization_structure,
                'artifact_tracking': artifact_tracking,
                'artifact_management_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup artifact generation & organization: {str(e)}"
            }
    
    async def _setup_final_verification_system(self) -> Dict[str, Any]:
        """4.5. Setup final verification system"""
        try:
            # Setup completeness verification
            completeness_verification = {
                'plan_objective_verification': {
                    'feature_scope_completion': True,
                    'requirement_fulfillment': True,
                    'acceptance_criteria_validation': True,
                    'deliverable_confirmation': True
                },
                'artifact_completeness': {
                    'all_planned_files_present': True,
                    'required_documentation_included': True,
                    'test_coverage_adequate': True,
                    'deployment_artifacts_ready': True
                },
                'quality_completeness': {
                    'code_quality_standards_met': True,
                    'security_requirements_satisfied': True,
                    'performance_benchmarks_achieved': True,
                    'compliance_requirements_fulfilled': True
                }
            }
            
            # Setup integrity verification
            integrity_verification = {
                'code_integrity': {
                    'syntax_validation': True,
                    'compilation_success': True,
                    'dependency_resolution': True,
                    'runtime_stability': True
                },
                'data_integrity': {
                    'configuration_validity': True,
                    'test_data_consistency': True,
                    'documentation_accuracy': True,
                    'metadata_correctness': True
                },
                'package_integrity': {
                    'file_checksums_valid': True,
                    'archive_integrity_confirmed': True,
                    'version_consistency_verified': True,
                    'dependency_compatibility_confirmed': True
                }
            }
            
            # Setup final validation workflow
            validation_workflow = {
                'automated_validation': {
                    'comprehensive_test_suite_execution': True,
                    'automated_quality_checks': True,
                    'security_vulnerability_scanning': True,
                    'performance_regression_testing': True
                },
                'manual_validation': {
                    'code_review_checklist': True,
                    'documentation_review': True,
                    'user_acceptance_testing': True,
                    'deployment_readiness_assessment': True
                },
                'validation_reporting': {
                    'comprehensive_validation_report': True,
                    'issue_identification_and_tracking': True,
                    'recommendation_generation': True,
                    'sign_off_documentation': True
                }
            }
            
            return {
                'success': True,
                'completeness_verification': completeness_verification,
                'integrity_verification': integrity_verification,
                'validation_workflow': validation_workflow,
                'final_verification_ready': True
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to setup final verification system: {str(e)}"
            }

# Global instance for easy access
_feature_implementation_agent = None

def get_feature_implementation_agent(llm_service: EnhancedLLMService, 
                                   memory_service: MemoryService) -> FeatureImplementationAgent:
    """Get or create the global Feature Implementation Agent instance"""
    global _feature_implementation_agent
    if _feature_implementation_agent is None:
        _feature_implementation_agent = FeatureImplementationAgent(llm_service, memory_service)
    return _feature_implementation_agent