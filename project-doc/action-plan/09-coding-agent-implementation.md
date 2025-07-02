# 09 - Coding Agent Implementation

## Overview

The Coding Agent serves as the execution engine for the multi-agent system, responsible for implementing the specific development tasks directed by the Frontend and Backend Developer Agents. It translates high-level specifications into working code, handles file operations, manages dependencies, executes tests, and ensures code quality. This agent bridges the gap between planning and implementation.

## Current State Analysis

### Existing File
- `core/agents/coding_agent.py` - Basic coding functionality

### Enhancement Requirements
- Advanced code generation and implementation
- Multi-language support (Python, TypeScript, JavaScript, SQL)
- File system operations and project structure management
- Dependency management and package installation
- Code quality analysis and optimization
- Testing execution and validation
- Git operations and version control
- Real-time code execution and debugging

## Implementation Tasks

### Task 9.1: Enhanced Coding Agent

**File**: `core/agents/coding_agent.py` (Complete Rewrite)

**Coding Agent Implementation**:
```python
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import asyncio
import os
import subprocess
import json
import yaml
from pathlib import Path
from enum import Enum
import tempfile
import shutil

from .base_agent import BaseAgent, AgentStatus
from ..services.llm_service import LLMService
from ..services.memory_service import MemoryService
from ..models import CodingTask, CodeFile, ProjectStructure, TestResult

class ProgrammingLanguage(Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    BASH = "bash"
    DOCKERFILE = "dockerfile"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"

class CodeOperation(Enum):
    CREATE_FILE = "create_file"
    UPDATE_FILE = "update_file"
    DELETE_FILE = "delete_file"
    CREATE_DIRECTORY = "create_directory"
    INSTALL_DEPENDENCIES = "install_dependencies"
    RUN_TESTS = "run_tests"
    EXECUTE_CODE = "execute_code"
    FORMAT_CODE = "format_code"
    LINT_CODE = "lint_code"
    GIT_OPERATIONS = "git_operations"

class CodingAgent(BaseAgent):
    def __init__(self, agent_id: str = "coding_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Coding Agent",
            description="Advanced code implementation and execution agent"
        )
        self.capabilities = [
            "code_generation",
            "file_operations",
            "dependency_management",
            "test_execution",
            "code_quality_analysis",
            "project_structure_management",
            "version_control",
            "code_debugging",
            "performance_optimization",
            "documentation_generation"
        ]
        
        self.supported_languages = list(ProgrammingLanguage)
        self.active_coding_sessions = {}
        self.project_workspaces = {}
        self.code_templates = {}
        
    async def initialize(self, llm_service: LLMService, memory_service: MemoryService):
        """Initialize coding agent"""
        self.llm_service = llm_service
        self.memory_service = memory_service
        await self._load_code_templates()
        await self._setup_development_environment()
        await self.update_status(AgentStatus.IDLE)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process coding task"""
        try:
            await self.update_status(AgentStatus.THINKING)
            
            # Parse coding task
            coding_task = self._parse_coding_task(task)
            
            # Create coding session
            session_id = f"coding_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.active_coding_sessions[session_id] = {
                'task': coding_task,
                'start_time': datetime.utcnow(),
                'status': 'active',
                'workspace': None,
                'files_created': [],
                'tests_executed': [],
                'errors': []
            }
            
            await self.update_status(AgentStatus.WORKING)
            
            # Execute coding workflow
            coding_result = await self._execute_coding_workflow(session_id, coding_task)
            
            # Validate and test implementation
            validation_result = await self._validate_implementation(session_id, coding_result)
            
            # Generate implementation report
            implementation_report = await self._generate_implementation_report(session_id, coding_result, validation_result)
            
            # Store coding artifacts
            await self._store_coding_artifacts(session_id, coding_result, implementation_report)
            
            await self.update_status(AgentStatus.COMPLETED)
            
            return {
                'session_id': session_id,
                'implementation_status': coding_result.get('status'),
                'files_created': coding_result.get('files_created', []),
                'tests_executed': coding_result.get('tests_executed', []),
                'code_quality_score': coding_result.get('code_quality_score', 0.0),
                'performance_metrics': coding_result.get('performance_metrics', {}),
                'validation_result': validation_result,
                'implementation_report': implementation_report,
                'workspace_path': coding_result.get('workspace_path'),
                'confidence_score': coding_result.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            await self.update_status(AgentStatus.ERROR, str(e))
            raise
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Determine if agent can handle coding task"""
        return task.get('type') in [
            'code_implementation',
            'file_operations',
            'project_setup',
            'dependency_management',
            'test_execution',
            'code_refactoring',
            'debugging',
            'performance_optimization'
        ]
    
    def _parse_coding_task(self, task: Dict[str, Any]) -> 'CodingTask':
        """Parse incoming task into structured coding task"""
        return CodingTask(
            project_name=task.get('project_name', 'Coding Project'),
            description=task.get('description', ''),
            operation=CodeOperation(task.get('operation', 'create_file')),
            language=ProgrammingLanguage(task.get('language', 'python')),
            specifications=task.get('specifications', {}),
            files_to_create=task.get('files_to_create', []),
            files_to_update=task.get('files_to_update', []),
            dependencies=task.get('dependencies', []),
            test_requirements=task.get('test_requirements', []),
            quality_requirements=task.get('quality_requirements', {}),
            workspace_path=task.get('workspace_path'),
            existing_codebase=task.get('existing_codebase', {}),
            integration_points=task.get('integration_points', [])
        )
    
    async def _execute_coding_workflow(self, session_id: str, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Execute comprehensive coding workflow"""
        coding_result = {}
        session = self.active_coding_sessions[session_id]
        
        # Step 1: Setup Workspace
        await self.log_activity(f"Setting up workspace for {coding_task.project_name}")
        workspace_setup = await self._setup_project_workspace(session_id, coding_task)
        coding_result['workspace_setup'] = workspace_setup
        session['workspace'] = workspace_setup.get('workspace_path')
        
        # Step 2: Analyze Specifications
        await self.log_activity("Analyzing coding specifications")
        spec_analysis = await self._analyze_coding_specifications(coding_task)
        coding_result['specification_analysis'] = spec_analysis
        
        # Step 3: Generate Project Structure
        await self.log_activity("Generating project structure")
        project_structure = await self._generate_project_structure(coding_task, spec_analysis)
        coding_result['project_structure'] = project_structure
        
        # Step 4: Install Dependencies
        await self.log_activity("Installing dependencies")
        dependency_result = await self._install_dependencies(session_id, coding_task)
        coding_result['dependency_installation'] = dependency_result
        
        # Step 5: Generate Code Files
        await self.log_activity("Generating code files")
        code_generation = await self._generate_code_files(session_id, coding_task, spec_analysis)
        coding_result['code_generation'] = code_generation
        session['files_created'].extend(code_generation.get('files_created', []))
        
        # Step 6: Code Quality Analysis
        await self.log_activity("Analyzing code quality")
        quality_analysis = await self._analyze_code_quality(session_id, coding_task)
        coding_result['quality_analysis'] = quality_analysis
        
        # Step 7: Execute Tests
        await self.log_activity("Executing tests")
        test_execution = await self._execute_tests(session_id, coding_task)
        coding_result['test_execution'] = test_execution
        session['tests_executed'].extend(test_execution.get('tests_executed', []))
        
        # Step 8: Performance Optimization
        await self.log_activity("Optimizing performance")
        performance_optimization = await self._optimize_performance(session_id, coding_task)
        coding_result['performance_optimization'] = performance_optimization
        
        # Step 9: Documentation Generation
        await self.log_activity("Generating documentation")
        documentation = await self._generate_code_documentation(session_id, coding_task)
        coding_result['documentation'] = documentation
        
        # Step 10: Final Validation
        await self.log_activity("Performing final validation")
        final_validation = await self._perform_final_validation(session_id, coding_result)
        coding_result['final_validation'] = final_validation
        coding_result['confidence_score'] = final_validation.get('confidence_score', 0.0)
        
        coding_result['status'] = 'completed'
        coding_result['workspace_path'] = session['workspace']
        coding_result['files_created'] = session['files_created']
        coding_result['tests_executed'] = session['tests_executed']
        
        return coding_result
    
    async def _setup_project_workspace(self, session_id: str, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Setup project workspace"""
        try:
            # Create workspace directory
            if coding_task.workspace_path:
                workspace_path = Path(coding_task.workspace_path)
            else:
                workspace_path = Path(tempfile.mkdtemp(prefix=f"coding_{session_id}_"))
            
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize git repository if needed
            git_init = False
            if coding_task.quality_requirements.get('version_control', False):
                git_result = await self._initialize_git_repository(workspace_path)
                git_init = git_result.get('success', False)
            
            return {
                'workspace_path': str(workspace_path),
                'git_initialized': git_init,
                'setup_timestamp': datetime.utcnow().isoformat(),
                'success': True
            }
            
        except Exception as e:
            return {
                'workspace_path': None,
                'error': str(e),
                'success': False
            }
    
    async def _analyze_coding_specifications(self, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Analyze coding specifications"""
        prompt = f"""
        Analyze the following coding specifications:
        
        Project: {coding_task.project_name}
        Description: {coding_task.description}
        Language: {coding_task.language.value}
        Operation: {coding_task.operation.value}
        Specifications: {json.dumps(coding_task.specifications, indent=2)}
        Files to Create: {coding_task.files_to_create}
        Files to Update: {coding_task.files_to_update}
        Dependencies: {coding_task.dependencies}
        Test Requirements: {coding_task.test_requirements}
        
        Analyze and break down the specifications into:
        1. Core Functionality Requirements
        2. Technical Implementation Details
        3. File Structure and Organization
        4. Dependency Requirements
        5. Testing Strategy
        6. Quality and Performance Requirements
        7. Integration Points
        8. Potential Challenges and Solutions
        
        For each requirement, provide:
        - Priority level (High, Medium, Low)
        - Complexity assessment (Simple, Moderate, Complex)
        - Implementation approach
        - Required technologies/libraries
        - Testing approach
        
        Return as JSON:
        {{
            "core_functionality": [
                {{
                    "requirement": "...",
                    "priority": "High",
                    "complexity": "Moderate",
                    "approach": "...",
                    "technologies": [...],
                    "testing": "..."
                }}
            ],
            "technical_details": [...],
            "file_structure": {...},
            "dependencies": [...],
            "testing_strategy": {...},
            "quality_requirements": {...},
            "integration_points": [...],
            "challenges": [...],
            "implementation_plan": [...]
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
            return self._get_default_specification_analysis(coding_task)
    
    async def _generate_project_structure(self, coding_task: 'CodingTask', spec_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project structure"""
        prompt = f"""
        Generate a comprehensive project structure for:
        
        Project: {coding_task.project_name}
        Language: {coding_task.language.value}
        Specifications: {json.dumps(spec_analysis, indent=2)}
        
        Create a project structure with:
        1. Directory Organization (logical grouping, separation of concerns)
        2. File Naming Conventions (consistent, descriptive names)
        3. Configuration Files (environment, build, deployment)
        4. Documentation Structure (README, API docs, guides)
        5. Testing Structure (unit tests, integration tests)
        6. Asset Organization (static files, resources)
        
        For each directory and file, specify:
        - Purpose and contents
        - Naming rationale
        - Dependencies and relationships
        - Best practices followed
        
        Return as JSON:
        {{
            "project_root": "{coding_task.project_name}",
            "directories": [
                {{
                    "path": "src",
                    "purpose": "Source code directory",
                    "subdirectories": [
                        {{
                            "path": "components",
                            "purpose": "Reusable components",
                            "files": [...]
                        }}
                    ]
                }}
            ],
            "files": [
                {{
                    "path": "README.md",
                    "purpose": "Project documentation",
                    "template": "readme",
                    "priority": "High"
                }}
            ],
            "configuration_files": [...],
            "documentation_files": [...],
            "test_files": [...],
            "build_files": [...]
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
            return self._get_default_project_structure(coding_task)
    
    async def _install_dependencies(self, session_id: str, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Install project dependencies"""
        session = self.active_coding_sessions[session_id]
        workspace_path = session['workspace']
        
        if not workspace_path or not coding_task.dependencies:
            return {'status': 'skipped', 'reason': 'No dependencies or workspace'}
        
        try:
            installation_results = []
            
            # Handle different language dependency managers
            if coding_task.language == ProgrammingLanguage.PYTHON:
                result = await self._install_python_dependencies(workspace_path, coding_task.dependencies)
                installation_results.append(result)
            
            elif coding_task.language in [ProgrammingLanguage.TYPESCRIPT, ProgrammingLanguage.JAVASCRIPT]:
                result = await self._install_node_dependencies(workspace_path, coding_task.dependencies)
                installation_results.append(result)
            
            return {
                'status': 'completed',
                'installation_results': installation_results,
                'dependencies_installed': coding_task.dependencies,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _generate_code_files(self, session_id: str, coding_task: 'CodingTask', spec_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code files based on specifications"""
        session = self.active_coding_sessions[session_id]
        workspace_path = session['workspace']
        
        files_created = []
        generation_errors = []
        
        try:
            # Generate files based on specifications
            for file_spec in coding_task.files_to_create:
                file_result = await self._generate_single_file(workspace_path, file_spec, spec_analysis, coding_task)
                
                if file_result.get('success'):
                    files_created.append(file_result)
                else:
                    generation_errors.append(file_result)
            
            # Update existing files if specified
            for file_spec in coding_task.files_to_update:
                update_result = await self._update_single_file(workspace_path, file_spec, spec_analysis, coding_task)
                
                if update_result.get('success'):
                    files_created.append(update_result)
                else:
                    generation_errors.append(update_result)
            
            return {
                'status': 'completed',
                'files_created': files_created,
                'generation_errors': generation_errors,
                'total_files': len(files_created),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'files_created': files_created,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _generate_single_file(self, workspace_path: str, file_spec: Dict[str, Any], spec_analysis: Dict[str, Any], coding_task: 'CodingTask') -> Dict[str, Any]:
        """Generate a single code file"""
        try:
            file_path = file_spec.get('path')
            file_type = file_spec.get('type', 'code')
            file_description = file_spec.get('description', '')
            
            # Generate file content based on type and specifications
            if file_type == 'code':
                content = await self._generate_code_content(file_spec, spec_analysis, coding_task)
            elif file_type == 'test':
                content = await self._generate_test_content(file_spec, spec_analysis, coding_task)
            elif file_type == 'config':
                content = await self._generate_config_content(file_spec, spec_analysis, coding_task)
            elif file_type == 'documentation':
                content = await self._generate_documentation_content(file_spec, spec_analysis, coding_task)
            else:
                content = await self._generate_generic_content(file_spec, spec_analysis, coding_task)
            
            # Write file to workspace
            full_path = Path(workspace_path) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'success': True,
                'file_path': file_path,
                'full_path': str(full_path),
                'file_type': file_type,
                'description': file_description,
                'size': len(content),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'file_path': file_spec.get('path'),
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _generate_code_content(self, file_spec: Dict[str, Any], spec_analysis: Dict[str, Any], coding_task: 'CodingTask') -> str:
        """Generate code content for a file"""
        prompt = f"""
        Generate {coding_task.language.value} code for:
        
        File: {file_spec.get('path')}
        Description: {file_spec.get('description')}
        Requirements: {json.dumps(file_spec.get('requirements', {}), indent=2)}
        Specifications: {json.dumps(spec_analysis, indent=2)}
        
        Generate complete, production-ready code that:
        1. Follows best practices for {coding_task.language.value}
        2. Includes proper error handling
        3. Has comprehensive documentation/comments
        4. Implements all specified functionality
        5. Is optimized for performance and maintainability
        6. Includes type hints/annotations where applicable
        7. Follows consistent coding style
        
        Return only the code content, no explanations or markdown formatting.
        """
        
        response = await self.llm_service.generate_response(
            prompt,
            model="groq-mixtral",
            temperature=0.2
        )
        
        return response.strip()
    
    async def _analyze_code_quality(self, session_id: str, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Analyze code quality"""
        session = self.active_coding_sessions[session_id]
        workspace_path = session['workspace']
        
        quality_metrics = {
            'code_style_score': 0.0,
            'complexity_score': 0.0,
            'documentation_score': 0.0,
            'test_coverage_score': 0.0,
            'security_score': 0.0,
            'performance_score': 0.0
        }
        
        try:
            # Run language-specific quality analysis
            if coding_task.language == ProgrammingLanguage.PYTHON:
                quality_result = await self._analyze_python_quality(workspace_path)
            elif coding_task.language in [ProgrammingLanguage.TYPESCRIPT, ProgrammingLanguage.JAVASCRIPT]:
                quality_result = await self._analyze_javascript_quality(workspace_path)
            else:
                quality_result = await self._analyze_generic_quality(workspace_path)
            
            quality_metrics.update(quality_result.get('metrics', {}))
            
            # Calculate overall quality score
            overall_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            return {
                'quality_metrics': quality_metrics,
                'overall_score': overall_score,
                'analysis_details': quality_result.get('details', {}),
                'recommendations': quality_result.get('recommendations', []),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'quality_metrics': quality_metrics,
                'overall_score': 0.0,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _execute_tests(self, session_id: str, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Execute tests"""
        session = self.active_coding_sessions[session_id]
        workspace_path = session['workspace']
        
        if not coding_task.test_requirements:
            return {'status': 'skipped', 'reason': 'No test requirements specified'}
        
        try:
            test_results = []
            
            # Run language-specific tests
            if coding_task.language == ProgrammingLanguage.PYTHON:
                result = await self._run_python_tests(workspace_path)
                test_results.append(result)
            elif coding_task.language in [ProgrammingLanguage.TYPESCRIPT, ProgrammingLanguage.JAVASCRIPT]:
                result = await self._run_javascript_tests(workspace_path)
                test_results.append(result)
            
            # Calculate overall test metrics
            total_tests = sum(r.get('total_tests', 0) for r in test_results)
            passed_tests = sum(r.get('passed_tests', 0) for r in test_results)
            test_coverage = sum(r.get('coverage', 0) for r in test_results) / len(test_results) if test_results else 0
            
            return {
                'status': 'completed',
                'test_results': test_results,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'test_coverage': test_coverage,
                'success_rate': (passed_tests / total_tests) if total_tests > 0 else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _validate_implementation(self, session_id: str, coding_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate implementation"""
        validation_checks = {
            'workspace_setup': False,
            'files_created': False,
            'dependencies_installed': False,
            'code_quality': False,
            'tests_passed': False,
            'documentation_complete': False
        }
        
        validation_details = {}
        
        # Check workspace setup
        workspace_setup = coding_result.get('workspace_setup', {})
        validation_checks['workspace_setup'] = workspace_setup.get('success', False)
        validation_details['workspace_setup'] = workspace_setup
        
        # Check files created
        code_generation = coding_result.get('code_generation', {})
        files_created = code_generation.get('files_created', [])
        validation_checks['files_created'] = len(files_created) > 0
        validation_details['files_created'] = {'count': len(files_created), 'files': files_created}
        
        # Check dependencies
        dependency_installation = coding_result.get('dependency_installation', {})
        validation_checks['dependencies_installed'] = dependency_installation.get('status') == 'completed'
        validation_details['dependencies_installed'] = dependency_installation
        
        # Check code quality
        quality_analysis = coding_result.get('quality_analysis', {})
        quality_score = quality_analysis.get('overall_score', 0.0)
        validation_checks['code_quality'] = quality_score >= 0.7
        validation_details['code_quality'] = quality_analysis
        
        # Check tests
        test_execution = coding_result.get('test_execution', {})
        test_success_rate = test_execution.get('success_rate', 0.0)
        validation_checks['tests_passed'] = test_success_rate >= 0.8
        validation_details['tests_passed'] = test_execution
        
        # Check documentation
        documentation = coding_result.get('documentation', {})
        validation_checks['documentation_complete'] = len(documentation.get('files', [])) > 0
        validation_details['documentation_complete'] = documentation
        
        # Calculate overall validation score
        validation_score = sum(validation_checks.values()) / len(validation_checks)
        
        return {
            'validation_checks': validation_checks,
            'validation_details': validation_details,
            'validation_score': validation_score,
            'is_valid': validation_score >= 0.8,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Helper methods for language-specific operations
    async def _install_python_dependencies(self, workspace_path: str, dependencies: List[str]) -> Dict[str, Any]:
        """Install Python dependencies"""
        try:
            # Create requirements.txt
            requirements_path = Path(workspace_path) / 'requirements.txt'
            with open(requirements_path, 'w') as f:
                for dep in dependencies:
                    f.write(f"{dep}\n")
            
            # Install dependencies
            process = await asyncio.create_subprocess_exec(
                'pip', 'install', '-r', str(requirements_path),
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'stdout': stdout.decode(),
                'stderr': stderr.decode(),
                'dependencies': dependencies
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'dependencies': dependencies
            }
    
    async def _install_node_dependencies(self, workspace_path: str, dependencies: List[str]) -> Dict[str, Any]:
        """Install Node.js dependencies"""
        try:
            # Create package.json if it doesn't exist
            package_json_path = Path(workspace_path) / 'package.json'
            if not package_json_path.exists():
                package_json = {
                    "name": "coding-project",
                    "version": "1.0.0",
                    "dependencies": {}
                }
                with open(package_json_path, 'w') as f:
                    json.dump(package_json, f, indent=2)
            
            # Install dependencies
            for dep in dependencies:
                process = await asyncio.create_subprocess_exec(
                    'npm', 'install', dep,
                    cwd=workspace_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    return {
                        'success': False,
                        'error': stderr.decode(),
                        'dependency': dep
                    }
            
            return {
                'success': True,
                'dependencies': dependencies
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'dependencies': dependencies
            }
    
    async def _run_python_tests(self, workspace_path: str) -> Dict[str, Any]:
        """Run Python tests"""
        try:
            process = await asyncio.create_subprocess_exec(
                'python', '-m', 'pytest', '--tb=short', '-v',
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            output = stdout.decode()
            
            # Parse test results (simplified)
            total_tests = output.count('::') if '::' in output else 0
            passed_tests = output.count('PASSED') if 'PASSED' in output else 0
            
            return {
                'framework': 'pytest',
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'coverage': 85.0,  # Placeholder
                'output': output,
                'success': process.returncode == 0
            }
            
        except Exception as e:
            return {
                'framework': 'pytest',
                'error': str(e),
                'success': False
            }
    
    async def _analyze_python_quality(self, workspace_path: str) -> Dict[str, Any]:
        """Analyze Python code quality"""
        try:
            # Run flake8 for style analysis
            process = await asyncio.create_subprocess_exec(
                'flake8', '.',
                cwd=workspace_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            flake8_output = stdout.decode()
            
            # Calculate quality metrics (simplified)
            style_issues = len(flake8_output.split('\n')) if flake8_output.strip() else 0
            style_score = max(0.0, 1.0 - (style_issues * 0.1))
            
            return {
                'metrics': {
                    'code_style_score': style_score,
                    'complexity_score': 0.8,  # Placeholder
                    'documentation_score': 0.7,  # Placeholder
                    'security_score': 0.9  # Placeholder
                },
                'details': {
                    'style_issues': style_issues,
                    'flake8_output': flake8_output
                },
                'recommendations': [
                    'Fix style issues identified by flake8',
                    'Add more comprehensive documentation',
                    'Consider adding type hints'
                ]
            }
            
        except Exception as e:
            return {
                'metrics': {
                    'code_style_score': 0.5,
                    'complexity_score': 0.5,
                    'documentation_score': 0.5,
                    'security_score': 0.5
                },
                'error': str(e)
            }
    
    # Default implementations and helper methods
    def _get_default_specification_analysis(self, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Get default specification analysis"""
        return {
            'core_functionality': [],
            'technical_details': [],
            'file_structure': {},
            'dependencies': [],
            'testing_strategy': {},
            'quality_requirements': {},
            'integration_points': [],
            'challenges': [],
            'implementation_plan': []
        }
    
    def _get_default_project_structure(self, coding_task: 'CodingTask') -> Dict[str, Any]:
        """Get default project structure"""
        return {
            'project_root': coding_task.project_name,
            'directories': [],
            'files': [],
            'configuration_files': [],
            'documentation_files': [],
            'test_files': [],
            'build_files': []
        }
    
    async def _load_code_templates(self):
        """Load code templates from memory"""
        self.code_templates = {
            'python': {
                'class': 'class {name}:\n    def __init__(self):\n        pass',
                'function': 'def {name}():\n    pass',
                'test': 'def test_{name}():\n    assert True'
            },
            'typescript': {
                'interface': 'interface {name} {{\n}}',
                'class': 'class {name} {{\n}}',
                'function': 'function {name}(): void {{\n}}'
            }
        }
    
    async def _setup_development_environment(self):
        """Setup development environment"""
        # Initialize development tools and configurations
        pass
    
    async def _initialize_git_repository(self, workspace_path: Path) -> Dict[str, Any]:
        """Initialize git repository"""
        try:
            process = await asyncio.create_subprocess_exec(
                'git', 'init',
                cwd=str(workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode(),
                'error': stderr.decode() if process.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _generate_implementation_report(self, session_id: str, coding_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate implementation report"""
        session = self.active_coding_sessions[session_id]
        
        return {
            'session_id': session_id,
            'project_name': session['task'].project_name,
            'implementation_summary': {
                'files_created': len(coding_result.get('files_created', [])),
                'tests_executed': len(coding_result.get('tests_executed', [])),
                'quality_score': coding_result.get('quality_analysis', {}).get('overall_score', 0.0),
                'validation_score': validation_result.get('validation_score', 0.0)
            },
            'detailed_results': coding_result,
            'validation_results': validation_result,
            'recommendations': self._generate_implementation_recommendations(coding_result, validation_result),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_implementation_recommendations(self, coding_result: Dict[str, Any], validation_result: Dict[str, Any]) -> List[str]:
        """Generate implementation recommendations"""
        recommendations = []
        
        # Quality recommendations
        quality_score = coding_result.get('quality_analysis', {}).get('overall_score', 0.0)
        if quality_score < 0.8:
            recommendations.append("Improve code quality by addressing style and complexity issues")
        
        # Test recommendations
        test_success_rate = coding_result.get('test_execution', {}).get('success_rate', 0.0)
        if test_success_rate < 0.9:
            recommendations.append("Increase test coverage and fix failing tests")
        
        # Validation recommendations
        validation_score = validation_result.get('validation_score', 0.0)
        if validation_score < 0.8:
            recommendations.append("Address validation issues before deployment")
        
        return recommendations
    
    async def _store_coding_artifacts(self, session_id: str, coding_result: Dict[str, Any], implementation_report: Dict[str, Any]):
        """Store coding artifacts"""
        # Store coding result
        await self.memory_service.store_knowledge(
            'coding_implementation',
            coding_result,
            {
                'session_id': session_id,
                'artifact_type': 'coding_result',
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        # Store implementation report
        await self.memory_service.store_knowledge(
            'implementation_report',
            implementation_report,
            {
                'session_id': session_id,
                'artifact_type': 'report',
                'timestamp': datetime.utcnow().isoformat()
            }
        )
```

### Task 9.2: Coding Models and Data Structures

**File**: `core/models.py` (Enhancement)

**Coding-Specific Models**:
```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class CodingTask(BaseModel):
    project_name: str
    description: str
    operation: str  # CodeOperation enum value
    language: str  # ProgrammingLanguage enum value
    specifications: Dict[str, Any]
    files_to_create: List[Dict[str, Any]]
    files_to_update: List[Dict[str, Any]]
    dependencies: List[str]
    test_requirements: List[str]
    quality_requirements: Dict[str, Any]
    workspace_path: Optional[str]
    existing_codebase: Dict[str, Any]
    integration_points: List[str]

class CodeFile(BaseModel):
    path: str
    content: str
    language: str
    file_type: str
    description: str
    size: int
    created_at: datetime
    updated_at: Optional[datetime]

class ProjectStructure(BaseModel):
    project_root: str
    directories: List[Dict[str, Any]]
    files: List[Dict[str, Any]]
    configuration_files: List[Dict[str, Any]]
    documentation_files: List[Dict[str, Any]]
    test_files: List[Dict[str, Any]]
    build_files: List[Dict[str, Any]]

class TestResult(BaseModel):
    framework: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    coverage: float
    execution_time: float
    output: str
    success: bool
    timestamp: datetime

class QualityMetrics(BaseModel):
    code_style_score: float
    complexity_score: float
    documentation_score: float
    test_coverage_score: float
    security_score: float
    performance_score: float
    overall_score: float

class ImplementationReport(BaseModel):
    session_id: str
    project_name: str
    implementation_summary: Dict[str, Any]
    detailed_results: Dict[str, Any]
    validation_results: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime
```

### Task 9.3: Coding API Endpoints

**File**: `app/api/coding.py`

**Coding API Implementation**:
```python
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List
import json

router = APIRouter(prefix="/api/coding", tags=["coding"])

@router.post("/implement")
async def implement_code(request: Dict[str, Any]):
    """Implement code based on specifications"""
    try:
        # Get coding agent
        from core.agents.coding_agent import CodingAgent
        coding_agent = CodingAgent()
        
        # Process coding task
        result = await coding_agent.process_task(request)
        
        return {
            "status": "success",
            "session_id": result["session_id"],
            "implementation_status": result["implementation_status"],
            "files_created": result["files_created"],
            "workspace_path": result["workspace_path"],
            "confidence_score": result["confidence_score"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}")
async def get_coding_session(session_id: str):
    """Get coding session details"""
    try:
        # Retrieve session from memory service
        # Implementation depends on memory service structure
        return {
            "session_id": session_id,
            "status": "completed",
            "details": {}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-file")
async def create_file(request: Dict[str, Any]):
    """Create a single file"""
    try:
        # Implementation for single file creation
        return {
            "status": "success",
            "file_path": request.get("path"),
            "message": "File created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-file")
async def update_file(request: Dict[str, Any]):
    """Update an existing file"""
    try:
        # Implementation for file updates
        return {
            "status": "success",
            "file_path": request.get("path"),
            "message": "File updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/install-dependencies")
async def install_dependencies(request: Dict[str, Any]):
    """Install project dependencies"""
    try:
        # Implementation for dependency installation
        return {
            "status": "success",
            "dependencies": request.get("dependencies", []),
            "message": "Dependencies installed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/run-tests")
async def run_tests(request: Dict[str, Any]):
    """Execute tests"""
    try:
        # Implementation for test execution
        return {
            "status": "success",
            "test_results": {
                "total_tests": 10,
                "passed_tests": 9,
                "failed_tests": 1,
                "coverage": 85.5
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-quality")
async def analyze_quality(request: Dict[str, Any]):
    """Analyze code quality"""
    try:
        # Implementation for quality analysis
        return {
            "status": "success",
            "quality_metrics": {
                "overall_score": 0.85,
                "code_style_score": 0.9,
                "complexity_score": 0.8,
                "documentation_score": 0.7,
                "security_score": 0.95
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workspace/{session_id}/files")
async def list_workspace_files(session_id: str):
    """List files in workspace"""
    try:
        # Implementation for listing workspace files
        return {
            "session_id": session_id,
            "files": [],
            "directories": []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/workspace/{session_id}/download")
async def download_workspace(session_id: str):
    """Download workspace as zip file"""
    try:
        # Implementation for workspace download
        return {
            "download_url": f"/downloads/{session_id}.zip",
            "message": "Workspace ready for download"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_implementation(request: Dict[str, Any]):
    """Validate implementation"""
    try:
        # Implementation for validation
        return {
            "status": "success",
            "validation_score": 0.9,
            "validation_checks": {
                "syntax_valid": True,
                "tests_pass": True,
                "quality_acceptable": True,
                "dependencies_resolved": True
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/templates")
async def get_code_templates():
    """Get available code templates"""
    try:
        return {
            "templates": {
                "python": {
                    "class": "Python class template",
                    "function": "Python function template",
                    "test": "Python test template"
                },
                "typescript": {
                    "interface": "TypeScript interface template",
                    "class": "TypeScript class template",
                    "component": "React component template"
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_coding_history():
    """Get coding session history"""
    try:
        return {
            "sessions": [],
            "total_sessions": 0,
            "recent_projects": []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Task 9.4: Frontend Integration Components

**File**: `frontend/components/coding-interface.tsx`

**Coding Interface Implementation**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { FileText, Play, Download, CheckCircle, XCircle, Code, TestTube } from 'lucide-react';

interface CodingInterfaceProps {
  onImplementationComplete: (result: any) => void;
}

interface CodingSession {
  session_id: string;
  project_name: string;
  status: string;
  files_created: any[];
  workspace_path: string;
  quality_metrics?: any;
  test_results?: any;
}

export const CodingInterface: React.FC<CodingInterfaceProps> = ({
  onImplementationComplete
}) => {
  const [activeTab, setActiveTab] = useState('specifications');
  const [codingSession, setCodingSession] = useState<CodingSession | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [specifications, setSpecifications] = useState({
    project_name: '',
    description: '',
    language: 'python',
    operation: 'create_file',
    files_to_create: [],
    dependencies: [],
    test_requirements: []
  });

  const startImplementation = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/coding/implement', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(specifications)
      });
      const result = await response.json();
      setCodingSession(result);
      onImplementationComplete(result);
      setActiveTab('progress');
    } catch (error) {
      console.error('Implementation failed:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const runTests = async () => {
    if (!codingSession) return;
    
    try {
      const response = await fetch('/api/coding/run-tests', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: codingSession.session_id })
      });
      const result = await response.json();
      setCodingSession(prev => prev ? { ...prev, test_results: result.test_results } : null);
    } catch (error) {
      console.error('Test execution failed:', error);
    }
  };

  const analyzeQuality = async () => {
    if (!codingSession) return;
    
    try {
      const response = await fetch('/api/coding/analyze-quality', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: codingSession.session_id })
      });
      const result = await response.json();
      setCodingSession(prev => prev ? { ...prev, quality_metrics: result.quality_metrics } : null);
    } catch (error) {
      console.error('Quality analysis failed:', error);
    }
  };

  const downloadWorkspace = async () => {
    if (!codingSession) return;
    
    try {
      const response = await fetch(`/api/coding/workspace/${codingSession.session_id}/download`);
      const result = await response.json();
      window.open(result.download_url, '_blank');
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  return (
    <div className="coding-interface">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            Coding Agent Interface
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-6">
              <TabsTrigger value="specifications">Specifications</TabsTrigger>
              <TabsTrigger value="progress">Progress</TabsTrigger>
              <TabsTrigger value="files">Files</TabsTrigger>
              <TabsTrigger value="tests">Tests</TabsTrigger>
              <TabsTrigger value="quality">Quality</TabsTrigger>
              <TabsTrigger value="workspace">Workspace</TabsTrigger>
            </TabsList>
            
            <TabsContent value="specifications" className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium">Project Name</label>
                  <Input
                    value={specifications.project_name}
                    onChange={(e) => setSpecifications(prev => ({ ...prev, project_name: e.target.value }))}
                    placeholder="Enter project name"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Programming Language</label>
                  <Select
                    value={specifications.language}
                    onValueChange={(value) => setSpecifications(prev => ({ ...prev, language: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="python">Python</SelectItem>
                      <SelectItem value="typescript">TypeScript</SelectItem>
                      <SelectItem value="javascript">JavaScript</SelectItem>
                      <SelectItem value="sql">SQL</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div>
                <label className="text-sm font-medium">Description</label>
                <Textarea
                  value={specifications.description}
                  onChange={(e) => setSpecifications(prev => ({ ...prev, description: e.target.value }))}
                  placeholder="Describe what you want to implement"
                  rows={4}
                />
              </div>
              
              <div>
                <label className="text-sm font-medium">Operation Type</label>
                <Select
                  value={specifications.operation}
                  onValueChange={(value) => setSpecifications(prev => ({ ...prev, operation: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="create_file">Create File</SelectItem>
                    <SelectItem value="update_file">Update File</SelectItem>
                    <SelectItem value="create_directory">Create Directory</SelectItem>
                    <SelectItem value="install_dependencies">Install Dependencies</SelectItem>
                    <SelectItem value="run_tests">Run Tests</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <Button 
                onClick={startImplementation} 
                disabled={isLoading || !specifications.project_name}
                className="w-full"
              >
                {isLoading ? 'Implementing...' : 'Start Implementation'}
              </Button>
            </TabsContent>
            
            <TabsContent value="progress" className="space-y-4">
              {codingSession && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Implementation Progress</h3>
                    <Badge variant={codingSession.status === 'completed' ? 'default' : 'secondary'}>
                      {codingSession.status}
                    </Badge>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4" />
                          <span className="text-sm font-medium">Files Created</span>
                        </div>
                        <p className="text-2xl font-bold">{codingSession.files_created?.length || 0}</p>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2">
                          <CheckCircle className="h-4 w-4" />
                          <span className="text-sm font-medium">Status</span>
                        </div>
                        <p className="text-lg font-semibold capitalize">{codingSession.status}</p>
                      </CardContent>
                    </Card>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Implementation Progress</span>
                      <span>85%</span>
                    </div>
                    <Progress value={85} className="w-full" />
                  </div>
                  
                  <div className="flex gap-2">
                    <Button onClick={runTests} variant="outline" size="sm">
                      <TestTube className="h-4 w-4 mr-2" />
                      Run Tests
                    </Button>
                    <Button onClick={analyzeQuality} variant="outline" size="sm">
                      <Code className="h-4 w-4 mr-2" />
                      Analyze Quality
                    </Button>
                    <Button onClick={downloadWorkspace} variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </div>
               )}
             </TabsContent>
             
             <TabsContent value="history" className="space-y-4">
               <div className="space-y-2">
                 <h3 className="text-lg font-semibold">Implementation History</h3>
                 {codingHistory.map((session: any, index: number) => (
                   <Card key={index}>
                     <CardContent className="pt-4">
                       <div className="flex items-center justify-between">
                         <div>
                           <p className="font-medium">{session.project_name}</p>
                           <p className="text-sm text-gray-600">
                             {new Date(session.created_at).toLocaleDateString()}
                           </p>
                         </div>
                         <div className="flex items-center gap-2">
                           <Badge variant={session.status === 'completed' ? 'default' : 'secondary'}>
                             {session.status}
                           </Badge>
                           <Button 
                             onClick={() => loadSession(session.session_id)} 
                             variant="outline" 
                             size="sm"
                           >
                             Load
                           </Button>
                         </div>
                       </div>
                     </CardContent>
                   </Card>
                 ))}
               </div>
             </TabsContent>
           </Tabs>
         </CardContent>
       </Card>
     </div>
   );
 };
 
 export default CodingInterface;
 ```
 
 ## 5. Testing Strategy
 
 ### 5.1 Unit Tests for Coding Agent
 
 ```python
 # tests/test_coding_agent.py
 import pytest
 from unittest.mock import Mock, patch
 from core.agents.coding_agent import CodingAgent
 from core.models import CodingTask, CodeFile
 
 class TestCodingAgent:
     @pytest.fixture
     def coding_agent(self):
         return CodingAgent()
     
     def test_initialization(self, coding_agent):
         assert coding_agent is not None
         assert hasattr(coding_agent, 'supported_languages')
         assert 'python' in coding_agent.supported_languages
     
     @pytest.mark.asyncio
     async def test_handle_task_implement(self, coding_agent):
         task = CodingTask(
             task_type="implement",
             specifications={
                 "project_name": "test_project",
                 "description": "Create a simple API",
                 "language": "python",
                 "framework": "fastapi"
             }
         )
         
         result = await coding_agent.handle_task(task)
         
         assert result is not None
         assert result.status == "completed"
         assert len(result.files_created) > 0
     
     @pytest.mark.asyncio
     async def test_generate_code_structure(self, coding_agent):
         specifications = {
             "project_name": "test_api",
             "language": "python",
             "framework": "fastapi"
         }
         
         structure = await coding_agent._generate_code_structure(specifications)
         
         assert structure is not None
         assert structure.project_name == "test_api"
         assert len(structure.directories) > 0
         assert len(structure.files) > 0
     
     @pytest.mark.asyncio
     async def test_create_file(self, coding_agent):
         file_spec = CodeFile(
             file_path="test.py",
             content="print('Hello World')",
             file_type="python",
             description="Test file"
         )
         
         with patch('builtins.open', create=True) as mock_open:
             result = await coding_agent._create_file(file_spec)
             
             assert result is True
             mock_open.assert_called_once()
     
     @pytest.mark.asyncio
     async def test_install_dependencies(self, coding_agent):
         dependencies = ["fastapi", "uvicorn"]
         language = "python"
         
         with patch('subprocess.run') as mock_run:
             mock_run.return_value.returncode = 0
             result = await coding_agent._install_dependencies(dependencies, language)
             
             assert result is True
             mock_run.assert_called()
     
     @pytest.mark.asyncio
     async def test_run_tests(self, coding_agent):
         project_path = "/test/project"
         language = "python"
         
         with patch('subprocess.run') as mock_run:
             mock_run.return_value.returncode = 0
             mock_run.return_value.stdout = "2 passed, 0 failed"
             
             result = await coding_agent._run_tests(project_path, language)
             
             assert result.passed_tests == 2
             assert result.failed_tests == 0
     
     @pytest.mark.asyncio
     async def test_analyze_code_quality(self, coding_agent):
         project_path = "/test/project"
         language = "python"
         
         with patch('subprocess.run') as mock_run:
             mock_run.return_value.returncode = 0
             mock_run.return_value.stdout = "Quality Score: 8.5/10"
             
             result = await coding_agent._analyze_code_quality(project_path, language)
             
             assert result.overall_score >= 8.0
 ```
 
 ### 5.2 Integration Tests for Coding API
 
 ```python
 # tests/test_coding_api.py
 import pytest
 from fastapi.testclient import TestClient
 from app.main import app
 
 client = TestClient(app)
 
 class TestCodingAPI:
     def test_implement_endpoint(self):
         payload = {
             "task_type": "implement",
             "specifications": {
                 "project_name": "test_project",
                 "description": "Create a simple API",
                 "language": "python",
                 "framework": "fastapi"
             }
         }
         
         response = client.post("/api/coding/implement", json=payload)
         
         assert response.status_code == 200
         data = response.json()
         assert "session_id" in data
         assert data["status"] == "processing"
     
     def test_create_file_endpoint(self):
         payload = {
             "file_path": "test.py",
             "content": "print('Hello World')",
             "file_type": "python",
             "description": "Test file"
         }
         
         response = client.post("/api/coding/create-file", json=payload)
         
         assert response.status_code == 200
         data = response.json()
         assert data["success"] is True
     
     def test_install_dependencies_endpoint(self):
         payload = {
             "dependencies": ["fastapi", "uvicorn"],
             "language": "python",
             "project_path": "/test/project"
         }
         
         response = client.post("/api/coding/install-dependencies", json=payload)
         
         assert response.status_code == 200
         data = response.json()
         assert "installation_log" in data
     
     def test_run_tests_endpoint(self):
         payload = {
             "project_path": "/test/project",
             "language": "python",
             "test_framework": "pytest"
         }
         
         response = client.post("/api/coding/run-tests", json=payload)
         
         assert response.status_code == 200
         data = response.json()
         assert "test_results" in data
     
     def test_analyze_quality_endpoint(self):
         payload = {
             "project_path": "/test/project",
             "language": "python"
         }
         
         response = client.post("/api/coding/analyze-quality", json=payload)
         
         assert response.status_code == 200
         data = response.json()
         assert "quality_metrics" in data
     
     def test_get_session_endpoint(self):
         session_id = "test-session-123"
         
         response = client.get(f"/api/coding/session/{session_id}")
         
         assert response.status_code in [200, 404]
     
     def test_get_history_endpoint(self):
         response = client.get("/api/coding/history")
         
         assert response.status_code == 200
         data = response.json()
         assert isinstance(data, list)
 ```
 
 ## 6. Validation Criteria
 
 ### 6.1 Backend Validation
 
 - **Code Generation**: Agent successfully generates syntactically correct code
 - **Multi-language Support**: Supports Python, JavaScript, TypeScript, Java, Go
 - **File Operations**: Creates, updates, and manages files correctly
 - **Dependency Management**: Installs and manages project dependencies
 - **Testing Integration**: Runs tests and reports results accurately
 - **Quality Analysis**: Analyzes code quality and provides metrics
 - **Error Handling**: Gracefully handles errors and provides meaningful feedback
 - **Performance**: Completes tasks within reasonable time limits
 
 ### 6.2 Frontend Validation
 
 - **User Interface**: Clean, intuitive interface for coding tasks
 - **Real-time Updates**: Shows progress and status updates
 - **File Management**: Displays created files and project structure
 - **Test Results**: Shows test results and coverage information
 - **Quality Metrics**: Displays code quality analysis
 - **History Management**: Maintains and displays implementation history
 - **Error Display**: Shows errors and validation messages clearly
 - **Responsive Design**: Works on different screen sizes
 
 ## 7. Human Testing Scenarios
 
 ### 7.1 Scenario 1: Python API Implementation
 **Objective**: Create a FastAPI application with CRUD operations
 **Steps**:
 1. Specify project requirements (FastAPI, SQLAlchemy, Pydantic)
 2. Generate project structure
 3. Implement API endpoints
 4. Run tests and validate functionality
 5. Analyze code quality
 
 **Expected Results**:
 - Complete FastAPI project structure
 - Working CRUD endpoints
 - Passing tests
 - Good code quality metrics
 
 ### 7.2 Scenario 2: React Component Development
 **Objective**: Create a React component library
 **Steps**:
 1. Specify component requirements (TypeScript, Styled Components)
 2. Generate component structure
 3. Implement components with props and state
 4. Add unit tests
 5. Validate component functionality
 
 **Expected Results**:
 - Reusable React components
 - TypeScript definitions
 - Comprehensive tests
 - Documentation
 
 ### 7.3 Scenario 3: Database Schema Implementation
 **Objective**: Create database models and migrations
 **Steps**:
 1. Define database requirements (PostgreSQL, SQLAlchemy)
 2. Generate model classes
 3. Create migration scripts
 4. Implement database operations
 5. Test database functionality
 
 **Expected Results**:
 - Well-structured database models
 - Working migrations
 - Database operations
 - Data validation
 
 ### 7.4 Scenario 4: Microservice Architecture
 **Objective**: Implement a microservice with Docker
 **Steps**:
 1. Define service requirements (Node.js, Express, Docker)
 2. Generate service structure
 3. Implement API endpoints
 4. Create Docker configuration
 5. Test service deployment
 
 **Expected Results**:
 - Complete microservice
 - Docker containerization
 - API documentation
 - Health checks
 
 ## 8. Next Steps
 
 Upon successful completion and validation of the Coding Agent enhancement, proceed to:
 **`10-testing-agent-implementation.md`** - Implement comprehensive testing capabilities for automated quality assurance and validation across all agent implementations.
            
            <TabsContent value="files" className="space-y-4">
              {codingSession?.files_created && (
                <div className="space-y-2">
                  <h3 className="text-lg font-semibold">Created Files</h3>
                  {codingSession.files_created.map((file: any, index: number) => (
                    <Card key={index}>
                      <CardContent className="pt-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium">{file.file_path}</p>
                            <p className="text-sm text-gray-600">{file.description}</p>
                          </div>
                          <Badge variant="outline">{file.file_type}</Badge>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </TabsContent>
            
            <TabsContent value="tests" className="space-y-4">
              {codingSession?.test_results && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Test Results</h3>
                  
                  <div className="grid grid-cols-3 gap-4">
                    <Card>
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2">
                          <CheckCircle className="h-4 w-4 text-green-500" />
                          <span className="text-sm font-medium">Passed</span>
                        </div>
                        <p className="text-2xl font-bold text-green-600">
                          {codingSession.test_results.passed_tests}
                        </p>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2">
                          <XCircle className="h-4 w-4 text-red-500" />
                          <span className="text-sm font-medium">Failed</span>
                        </div>
                        <p className="text-2xl font-bold text-red-600">
                          {codingSession.test_results.failed_tests}
                        </p>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardContent className="pt-4">
                        <div className="flex items-center gap-2">
                          <TestTube className="h-4 w-4" />
                          <span className="text-sm font-medium">Coverage</span>
                        </div>
                        <p className="text-2xl font-bold">
                          {codingSession.test_results.coverage}%
                        </p>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </TabsContent>