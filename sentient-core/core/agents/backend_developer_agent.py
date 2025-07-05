"""
The Back-End Developer Agent
Handles all tasks related to server-side logic, databases, and APIs.
"""

from core.models import AppState, EnhancedTask, AgentType, LogEntry
from core.services.llm_service import EnhancedLLMService
from core.agents.base_agent import BaseAgent, AgentCapability, ActivityType
from typing import Dict, Any, Optional, List
import json
import asyncio

class BackendDeveloperAgent(BaseAgent):
    """
    The Back-End Developer Agent is responsible for:
    - Designing and implementing the server-side architecture.
    - Creating and managing databases.
    - Building robust and scalable APIs.
    - Implementing business logic and data processing.
    """

    def __init__(self, llm_service: EnhancedLLMService, agent_id: str = "backend_developer_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Backend Developer Agent",
            capabilities=[AgentCapability.BACKEND_DEVELOPMENT, AgentCapability.DATABASE_DESIGN, AgentCapability.API_DEVELOPMENT],
            description="Specialized agent for backend development, API design, and database management"
        )
        self.llm_service = llm_service
        self.supported_frameworks = ['fastapi', 'django', 'flask', 'express', 'spring', 'rails']
        self.supported_databases = ['postgresql', 'mysql', 'mongodb', 'redis', 'sqlite']
        self.api_patterns = ['rest', 'graphql', 'grpc', 'websocket']
        self.architecture_patterns = ['microservices', 'monolith', 'serverless', 'event-driven']

    def can_handle_task(self, task: EnhancedTask) -> bool:
        """
        Determines if this agent can handle the given task.
        """
        backend_keywords = [
            'backend', 'back-end', 'server', 'api', 'database', 'db', 'sql', 'nosql',
            'endpoint', 'service', 'microservice', 'authentication', 'authorization',
            'business logic', 'data processing', 'orm', 'migration', 'schema',
            'rest', 'graphql', 'websocket', 'middleware', 'routing'
        ]
        task_description = task.description.lower()
        return any(keyword in task_description for keyword in backend_keywords)
    
    async def process_task(self, task: EnhancedTask, state: Optional[AppState] = None) -> Dict:
        """
        Processes a backend development task.
        """
        try:
            self.log_activity(ActivityType.TASK_STARTED, f"Processing backend task: {task.description}")
            
            # Determine the type of backend task
            task_type = self._determine_task_type(task.description)
            
            result = {}
            
            if task_type == 'api':
                result = await self._handle_api_task(task)
            elif task_type == 'database':
                result = await self._handle_database_task(task)
            elif task_type == 'architecture':
                result = await self._handle_architecture_task(task)
            elif task_type == 'business_logic':
                result = await self._handle_business_logic_task(task)
            elif task_type == 'authentication':
                result = await self._handle_authentication_task(task)
            elif task_type == 'integration':
                result = await self._handle_integration_task(task)
            else:
                result = await self._handle_general_backend_task(task)
            
            self.log_activity(ActivityType.TASK_COMPLETED, f"Backend task completed: {task.description}")
            
            return {
                "status": "completed",
                "task_type": task_type,
                "result": result
            }
            
        except Exception as e:
            self.handle_error(e, f"Error processing backend task: {task.description}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _determine_task_type(self, description: str) -> str:
        """
        Determines the type of backend task based on description.
        """
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['api', 'endpoint', 'rest', 'graphql']):
            return 'api'
        elif any(word in description_lower for word in ['database', 'db', 'schema', 'migration', 'orm']):
            return 'database'
        elif any(word in description_lower for word in ['architecture', 'microservice', 'design pattern']):
            return 'architecture'
        elif any(word in description_lower for word in ['business logic', 'processing', 'algorithm']):
            return 'business_logic'
        elif any(word in description_lower for word in ['auth', 'login', 'security', 'jwt', 'oauth']):
            return 'authentication'
        elif any(word in description_lower for word in ['integration', 'third-party', 'webhook', 'external']):
            return 'integration'
        else:
            return 'general'
    
    async def _handle_api_task(self, task: EnhancedTask) -> Dict:
        """
        Handles API development tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Developing API endpoints")
        
        try:
            api_prompt = f"""
            Design and implement API endpoints for: {task.description}
            
            Include:
            1. RESTful API design
            2. Request/response schemas
            3. Error handling
            4. Input validation
            5. Documentation (OpenAPI/Swagger)
            6. Authentication/authorization
            7. Rate limiting considerations
            
            Provide complete implementation with best practices.
            """
            
            response = await self.llm_service.generate_response(
                prompt=api_prompt,
                model="compound-beta"
            )
            
            return {
                "api_implementation": response,
                "api_pattern": self._identify_api_pattern(task.description),
                "endpoints": self._extract_endpoints(response),
                "security_features": self._extract_security_features(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in API task: {e}")
            raise
    
    async def _handle_database_task(self, task: EnhancedTask) -> Dict:
        """
        Handles database design and implementation tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Designing database schema")
        
        try:
            db_prompt = f"""
            Design and implement database solution for: {task.description}
            
            Include:
            1. Database schema design
            2. Table relationships
            3. Indexes and constraints
            4. Migration scripts
            5. ORM models
            6. Query optimization
            7. Data validation
            
            Provide SQL scripts and ORM code.
            """
            
            response = await self.llm_service.generate_response(
                prompt=db_prompt,
                model="compound-beta"
            )
            
            return {
                "database_design": response,
                "database_type": self._identify_database_type(task.description),
                "tables": self._extract_tables(response),
                "relationships": self._extract_relationships(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in database task: {e}")
            raise
    
    async def _handle_architecture_task(self, task: EnhancedTask) -> Dict:
        """
        Handles system architecture design tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Designing system architecture")
        
        try:
            arch_prompt = f"""
            Design system architecture for: {task.description}
            
            Include:
            1. High-level architecture diagram
            2. Component interactions
            3. Data flow
            4. Scalability considerations
            5. Performance optimization
            6. Security architecture
            7. Deployment strategy
            
            Provide detailed architectural documentation.
            """
            
            response = await self.llm_service.generate_response(
                prompt=arch_prompt,
                model="compound-beta"
            )
            
            return {
                "architecture_design": response,
                "architecture_pattern": self._identify_architecture_pattern(task.description),
                "components": self._extract_components(response),
                "scalability_features": self._extract_scalability_features(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in architecture task: {e}")
            raise
    
    async def _handle_business_logic_task(self, task: EnhancedTask) -> Dict:
        """
        Handles business logic implementation tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Implementing business logic")
        
        try:
            logic_prompt = f"""
            Implement business logic for: {task.description}
            
            Include:
            1. Core business rules
            2. Data processing algorithms
            3. Validation logic
            4. Error handling
            5. Unit tests
            6. Performance considerations
            7. Documentation
            
            Provide clean, maintainable code with tests.
            """
            
            response = await self.llm_service.generate_response(
                prompt=logic_prompt,
                model="compound-beta"
            )
            
            return {
                "business_logic": response,
                "algorithms": self._extract_algorithms(response),
                "validation_rules": self._extract_validation_rules(response),
                "test_coverage": self._extract_test_coverage(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in business logic task: {e}")
            raise
    
    async def _handle_authentication_task(self, task: EnhancedTask) -> Dict:
        """
        Handles authentication and authorization tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Implementing authentication system")
        
        try:
            auth_prompt = f"""
            Implement authentication and authorization for: {task.description}
            
            Include:
            1. User authentication flow
            2. JWT token management
            3. Role-based access control
            4. Password security
            5. Session management
            6. OAuth integration
            7. Security best practices
            
            Provide secure implementation with proper error handling.
            """
            
            response = await self.llm_service.generate_response(
                prompt=auth_prompt,
                model="compound-beta"
            )
            
            return {
                "auth_implementation": response,
                "auth_methods": self._extract_auth_methods(response),
                "security_features": self._extract_security_features(response),
                "compliance": self._extract_compliance_features(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in authentication task: {e}")
            raise
    
    async def _handle_integration_task(self, task: EnhancedTask) -> Dict:
        """
        Handles third-party integration tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Implementing third-party integrations")
        
        try:
            integration_prompt = f"""
            Implement third-party integration for: {task.description}
            
            Include:
            1. API client implementation
            2. Error handling and retries
            3. Rate limiting
            4. Data transformation
            5. Webhook handling
            6. Configuration management
            7. Monitoring and logging
            
            Provide robust integration code with proper error handling.
            """
            
            response = await self.llm_service.generate_response(
                prompt=integration_prompt,
                model="compound-beta"
            )
            
            return {
                "integration_code": response,
                "integration_type": self._identify_integration_type(task.description),
                "error_handling": self._extract_error_handling(response),
                "monitoring": self._extract_monitoring_features(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in integration task: {e}")
            raise
    
    async def _handle_general_backend_task(self, task: EnhancedTask) -> Dict:
        """
        Handles general backend development tasks.
        """
        self.log_activity(ActivityType.PROCESSING, "Processing general backend task")
        
        try:
            general_prompt = f"""
            Implement backend solution for: {task.description}
            
            Provide:
            1. Complete implementation
            2. Best practices
            3. Performance optimization
            4. Security considerations
            5. Testing strategy
            6. Documentation
            7. Deployment guidelines
            
            Use modern backend technologies and patterns.
            """
            
            response = await self.llm_service.generate_response(
                prompt=general_prompt,
                model="compound-beta"
            )
            
            return {
                "implementation": response,
                "technologies_used": self._identify_technologies(response),
                "best_practices": self._extract_best_practices(response),
                "performance_features": self._extract_performance_features(response)
            }
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error in general backend task: {e}")
            raise
    
    def _identify_api_pattern(self, description: str) -> str:
        """
        Identifies the API pattern from task description.
        """
        description_lower = description.lower()
        
        for pattern in self.api_patterns:
            if pattern in description_lower:
                return pattern
        return 'rest'  # Default to REST
    
    def _extract_endpoints(self, code: str) -> List[str]:
        """
        Extracts API endpoints from implementation code.
        """
        endpoints = []
        import re
        
        # Extract route decorators and path definitions
        route_patterns = [
            r'@app\.route\([\'"]([^\'"]+)[\'"]',
            r'@router\.[a-z]+\([\'"]([^\'"]+)[\'"]',
            r'path\([\'"]([^\'"]+)[\'"]',
            r'Route\([\'"]([^\'"]+)[\'"]'
        ]
        
        for pattern in route_patterns:
            matches = re.findall(pattern, code)
            endpoints.extend(matches)
        
        return list(set(endpoints))
    
    def _extract_security_features(self, code: str) -> List[str]:
        """
        Extracts security features from code.
        """
        features = []
        code_lower = code.lower()
        
        security_keywords = {
            'jwt': 'JWT Authentication',
            'oauth': 'OAuth Integration',
            'cors': 'CORS Protection',
            'csrf': 'CSRF Protection',
            'rate_limit': 'Rate Limiting',
            'encryption': 'Data Encryption',
            'hash': 'Password Hashing',
            'validation': 'Input Validation'
        }
        
        for keyword, feature in security_keywords.items():
            if keyword in code_lower:
                features.append(feature)
        
        return features
    
    def _identify_database_type(self, description: str) -> str:
        """
        Identifies database type from task description.
        """
        description_lower = description.lower()
        
        for db_type in self.supported_databases:
            if db_type in description_lower:
                return db_type
        
        # Default based on common patterns
        if 'nosql' in description_lower or 'document' in description_lower:
            return 'mongodb'
        elif 'cache' in description_lower:
            return 'redis'
        else:
            return 'postgresql'  # Default to PostgreSQL
    
    def _extract_tables(self, code: str) -> List[str]:
        """
        Extracts table names from database code.
        """
        tables = []
        import re
        
        # Extract CREATE TABLE statements
        table_patterns = [
            r'CREATE TABLE\s+([\w_]+)',
            r'class\s+([A-Z][\w]+)\s*\(',
            r'Table\([\'"]([\w_]+)[\'"]'
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            tables.extend(matches)
        
        return list(set(tables))
    
    def _extract_relationships(self, code: str) -> List[str]:
        """
        Extracts database relationships from code.
        """
        relationships = []
        code_lower = code.lower()
        
        relationship_keywords = {
            'foreign key': 'Foreign Key',
            'one-to-many': 'One-to-Many',
            'many-to-many': 'Many-to-Many',
            'one-to-one': 'One-to-One',
            'join': 'Join Relationship'
        }
        
        for keyword, relationship in relationship_keywords.items():
            if keyword in code_lower:
                relationships.append(relationship)
        
        return relationships
    
    def _identify_architecture_pattern(self, description: str) -> str:
        """
        Identifies architecture pattern from description.
        """
        description_lower = description.lower()
        
        for pattern in self.architecture_patterns:
            if pattern in description_lower:
                return pattern
        return 'monolith'  # Default
    
    def _extract_components(self, code: str) -> List[str]:
        """
        Extracts system components from architecture code.
        """
        components = []
        code_lower = code.lower()
        
        component_keywords = [
            'service', 'controller', 'repository', 'middleware',
            'gateway', 'proxy', 'cache', 'queue', 'worker'
        ]
        
        for keyword in component_keywords:
            if keyword in code_lower:
                components.append(keyword.title())
        
        return list(set(components))
    
    def _extract_scalability_features(self, code: str) -> List[str]:
        """
        Extracts scalability features from code.
        """
        features = []
        code_lower = code.lower()
        
        scalability_keywords = {
            'load balancer': 'Load Balancing',
            'caching': 'Caching Strategy',
            'horizontal scaling': 'Horizontal Scaling',
            'database sharding': 'Database Sharding',
            'async': 'Asynchronous Processing',
            'queue': 'Message Queuing'
        }
        
        for keyword, feature in scalability_keywords.items():
            if keyword in code_lower:
                features.append(feature)
        
        return features
    
    def _extract_algorithms(self, code: str) -> List[str]:
        """
        Extracts algorithms from business logic code.
        """
        algorithms = []
        code_lower = code.lower()
        
        algorithm_keywords = [
            'sort', 'search', 'filter', 'aggregate', 'calculate',
            'validate', 'transform', 'process', 'optimize'
        ]
        
        for keyword in algorithm_keywords:
            if keyword in code_lower:
                algorithms.append(keyword.title())
        
        return list(set(algorithms))
    
    def _extract_validation_rules(self, code: str) -> List[str]:
        """
        Extracts validation rules from code.
        """
        rules = []
        code_lower = code.lower()
        
        validation_keywords = {
            'required': 'Required Fields',
            'email': 'Email Validation',
            'length': 'Length Validation',
            'pattern': 'Pattern Validation',
            'range': 'Range Validation',
            'unique': 'Uniqueness Validation'
        }
        
        for keyword, rule in validation_keywords.items():
            if keyword in code_lower:
                rules.append(rule)
        
        return rules
    
    def _extract_test_coverage(self, code: str) -> Dict:
        """
        Extracts test coverage information from code.
        """
        coverage = {
            "unit_tests": 'unittest' in code.lower() or 'pytest' in code.lower(),
            "integration_tests": 'integration' in code.lower(),
            "mocking": 'mock' in code.lower(),
            "fixtures": 'fixture' in code.lower()
        }
        
        return coverage
    
    def _extract_auth_methods(self, code: str) -> List[str]:
        """
        Extracts authentication methods from code.
        """
        methods = []
        code_lower = code.lower()
        
        auth_keywords = {
            'jwt': 'JWT Tokens',
            'oauth': 'OAuth',
            'session': 'Session-based',
            'basic auth': 'Basic Authentication',
            'api key': 'API Key',
            'bearer token': 'Bearer Token'
        }
        
        for keyword, method in auth_keywords.items():
            if keyword in code_lower:
                methods.append(method)
        
        return methods
    
    def _extract_compliance_features(self, code: str) -> List[str]:
        """
        Extracts compliance features from authentication code.
        """
        features = []
        code_lower = code.lower()
        
        compliance_keywords = {
            'gdpr': 'GDPR Compliance',
            'hipaa': 'HIPAA Compliance',
            'pci': 'PCI Compliance',
            'audit': 'Audit Logging',
            'encryption': 'Data Encryption'
        }
        
        for keyword, feature in compliance_keywords.items():
            if keyword in code_lower:
                features.append(feature)
        
        return features
    
    def _identify_integration_type(self, description: str) -> str:
        """
        Identifies integration type from description.
        """
        description_lower = description.lower()
        
        if 'webhook' in description_lower:
            return 'webhook'
        elif 'api' in description_lower:
            return 'api_client'
        elif 'queue' in description_lower:
            return 'message_queue'
        elif 'stream' in description_lower:
            return 'data_stream'
        else:
            return 'general'
    
    def _extract_error_handling(self, code: str) -> List[str]:
        """
        Extracts error handling patterns from code.
        """
        patterns = []
        code_lower = code.lower()
        
        error_keywords = {
            'try-catch': 'Exception Handling',
            'retry': 'Retry Logic',
            'timeout': 'Timeout Handling',
            'circuit breaker': 'Circuit Breaker',
            'fallback': 'Fallback Strategy'
        }
        
        for keyword, pattern in error_keywords.items():
            if keyword in code_lower:
                patterns.append(pattern)
        
        return patterns
    
    def _extract_monitoring_features(self, code: str) -> List[str]:
        """
        Extracts monitoring features from code.
        """
        features = []
        code_lower = code.lower()
        
        monitoring_keywords = {
            'logging': 'Logging',
            'metrics': 'Metrics Collection',
            'tracing': 'Distributed Tracing',
            'health check': 'Health Checks',
            'alerting': 'Alerting'
        }
        
        for keyword, feature in monitoring_keywords.items():
            if keyword in code_lower:
                features.append(feature)
        
        return features
    
    def _identify_technologies(self, code: str) -> List[str]:
        """
        Identifies technologies used in the implementation.
        """
        technologies = []
        code_lower = code.lower()
        
        all_technologies = self.supported_frameworks + self.supported_databases
        
        for tech in all_technologies:
            if tech in code_lower:
                technologies.append(tech)
        
        return technologies
    
    def _extract_best_practices(self, code: str) -> List[str]:
        """
        Extracts best practices mentioned in the code.
        """
        practices = []
        code_lower = code.lower()
        
        practice_keywords = {
            'solid': 'SOLID Principles',
            'dry': 'DRY Principle',
            'separation of concerns': 'Separation of Concerns',
            'dependency injection': 'Dependency Injection',
            'clean code': 'Clean Code',
            'design pattern': 'Design Patterns'
        }
        
        for keyword, practice in practice_keywords.items():
            if keyword in code_lower:
                practices.append(practice)
        
        return practices
    
    def _extract_performance_features(self, code: str) -> List[str]:
        """
        Extracts performance optimization features from code.
        """
        features = []
        code_lower = code.lower()
        
        performance_keywords = {
            'caching': 'Caching',
            'indexing': 'Database Indexing',
            'pagination': 'Pagination',
            'lazy loading': 'Lazy Loading',
            'connection pooling': 'Connection Pooling',
            'async': 'Asynchronous Processing'
        }
        
        for keyword, feature in performance_keywords.items():
            if keyword in code_lower:
                features.append(feature)
        
        return features

    def invoke(self, state: AppState) -> AppState:
        """
        Legacy method for backward compatibility.
        The main entry point for the back-end development process.
        """
        print("---BACK-END DEVELOPER AGENT---")
        
        self.log_activity(ActivityType.PROCESSING, "Legacy invoke method called")
        
        # Placeholder logic
        state.logs.append(LogEntry(
            source="BackendDeveloperAgent",
            message="Back-end development logic not yet implemented."
        ))
        
        return state