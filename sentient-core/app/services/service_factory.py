from typing import Dict, Optional, Any, Type
import os
import logging
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from .memory_service import MemoryService
from .state_manager import EnhancedStateManager, PersistenceMode
from .sse_manager import SSEConnectionManager
from .llm_service import EnhancedLLMService, LLMConfig, LLMProvider
from .workflow_service import WorkflowOrchestrator
from .research_service import EnhancedResearchService, ResearchConfig
from .agent_service import AgentService, AgentConfig
from ..core.error_handling import error_handler, SentientCoreError, ErrorContext
from ..core.health_monitor import health_monitor, ComponentType, HealthStatus


@dataclass
class ServiceConfig:
    """Configuration for all services"""
    # Database paths
    memory_db_path: str = "app/memory_management.db"
    state_db_path: str = "app/state_management.db"
    state_storage_path: str = "app/state_storage"
    
    # LLM Configuration
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    default_llm_provider: str = "groq"
    default_model: str = "llama-3.1-70b-versatile"
    
    # Research Configuration
    tavily_api_key: Optional[str] = None
    exa_api_key: Optional[str] = None
    enable_research_cache: bool = True
    research_cache_hours: int = 24
    
    # Agent Configuration
    max_agents: int = 10
    agent_task_timeout: int = 300
    enable_agent_learning: bool = True
    
    # Workflow Configuration
    max_concurrent_workflows: int = 5
    workflow_timeout_minutes: int = 30
    
    # SSE Configuration
    sse_heartbeat_interval: int = 30
    max_sse_connections: int = 100
    
    # General
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True


class ServiceFactory:
    """Factory for creating and managing all services"""
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Service instances
        self._memory_service: Optional[MemoryService] = None
        self._state_manager: Optional[EnhancedStateManager] = None
        self._sse_manager: Optional[SSEConnectionManager] = None
        self._llm_service: Optional[EnhancedLLMService] = None
        self._workflow_service: Optional[WorkflowOrchestrator] = None
        self._research_service: Optional[EnhancedResearchService] = None
        self._agent_service: Optional[AgentService] = None
        
        # Initialization status
        self._initialized = False
        self._initialization_order = [
            "memory_service",
            "state_manager", 
            "sse_manager",
            "llm_service",
            "workflow_service",
            "research_service",
            "agent_service"
        ]
        
        # Load configuration from environment
        self._load_env_config()
    
    def _load_env_config(self):
        """Load configuration from environment variables"""
        # API Keys
        self.config.groq_api_key = os.getenv("GROQ_API_KEY", self.config.groq_api_key)
        self.config.google_api_key = os.getenv("GOOGLE_API_KEY", self.config.google_api_key)
        self.config.openai_api_key = os.getenv("OPENAI_API_KEY", self.config.openai_api_key)
        self.config.tavily_api_key = os.getenv("TAVILY_API_KEY", self.config.tavily_api_key)
        self.config.exa_api_key = os.getenv("EXA_API_KEY", self.config.exa_api_key)
        
        # LLM Settings
        self.config.default_llm_provider = os.getenv("DEFAULT_LLM_PROVIDER", self.config.default_llm_provider)
        self.config.default_model = os.getenv("DEFAULT_MODEL", self.config.default_model)
        
        # Paths
        self.config.memory_db_path = os.getenv("MEMORY_DB_PATH", self.config.memory_db_path)
        self.config.state_db_path = os.getenv("STATE_DB_PATH", self.config.state_db_path)
        self.config.state_storage_path = os.getenv("STATE_STORAGE_PATH", self.config.state_storage_path)
        
        # Feature flags
        self.config.enable_research_cache = os.getenv("ENABLE_RESEARCH_CACHE", "true").lower() == "true"
        self.config.enable_agent_learning = os.getenv("ENABLE_AGENT_LEARNING", "true").lower() == "true"
        self.config.enable_performance_monitoring = os.getenv("ENABLE_PERFORMANCE_MONITORING", "true").lower() == "true"
        
        # Numeric settings
        try:
            self.config.max_agents = int(os.getenv("MAX_AGENTS", str(self.config.max_agents)))
            self.config.agent_task_timeout = int(os.getenv("AGENT_TASK_TIMEOUT", str(self.config.agent_task_timeout)))
            self.config.max_concurrent_workflows = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", str(self.config.max_concurrent_workflows)))
            self.config.workflow_timeout_minutes = int(os.getenv("WORKFLOW_TIMEOUT_MINUTES", str(self.config.workflow_timeout_minutes)))
            self.config.research_cache_hours = int(os.getenv("RESEARCH_CACHE_HOURS", str(self.config.research_cache_hours)))
            self.config.sse_heartbeat_interval = int(os.getenv("SSE_HEARTBEAT_INTERVAL", str(self.config.sse_heartbeat_interval)))
            self.config.max_sse_connections = int(os.getenv("MAX_SSE_CONNECTIONS", str(self.config.max_sse_connections)))
        except ValueError as e:
            self.logger.warning(f"Invalid numeric environment variable: {e}")
        
        # Log level
        self.config.log_level = os.getenv("LOG_LEVEL", self.config.log_level)
        
        self.logger.info("Configuration loaded from environment")
    
    async def initialize_all(self) -> bool:
        """Initialize all services in correct order"""
        if self._initialized:
            self.logger.warning("Services already initialized")
            return True
        
        try:
            self.logger.info("Starting service initialization...")
            
            for service_name in self._initialization_order:
                self.logger.info(f"Initializing {service_name}...")
                
                try:
                    if service_name == "memory_service":
                        await self._init_memory_service()
                    elif service_name == "state_manager":
                        await self._init_state_manager()
                    elif service_name == "sse_manager":
                        await self._init_sse_manager()
                    elif service_name == "llm_service":
                        await self._init_llm_service()
                    elif service_name == "workflow_service":
                        await self._init_workflow_service()
                    elif service_name == "research_service":
                        await self._init_research_service()
                    elif service_name == "agent_service":
                        await self._init_agent_service()
                    
                    # Register successful initialization with health monitor
                    health_monitor.register_component(
                        service_name, ComponentType.SERVICE, HealthStatus.HEALTHY
                    )
                    self.logger.info(f"✓ {service_name} initialized successfully")
                    
                except Exception as service_error:
                    # Handle error with error handler
                    error_handler.handle_error(
                        service_error, ErrorContext(
                            component=service_name,
                            operation="initialization",
                            user_id="system"
                        )
                    )
                    # Register failed initialization with health monitor
                    health_monitor.register_component(
                        service_name, ComponentType.SERVICE, HealthStatus.UNHEALTHY
                    )
                    self.logger.error(f"✗ {service_name} initialization failed: {service_error}")
                    self.logger.error(f"Error type: {type(service_error).__name__}")
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise
            
            self._initialized = True
            # Register overall factory health
            health_monitor.register_component(
                "service_factory", ComponentType.SERVICE, HealthStatus.HEALTHY
            )
            self.logger.info("🎉 All services initialized successfully")
            return True
            
        except Exception as e:
            # Handle overall initialization error
            error_handler.handle_error(
                e, ErrorContext(
                    component="service_factory",
                    operation="initialize_all",
                    user_id="system"
                )
            )
            # Register factory as unhealthy
            health_monitor.register_component(
                "service_factory", ComponentType.SERVICE, HealthStatus.UNHEALTHY
            )
            self.logger.error(f"💥 Service initialization failed: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            await self.cleanup_all()
            return False
    
    async def _init_memory_service(self):
        """Initialize memory service"""
        self._memory_service = MemoryService(db_path=self.config.memory_db_path)
        # MemoryService initializes itself in __init__ via _init_database()
    
    async def _init_state_manager(self):
        """Initialize state manager"""
        self._state_manager = EnhancedStateManager(
            db_path=self.config.state_db_path,
            storage_path=self.config.state_storage_path,
            persistence_mode=PersistenceMode.IMMEDIATE
        )
        # EnhancedStateManager doesn't need async initialization
    
    async def _init_sse_manager(self):
        """Initialize SSE manager"""
        self._sse_manager = SSEConnectionManager(
            heartbeat_interval=self.config.sse_heartbeat_interval,
            max_queue_size=self.config.max_sse_connections
        )
        # SSE manager doesn't need async initialization
    
    async def _init_llm_service(self):
        """Initialize LLM service"""
        # Determine the primary provider and create a single LLMConfig
        default_config = None
        
        # Priority order: Groq -> Google -> OpenAI
        if self.config.groq_api_key:
            default_config = LLMConfig(
                provider=LLMProvider.GROQ,
                model=self.config.default_model or "mixtral-8x7b-32768",
                api_key=self.config.groq_api_key,
                temperature=0.7,
                max_tokens=4000,
                timeout=60,
                retry_attempts=3
            )
        elif self.config.google_api_key:
            default_config = LLMConfig(
                provider=LLMProvider.GOOGLE,
                model=self.config.default_model or "gemini-pro",
                api_key=self.config.google_api_key,
                temperature=0.7,
                max_tokens=4000,
                timeout=60,
                retry_attempts=3
            )
        elif self.config.openai_api_key:
            default_config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model=self.config.default_model or "gpt-3.5-turbo",
                api_key=self.config.openai_api_key,
                temperature=0.7,
                max_tokens=4000,
                timeout=60,
                retry_attempts=3
            )
        
        if not default_config:
            raise ValueError("No LLM API keys configured")
        
        self._llm_service = EnhancedLLMService(default_config)
        
        # Add additional provider configs if available
        if self.config.groq_api_key and default_config.provider != LLMProvider.GROQ:
            groq_config = LLMConfig(
                provider=LLMProvider.GROQ,
                model="mixtral-8x7b-32768",
                api_key=self.config.groq_api_key,
                temperature=0.7,
                max_tokens=4000,
                timeout=60,
                retry_attempts=3
            )
            self._llm_service.add_provider_config("groq", groq_config)
        
        if self.config.google_api_key and default_config.provider != LLMProvider.GOOGLE:
            google_config = LLMConfig(
                provider=LLMProvider.GOOGLE,
                model="gemini-pro",
                api_key=self.config.google_api_key,
                temperature=0.7,
                max_tokens=4000,
                timeout=60,
                retry_attempts=3
            )
            self._llm_service.add_provider_config("google", google_config)
        
        if self.config.openai_api_key and default_config.provider != LLMProvider.OPENAI:
            openai_config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-3.5-turbo",
                api_key=self.config.openai_api_key,
                temperature=0.7,
                max_tokens=4000,
                timeout=60,
                retry_attempts=3
            )
            self._llm_service.add_provider_config("openai", openai_config)
        
        # LLM service doesn't need async initialization
    
    async def _init_workflow_service(self):
        """Initialize workflow service"""
        if not all([self._memory_service, self._state_manager, self._llm_service, self._sse_manager]):
            raise ValueError("Required services not initialized for workflow service")
        
        self._workflow_service = WorkflowOrchestrator(
            memory_service=self._memory_service,
            state_manager=self._state_manager,
            llm_service=self._llm_service,
            sse_manager=self._sse_manager
        )
        # Workflow service doesn't need async initialization
    
    async def _init_research_service(self):
        """Initialize research service"""
        if not all([self._memory_service, self._llm_service, self._sse_manager]):
            raise ValueError("Required services not initialized for research service")
        
        research_config = ResearchConfig(
            tavily_api_key=self.config.tavily_api_key,
            exa_api_key=self.config.exa_api_key,
            cache_results=self.config.enable_research_cache,
            cache_duration_hours=self.config.research_cache_hours,
            max_concurrent_searches=5,
            default_timeout=30,
            enable_content_extraction=True,
            max_content_length=10000
        )
        
        self._research_service = EnhancedResearchService(
            config=research_config,
            memory_service=self._memory_service,
            llm_service=self._llm_service,
            sse_manager=self._sse_manager
        )
        # Research service doesn't need async initialization
    
    async def _init_agent_service(self):
        """Initialize agent service"""
        if not all([self._memory_service, self._llm_service, self._sse_manager, self._workflow_service, self._research_service]):
            raise ValueError("Required services not initialized for agent service")
        
        self._agent_service = AgentService(
            memory_service=self._memory_service,
            llm_service=self._llm_service,
            sse_manager=self._sse_manager,
            workflow_service=self._workflow_service,
            research_service=self._research_service
        )
        
        # Initialize default agents
        await self._agent_service.initialize_default_agents()
    
    async def cleanup_all(self):
        """Cleanup all services"""
        self.logger.info("Starting service cleanup...")
        
        # Cleanup in reverse order
        cleanup_order = list(reversed(self._initialization_order))
        
        for service_name in cleanup_order:
            try:
                if service_name == "agent_service" and self._agent_service:
                    # Agent service cleanup (if needed)
                    pass
                elif service_name == "research_service" and self._research_service:
                    await self._research_service.clear_cache()
                elif service_name == "workflow_service" and self._workflow_service:
                    # Workflow service cleanup (if needed)
                    pass
                elif service_name == "llm_service" and self._llm_service:
                    # LLM service cleanup (if needed)
                    pass
                elif service_name == "sse_manager" and self._sse_manager:
                    await self._sse_manager.cleanup()
                elif service_name == "state_manager" and self._state_manager:
                    # State manager cleanup (if needed)
                    pass
                elif service_name == "memory_service" and self._memory_service:
                    # Memory service cleanup (if needed)
                    pass
                
                self.logger.info(f"{service_name} cleaned up")
                
            except Exception as e:
                self.logger.error(f"Error cleaning up {service_name}: {e}")
        
        # Reset all services
        self._memory_service = None
        self._state_manager = None
        self._sse_manager = None
        self._llm_service = None
        self._workflow_service = None
        self._research_service = None
        self._agent_service = None
        
        self._initialized = False
        self.logger.info("Service cleanup completed")
    
    # Service getters
    @property
    def memory_service(self) -> MemoryService:
        """Get memory service"""
        if not self._memory_service:
            raise RuntimeError("Memory service not initialized")
        return self._memory_service
    
    @property
    def state_manager(self) -> EnhancedStateManager:
        """Get state manager"""
        if not self._state_manager:
            raise RuntimeError("State manager not initialized")
        return self._state_manager
    
    @property
    def sse_manager(self) -> SSEConnectionManager:
        """Get SSE manager"""
        if not self._sse_manager:
            raise RuntimeError("SSE manager not initialized")
        return self._sse_manager
    
    @property
    def llm_service(self) -> EnhancedLLMService:
        """Get LLM service"""
        if not self._llm_service:
            raise RuntimeError("LLM service not initialized")
        return self._llm_service
    
    @property
    def workflow_service(self) -> WorkflowOrchestrator:
        """Get workflow service"""
        if not self._workflow_service:
            raise RuntimeError("Workflow service not initialized")
        return self._workflow_service
    
    @property
    def research_service(self) -> EnhancedResearchService:
        """Get research service"""
        if not self._research_service:
            raise RuntimeError("Research service not initialized")
        return self._research_service
    
    @property
    def agent_service(self) -> AgentService:
        """Get agent service"""
        if not self._agent_service:
            raise RuntimeError("Agent service not initialized")
        return self._agent_service
    
    def is_initialized(self) -> bool:
        """Check if all services are initialized"""
        return self._initialized
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        return {
            "initialized": self._initialized,
            "services": {
                "memory_service": self._memory_service is not None,
                "state_manager": self._state_manager is not None,
                "sse_manager": self._sse_manager is not None,
                "llm_service": self._llm_service is not None,
                "workflow_service": self._workflow_service is not None,
                "research_service": self._research_service is not None,
                "agent_service": self._agent_service is not None
            },
            "config": {
                "default_llm_provider": self.config.default_llm_provider,
                "default_model": self.config.default_model,
                "research_cache_enabled": self.config.enable_research_cache,
                "agent_learning_enabled": self.config.enable_agent_learning,
                "performance_monitoring_enabled": self.config.enable_performance_monitoring,
                "max_agents": self.config.max_agents,
                "max_concurrent_workflows": self.config.max_concurrent_workflows
            }
        }
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get statistics from all services"""
        stats = {
            "factory": {
                "initialized": self._initialized,
                "services_count": len([s for s in [
                    self._memory_service, self._state_manager, self._sse_manager,
                    self._llm_service, self._workflow_service, self._research_service,
                    self._agent_service
                ] if s is not None])
            }
        }
        
        try:
            if self._llm_service:
                stats["llm"] = self._llm_service.get_stats()
        except Exception as e:
            self.logger.debug(f"Error getting LLM stats: {e}")
        
        try:
            if self._research_service:
                stats["research"] = self._research_service.get_stats()
        except Exception as e:
            self.logger.debug(f"Error getting research stats: {e}")
        
        try:
            if self._agent_service:
                stats["agents"] = self._agent_service.get_system_stats()
        except Exception as e:
            self.logger.debug(f"Error getting agent stats: {e}")
        
        try:
            if self._sse_manager:
                stats["sse"] = {
                    "active_connections": len(self._sse_manager._connections),
                    "total_events_sent": getattr(self._sse_manager, '_total_events_sent', 0)
                }
        except Exception as e:
            self.logger.debug(f"Error getting SSE stats: {e}")
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services"""
        health = {
            "overall": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
        issues = []
        
        # Check each service
        services_to_check = [
            ("memory_service", self._memory_service),
            ("state_manager", self._state_manager),
            ("sse_manager", self._sse_manager),
            ("llm_service", self._llm_service),
            ("workflow_service", self._workflow_service),
            ("research_service", self._research_service),
            ("agent_service", self._agent_service)
        ]
        
        for service_name, service in services_to_check:
            try:
                if service is None:
                    health["services"][service_name] = {
                        "status": "not_initialized",
                        "healthy": False
                    }
                    issues.append(f"{service_name} not initialized")
                else:
                    # Basic health check - service exists and has expected attributes
                    service_health = {
                        "status": "healthy",
                        "healthy": True
                    }
                    
                    # Add service-specific health checks
                    if service_name == "memory_service":
                        # Check if database is accessible
                        try:
                            await service.get_memories(limit=1)
                            service_health["database_accessible"] = True
                        except Exception as e:
                            service_health["database_accessible"] = False
                            service_health["error"] = str(e)
                            issues.append(f"Memory service database issue: {e}")
                    
                    elif service_name == "llm_service":
                        # Check if at least one provider is available
                        available_providers = len(service.config.providers)
                        service_health["available_providers"] = available_providers
                        if available_providers == 0:
                            service_health["healthy"] = False
                            issues.append("No LLM providers available")
                    
                    elif service_name == "agent_service":
                        # Check if agents are available
                        agent_count = len(service.agents)
                        service_health["agent_count"] = agent_count
                        if agent_count == 0:
                            service_health["healthy"] = False
                            issues.append("No agents available")
                    
                    health["services"][service_name] = service_health
                    
            except Exception as e:
                health["services"][service_name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e)
                }
                issues.append(f"{service_name} health check failed: {e}")
        
        # Determine overall health
        if issues:
            health["overall"] = "degraded" if len(issues) < len(services_to_check) // 2 else "unhealthy"
            health["issues"] = issues
        
        return health
    
    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Updated config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")


# Global service factory instance
_service_factory: Optional[ServiceFactory] = None


def get_service_factory() -> ServiceFactory:
    """Get global service factory instance"""
    global _service_factory
    if _service_factory is None:
        _service_factory = ServiceFactory()
    return _service_factory


def set_service_factory(factory: ServiceFactory):
    """Set global service factory instance"""
    global _service_factory
    _service_factory = factory


async def initialize_services(config: Optional[ServiceConfig] = None) -> bool:
    """Initialize all services with optional config"""
    factory = get_service_factory()
    if config:
        factory.config = config
    return await factory.initialize_all()


async def cleanup_services():
    """Cleanup all services"""
    factory = get_service_factory()
    await factory.cleanup_all()