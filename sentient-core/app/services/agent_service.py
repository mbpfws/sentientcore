from typing import Dict, List, Optional, Any, AsyncGenerator, Union, Callable
import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
from pathlib import Path
import inspect
from abc import ABC, abstractmethod

from .memory_service import MemoryService, MemoryType
from .llm_service import EnhancedLLMService, ChatMessage, MessageRole
from .sse_manager import SSEConnectionManager, EventType
from .workflow_service import WorkflowOrchestrator, WorkflowTask, TaskType
from .research_service import EnhancedResearchService, ResearchQuery, SearchProvider


class AgentType(Enum):
    """Types of agents in the system"""
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    PLANNING = "planning"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class CommunicationProtocol(Enum):
    """Agent communication protocols"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    REQUEST_RESPONSE = "request_response"
    WORKFLOW = "workflow"


@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task assigned to an agent"""
    id: str
    type: TaskType
    priority: TaskPriority
    payload: Dict[str, Any]
    requester_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Message between agents"""
    id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    protocol: CommunicationProtocol
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agent configuration"""
    max_concurrent_tasks: int = 3
    task_timeout_seconds: int = 300
    memory_retention_hours: int = 24
    enable_learning: bool = True
    enable_collaboration: bool = True
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        name: str,
        description: str,
        config: AgentConfig,
        memory_service: MemoryService,
        llm_service: EnhancedLLMService,
        sse_manager: SSEConnectionManager
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.config = config
        
        # Core services
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.sse_manager = sse_manager
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.capabilities: List[AgentCapability] = []
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_history: List[AgentTask] = []
        self.message_handlers: Dict[str, Callable] = {}
        
        # Performance tracking
        self.stats = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_task_time": 0.0,
            "messages_sent": 0,
            "messages_received": 0,
            "uptime_start": datetime.now()
        }
        
        # Concurrency control
        self.task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Initialize agent
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the agent"""
        try:
            await self.initialize()
            self.status = AgentStatus.IDLE
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            
            # Notify initialization
            await self.sse_manager.broadcast_agent_status(
                self.agent_id,
                {
                    "status": self.status.value,
                    "type": self.agent_type.value,
                    "name": self.name,
                    "capabilities": [cap.name for cap in self.capabilities]
                }
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"Agent {self.agent_id} initialization failed: {e}")
    
    @abstractmethod
    async def initialize(self):
        """Initialize agent-specific resources"""
        pass
    
    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process a task assigned to this agent"""
        pass
    
    async def assign_task(self, task: AgentTask) -> bool:
        """Assign a task to this agent"""
        if self.status != AgentStatus.IDLE:
            return False
        
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            return False
        
        self.active_tasks[task.id] = task
        asyncio.create_task(self._execute_task(task))
        return True
    
    async def _execute_task(self, task: AgentTask):
        """Execute a task"""
        async with self.task_semaphore:
            try:
                self.status = AgentStatus.BUSY
                task.started_at = datetime.now()
                
                # Notify task start
                await self.sse_manager.broadcast_agent_status(
                    self.agent_id,
                    {
                        "status": self.status.value,
                        "current_task": task.id,
                        "task_type": task.type.value
                    },
                    task.session_id
                )
                
                # Process task with timeout
                result = await asyncio.wait_for(
                    self.process_task(task),
                    timeout=self.config.task_timeout_seconds
                )
                
                task.result = result
                task.completed_at = datetime.now()
                
                # Update stats
                self.stats["tasks_completed"] += 1
                self._update_average_task_time(task)
                
                # Store task result
                await self._store_task_result(task)
                
                self.logger.info(f"Task {task.id} completed successfully")
                
            except asyncio.TimeoutError:
                task.error = "Task timeout"
                task.completed_at = datetime.now()
                self.stats["tasks_failed"] += 1
                self.logger.error(f"Task {task.id} timed out")
                
            except Exception as e:
                task.error = str(e)
                task.completed_at = datetime.now()
                self.stats["tasks_failed"] += 1
                self.logger.error(f"Task {task.id} failed: {e}")
            
            finally:
                # Clean up
                self.active_tasks.pop(task.id, None)
                self.task_history.append(task)
                
                # Limit history
                if len(self.task_history) > 1000:
                    self.task_history = self.task_history[-500:]
                
                # Update status
                self.status = AgentStatus.IDLE if not self.active_tasks else AgentStatus.BUSY
                
                # Notify completion
                await self.sse_manager.broadcast_agent_status(
                    self.agent_id,
                    {
                        "status": self.status.value,
                        "task_completed": task.id,
                        "success": task.error is None
                    },
                    task.session_id
                )
    
    async def send_message(
        self,
        recipient_id: Optional[str],
        message_type: str,
        payload: Dict[str, Any],
        protocol: CommunicationProtocol = CommunicationProtocol.DIRECT,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Send message to another agent or broadcast"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            protocol=protocol,
            message_type=message_type,
            payload=payload,
            session_id=session_id,
            correlation_id=correlation_id
        )
        
        # This would be handled by the AgentService
        # For now, just log and update stats
        self.stats["messages_sent"] += 1
        self.logger.debug(f"Sent message {message.id} to {recipient_id or 'broadcast'}")
        
        return message.id
    
    async def handle_message(self, message: AgentMessage):
        """Handle incoming message"""
        self.stats["messages_received"] += 1
        
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Message handler failed: {e}")
        else:
            self.logger.warning(f"No handler for message type: {message.message_type}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
    
    async def _store_task_result(self, task: AgentTask):
        """Store task result in memory"""
        await self.memory_service.store_memory(
            content=json.dumps({
                "agent_id": self.agent_id,
                "task_id": task.id,
                "task_type": task.type.value,
                "result": task.result,
                "error": task.error,
                "execution_time": (task.completed_at - task.started_at).total_seconds() if task.completed_at and task.started_at else None
            }),
            memory_type=MemoryType.AGENT,
            metadata={
                "agent_id": self.agent_id,
                "task_id": task.id,
                "type": "task_result"
            }
        )
    
    def _update_average_task_time(self, task: AgentTask):
        """Update average task execution time"""
        if task.started_at and task.completed_at:
            execution_time = (task.completed_at - task.started_at).total_seconds()
            current_avg = self.stats["average_task_time"]
            completed_tasks = self.stats["tasks_completed"]
            
            if completed_tasks == 1:
                self.stats["average_task_time"] = execution_time
            else:
                self.stats["average_task_time"] = (
                    (current_avg * (completed_tasks - 1) + execution_time) / completed_tasks
                )
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "name": self.name,
            "status": self.status.value,
            "active_tasks": len(self.active_tasks),
            "capabilities": [cap.name for cap in self.capabilities],
            "stats": self.stats
        }


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent for coordinating other agents"""
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        memory_service: MemoryService,
        llm_service: EnhancedLLMService,
        sse_manager: SSEConnectionManager,
        workflow_service: WorkflowOrchestrator
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ORCHESTRATOR,
            name="Ultra Orchestrator",
            description="Coordinates and manages multi-agent workflows",
            config=config,
            memory_service=memory_service,
            llm_service=llm_service,
            sse_manager=sse_manager
        )
        
        self.workflow_service = workflow_service
        self.agent_registry: Dict[str, BaseAgent] = {}
        self.active_workflows: Dict[str, str] = {}  # task_id -> workflow_id
    
    async def initialize(self):
        """Initialize orchestrator capabilities"""
        self.capabilities = [
            AgentCapability(
                name="workflow_orchestration",
                description="Coordinate multi-agent workflows",
                input_schema={"workflow_definition": "object", "context": "object"},
                output_schema={"workflow_id": "string", "status": "string"}
            ),
            AgentCapability(
                name="agent_coordination",
                description="Coordinate agent interactions",
                input_schema={"agents": "array", "task": "object"},
                output_schema={"coordination_plan": "object"}
            ),
            AgentCapability(
                name="task_delegation",
                description="Delegate tasks to appropriate agents",
                input_schema={"task": "object", "requirements": "array"},
                output_schema={"assigned_agent": "string", "task_id": "string"}
            )
        ]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process orchestration task"""
        if task.type == TaskType.WORKFLOW:
            return await self._handle_workflow_task(task)
        elif task.type == TaskType.COORDINATION:
            return await self._handle_coordination_task(task)
        elif task.type == TaskType.DELEGATION:
            return await self._handle_delegation_task(task)
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
    
    async def _handle_workflow_task(self, task: AgentTask) -> Dict[str, Any]:
        """Handle workflow orchestration"""
        workflow_def = task.payload.get("workflow_definition")
        context = task.payload.get("context", {})
        
        if not workflow_def:
            raise ValueError("Workflow definition required")
        
        # Start workflow
        workflow_id = await self.workflow_service.start_workflow(
            workflow_def,
            context,
            task.session_id
        )
        
        self.active_workflows[task.id] = workflow_id
        
        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "Workflow orchestration initiated"
        }
    
    async def _handle_coordination_task(self, task: AgentTask) -> Dict[str, Any]:
        """Handle agent coordination"""
        agents = task.payload.get("agents", [])
        coordination_task = task.payload.get("task")
        
        # Create coordination plan
        plan = await self._create_coordination_plan(agents, coordination_task)
        
        return {
            "coordination_plan": plan,
            "status": "planned",
            "message": "Agent coordination plan created"
        }
    
    async def _handle_delegation_task(self, task: AgentTask) -> Dict[str, Any]:
        """Handle task delegation"""
        delegation_task = task.payload.get("task")
        requirements = task.payload.get("requirements", [])
        
        # Find best agent for task
        best_agent = await self._find_best_agent(delegation_task, requirements)
        
        if not best_agent:
            raise ValueError("No suitable agent found for task")
        
        # Create and assign task
        agent_task = AgentTask(
            id=str(uuid.uuid4()),
            type=TaskType(delegation_task.get("type", "general")),
            priority=TaskPriority(delegation_task.get("priority", 2)),
            payload=delegation_task.get("payload", {}),
            requester_id=self.agent_id,
            session_id=task.session_id
        )
        
        success = await best_agent.assign_task(agent_task)
        
        return {
            "assigned_agent": best_agent.agent_id,
            "task_id": agent_task.id,
            "success": success,
            "message": f"Task delegated to {best_agent.name}"
        }
    
    async def _create_coordination_plan(
        self,
        agents: List[str],
        coordination_task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create coordination plan using LLM"""
        prompt = f"""
        Create a coordination plan for the following agents and task:
        
        Agents: {', '.join(agents)}
        Task: {json.dumps(coordination_task, indent=2)}
        
        Provide a detailed coordination plan including:
        1. Task breakdown
        2. Agent assignments
        3. Communication flow
        4. Dependencies
        5. Timeline
        
        Return as structured JSON.
        """
        
        messages = [ChatMessage(MessageRole.USER, prompt)]
        response = await self.llm_service.chat_completion(messages=messages)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"plan": response.content, "structured": False}
    
    async def _find_best_agent(
        self,
        task: Dict[str, Any],
        requirements: List[str]
    ) -> Optional[BaseAgent]:
        """Find best agent for task"""
        best_agent = None
        best_score = 0
        
        for agent in self.agent_registry.values():
            if agent.status != AgentStatus.IDLE:
                continue
            
            score = self._calculate_agent_score(agent, task, requirements)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _calculate_agent_score(
        self,
        agent: BaseAgent,
        task: Dict[str, Any],
        requirements: List[str]
    ) -> float:
        """Calculate agent suitability score"""
        score = 0.0
        
        # Check capabilities
        agent_capabilities = [cap.name for cap in agent.capabilities]
        for req in requirements:
            if req in agent_capabilities:
                score += 1.0
        
        # Consider performance
        if agent.stats["tasks_completed"] > 0:
            success_rate = agent.stats["tasks_completed"] / (
                agent.stats["tasks_completed"] + agent.stats["tasks_failed"]
            )
            score += success_rate * 0.5
        
        # Consider current load
        load_factor = len(agent.active_tasks) / agent.config.max_concurrent_tasks
        score -= load_factor * 0.3
        
        return max(0, score)
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agent_registry[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")


class ResearchAgent(BaseAgent):
    """Research agent for information gathering"""
    
    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        memory_service: MemoryService,
        llm_service: EnhancedLLMService,
        sse_manager: SSEConnectionManager,
        research_service: EnhancedResearchService
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.RESEARCH,
            name="Research Agent",
            description="Conducts comprehensive research and information gathering",
            config=config,
            memory_service=memory_service,
            llm_service=llm_service,
            sse_manager=sse_manager
        )
        
        self.research_service = research_service
    
    async def initialize(self):
        """Initialize research capabilities"""
        self.capabilities = [
            AgentCapability(
                name="web_research",
                description="Conduct web research using multiple search providers",
                input_schema={"query": "string", "providers": "array", "max_results": "integer"},
                output_schema={"research_id": "string", "results": "array"}
            ),
            AgentCapability(
                name="academic_research",
                description="Conduct academic and scholarly research",
                input_schema={"topic": "string", "depth": "string"},
                output_schema={"findings": "object", "sources": "array"}
            ),
            AgentCapability(
                name="fact_verification",
                description="Verify facts and claims",
                input_schema={"claims": "array", "sources": "array"},
                output_schema={"verification_results": "array"}
            )
        ]
    
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """Process research task"""
        if task.type == TaskType.RESEARCH:
            return await self._handle_research_task(task)
        elif task.type == TaskType.ANALYSIS:
            return await self._handle_analysis_task(task)
        else:
            raise ValueError(f"Unsupported task type: {task.type}")
    
    async def _handle_research_task(self, task: AgentTask) -> Dict[str, Any]:
        """Handle research task"""
        query_text = task.payload.get("query")
        providers = task.payload.get("providers", ["tavily"])
        max_results = task.payload.get("max_results", 10)
        
        if not query_text:
            raise ValueError("Research query required")
        
        # Create research query
        research_query = ResearchQuery(
            query=query_text,
            providers=[SearchProvider(p) for p in providers],
            max_results=max_results
        )
        
        # Start research
        research_id = await self.research_service.start_research(
            research_query,
            task.session_id
        )
        
        # Wait for completion (with timeout)
        timeout = 60  # 1 minute timeout
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < timeout:
            research_task = await self.research_service.get_research_status(research_id)
            if research_task and research_task.status.value in ["completed", "failed"]:
                break
            await asyncio.sleep(1)
        
        # Get results
        results = await self.research_service.get_research_results(research_id)
        
        return {
            "research_id": research_id,
            "results": results,
            "status": "completed",
            "message": "Research completed successfully"
        }
    
    async def _handle_analysis_task(self, task: AgentTask) -> Dict[str, Any]:
        """Handle analysis task"""
        data = task.payload.get("data")
        analysis_type = task.payload.get("type", "general")
        
        if not data:
            raise ValueError("Data for analysis required")
        
        # Perform analysis using LLM
        prompt = f"""
        Perform {analysis_type} analysis on the following data:
        
        {json.dumps(data, indent=2)}
        
        Provide:
        1. Key insights
        2. Patterns and trends
        3. Recommendations
        4. Confidence assessment
        
        Return as structured JSON.
        """
        
        messages = [ChatMessage(MessageRole.USER, prompt)]
        response = await self.llm_service.chat_completion(messages=messages)
        
        try:
            analysis_result = json.loads(response.content)
        except json.JSONDecodeError:
            analysis_result = {"analysis": response.content, "structured": False}
        
        return {
            "analysis": analysis_result,
            "type": analysis_type,
            "status": "completed",
            "message": "Analysis completed successfully"
        }


class AgentService:
    """Service for managing multiple agents"""
    
    def __init__(
        self,
        memory_service: MemoryService,
        llm_service: EnhancedLLMService,
        sse_manager: SSEConnectionManager,
        workflow_service: WorkflowOrchestrator,
        research_service: EnhancedResearchService
    ):
        self.logger = logging.getLogger(__name__)
        
        # Core services
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.sse_manager = sse_manager
        self.workflow_service = workflow_service
        self.research_service = research_service
        
        # Agent management
        self.agents: Dict[str, BaseAgent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.orchestrator: Optional[OrchestratorAgent] = None
        
        # Communication
        self.subscriptions: Dict[str, List[str]] = {}  # topic -> agent_ids
        self.message_history: List[AgentMessage] = []
        
        # Start message processing
        asyncio.create_task(self._process_messages())
    
    async def initialize_default_agents(self):
        """Initialize default agents"""
        default_config = AgentConfig(
            max_concurrent_tasks=3,
            task_timeout_seconds=300,
            enable_learning=True,
            enable_collaboration=True
        )
        
        # Create orchestrator
        self.orchestrator = OrchestratorAgent(
            agent_id="orchestrator-001",
            config=default_config,
            memory_service=self.memory_service,
            llm_service=self.llm_service,
            sse_manager=self.sse_manager,
            workflow_service=self.workflow_service
        )
        
        await self.register_agent(self.orchestrator)
        
        # Create research agent
        research_agent = ResearchAgent(
            agent_id="research-001",
            config=default_config,
            memory_service=self.memory_service,
            llm_service=self.llm_service,
            sse_manager=self.sse_manager,
            research_service=self.research_service
        )
        
        await self.register_agent(research_agent)
        
        self.logger.info("Default agents initialized")
    
    async def register_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents[agent.agent_id] = agent
        
        # Register with orchestrator if available
        if self.orchestrator and agent.agent_id != self.orchestrator.agent_id:
            self.orchestrator.register_agent(agent)
        
        self.logger.info(f"Registered agent: {agent.agent_id}")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent.status = AgentStatus.OFFLINE
            
            # Unregister from orchestrator
            if self.orchestrator:
                self.orchestrator.unregister_agent(agent_id)
            
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def assign_task_to_agent(
        self,
        agent_id: str,
        task: AgentTask
    ) -> bool:
        """Assign task to specific agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False
        
        return await agent.assign_task(task)
    
    async def delegate_task(
        self,
        task: AgentTask,
        requirements: Optional[List[str]] = None
    ) -> Optional[str]:
        """Delegate task to best available agent"""
        if not self.orchestrator:
            return None
        
        delegation_task = AgentTask(
            id=str(uuid.uuid4()),
            type=TaskType.DELEGATION,
            priority=TaskPriority.NORMAL,
            payload={
                "task": {
                    "type": task.type.value,
                    "priority": task.priority.value,
                    "payload": task.payload
                },
                "requirements": requirements or []
            },
            session_id=task.session_id
        )
        
        success = await self.orchestrator.assign_task(delegation_task)
        return delegation_task.id if success else None
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: Optional[str],
        message_type: str,
        payload: Dict[str, Any],
        protocol: CommunicationProtocol = CommunicationProtocol.DIRECT,
        session_id: Optional[str] = None
    ) -> str:
        """Send message between agents"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            protocol=protocol,
            message_type=message_type,
            payload=payload,
            session_id=session_id
        )
        
        await self.message_queue.put(message)
        return message.id
    
    async def _process_messages(self):
        """Process message queue"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._route_message(message)
                self.message_history.append(message)
                
                # Limit history
                if len(self.message_history) > 10000:
                    self.message_history = self.message_history[-5000:]
                    
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    async def _route_message(self, message: AgentMessage):
        """Route message to appropriate agents"""
        if message.protocol == CommunicationProtocol.BROADCAST:
            # Send to all agents
            for agent in self.agents.values():
                if agent.agent_id != message.sender_id:
                    await agent.handle_message(message)
        
        elif message.protocol == CommunicationProtocol.PUBLISH_SUBSCRIBE:
            # Send to subscribed agents
            topic = message.payload.get("topic")
            if topic and topic in self.subscriptions:
                for agent_id in self.subscriptions[topic]:
                    agent = self.agents.get(agent_id)
                    if agent and agent.agent_id != message.sender_id:
                        await agent.handle_message(message)
        
        else:
            # Direct message
            if message.recipient_id and message.recipient_id in self.agents:
                agent = self.agents[message.recipient_id]
                await agent.handle_message(message)
    
    def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe agent to topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        
        if agent_id not in self.subscriptions[topic]:
            self.subscriptions[topic].append(agent_id)
    
    def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """Unsubscribe agent from topic"""
        if topic in self.subscriptions and agent_id in self.subscriptions[topic]:
            self.subscriptions[topic].remove(agent_id)
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status"""
        agent = self.agents.get(agent_id)
        return agent.get_status() if agent else None
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agents"""
        return [agent.get_status() for agent in self.agents.values()]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide agent statistics"""
        total_tasks = sum(agent.stats["tasks_completed"] + agent.stats["tasks_failed"] for agent in self.agents.values())
        active_tasks = sum(len(agent.active_tasks) for agent in self.agents.values())
        
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status != AgentStatus.OFFLINE]),
            "total_tasks_processed": total_tasks,
            "active_tasks": active_tasks,
            "messages_in_history": len(self.message_history),
            "subscriptions": len(self.subscriptions),
            "orchestrator_available": self.orchestrator is not None
        }