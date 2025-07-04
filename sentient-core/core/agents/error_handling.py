"""Error Handling for Multi-Agent System

This module provides comprehensive error handling, recovery strategies,
and resilience mechanisms for the agent framework.
"""

import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Type
from enum import Enum
from dataclasses import dataclass, field
import json

from .communication import MessageBus, MessageType, MessagePriority, get_message_bus
from ..models import LogEntry

logger = logging.getLogger(__name__)

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(str, Enum):
    """Error categories for classification"""
    INITIALIZATION = "initialization"
    COMMUNICATION = "communication"
    TASK_EXECUTION = "task_execution"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class RecoveryStrategy(str, Enum):
    """Recovery strategies for error handling"""
    RETRY = "retry"
    RESTART = "restart"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    SHUTDOWN = "shutdown"

@dataclass
class ErrorInfo:
    """Comprehensive error information"""
    error_id: str
    agent_id: str
    timestamp: datetime
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    traceback: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    recovery_strategy: Optional[RecoveryStrategy] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "error_id": self.error_id,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "traceback": self.traceback,
            "context": self.context,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorInfo':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['severity'] = ErrorSeverity(data['severity'])
        data['category'] = ErrorCategory(data['category'])
        if data.get('recovery_strategy'):
            data['recovery_strategy'] = RecoveryStrategy(data['recovery_strategy'])
        if data.get('resolution_time'):
            data['resolution_time'] = datetime.fromisoformat(data['resolution_time'])
        return cls(**data)

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    strategy: RecoveryStrategy
    action: Callable
    max_attempts: int = 3
    delay: float = 1.0
    backoff_multiplier: float = 2.0
    conditions: List[Callable] = field(default_factory=list)
    
class ErrorHandler:
    """Comprehensive error handling system for agents
    
    Provides:
    - Error classification and severity assessment
    - Recovery strategy selection and execution
    - Error tracking and analytics
    - Circuit breaker patterns
    - Escalation mechanisms
    """
    
    def __init__(self, 
                 agent_id: str,
                 message_bus: Optional[MessageBus] = None):
        self.agent_id = agent_id
        self.message_bus = message_bus or get_message_bus()
        
        # Error storage
        self.errors: Dict[str, ErrorInfo] = {}
        self.error_history: List[ErrorInfo] = []
        
        # Recovery strategies
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryAction]] = {
            ErrorCategory.INITIALIZATION: [
                RecoveryAction(RecoveryStrategy.RETRY, self._retry_initialization, max_attempts=3),
                RecoveryAction(RecoveryStrategy.RESTART, self._restart_agent, max_attempts=1),
                RecoveryAction(RecoveryStrategy.ESCALATE, self._escalate_error)
            ],
            ErrorCategory.COMMUNICATION: [
                RecoveryAction(RecoveryStrategy.RETRY, self._retry_communication, max_attempts=5, delay=0.5),
                RecoveryAction(RecoveryStrategy.FALLBACK, self._fallback_communication),
                RecoveryAction(RecoveryStrategy.ESCALATE, self._escalate_error)
            ],
            ErrorCategory.TASK_EXECUTION: [
                RecoveryAction(RecoveryStrategy.RETRY, self._retry_task, max_attempts=3),
                RecoveryAction(RecoveryStrategy.FALLBACK, self._fallback_task),
                RecoveryAction(RecoveryStrategy.ESCALATE, self._escalate_error)
            ],
            ErrorCategory.RESOURCE: [
                RecoveryAction(RecoveryStrategy.RETRY, self._retry_with_delay, max_attempts=3, delay=5.0),
                RecoveryAction(RecoveryStrategy.ESCALATE, self._escalate_error)
            ],
            ErrorCategory.TIMEOUT: [
                RecoveryAction(RecoveryStrategy.RETRY, self._retry_with_extended_timeout, max_attempts=2),
                RecoveryAction(RecoveryStrategy.ESCALATE, self._escalate_error)
            ],
            ErrorCategory.NETWORK: [
                RecoveryAction(RecoveryStrategy.RETRY, self._retry_network, max_attempts=5, delay=1.0),
                RecoveryAction(RecoveryStrategy.FALLBACK, self._fallback_network),
                RecoveryAction(RecoveryStrategy.ESCALATE, self._escalate_error)
            ]
        }
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "resolved_errors": 0,
            "escalated_errors": 0,
            "recovery_success_rate": 0.0,
            "error_categories": {cat.value: 0 for cat in ErrorCategory},
            "error_severities": {sev.value: 0 for sev in ErrorSeverity}
        }
        
        logger.info(f"ErrorHandler initialized for agent {agent_id}")
    
    async def handle_error(self, 
                          error: Exception, 
                          context: Dict[str, Any] = None,
                          severity: Optional[ErrorSeverity] = None,
                          category: Optional[ErrorCategory] = None) -> bool:
        """Handle an error with automatic classification and recovery
        
        Args:
            error: Exception that occurred
            context: Additional context information
            severity: Override severity classification
            category: Override category classification
            
        Returns:
            True if error was handled successfully
        """
        try:
            # Create error info
            error_info = self._create_error_info(
                error, context or {}, severity, category
            )
            
            # Store error
            self.errors[error_info.error_id] = error_info
            self.error_history.append(error_info)
            
            # Update statistics
            self.stats["total_errors"] += 1
            self.stats["error_categories"][error_info.category.value] += 1
            self.stats["error_severities"][error_info.severity.value] += 1
            
            # Log error
            await self._log_error(error_info)
            
            # Check circuit breaker
            if self._is_circuit_open(error_info.category):
                logger.warning(f"Circuit breaker open for {error_info.category.value}")
                return False
            
            # Attempt recovery
            recovery_success = await self._attempt_recovery(error_info)
            
            if recovery_success:
                error_info.resolved = True
                error_info.resolution_time = datetime.utcnow()
                self.stats["resolved_errors"] += 1
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker(error_info.category)
                
                logger.info(f"Error {error_info.error_id} resolved successfully")
            else:
                # Update circuit breaker on failure
                self._update_circuit_breaker(error_info.category)
                
                logger.error(f"Failed to resolve error {error_info.error_id}")
            
            # Update success rate
            self._update_success_rate()
            
            return recovery_success
            
        except Exception as e:
            logger.error(f"Error in error handler: {str(e)}")
            return False
    
    def _create_error_info(self, 
                          error: Exception, 
                          context: Dict[str, Any],
                          severity: Optional[ErrorSeverity] = None,
                          category: Optional[ErrorCategory] = None) -> ErrorInfo:
        """Create comprehensive error information"""
        import uuid
        
        error_id = str(uuid.uuid4())
        error_type = type(error).__name__
        message = str(error)
        traceback_str = traceback.format_exc()
        
        # Auto-classify if not provided
        if severity is None:
            severity = self._classify_severity(error, context)
        
        if category is None:
            category = self._classify_category(error, context)
        
        return ErrorInfo(
            error_id=error_id,
            agent_id=self.agent_id,
            timestamp=datetime.utcnow(),
            error_type=error_type,
            message=message,
            severity=severity,
            category=category,
            traceback=traceback_str,
            context=context
        )
    
    def _classify_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Classify error severity based on error type and context"""
        # Critical errors
        if isinstance(error, (SystemExit, KeyboardInterrupt, MemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(error, (ConnectionError, TimeoutError, OSError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        
        # Check context for severity hints
        if context.get('critical', False):
            return ErrorSeverity.CRITICAL
        if context.get('high_priority', False):
            return ErrorSeverity.HIGH
        
        return ErrorSeverity.LOW
    
    def _classify_category(self, error: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Classify error category based on error type and context"""
        error_type = type(error).__name__
        message = str(error).lower()
        
        # Network-related errors
        if isinstance(error, (ConnectionError, TimeoutError)) or 'network' in message:
            return ErrorCategory.NETWORK
        
        # Resource-related errors
        if isinstance(error, (MemoryError, OSError)) or 'resource' in message:
            return ErrorCategory.RESOURCE
        
        # Timeout errors
        if 'timeout' in message or isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT
        
        # Validation errors
        if isinstance(error, (ValueError, TypeError)) or 'validation' in message:
            return ErrorCategory.VALIDATION
        
        # Configuration errors
        if 'config' in message or 'configuration' in message:
            return ErrorCategory.CONFIGURATION
        
        # Check context for category hints
        if 'category' in context:
            try:
                return ErrorCategory(context['category'])
            except ValueError:
                pass
        
        # Check operation context
        operation = context.get('operation', '')
        if 'init' in operation:
            return ErrorCategory.INITIALIZATION
        if 'task' in operation:
            return ErrorCategory.TASK_EXECUTION
        if 'comm' in operation:
            return ErrorCategory.COMMUNICATION
        
        return ErrorCategory.UNKNOWN
    
    async def _attempt_recovery(self, error_info: ErrorInfo) -> bool:
        """Attempt to recover from an error using appropriate strategies"""
        strategies = self.recovery_strategies.get(error_info.category, [])
        
        for strategy_action in strategies:
            if error_info.recovery_attempts >= error_info.max_recovery_attempts:
                logger.warning(f"Max recovery attempts reached for error {error_info.error_id}")
                break
            
            # Check conditions
            if strategy_action.conditions:
                if not all(condition(error_info) for condition in strategy_action.conditions):
                    continue
            
            try:
                error_info.recovery_attempts += 1
                error_info.recovery_strategy = strategy_action.strategy
                
                logger.info(f"Attempting recovery strategy {strategy_action.strategy.value} for error {error_info.error_id}")
                
                # Calculate delay with backoff
                delay = strategy_action.delay * (strategy_action.backoff_multiplier ** (error_info.recovery_attempts - 1))
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Execute recovery action
                success = await strategy_action.action(error_info)
                
                if success:
                    logger.info(f"Recovery successful using strategy {strategy_action.strategy.value}")
                    return True
                
            except Exception as e:
                logger.error(f"Recovery strategy {strategy_action.strategy.value} failed: {str(e)}")
        
        # If all strategies failed, escalate
        await self._escalate_error(error_info)
        return False
    
    async def _log_error(self, error_info: ErrorInfo):
        """Log error information"""
        # Create log entry
        log_entry = LogEntry(
            timestamp=error_info.timestamp,
            level="ERROR",
            message=f"Agent {self.agent_id} error: {error_info.message}",
            details={
                "error_id": error_info.error_id,
                "error_type": error_info.error_type,
                "severity": error_info.severity.value,
                "category": error_info.category.value,
                "context": error_info.context
            }
        )
        
        # Send to message bus for centralized logging
        if self.message_bus:
            await self.message_bus.send_message(
                sender_id=self.agent_id,
                recipient_id="logging_service",
                message_type=MessageType.LOG_EVENT,
                payload={
                    "log_entry": log_entry.dict(),
                    "error_info": error_info.to_dict()
                },
                priority=MessagePriority.HIGH if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else MessagePriority.NORMAL
            )
    
    # Recovery action implementations
    async def _retry_initialization(self, error_info: ErrorInfo) -> bool:
        """Retry initialization operation"""
        # This would be implemented by the specific agent
        logger.info(f"Retrying initialization for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _restart_agent(self, error_info: ErrorInfo) -> bool:
        """Restart the agent"""
        logger.info(f"Restarting agent {self.agent_id}")
        # Send restart message
        if self.message_bus:
            await self.message_bus.send_message(
                sender_id=self.agent_id,
                recipient_id="agent_manager",
                message_type=MessageType.CONTROL,
                payload={
                    "action": "restart",
                    "agent_id": self.agent_id,
                    "reason": "error_recovery",
                    "error_id": error_info.error_id
                },
                priority=MessagePriority.HIGH
            )
        return True
    
    async def _retry_communication(self, error_info: ErrorInfo) -> bool:
        """Retry communication operation"""
        logger.info(f"Retrying communication for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _fallback_communication(self, error_info: ErrorInfo) -> bool:
        """Use fallback communication method"""
        logger.info(f"Using fallback communication for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _retry_task(self, error_info: ErrorInfo) -> bool:
        """Retry task execution"""
        logger.info(f"Retrying task execution for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _fallback_task(self, error_info: ErrorInfo) -> bool:
        """Use fallback task execution method"""
        logger.info(f"Using fallback task execution for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _retry_with_delay(self, error_info: ErrorInfo) -> bool:
        """Retry with extended delay"""
        logger.info(f"Retrying with delay for agent {self.agent_id}")
        await asyncio.sleep(5.0)
        return False  # Placeholder
    
    async def _retry_with_extended_timeout(self, error_info: ErrorInfo) -> bool:
        """Retry with extended timeout"""
        logger.info(f"Retrying with extended timeout for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _retry_network(self, error_info: ErrorInfo) -> bool:
        """Retry network operation"""
        logger.info(f"Retrying network operation for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _fallback_network(self, error_info: ErrorInfo) -> bool:
        """Use fallback network configuration"""
        logger.info(f"Using fallback network for agent {self.agent_id}")
        return False  # Placeholder
    
    async def _escalate_error(self, error_info: ErrorInfo) -> bool:
        """Escalate error to higher level"""
        logger.warning(f"Escalating error {error_info.error_id} for agent {self.agent_id}")
        
        self.stats["escalated_errors"] += 1
        
        # Send escalation message
        if self.message_bus:
            await self.message_bus.send_message(
                sender_id=self.agent_id,
                recipient_id="error_manager",
                message_type=MessageType.ERROR,
                payload={
                    "action": "escalate",
                    "error_info": error_info.to_dict(),
                    "agent_id": self.agent_id
                },
                priority=MessagePriority.CRITICAL if error_info.severity == ErrorSeverity.CRITICAL else MessagePriority.HIGH
            )
        
        return True
    
    # Circuit breaker implementation
    def _is_circuit_open(self, category: ErrorCategory) -> bool:
        """Check if circuit breaker is open for a category"""
        breaker = self.circuit_breakers.get(category.value)
        if not breaker:
            return False
        
        if breaker['state'] == 'open':
            # Check if timeout has passed
            if datetime.utcnow() > breaker['timeout']:
                breaker['state'] = 'half_open'
                return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, category: ErrorCategory):
        """Update circuit breaker state on failure"""
        breaker = self.circuit_breakers.setdefault(category.value, {
            'failures': 0,
            'state': 'closed',
            'timeout': None,
            'threshold': 5,
            'timeout_duration': timedelta(minutes=5)
        })
        
        breaker['failures'] += 1
        
        if breaker['failures'] >= breaker['threshold'] and breaker['state'] == 'closed':
            breaker['state'] = 'open'
            breaker['timeout'] = datetime.utcnow() + breaker['timeout_duration']
            logger.warning(f"Circuit breaker opened for {category.value}")
    
    def _reset_circuit_breaker(self, category: ErrorCategory):
        """Reset circuit breaker on success"""
        if category.value in self.circuit_breakers:
            self.circuit_breakers[category.value]['failures'] = 0
            self.circuit_breakers[category.value]['state'] = 'closed'
            self.circuit_breakers[category.value]['timeout'] = None
    
    def _update_success_rate(self):
        """Update recovery success rate"""
        if self.stats["total_errors"] > 0:
            self.stats["recovery_success_rate"] = self.stats["resolved_errors"] / self.stats["total_errors"]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics"""
        return {
            **self.stats,
            "active_errors": len([e for e in self.errors.values() if not e.resolved]),
            "circuit_breakers": {k: v['state'] for k, v in self.circuit_breakers.items()},
            "recent_errors": len([e for e in self.error_history 
                                if (datetime.utcnow() - e.timestamp) < timedelta(hours=1)])
        }
    
    def get_recent_errors(self, hours: int = 24) -> List[ErrorInfo]:
        """Get recent errors within specified time window"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [error for error in self.error_history if error.timestamp > cutoff]
    
    def clear_resolved_errors(self, older_than_hours: int = 24):
        """Clear resolved errors older than specified time"""
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        # Remove from history
        self.error_history = [error for error in self.error_history 
                             if not error.resolved or error.timestamp > cutoff]
        
        # Remove from active errors
        to_remove = [error_id for error_id, error in self.errors.items() 
                    if error.resolved and error.timestamp < cutoff]
        
        for error_id in to_remove:
            del self.errors[error_id]
        
        logger.info(f"Cleared {len(to_remove)} resolved errors older than {older_than_hours} hours")

# Decorator for automatic error handling
def handle_agent_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       category: ErrorCategory = ErrorCategory.UNKNOWN,
                       retry_attempts: int = 3):
    """Decorator for automatic error handling in agent methods"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'error_handler'):
                logger.warning(f"Agent {self.agent_id} does not have error_handler")
                return await func(self, *args, **kwargs)
            
            for attempt in range(retry_attempts):
                try:
                    return await func(self, *args, **kwargs)
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'attempt': attempt + 1,
                        'max_attempts': retry_attempts,
                        'args': str(args),
                        'kwargs': str(kwargs)
                    }
                    
                    if attempt == retry_attempts - 1:  # Last attempt
                        await self.error_handler.handle_error(e, context, severity, category)
                        raise
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
        
        return wrapper
    return decorator