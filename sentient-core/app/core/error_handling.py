"""Enhanced error handling system for Sentient Core."""

import logging
import traceback
from typing import Any, Dict, Optional, Type
from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for better classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    DATABASE = "database"
    LLM_SERVICE = "llm_service"
    MEMORY_SERVICE = "memory_service"
    WORKFLOW = "workflow"
    RESEARCH = "research"
    AGENT = "agent"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorContext(BaseModel):
    """Context information for errors."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    additional_data: Dict[str, Any] = {}


class SentientCoreError(Exception):
    """Base exception class for Sentient Core."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_error = original_error
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc() if original_error else None


class ValidationError(SentientCoreError):
    """Validation-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class ServiceError(SentientCoreError):
    """Service-related errors."""
    
    def __init__(self, message: str, service_name: str, **kwargs):
        context = kwargs.get('context', ErrorContext())
        context.component = service_name
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class LLMServiceError(ServiceError):
    """LLM service-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="llm_service", category=ErrorCategory.LLM_SERVICE, **kwargs)


class MemoryServiceError(ServiceError):
    """Memory service-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="memory_service", category=ErrorCategory.MEMORY_SERVICE, **kwargs)


class WorkflowError(ServiceError):
    """Workflow-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="workflow_service", category=ErrorCategory.WORKFLOW, **kwargs)


class ResearchError(ServiceError):
    """Research service-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="research_service", category=ErrorCategory.RESEARCH, **kwargs)


class AgentError(ServiceError):
    """Agent service-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, service_name="agent_service", category=ErrorCategory.AGENT, **kwargs)


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._error_counts = {}
        self._error_history = []
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        reraise: bool = True
    ) -> Optional[SentientCoreError]:
        """Handle and log errors with proper categorization."""
        
        # Convert to SentientCoreError if needed
        if isinstance(error, SentientCoreError):
            sentient_error = error
        else:
            sentient_error = self._categorize_error(error, context)
        
        # Log the error
        self._log_error(sentient_error)
        
        # Track error statistics
        self._track_error(sentient_error)
        
        # Store in history
        self._error_history.append(sentient_error)
        
        # Keep only last 100 errors
        if len(self._error_history) > 100:
            self._error_history.pop(0)
        
        if reraise:
            raise sentient_error
        
        return sentient_error
    
    def _categorize_error(self, error: Exception, context: Optional[ErrorContext] = None) -> SentientCoreError:
        """Categorize unknown errors."""
        error_type = type(error).__name__
        message = str(error)
        
        # Categorize based on error type and message
        if "validation" in message.lower() or "invalid" in message.lower():
            category = ErrorCategory.VALIDATION
        elif "connection" in message.lower() or "network" in message.lower():
            category = ErrorCategory.NETWORK
        elif "database" in message.lower() or "sql" in message.lower():
            category = ErrorCategory.DATABASE
        elif "llm" in message.lower() or "model" in message.lower():
            category = ErrorCategory.LLM_SERVICE
        elif "memory" in message.lower():
            category = ErrorCategory.MEMORY_SERVICE
        elif "workflow" in message.lower():
            category = ErrorCategory.WORKFLOW
        elif "research" in message.lower():
            category = ErrorCategory.RESEARCH
        elif "agent" in message.lower():
            category = ErrorCategory.AGENT
        else:
            category = ErrorCategory.SYSTEM
        
        # Determine severity
        if any(word in message.lower() for word in ["critical", "fatal", "crash"]):
            severity = ErrorSeverity.CRITICAL
        elif any(word in message.lower() for word in ["error", "failed", "exception"]):
            severity = ErrorSeverity.HIGH
        elif any(word in message.lower() for word in ["warning", "warn"]):
            severity = ErrorSeverity.MEDIUM
        else:
            severity = ErrorSeverity.LOW
        
        return SentientCoreError(
            message=f"{error_type}: {message}",
            category=category,
            severity=severity,
            context=context,
            original_error=error
        )
    
    def _log_error(self, error: SentientCoreError):
        """Log error with appropriate level."""
        log_message = f"[{error.category.value.upper()}] {error.message}"
        
        if error.context:
            context_info = []
            if error.context.user_id:
                context_info.append(f"user={error.context.user_id}")
            if error.context.session_id:
                context_info.append(f"session={error.context.session_id}")
            if error.context.component:
                context_info.append(f"component={error.context.component}")
            if error.context.operation:
                context_info.append(f"operation={error.context.operation}")
            
            if context_info:
                log_message += f" | Context: {', '.join(context_info)}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=error.original_error)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=error.original_error)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _track_error(self, error: SentientCoreError):
        """Track error statistics."""
        key = f"{error.category.value}:{error.severity.value}"
        self._error_counts[key] = self._error_counts.get(key, 0) + 1
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self._error_counts.copy(),
            "recent_errors": len(self._error_history),
            "critical_errors": len([e for e in self._error_history if e.severity == ErrorSeverity.CRITICAL]),
            "high_errors": len([e for e in self._error_history if e.severity == ErrorSeverity.HIGH])
        }
    
    def get_recent_errors(self, limit: int = 10) -> list[SentientCoreError]:
        """Get recent errors."""
        return self._error_history[-limit:]


# Global error handler instance
error_handler = ErrorHandler()