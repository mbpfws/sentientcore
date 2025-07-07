"""Health monitoring system for Sentient Core."""

import asyncio
import logging
import psutil
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


class ComponentType(str, Enum):
    """Component types for monitoring."""
    SERVICE = "service"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    SYSTEM_RESOURCE = "system_resource"
    MEMORY_LAYER = "memory_layer"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    component_type: ComponentType
    status: HealthStatus
    metrics: List[HealthMetric] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    uptime: Optional[float] = None


class SystemHealth(BaseModel):
    """Overall system health."""
    status: HealthStatus
    components: Dict[str, ComponentHealth]
    system_metrics: List[HealthMetric]
    timestamp: datetime
    uptime: float
    error_count: int = 0
    warning_count: int = 0


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._components: Dict[str, ComponentHealth] = {}
        self._start_time = time.time()
        self._check_interval = 30  # seconds
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_history: List[SystemHealth] = []
        self._max_history = 100
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            self.logger.warning("Health monitoring already running")
            return
        
        self.logger.info("Starting health monitoring")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self._check_interval)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        # Check system resources
        await self._check_system_resources()
        
        # Check registered components
        for component_name in list(self._components.keys()):
            await self._check_component(component_name)
        
        # Update overall health
        system_health = self._calculate_system_health()
        self._health_history.append(system_health)
        
        # Keep history limited
        if len(self._health_history) > self._max_history:
            self._health_history.pop(0)
        
        # Log critical issues
        if system_health.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            self.logger.critical(f"System health critical: {system_health.error_count} errors, {system_health.warning_count} warnings")
    
    async def _check_system_resources(self):
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._get_status_from_thresholds(cpu_percent, 70, 90)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = self._get_status_from_thresholds(memory_percent, 80, 95)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = self._get_status_from_thresholds(disk_percent, 80, 95)
            
            # Network connections
            connections = len(psutil.net_connections())
            connection_status = self._get_status_from_thresholds(connections, 100, 200)
            
            # Update system resources component
            system_component = ComponentHealth(
                name="system_resources",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=max([cpu_status, memory_status, disk_status, connection_status], key=lambda x: list(HealthStatus).index(x)),
                metrics=[
                    HealthMetric("cpu_usage", cpu_percent, "%", cpu_status, 70, 90),
                    HealthMetric("memory_usage", memory_percent, "%", memory_status, 80, 95),
                    HealthMetric("disk_usage", disk_percent, "%", disk_status, 80, 95),
                    HealthMetric("network_connections", connections, "count", connection_status, 100, 200)
                ],
                uptime=time.time() - self._start_time
            )
            
            self._components["system_resources"] = system_component
            
        except Exception as e:
            self.logger.error(f"Error checking system resources: {e}")
            self._components["system_resources"] = ComponentHealth(
                name="system_resources",
                component_type=ComponentType.SYSTEM_RESOURCE,
                status=HealthStatus.CRITICAL,
                error_message=str(e)
            )
    
    async def _check_component(self, component_name: str):
        """Check individual component health."""
        component = self._components.get(component_name)
        if not component:
            return
        
        try:
            # Update last check time
            component.last_check = datetime.utcnow()
            
            # Component-specific health checks would go here
            # For now, we'll just mark as healthy if no errors
            if not component.error_message:
                component.status = HealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Error checking component {component_name}: {e}")
            component.status = HealthStatus.CRITICAL
            component.error_message = str(e)
    
    def register_component(
        self,
        name: str,
        component_type: ComponentType,
        initial_status: HealthStatus = HealthStatus.HEALTHY
    ):
        """Register a component for monitoring."""
        self._components[name] = ComponentHealth(
            name=name,
            component_type=component_type,
            status=initial_status
        )
        self.logger.info(f"Registered component for monitoring: {name}")
    
    def update_component_status(
        self,
        name: str,
        status: HealthStatus,
        error_message: Optional[str] = None,
        metrics: Optional[List[HealthMetric]] = None
    ):
        """Update component status."""
        if name not in self._components:
            self.logger.warning(f"Component {name} not registered")
            return
        
        component = self._components[name]
        component.status = status
        component.error_message = error_message
        component.last_check = datetime.utcnow()
        
        if metrics:
            component.metrics = metrics
    
    def _get_status_from_thresholds(
        self,
        value: float,
        warning_threshold: float,
        critical_threshold: float
    ) -> HealthStatus:
        """Get health status based on thresholds."""
        if value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _calculate_system_health(self) -> SystemHealth:
        """Calculate overall system health."""
        if not self._components:
            return SystemHealth(
                status=HealthStatus.DOWN,
                components={},
                system_metrics=[],
                timestamp=datetime.utcnow(),
                uptime=time.time() - self._start_time
            )
        
        # Count status types
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.DOWN: 0
        }
        
        for component in self._components.values():
            status_counts[component.status] += 1
        
        # Determine overall status
        if status_counts[HealthStatus.DOWN] > 0:
            overall_status = HealthStatus.DOWN
        elif status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Collect system metrics
        system_metrics = []
        system_component = self._components.get("system_resources")
        if system_component:
            system_metrics = system_component.metrics
        
        return SystemHealth(
            status=overall_status,
            components=self._components.copy(),
            system_metrics=system_metrics,
            timestamp=datetime.utcnow(),
            uptime=time.time() - self._start_time,
            error_count=status_counts[HealthStatus.CRITICAL] + status_counts[HealthStatus.DOWN],
            warning_count=status_counts[HealthStatus.WARNING]
        )
    
    def get_current_health(self) -> SystemHealth:
        """Get current system health."""
        return self._calculate_system_health()
    
    def get_health_history(self, hours: int = 1) -> List[SystemHealth]:
        """Get health history for specified hours."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        return [h for h in self._health_history if h.timestamp >= cutoff_time]
    
    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health of specific component."""
        return self._components.get(name)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for API responses."""
        current_health = self.get_current_health()
        
        return {
            "status": current_health.status.value,
            "timestamp": current_health.timestamp.isoformat(),
            "uptime": current_health.uptime,
            "components": {
                name: {
                    "status": comp.status.value,
                    "type": comp.component_type.value,
                    "last_check": comp.last_check.isoformat(),
                    "error_message": comp.error_message,
                    "metrics_count": len(comp.metrics)
                }
                for name, comp in current_health.components.items()
            },
            "error_count": current_health.error_count,
            "warning_count": current_health.warning_count,
            "system_metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "status": metric.status.value
                }
                for metric in current_health.system_metrics
            ]
        }


# Global health monitor instance
health_monitor = HealthMonitor()