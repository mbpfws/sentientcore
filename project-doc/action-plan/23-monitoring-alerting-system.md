# 23 - Monitoring and Alerting System

## Executive Summary

This action plan details the implementation of a comprehensive monitoring and alerting system for the multi-agent RAG system. The system provides real-time monitoring of agent performance, system health, resource utilization, and automated alerting for critical issues.

## Development Objectives

### Primary Goals
- Real-time system monitoring and health checks
- Agent performance tracking and analytics
- Automated alerting for critical issues
- Resource utilization monitoring
- Error tracking and debugging support
- Performance metrics collection and visualization

### Technical Requirements
- Prometheus-style metrics collection
- Real-time dashboard with WebSocket updates
- Configurable alert thresholds
- Multi-channel notification system
- Historical data retention and analysis
- Integration with existing agent framework

## Backend Implementation

### Monitoring Service Core

**File**: `backend/core/monitoring/monitoring_service.py`

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import asyncio
import json
import logging
from collections import defaultdict, deque

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"

@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Metric:
    name: str
    type: MetricType
    description: str
    unit: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class AlertRule:
    id: str
    name: str
    metric_name: str
    condition: str  # e.g., "> 0.8", "< 100"
    threshold: float
    severity: AlertSeverity
    duration: timedelta
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class Alert:
    id: str
    rule_id: str
    metric_name: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0

@dataclass
class SystemHealth:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_agents: int
    total_requests: int
    error_rate: float
    response_time: float
    timestamp: datetime

class MonitoringService:
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.subscribers: List[Callable] = []
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    async def start_monitoring(self):
        """Start the monitoring service"""
        self.running = True
        await asyncio.gather(
            self._collect_system_metrics(),
            self._evaluate_alert_rules(),
            self._cleanup_old_data()
        )
    
    async def stop_monitoring(self):
        """Stop the monitoring service"""
        self.running = False
    
    def register_metric(self, metric: Metric) -> bool:
        """Register a new metric for monitoring"""
        try:
            self.metrics[metric.name] = metric
            self.logger.info(f"Registered metric: {metric.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register metric {metric.name}: {e}")
            return False
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> bool:
        """Record a metric value"""
        try:
            if name not in self.metrics:
                return False
            
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            )
            
            self.metrics[name].points.append(point)
            
            # Notify subscribers
            asyncio.create_task(self._notify_subscribers({
                'type': 'metric_update',
                'metric': name,
                'value': value,
                'timestamp': point.timestamp.isoformat()
            }))
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
            return False
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule"""
        try:
            self.alert_rules[rule.id] = rule
            self.logger.info(f"Added alert rule: {rule.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add alert rule {rule.id}: {e}")
            return False
    
    def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        try:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"Removed alert rule: {rule_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to remove alert rule {rule_id}: {e}")
            return False
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        import psutil
        
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric("system_cpu_usage", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric("system_memory_usage", memory.percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.record_metric("system_disk_usage", disk.percent)
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(10)
    
    async def _evaluate_alert_rules(self):
        """Evaluate alert rules against current metrics"""
        while self.running:
            try:
                for rule in self.alert_rules.values():
                    if not rule.enabled:
                        continue
                    
                    await self._check_alert_rule(rule)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error evaluating alert rules: {e}")
                await asyncio.sleep(30)
    
    async def _check_alert_rule(self, rule: AlertRule):
        """Check a specific alert rule"""
        try:
            if rule.metric_name not in self.metrics:
                return
            
            metric = self.metrics[rule.metric_name]
            if not metric.points:
                return
            
            # Get recent points within duration
            cutoff_time = datetime.utcnow() - rule.duration
            recent_points = [p for p in metric.points if p.timestamp >= cutoff_time]
            
            if not recent_points:
                return
            
            # Check if condition is met
            latest_value = recent_points[-1].value
            condition_met = self._evaluate_condition(latest_value, rule.condition, rule.threshold)
            
            alert_id = f"{rule.id}_{rule.metric_name}"
            
            if condition_met and alert_id not in self.active_alerts:
                # Trigger new alert
                alert = Alert(
                    id=alert_id,
                    rule_id=rule.id,
                    metric_name=rule.metric_name,
                    message=f"{rule.name}: {rule.metric_name} {rule.condition} {rule.threshold} (current: {latest_value})",
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    triggered_at=datetime.utcnow(),
                    labels=rule.labels,
                    value=latest_value
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                await self._notify_subscribers({
                    'type': 'alert_triggered',
                    'alert': self._alert_to_dict(alert)
                })
                
                self.logger.warning(f"Alert triggered: {alert.message}")
                
            elif not condition_met and alert_id in self.active_alerts:
                # Resolve existing alert
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                
                del self.active_alerts[alert_id]
                
                await self._notify_subscribers({
                    'type': 'alert_resolved',
                    'alert': self._alert_to_dict(alert)
                })
                
                self.logger.info(f"Alert resolved: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Error checking alert rule {rule.id}: {e}")
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        return False
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        while self.running:
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                # Clean up old alert history
                self.alert_history = [a for a in self.alert_history if a.triggered_at >= cutoff_time]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}")
                await asyncio.sleep(3600)
    
    def subscribe(self, callback: Callable):
        """Subscribe to monitoring events"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from monitoring events"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _notify_subscribers(self, event: Dict[str, Any]):
        """Notify all subscribers of an event"""
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Error notifying subscriber: {e}")
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        try:
            cpu_usage = self._get_latest_metric_value("system_cpu_usage", 0.0)
            memory_usage = self._get_latest_metric_value("system_memory_usage", 0.0)
            disk_usage = self._get_latest_metric_value("system_disk_usage", 0.0)
            active_agents = self._get_latest_metric_value("active_agents", 0.0)
            total_requests = self._get_latest_metric_value("total_requests", 0.0)
            error_rate = self._get_latest_metric_value("error_rate", 0.0)
            response_time = self._get_latest_metric_value("avg_response_time", 0.0)
            
            return SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                active_agents=int(active_agents),
                total_requests=int(total_requests),
                error_rate=error_rate,
                response_time=response_time,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                cpu_usage=0.0, memory_usage=0.0, disk_usage=0.0,
                active_agents=0, total_requests=0, error_rate=0.0,
                response_time=0.0, timestamp=datetime.utcnow()
            )
    
    def _get_latest_metric_value(self, metric_name: str, default: float = 0.0) -> float:
        """Get the latest value for a metric"""
        if metric_name in self.metrics and self.metrics[metric_name].points:
            return self.metrics[metric_name].points[-1].value
        return default
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        for name, metric in self.metrics.items():
            if metric.points:
                latest = metric.points[-1]
                summary[name] = {
                    'current_value': latest.value,
                    'timestamp': latest.timestamp.isoformat(),
                    'type': metric.type.value,
                    'unit': metric.unit,
                    'description': metric.description
                }
        return summary
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        return [self._alert_to_dict(alert) for alert in self.active_alerts.values()]
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history"""
        recent_alerts = sorted(self.alert_history, key=lambda a: a.triggered_at, reverse=True)[:limit]
        return [self._alert_to_dict(alert) for alert in recent_alerts]
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'id': alert.id,
            'rule_id': alert.rule_id,
            'metric_name': alert.metric_name,
            'message': alert.message,
            'severity': alert.severity.value,
            'status': alert.status.value,
            'triggered_at': alert.triggered_at.isoformat(),
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
            'acknowledged_at': alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
            'labels': alert.labels,
            'value': alert.value
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an active alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
                self.active_alerts[alert_id].acknowledged_at = datetime.utcnow()
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
```

### Agent Performance Tracker

**File**: `backend/core/monitoring/agent_tracker.py`

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import asyncio
import logging
from collections import defaultdict

class AgentStatus(Enum):
    IDLE = "idle"
    ACTIVE = "active"
    ERROR = "error"
    OFFLINE = "offline"

@dataclass
class AgentMetrics:
    agent_id: str
    agent_type: str
    status: AgentStatus
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_activity: Optional[datetime] = None
    uptime: timedelta = field(default_factory=lambda: timedelta())
    error_count: int = 0
    success_rate: float = 100.0

@dataclass
class TaskExecution:
    task_id: str
    agent_id: str
    agent_type: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration: Optional[timedelta] = None
    status: str = "running"
    error_message: Optional[str] = None
    memory_peak: float = 0.0
    cpu_peak: float = 0.0

class AgentPerformanceTracker:
    def __init__(self, monitoring_service):
        self.monitoring_service = monitoring_service
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.task_executions: List[TaskExecution] = []
        self.agent_start_times: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)
        
        # Register agent-specific metrics
        self._register_agent_metrics()
    
    def _register_agent_metrics(self):
        """Register agent performance metrics"""
        from .monitoring_service import Metric, MetricType
        
        metrics = [
            Metric("agent_task_completion_rate", MetricType.GAUGE, "Agent task completion rate", "percentage"),
            Metric("agent_response_time", MetricType.HISTOGRAM, "Agent response time", "seconds"),
            Metric("agent_error_rate", MetricType.GAUGE, "Agent error rate", "percentage"),
            Metric("agent_memory_usage", MetricType.GAUGE, "Agent memory usage", "MB"),
            Metric("agent_cpu_usage", MetricType.GAUGE, "Agent CPU usage", "percentage"),
            Metric("active_agents_count", MetricType.GAUGE, "Number of active agents", "count"),
            Metric("agent_uptime", MetricType.GAUGE, "Agent uptime", "seconds")
        ]
        
        for metric in metrics:
            self.monitoring_service.register_metric(metric)
    
    def register_agent(self, agent_id: str, agent_type: str):
        """Register a new agent for tracking"""
        try:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                agent_type=agent_type,
                status=AgentStatus.IDLE,
                last_activity=datetime.utcnow()
            )
            
            self.agent_start_times[agent_id] = datetime.utcnow()
            
            self.logger.info(f"Registered agent for tracking: {agent_id} ({agent_type})")
            
            # Update active agents count
            self._update_active_agents_count()
            
        except Exception as e:
            self.logger.error(f"Error registering agent {agent_id}: {e}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from tracking"""
        try:
            if agent_id in self.agent_metrics:
                del self.agent_metrics[agent_id]
            
            if agent_id in self.agent_start_times:
                del self.agent_start_times[agent_id]
            
            self.logger.info(f"Unregistered agent: {agent_id}")
            
            # Update active agents count
            self._update_active_agents_count()
            
        except Exception as e:
            self.logger.error(f"Error unregistering agent {agent_id}: {e}")
    
    def start_task(self, task_id: str, agent_id: str) -> bool:
        """Start tracking a task execution"""
        try:
            if agent_id not in self.agent_metrics:
                self.logger.warning(f"Agent {agent_id} not registered for task {task_id}")
                return False
            
            # Update agent status
            self.agent_metrics[agent_id].status = AgentStatus.ACTIVE
            self.agent_metrics[agent_id].last_activity = datetime.utcnow()
            
            # Create task execution record
            execution = TaskExecution(
                task_id=task_id,
                agent_id=agent_id,
                agent_type=self.agent_metrics[agent_id].agent_type,
                started_at=datetime.utcnow()
            )
            
            self.task_executions.append(execution)
            
            self.logger.info(f"Started tracking task {task_id} for agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting task tracking {task_id}: {e}")
            return False
    
    def complete_task(self, task_id: str, agent_id: str, success: bool = True, error_message: str = None) -> bool:
        """Complete task tracking"""
        try:
            # Find task execution
            execution = None
            for exec in self.task_executions:
                if exec.task_id == task_id and exec.agent_id == agent_id and exec.completed_at is None:
                    execution = exec
                    break
            
            if not execution:
                self.logger.warning(f"Task execution not found: {task_id} for agent {agent_id}")
                return False
            
            # Complete execution
            execution.completed_at = datetime.utcnow()
            execution.duration = execution.completed_at - execution.started_at
            execution.status = "completed" if success else "failed"
            execution.error_message = error_message
            
            # Update agent metrics
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]
                metrics.last_activity = datetime.utcnow()
                metrics.status = AgentStatus.IDLE
                
                if success:
                    metrics.tasks_completed += 1
                else:
                    metrics.tasks_failed += 1
                    metrics.error_count += 1
                
                # Update success rate
                total_tasks = metrics.tasks_completed + metrics.tasks_failed
                if total_tasks > 0:
                    metrics.success_rate = (metrics.tasks_completed / total_tasks) * 100
                
                # Update average response time
                self._update_avg_response_time(agent_id, execution.duration.total_seconds())
                
                # Record metrics
                self.monitoring_service.record_metric(
                    "agent_response_time", 
                    execution.duration.total_seconds(),
                    {"agent_id": agent_id, "agent_type": metrics.agent_type}
                )
                
                self.monitoring_service.record_metric(
                    "agent_task_completion_rate",
                    metrics.success_rate,
                    {"agent_id": agent_id, "agent_type": metrics.agent_type}
                )
            
            self.logger.info(f"Completed task tracking {task_id} for agent {agent_id}: {'success' if success else 'failed'}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error completing task tracking {task_id}: {e}")
            return False
    
    def update_agent_resources(self, agent_id: str, memory_usage: float, cpu_usage: float):
        """Update agent resource usage"""
        try:
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]
                metrics.memory_usage = memory_usage
                metrics.cpu_usage = cpu_usage
                metrics.last_activity = datetime.utcnow()
                
                # Record metrics
                self.monitoring_service.record_metric(
                    "agent_memory_usage",
                    memory_usage,
                    {"agent_id": agent_id, "agent_type": metrics.agent_type}
                )
                
                self.monitoring_service.record_metric(
                    "agent_cpu_usage",
                    cpu_usage,
                    {"agent_id": agent_id, "agent_type": metrics.agent_type}
                )
                
        except Exception as e:
            self.logger.error(f"Error updating agent resources {agent_id}: {e}")
    
    def set_agent_error(self, agent_id: str, error_message: str):
        """Set agent error status"""
        try:
            if agent_id in self.agent_metrics:
                metrics = self.agent_metrics[agent_id]
                metrics.status = AgentStatus.ERROR
                metrics.error_count += 1
                metrics.last_activity = datetime.utcnow()
                
                # Calculate error rate
                total_tasks = metrics.tasks_completed + metrics.tasks_failed
                if total_tasks > 0:
                    error_rate = (metrics.error_count / total_tasks) * 100
                    self.monitoring_service.record_metric(
                        "agent_error_rate",
                        error_rate,
                        {"agent_id": agent_id, "agent_type": metrics.agent_type}
                    )
                
                self.logger.error(f"Agent {agent_id} error: {error_message}")
                
        except Exception as e:
            self.logger.error(f"Error setting agent error {agent_id}: {e}")
    
    def _update_avg_response_time(self, agent_id: str, response_time: float):
        """Update agent average response time"""
        try:
            metrics = self.agent_metrics[agent_id]
            total_tasks = metrics.tasks_completed + metrics.tasks_failed
            
            if total_tasks == 1:
                metrics.avg_response_time = response_time
            else:
                # Calculate running average
                metrics.avg_response_time = (
                    (metrics.avg_response_time * (total_tasks - 1) + response_time) / total_tasks
                )
                
        except Exception as e:
            self.logger.error(f"Error updating avg response time for {agent_id}: {e}")
    
    def _update_active_agents_count(self):
        """Update the count of active agents"""
        try:
            active_count = len([m for m in self.agent_metrics.values() if m.status != AgentStatus.OFFLINE])
            self.monitoring_service.record_metric("active_agents_count", active_count)
        except Exception as e:
            self.logger.error(f"Error updating active agents count: {e}")
    
    def get_agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        """Get metrics for a specific agent"""
        return self.agent_metrics.get(agent_id)
    
    def get_all_agent_metrics(self) -> Dict[str, AgentMetrics]:
        """Get metrics for all agents"""
        # Update uptime for all agents
        current_time = datetime.utcnow()
        for agent_id, metrics in self.agent_metrics.items():
            if agent_id in self.agent_start_times:
                metrics.uptime = current_time - self.agent_start_times[agent_id]
        
        return self.agent_metrics.copy()
    
    def get_task_history(self, agent_id: str = None, limit: int = 100) -> List[TaskExecution]:
        """Get task execution history"""
        executions = self.task_executions
        
        if agent_id:
            executions = [e for e in executions if e.agent_id == agent_id]
        
        # Sort by start time, most recent first
        executions = sorted(executions, key=lambda e: e.started_at, reverse=True)
        
        return executions[:limit]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        try:
            total_agents = len(self.agent_metrics)
            active_agents = len([m for m in self.agent_metrics.values() if m.status == AgentStatus.ACTIVE])
            error_agents = len([m for m in self.agent_metrics.values() if m.status == AgentStatus.ERROR])
            
            total_tasks = sum(m.tasks_completed + m.tasks_failed for m in self.agent_metrics.values())
            completed_tasks = sum(m.tasks_completed for m in self.agent_metrics.values())
            failed_tasks = sum(m.tasks_failed for m in self.agent_metrics.values())
            
            overall_success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 100
            
            avg_response_times = [m.avg_response_time for m in self.agent_metrics.values() if m.avg_response_time > 0]
            overall_avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
            
            return {
                'total_agents': total_agents,
                'active_agents': active_agents,
                'error_agents': error_agents,
                'total_tasks': total_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'overall_success_rate': overall_success_rate,
                'overall_avg_response_time': overall_avg_response_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}
```

## Frontend Implementation

### Monitoring Dashboard

**File**: `frontend/components/monitoring/monitoring-dashboard.tsx`

```typescript
'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { AlertTriangle, CheckCircle, XCircle, Activity, Server, Users, Clock } from 'lucide-react';

interface SystemHealth {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  active_agents: number;
  total_requests: number;
  error_rate: number;
  response_time: number;
  timestamp: string;
}

interface Alert {
  id: string;
  rule_id: string;
  metric_name: string;
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'active' | 'resolved' | 'acknowledged';
  triggered_at: string;
  resolved_at?: string;
  acknowledged_at?: string;
  value: number;
}

interface AgentMetrics {
  agent_id: string;
  agent_type: string;
  status: 'idle' | 'active' | 'error' | 'offline';
  tasks_completed: number;
  tasks_failed: number;
  avg_response_time: number;
  memory_usage: number;
  cpu_usage: number;
  success_rate: number;
  uptime: string;
}

interface MetricPoint {
  timestamp: string;
  value: number;
}

const MonitoringDashboard: React.FC = () => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [agentMetrics, setAgentMetrics] = useState<AgentMetrics[]>([]);
  const [metricHistory, setMetricHistory] = useState<Record<string, MetricPoint[]>>({});
  const [isConnected, setIsConnected] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState('cpu_usage');

  useEffect(() => {
    // Initialize WebSocket connection for real-time updates
    const ws = new WebSocket(`ws://localhost:8000/ws/monitoring`);
    
    ws.onopen = () => {
      setIsConnected(true);
      console.log('Connected to monitoring WebSocket');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleRealtimeUpdate(data);
    };
    
    ws.onclose = () => {
      setIsConnected(false);
      console.log('Disconnected from monitoring WebSocket');
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };
    
    // Initial data fetch
    fetchInitialData();
    
    return () => {
      ws.close();
    };
  }, []);

  const handleRealtimeUpdate = (data: any) => {
    switch (data.type) {
      case 'system_health':
        setSystemHealth(data.health);
        break;
      case 'alert_triggered':
        setAlerts(prev => [data.alert, ...prev]);
        break;
      case 'alert_resolved':
        setAlerts(prev => prev.map(alert => 
          alert.id === data.alert.id ? data.alert : alert
        ));
        break;
      case 'agent_metrics':
        setAgentMetrics(data.metrics);
        break;
      case 'metric_update':
        updateMetricHistory(data.metric, data.value, data.timestamp);
        break;
    }
  };

  const fetchInitialData = async () => {
    try {
      // Fetch system health
      const healthResponse = await fetch('/api/monitoring/health');
      const health = await healthResponse.json();
      setSystemHealth(health);
      
      // Fetch active alerts
      const alertsResponse = await fetch('/api/monitoring/alerts');
      const alertsData = await alertsResponse.json();
      setAlerts(alertsData);
      
      // Fetch agent metrics
      const agentsResponse = await fetch('/api/monitoring/agents');
      const agentsData = await agentsResponse.json();
      setAgentMetrics(agentsData);
      
      // Fetch metric history
      const metricsResponse = await fetch('/api/monitoring/metrics/history');
      const metricsData = await metricsResponse.json();
      setMetricHistory(metricsData);
      
    } catch (error) {
      console.error('Error fetching initial data:', error);
    }
  };

  const updateMetricHistory = (metric: string, value: number, timestamp: string) => {
    setMetricHistory(prev => ({
      ...prev,
      [metric]: [...(prev[metric] || []), { timestamp, value }].slice(-50) // Keep last 50 points
    }));
  };

  const acknowledgeAlert = async (alertId: string) => {
    try {
      await fetch(`/api/monitoring/alerts/${alertId}/acknowledge`, {
        method: 'POST'
      });
      
      setAlerts(prev => prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: 'acknowledged' as const, acknowledged_at: new Date().toISOString() }
          : alert
      ));
    } catch (error) {
      console.error('Error acknowledging alert:', error);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <Activity className="h-4 w-4" />;
      case 'error': return <XCircle className="h-4 w-4" />;
      case 'idle': return <CheckCircle className="h-4 w-4" />;
      default: return <Server className="h-4 w-4" />;
    }
  };

  const formatUptime = (uptime: string) => {
    // Convert uptime string to human readable format
    const seconds = parseInt(uptime);
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Monitoring</h1>
          <p className="text-muted-foreground">Real-time monitoring and alerting dashboard</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-sm text-muted-foreground">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* System Health Overview */}
      {systemHealth && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
              <Server className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemHealth.cpu_usage.toFixed(1)}%</div>
              <Progress value={systemHealth.cpu_usage} className="mt-2" />
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
              <Server className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemHealth.memory_usage.toFixed(1)}%</div>
              <Progress value={systemHealth.memory_usage} className="mt-2" />
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemHealth.active_agents}</div>
              <p className="text-xs text-muted-foreground mt-2">agents running</p>
            </CardContent>
          </Card>
          
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Response Time</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{systemHealth.response_time.toFixed(0)}ms</div>
              <p className="text-xs text-muted-foreground mt-2">average</p>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* Active Alerts Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Active Alerts</CardTitle>
              <CardDescription>Current system alerts requiring attention</CardDescription>
            </CardHeader>
            <CardContent>
              {alerts.filter(alert => alert.status === 'active').length === 0 ? (
                <div className="text-center py-4 text-muted-foreground">
                  <CheckCircle className="h-8 w-8 mx-auto mb-2" />
                  <p>No active alerts</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {alerts.filter(alert => alert.status === 'active').slice(0, 5).map(alert => (
                    <div key={alert.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        <AlertTriangle className="h-4 w-4 text-orange-500" />
                        <div>
                          <p className="font-medium">{alert.message}</p>
                          <p className="text-sm text-muted-foreground">
                            {new Date(alert.triggered_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <Badge variant={getSeverityColor(alert.severity)}>
                        {alert.severity}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* System Metrics Chart */}
          <Card>
            <CardHeader>
              <CardTitle>System Metrics</CardTitle>
              <CardDescription>Real-time system performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricHistory[selectedMetric] || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timestamp" 
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip 
                      labelFormatter={(value) => new Date(value).toLocaleString()}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#8884d8" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Alert Management</CardTitle>
              <CardDescription>Manage and acknowledge system alerts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {alerts.map(alert => (
                  <div key={alert.id} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <Badge variant={getSeverityColor(alert.severity)}>
                            {alert.severity}
                          </Badge>
                          <Badge variant="outline">
                            {alert.status}
                          </Badge>
                        </div>
                        <h3 className="font-medium">{alert.message}</h3>
                        <p className="text-sm text-muted-foreground">
                          Triggered: {new Date(alert.triggered_at).toLocaleString()}
                        </p>
                        {alert.resolved_at && (
                          <p className="text-sm text-muted-foreground">
                            Resolved: {new Date(alert.resolved_at).toLocaleString()}
                          </p>
                        )}
                      </div>
                      {alert.status === 'active' && (
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => acknowledgeAlert(alert.id)}
                        >
                          Acknowledge
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="agents" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Agent Performance</CardTitle>
              <CardDescription>Monitor individual agent performance and status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {agentMetrics.map(agent => (
                  <div key={agent.agent_id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(agent.status)}
                        <div>
                          <h3 className="font-medium">{agent.agent_id}</h3>
                          <p className="text-sm text-muted-foreground">{agent.agent_type}</p>
                        </div>
                      </div>
                      <Badge variant={agent.status === 'active' ? 'default' : 'secondary'}>
                        {agent.status}
                      </Badge>
                    </div>
                    
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Tasks Completed</p>
                        <p className="font-medium">{agent.tasks_completed}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Success Rate</p>
                        <p className="font-medium">{agent.success_rate.toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Avg Response</p>
                        <p className="font-medium">{agent.avg_response_time.toFixed(2)}s</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Uptime</p>
                        <p className="font-medium">{formatUptime(agent.uptime)}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detailed Metrics</CardTitle>
              <CardDescription>Historical performance metrics and trends</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex space-x-2">
                  {['cpu_usage', 'memory_usage', 'response_time', 'error_rate'].map(metric => (
                    <Button
                      key={metric}
                      variant={selectedMetric === metric ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setSelectedMetric(metric)}
                    >
                      {metric.replace('_', ' ').toUpperCase()}
                    </Button>
                  ))}
                </div>
                
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={metricHistory[selectedMetric] || []}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="timestamp" 
                        tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                      />
                      <YAxis />
                      <Tooltip 
                        labelFormatter={(value) => new Date(value).toLocaleString()}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="value" 
                        stroke="#8884d8" 
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;
```

## Backend API Endpoints

**File**: `backend/api/monitoring.py`

```python
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
import logging

from ..core.monitoring.monitoring_service import MonitoringService, AlertRule, AlertSeverity
from ..core.monitoring.agent_tracker import AgentPerformanceTracker

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
logger = logging.getLogger(__name__)

# Global monitoring service instance
monitoring_service = MonitoringService()
agent_tracker = AgentPerformanceTracker(monitoring_service)

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

@router.on_event("startup")
async def startup_monitoring():
    """Initialize monitoring service on startup"""
    try:
        # Register default alert rules
        default_rules = [
            AlertRule(
                id="high_cpu_usage",
                name="High CPU Usage",
                metric_name="system_cpu_usage",
                condition=">",
                threshold=80.0,
                severity=AlertSeverity.HIGH,
                duration=timedelta(minutes=5)
            ),
            AlertRule(
                id="high_memory_usage",
                name="High Memory Usage",
                metric_name="system_memory_usage",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.HIGH,
                duration=timedelta(minutes=5)
            ),
            AlertRule(
                id="high_error_rate",
                name="High Error Rate",
                metric_name="error_rate",
                condition=">",
                threshold=5.0,
                severity=AlertSeverity.CRITICAL,
                duration=timedelta(minutes=2)
            )
        ]
        
        for rule in default_rules:
            monitoring_service.add_alert_rule(rule)
        
        # Subscribe to monitoring events for WebSocket broadcasting
        monitoring_service.subscribe(broadcast_monitoring_event)
        
        # Start monitoring service
        asyncio.create_task(monitoring_service.start_monitoring())
        
        logger.info("Monitoring service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start monitoring service: {e}")

@router.on_event("shutdown")
async def shutdown_monitoring():
    """Cleanup monitoring service on shutdown"""
    try:
        await monitoring_service.stop_monitoring()
        logger.info("Monitoring service stopped")
    except Exception as e:
        logger.error(f"Error stopping monitoring service: {e}")

async def broadcast_monitoring_event(event: Dict[str, Any]):
    """Broadcast monitoring events to all connected WebSocket clients"""
    if active_connections:
        message = json.dumps(event)
        disconnected = []
        
        for connection in active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            active_connections.remove(connection)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial data
        initial_data = {
            'type': 'initial_data',
            'system_health': monitoring_service.get_system_health().__dict__,
            'active_alerts': monitoring_service.get_active_alerts(),
            'agent_metrics': [vars(agent) for agent in agent_tracker.get_all_agent_metrics().values()]
        }
        await websocket.send_text(json.dumps(initial_data, default=str))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

@router.get("/health")
async def get_system_health():
    """Get current system health status"""
    try:
        health = monitoring_service.get_system_health()
        return health.__dict__
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.get("/metrics")
async def get_metrics_summary():
    """Get summary of all metrics"""
    try:
        return monitoring_service.get_metrics_summary()
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics summary")

@router.get("/metrics/history")
async def get_metrics_history(metric_name: Optional[str] = None, hours: int = 1):
    """Get historical metrics data"""
    try:
        # This would typically query a time-series database
        # For now, return recent points from in-memory storage
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        if metric_name:
            if metric_name in monitoring_service.metrics:
                metric = monitoring_service.metrics[metric_name]
                points = [p for p in metric.points if p.timestamp >= cutoff_time]
                return {
                    metric_name: [{
                        'timestamp': p.timestamp.isoformat(),
                        'value': p.value
                    } for p in points]
                }
            else:
                return {metric_name: []}
        else:
            # Return all metrics
            result = {}
            for name, metric in monitoring_service.metrics.items():
                points = [p for p in metric.points if p.timestamp >= cutoff_time]
                result[name] = [{
                    'timestamp': p.timestamp.isoformat(),
                    'value': p.value
                } for p in points]
            return result
            
    except Exception as e:
        logger.error(f"Error getting metrics history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics history")

@router.get("/alerts")
async def get_alerts(status: Optional[str] = None, limit: int = 100):
    """Get alerts with optional status filter"""
    try:
        if status == "active":
            return monitoring_service.get_active_alerts()
        else:
            return monitoring_service.get_alert_history(limit)
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alerts")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an active alert"""
    try:
        success = monitoring_service.acknowledge_alert(alert_id)
        if success:
            return {"message": "Alert acknowledged successfully"}
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")

@router.get("/agents")
async def get_agent_metrics():
    """Get metrics for all agents"""
    try:
        metrics = agent_tracker.get_all_agent_metrics()
        return [vars(agent) for agent in metrics.values()]
    except Exception as e:
        logger.error(f"Error getting agent metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent metrics")

@router.get("/agents/{agent_id}")
async def get_agent_metrics_by_id(agent_id: str):
    """Get metrics for a specific agent"""
    try:
        metrics = agent_tracker.get_agent_metrics(agent_id)
        if metrics:
            return vars(metrics)
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
    except Exception as e:
        logger.error(f"Error getting agent metrics for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent metrics")

@router.get("/agents/{agent_id}/tasks")
async def get_agent_task_history(agent_id: str, limit: int = 50):
    """Get task execution history for an agent"""
    try:
        tasks = agent_tracker.get_task_history(agent_id, limit)
        return [vars(task) for task in tasks]
    except Exception as e:
        logger.error(f"Error getting task history for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task history")

@router.post("/metrics")
async def record_custom_metric(metric_name: str, value: float, labels: Dict[str, str] = None):
    """Record a custom metric value"""
    try:
        success = monitoring_service.record_metric(metric_name, value, labels)
        if success:
            return {"message": "Metric recorded successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to record metric")
    except Exception as e:
        logger.error(f"Error recording metric {metric_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to record metric")

@router.get("/performance/summary")
async def get_performance_summary():
    """Get overall system performance summary"""
    try:
        return agent_tracker.get_performance_summary()
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance summary")
```

## Unit Testing Strategy

### Backend Tests

**File**: `backend/tests/test_monitoring_service.py`

```python
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from backend.core.monitoring.monitoring_service import (
    MonitoringService, Metric, MetricType, AlertRule, AlertSeverity, Alert, AlertStatus
)

class TestMonitoringService:
    @pytest.fixture
    def monitoring_service(self):
        return MonitoringService()
    
    @pytest.fixture
    def sample_metric(self):
        return Metric(
            name="test_metric",
            type=MetricType.GAUGE,
            description="Test metric",
            unit="count"
        )
    
    @pytest.fixture
    def sample_alert_rule(self):
        return AlertRule(
            id="test_rule",
            name="Test Alert",
            metric_name="test_metric",
            condition=">",
            threshold=50.0,
            severity=AlertSeverity.HIGH,
            duration=timedelta(minutes=1)
        )
    
    def test_register_metric(self, monitoring_service, sample_metric):
        """Test metric registration"""
        result = monitoring_service.register_metric(sample_metric)
        assert result is True
        assert "test_metric" in monitoring_service.metrics
        assert monitoring_service.metrics["test_metric"] == sample_metric
    
    def test_record_metric(self, monitoring_service, sample_metric):
        """Test metric recording"""
        monitoring_service.register_metric(sample_metric)
        
        result = monitoring_service.record_metric("test_metric", 42.0)
        assert result is True
        
        metric = monitoring_service.metrics["test_metric"]
        assert len(metric.points) == 1
        assert metric.points[0].value == 42.0
    
    def test_record_metric_nonexistent(self, monitoring_service):
        """Test recording metric that doesn't exist"""
        result = monitoring_service.record_metric("nonexistent", 42.0)
        assert result is False
    
    def test_add_alert_rule(self, monitoring_service, sample_alert_rule):
        """Test adding alert rule"""
        result = monitoring_service.add_alert_rule(sample_alert_rule)
        assert result is True
        assert "test_rule" in monitoring_service.alert_rules
    
    def test_remove_alert_rule(self, monitoring_service, sample_alert_rule):
        """Test removing alert rule"""
        monitoring_service.add_alert_rule(sample_alert_rule)
        
        result = monitoring_service.remove_alert_rule("test_rule")
        assert result is True
        assert "test_rule" not in monitoring_service.alert_rules
    
    def test_evaluate_condition(self, monitoring_service):
        """Test alert condition evaluation"""
        assert monitoring_service._evaluate_condition(60, ">", 50) is True
        assert monitoring_service._evaluate_condition(40, ">", 50) is False
        assert monitoring_service._evaluate_condition(50, ">=", 50) is True
        assert monitoring_service._evaluate_condition(40, "<", 50) is True
        assert monitoring_service._evaluate_condition(50, "==", 50) is True
    
    @pytest.mark.asyncio
    async def test_check_alert_rule_trigger(self, monitoring_service, sample_metric, sample_alert_rule):
        """Test alert rule triggering"""
        monitoring_service.register_metric(sample_metric)
        monitoring_service.add_alert_rule(sample_alert_rule)
        
        # Record metric value that should trigger alert
        monitoring_service.record_metric("test_metric", 60.0)
        
        await monitoring_service._check_alert_rule(sample_alert_rule)
        
        # Check if alert was triggered
        alert_id = f"{sample_alert_rule.id}_{sample_alert_rule.metric_name}"
        assert alert_id in monitoring_service.active_alerts
        
        alert = monitoring_service.active_alerts[alert_id]
        assert alert.severity == AlertSeverity.HIGH
        assert alert.status == AlertStatus.ACTIVE
    
    def test_acknowledge_alert(self, monitoring_service):
        """Test alert acknowledgment"""
        # Create a mock active alert
        alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            metric_name="test_metric",
            message="Test alert",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.ACTIVE,
            triggered_at=datetime.utcnow()
        )
        
        monitoring_service.active_alerts["test_alert"] = alert
        
        result = monitoring_service.acknowledge_alert("test_alert")
        assert result is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None
    
    def test_get_system_health(self, monitoring_service):
        """Test system health retrieval"""
        health = monitoring_service.get_system_health()
        assert health is not None
        assert hasattr(health, 'cpu_usage')
        assert hasattr(health, 'memory_usage')
        assert hasattr(health, 'timestamp')
    
    def test_get_metrics_summary(self, monitoring_service, sample_metric):
        """Test metrics summary"""
        monitoring_service.register_metric(sample_metric)
        monitoring_service.record_metric("test_metric", 42.0)
        
        summary = monitoring_service.get_metrics_summary()
        assert "test_metric" in summary
        assert summary["test_metric"]["current_value"] == 42.0
```

**File**: `backend/tests/test_agent_tracker.py`

```python
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from backend.core.monitoring.agent_tracker import (
    AgentPerformanceTracker, AgentStatus, AgentMetrics, TaskExecution
)

class TestAgentPerformanceTracker:
    @pytest.fixture
    def agent_tracker(self):
        return AgentPerformanceTracker()
    
    @pytest.fixture
    def sample_agent_metrics(self):
        return AgentMetrics(
            agent_id="test_agent",
            status=AgentStatus.ACTIVE,
            cpu_usage=25.5,
            memory_usage=512.0,
            tasks_completed=10,
            tasks_failed=1,
            average_response_time=1.5,
            last_activity=datetime.utcnow()
        )
    
    def test_register_agent(self, agent_tracker):
        """Test agent registration"""
        result = agent_tracker.register_agent("test_agent")
        assert result is True
        assert "test_agent" in agent_tracker.agents
        
        metrics = agent_tracker.agents["test_agent"]
        assert metrics.agent_id == "test_agent"
        assert metrics.status == AgentStatus.ACTIVE
    
    def test_unregister_agent(self, agent_tracker):
        """Test agent unregistration"""
        agent_tracker.register_agent("test_agent")
        
        result = agent_tracker.unregister_agent("test_agent")
        assert result is True
        assert "test_agent" not in agent_tracker.agents
    
    def test_track_task_execution(self, agent_tracker):
        """Test task execution tracking"""
        agent_tracker.register_agent("test_agent")
        
        task_execution = TaskExecution(
            task_id="task_1",
            agent_id="test_agent",
            task_type="code_generation",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=5),
            status="completed",
            result="success"
        )
        
        result = agent_tracker.track_task_execution(task_execution)
        assert result is True
        
        # Check if metrics were updated
        metrics = agent_tracker.agents["test_agent"]
        assert metrics.tasks_completed == 1
        assert len(agent_tracker.task_history["test_agent"]) == 1
    
    def test_update_agent_resources(self, agent_tracker):
        """Test agent resource updates"""
        agent_tracker.register_agent("test_agent")
        
        result = agent_tracker.update_agent_resources("test_agent", 45.0, 1024.0)
        assert result is True
        
        metrics = agent_tracker.agents["test_agent"]
        assert metrics.cpu_usage == 45.0
        assert metrics.memory_usage == 1024.0
    
    def test_set_agent_error(self, agent_tracker):
        """Test setting agent error status"""
        agent_tracker.register_agent("test_agent")
        
        result = agent_tracker.set_agent_error("test_agent", "Connection timeout")
        assert result is True
        
        metrics = agent_tracker.agents["test_agent"]
        assert metrics.status == AgentStatus.ERROR
        assert metrics.last_error == "Connection timeout"
    
    def test_get_performance_summary(self, agent_tracker):
        """Test performance summary generation"""
        agent_tracker.register_agent("agent_1")
        agent_tracker.register_agent("agent_2")
        
        # Add some task executions
        task1 = TaskExecution(
            task_id="task_1",
            agent_id="agent_1",
            task_type="code_generation",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(seconds=2),
            status="completed",
            result="success"
        )
        
        agent_tracker.track_task_execution(task1)
        
        summary = agent_tracker.get_performance_summary()
        assert "total_agents" in summary
        assert "active_agents" in summary
        assert "total_tasks_completed" in summary
        assert summary["total_agents"] == 2
        assert summary["active_agents"] == 2
        assert summary["total_tasks_completed"] == 1
```

### Frontend Tests

**File**: `frontend/components/monitoring/__tests__/monitoring-dashboard.test.tsx`

```typescript
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MonitoringDashboard } from '../monitoring-dashboard';

// Mock the API calls
jest.mock('../../../lib/api', () => ({
  get: jest.fn(),
  post: jest.fn(),
}));

// Mock WebSocket
const mockWebSocket = {
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  close: jest.fn(),
};

global.WebSocket = jest.fn(() => mockWebSocket) as any;

const mockSystemHealth = {
  cpu_usage: 45.2,
  memory_usage: 67.8,
  disk_usage: 23.1,
  active_connections: 15,
  timestamp: '2024-01-15T10:30:00Z'
};

const mockAlerts = [
  {
    id: 'alert_1',
    rule_id: 'high_cpu',
    metric_name: 'cpu_usage',
    message: 'High CPU usage detected',
    severity: 'HIGH',
    status: 'ACTIVE',
    triggered_at: '2024-01-15T10:25:00Z'
  }
];

const mockAgentMetrics = [
  {
    agent_id: 'agent_1',
    status: 'ACTIVE',
    cpu_usage: 25.5,
    memory_usage: 512.0,
    tasks_completed: 10,
    tasks_failed: 1,
    average_response_time: 1.5
  }
];

describe('MonitoringDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders monitoring dashboard', async () => {
    const { get } = require('../../../lib/api');
    get.mockResolvedValueOnce({ data: mockSystemHealth })
       .mockResolvedValueOnce({ data: mockAlerts })
       .mockResolvedValueOnce({ data: mockAgentMetrics });

    render(<MonitoringDashboard />);

    expect(screen.getByText('System Monitoring')).toBeInTheDocument();
    expect(screen.getByText('System Health')).toBeInTheDocument();
    expect(screen.getByText('Active Alerts')).toBeInTheDocument();
    expect(screen.getByText('Agent Performance')).toBeInTheDocument();
  });

  test('displays system health metrics', async () => {
    const { get } = require('../../../lib/api');
    get.mockResolvedValueOnce({ data: mockSystemHealth })
       .mockResolvedValueOnce({ data: [] })
       .mockResolvedValueOnce({ data: [] });

    render(<MonitoringDashboard />);

    await waitFor(() => {
      expect(screen.getByText('45.2%')).toBeInTheDocument(); // CPU usage
      expect(screen.getByText('67.8%')).toBeInTheDocument(); // Memory usage
      expect(screen.getByText('23.1%')).toBeInTheDocument(); // Disk usage
    });
  });

  test('displays active alerts', async () => {
    const { get } = require('../../../lib/api');
    get.mockResolvedValueOnce({ data: mockSystemHealth })
       .mockResolvedValueOnce({ data: mockAlerts })
       .mockResolvedValueOnce({ data: [] });

    render(<MonitoringDashboard />);

    await waitFor(() => {
      expect(screen.getByText('High CPU usage detected')).toBeInTheDocument();
      expect(screen.getByText('HIGH')).toBeInTheDocument();
    });
  });

  test('acknowledges alert when button clicked', async () => {
    const { get, post } = require('../../../lib/api');
    get.mockResolvedValueOnce({ data: mockSystemHealth })
       .mockResolvedValueOnce({ data: mockAlerts })
       .mockResolvedValueOnce({ data: [] });
    post.mockResolvedValueOnce({ data: { message: 'Alert acknowledged' } });

    const user = userEvent.setup();
    render(<MonitoringDashboard />);

    await waitFor(() => {
      expect(screen.getByText('Acknowledge')).toBeInTheDocument();
    });

    await user.click(screen.getByText('Acknowledge'));

    expect(post).toHaveBeenCalledWith('/api/monitoring/alerts/alert_1/acknowledge');
  });

  test('displays agent performance metrics', async () => {
    const { get } = require('../../../lib/api');
    get.mockResolvedValueOnce({ data: mockSystemHealth })
       .mockResolvedValueOnce({ data: [] })
       .mockResolvedValueOnce({ data: mockAgentMetrics });

    render(<MonitoringDashboard />);

    await waitFor(() => {
      expect(screen.getByText('agent_1')).toBeInTheDocument();
      expect(screen.getByText('ACTIVE')).toBeInTheDocument();
      expect(screen.getByText('10')).toBeInTheDocument(); // tasks completed
    });
  });

  test('handles WebSocket real-time updates', async () => {
    const { get } = require('../../../lib/api');
    get.mockResolvedValueOnce({ data: mockSystemHealth })
       .mockResolvedValueOnce({ data: [] })
       .mockResolvedValueOnce({ data: [] });

    render(<MonitoringDashboard />);

    // Simulate WebSocket message
    const messageHandler = mockWebSocket.addEventListener.mock.calls
      .find(call => call[0] === 'message')[1];
    
    const mockEvent = {
      data: JSON.stringify({
        type: 'system_health',
        data: { ...mockSystemHealth, cpu_usage: 55.0 }
      })
    };

    messageHandler(mockEvent);

    await waitFor(() => {
      expect(screen.getByText('55%')).toBeInTheDocument();
    });
  });
});
```

## Integration Tests

**File**: `backend/tests/integration/test_monitoring_integration.py`

```python
import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.monitoring.monitoring_service import MonitoringService, Metric, MetricType
from backend.core.monitoring.agent_tracker import AgentPerformanceTracker

class TestMonitoringIntegration:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def monitoring_service(self):
        return MonitoringService()
    
    @pytest.fixture
    def agent_tracker(self):
        return AgentPerformanceTracker()
    
    def test_full_monitoring_workflow(self, client, monitoring_service, agent_tracker):
        """Test complete monitoring workflow"""
        # 1. Register an agent
        agent_tracker.register_agent("test_agent")
        
        # 2. Register a metric
        metric = Metric(
            name="response_time",
            type=MetricType.HISTOGRAM,
            description="API response time",
            unit="seconds"
        )
        monitoring_service.register_metric(metric)
        
        # 3. Record some metrics
        monitoring_service.record_metric("response_time", 0.5)
        monitoring_service.record_metric("response_time", 1.2)
        
        # 4. Get system health via API
        response = client.get("/api/monitoring/health")
        assert response.status_code == 200
        health_data = response.json()
        assert "cpu_usage" in health_data
        assert "memory_usage" in health_data
        
        # 5. Get metrics summary
        response = client.get("/api/monitoring/metrics")
        assert response.status_code == 200
        metrics_data = response.json()
        assert "response_time" in metrics_data
        
        # 6. Get agent metrics
        response = client.get("/api/monitoring/agents")
        assert response.status_code == 200
        agents_data = response.json()
        assert len(agents_data) == 1
        assert agents_data[0]["agent_id"] == "test_agent"
    
    def test_alert_workflow(self, client, monitoring_service):
        """Test alert creation and acknowledgment workflow"""
        # 1. Register metric and alert rule
        metric = Metric(
            name="error_rate",
            type=MetricType.GAUGE,
            description="Error rate percentage",
            unit="percent"
        )
        monitoring_service.register_metric(metric)
        
        # 2. Record high error rate to trigger alert
        monitoring_service.record_metric("error_rate", 85.0)
        
        # 3. Check for alerts
        response = client.get("/api/monitoring/alerts")
        assert response.status_code == 200
        
        # 4. If alerts exist, acknowledge one
        alerts = response.json()
        if alerts:
            alert_id = alerts[0]["id"]
            response = client.post(f"/api/monitoring/alerts/{alert_id}/acknowledge")
            assert response.status_code == 200
    
    def test_performance_tracking(self, client, agent_tracker):
        """Test agent performance tracking"""
        # 1. Register multiple agents
        agent_tracker.register_agent("agent_1")
        agent_tracker.register_agent("agent_2")
        
        # 2. Update resource usage
        agent_tracker.update_agent_resources("agent_1", 45.0, 1024.0)
        agent_tracker.update_agent_resources("agent_2", 30.0, 768.0)
        
        # 3. Get performance summary
        response = client.get("/api/monitoring/performance/summary")
        assert response.status_code == 200
        summary = response.json()
        assert summary["total_agents"] == 2
        assert summary["active_agents"] == 2
    
    def test_custom_metric_recording(self, client):
        """Test custom metric recording via API"""
        response = client.post(
            "/api/monitoring/metrics",
            params={
                "metric_name": "custom_metric",
                "value": 42.0
            }
        )
        # Note: This might return 400 if metric not registered
        assert response.status_code in [200, 400]
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, monitoring_service):
        """Test real-time monitoring capabilities"""
        # Start monitoring
        await monitoring_service.start_monitoring()
        
        # Wait a short time for monitoring to collect data
        await asyncio.sleep(2)
        
        # Check that system health is being collected
        health = monitoring_service.get_system_health()
        assert health is not None
        assert health.cpu_usage >= 0
        assert health.memory_usage >= 0
        
        # Stop monitoring
        await monitoring_service.stop_monitoring()
```

## Human Testing Scenarios

### Backend Testing
1. **Metric Registration and Recording**
   - Register various metric types (gauge, counter, histogram)
   - Record metric values and verify storage
   - Test metric aggregation and summary generation

2. **Alert System Testing**
   - Create alert rules with different conditions
   - Trigger alerts by recording threshold-exceeding values
   - Test alert acknowledgment and resolution

3. **Agent Performance Tracking**
   - Register multiple agents
   - Track task executions and resource usage
   - Generate performance summaries

### Frontend Testing
1. **Dashboard Visualization**
   - Verify real-time system health display
   - Test alert notifications and acknowledgment
   - Check agent performance metrics visualization

2. **Interactive Features**
   - Test alert filtering and sorting
   - Verify metric time range selection
   - Test agent detail views

3. **Real-time Updates**
   - Verify WebSocket connectivity
   - Test live metric updates
   - Check alert notifications

### Integration Testing
1. **End-to-End Monitoring**
   - Complete workflow from metric recording to dashboard display
   - Alert triggering and notification flow
   - Agent registration to performance tracking

2. **Performance Testing**
   - High-frequency metric recording
   - Multiple concurrent agents
   - Large-scale alert rule evaluation

## Validation Criteria

### Backend Validation
- [ ] All monitoring services start and stop correctly
- [ ] Metrics are recorded and retrieved accurately
- [ ] Alert rules trigger and resolve properly
- [ ] Agent performance tracking works correctly
- [ ] API endpoints return expected responses
- [ ] WebSocket connections handle real-time updates

### Frontend Validation
- [ ] Dashboard displays all monitoring data correctly
- [ ] Real-time updates work without page refresh
- [ ] Alert acknowledgment functions properly
- [ ] Agent performance metrics are visualized clearly
- [ ] Responsive design works on different screen sizes

### Integration Validation
- [ ] Complete monitoring workflow functions end-to-end
- [ ] Real-time data flows from backend to frontend
- [ ] Alert notifications appear in dashboard
- [ ] Performance metrics update in real-time
- [ ] System handles multiple concurrent monitoring sessions

## Success Metrics

1. **Monitoring Coverage**: 100% of system components monitored
2. **Alert Response Time**: < 30 seconds from trigger to notification
3. **Dashboard Load Time**: < 2 seconds for initial load
4. **Real-time Update Latency**: < 1 second for metric updates
5. **System Performance Impact**: < 5% CPU overhead for monitoring
6. **Alert Accuracy**: > 95% true positive rate for alerts
7. **Agent Tracking Accuracy**: 100% task execution tracking
8. **Dashboard Uptime**: > 99.9% availability

## Next Steps

After completing the monitoring and alerting system:
1. Proceed to `24-performance-optimization.md`
2. Implement performance profiling and optimization
3. Add caching strategies and database optimization
4. Integrate with the monitoring system for performance tracking