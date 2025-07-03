# 17 - Performance Optimization Framework

## Overview

The Performance Optimization Framework provides comprehensive monitoring, analysis, and optimization capabilities for the multi-agent system. This includes real-time performance metrics collection, bottleneck identification, automatic optimization suggestions, resource usage monitoring, and performance-based scaling decisions.

## Current State Analysis

### Performance Requirements
- Real-time performance metrics collection
- Agent performance profiling and analysis
- Resource usage monitoring (CPU, memory, I/O)
- Bottleneck identification and resolution
- Automatic performance optimization
- Performance-based scaling recommendations
- Historical performance trend analysis
- Performance alerting and notifications

### Integration Points
- Agent framework performance hooks
- Workflow engine performance tracking
- Database query optimization
- API response time monitoring
- Frontend performance metrics
- External service performance tracking

## Implementation Tasks

### Task 17.1: Performance Monitoring Engine

**File**: `core/performance/monitoring_engine.py`

**Performance Monitoring Engine**:
```python
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import asyncio
import time
import psutil
import threading
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import defaultdict, deque
import json
import uuid

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PerformanceMetric:
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""

@dataclass
class PerformanceAlert:
    id: str
    metric_name: str
    severity: AlertSeverity
    threshold: float
    current_value: float
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class AgentPerformanceProfile:
    agent_id: str
    agent_type: str
    task_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_activity: Optional[datetime] = None
    performance_score: float = 1.0
    bottlenecks: List[str] = field(default_factory=list)

@dataclass
class SystemPerformanceSnapshot:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_agents: int
    active_workflows: int
    queue_sizes: Dict[str, int]
    response_times: Dict[str, float]
    error_rates: Dict[str, float]

class PerformanceMonitoringEngine:
    def __init__(self, state_service, memory_service, alert_service):
        self.state_service = state_service
        self.memory_service = memory_service
        self.alert_service = alert_service
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.system_snapshots: deque = deque(maxlen=1000)
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Monitoring configuration
        self.monitoring_interval = 5  # seconds
        self.metric_retention_hours = 24
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Performance hooks
        self.performance_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Optimization engine
        self.optimization_rules: List[Callable] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize performance monitoring"""
        await self._setup_default_thresholds()
        await self._setup_optimization_rules()
        await self._load_historical_data()
        self.start_monitoring()
        
    def start_monitoring(self):
        """Start performance monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                asyncio.run(self._collect_system_metrics())
                
                # Update agent profiles
                asyncio.run(self._update_agent_profiles())
                
                # Check alert conditions
                asyncio.run(self._check_alert_conditions())
                
                # Run optimization analysis
                asyncio.run(self._run_optimization_analysis())
                
                # Clean old metrics
                self._cleanup_old_metrics()
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
            
            time.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric("system.cpu.usage", cpu_percent, MetricType.GAUGE, unit="%")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_metric("system.memory.usage", memory.percent, MetricType.GAUGE, unit="%")
            await self.record_metric("system.memory.available", memory.available / (1024**3), MetricType.GAUGE, unit="GB")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self.record_metric("system.disk.usage", disk.percent, MetricType.GAUGE, unit="%")
            
            # Network metrics
            network = psutil.net_io_counters()
            await self.record_metric("system.network.bytes_sent", network.bytes_sent, MetricType.COUNTER, unit="bytes")
            await self.record_metric("system.network.bytes_recv", network.bytes_recv, MetricType.COUNTER, unit="bytes")
            
            # Create system snapshot
            snapshot = SystemPerformanceSnapshot(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                active_agents=len(self.agent_profiles),
                active_workflows=await self._get_active_workflow_count(),
                queue_sizes=await self._get_queue_sizes(),
                response_times=await self._get_average_response_times(),
                error_rates=await self._get_error_rates()
            )
            
            self.system_snapshots.append(snapshot)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    async def record_metric(self, name: str, value: float, metric_type: MetricType, 
                          tags: Dict[str, str] = None, unit: str = "", description: str = ""):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            unit=unit,
            description=description
        )
        
        self.metrics[name].append(metric)
        
        # Store in state service for real-time access
        await self.state_service.update_performance_metric(name, {
            "value": value,
            "timestamp": metric.timestamp.isoformat(),
            "type": metric_type.value,
            "tags": tags or {},
            "unit": unit
        })
        
        # Trigger performance hooks
        for hook in self.performance_hooks[name]:
            try:
                await hook(metric)
            except Exception as e:
                print(f"Error in performance hook for {name}: {e}")
    
    async def record_agent_performance(self, agent_id: str, agent_type: str, 
                                     execution_time: float, success: bool, 
                                     memory_usage: float = 0, cpu_usage: float = 0):
        """Record agent performance metrics"""
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentPerformanceProfile(
                agent_id=agent_id,
                agent_type=agent_type
            )
        
        profile = self.agent_profiles[agent_id]
        profile.task_count += 1
        profile.total_execution_time += execution_time
        profile.average_execution_time = profile.total_execution_time / profile.task_count
        profile.last_activity = datetime.utcnow()
        
        if success:
            profile.success_rate = (profile.success_rate * (profile.task_count - 1) + 1) / profile.task_count
        else:
            profile.error_count += 1
            profile.success_rate = (profile.success_rate * (profile.task_count - 1)) / profile.task_count
        
        if memory_usage > 0:
            profile.memory_usage = memory_usage
        if cpu_usage > 0:
            profile.cpu_usage = cpu_usage
        
        # Calculate performance score
        profile.performance_score = self._calculate_performance_score(profile)
        
        # Record metrics
        await self.record_metric(f"agent.{agent_type}.execution_time", execution_time, MetricType.TIMER, 
                               {"agent_id": agent_id}, "seconds")
        await self.record_metric(f"agent.{agent_type}.success_rate", profile.success_rate, MetricType.GAUGE, 
                               {"agent_id": agent_id}, "%")
        await self.record_metric(f"agent.{agent_type}.performance_score", profile.performance_score, MetricType.GAUGE, 
                               {"agent_id": agent_id})
    
    def _calculate_performance_score(self, profile: AgentPerformanceProfile) -> float:
        """Calculate overall performance score for an agent"""
        # Weighted scoring based on multiple factors
        success_weight = 0.4
        speed_weight = 0.3
        resource_weight = 0.3
        
        # Success rate component (0-1)
        success_score = profile.success_rate
        
        # Speed component (inverse of execution time, normalized)
        if profile.average_execution_time > 0:
            # Normalize to 0-1 range (assuming 10 seconds is baseline)
            speed_score = max(0, 1 - (profile.average_execution_time / 10))
        else:
            speed_score = 1.0
        
        # Resource efficiency component
        memory_score = max(0, 1 - (profile.memory_usage / 1000))  # Assuming 1GB baseline
        cpu_score = max(0, 1 - (profile.cpu_usage / 100))
        resource_score = (memory_score + cpu_score) / 2
        
        # Calculate weighted score
        total_score = (
            success_score * success_weight +
            speed_score * speed_weight +
            resource_score * resource_weight
        )
        
        return min(1.0, max(0.0, total_score))
    
    async def _check_alert_conditions(self):
        """Check for alert conditions"""
        for metric_name, thresholds in self.alert_thresholds.items():
            if metric_name in self.metrics and self.metrics[metric_name]:
                latest_metric = self.metrics[metric_name][-1]
                
                for severity_str, threshold in thresholds.items():
                    severity = AlertSeverity(severity_str)
                    
                    # Check if threshold is exceeded
                    if self._should_trigger_alert(latest_metric.value, threshold, metric_name):
                        alert_id = f"{metric_name}_{severity_str}"
                        
                        if alert_id not in self.active_alerts:
                            alert = PerformanceAlert(
                                id=alert_id,
                                metric_name=metric_name,
                                severity=severity,
                                threshold=threshold,
                                current_value=latest_metric.value,
                                message=f"{metric_name} exceeded {severity_str} threshold: {latest_metric.value} > {threshold}",
                                timestamp=datetime.utcnow()
                            )
                            
                            self.active_alerts[alert_id] = alert
                            await self.alert_service.send_alert(alert)
                    
                    else:
                        # Check if alert should be resolved
                        alert_id = f"{metric_name}_{severity_str}"
                        if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
                            self.active_alerts[alert_id].resolved = True
                            self.active_alerts[alert_id].resolution_time = datetime.utcnow()
                            await self.alert_service.resolve_alert(alert_id)
    
    def _should_trigger_alert(self, value: float, threshold: float, metric_name: str) -> bool:
        """Determine if an alert should be triggered"""
        # Different logic for different metric types
        if "usage" in metric_name or "error_rate" in metric_name:
            return value > threshold
        elif "response_time" in metric_name:
            return value > threshold
        elif "success_rate" in metric_name:
            return value < threshold
        else:
            return value > threshold
    
    async def _run_optimization_analysis(self):
        """Run optimization analysis and generate recommendations"""
        try:
            recommendations = []
            
            # Analyze agent performance
            agent_recommendations = await self._analyze_agent_performance()
            recommendations.extend(agent_recommendations)
            
            # Analyze system resources
            system_recommendations = await self._analyze_system_performance()
            recommendations.extend(system_recommendations)
            
            # Analyze workflow efficiency
            workflow_recommendations = await self._analyze_workflow_performance()
            recommendations.extend(workflow_recommendations)
            
            # Store recommendations
            if recommendations:
                optimization_result = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "recommendations": recommendations,
                    "system_state": await self._get_current_system_state()
                }
                
                self.optimization_history.append(optimization_result)
                
                # Store in memory service
                await self.memory_service.store_knowledge(
                    "performance_optimization",
                    optimization_result,
                    {"type": "optimization_analysis"}
                )
        
        except Exception as e:
            print(f"Error in optimization analysis: {e}")
    
    async def _analyze_agent_performance(self) -> List[Dict[str, Any]]:
        """Analyze agent performance and generate recommendations"""
        recommendations = []
        
        for agent_id, profile in self.agent_profiles.items():
            # Check for poor performance
            if profile.performance_score < 0.7:
                recommendations.append({
                    "type": "agent_optimization",
                    "priority": "high" if profile.performance_score < 0.5 else "medium",
                    "agent_id": agent_id,
                    "agent_type": profile.agent_type,
                    "issue": "low_performance_score",
                    "current_score": profile.performance_score,
                    "recommendations": self._get_agent_optimization_suggestions(profile)
                })
            
            # Check for high resource usage
            if profile.memory_usage > 500:  # 500MB
                recommendations.append({
                    "type": "resource_optimization",
                    "priority": "medium",
                    "agent_id": agent_id,
                    "agent_type": profile.agent_type,
                    "issue": "high_memory_usage",
                    "current_usage": profile.memory_usage,
                    "recommendations": ["Consider memory optimization", "Check for memory leaks"]
                })
            
            # Check for slow execution
            if profile.average_execution_time > 30:  # 30 seconds
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "medium",
                    "agent_id": agent_id,
                    "agent_type": profile.agent_type,
                    "issue": "slow_execution",
                    "current_time": profile.average_execution_time,
                    "recommendations": ["Optimize task processing", "Consider parallel processing"]
                })
        
        return recommendations
    
    def _get_agent_optimization_suggestions(self, profile: AgentPerformanceProfile) -> List[str]:
        """Get optimization suggestions for an agent"""
        suggestions = []
        
        if profile.success_rate < 0.8:
            suggestions.append("Improve error handling and retry logic")
            suggestions.append("Review task validation and input processing")
        
        if profile.average_execution_time > 20:
            suggestions.append("Optimize task processing algorithms")
            suggestions.append("Consider caching frequently used data")
        
        if profile.memory_usage > 300:
            suggestions.append("Implement memory cleanup routines")
            suggestions.append("Review data structures for memory efficiency")
        
        if profile.cpu_usage > 80:
            suggestions.append("Optimize CPU-intensive operations")
            suggestions.append("Consider asynchronous processing")
        
        return suggestions
    
    async def get_performance_summary(self, time_range_hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for specified time range"""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
        
        # Filter metrics by time range
        recent_metrics = {}
        for name, metric_list in self.metrics.items():
            recent_metrics[name] = [
                m for m in metric_list if m.timestamp >= cutoff_time
            ]
        
        # Calculate summary statistics
        summary = {
            "time_range_hours": time_range_hours,
            "timestamp": datetime.utcnow().isoformat(),
            "system_metrics": {},
            "agent_performance": {},
            "alerts": {
                "active": len([a for a in self.active_alerts.values() if not a.resolved]),
                "total": len(self.active_alerts)
            },
            "recommendations": len(self.optimization_history[-1]["recommendations"]) if self.optimization_history else 0
        }
        
        # System metrics summary
        for metric_name, metrics in recent_metrics.items():
            if metrics and metric_name.startswith("system."):
                values = [m.value for m in metrics]
                summary["system_metrics"][metric_name] = {
                    "current": values[-1] if values else 0,
                    "average": statistics.mean(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "count": len(values)
                }
        
        # Agent performance summary
        for agent_id, profile in self.agent_profiles.items():
            if profile.last_activity and profile.last_activity >= cutoff_time:
                summary["agent_performance"][agent_id] = {
                    "agent_type": profile.agent_type,
                    "performance_score": profile.performance_score,
                    "task_count": profile.task_count,
                    "success_rate": profile.success_rate,
                    "average_execution_time": profile.average_execution_time,
                    "memory_usage": profile.memory_usage,
                    "cpu_usage": profile.cpu_usage
                }
        
        return summary
    
    async def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get latest optimization recommendations"""
        if not self.optimization_history:
            return []
        
        return self.optimization_history[-1]["recommendations"]
    
    async def _setup_default_thresholds(self):
        """Setup default alert thresholds"""
        self.alert_thresholds = {
            "system.cpu.usage": {
                "medium": 70.0,
                "high": 85.0,
                "critical": 95.0
            },
            "system.memory.usage": {
                "medium": 75.0,
                "high": 90.0,
                "critical": 95.0
            },
            "system.disk.usage": {
                "medium": 80.0,
                "high": 90.0,
                "critical": 95.0
            }
        }
    
    def add_performance_hook(self, metric_name: str, hook: Callable):
        """Add a performance hook for a specific metric"""
        self.performance_hooks[metric_name].append(hook)
    
    def remove_performance_hook(self, metric_name: str, hook: Callable):
        """Remove a performance hook"""
        if hook in self.performance_hooks[metric_name]:
            self.performance_hooks[metric_name].remove(hook)
    
    async def _get_active_workflow_count(self) -> int:
        """Get count of active workflows"""
        # This would integrate with workflow engine
        return 0
    
    async def _get_queue_sizes(self) -> Dict[str, int]:
        """Get current queue sizes"""
        # This would integrate with task queues
        return {}
    
    async def _get_average_response_times(self) -> Dict[str, float]:
        """Get average response times for different endpoints"""
        # This would integrate with API monitoring
        return {}
    
    async def _get_error_rates(self) -> Dict[str, float]:
        """Get error rates for different components"""
        # This would integrate with error tracking
        return {}
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.metric_retention_hours)
        
        for metric_name, metric_list in self.metrics.items():
            # Remove old metrics
            while metric_list and metric_list[0].timestamp < cutoff_time:
                metric_list.popleft()
```

### Task 17.2: Performance Dashboard Component

**File**: `frontend/components/performance/performance-dashboard.tsx`

**Performance Dashboard Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Activity, Cpu, HardDrive, Wifi, AlertTriangle, TrendingUp, TrendingDown } from 'lucide-react';

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    bytes_sent: number;
    bytes_recv: number;
  };
}

interface AgentPerformance {
  agent_id: string;
  agent_type: string;
  performance_score: number;
  task_count: number;
  success_rate: number;
  average_execution_time: number;
  memory_usage: number;
  cpu_usage: number;
}

interface PerformanceAlert {
  id: string;
  metric_name: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: string;
  resolved: boolean;
}

interface OptimizationRecommendation {
  type: string;
  priority: 'low' | 'medium' | 'high';
  issue: string;
  recommendations: string[];
  agent_id?: string;
  agent_type?: string;
}

interface PerformanceSummary {
  system_metrics: Record<string, {
    current: number;
    average: number;
    min: number;
    max: number;
  }>;
  agent_performance: Record<string, AgentPerformance>;
  alerts: {
    active: number;
    total: number;
  };
  recommendations: number;
}

export const PerformanceDashboard: React.FC = () => {
  const [summary, setSummary] = useState<PerformanceSummary | null>(null);
  const [alerts, setAlerts] = useState<PerformanceAlert[]>([]);
  const [recommendations, setRecommendations] = useState<OptimizationRecommendation[]>([]);
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeRange, setSelectedTimeRange] = useState('1h');

  useEffect(() => {
    fetchPerformanceData();
    const interval = setInterval(fetchPerformanceData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, [selectedTimeRange]);

  const fetchPerformanceData = async () => {
    try {
      // Fetch performance summary
      const summaryResponse = await fetch(`/api/performance/summary?hours=${selectedTimeRange.replace('h', '')}`);
      if (!summaryResponse.ok) throw new Error('Failed to fetch performance summary');
      const summaryData = await summaryResponse.json();
      setSummary(summaryData);

      // Fetch alerts
      const alertsResponse = await fetch('/api/performance/alerts');
      if (!alertsResponse.ok) throw new Error('Failed to fetch alerts');
      const alertsData = await alertsResponse.json();
      setAlerts(alertsData.alerts || []);

      // Fetch recommendations
      const recommendationsResponse = await fetch('/api/performance/recommendations');
      if (!recommendationsResponse.ok) throw new Error('Failed to fetch recommendations');
      const recommendationsData = await recommendationsResponse.json();
      setRecommendations(recommendationsData.recommendations || []);

      // Fetch historical data
      const historicalResponse = await fetch(`/api/performance/historical?hours=${selectedTimeRange.replace('h', '')}`);
      if (!historicalResponse.ok) throw new Error('Failed to fetch historical data');
      const historicalData = await historicalResponse.json();
      setHistoricalData(historicalData.data || []);

      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const getMetricIcon = (metricName: string) => {
    if (metricName.includes('cpu')) return <Cpu className="w-4 h-4" />;
    if (metricName.includes('memory')) return <Activity className="w-4 h-4" />;
    if (metricName.includes('disk')) return <HardDrive className="w-4 h-4" />;
    if (metricName.includes('network')) return <Wifi className="w-4 h-4" />;
    return <Activity className="w-4 h-4" />;
  };

  const getAlertColor = (severity: string) => {
    const colors = {
      low: 'bg-blue-500',
      medium: 'bg-yellow-500',
      high: 'bg-orange-500',
      critical: 'bg-red-500'
    };
    return colors[severity] || 'bg-gray-500';
  };

  const getPriorityColor = (priority: string) => {
    const colors = {
      low: 'bg-green-500',
      medium: 'bg-yellow-500',
      high: 'bg-red-500'
    };
    return colors[priority] || 'bg-gray-500';
  };

  const formatBytes = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 B';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2">Loading performance data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>Error loading performance data: {error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Performance Dashboard</h1>
        <div className="flex items-center space-x-2">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-2 border rounded-md"
          >
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
          </select>
          <Button onClick={fetchPerformanceData} variant="outline" size="sm">
            Refresh
          </Button>
        </div>
      </div>

      {/* System Overview */}
      {summary && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.entries(summary.system_metrics).map(([metricName, data]) => {
            const shortName = metricName.split('.').pop() || metricName;
            return (
              <Card key={metricName}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium capitalize">
                    {shortName.replace('_', ' ')}
                  </CardTitle>
                  {getMetricIcon(metricName)}
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {data.current.toFixed(1)}
                    {metricName.includes('usage') ? '%' : ''}
                  </div>
                  <Progress 
                    value={metricName.includes('usage') ? data.current : (data.current / data.max) * 100} 
                    className="mt-2" 
                  />
                  <div className="text-xs text-gray-600 mt-1">
                    Avg: {data.average.toFixed(1)} | Max: {data.max.toFixed(1)}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      {/* Alerts Summary */}
      {alerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <AlertTriangle className="w-5 h-5 mr-2" />
              Active Alerts ({alerts.filter(a => !a.resolved).length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {alerts.filter(a => !a.resolved).slice(0, 5).map((alert) => (
                <div key={alert.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Badge className={getAlertColor(alert.severity)}>
                      {alert.severity.toUpperCase()}
                    </Badge>
                    <div>
                      <p className="font-medium">{alert.metric_name}</p>
                      <p className="text-sm text-gray-600">{alert.message}</p>
                    </div>
                  </div>
                  <div className="text-sm text-gray-500">
                    {new Date(alert.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList>
          <TabsTrigger value="metrics">System Metrics</TabsTrigger>
          <TabsTrigger value="agents">Agent Performance</TabsTrigger>
          <TabsTrigger value="recommendations">Optimization</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="space-y-4">
          {/* Historical Charts */}
          <Card>
            <CardHeader>
              <CardTitle>System Metrics Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={historicalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="cpu_usage" stroke="#8884d8" name="CPU %" />
                    <Line type="monotone" dataKey="memory_usage" stroke="#82ca9d" name="Memory %" />
                    <Line type="monotone" dataKey="disk_usage" stroke="#ffc658" name="Disk %" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="agents" className="space-y-4">
          {/* Agent Performance */}
          {summary && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {Object.entries(summary.agent_performance).map(([agentId, agent]) => (
                <Card key={agentId}>
                  <CardHeader>
                    <CardTitle className="text-lg">
                      {agent.agent_type}
                      <span className="text-sm text-gray-500 ml-2">({agentId.slice(0, 8)})</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span>Performance Score</span>
                        <div className="flex items-center space-x-2">
                          <Progress value={agent.performance_score * 100} className="w-20" />
                          <span className="text-sm font-medium">
                            {(agent.performance_score * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-gray-600">Tasks:</span>
                          <span className="ml-2 font-semibold">{agent.task_count}</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Success Rate:</span>
                          <span className="ml-2 font-semibold">{(agent.success_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Avg Time:</span>
                          <span className="ml-2 font-semibold">{agent.average_execution_time.toFixed(1)}s</span>
                        </div>
                        <div>
                          <span className="text-gray-600">Memory:</span>
                          <span className="ml-2 font-semibold">{formatBytes(agent.memory_usage * 1024 * 1024)}</span>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-4">
          {/* Optimization Recommendations */}
          <Card>
            <CardHeader>
              <CardTitle>Optimization Recommendations</CardTitle>
            </CardHeader>
            <CardContent>
              {recommendations.length === 0 ? (
                <div className="text-center py-8 text-gray-500">
                  <TrendingUp className="w-12 h-12 mx-auto mb-4" />
                  <p>No optimization recommendations at this time.</p>
                  <p className="text-sm">Your system is performing well!</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {recommendations.map((rec, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <Badge className={getPriorityColor(rec.priority)}>
                            {rec.priority.toUpperCase()}
                          </Badge>
                          <span className="font-medium">{rec.type.replace('_', ' ')}</span>
                          {rec.agent_type && (
                            <span className="text-sm text-gray-600">({rec.agent_type})</span>
                          )}
                        </div>
                      </div>
                      
                      <p className="text-sm text-gray-700 mb-2">
                        Issue: {rec.issue.replace('_', ' ')}
                      </p>
                      
                      <div className="space-y-1">
                        <p className="text-sm font-medium">Recommendations:</p>
                        <ul className="text-sm text-gray-600 list-disc list-inside space-y-1">
                          {rec.recommendations.map((suggestion, i) => (
                            <li key={i}>{suggestion}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          {/* All Alerts */}
          <Card>
            <CardHeader>
              <CardTitle>Alert History</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <div 
                    key={alert.id} 
                    className={`p-4 border rounded-lg ${
                      alert.resolved ? 'bg-gray-50' : 'bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Badge className={getAlertColor(alert.severity)}>
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <div>
                          <p className="font-medium">{alert.metric_name}</p>
                          <p className="text-sm text-gray-600">{alert.message}</p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-sm text-gray-500">
                          {new Date(alert.timestamp).toLocaleString()}
                        </div>
                        {alert.resolved && (
                          <Badge variant="outline" className="mt-1">
                            Resolved
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};
```

### Task 17.3: Performance API Endpoints

**File**: `app/api/performance.py`

**Performance API Endpoints**:
```python
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
from core.performance.monitoring_engine import PerformanceMonitoringEngine

router = APIRouter(prefix="/api/performance", tags=["performance"])

# Initialize performance engine (would be dependency injected)
performance_engine = None

class MetricRequest(BaseModel):
    name: str
    value: float
    metric_type: str
    tags: Dict[str, str] = {}
    unit: str = ""
    description: str = ""

class AgentPerformanceRequest(BaseModel):
    agent_id: str
    agent_type: str
    execution_time: float
    success: bool
    memory_usage: float = 0
    cpu_usage: float = 0

@router.get("/summary")
async def get_performance_summary(hours: int = Query(1, ge=1, le=168)):
    """Get performance summary for specified time range"""
    try:
        summary = await performance_engine.get_performance_summary(hours)
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_performance_alerts(active_only: bool = Query(False)):
    """Get performance alerts"""
    try:
        alerts = list(performance_engine.active_alerts.values())
        
        if active_only:
            alerts = [alert for alert in alerts if not alert.resolved]
        
        # Convert to dict format
        alert_data = []
        for alert in alerts:
            alert_data.append({
                "id": alert.id,
                "metric_name": alert.metric_name,
                "severity": alert.severity.value,
                "threshold": alert.threshold,
                "current_value": alert.current_value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolution_time": alert.resolution_time.isoformat() if alert.resolution_time else None
            })
        
        return {
            "success": True,
            "alerts": alert_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations")
async def get_optimization_recommendations():
    """Get optimization recommendations"""
    try:
        recommendations = await performance_engine.get_optimization_recommendations()
        return {
            "success": True,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/historical")
async def get_historical_data(hours: int = Query(1, ge=1, le=168)):
    """Get historical performance data"""
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        # Get system snapshots
        historical_data = []
        for snapshot in performance_engine.system_snapshots:
            if snapshot.timestamp >= cutoff_time:
                historical_data.append({
                    "timestamp": snapshot.timestamp.isoformat(),
                    "cpu_usage": snapshot.cpu_usage,
                    "memory_usage": snapshot.memory_usage,
                    "disk_usage": snapshot.disk_usage,
                    "active_agents": snapshot.active_agents,
                    "active_workflows": snapshot.active_workflows
                })
        
        return {
            "success": True,
            "data": historical_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics")
async def record_metric(request: MetricRequest):
    """Record a performance metric"""
    try:
        from core.performance.monitoring_engine import MetricType
        
        metric_type = MetricType(request.metric_type)
        await performance_engine.record_metric(
            request.name,
            request.value,
            metric_type,
            request.tags,
            request.unit,
            request.description
        )
        
        return {
            "success": True,
            "message": "Metric recorded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/agent-performance")
async def record_agent_performance(request: AgentPerformanceRequest):
    """Record agent performance metrics"""
    try:
        await performance_engine.record_agent_performance(
            request.agent_id,
            request.agent_type,
            request.execution_time,
            request.success,
            request.memory_usage,
            request.cpu_usage
        )
        
        return {
            "success": True,
            "message": "Agent performance recorded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/agents/{agent_id}")
async def get_agent_performance(agent_id: str):
    """Get performance data for specific agent"""
    try:
        if agent_id not in performance_engine.agent_profiles:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        profile = performance_engine.agent_profiles[agent_id]
        
        return {
            "success": True,
            "agent_performance": {
                "agent_id": profile.agent_id,
                "agent_type": profile.agent_type,
                "task_count": profile.task_count,
                "total_execution_time": profile.total_execution_time,
                "average_execution_time": profile.average_execution_time,
                "success_rate": profile.success_rate,
                "error_count": profile.error_count,
                "memory_usage": profile.memory_usage,
                "cpu_usage": profile.cpu_usage,
                "last_activity": profile.last_activity.isoformat() if profile.last_activity else None,
                "performance_score": profile.performance_score,
                "bottlenecks": profile.bottlenecks
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/status")
async def get_system_status():
    """Get current system status"""
    try:
        if not performance_engine.system_snapshots:
            raise HTTPException(status_code=404, detail="No system data available")
        
        latest_snapshot = performance_engine.system_snapshots[-1]
        
        return {
            "success": True,
            "system_status": {
                "timestamp": latest_snapshot.timestamp.isoformat(),
                "cpu_usage": latest_snapshot.cpu_usage,
                "memory_usage": latest_snapshot.memory_usage,
                "disk_usage": latest_snapshot.disk_usage,
                "network_io": latest_snapshot.network_io,
                "active_agents": latest_snapshot.active_agents,
                "active_workflows": latest_snapshot.active_workflows,
                "queue_sizes": latest_snapshot.queue_sizes,
                "response_times": latest_snapshot.response_times,
                "error_rates": latest_snapshot.error_rates
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/thresholds")
async def get_alert_thresholds():
    """Get current alert thresholds"""
    try:
        return {
            "success": True,
            "thresholds": performance_engine.alert_thresholds
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/thresholds")
async def update_alert_thresholds(thresholds: Dict[str, Dict[str, float]]):
    """Update alert thresholds"""
    try:
        performance_engine.alert_thresholds.update(thresholds)
        
        return {
            "success": True,
            "message": "Alert thresholds updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

## Testing Strategy

### Unit Tests
- Performance monitoring engine functionality
- Metric collection and storage
- Alert threshold evaluation
- Optimization recommendation generation

### Integration Tests
- End-to-end performance monitoring
- Dashboard data visualization
- Alert notification system
- API endpoint functionality

### Human Testing Scenarios
1. **Performance Monitoring**: Monitor system performance during normal operation
2. **Alert Testing**: Trigger performance alerts and verify notifications
3. **Optimization Recommendations**: Review and apply optimization suggestions
4. **Dashboard Interaction**: Navigate performance dashboard and analyze trends
5. **Threshold Configuration**: Adjust alert thresholds and test effectiveness

## Validation Criteria

### Backend Validation
- [ ] Performance metrics are collected accurately
- [ ] Alert system triggers appropriately
- [ ] Optimization recommendations are relevant
- [ ] Historical data is stored and retrievable
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Dashboard displays real-time performance data
- [ ] Charts and visualizations are accurate
- [ ] Alert notifications are visible
- [ ] Time range selection works
- [ ] Performance trends are clear

### Integration Validation
- [ ] Performance data flows correctly
- [ ] Alerts integrate with notification system
- [ ] Optimization recommendations are actionable
- [ ] System performance improves with recommendations

## Next Steps

After successful validation of the performance optimization framework, proceed to **18-advanced-ai-model-integration.md** for implementing enhanced AI model capabilities and multi-model orchestration.

---

**Dependencies**: This phase requires the core services, agent framework, and monitoring infrastructure to be functional.