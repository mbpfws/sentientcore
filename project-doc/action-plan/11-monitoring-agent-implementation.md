# 11 - Monitoring Agent Implementation

## Overview

The Monitoring Agent serves as the observability and health management hub for the entire multi-agent system. It provides real-time monitoring, performance tracking, error detection, system health assessment, and automated alerting capabilities. This agent ensures system reliability, performance optimization, and proactive issue resolution across all components.

## Current State Analysis

### Existing File
- `core/agents/monitoring_agent.py` - Basic monitoring functionality

### Enhancement Requirements
- Real-time system monitoring and health checks
- Performance metrics collection and analysis
- Error detection and automated alerting
- Resource utilization tracking
- Agent performance monitoring
- Workflow execution monitoring
- Automated recovery mechanisms
- Comprehensive logging and audit trails

## Implementation Tasks

### Task 11.1: Enhanced Monitoring Agent

**File**: `core/agents/monitoring_agent.py` (Complete Rewrite)

**Monitoring Agent Implementation**:
```python
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import json
import psutil
import logging
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentStatus
from ..services.llm_service import LLMService
from ..services.memory_service import MemoryService
from ..models import MonitoringTask, SystemMetrics, AlertRule, HealthCheck

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MonitoringScope(Enum):
    SYSTEM = "system"
    AGENTS = "agents"
    WORKFLOWS = "workflows"
    SERVICES = "services"
    PERFORMANCE = "performance"
    SECURITY = "security"

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_agents: int
    active_workflows: int
    error_count: int
    response_time: float

@dataclass
class AlertRule:
    name: str
    condition: str
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 5

class MonitoringAgent(BaseAgent):
    def __init__(self, agent_id: str = "monitoring_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Monitoring Agent",
            description="Advanced system monitoring and observability agent"
        )
        self.capabilities = [
            "system_monitoring",
            "performance_tracking",
            "error_detection",
            "health_assessment",
            "automated_alerting",
            "resource_monitoring",
            "workflow_monitoring",
            "security_monitoring",
            "log_analysis",
            "predictive_analysis"
        ]
        
        self.monitoring_sessions = {}
        self.alert_rules = []
        self.health_checks = []
        self.metrics_history = []
        self.active_alerts = {}
        self.monitoring_interval = 30  # seconds
        self.is_monitoring = False
        
    async def initialize(self, llm_service: LLMService, memory_service: MemoryService):
        """Initialize monitoring agent"""
        self.llm_service = llm_service
        self.memory_service = memory_service
        await self._setup_default_alert_rules()
        await self._setup_health_checks()
        await self.update_status(AgentStatus.IDLE)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process monitoring task"""
        try:
            await self.update_status(AgentStatus.THINKING)
            
            # Parse monitoring task
            monitoring_task = self._parse_monitoring_task(task)
            
            # Create monitoring session
            session_id = f"monitoring_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.monitoring_sessions[session_id] = {
                'task': monitoring_task,
                'start_time': datetime.utcnow(),
                'status': 'active',
                'metrics_collected': [],
                'alerts_triggered': []
            }
            
            await self.update_status(AgentStatus.WORKING)
            
            # Execute monitoring workflow
            monitoring_result = await self._execute_monitoring_workflow(session_id, monitoring_task)
            
            # Generate monitoring report
            monitoring_report = await self._generate_monitoring_report(session_id, monitoring_result)
            
            # Store monitoring data
            await self._store_monitoring_data(session_id, monitoring_result, monitoring_report)
            
            await self.update_status(AgentStatus.COMPLETED)
            
            return {
                'session_id': session_id,
                'monitoring_scope': monitoring_task.scope.value,
                'system_health': monitoring_result.get('system_health'),
                'performance_metrics': monitoring_result.get('performance_metrics'),
                'alerts_triggered': monitoring_result.get('alerts_triggered', []),
                'recommendations': monitoring_result.get('recommendations', []),
                'monitoring_report': monitoring_report,
                'confidence_score': monitoring_result.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            await self.update_status(AgentStatus.ERROR, str(e))
            raise
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Determine if agent can handle monitoring task"""
        return task.get('type') in [
            'system_monitoring',
            'performance_monitoring',
            'health_check',
            'error_monitoring',
            'workflow_monitoring',
            'agent_monitoring',
            'security_monitoring'
        ]
    
    async def start_continuous_monitoring(self):
        """Start continuous system monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        await self.log_activity("Starting continuous monitoring")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_continuous_monitoring(self):
        """Stop continuous system monitoring"""
        self.is_monitoring = False
        await self.log_activity("Stopping continuous monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check alert rules
                await self._check_alert_rules(metrics)
                
                # Perform health checks
                await self._perform_health_checks()
                
                # Store metrics
                await self._store_metrics(metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                await self.log_activity(f"Monitoring loop error: {str(e)}", "error")
                await asyncio.sleep(self.monitoring_interval)
    
    def _parse_monitoring_task(self, task: Dict[str, Any]) -> 'MonitoringTask':
        """Parse incoming task into structured monitoring task"""
        return MonitoringTask(
            scope=MonitoringScope(task.get('scope', 'system')),
            duration=task.get('duration', 3600),  # 1 hour default
            metrics_to_collect=task.get('metrics', []),
            alert_thresholds=task.get('alert_thresholds', {}),
            targets=task.get('targets', []),
            reporting_interval=task.get('reporting_interval', 300),  # 5 minutes
            custom_checks=task.get('custom_checks', [])
        )
    
    async def _execute_monitoring_workflow(self, session_id: str, monitoring_task: 'MonitoringTask') -> Dict[str, Any]:
        """Execute comprehensive monitoring workflow"""
        monitoring_result = {}
        session = self.monitoring_sessions[session_id]
        
        # Step 1: System Health Assessment
        await self.log_activity("Assessing system health")
        system_health = await self._assess_system_health(monitoring_task)
        monitoring_result['system_health'] = system_health
        
        # Step 2: Performance Metrics Collection
        await self.log_activity("Collecting performance metrics")
        performance_metrics = await self._collect_performance_metrics(monitoring_task)
        monitoring_result['performance_metrics'] = performance_metrics
        session['metrics_collected'].extend(performance_metrics)
        
        # Step 3: Agent Monitoring
        if monitoring_task.scope in [MonitoringScope.AGENTS, MonitoringScope.SYSTEM]:
            await self.log_activity("Monitoring agent performance")
            agent_metrics = await self._monitor_agents(monitoring_task)
            monitoring_result['agent_metrics'] = agent_metrics
        
        # Step 4: Workflow Monitoring
        if monitoring_task.scope in [MonitoringScope.WORKFLOWS, MonitoringScope.SYSTEM]:
            await self.log_activity("Monitoring workflow execution")
            workflow_metrics = await self._monitor_workflows(monitoring_task)
            monitoring_result['workflow_metrics'] = workflow_metrics
        
        # Step 5: Error Detection and Analysis
        await self.log_activity("Detecting and analyzing errors")
        error_analysis = await self._detect_and_analyze_errors(monitoring_task)
        monitoring_result['error_analysis'] = error_analysis
        
        # Step 6: Security Monitoring
        if monitoring_task.scope in [MonitoringScope.SECURITY, MonitoringScope.SYSTEM]:
            await self.log_activity("Performing security monitoring")
            security_metrics = await self._monitor_security(monitoring_task)
            monitoring_result['security_metrics'] = security_metrics
        
        # Step 7: Alert Processing
        await self.log_activity("Processing alerts")
        alerts_triggered = await self._process_alerts(monitoring_result)
        monitoring_result['alerts_triggered'] = alerts_triggered
        session['alerts_triggered'].extend(alerts_triggered)
        
        # Step 8: Predictive Analysis
        await self.log_activity("Performing predictive analysis")
        predictive_analysis = await self._perform_predictive_analysis(monitoring_result)
        monitoring_result['predictive_analysis'] = predictive_analysis
        
        # Step 9: Recommendations Generation
        await self.log_activity("Generating recommendations")
        recommendations = await self._generate_recommendations(monitoring_result)
        monitoring_result['recommendations'] = recommendations
        
        # Step 10: Quality Assessment
        await self.log_activity("Assessing monitoring quality")
        quality_assessment = await self._assess_monitoring_quality(monitoring_result)
        monitoring_result['quality_assessment'] = quality_assessment
        monitoring_result['confidence_score'] = quality_assessment.get('confidence_score', 0.0)
        
        return monitoring_result
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Agent and workflow counts (would be retrieved from registry/state service)
            active_agents = await self._count_active_agents()
            active_workflows = await self._count_active_workflows()
            
            # Error count (from recent logs)
            error_count = await self._count_recent_errors()
            
            # Response time (average from recent requests)
            response_time = await self._calculate_average_response_time()
            
            return SystemMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                active_agents=active_agents,
                active_workflows=active_workflows,
                error_count=error_count,
                response_time=response_time
            )
            
        except Exception as e:
            await self.log_activity(f"Error collecting system metrics: {str(e)}", "error")
            return None
    
    async def _assess_system_health(self, monitoring_task: 'MonitoringTask') -> Dict[str, Any]:
        """Assess overall system health"""
        health_assessment = {
            'overall_status': 'healthy',
            'component_health': {},
            'health_score': 0.0,
            'issues_detected': [],
            'recommendations': []
        }
        
        try:
            # Check system resources
            current_metrics = await self._collect_system_metrics()
            if current_metrics:
                resource_health = await self._assess_resource_health(current_metrics)
                health_assessment['component_health']['resources'] = resource_health
            
            # Check service health
            service_health = await self._check_service_health()
            health_assessment['component_health']['services'] = service_health
            
            # Check agent health
            agent_health = await self._check_agent_health()
            health_assessment['component_health']['agents'] = agent_health
            
            # Check workflow health
            workflow_health = await self._check_workflow_health()
            health_assessment['component_health']['workflows'] = workflow_health
            
            # Calculate overall health score
            health_score = await self._calculate_health_score(health_assessment['component_health'])
            health_assessment['health_score'] = health_score
            
            # Determine overall status
            if health_score >= 0.9:
                health_assessment['overall_status'] = 'excellent'
            elif health_score >= 0.7:
                health_assessment['overall_status'] = 'good'
            elif health_score >= 0.5:
                health_assessment['overall_status'] = 'fair'
            else:
                health_assessment['overall_status'] = 'poor'
            
            return health_assessment
            
        except Exception as e:
            health_assessment['overall_status'] = 'error'
            health_assessment['issues_detected'].append(f"Health assessment error: {str(e)}")
            return health_assessment
    
    async def _collect_performance_metrics(self, monitoring_task: 'MonitoringTask') -> List[Dict[str, Any]]:
        """Collect comprehensive performance metrics"""
        performance_metrics = []
        
        try:
            # System performance metrics
            system_metrics = await self._collect_system_metrics()
            if system_metrics:
                performance_metrics.append({
                    'type': 'system',
                    'timestamp': system_metrics.timestamp.isoformat(),
                    'metrics': {
                        'cpu_usage': system_metrics.cpu_usage,
                        'memory_usage': system_metrics.memory_usage,
                        'disk_usage': system_metrics.disk_usage,
                        'response_time': system_metrics.response_time
                    }
                })
            
            # Application performance metrics
            app_metrics = await self._collect_application_metrics()
            performance_metrics.extend(app_metrics)
            
            # Database performance metrics (if applicable)
            db_metrics = await self._collect_database_metrics()
            if db_metrics:
                performance_metrics.extend(db_metrics)
            
            # API performance metrics
            api_metrics = await self._collect_api_metrics()
            performance_metrics.extend(api_metrics)
            
            return performance_metrics
            
        except Exception as e:
            await self.log_activity(f"Error collecting performance metrics: {str(e)}", "error")
            return []
    
    async def _monitor_agents(self, monitoring_task: 'MonitoringTask') -> Dict[str, Any]:
        """Monitor agent performance and health"""
        agent_monitoring = {
            'total_agents': 0,
            'active_agents': 0,
            'idle_agents': 0,
            'error_agents': 0,
            'agent_details': [],
            'performance_summary': {}
        }
        
        try:
            # Get agent registry (would be injected)
            agents = await self._get_all_agents()
            
            agent_monitoring['total_agents'] = len(agents)
            
            for agent in agents:
                agent_status = await self._get_agent_status(agent.agent_id)
                agent_metrics = await self._get_agent_metrics(agent.agent_id)
                
                agent_detail = {
                    'agent_id': agent.agent_id,
                    'name': agent.name,
                    'status': agent_status,
                    'metrics': agent_metrics,
                    'health_score': await self._calculate_agent_health_score(agent_metrics)
                }
                
                agent_monitoring['agent_details'].append(agent_detail)
                
                # Count by status
                if agent_status == 'active':
                    agent_monitoring['active_agents'] += 1
                elif agent_status == 'idle':
                    agent_monitoring['idle_agents'] += 1
                elif agent_status == 'error':
                    agent_monitoring['error_agents'] += 1
            
            # Calculate performance summary
            agent_monitoring['performance_summary'] = await self._calculate_agent_performance_summary(
                agent_monitoring['agent_details']
            )
            
            return agent_monitoring
            
        except Exception as e:
            await self.log_activity(f"Error monitoring agents: {str(e)}", "error")
            return agent_monitoring
    
    async def _monitor_workflows(self, monitoring_task: 'MonitoringTask') -> Dict[str, Any]:
        """Monitor workflow execution and performance"""
        workflow_monitoring = {
            'total_workflows': 0,
            'active_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0,
            'workflow_details': [],
            'performance_summary': {}
        }
        
        try:
            # Get workflow states (would be from state service)
            workflows = await self._get_all_workflows()
            
            workflow_monitoring['total_workflows'] = len(workflows)
            
            for workflow in workflows:
                workflow_status = await self._get_workflow_status(workflow.workflow_id)
                workflow_metrics = await self._get_workflow_metrics(workflow.workflow_id)
                
                workflow_detail = {
                    'workflow_id': workflow.workflow_id,
                    'name': workflow.name,
                    'status': workflow_status,
                    'progress': workflow.progress,
                    'metrics': workflow_metrics,
                    'health_score': await self._calculate_workflow_health_score(workflow_metrics)
                }
                
                workflow_monitoring['workflow_details'].append(workflow_detail)
                
                # Count by status
                if workflow_status == 'active':
                    workflow_monitoring['active_workflows'] += 1
                elif workflow_status == 'completed':
                    workflow_monitoring['completed_workflows'] += 1
                elif workflow_status == 'failed':
                    workflow_monitoring['failed_workflows'] += 1
            
            # Calculate performance summary
            workflow_monitoring['performance_summary'] = await self._calculate_workflow_performance_summary(
                workflow_monitoring['workflow_details']
            )
            
            return workflow_monitoring
            
        except Exception as e:
            await self.log_activity(f"Error monitoring workflows: {str(e)}", "error")
            return workflow_monitoring
    
    async def _detect_and_analyze_errors(self, monitoring_task: 'MonitoringTask') -> Dict[str, Any]:
        """Detect and analyze system errors"""
        error_analysis = {
            'total_errors': 0,
            'error_categories': {},
            'recent_errors': [],
            'error_trends': {},
            'critical_errors': [],
            'recommendations': []
        }
        
        try:
            # Get recent errors from logs
            recent_errors = await self._get_recent_errors(hours=24)
            error_analysis['total_errors'] = len(recent_errors)
            error_analysis['recent_errors'] = recent_errors[:10]  # Last 10 errors
            
            # Categorize errors
            error_categories = await self._categorize_errors(recent_errors)
            error_analysis['error_categories'] = error_categories
            
            # Identify critical errors
            critical_errors = await self._identify_critical_errors(recent_errors)
            error_analysis['critical_errors'] = critical_errors
            
            # Analyze error trends
            error_trends = await self._analyze_error_trends(recent_errors)
            error_analysis['error_trends'] = error_trends
            
            # Generate recommendations
            recommendations = await self._generate_error_recommendations(error_analysis)
            error_analysis['recommendations'] = recommendations
            
            return error_analysis
            
        except Exception as e:
            await self.log_activity(f"Error in error analysis: {str(e)}", "error")
            return error_analysis
    
    async def _monitor_security(self, monitoring_task: 'MonitoringTask') -> Dict[str, Any]:
        """Monitor security-related metrics and events"""
        security_monitoring = {
            'security_score': 0.0,
            'threats_detected': [],
            'security_events': [],
            'access_patterns': {},
            'vulnerability_scan': {},
            'recommendations': []
        }
        
        try:
            # Monitor access patterns
            access_patterns = await self._monitor_access_patterns()
            security_monitoring['access_patterns'] = access_patterns
            
            # Detect security events
            security_events = await self._detect_security_events()
            security_monitoring['security_events'] = security_events
            
            # Check for threats
            threats = await self._detect_threats()
            security_monitoring['threats_detected'] = threats
            
            # Perform basic vulnerability scan
            vulnerability_scan = await self._perform_vulnerability_scan()
            security_monitoring['vulnerability_scan'] = vulnerability_scan
            
            # Calculate security score
            security_score = await self._calculate_security_score(security_monitoring)
            security_monitoring['security_score'] = security_score
            
            # Generate security recommendations
            recommendations = await self._generate_security_recommendations(security_monitoring)
            security_monitoring['recommendations'] = recommendations
            
            return security_monitoring
            
        except Exception as e:
            await self.log_activity(f"Error in security monitoring: {str(e)}", "error")
            return security_monitoring
    
    async def _process_alerts(self, monitoring_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and trigger alerts based on monitoring results"""
        alerts_triggered = []
        
        try:
            # Check system health alerts
            system_health = monitoring_result.get('system_health', {})
            health_score = system_health.get('health_score', 1.0)
            
            if health_score < 0.5:
                alert = await self._create_alert(
                    "System Health Critical",
                    f"System health score is {health_score:.2f}",
                    AlertSeverity.CRITICAL
                )
                alerts_triggered.append(alert)
            
            # Check performance alerts
            performance_metrics = monitoring_result.get('performance_metrics', [])
            for metric in performance_metrics:
                if metric.get('type') == 'system':
                    metrics_data = metric.get('metrics', {})
                    
                    # CPU usage alert
                    cpu_usage = metrics_data.get('cpu_usage', 0)
                    if cpu_usage > 90:
                        alert = await self._create_alert(
                            "High CPU Usage",
                            f"CPU usage is {cpu_usage:.1f}%",
                            AlertSeverity.HIGH
                        )
                        alerts_triggered.append(alert)
                    
                    # Memory usage alert
                    memory_usage = metrics_data.get('memory_usage', 0)
                    if memory_usage > 85:
                        alert = await self._create_alert(
                            "High Memory Usage",
                            f"Memory usage is {memory_usage:.1f}%",
                            AlertSeverity.HIGH
                        )
                        alerts_triggered.append(alert)
            
            # Check error alerts
            error_analysis = monitoring_result.get('error_analysis', {})
            critical_errors = error_analysis.get('critical_errors', [])
            
            if critical_errors:
                alert = await self._create_alert(
                    "Critical Errors Detected",
                    f"{len(critical_errors)} critical errors detected",
                    AlertSeverity.CRITICAL
                )
                alerts_triggered.append(alert)
            
            # Check security alerts
            security_metrics = monitoring_result.get('security_metrics', {})
            threats = security_metrics.get('threats_detected', [])
            
            if threats:
                alert = await self._create_alert(
                    "Security Threats Detected",
                    f"{len(threats)} security threats detected",
                    AlertSeverity.HIGH
                )
                alerts_triggered.append(alert)
            
            # Store alerts
            for alert in alerts_triggered:
                await self._store_alert(alert)
            
            return alerts_triggered
            
        except Exception as e:
            await self.log_activity(f"Error processing alerts: {str(e)}", "error")
            return alerts_triggered
    
    async def _perform_predictive_analysis(self, monitoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform predictive analysis based on monitoring data"""
        predictive_analysis = {
            'resource_predictions': {},
            'failure_predictions': {},
            'capacity_planning': {},
            'trend_analysis': {},
            'recommendations': []
        }
        
        try:
            # Analyze resource trends
            if len(self.metrics_history) >= 10:
                resource_predictions = await self._predict_resource_usage()
                predictive_analysis['resource_predictions'] = resource_predictions
            
            # Predict potential failures
            failure_predictions = await self._predict_failures(monitoring_result)
            predictive_analysis['failure_predictions'] = failure_predictions
            
            # Capacity planning analysis
            capacity_planning = await self._analyze_capacity_planning()
            predictive_analysis['capacity_planning'] = capacity_planning
            
            # Trend analysis
            trend_analysis = await self._analyze_trends()
            predictive_analysis['trend_analysis'] = trend_analysis
            
            # Generate predictive recommendations
            recommendations = await self._generate_predictive_recommendations(predictive_analysis)
            predictive_analysis['recommendations'] = recommendations
            
            return predictive_analysis
            
        except Exception as e:
            await self.log_activity(f"Error in predictive analysis: {str(e)}", "error")
            return predictive_analysis
    
    async def _generate_recommendations(self, monitoring_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on monitoring results"""
        recommendations = []
        
        try:
            # System health recommendations
            system_health = monitoring_result.get('system_health', {})
            health_score = system_health.get('health_score', 1.0)
            
            if health_score < 0.7:
                recommendations.append({
                    'category': 'system_health',
                    'priority': 'high',
                    'title': 'Improve System Health',
                    'description': 'System health score is below optimal threshold',
                    'actions': [
                        'Review system resource usage',
                        'Check for failing services',
                        'Investigate error logs',
                        'Consider scaling resources'
                    ]
                })
            
            # Performance recommendations
            performance_metrics = monitoring_result.get('performance_metrics', [])
            for metric in performance_metrics:
                if metric.get('type') == 'system':
                    metrics_data = metric.get('metrics', {})
                    
                    if metrics_data.get('cpu_usage', 0) > 80:
                        recommendations.append({
                            'category': 'performance',
                            'priority': 'medium',
                            'title': 'Optimize CPU Usage',
                            'description': 'CPU usage is consistently high',
                            'actions': [
                                'Profile CPU-intensive processes',
                                'Consider horizontal scaling',
                                'Optimize algorithms and queries',
                                'Implement caching strategies'
                            ]
                        })
            
            # Error-based recommendations
            error_analysis = monitoring_result.get('error_analysis', {})
            error_categories = error_analysis.get('error_categories', {})
            
            for category, count in error_categories.items():
                if count > 10:  # More than 10 errors in category
                    recommendations.append({
                        'category': 'error_handling',
                        'priority': 'medium',
                        'title': f'Address {category} Errors',
                        'description': f'High frequency of {category} errors detected',
                        'actions': [
                            f'Investigate root cause of {category} errors',
                            'Implement better error handling',
                            'Add monitoring for early detection',
                            'Consider circuit breaker patterns'
                        ]
                    })
            
            # Security recommendations
            security_metrics = monitoring_result.get('security_metrics', {})
            security_score = security_metrics.get('security_score', 1.0)
            
            if security_score < 0.8:
                recommendations.append({
                    'category': 'security',
                    'priority': 'high',
                    'title': 'Enhance Security Posture',
                    'description': 'Security score indicates potential vulnerabilities',
                    'actions': [
                        'Review access controls and permissions',
                        'Update security policies',
                        'Implement additional monitoring',
                        'Conduct security audit'
                    ]
                })
            
            return recommendations
            
        except Exception as e:
            await self.log_activity(f"Error generating recommendations: {str(e)}", "error")
            return recommendations
    
    # Helper methods would continue here...
    # (Additional helper methods for metrics collection, alert processing, etc.)
    
    async def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="High CPU Usage",
                condition="cpu_usage > 90",
                threshold=90.0,
                severity=AlertSeverity.HIGH
            ),
            AlertRule(
                name="High Memory Usage",
                condition="memory_usage > 85",
                threshold=85.0,
                severity=AlertSeverity.HIGH
            ),
            AlertRule(
                name="Low Disk Space",
                condition="disk_usage > 90",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL
            ),
            AlertRule(
                name="High Error Rate",
                condition="error_count > 50",
                threshold=50.0,
                severity=AlertSeverity.MEDIUM
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    async def _setup_health_checks(self):
        """Setup system health checks"""
        health_checks = [
            {
                'name': 'Database Connectivity',
                'check_function': self._check_database_health,
                'interval': 60
            },
            {
                'name': 'API Endpoints',
                'check_function': self._check_api_health,
                'interval': 30
            },
            {
                'name': 'External Services',
                'check_function': self._check_external_services,
                'interval': 120
            }
        ]
        
        self.health_checks.extend(health_checks)
```

### Task 11.2: Monitoring Service Integration

**File**: `core/services/monitoring_service.py`

**Monitoring Service Implementation**:
```python
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
from dataclasses import asdict

from ..models import SystemMetrics, AlertRule, HealthCheck
from .database_service import DatabaseService

class MonitoringService:
    """Service for managing monitoring data and operations"""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.active_monitors = {}
        self.alert_handlers = []
    
    async def store_metrics(self, metrics: SystemMetrics):
        """Store system metrics in database"""
        try:
            await self.db_service.execute(
                """
                INSERT INTO system_metrics 
                (timestamp, cpu_usage, memory_usage, disk_usage, 
                 network_io, active_agents, active_workflows, 
                 error_count, response_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metrics.timestamp,
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.disk_usage,
                    json.dumps(metrics.network_io),
                    metrics.active_agents,
                    metrics.active_workflows,
                    metrics.error_count,
                    metrics.response_time
                )
            )
        except Exception as e:
            print(f"Error storing metrics: {e}")
    
    async def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Retrieve metrics history"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            rows = await self.db_service.fetch_all(
                """
                SELECT * FROM system_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
                """,
                (cutoff_time,)
            )
            
            metrics_list = []
            for row in rows:
                metrics = SystemMetrics(
                    timestamp=row['timestamp'],
                    cpu_usage=row['cpu_usage'],
                    memory_usage=row['memory_usage'],
                    disk_usage=row['disk_usage'],
                    network_io=json.loads(row['network_io']),
                    active_agents=row['active_agents'],
                    active_workflows=row['active_workflows'],
                    error_count=row['error_count'],
                    response_time=row['response_time']
                )
                metrics_list.append(metrics)
            
            return metrics_list
            
        except Exception as e:
            print(f"Error retrieving metrics history: {e}")
            return []
    
    async def store_alert(self, alert: Dict[str, Any]):
        """Store alert in database"""
        try:
            await self.db_service.execute(
                """
                INSERT INTO alerts 
                (alert_id, title, description, severity, timestamp, 
                 status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert['alert_id'],
                    alert['title'],
                    alert['description'],
                    alert['severity'],
                    alert['timestamp'],
                    alert.get('status', 'active'),
                    json.dumps(alert.get('metadata', {}))
                )
            )
        except Exception as e:
            print(f"Error storing alert: {e}")
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts"""
        try:
            rows = await self.db_service.fetch_all(
                """
                SELECT * FROM alerts 
                WHERE status = 'active' 
                ORDER BY timestamp DESC
                """
            )
            
            alerts = []
            for row in rows:
                alert = {
                    'alert_id': row['alert_id'],
                    'title': row['title'],
                    'description': row['description'],
                    'severity': row['severity'],
                    'timestamp': row['timestamp'],
                    'status': row['status'],
                    'metadata': json.loads(row['metadata'])
                }
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            print(f"Error retrieving active alerts: {e}")
            return []
    
    async def register_alert_handler(self, handler_func):
        """Register alert handler function"""
        self.alert_handlers.append(handler_func)
    
    async def trigger_alert_handlers(self, alert: Dict[str, Any]):
        """Trigger all registered alert handlers"""
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")
```

### Task 11.3: Database Schema for Monitoring

**File**: `core/database/monitoring_schema.sql`

```sql
-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    cpu_usage REAL NOT NULL,
    memory_usage REAL NOT NULL,
    disk_usage REAL NOT NULL,
    network_io TEXT NOT NULL,  -- JSON string
    active_agents INTEGER NOT NULL,
    active_workflows INTEGER NOT NULL,
    error_count INTEGER NOT NULL,
    response_time REAL NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    status TEXT DEFAULT 'active',
    metadata TEXT,  -- JSON string
    resolved_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Agent metrics table
CREATE TABLE IF NOT EXISTS agent_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    status TEXT NOT NULL,
    task_count INTEGER DEFAULT 0,
    success_rate REAL DEFAULT 0.0,
    average_response_time REAL DEFAULT 0.0,
    error_count INTEGER DEFAULT 0,
    memory_usage REAL DEFAULT 0.0,
    cpu_usage REAL DEFAULT 0.0,
    metadata TEXT,  -- JSON string
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Workflow metrics table
CREATE TABLE IF NOT EXISTS workflow_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    status TEXT NOT NULL,
    progress REAL DEFAULT 0.0,
    duration REAL DEFAULT 0.0,
    task_count INTEGER DEFAULT 0,
    completed_tasks INTEGER DEFAULT 0,
    failed_tasks INTEGER DEFAULT 0,
    metadata TEXT,  -- JSON string
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Health checks table
CREATE TABLE IF NOT EXISTS health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    check_name TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    status TEXT NOT NULL,
    response_time REAL,
    details TEXT,
    metadata TEXT,  -- JSON string
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_type TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    tags TEXT,  -- JSON string
    metadata TEXT,  -- JSON string
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_agent_id ON agent_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_metrics_timestamp ON agent_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_workflow_metrics_workflow_id ON workflow_metrics(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_metrics_timestamp ON workflow_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_health_checks_timestamp ON health_checks(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
```

### Task 11.4: Frontend Monitoring Dashboard

**File**: `frontend/components/monitoring-dashboard.tsx`

```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface SystemMetrics {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  response_time: number;
  active_agents: number;
  active_workflows: number;
  error_count: number;
}

interface Alert {
  alert_id: string;
  title: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  status: string;
}

interface MonitoringDashboardProps {
  refreshInterval?: number;
}

export const MonitoringDashboard: React.FC<MonitoringDashboardProps> = ({ 
  refreshInterval = 30000 
}) => {
  const [metrics, setMetrics] = useState<SystemMetrics[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<SystemMetrics | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [systemHealth, setSystemHealth] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMonitoringData();
    const interval = setInterval(fetchMonitoringData, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  const fetchMonitoringData = async () => {
    try {
      // Fetch current metrics
      const metricsResponse = await fetch('/api/monitoring/metrics/current');
      const currentMetricsData = await metricsResponse.json();
      setCurrentMetrics(currentMetricsData);

      // Fetch metrics history
      const historyResponse = await fetch('/api/monitoring/metrics/history?hours=1');
      const historyData = await historyResponse.json();
      setMetrics(historyData);

      // Fetch active alerts
      const alertsResponse = await fetch('/api/monitoring/alerts/active');
      const alertsData = await alertsResponse.json();
      setAlerts(alertsData);

      // Fetch system health
      const healthResponse = await fetch('/api/monitoring/health');
      const healthData = await healthResponse.json();
      setSystemHealth(healthData);

      setLoading(false);
    } catch (error) {
      console.error('Error fetching monitoring data:', error);
      setLoading(false);
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

  const getHealthStatusColor = (score: number) => {
    if (score >= 0.9) return 'text-green-600';
    if (score >= 0.7) return 'text-yellow-600';
    if (score >= 0.5) return 'text-orange-600';
    return 'text-red-600';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-lg">Loading monitoring data...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">System Health</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <span className={getHealthStatusColor(systemHealth?.health_score || 0)}>
                {((systemHealth?.health_score || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              {systemHealth?.overall_status || 'Unknown'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentMetrics?.active_agents || 0}</div>
            <p className="text-xs text-muted-foreground">Currently running</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Workflows</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{currentMetrics?.active_workflows || 0}</div>
            <p className="text-xs text-muted-foreground">In progress</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{alerts.length}</div>
            <p className="text-xs text-muted-foreground">Require attention</p>
          </CardContent>
        </Card>
      </div>

      {/* Resource Usage */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">CPU Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={currentMetrics?.cpu_usage || 0} className="mb-2" />
            <p className="text-sm text-muted-foreground">
              {(currentMetrics?.cpu_usage || 0).toFixed(1)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={currentMetrics?.memory_usage || 0} className="mb-2" />
            <p className="text-sm text-muted-foreground">
              {(currentMetrics?.memory_usage || 0).toFixed(1)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Disk Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={currentMetrics?.disk_usage || 0} className="mb-2" />
            <p className="text-sm text-muted-foreground">
              {(currentMetrics?.disk_usage || 0).toFixed(1)}%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Monitoring */}
      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
          <TabsTrigger value="health">Health Checks</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics (Last Hour)</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metrics}>
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
                    dataKey="cpu_usage" 
                    stroke="#8884d8" 
                    name="CPU %"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="memory_usage" 
                    stroke="#82ca9d" 
                    name="Memory %"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="response_time" 
                    stroke="#ffc658" 
                    name="Response Time (ms)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Active Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              {alerts.length === 0 ? (
                <p className="text-muted-foreground">No active alerts</p>
              ) : (
                <div className="space-y-3">
                  {alerts.map((alert) => (
                    <Alert key={alert.alert_id}>
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h4 className="font-semibold">{alert.title}</h4>
                            <Badge variant={getSeverityColor(alert.severity)}>
                              {alert.severity}
                            </Badge>
                          </div>
                          <AlertDescription>{alert.description}</AlertDescription>
                          <p className="text-xs text-muted-foreground mt-1">
                            {new Date(alert.timestamp).toLocaleString()}
                          </p>
                        </div>
                      </div>
                    </Alert>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="health" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>System Health Details</CardTitle>
            </CardHeader>
            <CardContent>
              {systemHealth?.component_health && (
                <div className="space-y-4">
                  {Object.entries(systemHealth.component_health).map(([component, health]: [string, any]) => (
                    <div key={component} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-semibold capitalize">{component}</h4>
                        <Badge 
                          variant={health.status === 'healthy' ? 'default' : 'destructive'}
                        >
                          {health.status}
                        </Badge>
                      </div>
                      {health.score && (
                        <div className="mb-2">
                          <Progress value={health.score * 100} className="mb-1" />
                          <p className="text-sm text-muted-foreground">
                            Score: {(health.score * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {health.issues && health.issues.length > 0 && (
                        <div>
                          <p className="text-sm font-medium mb-1">Issues:</p>
                          <ul className="text-sm text-muted-foreground list-disc list-inside">
                            {health.issues.map((issue: string, index: number) => (
                              <li key={index}>{issue}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MonitoringDashboard;
```

## Backend API Integration

### Task 11.5: Monitoring API Endpoints

**File**: `app/api/monitoring.py`

```python
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from core.agents.monitoring_agent import MonitoringAgent
from core.services.monitoring_service import MonitoringService
from core.models import MonitoringTask

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# Global monitoring agent instance
monitoring_agent: Optional[MonitoringAgent] = None
monitoring_service: Optional[MonitoringService] = None

@router.post("/start")
async def start_monitoring(task: Dict[str, Any]):
    """Start monitoring task"""
    try:
        if not monitoring_agent:
            raise HTTPException(status_code=500, detail="Monitoring agent not initialized")
        
        result = await monitoring_agent.process_task(task)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/current")
async def get_current_metrics():
    """Get current system metrics"""
    try:
        if not monitoring_agent:
            raise HTTPException(status_code=500, detail="Monitoring agent not initialized")
        
        metrics = await monitoring_agent._collect_system_metrics()
        if metrics:
            return {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "disk_usage": metrics.disk_usage,
                "network_io": metrics.network_io,
                "active_agents": metrics.active_agents,
                "active_workflows": metrics.active_workflows,
                "error_count": metrics.error_count,
                "response_time": metrics.response_time
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to collect metrics")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/history")
async def get_metrics_history(hours: int = Query(24, ge=1, le=168)):
    """Get metrics history"""
    try:
        if not monitoring_service:
            raise HTTPException(status_code=500, detail="Monitoring service not initialized")
        
        metrics_list = await monitoring_service.get_metrics_history(hours)
        return [
            {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "disk_usage": metrics.disk_usage,
                "response_time": metrics.response_time,
                "active_agents": metrics.active_agents,
                "active_workflows": metrics.active_workflows,
                "error_count": metrics.error_count