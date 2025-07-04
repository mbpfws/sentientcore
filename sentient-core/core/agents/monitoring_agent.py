"""
The Monitoring Agent
Observes the state of the system and provides verbose feedback.
"""
from core.models import AppState, LogEntry, EnhancedTask
from core.agents.base_agent import BaseAgent, AgentCapability, ActivityType
from typing import Dict, Any, Optional, List
import time
import psutil
import asyncio

class MonitoringAgent(BaseAgent):
    """
    The Monitoring Agent is a specialized agent that observes the entire workflow.
    It does not modify the state but provides logs and insights based on the
    actions of other agents. Its primary role is to make the system's
    internal workings transparent to the user and for debugging purposes.
    """
    
    def __init__(self, agent_id: str = "monitoring_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Monitoring Agent",
            capabilities=[AgentCapability.MONITORING, AgentCapability.ANALYSIS],
            description="Specialized agent for system monitoring and observability"
        )
        self.monitoring_active = False
        self.metrics_history = []
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 5.0
        }

    def can_handle_task(self, task: EnhancedTask) -> bool:
        """
        Determines if this agent can handle the given task.
        """
        monitoring_keywords = ['monitor', 'observe', 'track', 'watch', 'analyze', 'metrics', 'performance']
        task_description = task.description.lower()
        return any(keyword in task_description for keyword in monitoring_keywords)
    
    async def process_task(self, task: EnhancedTask, state: Optional[AppState] = None) -> Dict:
        """
        Processes a monitoring task by executing monitoring workflows.
        """
        try:
            self.log_activity(ActivityType.TASK_STARTED, f"Processing monitoring task: {task.description}")
            
            # Collect system metrics
            metrics = await self.collect_system_metrics()
            
            # Analyze metrics for alerts
            alerts = self.analyze_metrics_for_alerts(metrics)
            
            # Generate monitoring report
            report = self.generate_monitoring_report(metrics, alerts)
            
            self.log_activity(ActivityType.TASK_COMPLETED, f"Monitoring task completed: {task.description}")
            
            return {
                "status": "completed",
                "metrics": metrics,
                "alerts": alerts,
                "report": report
            }
            
        except Exception as e:
            self.handle_error(e, f"Error processing monitoring task: {task.description}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collects comprehensive system metrics.
        """
        self.log_activity(ActivityType.PROCESSING, "Collecting system metrics")
        
        try:
            metrics = {
                'timestamp': time.time(),
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
                'active_processes': len(psutil.pids()),
                'system_load': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
            
            self.metrics_history.append(metrics)
            
            # Keep only last 100 metrics entries
            if len(self.metrics_history) > 100:
                self.metrics_history = self.metrics_history[-100:]
            
            return metrics
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error collecting metrics: {e}")
            return {}
    
    def analyze_metrics_for_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes metrics and generates alerts if thresholds are exceeded.
        """
        alerts = []
        
        try:
            # Check CPU usage
            if metrics.get('cpu_usage', 0) > self.alert_thresholds['cpu_usage']:
                alerts.append({
                    'type': 'cpu_high',
                    'severity': 'warning',
                    'message': f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                    'threshold': self.alert_thresholds['cpu_usage'],
                    'current_value': metrics['cpu_usage']
                })
            
            # Check memory usage
            if metrics.get('memory_usage', 0) > self.alert_thresholds['memory_usage']:
                alerts.append({
                    'type': 'memory_high',
                    'severity': 'warning',
                    'message': f"High memory usage: {metrics['memory_usage']:.1f}%",
                    'threshold': self.alert_thresholds['memory_usage'],
                    'current_value': metrics['memory_usage']
                })
            
            if alerts:
                self.log_activity(ActivityType.WARNING, f"Generated {len(alerts)} alerts")
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error analyzing metrics: {e}")
        
        return alerts
    
    def generate_monitoring_report(self, metrics: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str:
        """
        Generates a comprehensive monitoring report.
        """
        try:
            report_lines = [
                "# System Monitoring Report",
                f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metrics.get('timestamp', time.time())))}",
                "",
                "## System Metrics",
                f"- **CPU Usage:** {metrics.get('cpu_usage', 0):.1f}%",
                f"- **Memory Usage:** {metrics.get('memory_usage', 0):.1f}%",
                f"- **Disk Usage:** {metrics.get('disk_usage', 0):.1f}%",
                f"- **Active Processes:** {metrics.get('active_processes', 0)}",
                ""
            ]
            
            if alerts:
                report_lines.extend([
                    "## Alerts",
                    ""
                ])
                for alert in alerts:
                    report_lines.append(f"- **{alert['severity'].upper()}:** {alert['message']}")
                report_lines.append("")
            else:
                report_lines.extend([
                    "## Status",
                    "✅ All systems operating within normal parameters",
                    ""
                ])
            
            return "\n".join(report_lines)
            
        except Exception as e:
            self.log_activity(ActivityType.ERROR, f"Error generating report: {e}")
            return f"Error generating monitoring report: {e}"

    def invoke(self, state: AppState) -> AppState:
        """
        Legacy method for backward compatibility.
        The entry point for the Monitoring Agent. It inspects the state
        and adds a log entry reflecting the orchestrator's last decision.
        """
        print("---MONITORING AGENT---")
        
        self.log_activity(ActivityType.PROCESSING, "Legacy invoke method called")
        
        last_decision = state.next_action
        log_message = f"Monitoring workflow. Last decision from orchestrator: '{last_decision}'."

        if last_decision == "create_plan":
            # Access the plan from the orchestrator's decision stored in the state
            if state.orchestrator_decision:
                plan = state.orchestrator_decision.get("plan", [])
                log_message += f" A new plan was created with {len(plan)} task(s)."
            else:
                log_message += " A new plan was created."
        elif last_decision == "request_clarification":
            log_message += " The orchestrator is asking the user for more information."
        elif last_decision == "redirect_off_topic":
            log_message += " The orchestrator is redirecting an off-topic conversation."
        elif last_decision == "issue_warning":
            log_message += " The orchestrator has issued a warning for an unproductive conversation."

        state.logs.append(LogEntry(
            source="MonitoringAgent",
            message=log_message
        ))
        
        # This agent only observes, it does not change the next_action
        return state