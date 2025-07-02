"""
The Monitoring Agent
Observes the state of the system and provides verbose feedback.
"""
from core.models import AppState, LogEntry, EnhancedTask
from typing import Dict, Any

class MonitoringAgent:
    """
    The Monitoring Agent is a specialized agent that observes the entire workflow.
    It does not modify the state but provides logs and insights based on the
    actions of other agents. Its primary role is to make the system's
    internal workings transparent to the user and for debugging purposes.
    """

    def invoke(self, state: AppState) -> AppState:
        """
        The entry point for the Monitoring Agent. It inspects the state
        and adds a log entry reflecting the orchestrator's last decision.
        """
        print("---MONITORING AGENT---")
        
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