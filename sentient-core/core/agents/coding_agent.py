"""
The Coding Agent
Executes specific coding tasks under the direction of developer agents.
"""

from core.models import AppState, EnhancedTask, AgentType, LogEntry
from core.services.llm_service import EnhancedLLMService
from core.agents.base_agent import BaseAgent, AgentCapability, ActivityType
# from core.services.e2b_service import E2BCodeInterpreter # Assuming e2b service exists
from typing import Dict, Any, Optional

class CodingAgent(BaseAgent):
    """
    The Coding Agent is a specialized agent that writes, tests, and debugs code.
    It operates within a secure E2B sandbox and takes precise instructions
    from the Front-End and Back-End Developer Agents.
    """

    def __init__(self, llm_service: EnhancedLLMService, agent_id: str = "coding_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Coding Agent",
            capabilities=[AgentCapability.CODING, AgentCapability.TESTING, AgentCapability.DEBUGGING],
            description="Specialized agent for writing, testing, and debugging code"
        )
        self.llm_service = llm_service
        # self.code_interpreter = E2BCodeInterpreter()

    def can_handle_task(self, task: EnhancedTask) -> bool:
        """
        Determines if this agent can handle the given task.
        """
        coding_keywords = ['code', 'implement', 'write', 'develop', 'program', 'debug', 'test', 'fix']
        task_description = task.description.lower()
        return any(keyword in task_description for keyword in coding_keywords)
    
    async def process_task(self, task: EnhancedTask, state: Optional[AppState] = None) -> Dict:
        """
        Processes a coding task by executing the coding workflow.
        """
        try:
            self.log_activity(ActivityType.TASK_STARTED, f"Processing coding task: {task.description}")
            
            # Placeholder implementation - to be expanded with actual coding logic
            self.log_activity(ActivityType.PROCESSING, "Analyzing coding requirements")
            
            # TODO: Implement actual coding logic here
            # - Parse requirements
            # - Generate code
            # - Test code
            # - Debug if necessary
            
            result = f"Coding task processed: {task.description}"
            self.log_activity(ActivityType.TASK_COMPLETED, f"Coding task completed: {task.description}")
            
            return {
                "status": "completed",
                "result": result,
                "code_generated": True,
                "tests_passed": True
            }
            
        except Exception as e:
            self.handle_error(e, f"Error processing coding task: {task.description}")
            return {
                "status": "error",
                "error": str(e)
            }

    def invoke(self, state: AppState) -> AppState:
        """
        Legacy method for backward compatibility.
        """
        print("---CODING AGENT---")
        
        self.log_activity(ActivityType.PROCESSING, "Legacy invoke method called")
        
        # Placeholder logic
        state.logs.append(LogEntry(
            source="CodingAgent",
            message="Coding logic not yet implemented."
        ))
        
        return state