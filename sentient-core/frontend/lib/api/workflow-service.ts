import { ApiResponse, Task, Workflow } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const WorkflowService = {
  /**
   * Get all available workflows
   */
  getWorkflows: async (): Promise<Workflow[]> => {
    try {
      const response = await fetch(`${API_URL}/api/workflows`);
      if (!response.ok) {
        throw new Error(`Error fetching workflows: ${response.statusText}`);
      }
      const data: ApiResponse<Workflow[]> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in getWorkflows:', error);
      throw error;
    }
  },

  /**
   * Execute a workflow
   */
  executeWorkflow: async (workflowId: string, input: any): Promise<{ workflow_execution_id: string }> => {
    try {
      const response = await fetch(`${API_URL}/api/workflows/${workflowId}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(input),
      });
      
      if (!response.ok) {
        throw new Error(`Error executing workflow: ${response.statusText}`);
      }
      
      const data: ApiResponse<{ workflow_execution_id: string }> = await response.json();
      return data.data;
    } catch (error) {
      console.error(`Error in executeWorkflow for id ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * Get tasks for a workflow execution
   */
  getTasks: async (workflowExecutionId?: string): Promise<Task[]> => {
    try {
      const url = workflowExecutionId 
        ? `${API_URL}/api/workflows/${workflowExecutionId}/tasks`
        : `${API_URL}/api/workflows/tasks`;
        
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Error fetching tasks: ${response.statusText}`);
      }
      
      const data: ApiResponse<Task[]> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in getTasks:', error);
      throw error;
    }
  },

  /**
   * Get a specific task by ID
   */
  getTaskById: async (taskId: string): Promise<Task> => {
    try {
      const response = await fetch(`${API_URL}/api/workflows/tasks/${taskId}`);
      if (!response.ok) {
        throw new Error(`Error fetching task: ${response.statusText}`);
      }
      
      const data: ApiResponse<Task> = await response.json();
      return data.data;
    } catch (error) {
      console.error(`Error in getTaskById for id ${taskId}:`, error);
      throw error;
    }
  },

  /**
   * Execute a specific task
   */
  executeTask: async (taskId: string, input: any): Promise<Task> => {
    try {
      const response = await fetch(`${API_URL}/api/workflows/tasks/${taskId}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(input),
      });
      
      if (!response.ok) {
        throw new Error(`Error executing task: ${response.statusText}`);
      }
      
      const data: ApiResponse<Task> = await response.json();
      return data.data;
    } catch (error) {
      console.error(`Error in executeTask for id ${taskId}:`, error);
      throw error;
    }
  },
};
