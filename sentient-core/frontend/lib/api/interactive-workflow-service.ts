import { ApiResponse } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Interactive Workflow Types
export interface InteractiveWorkflow {
  id: string;
  name: string;
  description: string;
  status: string;
  created_at: string;
  updated_at?: string;
  metadata: Record<string, any>;
  steps: WorkflowStep[];
  current_step_index: number;
  total_steps: number;
  completion_percentage: number;
}

export interface WorkflowStep {
  id: string;
  name: string;
  description: string;
  step_type: string;
  status: string;
  created_at: string;
  updated_at?: string;
  metadata: Record<string, any>;
  dependencies: string[];
  estimated_duration?: number;
  actual_duration?: number;
}

export interface PendingApproval {
  id: string;
  workflow_id: string;
  step_id: string;
  approval_type: string;
  message: string;
  context: Record<string, any>;
  created_at: string;
  timeout_seconds?: number;
}

export interface CreateWorkflowRequest {
  name: string;
  description: string;
  metadata?: Record<string, any>;
}

export interface AddStepRequest {
  name: string;
  description: string;
  step_type: string;
  metadata?: Record<string, any>;
  dependencies?: string[];
  estimated_duration?: number;
}

export interface SubmitApprovalRequest {
  decision: 'approve' | 'reject' | 'modify';
  feedback?: string;
  modifications?: Record<string, any>;
}

export interface TaskBreakdownRequest {
  task_description: string;
  strategy?: string;
  context?: Record<string, any>;
}

export interface WorkflowControlRequest {
  action: 'pause' | 'resume' | 'cancel' | 'restart';
  reason?: string;
}

export interface WorkflowListResponse {
  workflows: InteractiveWorkflow[];
  total_count: number;
  limit: number;
  offset: number;
}

export interface TaskBreakdownResponse {
  breakdown_id: string;
  original_task: string;
  steps: WorkflowStep[];
  estimated_total_duration: number;
  complexity_score: number;
  strategy_used: string;
}

export interface WorkflowMetrics {
  total_workflows: number;
  active_workflows: number;
  completed_workflows: number;
  failed_workflows: number;
  average_completion_time: number;
  approval_rate: number;
}

export interface ExecutionReport {
  workflow_id: string;
  total_steps: number;
  completed_steps: number;
  failed_steps: number;
  total_duration: number;
  approval_count: number;
  rejection_count: number;
  current_status: string;
}

export const InteractiveWorkflowService = {
  /**
   * Create a new interactive workflow
   */
  createWorkflow: async (request: CreateWorkflowRequest): Promise<InteractiveWorkflow> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Error creating workflow: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error in createWorkflow:', error);
      throw error;
    }
  },

  /**
   * Start a workflow
   */
  startWorkflow: async (workflowId: string): Promise<{ message: string; workflow_id: string; result: any }> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}/start`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Error starting workflow: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in startWorkflow for id ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * List all workflows with optional filtering
   */
  listWorkflows: async (status?: string, limit = 10, offset = 0): Promise<WorkflowListResponse> => {
    try {
      const params = new URLSearchParams({
        limit: limit.toString(),
        offset: offset.toString(),
      });
      
      if (status) {
        params.append('status', status);
      }
      
      const response = await fetch(`${API_URL}/api/interactive-workflows/?${params}`);
      
      if (!response.ok) {
        throw new Error(`Error listing workflows: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error in listWorkflows:', error);
      throw error;
    }
  },

  /**
   * Get a specific workflow by ID
   */
  getWorkflow: async (workflowId: string): Promise<InteractiveWorkflow> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}`);
      
      if (!response.ok) {
        throw new Error(`Error getting workflow: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in getWorkflow for id ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * Delete a workflow
   */
  deleteWorkflow: async (workflowId: string): Promise<{ message: string; workflow_id: string }> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error(`Error deleting workflow: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in deleteWorkflow for id ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * Add a step to a workflow
   */
  addStep: async (workflowId: string, request: AddStepRequest): Promise<WorkflowStep> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}/steps`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Error adding step: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in addStep for workflow ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * Get pending approvals for a workflow
   */
  getPendingApprovals: async (workflowId: string): Promise<PendingApproval[]> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}/approvals/pending`);
      
      if (!response.ok) {
        throw new Error(`Error getting pending approvals: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in getPendingApprovals for workflow ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * Submit an approval decision
   */
  submitApproval: async (workflowId: string, approvalId: string, request: SubmitApprovalRequest): Promise<{ message: string; approval_id: string; decision: string }> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}/approvals/${approvalId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Error submitting approval: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in submitApproval for approval ${approvalId}:`, error);
      throw error;
    }
  },

  /**
   * Control workflow execution (pause, resume, cancel, restart)
   */
  controlWorkflow: async (workflowId: string, request: WorkflowControlRequest): Promise<{ message: string; workflow_id: string; action: string }> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}/control`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Error controlling workflow: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in controlWorkflow for workflow ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * Break down a task into steps
   */
  breakdownTask: async (request: TaskBreakdownRequest): Promise<TaskBreakdownResponse> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/breakdown`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Error breaking down task: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error in breakdownTask:', error);
      throw error;
    }
  },

  /**
   * Get workflow metrics
   */
  getMetrics: async (): Promise<WorkflowMetrics> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/metrics`);
      
      if (!response.ok) {
        throw new Error(`Error getting metrics: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error in getMetrics:', error);
      throw error;
    }
  },

  /**
   * Get execution report for a workflow
   */
  getExecutionReport: async (workflowId: string): Promise<ExecutionReport> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/${workflowId}/report`);
      
      if (!response.ok) {
        throw new Error(`Error getting execution report: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`Error in getExecutionReport for workflow ${workflowId}:`, error);
      throw error;
    }
  },

  /**
   * Health check for interactive workflows
   */
  healthCheck: async (): Promise<{ status: string; message: string }> => {
    try {
      const response = await fetch(`${API_URL}/api/interactive-workflows/health`);
      
      if (!response.ok) {
        throw new Error(`Error in health check: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Error in healthCheck:', error);
      throw error;
    }
  },
};