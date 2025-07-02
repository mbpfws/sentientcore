// Common API types

export interface ApiResponse<T> {
  status: string;
  data: T;
}

// Agent types
export interface Capability {
  name: string;
  description: string;
}

export interface Agent {
  id: string;
  name: string;
  description: string;
  icon?: string;
  capabilities: Capability[];
}

// Task types
export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed';

export interface Task {
  id: string;
  title: string;
  description: string;
  status: TaskStatus;
  created_at: string;
  updated_at: string;
  agent_id?: string;
  parent_task_id?: string;
  output?: any;
}

// Chat types
export interface Message {
  id: string;
  sender: 'user' | 'assistant';
  content: string;
  created_at: string;
  metadata?: any;
}

// Workflow types
export interface Workflow {
  id: string;
  name: string;
  description: string;
  icon?: string;
}
