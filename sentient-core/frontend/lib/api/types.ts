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
export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'done';

export interface Task {
  id: string;
  title: string;
  description: string;
  status: TaskStatus;
  agent_type: string;
  sequence: number;
  dependencies: string[];
  result?: string;
  follow_up_questions?: string[];
  created_at: string;
  updated_at?: string;
  completed_at?: string;
  // Legacy compatibility fields
  agent_id?: string;
  parent_task_id?: string;
  output?: any;
  // Computed property for backward compatibility
  agent?: string;
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
