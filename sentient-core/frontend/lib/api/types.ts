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

// Agent State types
export interface AgentState {
  status: 'active' | 'busy' | 'idle' | 'error';
  current_task?: string;
  last_activity?: string;
  error_message?: string;
}

// Workflow types
export interface Workflow {
  id: string;
  name: string;
  description: string;
  icon?: string;
}

export interface WorkflowState {
  id: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  current_step?: string;
  progress?: number;
  created_at: string;
  updated_at?: string;
}

// Memory types
export interface MemoryItem {
  id: string;
  content: string;
  type: 'text' | 'image' | 'file';
  created_at: string;
  metadata?: any;
}

// Search types
export interface SearchResult {
  id: string;
  title: string;
  content: string;
  url?: string;
  source: string;
  relevance_score?: number;
  created_at: string;
}

// Service Health types
export interface ServiceHealth {
  memory_service: boolean;
  state_service: boolean;
  search_service: boolean;
  vector_service: boolean;
  llm_service?: boolean;
  overall_status: 'healthy' | 'degraded' | 'down';
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
