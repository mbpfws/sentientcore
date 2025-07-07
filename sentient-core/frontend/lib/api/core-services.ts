import { ApiResponse } from './types';

// Types for Core Services API
export interface MemoryStoreRequest {
  layer: 'knowledge_synthesis' | 'conversation_history' | 'codebase_knowledge' | 'stack_dependencies';
  memory_type: 'conversation' | 'code_snippet' | 'documentation' | 'research_finding' | 'architectural_decision' | 'dependency_info' | 'best_practice' | 'error_solution';
  content: string;
  metadata?: Record<string, any>;
  tags?: string[];
}

export interface MemoryRetrieveRequest {
  query: string;
  layer?: string;
  memory_type?: string;
  limit?: number;
  similarity_threshold?: number;
}

export interface SearchRequest {
  query: string;
  search_type?: 'knowledge' | 'code' | 'documentation';
  providers?: string[];
  max_results?: number;
  include_metadata?: boolean;
}

export interface StateUpdateRequest {
  entity_id: string;
  updates: Record<string, any>;
}

export interface MemoryItem {
  id: string;
  content: string;
  layer: string;
  memory_type: string;
  metadata: Record<string, any>;
  tags: string[];
  created_at: string;
  similarity_score?: number;
}

export interface AgentState {
  agent_id: string;
  status: 'idle' | 'active' | 'busy' | 'error' | 'offline';
  current_task?: string;
  last_activity?: string;
  metadata: Record<string, any>;
  performance_metrics: Record<string, any>;
  error_info?: string;
}

export interface WorkflowState {
  workflow_id: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
  current_step?: string;
  progress: number;
  started_at?: string;
  completed_at?: string;
  steps_completed: string[];
  steps_remaining: string[];
  metadata: Record<string, any>;
  error_info?: string;
}

export interface SearchResult {
  title: string;
  content: string;
  url?: string;
  source: string;
  relevance_score: number;
  metadata: Record<string, any>;
  created_at?: string;
}

export interface ServiceHealth {
  memory_service: boolean;
  state_service: boolean;
  search_service: boolean;
  vector_service: boolean;
  timestamp: string;
  state_performance?: Record<string, any>;
  memory_stats?: Record<string, any>;
}

// Enhanced types for conversation context and confirmations
export interface ConversationContext {
  current_focus: string;
  user_intent: string;
  requirements_gathered: boolean;
  research_needed: boolean;
  project_type?: string;
  last_updated: string;
}

export interface PendingConfirmation {
  confirmation_id: string;
  message: string;
  action_type: string;
  action_data: Record<string, any>;
  created_at: string;
}

export interface ConfirmationRequest {
  confirmation_id: string;
  confirmed: boolean;
  session_id?: string;
}

export interface ContextUpdateRequest {
  session_id: string;
  context: ConversationContext;
}

/**
 * Core Services Client for frontend integration
 * Handles memory management, state tracking, and search operations
 */
export class CoreServicesClient {
  constructor() {}

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `/api${endpoint}`;

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Memory Management Methods
  async storeMemory(request: MemoryStoreRequest): Promise<{ success: boolean; memory_id: string; layer: string; memory_type: string }> {
    return this.request('/memory/store', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async retrieveMemory(request: MemoryRetrieveRequest): Promise<{ success: boolean; memories: MemoryItem[]; count: number }> {
    return this.request('/memory/retrieve', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getMemoryStats(): Promise<{ success: boolean; stats: Record<string, any> }> {
    return this.request('/memory/stats');
  }

  // State Management Methods
  async getAgentStates(): Promise<{ success: boolean; agent_states: Record<string, AgentState> }> {
    return this.request('/state/agents');
  }

  async getWorkflowState(workflowId: string): Promise<{ success: boolean; workflow_state: WorkflowState }> {
    return this.request(`/state/workflow/${workflowId}`);
  }

  async updateAgentState(agentId: string, updates: Record<string, any>): Promise<{ success: boolean; agent_state: AgentState }> {
    return this.request(`/state/agents/${agentId}/update`, {
      method: 'POST',
      body: JSON.stringify({ entity_id: agentId, updates }),
    });
  }

  async updateWorkflowState(workflowId: string, updates: Record<string, any>): Promise<{ success: boolean; workflow_state: WorkflowState }> {
    return this.request(`/state/workflow/${workflowId}/update`, {
      method: 'POST',
      body: JSON.stringify({ entity_id: workflowId, updates }),
    });
  }

  // Search Methods
  async searchKnowledge(request: SearchRequest): Promise<{ success: boolean; results: SearchResult[]; count: number; query: string; search_type: string }> {
    return this.request('/search/knowledge', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getSearchProviders(): Promise<{ success: boolean; providers: string[] }> {
    return this.request('/search/providers');
  }

  // Service Health and Management
  async getServicesHealth(): Promise<{ success: boolean; health: ServiceHealth }> {
    return this.request('/health');
  }

  async initializeServices(): Promise<{ success: boolean; message: string; services: Record<string, boolean> }> {
    return this.request('/initialize', {
      method: 'POST',
    });
  }

  // Convenience methods for common operations
  async storeConversation(content: string, metadata?: Record<string, any>): Promise<string> {
    const response = await this.storeMemory({
      layer: 'conversation_history',
      memory_type: 'conversation',
      content,
      metadata,
    });
    return response.memory_id;
  }

  async storeCodeSnippet(content: string, metadata?: Record<string, any>, tags?: string[]): Promise<string> {
    const response = await this.storeMemory({
      layer: 'codebase_knowledge',
      memory_type: 'code_snippet',
      content,
      metadata,
      tags,
    });
    return response.memory_id;
  }

  async storeResearchFinding(content: string, metadata?: Record<string, any>, tags?: string[]): Promise<string> {
    const response = await this.storeMemory({
      layer: 'knowledge_synthesis',
      memory_type: 'research_finding',
      content,
      metadata,
      tags,
    });
    return response.memory_id;
  }

  async searchConversationHistory(query: string, limit: number = 10): Promise<MemoryItem[]> {
    const response = await this.retrieveMemory({
      query,
      layer: 'conversation_history',
      limit,
    });
    return response.memories;
  }

  async searchCodebase(query: string, limit: number = 10): Promise<MemoryItem[]> {
    const response = await this.retrieveMemory({
      query,
      layer: 'codebase_knowledge',
      limit,
    });
    return response.memories;
  }

  async searchKnowledgeBase(query: string, limit: number = 10): Promise<MemoryItem[]> {
    const response = await this.retrieveMemory({
      query,
      layer: 'knowledge_synthesis',
      limit,
    });
    return response.memories;
  }

  // Real-time state monitoring
  async monitorAgentStates(callback: (states: Record<string, AgentState>) => void, interval: number = 5000): Promise<() => void> {
    const intervalId = setInterval(async () => {
      try {
        const response = await this.getAgentStates();
        if (response.success) {
          callback(response.agent_states);
        }
      } catch (error) {
        console.error('Error monitoring agent states:', error);
      }
    }, interval);

    return () => clearInterval(intervalId);
  }

  async monitorWorkflowState(workflowId: string, callback: (state: WorkflowState) => void, interval: number = 2000): Promise<() => void> {
    const intervalId = setInterval(async () => {
      try {
        const response = await this.getWorkflowState(workflowId);
        if (response.success) {
          callback(response.workflow_state);
        }
      } catch (error) {
        console.error('Error monitoring workflow state:', error);
      }
    }, interval);

    return () => clearInterval(intervalId);
  }

  // Enhanced Chat API Methods for Confirmation and Context Management
  async handleConfirmation(request: ConfirmationRequest): Promise<{ confirmation_id: string; confirmed: boolean; action_executed: boolean; message: string }> {
    const url = `${this.baseUrl}/api/chat/confirmation`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  async getConversationContext(sessionId: string): Promise<ConversationContext> {
    const url = `${this.baseUrl}/api/chat/sessions/${sessionId}/context`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  async updateConversationContext(sessionId: string, context: ConversationContext): Promise<{ message: string; context: ConversationContext }> {
    const url = `${this.baseUrl}/api/chat/sessions/${sessionId}/context`;
    
    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ session_id: sessionId, context }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  async getPendingConfirmations(sessionId: string): Promise<{ confirmations: PendingConfirmation[]; total_count: number }> {
    const url = `${this.baseUrl}/api/chat/sessions/${sessionId}/confirmations`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  // Enhanced session management methods
  async getChatHistory(sessionId?: string, workflowMode?: string, taskId?: string): Promise<{ messages: any[]; workflow_mode: string; task_id?: string }> {
    const params = new URLSearchParams();
    if (sessionId) params.append('session_id', sessionId);
    if (workflowMode) params.append('workflow_mode', workflowMode);
    if (taskId) params.append('task_id', taskId);
    
    const url = `${this.baseUrl}/api/chat/history${params.toString() ? '?' + params.toString() : ''}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  async listSessions(): Promise<{ sessions: any[]; total_count: number }> {
    const url = `${this.baseUrl}/api/chat/sessions`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  async deleteSession(sessionId: string): Promise<{ message: string }> {
    const url = `${this.baseUrl}/api/chat/sessions/${sessionId}`;
    
    const response = await fetch(url, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  async getSessionStats(sessionId: string): Promise<any> {
    const url = `${this.baseUrl}/api/chat/sessions/${sessionId}/stats`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  // Research artifact methods
  async listResearchArtifacts(): Promise<{ artifacts: any[]; total_count: number }> {
    const url = `${this.baseUrl}/api/chat/research/artifacts`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    return result.data;
  }

  getResearchArtifactDownloadUrl(filename: string): string {
    return `${this.baseUrl}/api/chat/download/research/${filename}`;
  }
}

// Export singleton instance
export const coreServicesClient = new CoreServicesClient();