import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interfaces for API responses
export interface AgentState {
  // Define AgentState properties based on backend
  [key: string]: any;
}

export interface WorkflowState {
  // Define WorkflowState properties based on backend
  [key: string]: any;
}

export interface InitializeResponse {
  session_id: string;
  message: string;
}

export interface StateResponse {
  agent_states: Record<string, AgentState>;
  workflow_states: Record<string, WorkflowState>;
}

export interface MemoryStoreRequest {
  content: any;
  metadata?: any;
}

export interface MemoryRetrieveRequest {
  query: string;
  filters?: any;
}

// Core services client
export const coreServicesClient = {
  async initializeSession(apiKey: string): Promise<InitializeResponse> {
    const response = await apiClient.post('/initialize', { api_key: apiKey });
    return response.data;
  },

  async getState(sessionId: string): Promise<StateResponse> {
    const response = await apiClient.get(`/state?session_id=${sessionId}`);
    return response.data;
  },

  async storeMemory(sessionId: string, data: MemoryStoreRequest): Promise<any> {
    const response = await apiClient.post(`/memory/store?session_id=${sessionId}`, data);
    return response.data;
  },

  async retrieveMemory(sessionId: string, params: MemoryRetrieveRequest): Promise<any> {
    const response = await apiClient.get('/memory/retrieve', { params: { session_id: sessionId, ...params } });
    return response.data;
  },
};