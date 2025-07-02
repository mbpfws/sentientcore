import { ApiResponse, Agent } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const AgentService = {
  /**
   * Get all available agents
   */
  getAgents: async (): Promise<Agent[]> => {
    try {
      const response = await fetch(`${API_URL}/api/agents`);
      if (!response.ok) {
        throw new Error(`Error fetching agents: ${response.statusText}`);
      }
      const data: ApiResponse<Agent[]> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in getAgents:', error);
      throw error;
    }
  },

  /**
   * Get a specific agent by ID
   */
  getAgentById: async (id: string): Promise<Agent> => {
    try {
      const response = await fetch(`${API_URL}/api/agents/${id}`);
      if (!response.ok) {
        throw new Error(`Error fetching agent: ${response.statusText}`);
      }
      const data: ApiResponse<Agent> = await response.json();
      return data.data;
    } catch (error) {
      console.error(`Error in getAgentById for id ${id}:`, error);
      throw error;
    }
  },

  /**
   * Get agent capabilities by ID
   */
  getAgentCapabilities: async (id: string): Promise<Agent['capabilities']> => {
    try {
      const response = await fetch(`${API_URL}/api/agents/${id}/capabilities`);
      if (!response.ok) {
        throw new Error(`Error fetching agent capabilities: ${response.statusText}`);
      }
      const data: ApiResponse<Agent['capabilities']> = await response.json();
      return data.data;
    } catch (error) {
      console.error(`Error in getAgentCapabilities for id ${id}:`, error);
      throw error;
    }
  }
};
