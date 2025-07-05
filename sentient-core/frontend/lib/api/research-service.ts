import { ApiResponse } from './types';
import { ResearchResult } from '@/components/research-progress-view';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ResearchRequest {
  query: string;
  mode: 'knowledge' | 'deep' | 'best_in_class';
  persist_to_memory?: boolean;
}

export const ResearchService = {
  /**
   * Start a new research task
   */
  startResearch: async (request: ResearchRequest): Promise<ResearchResult> => {
    try {
      const response = await fetch(`${API_URL}/api/research/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Error starting research: ${response.statusText}`);
      }
      
      const data: ApiResponse<ResearchResult> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in startResearch:', error);
      throw error;
    }
  },

  /**
   * Get all research results
   */
  getResearchResults: async (): Promise<ResearchResult[]> => {
    try {
      const response = await fetch(`${API_URL}/api/research/results`);
      if (!response.ok) {
        throw new Error(`Error fetching research results: ${response.statusText}`);
      }
      
      const data: ApiResponse<ResearchResult[]> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in getResearchResults:', error);
      throw error;
    }
  },

  /**
   * Get a specific research result by ID
   */
  getResearchById: async (id: string): Promise<ResearchResult> => {
    try {
      const response = await fetch(`${API_URL}/api/research/results/${id}`);
      if (!response.ok) {
        throw new Error(`Error fetching research: ${response.statusText}`);
      }
      
      const data: ApiResponse<ResearchResult> = await response.json();
      return data.data;
    } catch (error) {
      console.error(`Error in getResearchById for id ${id}:`, error);
      throw error;
    }
  },

  /**
   * Cancel an in-progress research task
   */
  cancelResearch: async (id: string): Promise<{ success: boolean }> => {
    try {
      const response = await fetch(`${API_URL}/api/research/cancel/${id}`, {
        method: 'POST'
      });
      
      if (!response.ok) {
        throw new Error(`Error canceling research: ${response.statusText}`);
      }
      
      const data: ApiResponse<{ success: boolean }> = await response.json();
      return data.data;
    } catch (error) {
      console.error(`Error in cancelResearch for id ${id}:`, error);
      throw error;
    }
  },

  /**
   * Generate a PDF from research results
   */
  generatePDF: async (id: string): Promise<Blob> => {
    try {
      const response = await fetch(`${API_URL}/api/research/download/pdf/${id}`);
      
      if (!response.ok) {
        throw new Error(`Error generating PDF: ${response.statusText}`);
      }
      
      return await response.blob();
    } catch (error) {
      console.error(`Error in generatePDF for id ${id}:`, error);
      throw error;
    }
  },

  /**
   * Get markdown content for a research result
   */
  getMarkdownContent: async (id: string): Promise<string> => {
    try {
      const response = await fetch(`${API_URL}/api/research/download/markdown/${id}`);
      
      if (!response.ok) {
        throw new Error(`Error getting markdown content: ${response.statusText}`);
      }
      
      return await response.text();
    } catch (error) {
      console.error(`Error in getMarkdownContent for id ${id}:`, error);
      throw error;
    }
  },

  /**
   * Run multiple simultaneous research queries
   */
  batchResearch: async (requests: ResearchRequest[]): Promise<{ research_ids: string[] }> => {
    try {
      const response = await fetch(`${API_URL}/api/research/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ requests }),
      });
      
      if (!response.ok) {
        throw new Error(`Error starting batch research: ${response.statusText}`);
      }
      
      const data: ApiResponse<{ research_ids: string[] }> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in batchResearch:', error);
      throw error;
    }
  }
};
