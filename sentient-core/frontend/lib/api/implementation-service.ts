import { ApiResponse } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Implementation workflow types
export interface ImplementationPhase {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  progress: number;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  artifacts?: ImplementationArtifact[];
}

export interface ImplementationArtifact {
  id: string;
  name: string;
  type: 'file' | 'directory' | 'test_result' | 'build_output' | 'documentation';
  path: string;
  size?: number;
  created_at: string;
  content_preview?: string;
}

export interface ImplementationProgress {
  workflow_id: string;
  overall_progress: number;
  current_phase: string;
  phases: ImplementationPhase[];
  total_artifacts: number;
  test_results?: TestResult[];
  build_status?: BuildStatus;
}

export interface TestResult {
  id: string;
  test_name: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  error_message?: string;
  file_path: string;
}

export interface BuildStatus {
  status: 'success' | 'failed' | 'in_progress';
  build_time: number;
  errors: string[];
  warnings: string[];
}

export interface ImplementationRequest {
  feature_build_plan: string;
  synthesized_knowledge: string[];
  implementation_mode: 'validation' | 'phase2' | 'phase3' | 'phase4' | 'full';
  workflow_id: string;
}

export interface ImplementationResult {
  id: string;
  workflow_id: string;
  status: 'started' | 'in_progress' | 'completed' | 'failed';
  phase: string;
  progress: ImplementationProgress;
  message: string;
  error?: string;
  created_at: string;
  updated_at: string;
}

export interface ImplementationUpdate {
  id: string;
  type: 'phase_started' | 'phase_completed' | 'progress_update' | 'artifact_created' | 'test_completed' | 'error_occurred';
  phase: string;
  progress: number;
  message: string;
  timestamp: string;
  data?: any;
}

export const ImplementationService = {
  /**
   * Start implementation workflow
   */
  startImplementation: async (request: ImplementationRequest): Promise<ImplementationResult> => {
    try {
      const response = await fetch(`${API_URL}/api/implementation/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`Error starting implementation: ${errorData.detail || response.statusText}`);
      }
      
      const data: ApiResponse<ImplementationResult> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in startImplementation:', error);
      throw error;
    }
  },

  /**
   * Get implementation progress
   */
  getProgress: async (workflow_id: string): Promise<ImplementationProgress> => {
    try {
      const response = await fetch(`${API_URL}/api/implementation/progress/${workflow_id}`);
      
      if (!response.ok) {
        throw new Error(`Error fetching implementation progress: ${response.statusText}`);
      }
      
      const data: ApiResponse<ImplementationProgress> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in getProgress:', error);
      throw error;
    }
  },

  /**
   * Get implementation result
   */
  getResult: async (implementation_id: string): Promise<ImplementationResult> => {
    try {
      const response = await fetch(`${API_URL}/api/implementation/result/${implementation_id}`);
      
      if (!response.ok) {
        throw new Error(`Error fetching implementation result: ${response.statusText}`);
      }
      
      const data: ApiResponse<ImplementationResult> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in getResult:', error);
      throw error;
    }
  },

  /**
   * Subscribe to implementation updates
   */
  subscribeToUpdates: (implementation_id: string, callback: (update: ImplementationUpdate) => void): () => void => {
    const eventSource = new EventSource(`${API_URL}/api/implementation/updates/${implementation_id}`);
    
    eventSource.onmessage = (event) => {
      try {
        const update: ImplementationUpdate = JSON.parse(event.data);
        callback(update);
      } catch (error) {
        console.error('Error parsing implementation update:', error);
      }
    };
    
    eventSource.onerror = (error) => {
      console.error('Implementation updates stream error:', error);
    };
    
    // Return cleanup function
    return () => {
      eventSource.close();
    };
  },

  /**
   * Download artifact
   */
  downloadArtifact: async (workflow_id: string, artifact_id: string): Promise<Blob> => {
    try {
      const response = await fetch(`${API_URL}/api/implementation/artifact/${workflow_id}/${artifact_id}`);
      
      if (!response.ok) {
        throw new Error(`Error downloading artifact: ${response.statusText}`);
      }
      
      return await response.blob();
    } catch (error) {
      console.error('Error in downloadArtifact:', error);
      throw error;
    }
  },

  /**
   * Get artifact content
   */
  getArtifactContent: async (workflow_id: string, artifact_id: string): Promise<string> => {
    try {
      const response = await fetch(`${API_URL}/api/implementation/artifact/${workflow_id}/${artifact_id}/content`);
      
      if (!response.ok) {
        throw new Error(`Error fetching artifact content: ${response.statusText}`);
      }
      
      return await response.text();
    } catch (error) {
      console.error('Error in getArtifactContent:', error);
      throw error;
    }
  },

  /**
   * Cancel implementation
   */
  cancelImplementation: async (implementation_id: string): Promise<{ success: boolean }> => {
    try {
      const response = await fetch(`${API_URL}/api/implementation/cancel/${implementation_id}`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        throw new Error(`Error canceling implementation: ${response.statusText}`);
      }
      
      const data: ApiResponse<{ success: boolean }> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in cancelImplementation:', error);
      throw error;
    }
  },
};