import { ApiResponse } from './types';

export interface PlannerQuery {
  query: string;
  research_artifacts?: string[];
  workflow_id: string;
  context?: string;
  requirements?: string[];
}

export interface PlannerResult {
  id: string;
  query: string;
  status: 'pending' | 'analyzing' | 'planning' | 'completed' | 'error';
  progress: number;
  results?: {
    prd: {
      title: string;
      overview: string;
      objectives: string[];
      requirements: string[];
      success_criteria: string[];
    };
    high_level_plan: {
      phases: Array<{
        name: string;
        description: string;
        deliverables: string[];
        timeline: string;
      }>;
      dependencies: string[];
      risks: string[];
    };
    tech_specifications: {
      architecture: string;
      tech_stack: string[];
      frameworks: string[];
      databases: string[];
      apis: string[];
      deployment: string;
    };
    implementation_roadmap: {
      milestones: Array<{
        name: string;
        description: string;
        tasks: string[];
        timeline: string;
      }>;
      resource_requirements: string[];
      quality_gates: string[];
    };
  };
  created_at: string;
  completed_at?: string;
  verbose_log?: string[];
  workflow_id: string;
  research_context?: string[];
}

export interface PlannerSession {
  id: string;
  workflow_id: string;
  plans: PlannerResult[];
  created_at: string;
  updated_at: string;
  metadata?: {
    total_plans: number;
    completed_plans: number;
    failed_plans: number;
  };
}

class PlannerService {
  private baseUrl = '/api/planner';
  private sseConnection: EventSource | null = null;
  private listeners: Map<string, (data: any) => void> = new Map();

  // Start a new planning session
  async startPlanning(query: PlannerQuery): Promise<ApiResponse<PlannerResult>> {
    try {
      const response = await fetch(`${this.baseUrl}/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(query),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to start planning');
      }

      // Store in localStorage for persistence
      this.updateLocalStorage(data.data);
      
      // Start listening for updates
      this.subscribeToUpdates(data.data.id);

      return data;
    } catch (error) {
      console.error('Planning start error:', error);
      throw error;
    }
  }

  // Get planning results for a workflow
  async getPlanningResults(workflowId: string): Promise<ApiResponse<PlannerResult[]>> {
    try {
      const response = await fetch(`${this.baseUrl}/results?workflow_id=${workflowId}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch planning results');
      }

      // Update localStorage
      if (data.data) {
        data.data.forEach((result: PlannerResult) => {
          this.updateLocalStorage(result);
        });
      }

      return data;
    } catch (error) {
      console.error('Planning fetch error:', error);
      throw error;
    }
  }

  // Get a specific planning result
  async getPlanningResult(resultId: string): Promise<ApiResponse<PlannerResult>> {
    try {
      const response = await fetch(`${this.baseUrl}/result/${resultId}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch planning result');
      }

      // Update localStorage
      if (data.data) {
        this.updateLocalStorage(data.data);
      }

      return data;
    } catch (error) {
      console.error('Planning result fetch error:', error);
      throw error;
    }
  }

  // Get planning sessions for a workflow
  async getPlanningSessions(workflowId: string): Promise<ApiResponse<PlannerSession[]>> {
    try {
      const response = await fetch(`${this.baseUrl}/sessions?workflow_id=${workflowId}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch planning sessions');
      }

      return data;
    } catch (error) {
      console.error('Planning sessions fetch error:', error);
      throw error;
    }
  }

  // Export planning result as PDF
  async exportToPDF(resultId: string): Promise<Blob> {
    try {
      const response = await fetch(`${this.baseUrl}/export/pdf`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ result_id: resultId }),
      });

      if (!response.ok) {
        throw new Error('Failed to export PDF');
      }

      return await response.blob();
    } catch (error) {
      console.error('PDF export error:', error);
      throw error;
    }
  }

  // Export planning result as Markdown
  async exportToMarkdown(resultId: string): Promise<string> {
    try {
      const response = await fetch(`${this.baseUrl}/export/markdown`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ result_id: resultId }),
      });

      if (!response.ok) {
        throw new Error('Failed to export Markdown');
      }

      const data = await response.json();
      return data.markdown;
    } catch (error) {
      console.error('Markdown export error:', error);
      throw error;
    }
  }

  // Subscribe to real-time updates via SSE
  subscribeToUpdates(resultId: string, callback?: (data: PlannerResult) => void) {
    if (typeof window === 'undefined') return;

    try {
      // Close existing connection if any
      if (this.sseConnection) {
        this.sseConnection.close();
      }

      // Create SSE connection
      const sseUrl = `/api/sse/planner/${resultId}`;
      this.sseConnection = new EventSource(sseUrl);

      this.sseConnection.onopen = () => {
        console.log('Planner SSE connected');
      };

      // Listen for planner_update events
      this.sseConnection.addEventListener('planner_update', (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Update localStorage
          if (data) {
            this.updateLocalStorage(data);
            
            // Call callback if provided
            if (callback) {
              callback(data);
            }
            
            // Notify all listeners
            this.listeners.forEach((listener) => {
              listener(data);
            });
          }
        } catch (error) {
          console.error('SSE message parse error:', error);
        }
      });

      this.sseConnection.onerror = (error) => {
        console.error('Planner SSE error:', error);
        
        // Attempt to reconnect after 5 seconds
        setTimeout(() => {
          if (this.sseConnection?.readyState === EventSource.CLOSED) {
            this.subscribeToUpdates(resultId, callback);
          }
        }, 5000);
      };
    } catch (error) {
      console.error('SSE connection error:', error);
    }
  }

  // Add listener for planning updates
  addUpdateListener(id: string, callback: (data: PlannerResult) => void) {
    this.listeners.set(id, callback);
  }

  // Remove listener
  removeUpdateListener(id: string) {
    this.listeners.delete(id);
  }

  // Disconnect SSE
  disconnect() {
    if (this.sseConnection) {
      this.sseConnection.close();
      this.sseConnection = null;
    }
    this.listeners.clear();
  }

  // Local storage management
  private updateLocalStorage(result: PlannerResult) {
    if (typeof window === 'undefined') return;

    try {
      const key = `planner_results_${result.workflow_id}`;
      const stored = localStorage.getItem(key);
      let results: PlannerResult[] = stored ? JSON.parse(stored) : [];
      
      // Update or add result
      const existingIndex = results.findIndex(r => r.id === result.id);
      if (existingIndex >= 0) {
        results[existingIndex] = result;
      } else {
        results.push(result);
      }
      
      // Sort by created_at (newest first)
      results.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      
      localStorage.setItem(key, JSON.stringify(results));
      
      // Also store individual result
      localStorage.setItem(`planner_result_${result.id}`, JSON.stringify(result));
    } catch (error) {
      console.error('localStorage update error:', error);
    }
  }

  // Get results from localStorage
  getLocalResults(workflowId: string): PlannerResult[] {
    if (typeof window === 'undefined') return [];

    try {
      const key = `planner_results_${workflowId}`;
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('localStorage read error:', error);
      return [];
    }
  }

  // Clear local storage for a workflow
  clearLocalResults(workflowId: string) {
    if (typeof window === 'undefined') return;

    try {
      const key = `planner_results_${workflowId}`;
      localStorage.removeItem(key);
      
      // Also clear individual results
      const results = this.getLocalResults(workflowId);
      results.forEach(result => {
        localStorage.removeItem(`planner_result_${result.id}`);
      });
    } catch (error) {
      console.error('localStorage clear error:', error);
    }
  }

  // Search through stored planning results
  searchLocalResults(workflowId: string, searchTerm: string): PlannerResult[] {
    const results = this.getLocalResults(workflowId);
    const term = searchTerm.toLowerCase();
    
    return results.filter(result => 
      result.query.toLowerCase().includes(term) ||
      result.results?.prd.title.toLowerCase().includes(term) ||
      result.results?.prd.overview.toLowerCase().includes(term) ||
      result.results?.tech_specifications.architecture.toLowerCase().includes(term)
    );
  }

  // Get planning statistics
  getPlanningStats(workflowId: string): {
    total: number;
    completed: number;
    pending: number;
    failed: number;
  } {
    const results = this.getLocalResults(workflowId);
    
    const stats = {
      total: results.length,
      completed: 0,
      pending: 0,
      failed: 0
    };
    
    results.forEach(result => {
      if (result.status === 'completed') stats.completed++;
      else if (result.status === 'error') stats.failed++;
      else stats.pending++;
    });
    
    return stats;
  }
}

// Export singleton instance
export const plannerService = new PlannerService();
export default plannerService;