import { ApiResponse } from './types';

export interface ResearchQuery {
  query: string;
  mode: 'knowledge' | 'deep' | 'best_in_class';
  workflow_id: string;
  context?: string;
}

export interface ResearchResult {
  id: string;
  query: string;
  mode: 'knowledge' | 'deep' | 'best_in_class';
  status: 'pending' | 'searching' | 'synthesizing' | 'completed' | 'error';
  progress: number;
  results?: {
    summary: string;
    sources: Array<{
      title: string;
      url: string;
      snippet: string;
      relevance_score?: number;
    }>;
    insights?: string[];
    recommendations?: string[];
    citations?: string[];
  };
  created_at: string;
  completed_at?: string;
  verbose_log?: string[];
  workflow_id: string;
}

export interface ResearchSession {
  id: string;
  workflow_id: string;
  queries: ResearchResult[];
  created_at: string;
  updated_at: string;
  metadata?: {
    total_queries: number;
    completed_queries: number;
    failed_queries: number;
  };
}

class ResearchService {
  private baseUrl = '/api/research';
  private sseConnection: EventSource | null = null;
  private listeners: Map<string, (data: any) => void> = new Map();

  // Start a new research query
  async startResearch(query: ResearchQuery): Promise<ApiResponse<ResearchResult>> {
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
        throw new Error(data.error || 'Failed to start research');
      }

      // Store in localStorage for persistence
      this.updateLocalStorage(data.data);
      
      // Start listening for updates
      this.subscribeToUpdates(data.data.id);

      return data;
    } catch (error) {
      console.error('Research start error:', error);
      throw error;
    }
  }

  // Get research results for a workflow
  async getResearchResults(workflowId: string): Promise<ApiResponse<ResearchResult[]>> {
    try {
      const response = await fetch(`${this.baseUrl}/results?workflow_id=${workflowId}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch research results');
      }

      // Update localStorage
      if (data.data) {
        data.data.forEach((result: ResearchResult) => {
          this.updateLocalStorage(result);
        });
      }

      return data;
    } catch (error) {
      console.error('Research fetch error:', error);
      throw error;
    }
  }

  // Get a specific research result
  async getResearchResult(resultId: string): Promise<ApiResponse<ResearchResult>> {
    try {
      const response = await fetch(`${this.baseUrl}/result/${resultId}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch research result');
      }

      // Update localStorage
      if (data.data) {
        this.updateLocalStorage(data.data);
      }

      return data;
    } catch (error) {
      console.error('Research result fetch error:', error);
      throw error;
    }
  }

  // Get research sessions for a workflow
  async getResearchSessions(workflowId: string): Promise<ApiResponse<ResearchSession[]>> {
    try {
      const response = await fetch(`${this.baseUrl}/sessions?workflow_id=${workflowId}`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch research sessions');
      }

      return data;
    } catch (error) {
      console.error('Research sessions fetch error:', error);
      throw error;
    }
  }

  // Export research result as PDF
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

  // Export research result as Markdown
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
  subscribeToUpdates(resultId: string, callback?: (data: ResearchResult) => void) {
    if (typeof window === 'undefined') return;

    try {
      // Close existing connection if any
      if (this.sseConnection) {
        this.sseConnection.close();
      }

      // Create SSE connection
      const sseUrl = `/api/sse/research/${resultId}`;
      this.sseConnection = new EventSource(sseUrl);

      this.sseConnection.onopen = () => {
        console.log('Research SSE connected');
      };

      // Listen for research_update events
      this.sseConnection.addEventListener('research_update', (event) => {
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
        console.error('Research SSE error:', error);
        
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

  // Add listener for research updates
  addUpdateListener(id: string, callback: (data: ResearchResult) => void) {
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
  private updateLocalStorage(result: ResearchResult) {
    if (typeof window === 'undefined') return;

    try {
      const key = `research_results_${result.workflow_id}`;
      const stored = localStorage.getItem(key);
      let results: ResearchResult[] = stored ? JSON.parse(stored) : [];
      
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
      localStorage.setItem(`research_result_${result.id}`, JSON.stringify(result));
    } catch (error) {
      console.error('localStorage update error:', error);
    }
  }

  // Get results from localStorage
  getLocalResults(workflowId: string): ResearchResult[] {
    if (typeof window === 'undefined') return [];

    try {
      const key = `research_results_${workflowId}`;
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
      const key = `research_results_${workflowId}`;
      localStorage.removeItem(key);
      
      // Also clear individual results
      const results = this.getLocalResults(workflowId);
      results.forEach(result => {
        localStorage.removeItem(`research_result_${result.id}`);
      });
    } catch (error) {
      console.error('localStorage clear error:', error);
    }
  }

  // Search through stored research results
  searchLocalResults(workflowId: string, searchTerm: string): ResearchResult[] {
    const results = this.getLocalResults(workflowId);
    const term = searchTerm.toLowerCase();
    
    return results.filter(result => 
      result.query.toLowerCase().includes(term) ||
      result.results?.summary.toLowerCase().includes(term) ||
      result.results?.sources.some(source => 
        source.title.toLowerCase().includes(term) ||
        source.snippet.toLowerCase().includes(term)
      )
    );
  }

  // Get research statistics
  getResearchStats(workflowId: string): {
    total: number;
    completed: number;
    pending: number;
    failed: number;
    modes: Record<string, number>;
  } {
    const results = this.getLocalResults(workflowId);
    
    const stats = {
      total: results.length,
      completed: 0,
      pending: 0,
      failed: 0,
      modes: {} as Record<string, number>
    };
    
    results.forEach(result => {
      // Count by status
      if (result.status === 'completed') stats.completed++;
      else if (result.status === 'error') stats.failed++;
      else stats.pending++;
      
      // Count by mode
      stats.modes[result.mode] = (stats.modes[result.mode] || 0) + 1;
    });
    
    return stats;
  }
}

// Export singleton instance
export const researchService = new ResearchService();
export default researchService;