'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Tabs } from './ui/tabs';
import { coreServicesClient, AgentState, WorkflowState, MemoryItem, SearchResult, ServiceHealth } from '../lib/api';

interface CoreServicesDashboardProps {
  className?: string;
}

export function CoreServicesDashboard({ className }: CoreServicesDashboardProps) {
  const [activeTab, setActiveTab] = useState('overview');
  const [serviceHealth, setServiceHealth] = useState<ServiceHealth | null>(null);
  const [agentStates, setAgentStates] = useState<Record<string, AgentState>>({});
  const [memoryStats, setMemoryStats] = useState<any>(null);
  const [searchProviders, setSearchProviders] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);

  // Load initial data
  const loadData = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [healthResponse, agentsResponse, memoryResponse, providersResponse] = await Promise.allSettled([
        coreServicesClient.getServicesHealth(),
        coreServicesClient.getAgentStates(),
        coreServicesClient.getMemoryStats(),
        coreServicesClient.getSearchProviders()
      ]);

      if (healthResponse.status === 'fulfilled' && healthResponse.value.success) {
        setServiceHealth(healthResponse.value.health);
      }

      if (agentsResponse.status === 'fulfilled' && agentsResponse.value.success) {
        setAgentStates(agentsResponse.value.agent_states);
      }

      if (memoryResponse.status === 'fulfilled' && memoryResponse.value.success) {
        setMemoryStats(memoryResponse.value.stats);
      }

      if (providersResponse.status === 'fulfilled' && providersResponse.value.success) {
        setSearchProviders(providersResponse.value.providers);
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initialize services
  const initializeServices = async () => {
    try {
      setIsInitializing(true);
      const response = await coreServicesClient.initializeServices();
      if (response.success) {
        await loadData();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize services');
    } finally {
      setIsInitializing(false);
    }
  };

  useEffect(() => {
    loadData();
    
    // Set up periodic refresh
    const interval = setInterval(loadData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, [loadData]);

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Service Health Status */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Service Health</h3>
        {serviceHealth ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                serviceHealth.memory_service ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <p className="text-sm">Memory Service</p>
            </div>
            <div className="text-center">
              <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                serviceHealth.state_service ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <p className="text-sm">State Service</p>
            </div>
            <div className="text-center">
              <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                serviceHealth.search_service ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <p className="text-sm">Search Service</p>
            </div>
            <div className="text-center">
              <div className={`w-4 h-4 rounded-full mx-auto mb-2 ${
                serviceHealth.vector_service ? 'bg-green-500' : 'bg-red-500'
              }`} />
              <p className="text-sm">Vector Service</p>
            </div>
          </div>
        ) : (
          <div className="text-center py-4">
            <Button 
              onClick={initializeServices} 
              disabled={isInitializing}
              className="mb-2"
            >
              {isInitializing ? 'Initializing...' : 'Initialize Services'}
            </Button>
            <p className="text-sm text-gray-500">Services not yet initialized</p>
          </div>
        )}
      </Card>

      {/* Agent States */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Agent States</h3>
        {Object.keys(agentStates).length > 0 ? (
          <div className="space-y-3">
            {Object.entries(agentStates).map(([agentId, state]) => (
              <div key={agentId} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="font-medium">{agentId}</p>
                  <p className="text-sm text-gray-600">{state.current_task || 'No active task'}</p>
                </div>
                <div className="text-right">
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    state.status === 'active' ? 'bg-green-100 text-green-800' :
                    state.status === 'busy' ? 'bg-yellow-100 text-yellow-800' :
                    state.status === 'error' ? 'bg-red-100 text-red-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {state.status}
                  </span>
                  {state.last_activity && (
                    <p className="text-xs text-gray-500 mt-1">
                      {new Date(state.last_activity).toLocaleTimeString()}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">No active agents</p>
        )}
      </Card>

      {/* Memory Statistics */}
      {memoryStats && (
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">Memory Statistics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(memoryStats).map(([key, value]) => (
              <div key={key} className="text-center">
                <p className="text-2xl font-bold text-blue-600">{String(value)}</p>
                <p className="text-sm text-gray-600 capitalize">{key.replace(/_/g, ' ')}</p>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );

  const renderMemoryManagement = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Memory Management</h3>
        <MemoryInterface />
      </Card>
    </div>
  );

  const renderStateMonitoring = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">State Monitoring</h3>
        <StateMonitor agentStates={agentStates} />
      </Card>
    </div>
  );

  const renderSearchInterface = () => (
    <div className="space-y-6">
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Search Interface</h3>
        <SearchInterface providers={searchProviders} />
      </Card>
    </div>
  );

  if (isLoading && !serviceHealth) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p>Loading core services...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {error && (
        <Card className="p-4 bg-red-50 border-red-200">
          <p className="text-red-800">{error}</p>
          <Button onClick={loadData} className="mt-2" size="sm">
            Retry
          </Button>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'memory', label: 'Memory' },
              { id: 'state', label: 'State' },
              { id: 'search', label: 'Search' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        <div className="mt-6">
          {activeTab === 'overview' && renderOverview()}
          {activeTab === 'memory' && renderMemoryManagement()}
          {activeTab === 'state' && renderStateMonitoring()}
          {activeTab === 'search' && renderSearchInterface()}
        </div>
      </Tabs>
    </div>
  );
}

// Memory Interface Component
function MemoryInterface() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MemoryItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState<string>('all');

  const searchMemory = async () => {
    if (!query.trim()) return;
    
    try {
      setIsSearching(true);
      const response = await coreServicesClient.retrieveMemory({
        query,
        layer: selectedLayer === 'all' ? undefined : selectedLayer,
        limit: 10
      });
      
      if (response.success) {
        setResults(response.memories);
      }
    } catch (error) {
      console.error('Memory search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <select 
          value={selectedLayer} 
          onChange={(e) => setSelectedLayer(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-md"
        >
          <option value="all">All Layers</option>
          <option value="knowledge_synthesis">Knowledge Synthesis</option>
          <option value="conversation_history">Conversation History</option>
          <option value="codebase_knowledge">Codebase Knowledge</option>
          <option value="stack_dependencies">Stack Dependencies</option>
        </select>
        
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search memory..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md"
          onKeyPress={(e) => e.key === 'Enter' && searchMemory()}
        />
        
        <Button onClick={searchMemory} disabled={isSearching}>
          {isSearching ? 'Searching...' : 'Search'}
        </Button>
      </div>

      {results.length > 0 && (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {results.map((item) => (
            <div key={item.id} className="p-3 bg-gray-50 rounded-lg">
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                  {item.layer} - {item.memory_type}
                </span>
                {item.similarity_score && (
                  <span className="text-xs text-gray-500">
                    {(item.similarity_score * 100).toFixed(1)}% match
                  </span>
                )}
              </div>
              <p className="text-sm">{item.content}</p>
              {item.tags && item.tags.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {item.tags.map((tag) => (
                    <span key={tag} className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
                      {tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// State Monitor Component
function StateMonitor({ agentStates }: { agentStates: Record<string, AgentState> }) {
  return (
    <div className="space-y-4">
      {Object.entries(agentStates).map(([agentId, state]) => (
        <div key={agentId} className="p-4 border border-gray-200 rounded-lg">
          <div className="flex justify-between items-start mb-3">
            <h4 className="font-medium">{agentId}</h4>
            <span className={`px-2 py-1 rounded-full text-xs ${
              state.status === 'active' ? 'bg-green-100 text-green-800' :
              state.status === 'busy' ? 'bg-yellow-100 text-yellow-800' :
              state.status === 'error' ? 'bg-red-100 text-red-800' :
              'bg-gray-100 text-gray-800'
            }`}>
              {state.status}
            </span>
          </div>
          
          {state.current_task && (
            <p className="text-sm text-gray-600 mb-2">Task: {state.current_task}</p>
          )}
          
          {state.error_info && (
            <p className="text-sm text-red-600 mb-2">Error: {state.error_info}</p>
          )}
          
          {state.last_activity && (
            <p className="text-xs text-gray-500">
              Last activity: {new Date(state.last_activity).toLocaleString()}
            </p>
          )}
          
          {Object.keys(state.performance_metrics).length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-100">
              <p className="text-xs font-medium text-gray-700 mb-2">Performance Metrics:</p>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {Object.entries(state.performance_metrics).map(([key, value]) => (
                  <div key={key}>
                    <span className="text-gray-500">{key}:</span> {String(value)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// Search Interface Component
function SearchInterface({ providers }: { providers: string[] }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchType, setSearchType] = useState<'knowledge' | 'code' | 'documentation'>('knowledge');

  const performSearch = async () => {
    if (!query.trim()) return;
    
    try {
      setIsSearching(true);
      const response = await coreServicesClient.searchKnowledge({
        query,
        search_type: searchType,
        max_results: 10
      });
      
      if (response.success) {
        setResults(response.results);
      }
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <select 
          value={searchType} 
          onChange={(e) => setSearchType(e.target.value as any)}
          className="px-3 py-2 border border-gray-300 rounded-md"
        >
          <option value="knowledge">Knowledge</option>
          <option value="code">Code</option>
          <option value="documentation">Documentation</option>
        </select>
        
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search..."
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md"
          onKeyPress={(e) => e.key === 'Enter' && performSearch()}
        />
        
        <Button onClick={performSearch} disabled={isSearching}>
          {isSearching ? 'Searching...' : 'Search'}
        </Button>
      </div>

      {providers.length > 0 && (
        <div className="text-sm text-gray-600">
          Available providers: {providers.join(', ')}
        </div>
      )}

      {results.length > 0 && (
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {results.map((result, index) => (
            <div key={index} className="p-3 bg-gray-50 rounded-lg">
              <div className="flex justify-between items-start mb-2">
                <h5 className="font-medium">{result.title}</h5>
                <div className="text-right">
                  <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                    {result.source}
                  </span>
                  <p className="text-xs text-gray-500 mt-1">
                    {(result.relevance_score * 100).toFixed(1)}% relevant
                  </p>
                </div>
              </div>
              <p className="text-sm text-gray-700">{result.content}</p>
              {result.url && (
                <a 
                  href={result.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-xs text-blue-600 hover:underline mt-2 inline-block"
                >
                  View Source
                </a>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default CoreServicesDashboard;