'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { Agent, Workflow } from '../api/types';
import { AgentService, WorkflowService } from '../api';

interface AppState {
  activeWorkflow: string | null;
  workflowsList: Workflow[];
  agents: Agent[];
  isLoading: boolean;
  error: string | null;
}

interface AppContextValue extends AppState {
  setActiveWorkflow: (workflowId: string) => void;
  resetSession: () => void;
  refreshAgents: () => Promise<void>;
  refreshWorkflows: () => Promise<void>;
}

const initialState: AppState = {
  activeWorkflow: 'intelligent', // Default to intelligent workflow mode
  workflowsList: [],
  agents: [],
  isLoading: false,
  error: null,
};

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AppState>(initialState);

  const setActiveWorkflow = (workflowId: string) => {
    setState((prev) => ({ ...prev, activeWorkflow: workflowId }));
  };

  const resetSession = () => {
    setState((prev) => ({
      ...prev,
      activeWorkflow: 'intelligent',
    }));
    // Here we would also clear chat history via API
  };

  const refreshAgents = async () => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));
    try {
      const agentsData = await AgentService.getAgents();
      setState((prev) => ({
        ...prev,
        agents: agentsData,
        isLoading: false,
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: 'Failed to load agents',
        isLoading: false,
      }));
      console.error('Error loading agents:', error);
    }
  };

  const refreshWorkflows = async () => {
    setState((prev) => ({ ...prev, isLoading: true, error: null }));
    try {
      const workflowsData = await WorkflowService.getWorkflows();
      setState((prev) => ({
        ...prev,
        workflowsList: workflowsData,
        isLoading: false,
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: 'Failed to load workflows',
        isLoading: false,
      }));
      console.error('Error loading workflows:', error);
    }
  };

  // Initial data loading
  useEffect(() => {
    const loadInitialData = async () => {
      setState((prev) => ({ ...prev, isLoading: true }));
      try {
        await Promise.all([refreshAgents(), refreshWorkflows()]);
      } catch (error) {
        setState((prev) => ({
          ...prev,
          error: 'Failed to load initial data',
          isLoading: false,
        }));
      }
    };

    loadInitialData();
  }, []);

  return (
    <AppContext.Provider
      value={{
        ...state,
        setActiveWorkflow,
        resetSession,
        refreshAgents,
        refreshWorkflows,
      }}
    >
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}
