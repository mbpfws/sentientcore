import { useState, useCallback, useEffect, useRef } from 'react';
import { coreServicesClient, AgentState, WorkflowState } from '../api';

// Enhanced interfaces for better state management
export interface OrchestratorMessage {
  id: string;
  type: 'user' | 'orchestrator' | 'agent' | 'system' | 'confirmation' | 'artifact';
  content: string;
  timestamp: Date;
  metadata?: {
    agent_id?: string;
    workflow_id?: string;
    action_type?: string;
    status?: string;
    requires_confirmation?: boolean;
    confirmation_id?: string;
    artifact_type?: 'research' | 'plan' | 'specification';
    download_url?: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
  };
}

export interface ActiveWorkflow {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed';
  progress: number;
  current_step?: string;
  agents_involved: string[];
  started_at?: Date;
  artifacts?: Array<{
    id: string;
    type: 'research' | 'plan' | 'specification';
    title: string;
    url: string;
    created_at: Date;
  }>;
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

export interface PendingConfirmation {
  id: string;
  message: string;
  action: string;
  metadata?: any;
  timestamp: Date;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  timeout?: number; // Auto-timeout in seconds
}

export interface ConversationContext {
  user_intent: string;
  requirements_gathered: boolean;
  research_needed: boolean;
  planning_phase: boolean;
  current_focus: 'requirements_gathering' | 'research' | 'planning' | 'implementation' | 'testing';
  artifacts_generated: string[];
  session_metadata?: {
    start_time: Date;
    last_activity: Date;
    total_messages: number;
    completed_actions: number;
  };
}

export interface ExecutionState {
  current_action: string | null;
  action_queue: Array<{
    id: string;
    action: string;
    priority: 'low' | 'medium' | 'high' | 'critical';
    metadata?: any;
    created_at: Date;
  }>;
  is_executing: boolean;
  last_execution: Date | null;
}

export interface OrchestratorState {
  // Core state
  messages: OrchestratorMessage[];
  agentStates: Record<string, AgentState>;
  activeWorkflows: ActiveWorkflow[];
  pendingConfirmations: PendingConfirmation[];
  conversationContext: ConversationContext;
  executionState: ExecutionState;
  
  // Connection state
  isConnected: boolean;
  sessionId: string;
  orchestratorMode: 'intelligent' | 'multi_agent' | 'legacy';
  
  // UI state
  isProcessing: boolean;
  isWaitingForConfirmation: boolean;
  autoRefresh: boolean;
  selectedWorkflow: string | null;
  
  // Error state
  lastError: string | null;
  errorCount: number;
}

const initialState: OrchestratorState = {
  messages: [],
  agentStates: {},
  activeWorkflows: [],
  pendingConfirmations: [],
  conversationContext: {
    user_intent: '',
    requirements_gathered: false,
    research_needed: false,
    planning_phase: false,
    current_focus: 'requirements_gathering',
    artifacts_generated: [],
    session_metadata: {
      start_time: new Date(),
      last_activity: new Date(),
      total_messages: 0,
      completed_actions: 0
    }
  },
  executionState: {
    current_action: null,
    action_queue: [],
    is_executing: false,
    last_execution: null
  },
  isConnected: false,
  sessionId: '',
  orchestratorMode: 'intelligent',
  isProcessing: false,
  isWaitingForConfirmation: false,
  autoRefresh: true,
  selectedWorkflow: null,
  lastError: null,
  errorCount: 0
};

export function useOrchestratorState() {
  const [state, setState] = useState<OrchestratorState>(initialState);
  const stateRef = useRef(state);
  
  // Keep ref in sync with state for callbacks
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Enhanced state updater with persistence
  const updateState = useCallback((updates: Partial<OrchestratorState>) => {
    setState(prevState => {
      const newState = { ...prevState, ...updates };
      
      // Update session metadata
      if (updates.messages || updates.conversationContext) {
        newState.conversationContext = {
          ...newState.conversationContext,
          session_metadata: {
            ...newState.conversationContext.session_metadata!,
            last_activity: new Date(),
            total_messages: newState.messages.length,
            completed_actions: newState.conversationContext.session_metadata?.completed_actions || 0
          }
        };
      }
      
      // Persist critical state to localStorage
      if (typeof window !== 'undefined' && newState.sessionId) {
        try {
          const persistentState = {
            sessionId: newState.sessionId,
            conversationContext: newState.conversationContext,
            orchestratorMode: newState.orchestratorMode,
            lastActivity: new Date().toISOString()
          };
          localStorage.setItem(`orchestrator_state_${newState.sessionId}`, JSON.stringify(persistentState));
        } catch (error) {
          console.warn('Failed to persist state:', error);
        }
      }
      
      return newState;
    });
  }, []);

  // Add message with enhanced features
  const addMessage = useCallback((message: Omit<OrchestratorMessage, 'id' | 'timestamp'>) => {
    const newMessage: OrchestratorMessage = {
      ...message,
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
    };
    
    updateState({
      messages: [...stateRef.current.messages, newMessage],
      lastError: null // Clear error on successful message
    });
    
    // Store in backend if session exists
    if (stateRef.current.sessionId) {
      coreServicesClient.storeMemory(stateRef.current.sessionId, {
        content: { ...newMessage },
        metadata: {
          session_id: stateRef.current.sessionId,
          workflow_mode: stateRef.current.orchestratorMode
        }
      }).catch(error => {
        console.error('Failed to store message:', error);
        updateState({
          lastError: `Failed to store message: ${error.message}`,
          errorCount: stateRef.current.errorCount + 1
        });
      });
    }
    
    return newMessage.id;
  }, [updateState]);

  // Enhanced confirmation handling with queue management
  const addConfirmation = useCallback((confirmation: Omit<PendingConfirmation, 'id' | 'timestamp'>) => {
    const newConfirmation: PendingConfirmation = {
      ...confirmation,
      id: `conf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      priority: confirmation.priority || 'medium'
    };
    
    // Sort confirmations by priority
    const updatedConfirmations = [...stateRef.current.pendingConfirmations, newConfirmation]
      .sort((a, b) => {
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority || 'medium'] - priorityOrder[a.priority || 'medium'];
      });
    
    updateState({
      pendingConfirmations: updatedConfirmations,
      isWaitingForConfirmation: true
    });
    
    // Auto-timeout if specified
    if (newConfirmation.timeout) {
      setTimeout(() => {
        if (stateRef.current.pendingConfirmations.find(c => c.id === newConfirmation.id)) {
          removeConfirmation(newConfirmation.id);
          addMessage({
            type: 'system',
            content: `⏰ Confirmation timeout: ${newConfirmation.message}`,
            metadata: { action_type: 'timeout', confirmation_id: newConfirmation.id }
          });
        }
      }, newConfirmation.timeout * 1000);
    }
    
    return newConfirmation.id;
  }, [updateState, addMessage]);

  // Remove confirmation
  const removeConfirmation = useCallback((confirmationId: string) => {
    const updatedConfirmations = stateRef.current.pendingConfirmations.filter(c => c.id !== confirmationId);
    updateState({
      pendingConfirmations: updatedConfirmations,
      isWaitingForConfirmation: updatedConfirmations.length > 0
    });
  }, [updateState]);

  // Enhanced action queue management for one-action-at-a-time execution
  const queueAction = useCallback((action: string, priority: 'low' | 'medium' | 'high' | 'critical' = 'medium', metadata?: any) => {
    const newAction = {
      id: `action_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      action,
      priority,
      metadata,
      created_at: new Date()
    };
    
    const currentQueue = stateRef.current.executionState.action_queue || [];
    const updatedQueue = [...currentQueue, newAction]
      .sort((a, b) => {
        const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      });
    
    updateState({
      executionState: {
        ...stateRef.current.executionState,
        action_queue: updatedQueue
      }
    });
    
    // Auto-execute if not currently executing
    if (!stateRef.current.executionState.is_executing) {
      executeNextAction();
    }
    
    return newAction.id;
  }, [updateState]);

  // Execute next action in queue
  const executeNextAction = useCallback(async () => {
    if (stateRef.current.executionState.is_executing || !stateRef.current.executionState.action_queue || stateRef.current.executionState.action_queue.length === 0) {
      return;
    }
    
    const nextAction = stateRef.current.executionState.action_queue[0];
    const remainingQueue = (stateRef.current.executionState.action_queue || []).slice(1);
    
    updateState({
      executionState: {
        ...stateRef.current.executionState,
        current_action: nextAction.action,
        action_queue: remainingQueue,
        is_executing: true
      }
    });
    
    try {
      // Execute the action based on type
      await executeAction(nextAction);
      
      // Mark as completed
      updateState({
        executionState: {
          ...stateRef.current.executionState,
          current_action: null,
          is_executing: false,
          last_execution: new Date()
        },
        conversationContext: {
          ...stateRef.current.conversationContext,
          session_metadata: {
            ...stateRef.current.conversationContext.session_metadata!,
            completed_actions: (stateRef.current.conversationContext.session_metadata?.completed_actions || 0) + 1
          }
        }
      });
      
      // Execute next action if queue is not empty
      if (remainingQueue.length > 0) {
        setTimeout(executeNextAction, 100); // Small delay to prevent overwhelming
      }
    } catch (error) {
      console.error('Action execution failed:', error);
      updateState({
        executionState: {
          ...stateRef.current.executionState,
          current_action: null,
          is_executing: false
        },
        lastError: `Action failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        errorCount: stateRef.current.errorCount + 1
      });
      
      addMessage({
        type: 'system',
        content: `❌ Action failed: ${nextAction.action} - ${error instanceof Error ? error.message : 'Unknown error'}`,
        metadata: { action_type: 'error', action_id: nextAction.id }
      });
    }
  }, [updateState, addMessage]);

  // Execute specific action
  const executeAction = useCallback(async (action: any) => {
    switch (action.action) {
      case 'load_agent_states':
        const agentResponse = await coreServicesClient.getAgentStates();
        if (agentResponse.success) {
          updateState({ agentStates: agentResponse.agent_states });
        }
        break;
        
      case 'update_conversation_context':
        if (action.metadata?.context && stateRef.current.sessionId) {
          await coreServicesClient.updateConversationContext(stateRef.current.sessionId, action.metadata.context);
          updateState({ conversationContext: action.metadata.context });
        }
        break;
        
      case 'store_conversation':
        if (action.metadata?.message && stateRef.current.sessionId) {
          await coreServicesClient.storeConversation(action.metadata.message, {
            orchestrator_mode: stateRef.current.orchestratorMode,
            timestamp: new Date().toISOString(),
            session_id: stateRef.current.sessionId,
            context: stateRef.current.conversationContext
          });
        }
        break;
        
      default:
        console.warn('Unknown action type:', action.action);
    }
  }, [updateState]);

  // Initialize session with enhanced features
  const initializeSession = useCallback(async (mode: 'intelligent' | 'multi_agent' | 'legacy' = 'intelligent') => {
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Try to restore from localStorage
    let restoredState = null;
    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem(`orchestrator_state_${sessionId}`);
        if (stored) {
          restoredState = JSON.parse(stored);
        }
      } catch (error) {
        console.warn('Failed to restore state:', error);
      }
    }
    
    updateState({
      sessionId,
      orchestratorMode: mode,
      isConnected: false,
      conversationContext: restoredState?.conversationContext || {
        ...initialState.conversationContext,
        session_metadata: {
          start_time: new Date(),
          last_activity: new Date(),
          total_messages: 0,
          completed_actions: 0
        }
      }
    });
    
    // Initialize backend services
    queueAction('initialize_services', 'high');
    queueAction('load_agent_states', 'medium');
    
    return sessionId;
  }, [updateState, queueAction]);

  // Clear all state
  const clearState = useCallback(() => {
    updateState(initialState);
    
    // Clear localStorage
    if (typeof window !== 'undefined' && state.sessionId) {
      try {
        localStorage.removeItem(`orchestrator_state_${state.sessionId}`);
      } catch (error) {
        console.warn('Failed to clear stored state:', error);
      }
    }
  }, [updateState, state.sessionId]);

  return {
    // State
    state,
    
    // Actions
    updateState,
    addMessage,
    addConfirmation,
    removeConfirmation,
    queueAction,
    executeNextAction,
    initializeSession,
    clearState,
    
    // Computed values
    hasActiveActions: (state.executionState.action_queue?.length || 0) > 0 || state.executionState.is_executing,
    nextAction: state.executionState.action_queue?.[0] || null,
    isHealthy: state.errorCount < 5 && state.isConnected,
    sessionDuration: state.conversationContext.session_metadata ? 
      Date.now() - state.conversationContext.session_metadata.start_time.getTime() : 0
  };
}

export default useOrchestratorState;