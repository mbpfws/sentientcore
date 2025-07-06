import { useState, useCallback, useEffect, useRef } from 'react';
import { PendingConfirmation } from './useOrchestratorState';

export interface ConfirmationAction {
  id: string;
  label: string;
  action: () => Promise<void> | void;
  variant?: 'default' | 'destructive' | 'secondary' | 'ghost';
  requiresDoubleConfirm?: boolean;
  icon?: string;
}

export interface ConfirmationDialog {
  id: string;
  title: string;
  message: string;
  description?: string;
  actions: ConfirmationAction[];
  priority: 'low' | 'medium' | 'high' | 'critical';
  timeout?: number;
  metadata?: any;
  timestamp: Date;
  category?: 'workflow' | 'research' | 'planning' | 'system' | 'user_action';
  requiresContext?: boolean;
  contextData?: any;
}

export interface ConfirmationHistory {
  id: string;
  dialog: ConfirmationDialog;
  action_taken: string;
  timestamp: Date;
  execution_time_ms: number;
  success: boolean;
  error?: string;
}

export interface ConfirmationManagerState {
  activeDialogs: ConfirmationDialog[];
  history: ConfirmationHistory[];
  isProcessing: boolean;
  currentDialog: ConfirmationDialog | null;
  queuedDialogs: ConfirmationDialog[];
  autoApprovalRules: Array<{
    id: string;
    pattern: string;
    category?: string;
    priority?: string;
    enabled: boolean;
    created_at: Date;
  }>;
  settings: {
    enableAutoApproval: boolean;
    maxConcurrentDialogs: number;
    defaultTimeout: number;
    enableSounds: boolean;
    enableNotifications: boolean;
    priorityThreshold: 'low' | 'medium' | 'high' | 'critical';
  };
}

const initialState: ConfirmationManagerState = {
  activeDialogs: [],
  history: [],
  isProcessing: false,
  currentDialog: null,
  queuedDialogs: [],
  autoApprovalRules: [],
  settings: {
    enableAutoApproval: false,
    maxConcurrentDialogs: 3,
    defaultTimeout: 30,
    enableSounds: true,
    enableNotifications: true,
    priorityThreshold: 'medium'
  }
};

export function useConfirmationManager() {
  const [state, setState] = useState<ConfirmationManagerState>(initialState);
  const stateRef = useRef(state);
  const timeoutRefs = useRef<Map<string, NodeJS.Timeout>>(new Map());
  
  // Keep ref in sync
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Load settings from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const stored = localStorage.getItem('confirmation_manager_settings');
        if (stored) {
          const settings = JSON.parse(stored);
          setState(prev => ({ ...prev, settings: { ...prev.settings, ...settings } }));
        }
        
        const storedRules = localStorage.getItem('confirmation_auto_approval_rules');
        if (storedRules) {
          const rules = JSON.parse(storedRules);
          setState(prev => ({ ...prev, autoApprovalRules: rules }));
        }
      } catch (error) {
        console.warn('Failed to load confirmation manager settings:', error);
      }
    }
  }, []);

  // Save settings to localStorage
  const saveSettings = useCallback((newSettings: Partial<ConfirmationManagerState['settings']>) => {
    const updatedSettings = { ...stateRef.current.settings, ...newSettings };
    setState(prev => ({ ...prev, settings: updatedSettings }));
    
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('confirmation_manager_settings', JSON.stringify(updatedSettings));
      } catch (error) {
        console.warn('Failed to save confirmation manager settings:', error);
      }
    }
  }, []);

  // Add auto-approval rule
  const addAutoApprovalRule = useCallback((pattern: string, category?: string, priority?: string) => {
    const newRule = {
      id: `rule_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      pattern,
      category,
      priority,
      enabled: true,
      created_at: new Date()
    };
    
    const updatedRules = [...stateRef.current.autoApprovalRules, newRule];
    setState(prev => ({ ...prev, autoApprovalRules: updatedRules }));
    
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('confirmation_auto_approval_rules', JSON.stringify(updatedRules));
      } catch (error) {
        console.warn('Failed to save auto-approval rules:', error);
      }
    }
    
    return newRule.id;
  }, []);

  // Check if dialog matches auto-approval rules
  const checkAutoApproval = useCallback((dialog: ConfirmationDialog): boolean => {
    if (!stateRef.current.settings.enableAutoApproval) return false;
    
    const priorityOrder = { low: 1, medium: 2, high: 3, critical: 4 };
    const thresholdLevel = priorityOrder[stateRef.current.settings.priorityThreshold];
    const dialogLevel = priorityOrder[dialog.priority];
    
    // Don't auto-approve if priority is above threshold
    if (dialogLevel > thresholdLevel) return false;
    
    return stateRef.current.autoApprovalRules.some(rule => {
      if (!rule.enabled) return false;
      
      // Check category match
      if (rule.category && dialog.category !== rule.category) return false;
      
      // Check priority match
      if (rule.priority && dialog.priority !== rule.priority) return false;
      
      // Check pattern match
      try {
        const regex = new RegExp(rule.pattern, 'i');
        return regex.test(dialog.message) || regex.test(dialog.title);
      } catch (error) {
        console.warn('Invalid regex pattern in auto-approval rule:', rule.pattern);
        return false;
      }
    });
  }, []);

  // Create confirmation dialog
  const createConfirmation = useCallback((config: {
    title: string;
    message: string;
    description?: string;
    actions?: Omit<ConfirmationAction, 'id'>[];
    priority?: 'low' | 'medium' | 'high' | 'critical';
    timeout?: number;
    category?: 'workflow' | 'research' | 'planning' | 'system' | 'user_action';
    metadata?: any;
    requiresContext?: boolean;
    contextData?: any;
  }): string => {
    const dialogId = `dialog_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Default actions if none provided
    const defaultActions: ConfirmationAction[] = [
      {
        id: `${dialogId}_approve`,
        label: 'Approve',
        action: () => handleConfirmationResponse(dialogId, 'approve'),
        variant: 'default'
      },
      {
        id: `${dialogId}_reject`,
        label: 'Reject',
        action: () => handleConfirmationResponse(dialogId, 'reject'),
        variant: 'secondary'
      }
    ];
    
    const dialog: ConfirmationDialog = {
      id: dialogId,
      title: config.title,
      message: config.message,
      description: config.description,
      actions: config.actions?.map(action => ({
        ...action,
        id: `${dialogId}_${action.label.toLowerCase().replace(/\s+/g, '_')}`
      })) || defaultActions,
      priority: config.priority || 'medium',
      timeout: config.timeout || stateRef.current.settings.defaultTimeout,
      metadata: config.metadata,
      timestamp: new Date(),
      category: config.category || 'user_action',
      requiresContext: config.requiresContext,
      contextData: config.contextData
    };
    
    // Check for auto-approval
    if (checkAutoApproval(dialog)) {
      // Auto-approve with first action
      setTimeout(() => {
        if (dialog.actions.length > 0) {
          executeAction(dialog.actions[0], dialog);
        }
      }, 100);
      return dialogId;
    }
    
    // Add to queue or active dialogs
    if (stateRef.current.activeDialogs.length >= stateRef.current.settings.maxConcurrentDialogs) {
      setState(prev => ({
        ...prev,
        queuedDialogs: [...prev.queuedDialogs, dialog].sort((a, b) => {
          const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
          return priorityOrder[b.priority] - priorityOrder[a.priority];
        })
      }));
    } else {
      setState(prev => ({
        ...prev,
        activeDialogs: [...prev.activeDialogs, dialog].sort((a, b) => {
          const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
          return priorityOrder[b.priority] - priorityOrder[a.priority];
        }),
        currentDialog: prev.currentDialog || dialog
      }));
      
      // Set timeout
      if (dialog.timeout && dialog.timeout > 0) {
        const timeoutId = setTimeout(() => {
          handleTimeout(dialogId);
        }, dialog.timeout * 1000);
        timeoutRefs.current.set(dialogId, timeoutId);
      }
      
      // Play notification sound
      if (stateRef.current.settings.enableSounds) {
        playNotificationSound(dialog.priority);
      }
      
      // Show browser notification
      if (stateRef.current.settings.enableNotifications && 'Notification' in window) {
        if (Notification.permission === 'granted') {
          new Notification(`Confirmation Required: ${dialog.title}`, {
            body: dialog.message,
            icon: '/favicon.ico',
            tag: dialogId
          });
        } else if (Notification.permission !== 'denied') {
          Notification.requestPermission();
        }
      }
    }
    
    return dialogId;
  }, [checkAutoApproval]);

  // Execute action with error handling and timing
  const executeAction = useCallback(async (action: ConfirmationAction, dialog: ConfirmationDialog) => {
    const startTime = Date.now();
    setState(prev => ({ ...prev, isProcessing: true }));
    
    try {
      await action.action();
      
      const executionTime = Date.now() - startTime;
      const historyEntry: ConfirmationHistory = {
        id: `history_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        dialog,
        action_taken: action.label,
        timestamp: new Date(),
        execution_time_ms: executionTime,
        success: true
      };
      
      setState(prev => ({
        ...prev,
        history: [historyEntry, ...prev.history].slice(0, 100), // Keep last 100 entries
        isProcessing: false
      }));
      
      removeDialog(dialog.id);
    } catch (error) {
      const executionTime = Date.now() - startTime;
      const historyEntry: ConfirmationHistory = {
        id: `history_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        dialog,
        action_taken: action.label,
        timestamp: new Date(),
        execution_time_ms: executionTime,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
      
      setState(prev => ({
        ...prev,
        history: [historyEntry, ...prev.history].slice(0, 100),
        isProcessing: false
      }));
      
      console.error('Confirmation action failed:', error);
      // Don't remove dialog on error, let user retry
    }
  }, []);

  // Handle confirmation response
  const handleConfirmationResponse = useCallback((dialogId: string, response: string, metadata?: any) => {
    const dialog = stateRef.current.activeDialogs.find(d => d.id === dialogId);
    if (!dialog) return;
    
    const action = dialog.actions.find(a => a.label.toLowerCase() === response.toLowerCase());
    if (action) {
      executeAction(action, dialog);
    } else {
      console.warn('Unknown confirmation response:', response);
      removeDialog(dialogId);
    }
  }, [executeAction]);

  // Handle timeout
  const handleTimeout = useCallback((dialogId: string) => {
    const dialog = stateRef.current.activeDialogs.find(d => d.id === dialogId);
    if (!dialog) return;
    
    const historyEntry: ConfirmationHistory = {
      id: `history_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      dialog,
      action_taken: 'timeout',
      timestamp: new Date(),
      execution_time_ms: (dialog.timeout || 0) * 1000,
      success: false,
      error: 'Confirmation timed out'
    };
    
    setState(prev => ({
      ...prev,
      history: [historyEntry, ...prev.history].slice(0, 100)
    }));
    
    removeDialog(dialogId);
  }, []);

  // Remove dialog and process queue
  const removeDialog = useCallback((dialogId: string) => {
    // Clear timeout
    const timeoutId = timeoutRefs.current.get(dialogId);
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutRefs.current.delete(dialogId);
    }
    
    setState(prev => {
      const updatedActiveDialogs = prev.activeDialogs.filter(d => d.id !== dialogId);
      const updatedQueuedDialogs = [...prev.queuedDialogs];
      
      // Move queued dialog to active if space available
      let newCurrentDialog = prev.currentDialog;
      if (updatedActiveDialogs.length < prev.settings.maxConcurrentDialogs && updatedQueuedDialogs.length > 0) {
        const nextDialog = updatedQueuedDialogs.shift()!;
        updatedActiveDialogs.push(nextDialog);
        
        // Set timeout for new dialog
        if (nextDialog.timeout && nextDialog.timeout > 0) {
          const timeoutId = setTimeout(() => {
            handleTimeout(nextDialog.id);
          }, nextDialog.timeout * 1000);
          timeoutRefs.current.set(nextDialog.id, timeoutId);
        }
      }
      
      // Update current dialog
      if (prev.currentDialog?.id === dialogId) {
        newCurrentDialog = updatedActiveDialogs[0] || null;
      }
      
      return {
        ...prev,
        activeDialogs: updatedActiveDialogs,
        queuedDialogs: updatedQueuedDialogs,
        currentDialog: newCurrentDialog
      };
    });
  }, [handleTimeout]);

  // Play notification sound based on priority
  const playNotificationSound = useCallback((priority: 'low' | 'medium' | 'high' | 'critical') => {
    if (typeof window === 'undefined' || !stateRef.current.settings.enableSounds) return;
    
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      // Different frequencies for different priorities
      const frequencies = { low: 400, medium: 600, high: 800, critical: 1000 };
      oscillator.frequency.setValueAtTime(frequencies[priority], audioContext.currentTime);
      
      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
      
      oscillator.start(audioContext.currentTime);
      oscillator.stop(audioContext.currentTime + 0.3);
    } catch (error) {
      console.warn('Failed to play notification sound:', error);
    }
  }, []);

  // Clear all dialogs
  const clearAllDialogs = useCallback(() => {
    // Clear all timeouts
    timeoutRefs.current.forEach(timeoutId => clearTimeout(timeoutId));
    timeoutRefs.current.clear();
    
    setState(prev => ({
      ...prev,
      activeDialogs: [],
      queuedDialogs: [],
      currentDialog: null,
      isProcessing: false
    }));
  }, []);

  // Get statistics
  const getStats = useCallback(() => {
    const now = Date.now();
    const last24h = stateRef.current.history.filter(h => now - h.timestamp.getTime() < 24 * 60 * 60 * 1000);
    
    return {
      total_confirmations: stateRef.current.history.length,
      last_24h: last24h.length,
      success_rate: stateRef.current.history.length > 0 ? 
        stateRef.current.history.filter(h => h.success).length / stateRef.current.history.length : 0,
      average_response_time: stateRef.current.history.length > 0 ?
        stateRef.current.history.reduce((sum, h) => sum + h.execution_time_ms, 0) / stateRef.current.history.length : 0,
      active_dialogs: stateRef.current.activeDialogs.length,
      queued_dialogs: stateRef.current.queuedDialogs.length,
      auto_approval_rules: stateRef.current.autoApprovalRules.filter(r => r.enabled).length
    };
  }, []);

  return {
    // State
    state,
    
    // Actions
    createConfirmation,
    handleConfirmationResponse,
    removeDialog,
    clearAllDialogs,
    saveSettings,
    addAutoApprovalRule,
    
    // Computed values
    hasActiveConfirmations: state.activeDialogs.length > 0,
    hasQueuedConfirmations: state.queuedDialogs.length > 0,
    totalPendingConfirmations: state.activeDialogs.length + state.queuedDialogs.length,
    currentPriority: state.currentDialog?.priority || 'medium',
    stats: getStats()
  };
}

export default useConfirmationManager;