'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Textarea } from '@/components/ui/textarea';
import { 
  Send, 
  Bot, 
  User, 
  Settings, 
  RefreshCw, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  Search,
  FileText,
  Zap,
  Users,
  Activity,
  Download,
  Eye,
  MessageSquare,
  Lightbulb,
  Target,
  Workflow,
  Play,
  Pause,
  Square,
  Star,
  Filter,
  Grid,
  List,
  Timeline,
  Brain
} from 'lucide-react';
import { chatService, coreServicesClient, AgentState, WorkflowState } from '@/lib/api';
import { ChatService } from '@/lib/api/chat-service';
import { ResearchService } from '@/lib/api/research-service';
import { useOrchestratorState } from '@/lib/hooks/useOrchestratorState';
import { useConfirmationManager } from '@/lib/hooks/useConfirmationManager';
import { useArtifactManager } from '@/lib/hooks/useArtifactManager';
import { conversationProcessor } from '@/lib/conversation/conversationProcessor';
import { conversationFlowManager } from '@/lib/conversation/conversationFlowManager';

interface OrchestratorInterfaceProps {
  className?: string;
}

interface OrchestratorMessage {
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
  };
}

interface ActiveWorkflow {
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
}

interface PendingConfirmation {
  id: string;
  message: string;
  action: string;
  metadata?: any;
  timestamp: Date;
}

interface ConversationContext {
  user_intent: string;
  requirements_gathered: boolean;
  research_needed: boolean;
  planning_phase: boolean;
  current_focus: string;
  artifacts_generated: string[];
}

export function OrchestratorInterface({ className }: OrchestratorInterfaceProps) {
  // Enhanced state management hooks
  const {
    state: orchestratorState,
    addMessage,
    addConfirmation,
    removeConfirmation,
    queueAction,
    executeNextAction,
    initializeSession,
    clearState,
    hasActiveActions,
    nextAction,
    isHealthy,
    sessionDuration
  } = useOrchestratorState();
  
  const {
    createConfirmation,
    handleConfirmationResponse,
    removeDialog,
    clearAllDialogs,
    saveSettings: saveConfirmationSettings,
    hasActiveConfirmations,
    totalPendingConfirmations,
    currentPriority,
    stats: confirmationStats
  } = useConfirmationManager();
  
  const {
    state: artifactState,
    loadArtifacts,
    createArtifact,
    selectArtifact,
    toggleFavorite,
    searchArtifacts,
    filteredArtifacts,
    stats: artifactStats,
    hasArtifacts
  } = useArtifactManager();
  
  // Local UI state
   const [input, setInput] = useState('');
   const [selectedTab, setSelectedTab] = useState('conversation');
   const [showSettings, setShowSettings] = useState(false);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [orchestratorMode, setOrchestratorMode] = useState<'intelligent' | 'multi_agent' | 'legacy'>('intelligent');
  
  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatService = new ChatService();
  const researchService = new ResearchService();
  
  // Derived state from hooks
  const messages = orchestratorState.messages;
  const isProcessing = orchestratorState.isProcessing;
  const agentStates = orchestratorState.agentStates;
  const activeWorkflows = orchestratorState.activeWorkflows;
  const isConnected = orchestratorState.isConnected;
  const sessionId = orchestratorState.sessionId;
  const pendingConfirmations = orchestratorState.pendingConfirmations;
  const conversationContext = orchestratorState.conversationContext;
  const isWaitingForConfirmation = hasActiveConfirmations;
  
  // Additional derived states for enhanced UI
  const hasActiveActionsLocal = orchestratorState.actionQueue.length > 0;
  const nextActionLocal = orchestratorState.actionQueue[0] || null;
  const isHealthyLocal = orchestratorState.executionStats.errors < 3;
  
  const sessionDurationLocal = React.useMemo(() => {
    if (!orchestratorState.sessionStartTime) return '0m';
    const duration = Date.now() - orchestratorState.sessionStartTime;
    const minutes = Math.floor(duration / 60000);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    return `${minutes}m`;
  }, [orchestratorState.sessionStartTime]);

  // Initialize orchestrator connection
  const initializeOrchestrator = useCallback(async () => {
    try {
      // Initialize session using enhanced state management
      const newSessionId = await initializeSession();
      
      // Initialize core services
      const response = await coreServicesClient.initializeServices();
      if (response.success) {
        // Load artifacts
        await loadArtifacts();
        
        // Add welcome message
        addMessage({
          type: 'orchestrator',
          content: `🤖 AI Orchestrator initialized in ${orchestratorMode} mode with enhanced state management. How can I help you today?`,
          metadata: { action_type: 'initialization' }
        });
      }
    } catch (error) {
      console.error('Failed to initialize orchestrator:', error);
      addMessage({
        type: 'system',
        content: `❌ Failed to connect to orchestrator: ${error instanceof Error ? error.message : 'Unknown error'}`,
        metadata: { action_type: 'error' }
      });
    }
  }, [initializeSession, loadArtifacts, addMessage, orchestratorMode]);

  // Load agent states
  const loadAgentStates = useCallback(async () => {
    try {
      const response = await coreServicesClient.getAgentStates();
      if (response.success) {
        setAgentStates(response.agent_states);
      }
    } catch (error) {
      console.error('Failed to load agent states:', error);
    }
  }, []);

  // Scroll to bottom of messages
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Add message to conversation with enhanced features
  const addMessage = useCallback((message: Omit<OrchestratorMessage, 'id' | 'timestamp'>) => {
    const newMessage: OrchestratorMessage = {
      ...message,
      id: Date.now().toString(),
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newMessage]);
    
    // Store in session memory
    if (sessionId) {
      coreServicesClient.storeMemory({
        layer: 'session',
        type: 'conversation',
        key: `message_${newMessage.id}`,
        data: newMessage,
        metadata: { session_id: sessionId, workflow_mode: orchestratorMode }
      }).catch(console.error);
    }
  }, [sessionId, orchestratorMode]);

  // Handle confirmation responses using enhanced confirmation manager
  const handleConfirmationLocal = useCallback(async (confirmationId: string, approved: boolean) => {
    try {
      // Use the enhanced confirmation manager
      const result = await handleConfirmationResponse(confirmationId, approved);
      
      if (result.success) {
        if (approved && result.action) {
          // Queue the confirmed action for execution
          await queueAction({
            id: `action_${Date.now()}`,
            type: 'confirmed_action',
            priority: 'high',
            data: result.action,
            execute: async (actionData) => {
              await executeConfirmedAction(actionData);
            }
          });
          
          // Execute the next action in queue
          await executeNextAction();
        } else if (!approved) {
          // Add rejection message
          addMessage({
            type: 'system',
            content: `❌ Action was declined by user.`,
            metadata: { action_type: 'confirmation_declined', confirmation_id: confirmationId }
          });
        }
      }
      
    } catch (error) {
      console.error('Error handling confirmation:', error);
      addMessage({
        type: 'system',
        content: 'Error occurred while processing confirmation. Please try again.',
        metadata: { action_type: 'error' }
      });
    }
  }, [handleConfirmationResponse, queueAction, executeNextAction, addMessage]);

  // Execute confirmed actions
  const executeConfirmedAction = useCallback(async (confirmation: PendingConfirmation) => {
    try {
      switch (confirmation.action) {
        case 'start_research':
          const researchResult = await researchService.startResearch(confirmation.metadata);
          addMessage({
            type: 'system',
            content: `🔍 Research started: ${confirmation.metadata.query}`,
            metadata: { 
              action_type: 'research_started',
              workflow_id: researchResult.data?.workflow_id 
            }
          });
          break;
        case 'transition_to_planning':
          setConversationContext(prev => ({ ...prev, planning_phase: true, current_focus: 'planning' }));
          addMessage({
            type: 'orchestrator',
            content: '🎯 Transitioning to planning phase. I will now coordinate with the architect planner to create detailed specifications.',
            metadata: { action_type: 'phase_transition' }
          });
          break;
        default:
          console.warn('Unknown confirmation action:', confirmation.action);
      }
    } catch (error) {
      console.error('Error executing confirmed action:', error);
      addMessage({
        type: 'system',
        content: `❌ Error executing action: ${error}`,
        metadata: { action_type: 'error' }
      });
    }
  }, [researchService, addMessage, setConversationContext]);

  // Add system message
  const addSystemMessage = (content: string) => {
    addMessage({
      type: 'system',
      content
    });
  };

  // Add orchestrator message
  const addOrchestratorMessage = (content: string, metadata?: any) => {
    const message: OrchestratorMessage = {
      id: Date.now().toString(),
      type: 'orchestrator',
      content,
      timestamp: new Date(),
      metadata
    };
    setMessages(prev => [...prev, message]);
  };

  // Add agent message
  const addAgentMessage = (content: string, agentId: string, metadata?: any) => {
    const message: OrchestratorMessage = {
      id: Date.now().toString(),
      type: 'agent',
      content,
      timestamp: new Date(),
      metadata: { agent_id: agentId, ...metadata }
    };
    setMessages(prev => [...prev, message]);
  };

  // Enhanced send message with intelligent conversation flow and state management
  const sendMessage = useCallback(async () => {
    if (!input.trim() || isProcessing || isWaitingForConfirmation) return;

    const userMessage = input.trim();
    setInput('');

    // Add user message using enhanced state management
    addMessage({
      type: 'user',
      content: userMessage,
    });

    try {
      // Initialize conversation flow if needed
      if (!conversationFlowManager.getCurrentFlow()) {
        const flowType = userMessage.toLowerCase().includes('create') || userMessage.toLowerCase().includes('build') 
          ? 'development_project' : 'research_task';
        conversationFlowManager.initializeFlow(flowType, { user_message: userMessage });
      }

      // Process message through conversation processor
      const processedResponse = await conversationProcessor.processMessage(userMessage, {
        session_id: sessionId,
        orchestrator_mode: orchestratorMode,
        conversation_context: conversationContext,
        agent_states: agentStates,
        active_workflows: activeWorkflows
      });

      // Update conversation flow based on processed response
      const flowAnalysis = conversationFlowManager.analyzeMessage(userMessage, conversationContext);
      const updatedContext = conversationFlowManager.updateContext(conversationContext, flowAnalysis);
      
      // Queue the message processing action
      await queueAction({
        id: `action_${Date.now()}`,
        type: 'send_message',
        priority: 'high',
        data: {
          message: userMessage,
          session_id: sessionId,
          processed_response: processedResponse,
          flow_analysis: flowAnalysis,
          context: {
            orchestrator_mode: orchestratorMode,
            conversation_context: updatedContext,
            agent_states: agentStates,
            active_workflows: activeWorkflows.map(w => ({ id: w.id, status: w.status, progress: w.progress }))
          }
        },
        execute: async (actionData) => {
          // Store the conversation in memory with enhanced context
          await conversationProcessor.storeConversationMemory(
            actionData.session_id,
            `User: ${actionData.message}`,
            { 
              orchestrator_mode: actionData.context.orchestrator_mode,
              timestamp: new Date().toISOString(),
              session_id: actionData.session_id,
              context: actionData.context.conversation_context,
              flow_analysis: actionData.flow_analysis
            }
          );

          // Send to chat service with enhanced context
          const response = await chatService.sendMessage({
            message: actionData.message,
            workflow_mode: actionData.context.orchestrator_mode,
            research_mode: 'knowledge',
            session_id: actionData.session_id,
            context: actionData.context.conversation_context,
            enhanced_context: actionData.processed_response.enhanced_context
          });

          // Process orchestrator response with flow management
          await processOrchestratorResponseEnhanced(response, actionData.context.conversation_context, actionData.flow_analysis);
        }
      });
      
      // Execute the next action in queue (one-action-at-a-time)
      await executeNextAction();
      
    } catch (error) {
      console.error('Error sending message:', error);
      addMessage({
        type: 'system',
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
        metadata: { action_type: 'error' }
      });
    }
  }, [input, isProcessing, isWaitingForConfirmation, conversationContext, sessionId, orchestratorMode, agentStates, activeWorkflows, addMessage, queueAction, executeNextAction]);

  // Analyze user intent and update conversation context
  const analyzeUserIntent = async (message: string, currentContext: ConversationContext): Promise<ConversationContext> => {
    const lowerMessage = message.toLowerCase();
    
    // Detect user intent
    let newIntent = currentContext.user_intent;
    if (!newIntent) {
      if (lowerMessage.includes('create') || lowerMessage.includes('build') || lowerMessage.includes('develop')) {
        newIntent = 'development_project';
      } else if (lowerMessage.includes('research') || lowerMessage.includes('analyze')) {
        newIntent = 'research_task';
      } else if (lowerMessage.includes('plan') || lowerMessage.includes('design')) {
        newIntent = 'planning_task';
      } else {
        newIntent = 'general_inquiry';
      }
    }

    // Determine if requirements are gathered
    const requirementsGathered = currentContext.requirements_gathered || 
      (lowerMessage.includes('that\'s all') || lowerMessage.includes('complete') || lowerMessage.includes('enough'));

    // Determine if research is needed
    const researchNeeded = currentContext.research_needed || 
      (newIntent === 'development_project' && requirementsGathered) ||
      newIntent === 'research_task';

    return {
      ...currentContext,
      user_intent: newIntent,
      requirements_gathered: requirementsGathered,
      research_needed: researchNeeded,
      current_focus: requirementsGathered ? (researchNeeded ? 'research' : 'planning') : 'requirements_gathering'
    };
  };

  // Enhanced process orchestrator response with flow management
  const processOrchestratorResponseEnhanced = async (response: any, context: ConversationContext, flowAnalysis: any) => {
    // Process response through conversation processor
    const enhancedResponse = await conversationProcessor.enhanceResponse(
      response.content || response.response || 'I understand your request. Let me help you with that.',
      {
        session_id: sessionId,
        orchestrator_mode: orchestratorMode,
        conversation_context: context,
        flow_analysis: flowAnalysis
      }
    );

    // Add orchestrator response with enhanced content
    addMessage({
      type: 'orchestrator',
      content: enhancedResponse.content,
      metadata: { 
        action_type: 'response',
        confidence: enhancedResponse.confidence,
        suggestions: enhancedResponse.suggestions
      }
    });

    // Store orchestrator response with enhanced context
    await conversationProcessor.storeConversationMemory(
      sessionId,
      `Orchestrator: ${enhancedResponse.content}`,
      { 
        orchestrator_mode: orchestratorMode,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        context,
        flow_analysis: flowAnalysis,
        enhanced_response: enhancedResponse
      }
    );

    // Check for flow transitions
    const currentFlow = conversationFlowManager.getCurrentFlow();
    if (currentFlow) {
      const possibleTransitions = conversationFlowManager.getPossibleTransitions(currentFlow.current_phase);
      
      for (const transition of possibleTransitions) {
        if (transition.condition(context, flowAnalysis)) {
          // Execute transition
          const newPhase = conversationFlowManager.executeTransition(transition.to_phase, {
            context,
            flow_analysis: flowAnalysis,
            response: enhancedResponse
          });
          
          // Handle specific transitions
          if (transition.to_phase === 'research_planning' && !context.research_needed) {
            // Transition to planning
            const confirmationId = `confirm_${Date.now()}`;
            addConfirmation({
              id: confirmationId,
        message: 'I have gathered sufficient information about your requirements. Should I proceed to create a detailed plan and specifications?',
        action: 'transition_to_planning',
        metadata: { context },
        timestamp: new Date()
      }]);
      setIsWaitingForConfirmation(true);
      
      addMessage({
        type: 'confirmation',
        content: 'I have gathered sufficient information about your requirements. Should I proceed to create a detailed plan and specifications?',
        metadata: { 
          requires_confirmation: true,
          confirmation_id: confirmationId
        }
      });
    } else if (context.research_needed && context.current_focus === 'requirements_gathering') {
      // Suggest research
      const confirmationId = `confirm_${Date.now()}`;
      setPendingConfirmations(prev => [...prev, {
        id: confirmationId,
        message: 'Based on your requirements, I recommend conducting research to gather more information. Should I start the research process?',
        action: 'start_research',
        metadata: {
          query: context.user_intent,
          mode: 'deep',
          workflow_id: sessionId
        },
        timestamp: new Date()
      }]);
      setIsWaitingForConfirmation(true);
      
      addMessage({
        type: 'confirmation',
        content: 'Based on your requirements, I recommend conducting research to gather more information. Should I start the research process?',
        metadata: { 
          requires_confirmation: true,
          confirmation_id: confirmationId
        }
      });
    } else if (context.user_intent === 'development_project') {
      // Create workflow for development projects
      const workflowId = `workflow_${Date.now()}`;
      const newWorkflow: ActiveWorkflow = {
        id: workflowId,
        name: `Task: ${context.user_intent.substring(0, 50)}...`,
        status: 'running',
        progress: 0,
        current_step: 'Planning',
        agents_involved: ['research_agent', 'architect_agent'],
        started_at: new Date()
      };
      
      setActiveWorkflows(prev => [...prev, newWorkflow]);
      addSystemMessage(`Created workflow: ${workflowId}`);
      
      // Simulate agent activation
      setTimeout(() => {
        addAgentMessage(
          'Starting research phase for the requested task',
          'research_agent',
          { workflow_id: workflowId, action_type: 'research_start' }
        );
      }, 1000);
      
      setTimeout(() => {
        addAgentMessage(
          'Analyzing requirements and creating architectural plan',
          'architect_agent',
          { workflow_id: workflowId, action_type: 'planning_start' }
        );
      }, 2000);
    }
  };

  // Monitor agent states
  useEffect(() => {
    if (!autoRefresh || !isConnected) return;

    const interval = setInterval(loadAgentStates, 5000);
    return () => clearInterval(interval);
  }, [autoRefresh, isConnected, loadAgentStates]);

  // Initialize on mount
  useEffect(() => {
    initializeOrchestrator();
  }, [initializeOrchestrator]);

  // Simulate workflow progress updates
  useEffect(() => {
    if (activeWorkflows.length === 0) return;

    const interval = setInterval(() => {
      setActiveWorkflows(prev => prev.map(workflow => {
        if (workflow.status === 'running' && workflow.progress < 100) {
          const newProgress = Math.min(workflow.progress + Math.random() * 10, 100);
          const isCompleted = newProgress >= 100;
          
          if (isCompleted) {
            addSystemMessage(`Workflow ${workflow.id} completed successfully`);
          }
          
          return {
            ...workflow,
            progress: newProgress,
            status: isCompleted ? 'completed' : 'running',
            current_step: isCompleted ? 'Completed' : 
              newProgress > 75 ? 'Finalizing' :
              newProgress > 50 ? 'Implementation' :
              newProgress > 25 ? 'Design' : 'Planning'
          };
        }
        return workflow;
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, [activeWorkflows.length]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': case 'running': return 'bg-green-100 text-green-800';
      case 'busy': case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'error': case 'failed': return 'bg-red-100 text-red-800';
      case 'completed': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getMessageTypeColor = (type: string) => {
    switch (type) {
      case 'user': return 'bg-blue-50 border-blue-200';
      case 'orchestrator': return 'bg-purple-50 border-purple-200';
      case 'agent': return 'bg-green-50 border-green-200';
      case 'system': return 'bg-gray-50 border-gray-200';
      default: return 'bg-white border-gray-200';
    }
  };

  return (
    <div className={`grid grid-cols-1 lg:grid-cols-3 gap-6 h-full ${className}`}>
      {/* Main Chat Interface */}
      <div className="lg:col-span-2 flex flex-col">
        <Card className="flex-1 flex flex-col">
          {/* Header */}
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">AI Orchestrator</h2>
                <div className="flex items-center gap-2 mt-1">
                  <div className={`w-2 h-2 rounded-full ${
                    isConnected ? 'bg-green-500' : 'bg-red-500'
                  }`} />
                  <span className="text-sm text-gray-600">
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                  {sessionId && (
                    <Badge variant="outline" className="text-xs ml-2">
                      Session: {sessionId.slice(-8)}
                    </Badge>
                  )}
                  {conversationContext.current_focus && (
                    <Badge variant="secondary" className="text-xs">
                      Focus: {conversationContext.current_focus.replace('_', ' ')}
                    </Badge>
                  )}
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                <select 
                  value={orchestratorMode} 
                  onChange={(e) => setOrchestratorMode(e.target.value as any)}
                  className="px-3 py-1 border border-gray-300 rounded text-sm"
                >
                  <option value="intelligent">Intelligent Mode</option>
                  <option value="multi_agent">Multi-Agent Mode</option>
                  <option value="legacy">Legacy Mode</option>
                </select>
                
                <Button 
                  onClick={initializeOrchestrator} 
                  size="sm" 
                  disabled={isProcessing}
                >
                  Reconnect
                </Button>
              </div>
            </div>
            
            {/* Conversation Context Display */}
            {conversationContext.user_intent && (
              <div className="mt-3 p-3 bg-blue-50 rounded-lg">
                <div className="text-sm text-blue-800">
                  <strong>Intent:</strong> {conversationContext.user_intent.replace('_', ' ')}
                  {conversationContext.requirements_gathered && (
                    <span className="ml-2 text-green-600">✓ Requirements gathered</span>
                  )}
                  {conversationContext.research_needed && (
                    <span className="ml-2 text-orange-600">🔍 Research needed</span>
                  )}
                </div>
              </div>
            )}
            
            {/* Pending Confirmations */}
            {pendingConfirmations.length > 0 && (
              <div className="mt-3 space-y-2">
                {pendingConfirmations.map((confirmation) => (
                  <Alert key={confirmation.id} className="border-orange-200 bg-orange-50">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">{confirmation.message}</span>
                        <div className="flex gap-2 ml-4">
                          <Button
                            size="sm"
                            onClick={() => handleConfirmationLocal(confirmation.id, true)}
                            className="bg-green-600 hover:bg-green-700"
                          >
                            Yes
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleConfirmationLocal(confirmation.id, false)}
                          >
                            No
                          </Button>
                        </div>
                      </div>
                    </AlertDescription>
                  </Alert>
                ))}
              </div>
            )}
          </div>

          {/* Messages */}
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4">
              {messages.length === 0 ? (
                <div className="text-center text-gray-500 py-8">
                  <p>Welcome to the AI Orchestrator!</p>
                  <p className="text-sm mt-2">Start by describing what you'd like to build or accomplish.</p>
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${
                      message.type === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.type === 'user'
                          ? 'bg-blue-500 text-white'
                          : message.type === 'orchestrator'
                          ? 'bg-purple-100 text-purple-900 border border-purple-200'
                          : message.type === 'agent'
                          ? 'bg-green-100 text-green-900 border border-green-200'
                          : message.type === 'confirmation'
                          ? 'bg-orange-100 text-orange-900 border border-orange-200'
                          : message.type === 'artifact'
                          ? 'bg-indigo-100 text-indigo-900 border border-indigo-200'
                          : 'bg-gray-100 text-gray-900'
                      }`}
                    >
                      <div className="flex items-center gap-2 text-sm font-medium mb-1">
                        {message.type === 'user' && <Users className="h-4 w-4" />}
                        {message.type === 'orchestrator' && <Brain className="h-4 w-4" />}
                        {message.type === 'agent' && <Settings className="h-4 w-4" />}
                        {message.type === 'confirmation' && <AlertCircle className="h-4 w-4" />}
                        {message.type === 'artifact' && <FileText className="h-4 w-4" />}
                        
                        <span>
                          {message.type === 'user' ? 'You' : 
                           message.type === 'orchestrator' ? 'Ultra Orchestrator' :
                           message.type === 'agent' ? `${message.metadata?.agent_id || 'Agent'}` :
                           message.type === 'confirmation' ? 'Confirmation Required' :
                           message.type === 'artifact' ? 'Artifact Generated' :
                           'System'}
                        </span>
                      </div>
                      
                      <div className="whitespace-pre-wrap">{message.content}</div>
                      
                      {/* Artifact download buttons */}
                      {message.type === 'artifact' && message.metadata?.download_url && (
                        <div className="mt-2 flex gap-2">
                          <Button size="sm" variant="outline" className="text-xs">
                            <Download className="h-3 w-3 mr-1" />
                            Download PDF
                          </Button>
                          <Button size="sm" variant="outline" className="text-xs">
                            <FileText className="h-3 w-3 mr-1" />
                            Download Markdown
                          </Button>
                        </div>
                      )}
                      
                      <div className="text-xs opacity-70 mt-1 flex items-center justify-between">
                        <span>{message.timestamp.toLocaleTimeString()}</span>
                        {message.metadata?.status && (
                          <Badge variant="outline" className="text-xs">
                            {message.metadata.status}
                          </Badge>
                        )}
                      </div>
                      
                      {message.metadata && (message.metadata.workflow_id || message.metadata.action_type) && (
                        <div className="text-xs opacity-60 mt-1">
                          {message.metadata.workflow_id && `Workflow: ${message.metadata.workflow_id.slice(-8)}`}
                          {message.metadata.action_type && ` | Action: ${message.metadata.action_type}`}
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              
              {isProcessing && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg p-3 border">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-purple-600"></div>
                      <span className="text-sm text-gray-600">Ultra Orchestrator is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>

          {/* Input */}
          <div className="p-4 border-t border-gray-200">
            {isWaitingForConfirmation && (
              <Alert className="mb-4 border-orange-200 bg-orange-50">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  Please respond to the pending confirmation above before sending a new message.
                </AlertDescription>
              </Alert>
            )}
            
            {conversationContext.current_focus && (
              <div className="mb-2 text-xs text-gray-600">
                Current focus: {conversationContext.current_focus.replace('_', ' ')}
              </div>
            )}
            
            <div className="flex gap-2">
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={isWaitingForConfirmation 
                  ? "Please respond to the confirmation above first..."
                  : conversationContext.current_focus === 'requirements_gathering'
                  ? "Describe what you'd like to build or accomplish..."
                  : conversationContext.current_focus === 'research'
                  ? "Provide additional research requirements or say 'continue'..."
                  : "Continue the conversation..."}
                className="flex-1 min-h-[60px] resize-none"
                disabled={isWaitingForConfirmation}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isWaitingForConfirmation) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
              />
              <Button 
                onClick={sendMessage} 
                disabled={!input.trim() || isProcessing || isWaitingForConfirmation}
                className="self-end"
              >
                {isProcessing ? (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Sending</span>
                  </div>
                ) : (
                  'Send'
                )}
              </Button>
            </div>
            
            {/* Quick Actions */}
            {!isWaitingForConfirmation && conversationContext.current_focus === 'requirements_gathering' && (
              <div className="mt-2 flex gap-2">
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={() => {
                    setInput("That's all the requirements I have. Please proceed with the next steps.");
                  }}
                  className="text-xs"
                >
                  Complete Requirements
                </Button>
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={() => {
                    setInput("I need help defining my requirements. Can you guide me?");
                  }}
                  className="text-xs"
                >
                  Need Help
                </Button>
              </div>
            )}
            
            <p className="text-xs text-gray-500 mt-2">
              Press Enter to send, Shift+Enter for new line
            </p>
          </div>
        </Card>
      </div>

      {/* Sidebar - Enhanced Monitoring */}
      <div className="space-y-6">
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="workflows">Workflows</TabsTrigger>
            <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
            <TabsTrigger value="actions">Actions</TabsTrigger>
            <TabsTrigger value="agents">Agents</TabsTrigger>
          </TabsList>

          <TabsContent value="workflows" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Workflow className="h-5 w-5" />
                  Active Workflows ({activeWorkflows.length})
                </CardTitle>
                <CardDescription>
                  Monitor workflow progress and status
                </CardDescription>
              </CardHeader>
              <CardContent>
                {activeWorkflows.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Workflow className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No active workflows</p>
                    <p className="text-sm">Workflows will appear here when started</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {activeWorkflows.map((workflow) => (
                      <div 
                        key={workflow.id} 
                        className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                          selectedWorkflow === workflow.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                        }`}
                        onClick={() => setSelectedWorkflow(workflow.id)}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">{workflow.name}</span>
                          <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(workflow.status)}`}>
                            {workflow.status}
                          </span>
                        </div>
                        
                        <div className="mb-2">
                          <div className="flex justify-between text-xs text-gray-600 mb-1">
                            <span>{workflow.current_step}</span>
                            <span>{Math.round(workflow.progress)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${workflow.progress}%` }}
                            />
                          </div>
                        </div>
                        
                        <div className="flex flex-wrap gap-1">
                          {workflow.agents_involved.map((agentId) => (
                            <span key={agentId} className="text-xs bg-gray-100 text-gray-700 px-2 py-1 rounded">
                              {agentId.replace('_', ' ')}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="artifacts" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Artifacts ({artifactStats.total})
                </CardTitle>
                <CardDescription>
                  Generated files, documents, and code artifacts
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center gap-2">
                    <Input
                      placeholder="Search artifacts..."
                      value={artifactState.searchQuery}
                      onChange={(e) => searchArtifacts(e.target.value)}
                      className="flex-1"
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => loadArtifacts()}
                    >
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {hasArtifacts ? (
                    <div className="space-y-2">
                      {filteredArtifacts.map((artifact) => (
                        <div
                          key={artifact.id}
                          className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                            artifactState.selectedArtifact?.id === artifact.id
                              ? 'border-blue-500 bg-blue-50'
                              : 'hover:bg-gray-50'
                          }`}
                          onClick={() => selectArtifact(artifact.id)}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <FileText className="h-4 w-4" />
                              <span className="font-medium">{artifact.name}</span>
                              <Badge variant="outline">{artifact.type}</Badge>
                              {artifact.isFavorite && (
                                <Star className="h-4 w-4 text-yellow-500 fill-current" />
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleFavorite(artifact.id);
                                }}
                              >
                                <Star className={`h-4 w-4 ${
                                  artifact.isFavorite ? 'text-yellow-500 fill-current' : 'text-gray-400'
                                }`} />
                              </Button>
                              <Badge variant="secondary">
                                {artifact.size ? `${Math.round(artifact.size / 1024)}KB` : 'N/A'}
                              </Badge>
                            </div>
                          </div>
                          <div className="text-sm text-muted-foreground mt-1">
                            {artifact.description || 'No description'}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            Created: {new Date(artifact.createdAt).toLocaleString()}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>No artifacts generated yet</p>
                      <p className="text-sm">Artifacts will appear here as they are created</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="actions" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Action Queue
                </CardTitle>
                <CardDescription>
                  One-action-at-a-time execution monitoring
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant={hasActiveActionsLocal ? 'default' : 'secondary'}>
                         {hasActiveActionsLocal ? 'Active' : 'Idle'}
                       </Badge>
                       {nextActionLocal && (
                         <Badge variant="outline">
                           Next: {nextActionLocal.type}
                         </Badge>
                       )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">
                        Queue: {orchestratorState.actionQueue.length}
                      </Badge>
                      <Badge variant={isHealthyLocal ? 'default' : 'destructive'}>
                         {isHealthyLocal ? 'Healthy' : 'Issues'}
                       </Badge>
                    </div>
                  </div>
                  
                  {orchestratorState.actionQueue.length > 0 ? (
                    <div className="space-y-2">
                      {orchestratorState.actionQueue.slice(0, 5).map((action, index) => (
                        <div
                          key={action.id}
                          className={`p-3 border rounded-lg ${
                            index === 0 ? 'border-blue-500 bg-blue-50' : 'border-gray-200'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {index === 0 ? (
                                <Play className="h-4 w-4 text-blue-600" />
                              ) : (
                                <Clock className="h-4 w-4 text-gray-400" />
                              )}
                              <span className="font-medium">{action.type}</span>
                              <Badge variant={action.priority === 'high' ? 'destructive' : 'secondary'}>
                                {action.priority}
                              </Badge>
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {index === 0 ? 'Executing' : `Position ${index + 1}`}
                            </div>
                          </div>
                          {action.data && (
                            <div className="text-sm text-muted-foreground mt-1">
                              {typeof action.data === 'object' ? JSON.stringify(action.data).slice(0, 100) + '...' : action.data}
                            </div>
                          )}
                        </div>
                      ))}
                      {orchestratorState.actionQueue.length > 5 && (
                        <div className="text-center text-sm text-muted-foreground">
                          ... and {orchestratorState.actionQueue.length - 5} more actions
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>No actions in queue</p>
                      <p className="text-sm">Actions will appear here as they are queued</p>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t">
                    <div className="text-center">
                      <div className="text-2xl font-bold">{orchestratorState.executionStats.totalExecuted}</div>
                      <div className="text-sm text-muted-foreground">Total Executed</div>
                    </div>
                    <div className="text-center">
                       <div className="text-2xl font-bold">{sessionDurationLocal}</div>
                       <div className="text-sm text-muted-foreground">Session Duration</div>
                     </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="agents" className="space-y-4">

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Agent States ({Object.keys(agentStates).length})
                </CardTitle>
                <CardDescription>
                  Monitor individual agent status and activity
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={autoRefresh}
                        onChange={(e) => setAutoRefresh(e.target.checked)}
                        className="rounded"
                      />
                      Auto-refresh
                    </label>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={loadAgentStates}
                    >
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {Object.keys(agentStates).length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      <Settings className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p>No active agents</p>
                      <p className="text-sm">Agents will appear here when activated</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {Object.entries(agentStates).map(([agentId, state]) => (
                        <div key={agentId} className="p-3 border rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Settings className="h-4 w-4" />
                              <span className="text-sm font-medium">{agentId.replace('_', ' ')}</span>
                            </div>
                            <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(state.status)}`}>
                              {state.status}
                            </span>
                          </div>
                          
                          {state.current_task && (
                            <div className="text-xs text-gray-600 mb-1">
                              <strong>Task:</strong> {state.current_task}
                            </div>
                          )}
                          
                          {state.last_activity && (
                            <div className="text-xs text-gray-500">
                              <strong>Last Activity:</strong> {new Date(state.last_activity).toLocaleTimeString()}
                            </div>
                          )}
                          
                          {state.capabilities && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {state.capabilities.map((capability) => (
                                <Badge key={capability} variant="outline" className="text-xs">
                                  {capability}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Quick Actions */}
        <Card className="p-4">
          <h3 className="font-semibold mb-3">Quick Actions</h3>
          <div className="space-y-2">
            <Button 
              onClick={() => setInput('Create a new React component with TypeScript')}
              size="sm" 
              className="w-full justify-start"
            >
              Create Component
            </Button>
            <Button 
              onClick={() => setInput('Build a REST API with FastAPI')}
              size="sm" 
              className="w-full justify-start"
            >
              Build API
            </Button>
            <Button 
              onClick={() => setInput('Analyze the current codebase structure')}
              size="sm" 
              className="w-full justify-start"
            >
              Analyze Codebase
            </Button>
            <Button 
              onClick={() => setInput('Set up testing framework')}
              size="sm" 
              className="w-full justify-start"
            >
              Setup Testing
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}

export default OrchestratorInterface;