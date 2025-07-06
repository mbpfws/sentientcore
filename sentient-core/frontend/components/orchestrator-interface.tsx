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
  const [inputMessage, setInputMessage] = useState('');
  const [selectedTab, setSelectedTab] = useState('conversation');
  const [showSettings, setShowSettings] = useState(false);
  const [messages, setMessages] = useState<OrchestratorMessage[]>([]);
  const [agentStates, setAgentStates] = useState<Record<string, AgentState>>({});
  const [activeWorkflows, setActiveWorkflows] = useState<ActiveWorkflow[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [pendingConfirmations, setPendingConfirmations] = useState<PendingConfirmation[]>([]);
  const [conversationContext, setConversationContext] = useState<ConversationContext>({
    user_intent: '',
    requirements_gathered: false,
    research_needed: false,
    planning_phase: false,
    current_focus: 'requirements_gathering',
    artifacts_generated: []
  });
  const [isWaitingForConfirmation, setIsWaitingForConfirmation] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
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
      // Analyze user intent and update conversation context
      const updatedContext = await analyzeUserIntent(userMessage, conversationContext);
      
      // Queue the message processing action
      await queueAction({
        id: `action_${Date.now()}`,
        type: 'send_message',
        priority: 'high',
        data: {
          message: userMessage,
          session_id: sessionId,
          context: {
            orchestrator_mode: orchestratorMode,
            conversation_context: updatedContext,
            agent_states: agentStates,
            active_workflows: activeWorkflows.map(w => ({ id: w.id, status: w.status, progress: w.progress }))
          }
        },
        execute: async (actionData) => {
          // Store the conversation in memory
          await coreServicesClient.storeConversation(
            `User: ${actionData.message}`,
            { 
              orchestrator_mode: actionData.context.orchestrator_mode,
              timestamp: new Date().toISOString(),
              session_id: actionData.session_id,
              context: actionData.context.conversation_context
            }
          );

          // Send to chat service with enhanced context
          const response = await chatService.sendMessage({
            message: actionData.message,
            workflow_mode: actionData.context.orchestrator_mode,
            research_mode: 'knowledge',
            session_id: actionData.session_id,
            context: actionData.context.conversation_context
          });

          // Process orchestrator response
          await processOrchestratorResponse(response, actionData.context.conversation_context);
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
  }, [input, isProcessing, isWaitingForConfirmation, conversationContext, sessionId, orchestratorMode, agentStates, activeWorkflows, addMessage, queueAction, executeNextAction, analyzeUserIntent, processOrchestratorResponse]);

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

  // Process orchestrator response with intelligent flow
  const processOrchestratorResponse = async (response: any, context: ConversationContext) => {
    // Add orchestrator response
    addMessage({
      type: 'orchestrator',
      content: response.content || response.response || 'I understand your request. Let me help you with that.',
      metadata: { action_type: 'response' }
    });

    // Store orchestrator response
    await coreServicesClient.storeConversation(
      `Orchestrator: ${response.content || response.response}`,
      { 
        orchestrator_mode: orchestratorMode,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        context
      }
    );

    // Handle conversation flow based on context
    if (context.requirements_gathered && !context.research_needed && context.current_focus === 'requirements_gathering') {
      // Transition to planning
      const confirmationId = `confirm_${Date.now()}`;
      setPendingConfirmations(prev => [...prev, {
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
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
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
                disabled={!inputMessage.trim() || isProcessing || isWaitingForConfirmation}
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
                    setInputMessage("That's all the requirements I have. Please proceed with the next steps.");
                  }}
                  className="text-xs"
                >
                  Complete Requirements
                </Button>
                <Button 
                  size="sm" 
                  variant="outline" 
                  onClick={() => {
                    setInputMessage("I need help defining my requirements. Can you guide me?");
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

      {/* Sidebar - Agent States & Workflows */}
      <div className="space-y-6">
        {/* Active Workflows */}
        <Card className="p-4">
          <h3 className="font-semibold mb-3">Active Workflows</h3>
          {activeWorkflows.length === 0 ? (
            <p className="text-sm text-gray-500">No active workflows</p>
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
        </Card>

        {/* Agent States */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">Agent States</h3>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                className="rounded"
              />
              Auto-refresh
            </label>
          </div>
          
          {Object.keys(agentStates).length === 0 ? (
            <p className="text-sm text-gray-500">No active agents</p>
          ) : (
            <div className="space-y-2">
              {Object.entries(agentStates).map(([agentId, state]) => (
                <div key={agentId} className="p-2 bg-gray-50 rounded">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium">{agentId.replace('_', ' ')}</span>
                    <span className={`px-2 py-1 rounded-full text-xs ${getStatusColor(state.status)}`}>
                      {state.status}
                    </span>
                  </div>
                  
                  {state.current_task && (
                    <p className="text-xs text-gray-600 mb-1">{state.current_task}</p>
                  )}
                  
                  {state.last_activity && (
                    <p className="text-xs text-gray-500">
                      {new Date(state.last_activity).toLocaleTimeString()}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* Quick Actions */}
        <Card className="p-4">
          <h3 className="font-semibold mb-3">Quick Actions</h3>
          <div className="space-y-2">
            <Button 
              onClick={() => setInputMessage('Create a new React component with TypeScript')}
              size="sm" 
              className="w-full justify-start"
            >
              Create Component
            </Button>
            <Button 
              onClick={() => setInputMessage('Build a REST API with FastAPI')}
              size="sm" 
              className="w-full justify-start"
            >
              Build API
            </Button>
            <Button 
              onClick={() => setInputMessage('Analyze the current codebase structure')}
              size="sm" 
              className="w-full justify-start"
            >
              Analyze Codebase
            </Button>
            <Button 
              onClick={() => setInputMessage('Set up testing framework')}
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