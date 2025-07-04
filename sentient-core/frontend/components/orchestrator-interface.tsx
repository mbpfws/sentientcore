'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Textarea } from './ui/textarea';
import { coreServicesClient, AgentState, WorkflowState } from '../lib/api';
import { ChatService } from '../lib/api/chat-service';

interface OrchestratorInterfaceProps {
  className?: string;
}

interface OrchestratorMessage {
  id: string;
  type: 'user' | 'orchestrator' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    agent_id?: string;
    workflow_id?: string;
    action_type?: string;
    status?: string;
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
}

export function OrchestratorInterface({ className }: OrchestratorInterfaceProps) {
  const [messages, setMessages] = useState<OrchestratorMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [agentStates, setAgentStates] = useState<Record<string, AgentState>>({});
  const [activeWorkflows, setActiveWorkflows] = useState<ActiveWorkflow[]>([]);
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [orchestratorMode, setOrchestratorMode] = useState<'intelligent' | 'multi_agent' | 'legacy'>('intelligent');
  const [isConnected, setIsConnected] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Initialize orchestrator connection
  const initializeOrchestrator = useCallback(async () => {
    try {
      const response = await coreServicesClient.initializeServices();
      if (response.success) {
        setIsConnected(true);
        addSystemMessage('Orchestrator initialized successfully');
        await loadAgentStates();
      }
    } catch (error) {
      addSystemMessage(`Failed to initialize orchestrator: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }, []);

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

  // Add system message
  const addSystemMessage = (content: string) => {
    const message: OrchestratorMessage = {
      id: Date.now().toString(),
      type: 'system',
      content,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, message]);
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

  // Send message to orchestrator
  const sendMessage = async () => {
    if (!inputMessage.trim() || isProcessing) return;

    const userMessage: OrchestratorMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsProcessing(true);

    try {
      // Store the conversation in memory
      await coreServicesClient.storeConversation(
        `User: ${inputMessage}`,
        { 
          orchestrator_mode: orchestratorMode,
          timestamp: new Date().toISOString()
        }
      );

      // Send to chat service with orchestrator mode
      const response = await ChatService.sendMessage({
        message: inputMessage,
        workflow_mode: orchestratorMode,
        research_mode: 'knowledge'
      });

      if (response) {
        addOrchestratorMessage(response.content);
        
        // Store orchestrator response
        await coreServicesClient.storeConversation(
          `Orchestrator: ${response.content}`,
          { 
            orchestrator_mode: orchestratorMode,
            timestamp: new Date().toISOString()
          }
        );

        // Simulate workflow creation if this looks like a complex task
        if (inputMessage.toLowerCase().includes('create') || 
            inputMessage.toLowerCase().includes('build') ||
            inputMessage.toLowerCase().includes('develop')) {
          
          const workflowId = `workflow_${Date.now()}`;
          const newWorkflow: ActiveWorkflow = {
            id: workflowId,
            name: `Task: ${inputMessage.substring(0, 50)}...`,
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
      }
    } catch (error) {
      addSystemMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsProcessing(false);
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
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                <p>Welcome to the AI Orchestrator!</p>
                <p className="text-sm mt-2">Start by describing what you'd like to build or accomplish.</p>
              </div>
            ) : (
              messages.map((message) => (
                <div key={message.id} className={`p-3 rounded-lg border ${getMessageTypeColor(message.type)}`}>
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="font-medium capitalize">
                        {message.type === 'agent' && message.metadata?.agent_id 
                          ? message.metadata.agent_id.replace('_', ' ')
                          : message.type
                        }
                      </span>
                      {message.metadata?.workflow_id && (
                        <span className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded">
                          {message.metadata.workflow_id}
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-gray-500">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm">{message.content}</p>
                  {message.metadata?.action_type && (
                    <div className="mt-2">
                      <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        {message.metadata.action_type.replace('_', ' ')}
                      </span>
                    </div>
                  )}
                </div>
              ))
            )}
            
            {isProcessing && (
              <div className="flex items-center gap-2 text-gray-500">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-400"></div>
                <span className="text-sm">Orchestrator is thinking...</span>
              </div>
            )}
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex gap-2">
              <Textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Describe what you'd like to build or accomplish..."
                className="flex-1 min-h-[60px] resize-none"
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
              />
              <Button 
                onClick={sendMessage} 
                disabled={!inputMessage.trim() || isProcessing}
                className="self-end"
              >
                Send
              </Button>
            </div>
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