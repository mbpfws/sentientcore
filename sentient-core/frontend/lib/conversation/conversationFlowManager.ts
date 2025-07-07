import { ConversationContext, OrchestratorMessage } from '../hooks/useOrchestratorState';
import { coreServicesClient } from '../api/core-services';

export interface ConversationFlow {
  id: string;
  name: string;
  description: string;
  phases: ConversationPhase[];
  currentPhase: number;
  context: ConversationContext;
  metadata: Record<string, any>;
}

export interface ConversationPhase {
  id: string;
  name: string;
  description: string;
  requirements: string[];
  completionCriteria: string[];
  suggestedPrompts: string[];
  nextPhases: string[];
  allowedTransitions: string[];
}

export interface FlowTransition {
  from: string;
  to: string;
  condition: (context: ConversationContext, messages: OrchestratorMessage[]) => boolean;
  action?: (context: ConversationContext) => Promise<void>;
}

export interface ConversationAnalysis {
  intent: string;
  confidence: number;
  entities: Record<string, any>;
  sentiment: 'positive' | 'neutral' | 'negative';
  complexity: 'simple' | 'moderate' | 'complex';
  requiresResearch: boolean;
  requiresPlanning: boolean;
  suggestedActions: string[];
}

class ConversationFlowManager {
  private flows: Map<string, ConversationFlow> = new Map();
  private transitions: FlowTransition[] = [];
  private currentFlow: ConversationFlow | null = null;

  constructor() {
    this.initializeDefaultFlows();
    this.initializeTransitions();
  }

  private initializeDefaultFlows() {
    // Development Project Flow
    const developmentFlow: ConversationFlow = {
      id: 'development_project',
      name: 'Development Project',
      description: 'Complete software development lifecycle',
      phases: [
        {
          id: 'requirements_gathering',
          name: 'Requirements Gathering',
          description: 'Collect and clarify project requirements',
          requirements: ['project_description', 'target_platform', 'key_features'],
          completionCriteria: ['clear_scope', 'defined_objectives', 'technical_constraints'],
          suggestedPrompts: [
            'What type of application do you want to build?',
            'Who is your target audience?',
            'What are the main features you need?',
            'Do you have any technical preferences or constraints?'
          ],
          nextPhases: ['research', 'planning'],
          allowedTransitions: ['research', 'planning', 'implementation']
        },
        {
          id: 'research',
          name: 'Research & Analysis',
          description: 'Research technologies and best practices',
          requirements: ['technology_stack', 'architecture_patterns', 'best_practices'],
          completionCriteria: ['technology_selected', 'patterns_identified', 'dependencies_mapped'],
          suggestedPrompts: [
            'Let me research the best technologies for your project',
            'I\'ll analyze similar projects and patterns',
            'Let me gather information about best practices'
          ],
          nextPhases: ['planning'],
          allowedTransitions: ['planning', 'requirements_gathering']
        },
        {
          id: 'planning',
          name: 'Planning & Design',
          description: 'Create detailed project plan and architecture',
          requirements: ['architecture_design', 'implementation_plan', 'testing_strategy'],
          completionCriteria: ['plan_approved', 'architecture_defined', 'milestones_set'],
          suggestedPrompts: [
            'I\'ll create a detailed implementation plan',
            'Let me design the system architecture',
            'I\'ll break down the work into manageable tasks'
          ],
          nextPhases: ['implementation'],
          allowedTransitions: ['implementation', 'research']
        },
        {
          id: 'implementation',
          name: 'Implementation',
          description: 'Build the project according to the plan',
          requirements: ['code_generation', 'testing', 'documentation'],
          completionCriteria: ['features_implemented', 'tests_passing', 'documentation_complete'],
          suggestedPrompts: [
            'I\'ll start implementing the core features',
            'Let me generate the project structure',
            'I\'ll create the necessary components and modules'
          ],
          nextPhases: ['testing', 'deployment'],
          allowedTransitions: ['testing', 'deployment', 'planning']
        }
      ],
      currentPhase: 0,
      context: {
        user_intent: 'development_project',
        requirements_gathered: false,
        research_needed: false,
        current_focus: 'requirements_gathering'
      },
      metadata: {}
    };

    // Research Task Flow
    const researchFlow: ConversationFlow = {
      id: 'research_task',
      name: 'Research Task',
      description: 'Focused research and information gathering',
      phases: [
        {
          id: 'topic_definition',
          name: 'Topic Definition',
          description: 'Define research scope and objectives',
          requirements: ['research_topic', 'scope', 'objectives'],
          completionCriteria: ['topic_clear', 'scope_defined', 'objectives_set'],
          suggestedPrompts: [
            'What specific topic would you like me to research?',
            'What depth of research do you need?',
            'Are there specific aspects you want me to focus on?'
          ],
          nextPhases: ['information_gathering'],
          allowedTransitions: ['information_gathering']
        },
        {
          id: 'information_gathering',
          name: 'Information Gathering',
          description: 'Collect relevant information and data',
          requirements: ['sources_identified', 'data_collected', 'information_verified'],
          completionCriteria: ['sufficient_data', 'sources_credible', 'information_current'],
          suggestedPrompts: [
            'I\'ll gather information from reliable sources',
            'Let me research the latest developments',
            'I\'ll compile comprehensive data on this topic'
          ],
          nextPhases: ['analysis'],
          allowedTransitions: ['analysis', 'topic_definition']
        },
        {
          id: 'analysis',
          name: 'Analysis & Synthesis',
          description: 'Analyze findings and create insights',
          requirements: ['data_analyzed', 'insights_generated', 'conclusions_drawn'],
          completionCriteria: ['analysis_complete', 'insights_valuable', 'conclusions_supported'],
          suggestedPrompts: [
            'I\'ll analyze the gathered information',
            'Let me identify key insights and patterns',
            'I\'ll synthesize the findings into actionable conclusions'
          ],
          nextPhases: ['presentation'],
          allowedTransitions: ['presentation', 'information_gathering']
        }
      ],
      currentPhase: 0,
      context: {
        user_intent: 'research_task',
        requirements_gathered: false,
        research_needed: true,
        current_focus: 'topic_definition'
      },
      metadata: {}
    };

    this.flows.set('development_project', developmentFlow);
    this.flows.set('research_task', researchFlow);
  }

  private initializeTransitions() {
    this.transitions = [
      {
        from: 'requirements_gathering',
        to: 'research',
        condition: (context, messages) => {
          return context.research_needed && context.requirements_gathered;
        },
        action: async (context) => {
          await coreServicesClient.storeConversation(
            'Transitioning to research phase',
            { transition: 'requirements_to_research', context }
          );
        }
      },
      {
        from: 'requirements_gathering',
        to: 'planning',
        condition: (context, messages) => {
          return !context.research_needed && context.requirements_gathered;
        },
        action: async (context) => {
          await coreServicesClient.storeConversation(
            'Transitioning to planning phase',
            { transition: 'requirements_to_planning', context }
          );
        }
      },
      {
        from: 'research',
        to: 'planning',
        condition: (context, messages) => {
          const recentMessages = (messages || []).slice(-5);
          return recentMessages.some(msg => 
            msg.content.toLowerCase().includes('research complete') ||
            msg.content.toLowerCase().includes('proceed to planning')
          );
        }
      }
    ];
  }

  analyzeMessage(message: string, context: ConversationContext): ConversationAnalysis {
    const lowerMessage = message.toLowerCase();
    
    // Intent detection
    let intent = 'general_inquiry';
    let confidence = 0.5;
    
    if (lowerMessage.includes('build') || lowerMessage.includes('create') || lowerMessage.includes('develop')) {
      intent = 'development_project';
      confidence = 0.8;
    } else if (lowerMessage.includes('research') || lowerMessage.includes('analyze') || lowerMessage.includes('investigate')) {
      intent = 'research_task';
      confidence = 0.8;
    } else if (lowerMessage.includes('help') || lowerMessage.includes('guide') || lowerMessage.includes('explain')) {
      intent = 'assistance_request';
      confidence = 0.7;
    }

    // Entity extraction (simplified)
    const entities: Record<string, any> = {};
    if (lowerMessage.includes('react')) entities.framework = 'react';
    if (lowerMessage.includes('python')) entities.language = 'python';
    if (lowerMessage.includes('api')) entities.type = 'api';
    if (lowerMessage.includes('web')) entities.platform = 'web';
    if (lowerMessage.includes('mobile')) entities.platform = 'mobile';

    // Sentiment analysis (simplified)
    const sentiment = lowerMessage.includes('urgent') || lowerMessage.includes('asap') ? 'negative' :
                     lowerMessage.includes('excited') || lowerMessage.includes('great') ? 'positive' : 'neutral';

    // Complexity assessment
    const complexity = lowerMessage.length > 200 || Object.keys(entities).length > 3 ? 'complex' :
                      lowerMessage.length > 50 || Object.keys(entities).length > 1 ? 'moderate' : 'simple';

    // Requirements assessment
    const requiresResearch = intent === 'research_task' || 
                           (intent === 'development_project' && complexity === 'complex');
    const requiresPlanning = intent === 'development_project';

    // Suggested actions
    const suggestedActions: string[] = [];
    if (requiresResearch) suggestedActions.push('start_research');
    if (requiresPlanning) suggestedActions.push('create_plan');
    if (intent === 'development_project') suggestedActions.push('gather_requirements');

    return {
      intent,
      confidence,
      entities,
      sentiment,
      complexity,
      requiresResearch,
      requiresPlanning,
      suggestedActions
    };
  }

  updateContext(analysis: ConversationAnalysis, currentContext: ConversationContext): ConversationContext {
    const newContext = { ...currentContext };
    
    // Update intent if confidence is high
    if (analysis.confidence > 0.7) {
      newContext.user_intent = analysis.intent;
    }

    // Update research and planning flags
    newContext.research_needed = analysis.requiresResearch;
    
    // Update current focus based on analysis
    if (analysis.intent === 'development_project') {
      if (!newContext.requirements_gathered) {
        newContext.current_focus = 'requirements_gathering';
      } else if (newContext.research_needed) {
        newContext.current_focus = 'research';
      } else {
        newContext.current_focus = 'planning';
      }
    } else if (analysis.intent === 'research_task') {
      newContext.current_focus = 'research';
    }

    return newContext;
  }

  getFlow(flowId: string): ConversationFlow | undefined {
    return this.flows.get(flowId);
  }

  getCurrentFlow(): ConversationFlow | null {
    return this.currentFlow;
  }

  setCurrentFlow(flowId: string): boolean {
    const flow = this.flows.get(flowId);
    if (flow) {
      this.currentFlow = flow;
      return true;
    }
    return false;
  }

  initializeFlow(flowType: string, initialContext: any): boolean {
    const flow = this.flows.get(flowType);
    if (flow) {
      // Create a copy of the flow with initial context
      this.currentFlow = {
        ...flow,
        currentPhase: 0,
        context: {
          user_intent: flowType,
          requirements_gathered: false,
          research_needed: flowType === 'development_project',
          current_focus: 'requirements_gathering',
          ...initialContext
        }
      };
      return true;
    }
    return false;
  }

  getCurrentPhase(): ConversationPhase | null {
    if (!this.currentFlow) return null;
    return this.currentFlow.phases[this.currentFlow.currentPhase] || null;
  }

  canTransition(to: string): boolean {
    const currentPhase = this.getCurrentPhase();
    if (!currentPhase) return false;
    return currentPhase.allowedTransitions.includes(to);
  }

  async transitionTo(phaseId: string, context: ConversationContext, messages: OrchestratorMessage[]): Promise<boolean> {
    if (!this.currentFlow || !this.canTransition(phaseId)) {
      return false;
    }

    const currentPhase = this.getCurrentPhase();
    if (!currentPhase) return false;

    // Find applicable transition
    const transition = this.transitions.find(t => 
      t.from === currentPhase.id && t.to === phaseId
    );

    if (transition && transition.condition(context, messages)) {
      // Execute transition action if defined
      if (transition.action) {
        await transition.action(context);
      }

      // Update current phase
      const newPhaseIndex = this.currentFlow.phases.findIndex(p => p.id === phaseId);
      if (newPhaseIndex !== -1) {
        this.currentFlow.currentPhase = newPhaseIndex;
        this.currentFlow.context = context;
        return true;
      }
    }

    return false;
  }

  getSuggestedPrompts(): string[] {
    const currentPhase = this.getCurrentPhase();
    return currentPhase?.suggestedPrompts || [];
  }

  getNextPhases(): string[] {
    const currentPhase = this.getCurrentPhase();
    return currentPhase?.nextPhases || [];
  }

  isPhaseComplete(context: ConversationContext): boolean {
    const currentPhase = this.getCurrentPhase();
    if (!currentPhase) return false;

    // Check completion criteria based on context
    return currentPhase.completionCriteria.every(criteria => {
      switch (criteria) {
        case 'clear_scope':
          return context.requirements_gathered;
        case 'technology_selected':
          return context.research_needed === false;
        case 'plan_approved':
          return context.current_focus !== 'planning';
        default:
          return true;
      }
    });
  }

  generateContextualResponse(analysis: ConversationAnalysis, context: ConversationContext): string {
    const currentPhase = this.getCurrentPhase();
    
    if (!currentPhase) {
      return "I understand your request. Let me help you get started.";
    }

    const responses = {
      requirements_gathering: [
        "I'd be happy to help you build that! Let me gather some more details to ensure I understand your requirements correctly.",
        "Great idea! To create the best solution for you, I need to understand a few more details about your project.",
        "I can definitely help with that. Let me ask a few questions to make sure I build exactly what you need."
      ],
      research: [
        "Let me research the best approaches and technologies for your project.",
        "I'll gather information about the latest best practices and tools for this type of project.",
        "Allow me to investigate the most suitable solutions and patterns for your requirements."
      ],
      planning: [
        "Now I'll create a detailed plan and architecture for your project.",
        "Let me design a comprehensive implementation strategy based on our discussion.",
        "I'll develop a step-by-step plan to bring your project to life."
      ],
      implementation: [
        "Perfect! I'll start implementing your project according to the plan we've created.",
        "Time to build! I'll begin creating the code and components for your project.",
        "Let's get started with the implementation. I'll create the project structure first."
      ]
    };

    const phaseResponses = responses[currentPhase.id as keyof typeof responses] || responses.requirements_gathering;
    return phaseResponses[Math.floor(Math.random() * phaseResponses.length)];
  }
}

export const conversationFlowManager = new ConversationFlowManager();
export default ConversationFlowManager;