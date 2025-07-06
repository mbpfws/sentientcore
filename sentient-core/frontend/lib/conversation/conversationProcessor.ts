import { ConversationContext, OrchestratorMessage } from '../hooks/useOrchestratorState';
import { conversationFlowManager, ConversationAnalysis } from './conversationFlowManager';
import { coreServicesClient } from '../services/coreServicesClient';

export interface ProcessedResponse {
  content: string;
  suggestedActions: string[];
  contextUpdates: Partial<ConversationContext>;
  requiresConfirmation: boolean;
  confirmationMessage?: string;
  confirmationAction?: string;
  metadata: Record<string, any>;
}

export interface ConversationMemory {
  shortTerm: OrchestratorMessage[];
  longTerm: ConversationSummary[];
  entities: Record<string, any>;
  preferences: Record<string, any>;
}

export interface ConversationSummary {
  id: string;
  timestamp: Date;
  summary: string;
  keyPoints: string[];
  decisions: string[];
  context: ConversationContext;
}

class ConversationProcessor {
  private memory: ConversationMemory = {
    shortTerm: [],
    longTerm: [],
    entities: {},
    preferences: {}
  };

  private readonly MAX_SHORT_TERM_MESSAGES = 20;
  private readonly CONTEXT_WINDOW = 10;

  async processMessage(
    message: string,
    currentContext: ConversationContext,
    messageHistory: OrchestratorMessage[],
    sessionId: string
  ): Promise<ProcessedResponse> {
    // Update short-term memory
    this.updateShortTermMemory(messageHistory);

    // Analyze the message
    const analysis = conversationFlowManager.analyzeMessage(message, currentContext);
    
    // Update context based on analysis
    const updatedContext = conversationFlowManager.updateContext(analysis, currentContext);
    
    // Set appropriate flow if not already set
    if (analysis.intent !== 'general_inquiry' && analysis.confidence > 0.7) {
      conversationFlowManager.setCurrentFlow(analysis.intent);
    }

    // Generate contextual response
    const baseResponse = conversationFlowManager.generateContextualResponse(analysis, updatedContext);
    
    // Determine if confirmation is needed
    const confirmationResult = this.checkForConfirmation(analysis, updatedContext, messageHistory);
    
    // Generate enhanced response with context
    const enhancedResponse = await this.enhanceResponse(
      baseResponse,
      analysis,
      updatedContext,
      messageHistory,
      sessionId
    );

    // Store conversation in memory
    await this.storeConversationMemory(message, enhancedResponse, updatedContext, sessionId);

    return {
      content: enhancedResponse,
      suggestedActions: analysis.suggestedActions,
      contextUpdates: this.getContextUpdates(currentContext, updatedContext),
      requiresConfirmation: confirmationResult.required,
      confirmationMessage: confirmationResult.message,
      confirmationAction: confirmationResult.action,
      metadata: {
        analysis,
        flowPhase: conversationFlowManager.getCurrentPhase()?.id,
        confidence: analysis.confidence,
        entities: analysis.entities
      }
    };
  }

  private updateShortTermMemory(messages: OrchestratorMessage[]) {
    this.memory.shortTerm = messages.slice(-this.MAX_SHORT_TERM_MESSAGES);
    
    // Extract entities from recent messages
    this.memory.shortTerm.forEach(msg => {
      if (msg.metadata?.entities) {
        Object.assign(this.memory.entities, msg.metadata.entities);
      }
    });
  }

  private checkForConfirmation(
    analysis: ConversationAnalysis,
    context: ConversationContext,
    messages: OrchestratorMessage[]
  ): { required: boolean; message?: string; action?: string } {
    const currentPhase = conversationFlowManager.getCurrentPhase();
    
    // Check if we should transition to next phase
    if (currentPhase) {
      const isPhaseComplete = conversationFlowManager.isPhaseComplete(context);
      const nextPhases = conversationFlowManager.getNextPhases();
      
      if (isPhaseComplete && nextPhases.length > 0) {
        const nextPhase = nextPhases[0];
        
        switch (nextPhase) {
          case 'research':
            return {
              required: true,
              message: 'I have enough information to proceed. Should I start researching the best technologies and approaches for your project?',
              action: 'start_research'
            };
          case 'planning':
            return {
              required: true,
              message: 'Based on our discussion and research, should I create a detailed implementation plan for your project?',
              action: 'create_plan'
            };
          case 'implementation':
            return {
              required: true,
              message: 'The plan is ready! Should I start implementing your project according to the specifications we\'ve defined?',
              action: 'start_implementation'
            };
        }
      }
    }

    // Check for high-impact actions
    if (analysis.suggestedActions.includes('start_research') && analysis.confidence > 0.8) {
      return {
        required: true,
        message: 'This appears to be a complex project that would benefit from research. Should I investigate the best approaches and technologies?',
        action: 'start_research'
      };
    }

    return { required: false };
  }

  private async enhanceResponse(
    baseResponse: string,
    analysis: ConversationAnalysis,
    context: ConversationContext,
    messages: OrchestratorMessage[],
    sessionId: string
  ): Promise<string> {
    let enhancedResponse = baseResponse;
    
    // Add context-aware enhancements
    const recentContext = this.getRecentContext(messages);
    
    // Add clarifying questions if needed
    if (context.current_focus === 'requirements_gathering') {
      const missingInfo = this.identifyMissingInformation(analysis, recentContext);
      if (missingInfo.length > 0) {
        enhancedResponse += `\n\nTo better assist you, could you tell me more about:\n${missingInfo.map(info => `• ${info}`).join('\n')}`;
      }
    }

    // Add progress indicators
    const currentPhase = conversationFlowManager.getCurrentPhase();
    if (currentPhase) {
      const progress = this.calculatePhaseProgress(context, currentPhase.id);
      if (progress > 0) {
        enhancedResponse += `\n\n*Progress: ${currentPhase.name} (${Math.round(progress * 100)}% complete)*`;
      }
    }

    // Add helpful suggestions
    const suggestions = this.generateHelpfulSuggestions(analysis, context);
    if (suggestions.length > 0) {
      enhancedResponse += `\n\n**Suggestions:**\n${suggestions.map(s => `• ${s}`).join('\n')}`;
    }

    return enhancedResponse;
  }

  private getRecentContext(messages: OrchestratorMessage[]): Record<string, any> {
    const recentMessages = messages.slice(-this.CONTEXT_WINDOW);
    const context: Record<string, any> = {
      topics: new Set(),
      technologies: new Set(),
      requirements: new Set(),
      decisions: new Set()
    };

    recentMessages.forEach(msg => {
      const content = msg.content.toLowerCase();
      
      // Extract technologies mentioned
      const techKeywords = ['react', 'vue', 'angular', 'python', 'javascript', 'typescript', 'node', 'express', 'fastapi', 'django'];
      techKeywords.forEach(tech => {
        if (content.includes(tech)) {
          context.technologies.add(tech);
        }
      });
      
      // Extract requirement indicators
      if (content.includes('need') || content.includes('want') || content.includes('require')) {
        context.requirements.add(msg.content);
      }
      
      // Extract decisions
      if (content.includes('yes') || content.includes('no') || content.includes('choose') || content.includes('prefer')) {
        context.decisions.add(msg.content);
      }
    });

    return {
      topics: Array.from(context.topics),
      technologies: Array.from(context.technologies),
      requirements: Array.from(context.requirements),
      decisions: Array.from(context.decisions)
    };
  }

  private identifyMissingInformation(analysis: ConversationAnalysis, recentContext: Record<string, any>): string[] {
    const missing: string[] = [];
    
    if (analysis.intent === 'development_project') {
      if (recentContext.technologies.length === 0) {
        missing.push('Your preferred technology stack or programming language');
      }
      
      if (!recentContext.requirements.some((req: string) => req.includes('user') || req.includes('audience'))) {
        missing.push('Your target users or audience');
      }
      
      if (!recentContext.requirements.some((req: string) => req.includes('feature') || req.includes('function'))) {
        missing.push('The main features or functionality you need');
      }
      
      if (!recentContext.requirements.some((req: string) => req.includes('platform') || req.includes('deploy'))) {
        missing.push('Your target platform (web, mobile, desktop)');
      }
    }
    
    return missing;
  }

  private calculatePhaseProgress(context: ConversationContext, phaseId: string): number {
    switch (phaseId) {
      case 'requirements_gathering':
        let progress = 0;
        if (context.user_intent && context.user_intent !== 'general_inquiry') progress += 0.3;
        if (context.requirements_gathered) progress += 0.7;
        return progress;
        
      case 'research':
        return context.research_needed ? 0.5 : 1.0;
        
      case 'planning':
        return context.current_focus === 'planning' ? 0.5 : 
               context.current_focus === 'implementation' ? 1.0 : 0;
        
      default:
        return 0;
    }
  }

  private generateHelpfulSuggestions(analysis: ConversationAnalysis, context: ConversationContext): string[] {
    const suggestions: string[] = [];
    
    if (context.current_focus === 'requirements_gathering') {
      suggestions.push('Consider describing your project\'s main goals and target audience');
      suggestions.push('Think about any technical constraints or preferences you have');
      
      if (analysis.complexity === 'complex') {
        suggestions.push('For complex projects, I can break down the requirements into smaller, manageable parts');
      }
    }
    
    if (analysis.intent === 'development_project' && !context.research_needed) {
      suggestions.push('I can research the latest best practices and technologies for your project');
    }
    
    if (context.current_focus === 'planning') {
      suggestions.push('I can create a detailed project timeline and milestone breakdown');
      suggestions.push('Consider discussing testing and deployment strategies');
    }
    
    return suggestions;
  }

  private getContextUpdates(oldContext: ConversationContext, newContext: ConversationContext): Partial<ConversationContext> {
    const updates: Partial<ConversationContext> = {};
    
    Object.keys(newContext).forEach(key => {
      const typedKey = key as keyof ConversationContext;
      if (oldContext[typedKey] !== newContext[typedKey]) {
        updates[typedKey] = newContext[typedKey];
      }
    });
    
    return updates;
  }

  private async storeConversationMemory(
    userMessage: string,
    assistantResponse: string,
    context: ConversationContext,
    sessionId: string
  ) {
    try {
      await coreServicesClient.storeConversation(
        `User: ${userMessage}\nAssistant: ${assistantResponse}`,
        {
          session_id: sessionId,
          context,
          timestamp: new Date().toISOString(),
          phase: conversationFlowManager.getCurrentPhase()?.id
        }
      );
    } catch (error) {
      console.error('Failed to store conversation memory:', error);
    }
  }

  async summarizeConversation(messages: OrchestratorMessage[], context: ConversationContext): Promise<ConversationSummary> {
    const keyPoints: string[] = [];
    const decisions: string[] = [];
    
    // Extract key points from messages
    messages.forEach(msg => {
      if (msg.type === 'user') {
        const content = msg.content.toLowerCase();
        if (content.includes('want') || content.includes('need') || content.includes('build')) {
          keyPoints.push(msg.content);
        }
        if (content.includes('yes') || content.includes('no') || content.includes('choose')) {
          decisions.push(msg.content);
        }
      }
    });
    
    const summary = `Conversation focused on ${context.user_intent}. Current phase: ${context.current_focus}. Requirements gathered: ${context.requirements_gathered}.`;
    
    return {
      id: `summary_${Date.now()}`,
      timestamp: new Date(),
      summary,
      keyPoints,
      decisions,
      context
    };
  }

  getMemory(): ConversationMemory {
    return this.memory;
  }

  clearMemory() {
    this.memory = {
      shortTerm: [],
      longTerm: [],
      entities: {},
      preferences: {}
    };
  }
}

export const conversationProcessor = new ConversationProcessor();
export default ConversationProcessor;