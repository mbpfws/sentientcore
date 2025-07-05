'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { ChevronDown, ChevronRight, Clock, CheckCircle, XCircle, AlertCircle, Zap, Search, Brain, FileText } from 'lucide-react';

interface VerboseStep {
  id: string;
  type: 'search' | 'analysis' | 'synthesis' | 'validation' | 'error';
  title: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  timestamp: string;
  duration?: number;
  details?: {
    input?: string;
    output?: string;
    metadata?: Record<string, any>;
    sources?: Array<{
      url: string;
      title: string;
      relevance: number;
    }>;
    tokens_used?: number;
    model_used?: string;
  };
  substeps?: VerboseStep[];
}

interface VerboseFeedbackProps {
  steps: VerboseStep[];
  isActive: boolean;
  onToggle?: () => void;
  className?: string;
}

const VerboseFeedback: React.FC<VerboseFeedbackProps> = ({ 
  steps, 
  isActive, 
  onToggle,
  className = '' 
}) => {
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollAreaRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new steps are added
  useEffect(() => {
    if (autoScroll && scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [steps, autoScroll]);

  const toggleStep = (stepId: string) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    setExpandedSteps(newExpanded);
  };

  const getStepIcon = (type: string, status: string) => {
    if (status === 'error') return <XCircle className="h-4 w-4 text-red-500" />;
    if (status === 'completed') return <CheckCircle className="h-4 w-4 text-green-500" />;
    if (status === 'running') return <div className="h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
    
    switch (type) {
      case 'search': return <Search className="h-4 w-4 text-blue-500" />;
      case 'analysis': return <Brain className="h-4 w-4 text-purple-500" />;
      case 'synthesis': return <FileText className="h-4 w-4 text-green-500" />;
      case 'validation': return <CheckCircle className="h-4 w-4 text-orange-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'running': return 'bg-blue-100 text-blue-800';
      case 'error': return 'bg-red-100 text-red-800';
      case 'pending': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatDuration = (duration?: number) => {
    if (!duration) return '';
    if (duration < 1000) return `${duration}ms`;
    return `${(duration / 1000).toFixed(1)}s`;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const getOverallProgress = () => {
    if (steps.length === 0) return 0;
    const completedSteps = steps.filter(step => step.status === 'completed').length;
    return (completedSteps / steps.length) * 100;
  };

  const getOverallStats = () => {
    const stats = {
      total: steps.length,
      completed: 0,
      running: 0,
      error: 0,
      pending: 0,
      totalTokens: 0,
      totalDuration: 0
    };

    steps.forEach(step => {
      stats[step.status as keyof typeof stats]++;
      if (step.details?.tokens_used) stats.totalTokens += step.details.tokens_used;
      if (step.duration) stats.totalDuration += step.duration;
    });

    return stats;
  };

  const stats = getOverallStats();

  if (!isActive) {
    return (
      <Button 
        variant="outline" 
        size="sm" 
        onClick={onToggle}
        className={className}
      >
        <Zap className="h-4 w-4 mr-2" />
        Show Verbose Feedback
      </Button>
    );
  }

  return (
    <Card className={`${className} border-l-4 border-l-blue-500`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center">
            <Zap className="h-5 w-5 mr-2 text-blue-500" />
            Verbose Execution Feedback
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setAutoScroll(!autoScroll)}
              className={autoScroll ? 'text-blue-600' : 'text-gray-500'}
            >
              Auto-scroll
            </Button>
            <Button variant="ghost" size="sm" onClick={onToggle}>
              Hide
            </Button>
          </div>
        </div>
        
        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Overall Progress</span>
            <span>{Math.round(getOverallProgress())}%</span>
          </div>
          <Progress value={getOverallProgress()} className="h-2" />
          
          {/* Stats */}
          <div className="flex gap-4 text-xs text-muted-foreground">
            <span>Total: {stats.total}</span>
            <span>Completed: {stats.completed}</span>
            <span>Running: {stats.running}</span>
            {stats.error > 0 && <span className="text-red-600">Errors: {stats.error}</span>}
            {stats.totalTokens > 0 && <span>Tokens: {stats.totalTokens.toLocaleString()}</span>}
            {stats.totalDuration > 0 && <span>Duration: {formatDuration(stats.totalDuration)}</span>}
          </div>
        </div>
      </CardHeader>
      
      <CardContent>
        <ScrollArea className="h-[400px]" ref={scrollAreaRef}>
          <div className="space-y-3">
            {steps.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <AlertCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No execution steps yet</p>
                <p className="text-xs">Verbose feedback will appear here during execution</p>
              </div>
            ) : (
              steps.map((step, index) => (
                <div key={step.id} className="relative">
                  {/* Timeline connector */}
                  {index < steps.length - 1 && (
                    <div className="absolute left-6 top-12 w-0.5 h-8 bg-gray-200" />
                  )}
                  
                  <Collapsible 
                    open={expandedSteps.has(step.id)}
                    onOpenChange={() => toggleStep(step.id)}
                  >
                    <CollapsibleTrigger asChild>
                      <Card className="cursor-pointer hover:bg-muted/50 transition-colors">
                        <CardContent className="p-4">
                          <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 mt-0.5">
                              {getStepIcon(step.type, step.status)}
                            </div>
                            
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center justify-between">
                                <h4 className="font-medium text-sm truncate">{step.title}</h4>
                                <div className="flex items-center gap-2 ml-2">
                                  <Badge className={`text-xs ${getStatusColor(step.status)}`}>
                                    {step.status}
                                  </Badge>
                                  {expandedSteps.has(step.id) ? 
                                    <ChevronDown className="h-4 w-4" /> : 
                                    <ChevronRight className="h-4 w-4" />
                                  }
                                </div>
                              </div>
                              
                              <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                                {step.description}
                              </p>
                              
                              <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                                <div className="flex items-center">
                                  <Clock className="h-3 w-3 mr-1" />
                                  {formatTimestamp(step.timestamp)}
                                </div>
                                {step.duration && (
                                  <span>{formatDuration(step.duration)}</span>
                                )}
                                {step.details?.model_used && (
                                  <span>Model: {step.details.model_used}</span>
                                )}
                                {step.details?.tokens_used && (
                                  <span>Tokens: {step.details.tokens_used}</span>
                                )}
                              </div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </CollapsibleTrigger>
                    
                    <CollapsibleContent>
                      <div className="ml-9 mt-2 space-y-3">
                        {/* Step Details */}
                        {step.details && (
                          <Card className="bg-muted/30">
                            <CardContent className="p-3 space-y-3">
                              {step.details.input && (
                                <div>
                                  <h5 className="text-xs font-medium text-muted-foreground mb-1">Input</h5>
                                  <div className="bg-background p-2 rounded text-xs font-mono">
                                    {step.details.input}
                                  </div>
                                </div>
                              )}
                              
                              {step.details.output && (
                                <div>
                                  <h5 className="text-xs font-medium text-muted-foreground mb-1">Output</h5>
                                  <div className="bg-background p-2 rounded text-xs font-mono max-h-32 overflow-y-auto">
                                    {step.details.output}
                                  </div>
                                </div>
                              )}
                              
                              {step.details.sources && step.details.sources.length > 0 && (
                                <div>
                                  <h5 className="text-xs font-medium text-muted-foreground mb-1">Sources Found</h5>
                                  <div className="space-y-1">
                                    {step.details.sources.map((source, idx) => (
                                      <div key={idx} className="flex items-center justify-between text-xs">
                                        <a 
                                          href={source.url} 
                                          target="_blank" 
                                          rel="noopener noreferrer"
                                          className="text-blue-600 hover:underline truncate flex-1"
                                        >
                                          {source.title}
                                        </a>
                                        <Badge variant="outline" className="ml-2 text-xs">
                                          {Math.round(source.relevance * 100)}%
                                        </Badge>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                              
                              {step.details.metadata && Object.keys(step.details.metadata).length > 0 && (
                                <div>
                                  <h5 className="text-xs font-medium text-muted-foreground mb-1">Metadata</h5>
                                  <div className="bg-background p-2 rounded text-xs font-mono">
                                    <pre>{JSON.stringify(step.details.metadata, null, 2)}</pre>
                                  </div>
                                </div>
                              )}
                            </CardContent>
                          </Card>
                        )}
                        
                        {/* Substeps */}
                        {step.substeps && step.substeps.length > 0 && (
                          <div className="space-y-2">
                            <h5 className="text-xs font-medium text-muted-foreground">Substeps</h5>
                            {step.substeps.map((substep, subIndex) => (
                              <Card key={substep.id} className="bg-muted/20">
                                <CardContent className="p-3">
                                  <div className="flex items-start gap-2">
                                    <div className="flex-shrink-0 mt-0.5">
                                      {getStepIcon(substep.type, substep.status)}
                                    </div>
                                    <div className="flex-1">
                                      <div className="flex items-center justify-between">
                                        <h6 className="text-xs font-medium">{substep.title}</h6>
                                        <Badge className={`text-xs ${getStatusColor(substep.status)}`}>
                                          {substep.status}
                                        </Badge>
                                      </div>
                                      <p className="text-xs text-muted-foreground mt-1">
                                        {substep.description}
                                      </p>
                                      <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                                        <span>{formatTimestamp(substep.timestamp)}</span>
                                        {substep.duration && <span>{formatDuration(substep.duration)}</span>}
                                      </div>
                                    </div>
                                  </div>
                                </CardContent>
                              </Card>
                            ))}
                          </div>
                        )}
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};

export { VerboseFeedback };