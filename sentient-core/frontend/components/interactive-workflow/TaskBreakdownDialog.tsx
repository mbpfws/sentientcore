'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Loader2, AlertCircle, Lightbulb, Target, Clock, CheckCircle, ArrowRight } from 'lucide-react';
import { InteractiveWorkflowService, BreakdownTaskRequest } from '@/lib/api/interactive-workflow-service';

interface TaskBreakdownDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workflowId: string;
  onTaskBrokenDown: () => void;
}

interface BreakdownResult {
  subtasks: Array<{
    name: string;
    description: string;
    estimated_duration?: number;
    dependencies?: string[];
    priority?: string;
    type?: string;
  }>;
  analysis: {
    complexity_score: number;
    estimated_total_duration: number;
    risk_factors: string[];
    recommendations: string[];
  };
}

export function TaskBreakdownDialog({ open, onOpenChange, workflowId, onTaskBrokenDown }: TaskBreakdownDialogProps) {
  const [formData, setFormData] = useState<BreakdownTaskRequest>({
    task_description: '',
    complexity_level: 'medium',
    target_subtask_count: undefined,
    include_dependencies: true,
    include_estimates: true
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<BreakdownResult | null>(null);
  const [step, setStep] = useState<'input' | 'result'>('input');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.task_description.trim()) {
      setError('Task description is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await InteractiveWorkflowService.breakdownTask(workflowId, formData);
      setResult(response);
      setStep('result');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to breakdown task');
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptBreakdown = () => {
    onTaskBrokenDown();
    onOpenChange(false);
    handleReset();
  };

  const handleReset = () => {
    setFormData({
      task_description: '',
      complexity_level: 'medium',
      target_subtask_count: undefined,
      include_dependencies: true,
      include_estimates: true
    });
    setResult(null);
    setStep('input');
    setError(null);
  };

  const handleCancel = () => {
    onOpenChange(false);
    handleReset();
  };

  const getComplexityColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'secondary';
      case 'medium': return 'default';
      case 'high': return 'destructive';
      default: return 'default';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority?.toLowerCase()) {
      case 'urgent': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'default';
      case 'low': return 'secondary';
      default: return 'default';
    }
  };

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[900px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Task Breakdown Assistant
          </DialogTitle>
          <DialogDescription>
            Break down complex tasks into smaller, manageable subtasks with AI assistance.
          </DialogDescription>
        </DialogHeader>

        {step === 'input' ? (
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Error Alert */}
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Task Description */}
            <div className="space-y-2">
              <Label htmlFor="task-description">Task Description *</Label>
              <Textarea
                id="task-description"
                placeholder="Describe the complex task you want to break down into smaller steps..."
                value={formData.task_description}
                onChange={(e) => setFormData(prev => ({ ...prev, task_description: e.target.value }))}
                disabled={loading}
                rows={4}
                required
              />
              <p className="text-sm text-muted-foreground">
                Be as detailed as possible. Include context, requirements, and expected outcomes.
              </p>
            </div>

            {/* Complexity Level */}
            <div className="space-y-2">
              <Label htmlFor="complexity-level">Complexity Level</Label>
              <select
                id="complexity-level"
                className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
                value={formData.complexity_level}
                onChange={(e) => setFormData(prev => ({ ...prev, complexity_level: e.target.value as 'low' | 'medium' | 'high' }))}
                disabled={loading}
              >
                <option value="low">Low - Simple task with few dependencies</option>
                <option value="medium">Medium - Moderate complexity with some dependencies</option>
                <option value="high">High - Complex task with many dependencies</option>
              </select>
            </div>

            {/* Target Subtask Count */}
            <div className="space-y-2">
              <Label htmlFor="target-count">Target Number of Subtasks</Label>
              <Input
                id="target-count"
                type="number"
                placeholder="Leave empty for AI to decide (recommended)"
                value={formData.target_subtask_count || ''}
                onChange={(e) => {
                  const count = parseInt(e.target.value);
                  setFormData(prev => ({ 
                    ...prev, 
                    target_subtask_count: isNaN(count) ? undefined : count 
                  }));
                }}
                disabled={loading}
                min="2"
                max="20"
              />
              <p className="text-sm text-muted-foreground">
                Optional: Specify how many subtasks you want (2-20). AI will optimize if left empty.
              </p>
            </div>

            {/* Options */}
            <div className="space-y-4">
              <Label>Breakdown Options</Label>
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="include-dependencies"
                    checked={formData.include_dependencies}
                    onChange={(e) => setFormData(prev => ({ ...prev, include_dependencies: e.target.checked }))}
                    disabled={loading}
                    className="rounded border-gray-300"
                  />
                  <Label htmlFor="include-dependencies" className="text-sm font-normal">
                    Include dependency analysis between subtasks
                  </Label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="include-estimates"
                    checked={formData.include_estimates}
                    onChange={(e) => setFormData(prev => ({ ...prev, include_estimates: e.target.checked }))}
                    disabled={loading}
                    className="rounded border-gray-300"
                  />
                  <Label htmlFor="include-estimates" className="text-sm font-normal">
                    Include time estimates for each subtask
                  </Label>
                </div>
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
                disabled={loading}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Lightbulb className="h-4 w-4 mr-2" />
                    Break Down Task
                  </>
                )}
              </Button>
            </DialogFooter>
          </form>
        ) : (
          <div className="space-y-6">
            {/* Analysis Summary */}
            {result?.analysis && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5" />
                    Analysis Summary
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <Label className="text-muted-foreground">Complexity Score</Label>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="text-2xl font-bold">{result.analysis.complexity_score}/10</div>
                        <Badge variant={getComplexityColor(
                          result.analysis.complexity_score <= 3 ? 'low' :
                          result.analysis.complexity_score <= 7 ? 'medium' : 'high'
                        )}>
                          {result.analysis.complexity_score <= 3 ? 'Low' :
                           result.analysis.complexity_score <= 7 ? 'Medium' : 'High'}
                        </Badge>
                      </div>
                    </div>
                    <div>
                      <Label className="text-muted-foreground">Estimated Total Duration</Label>
                      <div className="flex items-center gap-2 mt-1">
                        <Clock className="h-4 w-4" />
                        <span className="text-lg font-semibold">
                          {formatDuration(result.analysis.estimated_total_duration)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {result.analysis.risk_factors.length > 0 && (
                    <div>
                      <Label className="text-muted-foreground">Risk Factors</Label>
                      <ul className="mt-1 space-y-1">
                        {result.analysis.risk_factors.map((risk, index) => (
                          <li key={index} className="text-sm flex items-start gap-2">
                            <AlertCircle className="h-3 w-3 text-yellow-500 mt-0.5 flex-shrink-0" />
                            {risk}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.analysis.recommendations.length > 0 && (
                    <div>
                      <Label className="text-muted-foreground">Recommendations</Label>
                      <ul className="mt-1 space-y-1">
                        {result.analysis.recommendations.map((rec, index) => (
                          <li key={index} className="text-sm flex items-start gap-2">
                            <CheckCircle className="h-3 w-3 text-green-500 mt-0.5 flex-shrink-0" />
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            <Separator />

            {/* Subtasks */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold">Generated Subtasks ({result?.subtasks.length || 0})</h4>
                <Badge variant="outline">
                  Total: {result?.analysis.estimated_total_duration ? 
                    formatDuration(result.analysis.estimated_total_duration) : 'Unknown'
                  }
                </Badge>
              </div>

              <div className="space-y-3">
                {result?.subtasks.map((subtask, index) => (
                  <Card key={index}>
                    <CardContent className="p-4">
                      <div className="space-y-3">
                        <div className="flex items-start justify-between">
                          <div className="space-y-1 flex-1">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-muted-foreground">
                                Step {index + 1}
                              </span>
                              {index > 0 && <ArrowRight className="h-3 w-3 text-muted-foreground" />}
                            </div>
                            <h5 className="font-medium">{subtask.name}</h5>
                            <p className="text-sm text-muted-foreground">{subtask.description}</p>
                          </div>
                          <div className="flex flex-col items-end gap-2">
                            {subtask.priority && (
                              <Badge variant={getPriorityColor(subtask.priority)} size="sm">
                                {subtask.priority}
                              </Badge>
                            )}
                            {subtask.estimated_duration && (
                              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                <Clock className="h-3 w-3" />
                                {formatDuration(subtask.estimated_duration)}
                              </div>
                            )}
                          </div>
                        </div>

                        {subtask.dependencies && subtask.dependencies.length > 0 && (
                          <div>
                            <Label className="text-xs text-muted-foreground">Dependencies:</Label>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {subtask.dependencies.map((dep, depIndex) => (
                                <Badge key={depIndex} variant="outline" size="sm">
                                  {dep}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}

                        {subtask.type && (
                          <div>
                            <Badge variant="secondary" size="sm">
                              {subtask.type}
                            </Badge>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setStep('input')}
              >
                Back to Edit
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
              >
                Cancel
              </Button>
              <Button onClick={handleAcceptBreakdown}>
                <CheckCircle className="h-4 w-4 mr-2" />
                Accept & Add to Workflow
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}