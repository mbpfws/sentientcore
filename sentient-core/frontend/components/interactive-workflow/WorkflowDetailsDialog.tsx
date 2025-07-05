'use client';

import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { 
  Loader2, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Play, 
  Pause, 
  Square, 
  RotateCcw,
  Plus,
  Eye,
  Timer
} from 'lucide-react';
import { 
  InteractiveWorkflowService, 
  InteractiveWorkflow, 
  WorkflowStep, 
  PendingApproval,
  ExecutionReport
} from '@/lib/api/interactive-workflow-service';
import { AddStepDialog } from './AddStepDialog';

interface WorkflowDetailsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workflow: InteractiveWorkflow;
  onWorkflowUpdated: () => void;
}

export function WorkflowDetailsDialog({ 
  open, 
  onOpenChange, 
  workflow: initialWorkflow, 
  onWorkflowUpdated 
}: WorkflowDetailsDialogProps) {
  const [workflow, setWorkflow] = useState<InteractiveWorkflow>(initialWorkflow);
  const [pendingApprovals, setPendingApprovals] = useState<PendingApproval[]>([]);
  const [executionReport, setExecutionReport] = useState<ExecutionReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAddStepDialog, setShowAddStepDialog] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // Load workflow details
  const loadWorkflowDetails = async () => {
    if (!workflow.id) return;
    
    setLoading(true);
    try {
      const [updatedWorkflow, approvals, report] = await Promise.all([
        InteractiveWorkflowService.getWorkflow(workflow.id),
        InteractiveWorkflowService.getPendingApprovals(workflow.id),
        InteractiveWorkflowService.getExecutionReport(workflow.id)
      ]);
      
      setWorkflow(updatedWorkflow);
      setPendingApprovals(approvals);
      setExecutionReport(report);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workflow details');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (open) {
      setWorkflow(initialWorkflow);
      loadWorkflowDetails();
    }
  }, [open, initialWorkflow.id]);

  const handleWorkflowAction = async (action: 'start' | 'pause' | 'resume' | 'cancel' | 'restart') => {
    try {
      if (action === 'start') {
        await InteractiveWorkflowService.startWorkflow(workflow.id);
      } else {
        await InteractiveWorkflowService.controlWorkflow(workflow.id, { action });
      }
      await loadWorkflowDetails();
      onWorkflowUpdated();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} workflow`);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'running':
      case 'active':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'paused':
        return <Pause className="h-4 w-4 text-yellow-500" />;
      case 'pending':
        return <Clock className="h-4 w-4 text-gray-500" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
      case 'error':
        return 'bg-red-100 text-red-800';
      case 'running':
      case 'active':
        return 'bg-blue-100 text-blue-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'pending':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-600';
    }
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div>
                <DialogTitle className="text-xl">{workflow.name}</DialogTitle>
                <DialogDescription className="mt-1">
                  {workflow.description}
                </DialogDescription>
              </div>
              <div className="flex items-center space-x-2">
                {getStatusIcon(workflow.status)}
                <Badge className={getStatusColor(workflow.status)}>
                  {workflow.status}
                </Badge>
              </div>
            </div>
          </DialogHeader>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="steps">Steps ({workflow.total_steps})</TabsTrigger>
              <TabsTrigger value="approvals">Approvals ({pendingApprovals.length})</TabsTrigger>
              <TabsTrigger value="report">Report</TabsTrigger>
            </TabsList>

            <ScrollArea className="h-[500px] mt-4">
              <TabsContent value="overview" className="space-y-4">
                {/* Progress Overview */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Progress Overview</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Overall Progress</span>
                        <span>{workflow.completion_percentage}%</span>
                      </div>
                      <Progress value={workflow.completion_percentage} className="h-3" />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Current Step:</span>
                        <p className="font-medium">{workflow.current_step_index + 1} of {workflow.total_steps}</p>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Created:</span>
                        <p className="font-medium">{new Date(workflow.created_at).toLocaleString()}</p>
                      </div>
                      {workflow.updated_at && (
                        <div>
                          <span className="text-muted-foreground">Last Updated:</span>
                          <p className="font-medium">{new Date(workflow.updated_at).toLocaleString()}</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Workflow Actions */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Actions</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center space-x-2">
                      {workflow.status === 'pending' && (
                        <Button onClick={() => handleWorkflowAction('start')}>
                          <Play className="h-4 w-4 mr-2" />
                          Start Workflow
                        </Button>
                      )}
                      {workflow.status === 'running' && (
                        <Button 
                          variant="outline" 
                          onClick={() => handleWorkflowAction('pause')}
                        >
                          <Pause className="h-4 w-4 mr-2" />
                          Pause
                        </Button>
                      )}
                      {workflow.status === 'paused' && (
                        <Button onClick={() => handleWorkflowAction('resume')}>
                          <Play className="h-4 w-4 mr-2" />
                          Resume
                        </Button>
                      )}
                      {(workflow.status === 'running' || workflow.status === 'paused') && (
                        <Button 
                          variant="destructive" 
                          onClick={() => handleWorkflowAction('cancel')}
                        >
                          <Square className="h-4 w-4 mr-2" />
                          Cancel
                        </Button>
                      )}
                      {(workflow.status === 'completed' || workflow.status === 'failed') && (
                        <Button 
                          variant="outline" 
                          onClick={() => handleWorkflowAction('restart')}
                        >
                          <RotateCcw className="h-4 w-4 mr-2" />
                          Restart
                        </Button>
                      )}
                      <Button 
                        variant="outline" 
                        onClick={() => setShowAddStepDialog(true)}
                      >
                        <Plus className="h-4 w-4 mr-2" />
                        Add Step
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                {/* Metadata */}
                {workflow.metadata && Object.keys(workflow.metadata).length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-lg">Metadata</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {Object.entries(workflow.metadata).map(([key, value]) => (
                          <div key={key} className="flex justify-between text-sm">
                            <span className="text-muted-foreground capitalize">{key}:</span>
                            <span className="font-medium">
                              {Array.isArray(value) ? value.join(', ') : String(value)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              <TabsContent value="steps" className="space-y-4">
                {workflow.steps.length === 0 ? (
                  <Card>
                    <CardContent className="flex items-center justify-center h-32">
                      <div className="text-center">
                        <p className="text-muted-foreground mb-2">No steps defined yet</p>
                        <Button onClick={() => setShowAddStepDialog(true)}>
                          <Plus className="h-4 w-4 mr-2" />
                          Add First Step
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ) : (
                  <div className="space-y-3">
                    {workflow.steps.map((step, index) => (
                      <Card key={step.id} className={index === workflow.current_step_index ? 'ring-2 ring-blue-500' : ''}>
                        <CardHeader className="pb-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-2">
                              <span className="text-sm font-medium text-muted-foreground">#{index + 1}</span>
                              <CardTitle className="text-base">{step.name}</CardTitle>
                              {index === workflow.current_step_index && (
                                <Badge variant="outline">Current</Badge>
                              )}
                            </div>
                            <div className="flex items-center space-x-1">
                              {getStatusIcon(step.status)}
                              <Badge className={getStatusColor(step.status)}>
                                {step.status}
                              </Badge>
                            </div>
                          </div>
                          <CardDescription>{step.description}</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-muted-foreground">Type:</span>
                              <p className="font-medium">{step.step_type}</p>
                            </div>
                            <div>
                              <span className="text-muted-foreground">Created:</span>
                              <p className="font-medium">{new Date(step.created_at).toLocaleString()}</p>
                            </div>
                            {step.estimated_duration && (
                              <div>
                                <span className="text-muted-foreground">Estimated Duration:</span>
                                <p className="font-medium">{formatDuration(step.estimated_duration)}</p>
                              </div>
                            )}
                            {step.actual_duration && (
                              <div>
                                <span className="text-muted-foreground">Actual Duration:</span>
                                <p className="font-medium">{formatDuration(step.actual_duration)}</p>
                              </div>
                            )}
                          </div>
                          {step.dependencies.length > 0 && (
                            <div className="mt-3">
                              <span className="text-sm text-muted-foreground">Dependencies:</span>
                              <div className="flex flex-wrap gap-1 mt-1">
                                {step.dependencies.map((dep, i) => (
                                  <Badge key={i} variant="outline" className="text-xs">
                                    {dep}
                                  </Badge>
                                ))}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="approvals" className="space-y-4">
                {pendingApprovals.length === 0 ? (
                  <Card>
                    <CardContent className="flex items-center justify-center h-32">
                      <p className="text-muted-foreground">No pending approvals</p>
                    </CardContent>
                  </Card>
                ) : (
                  <div className="space-y-3">
                    {pendingApprovals.map((approval) => (
                      <Card key={approval.id}>
                        <CardHeader>
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-base">{approval.approval_type}</CardTitle>
                            <Badge variant="outline">Pending</Badge>
                          </div>
                        </CardHeader>
                        <CardContent>
                          <p className="text-sm mb-3">{approval.message}</p>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <span className="text-muted-foreground">Created:</span>
                              <p className="font-medium">{new Date(approval.created_at).toLocaleString()}</p>
                            </div>
                            {approval.timeout_seconds && (
                              <div>
                                <span className="text-muted-foreground">Timeout:</span>
                                <p className="font-medium">{formatDuration(approval.timeout_seconds)}</p>
                              </div>
                            )}
                          </div>
                          <div className="flex items-center space-x-2 mt-3">
                            <Button size="sm" className="bg-green-600 hover:bg-green-700">
                              Approve
                            </Button>
                            <Button size="sm" variant="destructive">
                              Reject
                            </Button>
                            <Button size="sm" variant="outline">
                              Modify
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </TabsContent>

              <TabsContent value="report" className="space-y-4">
                {loading ? (
                  <div className="flex items-center justify-center h-32">
                    <Loader2 className="h-6 w-6 animate-spin" />
                  </div>
                ) : executionReport ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Execution Summary</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Steps:</span>
                          <span className="font-medium">{executionReport.total_steps}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Completed:</span>
                          <span className="font-medium text-green-600">{executionReport.completed_steps}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Failed:</span>
                          <span className="font-medium text-red-600">{executionReport.failed_steps}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Total Duration:</span>
                          <span className="font-medium">{formatDuration(executionReport.total_duration)}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle className="text-lg">Approval Summary</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Approvals:</span>
                          <span className="font-medium text-green-600">{executionReport.approval_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Rejections:</span>
                          <span className="font-medium text-red-600">{executionReport.rejection_count}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Current Status:</span>
                          <Badge className={getStatusColor(executionReport.current_status)}>
                            {executionReport.current_status}
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                ) : (
                  <Card>
                    <CardContent className="flex items-center justify-center h-32">
                      <p className="text-muted-foreground">No execution report available</p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>
            </ScrollArea>
          </Tabs>
        </DialogContent>
      </Dialog>

      <AddStepDialog
        open={showAddStepDialog}
        onOpenChange={setShowAddStepDialog}
        workflowId={workflow.id}
        onStepAdded={() => {
          loadWorkflowDetails();
          onWorkflowUpdated();
        }}
      />
    </>
  );
}