'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Play, Pause, Square, RotateCcw, Plus, Trash2, Eye, Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { InteractiveWorkflowService, InteractiveWorkflow, WorkflowMetrics } from '@/lib/api/interactive-workflow-service';
import { CreateWorkflowDialog } from './CreateWorkflowDialog';
import { WorkflowDetailsDialog } from './WorkflowDetailsDialog';
import { ApprovalDialog } from './ApprovalDialog';
import { TaskBreakdownDialog } from './TaskBreakdownDialog';

interface InteractiveWorkflowDashboardProps {
  className?: string;
}

export function InteractiveWorkflowDashboard({ className }: InteractiveWorkflowDashboardProps) {
  const [workflows, setWorkflows] = useState<InteractiveWorkflow[]>([]);
  const [metrics, setMetrics] = useState<WorkflowMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedWorkflow, setSelectedWorkflow] = useState<InteractiveWorkflow | null>(null);
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showDetailsDialog, setShowDetailsDialog] = useState(false);
  const [showApprovalDialog, setShowApprovalDialog] = useState(false);
  const [showTaskBreakdownDialog, setShowTaskBreakdownDialog] = useState(false);
  const [activeTab, setActiveTab] = useState('all');
  const [refreshing, setRefreshing] = useState(false);

  // Load workflows and metrics
  const loadData = async () => {
    try {
      setRefreshing(true);
      const [workflowsResponse, metricsResponse] = await Promise.all([
        InteractiveWorkflowService.listWorkflows(activeTab === 'all' ? undefined : activeTab),
        InteractiveWorkflowService.getMetrics()
      ]);
      setWorkflows(workflowsResponse.workflows);
      setMetrics(metricsResponse);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [activeTab]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [activeTab]);

  const handleWorkflowAction = async (workflowId: string, action: 'start' | 'pause' | 'resume' | 'cancel' | 'restart') => {
    try {
      if (action === 'start') {
        await InteractiveWorkflowService.startWorkflow(workflowId);
      } else {
        await InteractiveWorkflowService.controlWorkflow(workflowId, { action });
      }
      await loadData(); // Refresh data
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to ${action} workflow`);
    }
  };

  const handleDeleteWorkflow = async (workflowId: string) => {
    if (!confirm('Are you sure you want to delete this workflow?')) return;
    
    try {
      await InteractiveWorkflowService.deleteWorkflow(workflowId);
      await loadData(); // Refresh data
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete workflow');
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

  const filteredWorkflows = workflows.filter(workflow => {
    if (activeTab === 'all') return true;
    return workflow.status.toLowerCase() === activeTab;
  });

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading workflows...</span>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Interactive Workflows</h1>
          <p className="text-muted-foreground">Manage and monitor your interactive workflows</p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            onClick={loadData}
            disabled={refreshing}
          >
            {refreshing ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
            Refresh
          </Button>
          <Button onClick={() => setShowTaskBreakdownDialog(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Break Down Task
          </Button>
          <Button onClick={() => setShowCreateDialog(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Create Workflow
          </Button>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Metrics Cards */}
      {metrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Workflows</CardTitle>
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metrics.total_workflows}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active</CardTitle>
              <Play className="h-4 w-4 text-blue-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{metrics.active_workflows}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Completed</CardTitle>
              <CheckCircle className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{metrics.completed_workflows}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Approval Rate</CardTitle>
              <CheckCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{Math.round(metrics.approval_rate * 100)}%</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Workflow Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList>
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="pending">Pending</TabsTrigger>
          <TabsTrigger value="running">Running</TabsTrigger>
          <TabsTrigger value="paused">Paused</TabsTrigger>
          <TabsTrigger value="completed">Completed</TabsTrigger>
          <TabsTrigger value="failed">Failed</TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="space-y-4">
          {filteredWorkflows.length === 0 ? (
            <Card>
              <CardContent className="flex items-center justify-center h-32">
                <p className="text-muted-foreground">No workflows found</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
              {filteredWorkflows.map((workflow) => (
                <Card key={workflow.id} className="hover:shadow-md transition-shadow">
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{workflow.name}</CardTitle>
                      <div className="flex items-center space-x-1">
                        {getStatusIcon(workflow.status)}
                        <Badge className={getStatusColor(workflow.status)}>
                          {workflow.status}
                        </Badge>
                      </div>
                    </div>
                    <CardDescription className="line-clamp-2">
                      {workflow.description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {/* Progress */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span>Progress</span>
                          <span>{workflow.completion_percentage}%</span>
                        </div>
                        <Progress value={workflow.completion_percentage} className="h-2" />
                      </div>

                      {/* Steps Info */}
                      <div className="flex justify-between text-sm text-muted-foreground">
                        <span>Steps: {workflow.current_step_index + 1}/{workflow.total_steps}</span>
                        <span>Created: {new Date(workflow.created_at).toLocaleDateString()}</span>
                      </div>

                      {/* Action Buttons */}
                      <div className="flex items-center justify-between pt-2">
                        <div className="flex items-center space-x-1">
                          {workflow.status === 'pending' && (
                            <Button
                              size="sm"
                              onClick={() => handleWorkflowAction(workflow.id, 'start')}
                            >
                              <Play className="h-3 w-3" />
                            </Button>
                          )}
                          {workflow.status === 'running' && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleWorkflowAction(workflow.id, 'pause')}
                            >
                              <Pause className="h-3 w-3" />
                            </Button>
                          )}
                          {workflow.status === 'paused' && (
                            <Button
                              size="sm"
                              onClick={() => handleWorkflowAction(workflow.id, 'resume')}
                            >
                              <Play className="h-3 w-3" />
                            </Button>
                          )}
                          {(workflow.status === 'running' || workflow.status === 'paused') && (
                            <Button
                              size="sm"
                              variant="destructive"
                              onClick={() => handleWorkflowAction(workflow.id, 'cancel')}
                            >
                              <Square className="h-3 w-3" />
                            </Button>
                          )}
                          {(workflow.status === 'completed' || workflow.status === 'failed') && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleWorkflowAction(workflow.id, 'restart')}
                            >
                              <RotateCcw className="h-3 w-3" />
                            </Button>
                          )}
                        </div>
                        <div className="flex items-center space-x-1">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              setSelectedWorkflow(workflow);
                              setShowDetailsDialog(true);
                            }}
                          >
                            <Eye className="h-3 w-3" />
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleDeleteWorkflow(workflow.id)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Dialogs */}
      <CreateWorkflowDialog
        open={showCreateDialog}
        onOpenChange={setShowCreateDialog}
        onWorkflowCreated={loadData}
      />

      {selectedWorkflow && (
        <WorkflowDetailsDialog
          open={showDetailsDialog}
          onOpenChange={setShowDetailsDialog}
          workflow={selectedWorkflow}
          onWorkflowUpdated={loadData}
        />
      )}

      <ApprovalDialog
        open={showApprovalDialog}
        onOpenChange={setShowApprovalDialog}
        workflowId={selectedWorkflow?.id || ''}
        onApprovalSubmitted={loadData}
      />

      <TaskBreakdownDialog
        open={showTaskBreakdownDialog}
        onOpenChange={setShowTaskBreakdownDialog}
        onTaskBrokenDown={loadData}
      />
    </div>
  );
}