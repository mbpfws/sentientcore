'use client';

import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Separator } from '@/components/ui/separator';
import { Loader2, AlertCircle, CheckCircle, XCircle, Clock, User, Calendar, FileText } from 'lucide-react';
import { InteractiveWorkflowService, PendingApproval, SubmitApprovalRequest } from '@/lib/api/interactive-workflow-service';

interface ApprovalDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workflowId: string;
  onApprovalSubmitted: () => void;
}

export function ApprovalDialog({ open, onOpenChange, workflowId, onApprovalSubmitted }: ApprovalDialogProps) {
  const [approvals, setApprovals] = useState<PendingApproval[]>([]);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedApproval, setSelectedApproval] = useState<PendingApproval | null>(null);
  const [decision, setDecision] = useState<'approve' | 'reject' | null>(null);
  const [comments, setComments] = useState('');

  useEffect(() => {
    if (open) {
      fetchPendingApprovals();
    }
  }, [open, workflowId]);

  const fetchPendingApprovals = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await InteractiveWorkflowService.getPendingApprovals(workflowId);
      setApprovals(response.approvals || []);
      if (response.approvals && response.approvals.length > 0) {
        setSelectedApproval(response.approvals[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch pending approvals');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitApproval = async () => {
    if (!selectedApproval || !decision) {
      setError('Please select an approval and make a decision');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const request: SubmitApprovalRequest = {
        approval_id: selectedApproval.id,
        decision,
        comments: comments.trim() || undefined
      };

      await InteractiveWorkflowService.submitApproval(workflowId, request);
      onApprovalSubmitted();
      onOpenChange(false);
      // Reset state
      setSelectedApproval(null);
      setDecision(null);
      setComments('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit approval');
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = () => {
    onOpenChange(false);
    setError(null);
    setSelectedApproval(null);
    setDecision(null);
    setComments('');
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
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

  const getStatusIcon = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'pending': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'approved': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'rejected': return <XCircle className="h-4 w-4 text-red-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[800px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Pending Approvals</DialogTitle>
          <DialogDescription>
            Review and approve or reject pending workflow steps.
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin" />
            <span className="ml-2">Loading approvals...</span>
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : approvals.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">No Pending Approvals</h3>
            <p className="text-muted-foreground">All workflow steps have been approved or don't require approval.</p>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Approval List */}
            <div className="space-y-3">
              <h4 className="font-semibold">Select Approval to Review:</h4>
              <div className="grid gap-3 max-h-60 overflow-y-auto">
                {approvals.map((approval) => (
                  <Card 
                    key={approval.id} 
                    className={`cursor-pointer transition-colors ${
                      selectedApproval?.id === approval.id 
                        ? 'ring-2 ring-primary bg-primary/5' 
                        : 'hover:bg-muted/50'
                    }`}
                    onClick={() => setSelectedApproval(approval)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(approval.status)}
                            <h5 className="font-medium">{approval.step_name}</h5>
                            <Badge variant={getPriorityColor(approval.priority)}>
                              {approval.priority || 'Medium'}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">{approval.description}</p>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <div className="flex items-center gap-1">
                              <User className="h-3 w-3" />
                              {approval.requested_by || 'System'}
                            </div>
                            <div className="flex items-center gap-1">
                              <Calendar className="h-3 w-3" />
                              {formatDate(approval.created_at)}
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            {/* Selected Approval Details */}
            {selectedApproval && (
              <>
                <Separator />
                <div className="space-y-4">
                  <h4 className="font-semibold">Approval Details:</h4>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <FileText className="h-5 w-5" />
                        {selectedApproval.step_name}
                      </CardTitle>
                      <CardDescription>{selectedApproval.description}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <Label className="text-muted-foreground">Status</Label>
                          <div className="flex items-center gap-2 mt-1">
                            {getStatusIcon(selectedApproval.status)}
                            <span className="capitalize">{selectedApproval.status}</span>
                          </div>
                        </div>
                        <div>
                          <Label className="text-muted-foreground">Priority</Label>
                          <div className="mt-1">
                            <Badge variant={getPriorityColor(selectedApproval.priority)}>
                              {selectedApproval.priority || 'Medium'}
                            </Badge>
                          </div>
                        </div>
                        <div>
                          <Label className="text-muted-foreground">Requested By</Label>
                          <p className="mt-1">{selectedApproval.requested_by || 'System'}</p>
                        </div>
                        <div>
                          <Label className="text-muted-foreground">Created</Label>
                          <p className="mt-1">{formatDate(selectedApproval.created_at)}</p>
                        </div>
                      </div>

                      {selectedApproval.context && (
                        <div>
                          <Label className="text-muted-foreground">Context</Label>
                          <div className="mt-1 p-3 bg-muted rounded-md">
                            <pre className="text-sm whitespace-pre-wrap">
                              {typeof selectedApproval.context === 'string' 
                                ? selectedApproval.context 
                                : JSON.stringify(selectedApproval.context, null, 2)
                              }
                            </pre>
                          </div>
                        </div>
                      )}

                      {selectedApproval.metadata && Object.keys(selectedApproval.metadata).length > 0 && (
                        <div>
                          <Label className="text-muted-foreground">Additional Information</Label>
                          <div className="mt-1 p-3 bg-muted rounded-md">
                            <pre className="text-sm whitespace-pre-wrap">
                              {JSON.stringify(selectedApproval.metadata, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  {/* Decision Section */}
                  <div className="space-y-4">
                    <h4 className="font-semibold">Make Decision:</h4>
                    
                    {/* Decision Buttons */}
                    <div className="flex gap-3">
                      <Button
                        variant={decision === 'approve' ? 'default' : 'outline'}
                        onClick={() => setDecision('approve')}
                        disabled={submitting}
                        className="flex-1"
                      >
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Approve
                      </Button>
                      <Button
                        variant={decision === 'reject' ? 'destructive' : 'outline'}
                        onClick={() => setDecision('reject')}
                        disabled={submitting}
                        className="flex-1"
                      >
                        <XCircle className="h-4 w-4 mr-2" />
                        Reject
                      </Button>
                    </div>

                    {/* Comments */}
                    <div className="space-y-2">
                      <Label htmlFor="approval-comments">
                        Comments {decision === 'reject' ? '(Required for rejection)' : '(Optional)'}
                      </Label>
                      <Textarea
                        id="approval-comments"
                        placeholder={decision === 'approve' 
                          ? 'Optional: Add any comments about your approval...' 
                          : 'Please explain why you are rejecting this step...'
                        }
                        value={comments}
                        onChange={(e) => setComments(e.target.value)}
                        disabled={submitting}
                        rows={3}
                        required={decision === 'reject'}
                      />
                    </div>

                    {/* Submit Error */}
                    {error && (
                      <Alert variant="destructive">
                        <AlertCircle className="h-4 w-4" />
                        <AlertDescription>{error}</AlertDescription>
                      </Alert>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={handleCancel}
            disabled={submitting}
          >
            Cancel
          </Button>
          {selectedApproval && (
            <Button
              onClick={handleSubmitApproval}
              disabled={submitting || !decision || (decision === 'reject' && !comments.trim())}
            >
              {submitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Submitting...
                </>
              ) : (
                `Submit ${decision === 'approve' ? 'Approval' : 'Rejection'}`
              )}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}