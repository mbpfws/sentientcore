'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, AlertCircle } from 'lucide-react';
import { InteractiveWorkflowService, CreateWorkflowRequest } from '@/lib/api/interactive-workflow-service';

interface CreateWorkflowDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onWorkflowCreated: () => void;
}

export function CreateWorkflowDialog({ open, onOpenChange, onWorkflowCreated }: CreateWorkflowDialogProps) {
  const [formData, setFormData] = useState<CreateWorkflowRequest>({
    name: '',
    description: '',
    metadata: {}
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      setError('Workflow name is required');
      return;
    }

    if (!formData.description.trim()) {
      setError('Workflow description is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await InteractiveWorkflowService.createWorkflow(formData);
      onWorkflowCreated();
      onOpenChange(false);
      // Reset form
      setFormData({
        name: '',
        description: '',
        metadata: {}
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workflow');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    onOpenChange(false);
    setError(null);
    // Reset form
    setFormData({
      name: '',
      description: '',
      metadata: {}
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Create New Workflow</DialogTitle>
          <DialogDescription>
            Create a new interactive workflow to manage your tasks and processes.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Error Alert */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Workflow Name */}
          <div className="space-y-2">
            <Label htmlFor="workflow-name">Workflow Name *</Label>
            <Input
              id="workflow-name"
              placeholder="Enter workflow name"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              disabled={loading}
              required
            />
          </div>

          {/* Workflow Description */}
          <div className="space-y-2">
            <Label htmlFor="workflow-description">Description *</Label>
            <Textarea
              id="workflow-description"
              placeholder="Describe what this workflow will accomplish"
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              disabled={loading}
              rows={4}
              required
            />
          </div>

          {/* Metadata (Optional) */}
          <div className="space-y-2">
            <Label htmlFor="workflow-tags">Tags (Optional)</Label>
            <Input
              id="workflow-tags"
              placeholder="Enter tags separated by commas (e.g., automation, testing, deployment)"
              onChange={(e) => {
                const tags = e.target.value.split(',').map(tag => tag.trim()).filter(Boolean);
                setFormData(prev => ({ 
                  ...prev, 
                  metadata: { ...prev.metadata, tags } 
                }));
              }}
              disabled={loading}
            />
          </div>

          {/* Priority */}
          <div className="space-y-2">
            <Label htmlFor="workflow-priority">Priority</Label>
            <select
              id="workflow-priority"
              className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
              onChange={(e) => setFormData(prev => ({ 
                ...prev, 
                metadata: { ...prev.metadata, priority: e.target.value } 
              }))}
              disabled={loading}
            >
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>
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
                  Creating...
                </>
              ) : (
                'Create Workflow'
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}