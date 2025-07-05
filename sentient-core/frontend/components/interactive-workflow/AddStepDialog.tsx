'use client';

import React, { useState } from 'react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Loader2, AlertCircle, X } from 'lucide-react';
import { InteractiveWorkflowService, AddStepRequest } from '@/lib/api/interactive-workflow-service';

interface AddStepDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workflowId: string;
  onStepAdded: () => void;
}

const STEP_TYPES = [
  { value: 'manual', label: 'Manual Task', description: 'Requires human intervention' },
  { value: 'automated', label: 'Automated Task', description: 'Runs automatically' },
  { value: 'approval', label: 'Approval Gate', description: 'Requires approval to proceed' },
  { value: 'validation', label: 'Validation', description: 'Validates previous steps' },
  { value: 'notification', label: 'Notification', description: 'Sends notifications' },
  { value: 'integration', label: 'Integration', description: 'Integrates with external systems' },
  { value: 'analysis', label: 'Analysis', description: 'Analyzes data or results' },
  { value: 'deployment', label: 'Deployment', description: 'Deploys code or resources' },
  { value: 'testing', label: 'Testing', description: 'Runs tests or validations' },
  { value: 'cleanup', label: 'Cleanup', description: 'Cleans up resources' }
];

export function AddStepDialog({ open, onOpenChange, workflowId, onStepAdded }: AddStepDialogProps) {
  const [formData, setFormData] = useState<AddStepRequest>({
    name: '',
    description: '',
    step_type: 'manual',
    metadata: {},
    dependencies: [],
    estimated_duration: undefined
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dependencyInput, setDependencyInput] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.name.trim()) {
      setError('Step name is required');
      return;
    }

    if (!formData.description.trim()) {
      setError('Step description is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      await InteractiveWorkflowService.addStep(workflowId, formData);
      onStepAdded();
      onOpenChange(false);
      // Reset form
      setFormData({
        name: '',
        description: '',
        step_type: 'manual',
        metadata: {},
        dependencies: [],
        estimated_duration: undefined
      });
      setDependencyInput('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add step');
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
      step_type: 'manual',
      metadata: {},
      dependencies: [],
      estimated_duration: undefined
    });
    setDependencyInput('');
  };

  const addDependency = () => {
    if (dependencyInput.trim() && !formData.dependencies?.includes(dependencyInput.trim())) {
      setFormData(prev => ({
        ...prev,
        dependencies: [...(prev.dependencies || []), dependencyInput.trim()]
      }));
      setDependencyInput('');
    }
  };

  const removeDependency = (dependency: string) => {
    setFormData(prev => ({
      ...prev,
      dependencies: prev.dependencies?.filter(dep => dep !== dependency) || []
    }));
  };

  const handleDurationChange = (value: string) => {
    const duration = parseInt(value);
    setFormData(prev => ({
      ...prev,
      estimated_duration: isNaN(duration) ? undefined : duration
    }));
  };

  const selectedStepType = STEP_TYPES.find(type => type.value === formData.step_type);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add New Step</DialogTitle>
          <DialogDescription>
            Add a new step to the workflow. Steps will be executed in the order they are added.
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

          {/* Step Name */}
          <div className="space-y-2">
            <Label htmlFor="step-name">Step Name *</Label>
            <Input
              id="step-name"
              placeholder="Enter step name"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              disabled={loading}
              required
            />
          </div>

          {/* Step Description */}
          <div className="space-y-2">
            <Label htmlFor="step-description">Description *</Label>
            <Textarea
              id="step-description"
              placeholder="Describe what this step will do"
              value={formData.description}
              onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              disabled={loading}
              rows={3}
              required
            />
          </div>

          {/* Step Type */}
          <div className="space-y-2">
            <Label htmlFor="step-type">Step Type *</Label>
            <select
              id="step-type"
              className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm"
              value={formData.step_type}
              onChange={(e) => setFormData(prev => ({ ...prev, step_type: e.target.value }))}
              disabled={loading}
              required
            >
              {STEP_TYPES.map((type) => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
            {selectedStepType && (
              <p className="text-sm text-muted-foreground">{selectedStepType.description}</p>
            )}
          </div>

          {/* Estimated Duration */}
          <div className="space-y-2">
            <Label htmlFor="estimated-duration">Estimated Duration (seconds)</Label>
            <Input
              id="estimated-duration"
              type="number"
              placeholder="Enter estimated duration in seconds"
              value={formData.estimated_duration || ''}
              onChange={(e) => handleDurationChange(e.target.value)}
              disabled={loading}
              min="1"
            />
            <p className="text-sm text-muted-foreground">
              Optional: How long you expect this step to take
            </p>
          </div>

          {/* Dependencies */}
          <div className="space-y-2">
            <Label htmlFor="dependencies">Dependencies</Label>
            <div className="flex space-x-2">
              <Input
                id="dependencies"
                placeholder="Enter dependency name"
                value={dependencyInput}
                onChange={(e) => setDependencyInput(e.target.value)}
                disabled={loading}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    addDependency();
                  }
                }}
              />
              <Button
                type="button"
                variant="outline"
                onClick={addDependency}
                disabled={loading || !dependencyInput.trim()}
              >
                Add
              </Button>
            </div>
            {formData.dependencies && formData.dependencies.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-2">
                {formData.dependencies.map((dependency, index) => (
                  <Badge key={index} variant="secondary" className="flex items-center gap-1">
                    {dependency}
                    <button
                      type="button"
                      onClick={() => removeDependency(dependency)}
                      className="ml-1 hover:bg-destructive hover:text-destructive-foreground rounded-full p-0.5"
                      disabled={loading}
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
            <p className="text-sm text-muted-foreground">
              Optional: Other steps that must complete before this step can start
            </p>
          </div>

          {/* Priority */}
          <div className="space-y-2">
            <Label htmlFor="step-priority">Priority</Label>
            <select
              id="step-priority"
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

          {/* Retry Configuration */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="max-retries">Max Retries</Label>
              <Input
                id="max-retries"
                type="number"
                placeholder="3"
                onChange={(e) => {
                  const retries = parseInt(e.target.value);
                  setFormData(prev => ({ 
                    ...prev, 
                    metadata: { 
                      ...prev.metadata, 
                      max_retries: isNaN(retries) ? undefined : retries 
                    } 
                  }));
                }}
                disabled={loading}
                min="0"
                max="10"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="retry-delay">Retry Delay (seconds)</Label>
              <Input
                id="retry-delay"
                type="number"
                placeholder="30"
                onChange={(e) => {
                  const delay = parseInt(e.target.value);
                  setFormData(prev => ({ 
                    ...prev, 
                    metadata: { 
                      ...prev.metadata, 
                      retry_delay: isNaN(delay) ? undefined : delay 
                    } 
                  }));
                }}
                disabled={loading}
                min="1"
              />
            </div>
          </div>

          {/* Timeout */}
          <div className="space-y-2">
            <Label htmlFor="timeout">Timeout (seconds)</Label>
            <Input
              id="timeout"
              type="number"
              placeholder="Enter timeout in seconds"
              onChange={(e) => {
                const timeout = parseInt(e.target.value);
                setFormData(prev => ({ 
                  ...prev, 
                  metadata: { 
                    ...prev.metadata, 
                    timeout: isNaN(timeout) ? undefined : timeout 
                  } 
                }));
              }}
              disabled={loading}
              min="1"
            />
            <p className="text-sm text-muted-foreground">
              Optional: Maximum time allowed for this step to complete
            </p>
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
                  Adding...
                </>
              ) : (
                'Add Step'
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}