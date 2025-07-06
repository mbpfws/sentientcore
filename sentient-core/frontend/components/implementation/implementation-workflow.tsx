'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Play, 
  Square, 
  Upload, 
  FileText, 
  AlertCircle, 
  CheckCircle,
  Clock,
  Settings
} from 'lucide-react';
import { 
  ImplementationService,
  ImplementationRequest,
  ImplementationResult,
  ImplementationArtifact
} from '@/lib/api/implementation-service';
import { ImplementationProgress } from './implementation-progress';

interface ImplementationWorkflowProps {
  onWorkflowStart?: (workflowId: string) => void;
  onWorkflowComplete?: (result: ImplementationResult) => void;
}

const implementationModes = [
  { value: 'validation', label: 'Validation Only', description: 'Validate plan and knowledge documents' },
  { value: 'phase2', label: 'Code Generation', description: 'Execute Phase 2: Code Generation & Implementation' },
  { value: 'phase3', label: 'Testing & Validation', description: 'Execute Phase 3: Continuous Validation & Testing' },
  { value: 'phase4', label: 'Reporting & Output', description: 'Execute Phase 4: Reporting & Output Management' },
  { value: 'full', label: 'Full Implementation', description: 'Execute all phases sequentially' }
];

export const ImplementationWorkflow: React.FC<ImplementationWorkflowProps> = ({
  onWorkflowStart,
  onWorkflowComplete
}) => {
  const [featureBuildPlan, setFeatureBuildPlan] = useState<string>('');
  const [synthesizedKnowledge, setSynthesizedKnowledge] = useState<string[]>(['']);
  const [implementationMode, setImplementationMode] = useState<string>('validation');
  const [workflowId, setWorkflowId] = useState<string>('');
  const [currentResult, setCurrentResult] = useState<ImplementationResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedArtifact, setSelectedArtifact] = useState<ImplementationArtifact | null>(null);
  const [artifactContent, setArtifactContent] = useState<string>('');

  const generateWorkflowId = () => {
    return `impl_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const handleAddKnowledgeDocument = () => {
    setSynthesizedKnowledge([...synthesizedKnowledge, '']);
  };

  const handleRemoveKnowledgeDocument = (index: number) => {
    if (synthesizedKnowledge.length > 1) {
      setSynthesizedKnowledge(synthesizedKnowledge.filter((_, i) => i !== index));
    }
  };

  const handleKnowledgeDocumentChange = (index: number, value: string) => {
    const updated = [...synthesizedKnowledge];
    updated[index] = value;
    setSynthesizedKnowledge(updated);
  };

  const handleStartImplementation = async () => {
    if (!featureBuildPlan.trim()) {
      setError('Feature Build Plan is required');
      return;
    }

    if (!workflowId.trim()) {
      setError('Workflow ID is required');
      return;
    }

    const filteredKnowledge = synthesizedKnowledge.filter(doc => doc.trim());
    if (filteredKnowledge.length === 0) {
      setError('At least one Synthesized Knowledge document is required');
      return;
    }

    try {
      setIsRunning(true);
      setError(null);

      const request: ImplementationRequest = {
        feature_build_plan: featureBuildPlan,
        synthesized_knowledge: filteredKnowledge,
        implementation_mode: implementationMode as any,
        workflow_id: workflowId
      };

      const result = await ImplementationService.startImplementation(request);
      setCurrentResult(result);
      
      if (onWorkflowStart) {
        onWorkflowStart(workflowId);
      }

      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const updatedResult = await ImplementationService.getResult(result.id);
          setCurrentResult(updatedResult);
          
          if (updatedResult.status === 'completed' || updatedResult.status === 'failed') {
            clearInterval(pollInterval);
            setIsRunning(false);
            
            if (onWorkflowComplete) {
              onWorkflowComplete(updatedResult);
            }
          }
        } catch (err) {
          console.error('Error polling implementation result:', err);
        }
      }, 2000);

      // Cleanup interval after 30 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        setIsRunning(false);
      }, 30 * 60 * 1000);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start implementation');
      setIsRunning(false);
    }
  };

  const handleCancelImplementation = async () => {
    if (currentResult) {
      try {
        await ImplementationService.cancelImplementation(currentResult.id);
        setIsRunning(false);
        setCurrentResult(null);
      } catch (err) {
        console.error('Error canceling implementation:', err);
      }
    }
  };

  const handleArtifactView = async (artifact: ImplementationArtifact) => {
    try {
      setSelectedArtifact(artifact);
      if (artifact.type === 'file' || artifact.type === 'documentation') {
        const content = await ImplementationService.getArtifactContent(workflowId, artifact.id);
        setArtifactContent(content);
      } else {
        setArtifactContent(`Artifact: ${artifact.name}\nType: ${artifact.type}\nPath: ${artifact.path}\nSize: ${artifact.size || 'Unknown'}`);
      }
    } catch (err) {
      setArtifactContent(`Error loading artifact content: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'in_progress':
      case 'started':
        return <Clock className="h-4 w-4 text-blue-500 animate-pulse" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'completed':
        return 'default';
      case 'in_progress':
      case 'started':
        return 'secondary';
      case 'failed':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Settings className="h-5 w-5" />
            <span>Implementation Workflow Configuration</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Workflow ID */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Workflow ID</label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={workflowId}
                onChange={(e) => setWorkflowId(e.target.value)}
                placeholder="Enter workflow ID or generate one"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isRunning}
              />
              <Button
                onClick={() => setWorkflowId(generateWorkflowId())}
                variant="outline"
                disabled={isRunning}
              >
                Generate
              </Button>
            </div>
          </div>

          {/* Implementation Mode */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Implementation Mode</label>
            <Select value={implementationMode} onValueChange={setImplementationMode} disabled={isRunning}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {implementationModes.map((mode) => (
                  <SelectItem key={mode.value} value={mode.value}>
                    <div>
                      <div className="font-medium">{mode.label}</div>
                      <div className="text-sm text-gray-500">{mode.description}</div>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Feature Build Plan */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Feature Build Plan</label>
            <Textarea
              value={featureBuildPlan}
              onChange={(e) => setFeatureBuildPlan(e.target.value)}
              placeholder="Paste your Feature Build Plan (markdown format)..."
              rows={8}
              disabled={isRunning}
            />
          </div>

          {/* Synthesized Knowledge Documents */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Synthesized Knowledge Documents</label>
              <Button
                onClick={handleAddKnowledgeDocument}
                variant="outline"
                size="sm"
                disabled={isRunning}
              >
                Add Document
              </Button>
            </div>
            <div className="space-y-2">
              {synthesizedKnowledge.map((doc, index) => (
                <div key={index} className="flex space-x-2">
                  <Textarea
                    value={doc}
                    onChange={(e) => handleKnowledgeDocumentChange(index, e.target.value)}
                    placeholder={`Synthesized Knowledge Document ${index + 1} (markdown format)...`}
                    rows={4}
                    disabled={isRunning}
                    className="flex-1"
                  />
                  {synthesizedKnowledge.length > 1 && (
                    <Button
                      onClick={() => handleRemoveKnowledgeDocument(index)}
                      variant="outline"
                      size="sm"
                      disabled={isRunning}
                    >
                      Remove
                    </Button>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="flex items-center space-x-2 text-red-600 bg-red-50 p-3 rounded-md">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          {/* Control Buttons */}
          <div className="flex space-x-2">
            <Button
              onClick={handleStartImplementation}
              disabled={isRunning || !featureBuildPlan.trim() || !workflowId.trim()}
              className="flex items-center space-x-2"
            >
              <Play className="h-4 w-4" />
              <span>{isRunning ? 'Running...' : 'Start Implementation'}</span>
            </Button>
            
            {isRunning && (
              <Button
                onClick={handleCancelImplementation}
                variant="destructive"
                className="flex items-center space-x-2"
              >
                <Square className="h-4 w-4" />
                <span>Cancel</span>
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Current Implementation Status */}
      {currentResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {getStatusIcon(currentResult.status)}
                <span>Implementation Status</span>
              </div>
              <Badge variant={getStatusBadgeVariant(currentResult.status)}>
                {currentResult.status}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-sm">
                <strong>ID:</strong> {currentResult.id}
              </div>
              <div className="text-sm">
                <strong>Workflow:</strong> {currentResult.workflow_id}
              </div>
              <div className="text-sm">
                <strong>Phase:</strong> {currentResult.phase}
              </div>
              <div className="text-sm">
                <strong>Message:</strong> {currentResult.message}
              </div>
              {currentResult.error && (
                <div className="text-sm text-red-600">
                  <strong>Error:</strong> {currentResult.error}
                </div>
              )}
              <div className="text-xs text-gray-500">
                <strong>Started:</strong> {new Date(currentResult.created_at).toLocaleString()}
              </div>
              <div className="text-xs text-gray-500">
                <strong>Updated:</strong> {new Date(currentResult.updated_at).toLocaleString()}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Implementation Progress */}
      {workflowId && currentResult && (
        <ImplementationProgress
          workflowId={workflowId}
          onArtifactView={handleArtifactView}
        />
      )}

      {/* Artifact Viewer */}
      {selectedArtifact && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <FileText className="h-5 w-5" />
              <span>Artifact: {selectedArtifact.name}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-sm text-gray-600">
                <strong>Type:</strong> {selectedArtifact.type} | 
                <strong>Path:</strong> {selectedArtifact.path}
              </div>
              <ScrollArea className="h-64 w-full border rounded-md p-4">
                <pre className="text-sm whitespace-pre-wrap">{artifactContent}</pre>
              </ScrollArea>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};