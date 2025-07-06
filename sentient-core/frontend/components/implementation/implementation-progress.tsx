'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  Play, 
  Download, 
  FileText, 
  Folder, 
  TestTube,
  Building,
  Eye
} from 'lucide-react';
import { 
  ImplementationProgress as IProgress,
  ImplementationPhase,
  ImplementationArtifact,
  ImplementationUpdate,
  TestResult,
  ImplementationService
} from '@/lib/api/implementation-service';

interface ImplementationProgressProps {
  workflowId: string;
  onArtifactView?: (artifact: ImplementationArtifact) => void;
}

const getPhaseIcon = (status: string) => {
  switch (status) {
    case 'completed':
      return <CheckCircle className="h-4 w-4 text-green-500" />;
    case 'in_progress':
      return <Play className="h-4 w-4 text-blue-500 animate-pulse" />;
    case 'failed':
      return <AlertCircle className="h-4 w-4 text-red-500" />;
    default:
      return <Clock className="h-4 w-4 text-gray-400" />;
  }
};

const getArtifactIcon = (type: string) => {
  switch (type) {
    case 'file':
      return <FileText className="h-4 w-4" />;
    case 'directory':
      return <Folder className="h-4 w-4" />;
    case 'test_result':
      return <TestTube className="h-4 w-4" />;
    case 'build_output':
      return <Building className="h-4 w-4" />;
    default:
      return <FileText className="h-4 w-4" />;
  }
};

const getStatusBadgeVariant = (status: string) => {
  switch (status) {
    case 'completed':
      return 'default';
    case 'in_progress':
      return 'secondary';
    case 'failed':
      return 'destructive';
    default:
      return 'outline';
  }
};

export const ImplementationProgress: React.FC<ImplementationProgressProps> = ({
  workflowId,
  onArtifactView
}) => {
  const [progress, setProgress] = useState<IProgress | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [updates, setUpdates] = useState<ImplementationUpdate[]>([]);

  useEffect(() => {
    let cleanup: (() => void) | null = null;

    const loadProgress = async () => {
      try {
        setLoading(true);
        const progressData = await ImplementationService.getProgress(workflowId);
        setProgress(progressData);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load progress');
      } finally {
        setLoading(false);
      }
    };

    const subscribeToUpdates = () => {
      cleanup = ImplementationService.subscribeToUpdates(workflowId, (update) => {
        setUpdates(prev => [update, ...prev].slice(0, 50)); // Keep last 50 updates
        
        // Update progress when receiving updates
        if (update.type === 'progress_update') {
          setProgress(prev => prev ? {
            ...prev,
            overall_progress: update.progress,
            current_phase: update.phase
          } : null);
        }
      });
    };

    loadProgress().then(() => {
      subscribeToUpdates();
    });

    return () => {
      if (cleanup) cleanup();
    };
  }, [workflowId]);

  const handleDownloadArtifact = async (artifact: ImplementationArtifact) => {
    try {
      const blob = await ImplementationService.downloadArtifact(workflowId, artifact.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = artifact.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Failed to download artifact:', err);
    }
  };

  const handleViewArtifact = (artifact: ImplementationArtifact) => {
    if (onArtifactView) {
      onArtifactView(artifact);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <span className="ml-2">Loading implementation progress...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center text-red-500">
            <AlertCircle className="h-5 w-5 mr-2" />
            <span>Error loading progress: {error}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!progress) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="text-center text-gray-500">
            No implementation progress found
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Overall Progress */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Implementation Progress</span>
            <Badge variant={getStatusBadgeVariant(progress.current_phase)}>
              {progress.current_phase}
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Overall Progress</span>
              <span>{Math.round(progress.overall_progress)}%</span>
            </div>
            <Progress value={progress.overall_progress} className="w-full" />
          </div>
        </CardContent>
      </Card>

      {/* Detailed Progress */}
      <Tabs defaultValue="phases" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="phases">Phases</TabsTrigger>
          <TabsTrigger value="artifacts">Artifacts ({progress.total_artifacts})</TabsTrigger>
          <TabsTrigger value="tests">Tests</TabsTrigger>
          <TabsTrigger value="updates">Updates</TabsTrigger>
        </TabsList>

        <TabsContent value="phases">
          <Card>
            <CardContent className="p-4">
              <ScrollArea className="h-64">
                <div className="space-y-3">
                  {progress.phases.map((phase) => (
                    <div key={phase.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center space-x-3">
                        {getPhaseIcon(phase.status)}
                        <div>
                          <div className="font-medium">{phase.name}</div>
                          <div className="text-sm text-gray-500">{phase.description}</div>
                          {phase.error_message && (
                            <div className="text-sm text-red-500 mt-1">{phase.error_message}</div>
                          )}
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant={getStatusBadgeVariant(phase.status)}>
                          {phase.status}
                        </Badge>
                        <div className="text-sm text-gray-500 mt-1">
                          {Math.round(phase.progress)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="artifacts">
          <Card>
            <CardContent className="p-4">
              <ScrollArea className="h-64">
                <div className="space-y-2">
                  {progress.phases.flatMap(phase => phase.artifacts || []).map((artifact) => (
                    <div key={artifact.id} className="flex items-center justify-between p-2 border rounded">
                      <div className="flex items-center space-x-2">
                        {getArtifactIcon(artifact.type)}
                        <div>
                          <div className="font-medium text-sm">{artifact.name}</div>
                          <div className="text-xs text-gray-500">{artifact.path}</div>
                        </div>
                      </div>
                      <div className="flex space-x-1">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleViewArtifact(artifact)}
                        >
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleDownloadArtifact(artifact)}
                        >
                          <Download className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tests">
          <Card>
            <CardContent className="p-4">
              <ScrollArea className="h-64">
                {progress.test_results && progress.test_results.length > 0 ? (
                  <div className="space-y-2">
                    {progress.test_results.map((test) => (
                      <div key={test.id} className="flex items-center justify-between p-2 border rounded">
                        <div className="flex items-center space-x-2">
                          <TestTube className="h-4 w-4" />
                          <div>
                            <div className="font-medium text-sm">{test.test_name}</div>
                            <div className="text-xs text-gray-500">{test.file_path}</div>
                            {test.error_message && (
                              <div className="text-xs text-red-500 mt-1">{test.error_message}</div>
                            )}
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge variant={test.status === 'passed' ? 'default' : test.status === 'failed' ? 'destructive' : 'secondary'}>
                            {test.status}
                          </Badge>
                          <div className="text-xs text-gray-500 mt-1">
                            {test.duration}ms
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    No test results available
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="updates">
          <Card>
            <CardContent className="p-4">
              <ScrollArea className="h-64">
                {updates.length > 0 ? (
                  <div className="space-y-2">
                    {updates.map((update, index) => (
                      <div key={index} className="p-2 border rounded text-sm">
                        <div className="flex justify-between items-start">
                          <div className="font-medium">{update.type.replace('_', ' ')}</div>
                          <div className="text-xs text-gray-500">
                            {new Date(update.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                        <div className="text-gray-600 mt-1">{update.message}</div>
                        <div className="text-xs text-gray-500 mt-1">
                          Phase: {update.phase} | Progress: {Math.round(update.progress)}%
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    No updates available
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};