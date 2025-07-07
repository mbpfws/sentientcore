'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import ReactMarkdown from 'react-markdown';
import { Download, FileText, Search, Clock, ExternalLink, BookOpen } from 'lucide-react';
import { researchService } from '@/lib/api/research-service';

interface ResearchResult {
  id: string;
  query: string;
  mode: 'knowledge' | 'deep' | 'best_in_class';
  status: 'pending' | 'searching' | 'synthesizing' | 'completed' | 'error';
  progress: number;
  results?: {
    summary: string;
    sources: Array<{
      title: string;
      url: string;
      snippet: string;
      relevance_score?: number;
    }>;
    insights?: string[];
    recommendations?: string[];
    citations?: string[];
  };
  created_at: string;
  completed_at?: string;
  verbose_log?: string[];
}

interface ResearchResultsProps {
  workflowId: string;
}

const ResearchResults: React.FC<ResearchResultsProps> = ({ workflowId }) => {
  const [results, setResults] = useState<ResearchResult[]>([]);
  const [selectedResult, setSelectedResult] = useState<ResearchResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [verboseMode, setVerboseMode] = useState(false);

  // Handle real-time updates
  const handleResearchUpdate = useCallback((updatedResult: ResearchResult) => {
    setResults(prevResults => {
      const existingIndex = prevResults.findIndex(r => r.id === updatedResult.id);
      if (existingIndex >= 0) {
        // Update existing result
        const newResults = [...prevResults];
        newResults[existingIndex] = updatedResult;
        return newResults.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      } else {
        // Add new result
        return [updatedResult, ...prevResults].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      }
    });
    
    // Update selected result if it's the one being updated
    if (selectedResult?.id === updatedResult.id) {
      setSelectedResult(updatedResult);
    }
  }, [selectedResult]);

  // Load research results from localStorage and API
  useEffect(() => {
    loadResearchResults();
    
    // Set up SSE listener for real-time updates
    const listenerId = `research-results-${workflowId}`;
    researchService.addUpdateListener(listenerId, handleResearchUpdate);
    
    // Cleanup on unmount
    return () => {
      researchService.removeUpdateListener(listenerId);
    };
  }, [workflowId, handleResearchUpdate]);

  const loadResearchResults = async () => {
    setIsLoading(true);
    try {
      // Load from localStorage first for immediate display
      const localResults = researchService.getLocalResults(workflowId);
      if (localResults.length > 0) {
        setResults(localResults);
      }

      // Then fetch from API for latest updates
      const apiResponse = await researchService.getResearchResults(workflowId);
      if (apiResponse.data) {
        setResults(apiResponse.data);
      }
    } catch (error) {
      console.error('Failed to load research results:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const downloadAsMarkdown = async (result: ResearchResult) => {
    try {
      const markdown = await researchService.exportToMarkdown(result.id);
      const blob = new Blob([markdown], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `research_${result.query.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download Markdown:', error);
    }
  };

  const downloadAsPDF = async (result: ResearchResult) => {
    try {
      const blob = await researchService.exportToPDF(result.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `research_${result.query.replace(/[^a-zA-Z0-9]/g, '_')}_${Date.now()}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to download PDF:', error);
    }
  };

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case 'knowledge': return <BookOpen className="h-4 w-4" />;
      case 'deep': return <Search className="h-4 w-4" />;
      case 'best_in_class': return <ExternalLink className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getModeColor = (mode: string) => {
    switch (mode) {
      case 'knowledge': return 'bg-blue-100 text-blue-800';
      case 'deep': return 'bg-purple-100 text-purple-800';
      case 'best_in_class': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'searching': return 'bg-blue-100 text-blue-800';
      case 'synthesizing': return 'bg-purple-100 text-purple-800';
      case 'error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Research Results</h2>
        <div className="flex gap-2">
          <Button
            variant={verboseMode ? 'default' : 'outline'}
            size="sm"
            onClick={() => setVerboseMode(!verboseMode)}
          >
            Verbose Mode
          </Button>
          <Button variant="outline" size="sm" onClick={loadResearchResults}>
            Refresh
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Results List */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Research Sessions</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[600px]">
                {isLoading ? (
                  <div className="text-center py-8">
                    <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
                    <p className="text-sm text-muted-foreground">Loading results...</p>
                  </div>
                ) : results.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    <Search className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No research results yet</p>
                    <p className="text-sm">Start a research query to see results here</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {results.map((result) => (
                      <Card 
                        key={result.id}
                        className={`cursor-pointer transition-colors hover:bg-muted/50 ${
                          selectedResult?.id === result.id ? 'ring-2 ring-primary' : ''
                        }`}
                        onClick={() => setSelectedResult(result)}
                      >
                        <CardContent className="p-4">
                          <div className="space-y-2">
                            <div className="flex items-start justify-between">
                              <h4 className="font-medium text-sm line-clamp-2">{result.query}</h4>
                              <div className="flex gap-1 ml-2">
                                <Badge className={`text-xs ${getModeColor(result.mode)}`}>
                                  {getModeIcon(result.mode)}
                                  <span className="ml-1">{result.mode.replace('_', ' ')}</span>
                                </Badge>
                              </div>
                            </div>
                            
                            <div className="flex items-center justify-between">
                              <Badge className={`text-xs ${getStatusColor(result.status)}`}>
                                {result.status}
                              </Badge>
                              <div className="flex items-center text-xs text-muted-foreground">
                                <Clock className="h-3 w-3 mr-1" />
                                {new Date(result.created_at).toLocaleDateString()}
                              </div>
                            </div>
                            
                            {result.status !== 'completed' && result.status !== 'error' && (
                              <Progress value={result.progress} className="h-1" />
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Result Details */}
        <div className="lg:col-span-2">
          {selectedResult ? (
            <Card>
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-xl">{selectedResult.query}</CardTitle>
                    <div className="flex gap-2 mt-2">
                      <Badge className={getModeColor(selectedResult.mode)}>
                        {getModeIcon(selectedResult.mode)}
                        <span className="ml-1">{selectedResult.mode.replace('_', ' ').toUpperCase()}</span>
                      </Badge>
                      <Badge className={getStatusColor(selectedResult.status)}>
                        {selectedResult.status.toUpperCase()}
                      </Badge>
                    </div>
                  </div>
                  
                  {selectedResult.status === 'completed' && selectedResult.results && (
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadAsMarkdown(selectedResult)}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        MD
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => downloadAsPDF(selectedResult)}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        PDF
                      </Button>
                    </div>
                  )}
                </div>
              </CardHeader>
              
              <CardContent>
                {selectedResult.status === 'completed' && selectedResult.results ? (
                  <Tabs defaultValue="summary" className="w-full">
                    <TabsList className="grid w-full grid-cols-4">
                      <TabsTrigger value="summary">Summary</TabsTrigger>
                      <TabsTrigger value="sources">Sources</TabsTrigger>
                      <TabsTrigger value="insights">Insights</TabsTrigger>
                      {verboseMode && <TabsTrigger value="verbose">Verbose</TabsTrigger>}
                    </TabsList>
                    
                    <TabsContent value="summary" className="mt-4">
                      <ScrollArea className="h-[500px]">
                        <ReactMarkdown className="prose prose-sm max-w-none">
                          {selectedResult.results.summary || ''}
                        </ReactMarkdown>
                      </ScrollArea>
                    </TabsContent>
                    
                    <TabsContent value="sources" className="mt-4">
                      <ScrollArea className="h-[500px]">
                        <div className="space-y-4">
                          {selectedResult.results.sources.map((source, index) => (
                            <Card key={index}>
                              <CardContent className="p-4">
                                <div className="space-y-2">
                                  <div className="flex items-start justify-between">
                                    <h4 className="font-medium">{source.title}</h4>
                                    {source.relevance_score && (
                                      <Badge variant="outline">
                                        {Math.round(source.relevance_score * 100)}% relevant
                                      </Badge>
                                    )}
                                  </div>
                                  <p className="text-sm text-muted-foreground">{source.snippet}</p>
                                  <a 
                                    href={source.url} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="text-sm text-primary hover:underline flex items-center"
                                  >
                                    <ExternalLink className="h-3 w-3 mr-1" />
                                    View Source
                                  </a>
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </ScrollArea>
                    </TabsContent>
                    
                    <TabsContent value="insights" className="mt-4">
                      <ScrollArea className="h-[500px]">
                        <div className="space-y-4">
                          {selectedResult.results.insights && (
                            <div>
                              <h4 className="font-medium mb-3">Key Insights</h4>
                              <ul className="space-y-2">
                                {selectedResult.results.insights.map((insight, index) => (
                                  <li key={index} className="flex items-start">
                                    <span className="w-2 h-2 bg-primary rounded-full mt-2 mr-3 flex-shrink-0"></span>
                                    <span className="text-sm">{insight}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {selectedResult.results.recommendations && (
                            <div>
                              <Separator className="my-4" />
                              <h4 className="font-medium mb-3">Recommendations</h4>
                              <ul className="space-y-2">
                                {selectedResult.results.recommendations.map((rec, index) => (
                                  <li key={index} className="flex items-start">
                                    <span className="w-2 h-2 bg-green-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                                    <span className="text-sm">{rec}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </ScrollArea>
                    </TabsContent>
                    
                    {verboseMode && (
                      <TabsContent value="verbose" className="mt-4">
                        <ScrollArea className="h-[500px]">
                          <div className="space-y-2">
                            <h4 className="font-medium mb-3">Verbose Execution Log</h4>
                            {selectedResult.verbose_log ? (
                              <div className="bg-muted p-4 rounded-md font-mono text-sm">
                                {selectedResult.verbose_log.map((log, index) => (
                                  <div key={index} className="mb-1">
                                    <span className="text-muted-foreground">[{new Date().toLocaleTimeString()}]</span> {log}
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="text-sm text-muted-foreground">No verbose logs available</p>
                            )}
                          </div>
                        </ScrollArea>
                      </TabsContent>
                    )}
                  </Tabs>
                ) : (
                  <div className="text-center py-12">
                    {selectedResult.status === 'pending' && (
                      <div>
                        <Clock className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                        <p className="text-lg font-medium">Research Queued</p>
                        <p className="text-sm text-muted-foreground">Waiting to start...</p>
                      </div>
                    )}
                    
                    {(selectedResult.status === 'searching' || selectedResult.status === 'synthesizing') && (
                      <div>
                        <div className="animate-spin h-12 w-12 border-4 border-primary border-t-transparent rounded-full mx-auto mb-4"></div>
                        <p className="text-lg font-medium">
                          {selectedResult.status === 'searching' ? 'Searching...' : 'Synthesizing Results...'}
                        </p>
                        <Progress value={selectedResult.progress} className="w-64 mx-auto mt-4" />
                        <p className="text-sm text-muted-foreground mt-2">{selectedResult.progress}% complete</p>
                      </div>
                    )}
                    
                    {selectedResult.status === 'error' && (
                      <div>
                        <div className="h-12 w-12 mx-auto mb-4 bg-red-100 rounded-full flex items-center justify-center">
                          <span className="text-red-600 text-xl">!</span>
                        </div>
                        <p className="text-lg font-medium text-red-600">Research Failed</p>
                        <p className="text-sm text-muted-foreground">An error occurred during research</p>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center h-[600px]">
                <div className="text-center text-muted-foreground">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p className="text-lg">Select a research result</p>
                  <p className="text-sm">Choose a research session from the list to view details</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};

export { ResearchResults };