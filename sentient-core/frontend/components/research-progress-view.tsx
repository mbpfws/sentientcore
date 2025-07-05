'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import ReactMarkdown from 'react-markdown';
import { Download, ExternalLink, ChevronDown, ChevronRight } from 'lucide-react';
import { ResearchService } from '@/lib/api';

export interface ResearchResult {
  id: string;
  query: string;
  mode: string;
  status: 'in_progress' | 'completed' | 'failed';
  progress: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  source_count?: number;
  logs: Array<{
    timestamp: string;
    message: string;
    type: 'info' | 'progress' | 'error' | 'complete';
    progress?: number;
  }>;
  sources?: Array<{
    url: string;
    title: string;
    snippet: string;
  }>;
  summary?: string;
  markdown_content?: string;
}

interface ResearchProgressViewProps {
  activeResearchId?: string;
  onNewResearch?: () => void;
}

const ResearchProgressView = ({ activeResearchId, onNewResearch }: ResearchProgressViewProps) => {
  const [researchResults, setResearchResults] = useState<ResearchResult[]>([]);
  const [activeTabId, setActiveTabId] = useState<string>(activeResearchId || '');
  const [isLoading, setIsLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({});

  // Fetch all research results on component mount
  useEffect(() => {
    const fetchResearchResults = async () => {
      try {
        setIsLoading(true);
        const results = await ResearchService.getResearchResults();
        setResearchResults(results);
        
        // Set active tab to the provided ID or the most recent result
        if (results.length > 0) {
          if (activeResearchId && results.some(r => r.id === activeResearchId)) {
            setActiveTabId(activeResearchId);
          } else {
            setActiveTabId(results[0].id);
          }
        }
      } catch (error) {
        console.error('Error fetching research results:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchResearchResults();
    
    // Set up polling for in-progress research tasks
    const pollInterval = setInterval(async () => {
      try {
        const hasInProgressTasks = researchResults.some(r => r.status === 'in_progress');
        if (hasInProgressTasks) {
          const results = await ResearchService.getResearchResults();
          setResearchResults(results);
        }
      } catch (error) {
        console.error('Error polling research results:', error);
      }
    }, 5000); // Poll every 5 seconds
    
    return () => clearInterval(pollInterval);
  }, [activeResearchId]);

  // Listen for SSE updates from the server for real-time progress
  useEffect(() => {
    const inProgressTasks = researchResults.filter(r => r.status === 'in_progress');
    
    if (inProgressTasks.length === 0) return;
    
    const eventSources: Record<string, EventSource> = {};
    
    inProgressTasks.forEach(task => {
      const eventSource = new EventSource(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/research/stream/${task.id}`);
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          setResearchResults(prev => prev.map(r => {
            if (r.id === task.id) {
              return {
                ...r,
                progress: data.progress || r.progress,
                status: data.status || r.status,
                logs: [...r.logs, ...(data.new_logs || [])],
                sources: data.sources || r.sources,
                summary: data.summary || r.summary,
                completed_at: data.completed_at || r.completed_at,
                markdown_content: data.markdown_content || r.markdown_content
              };
            }
            return r;
          }));
        } catch (error) {
          console.error('Error processing SSE update:', error);
        }
      };
      
      eventSource.onerror = () => {
        console.error(`SSE connection error for research task ${task.id}`);
        eventSource.close();
      };
      
      eventSources[task.id] = eventSource;
    });
    
    // Clean up event sources when component unmounts or when tasks complete
    return () => {
      Object.values(eventSources).forEach(es => es.close());
    };
  }, [researchResults]);

  const toggleSourceExpansion = (sourceIndex: string) => {
    setExpandedSources(prev => ({
      ...prev,
      [sourceIndex]: !prev[sourceIndex]
    }));
  };

  const downloadMarkdown = (research: ResearchResult) => {
    if (!research.markdown_content) return;
    
    const blob = new Blob([research.markdown_content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research-${research.query.slice(0, 30).replace(/[^a-z0-9]/gi, '-')}-${new Date(research.created_at).toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadPDF = async (research: ResearchResult) => {
    try {
      const pdfBlob = await ResearchService.generatePDF(research.id);
      const url = URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `research-${research.query.slice(0, 30).replace(/[^a-z0-9]/gi, '-')}-${new Date(research.created_at).toISOString().split('T')[0]}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading PDF:', error);
      // Show error notification here
    }
  };

  const formatResearchMode = (mode: string): string => {
    switch (mode) {
      case 'knowledge':
        return 'Knowledge Research';
      case 'deep':
        return 'Deep Research';
      case 'best_in_class':
        return 'Best-in-Class Research';
      default:
        return mode;
    }
  };

  const getModeIcon = (mode: string): string => {
    switch (mode) {
      case 'knowledge':
        return '📚';
      case 'deep':
        return '🧠';
      case 'best_in_class':
        return '🏆';
      default:
        return '🔍';
    }
  };

  if (isLoading && researchResults.length === 0) {
    return (
      <Card className="w-full h-[60vh]">
        <CardContent className="flex justify-center items-center h-full">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
            <p>Loading research results...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (researchResults.length === 0) {
    return (
      <Card className="w-full h-[60vh]">
        <CardContent className="flex flex-col justify-center items-center h-full">
          <p className="mb-4 text-center">No research results found.</p>
          {onNewResearch && (
            <Button onClick={onNewResearch}>Start New Research</Button>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Research Results</h2>
        {onNewResearch && (
          <Button onClick={onNewResearch}>New Research</Button>
        )}
      </div>
      
      <Tabs value={activeTabId} onValueChange={setActiveTabId} className="w-full">
        <TabsList className="mb-4 overflow-x-auto flex w-full">
          {researchResults.map((result) => (
            <TabsTrigger key={result.id} value={result.id} className="flex items-center">
              <span className="mr-2">{getModeIcon(result.mode)}</span>
              <span className="truncate max-w-[150px]">{result.query}</span>
              {result.status === 'in_progress' && (
                <span className="ml-2 animate-pulse">•</span>
              )}
            </TabsTrigger>
          ))}
        </TabsList>

        {researchResults.map((result) => (
          <TabsContent key={result.id} value={result.id} className="mt-0">
            <Card>
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center mb-2">
                      <span className="text-2xl mr-2">{getModeIcon(result.mode)}</span>
                      <CardTitle className="mr-2">{result.query}</CardTitle>
                      <Badge variant={result.status === 'completed' ? 'default' : result.status === 'in_progress' ? 'outline' : 'destructive'}>
                        {result.status === 'in_progress' ? 'In Progress' : result.status === 'completed' ? 'Completed' : 'Failed'}
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {formatResearchMode(result.mode)} • Started {new Date(result.created_at).toLocaleString()}
                      {result.completed_at && ` • Completed ${new Date(result.completed_at).toLocaleString()}`}
                      {result.source_count && ` • ${result.source_count} sources`}
                    </div>
                  </div>
                  
                  {result.status === 'completed' && (
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => downloadMarkdown(result)}>
                        <Download className="h-4 w-4 mr-1" /> MD
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => downloadPDF(result)}>
                        <Download className="h-4 w-4 mr-1" /> PDF
                      </Button>
                    </div>
                  )}
                </div>
                
                {result.status === 'in_progress' && (
                  <div className="mt-4">
                    <Progress value={result.progress} className="h-2" />
                    <div className="text-xs text-right mt-1">{Math.round(result.progress)}%</div>
                  </div>
                )}
              </CardHeader>
              
              <CardContent>
                <Tabs defaultValue="summary" className="w-full">
                  <TabsList className="mb-4">
                    <TabsTrigger value="summary">Summary</TabsTrigger>
                    <TabsTrigger value="sources">Sources</TabsTrigger>
                    <TabsTrigger value="logs">Logs</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="summary" className="mt-0">
                    {result.summary ? (
                      <div className="prose dark:prose-invert max-w-none">
                        <ReactMarkdown>{result.summary}</ReactMarkdown>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        {result.status === 'in_progress' 
                          ? 'Summary will appear once the research is complete...' 
                          : 'No summary available.'}
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="sources" className="mt-0">
                    {result.sources && result.sources.length > 0 ? (
                      <div className="space-y-4">
                        {result.sources.map((source, idx) => (
                          <Collapsible 
                            key={idx}
                            open={expandedSources[`${result.id}-${idx}`]}
                            onOpenChange={() => toggleSourceExpansion(`${result.id}-${idx}`)}
                            className="border rounded-md p-2"
                          >
                            <CollapsibleTrigger className="flex items-center justify-between w-full text-left p-2">
                              <div className="flex-1">
                                <div className="font-medium">{source.title || 'Untitled Source'}</div>
                                <div className="text-sm text-muted-foreground truncate">{source.url}</div>
                              </div>
                              {expandedSources[`${result.id}-${idx}`] ? (
                                <ChevronDown className="h-4 w-4" />
                              ) : (
                                <ChevronRight className="h-4 w-4" />
                              )}
                            </CollapsibleTrigger>
                            <CollapsibleContent className="p-2 pt-4 border-t mt-2">
                              <div className="prose dark:prose-invert max-w-none text-sm">
                                <ReactMarkdown>{source.snippet}</ReactMarkdown>
                              </div>
                              <Button variant="ghost" size="sm" asChild className="mt-2">
                                <a href={source.url} target="_blank" rel="noopener noreferrer">
                                  <ExternalLink className="h-4 w-4 mr-1" /> Open Source
                                </a>
                              </Button>
                            </CollapsibleContent>
                          </Collapsible>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        {result.status === 'in_progress' 
                          ? 'Sources will appear as they are discovered...' 
                          : 'No sources available.'}
                      </div>
                    )}
                  </TabsContent>
                  
                  <TabsContent value="logs" className="mt-0">
                    <div className="space-y-2 max-h-[400px] overflow-y-auto border rounded-md p-4">
                      {result.logs.map((log, idx) => (
                        <div 
                          key={idx} 
                          className={`text-sm py-1 border-b last:border-0 ${
                            log.type === 'error' 
                              ? 'text-destructive' 
                              : log.type === 'complete' 
                                ? 'text-green-500 dark:text-green-400' 
                                : log.type === 'progress' 
                                  ? 'text-blue-500 dark:text-blue-400'
                                  : ''
                          }`}
                        >
                          <span className="text-muted-foreground mr-2 text-xs">
                            {new Date(log.timestamp).toLocaleTimeString()}
                          </span>
                          {log.message}
                          {log.progress !== undefined && (
                            <span className="ml-2 text-xs">{Math.round(log.progress)}%</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
              
              <CardFooter className="flex justify-end pt-2">
                {result.status === 'in_progress' && (
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={() => ResearchService.cancelResearch(result.id)}
                  >
                    Cancel Research
                  </Button>
                )}
              </CardFooter>
            </Card>
          </TabsContent>
        ))}
      </Tabs>
    </div>
  );
};

export default ResearchProgressView;
