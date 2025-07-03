'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { ThemeToggle } from '@/components/theme-toggle';
import ChatInterface from '@/components/chat-interface';
import TaskView from '@/components/task-view';
import AgentsList from '@/components/agents-list';
import { LLMStream } from '@/components/llm-stream';
import CoreServicesDashboard from '@/components/core-services-dashboard';
import OrchestratorInterface from '@/components/orchestrator-interface';
import { useAppContext } from '@/lib/context/app-context';
import ClientOnly from '@/components/client-only';

// Create a separate client component for the header to avoid hydration issues
function Header() {
  const [mounted, setMounted] = useState(false);
  
  // Only render the header after component has mounted on client
  useEffect(() => {
    setMounted(true);
  }, []);
  
  if (!mounted) {
    return null; // Return null on server-side or first render
  }
  
  return (
    <header className="border-b">
      <div className="container flex items-center justify-between py-4">
        <h1 className="text-2xl font-bold">Sentient Core</h1>
        <div className="flex items-center gap-4">
          <ClientOnly>
            <ThemeToggle />
          </ClientOnly>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const { activeWorkflow, setActiveWorkflow, resetSession } = useAppContext();
  const [activeTab, setActiveTab] = useState('chat');
  const [mounted, setMounted] = useState(false);
  
  // Handle client-side mounting
  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <ClientOnly>
      <div className="flex flex-col min-h-screen">
        {/* Use the client-only Header component */}
        <Header />

        {/* Main content */}
        <main className="flex-1 container py-6">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Sidebar */}
            <div className="lg:col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle>Workflows</CardTitle>
                  <CardDescription>
                    Select the workflow mode to use
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <Button 
                      variant={activeWorkflow === 'intelligent' ? 'default' : 'outline'}
                      className="w-full justify-start" 
                      onClick={() => setActiveWorkflow('intelligent')}
                    >
                      <span className="mr-2">🧠</span> Intelligent RAG
                    </Button>
                    <Button 
                      variant={activeWorkflow === 'multi_agent' ? 'default' : 'outline'}
                      className="w-full justify-start"
                      onClick={() => setActiveWorkflow('multi_agent')}
                    >
                      <span className="mr-2">🤖</span> Multi-Agent RAG
                    </Button>
                    <Button 
                      variant={activeWorkflow === 'legacy' ? 'default' : 'outline'}
                      className="w-full justify-start"
                      onClick={() => setActiveWorkflow('legacy')}
                    >
                      <span className="mr-2">🔄</span> Legacy Workflow
                    </Button>
                  </div>
                </CardContent>
                <CardFooter>
                  <Button 
                    variant="destructive" 
                    className="w-full"
                    onClick={resetSession}
                  >
                    Reset Session
                  </Button>
                </CardFooter>
              </Card>
            </div>

            {/* Main area */}
            <div className="lg:col-span-3">
              <Tabs defaultValue="orchestrator" value={activeTab} onValueChange={setActiveTab}>
                <div className="flex justify-between items-center mb-4">
                  <TabsList>
                    <TabsTrigger value="orchestrator">Orchestrator</TabsTrigger>
                    <TabsTrigger value="core-services">Core Services</TabsTrigger>
                    <TabsTrigger value="chat">Chat</TabsTrigger>
                    <TabsTrigger value="tasks">Tasks</TabsTrigger>
                    <TabsTrigger value="agents">Agents</TabsTrigger>
                    <TabsTrigger value="llm">LLM Stream</TabsTrigger>
                  </TabsList>
                </div>

                <TabsContent value="orchestrator" className="space-y-4">
                  <OrchestratorInterface />
                </TabsContent>

                <TabsContent value="core-services" className="space-y-4">
                  <CoreServicesDashboard />
                </TabsContent>

                <TabsContent value="chat" className="space-y-4">
                  <ChatInterface />
                </TabsContent>

                <TabsContent value="tasks">
                  <TaskView />
                </TabsContent>

                <TabsContent value="agents">
                  <AgentsList />
                </TabsContent>

                <TabsContent value="llm">
                  <LLMStream />
                </TabsContent>
              </Tabs>
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t py-4">
          <div className="container text-center text-sm text-muted-foreground">
            &copy; 2025 Sentient Core - Multi-Agent RAG System
          </div>
        </footer>
      </div>
    </ClientOnly>
  );
}
