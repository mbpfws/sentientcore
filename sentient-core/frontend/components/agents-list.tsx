'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

interface Capability {
  name: string;
  description: string;
}

interface Agent {
  id: string;
  name: string;
  description: string;
  icon?: string;
  capabilities: Capability[];
}

const AgentsList = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // In a real implementation, this would fetch agents from the API
    // const fetchAgents = async () => {
    //   try {
    //     const response = await fetch('/api/agents');
    //     const data = await response.json();
    //     setAgents(data);
    //   } catch (error) {
    //     console.error('Error fetching agents:', error);
    //   } finally {
    //     setIsLoading(false);
    //   }
    // };
    // 
    // fetchAgents();
    
    // For demonstration, use sample data
    const sampleAgents: Agent[] = [
      {
        id: '1',
        name: 'Ultra Orchestrator',
        description: 'Manages the overall development workflow and coordinates other agents',
        icon: '🧠',
        capabilities: [
          { 
            name: 'Workflow Coordination',
            description: 'Coordinates multiple agents to complete complex tasks'
          },
          { 
            name: 'Task Prioritization',
            description: 'Prioritizes tasks based on dependencies and importance'
          },
          { 
            name: 'Error Handling',
            description: 'Detects and manages error conditions across the system'
          }
        ]
      },
      {
        id: '2',
        name: 'Research Agent',
        description: 'Performs deep research on technologies, frameworks, and best practices',
        icon: '📚',
        capabilities: [
          { 
            name: 'Knowledge Base Search',
            description: 'Searches through documentation and code examples'
          },
          { 
            name: 'Web Research',
            description: 'Finds relevant information from the internet'
          },
          { 
            name: 'Technical Analysis',
            description: 'Analyzes and compares technical solutions'
          }
        ]
      },
      {
        id: '3',
        name: 'Architect Planner',
        description: 'Designs system architecture and plans implementation details',
        icon: '🏗️',
        capabilities: [
          { 
            name: 'Architecture Design',
            description: 'Creates high-level system architecture designs'
          },
          { 
            name: 'Component Planning',
            description: 'Plans individual components and their interactions'
          },
          { 
            name: 'Technical Specification',
            description: 'Provides detailed technical specifications'
          }
        ]
      },
      {
        id: '4',
        name: 'Coding Agent',
        description: 'Generates code based on specifications and requirements',
        icon: '💻',
        capabilities: [
          { 
            name: 'Code Generation',
            description: 'Generates code for various programming languages'
          },
          { 
            name: 'Code Review',
            description: 'Reviews code for quality and best practices'
          },
          { 
            name: 'Refactoring',
            description: 'Refactors existing code for improved quality'
          }
        ]
      },
    ];
    
    setTimeout(() => {
      setAgents(sampleAgents);
      setIsLoading(false);
    }, 500); // Simulate loading delay
  }, []);

  return (
    <div>
      <h2 className="text-2xl font-bold mb-4">Available Agents</h2>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
            <p>Loading agents...</p>
          </div>
        </div>
      ) : agents.length === 0 ? (
        <Card>
          <CardContent className="flex justify-center items-center h-64">
            <p>No agents available</p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {agents.map((agent) => (
            <Card key={agent.id}>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{agent.icon}</span>
                  <CardTitle>{agent.name}</CardTitle>
                </div>
                <CardDescription>{agent.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <h3 className="font-medium mb-2">Capabilities:</h3>
                <ul className="space-y-2">
                  {agent.capabilities.map((capability, index) => (
                    <li key={index}>
                      <div className="font-medium">{capability.name}</div>
                      <div className="text-sm text-muted-foreground">{capability.description}</div>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentsList;
