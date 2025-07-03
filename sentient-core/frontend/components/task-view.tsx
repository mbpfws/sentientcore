'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

import { Task } from '@/lib/api/types';

const TaskView = () => {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // In a real implementation, this would fetch tasks from the API
    // const fetchTasks = async () => {
    //   try {
    //     const response = await fetch('/api/workflows/tasks');
    //     const data = await response.json();
    //     setTasks(data);
    //   } catch (error) {
    //     console.error('Error fetching tasks:', error);
    //   } finally {
    //     setIsLoading(false);
    //   }
    // };
    // 
    // fetchTasks();
    
    // For demonstration, use sample data
    const sampleTasks: Task[] = [
      {
        id: '1',
        title: 'Analyze requirements',
        description: 'Analyze project requirements and constraints',
        status: 'completed',
        agent_type: 'research',
        sequence: 1,
        dependencies: [],
        result: 'Requirements analysis completed successfully',
        created_at: '2025-07-02T10:00:00Z',
        updated_at: '2025-07-02T11:30:00Z',
        completed_at: '2025-07-02T11:30:00Z'
      },
      {
        id: '2',
        title: 'Design system architecture',
        description: 'Create high-level system architecture',
        status: 'in_progress',
        agent_type: 'architect',
        sequence: 2,
        dependencies: ['1'],
        created_at: '2025-07-02T12:00:00Z',
        updated_at: '2025-07-02T14:15:00Z'
      },
      {
        id: '3',
        title: 'Implement backend API',
        description: 'Develop RESTful API endpoints for the application',
        status: 'pending',
        agent_type: 'backend_developer',
        sequence: 3,
        dependencies: ['2'],
        created_at: '2025-07-02T15:00:00Z',
        updated_at: '2025-07-02T15:00:00Z'
      },
      {
        id: '4',
        title: 'Set up CI/CD pipeline',
        description: 'Configure continuous integration and deployment',
        status: 'failed',
        agent_type: 'devops',
        sequence: 4,
        dependencies: ['3'],
        created_at: '2025-07-02T09:00:00Z',
        updated_at: '2025-07-02T10:45:00Z'
      }
    ];
    
    setTimeout(() => {
      setTasks(sampleTasks);
      setIsLoading(false);
    }, 500); // Simulate loading delay
  }, []);

  const getStatusClass = (status: string) => {
    switch (status) {
      case 'pending':
        return 'task-pending';
      case 'in_progress':
        return 'task-in-progress';
      case 'completed':
      case 'done':
        return 'task-completed';
      case 'failed':
        return 'task-failed';
      default:
        return '';
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'pending':
        return 'Pending';
      case 'in_progress':
        return 'In Progress';
      case 'completed':
        return 'Completed';
      case 'done':
        return 'Done';
      case 'failed':
        return 'Failed';
      default:
        return status;
    }
  };

  const getAgentEmoji = (agentType: string) => {
    switch (agentType) {
      case 'research':
        return '🔍';
      case 'architect':
        return '🏗️';
      case 'backend_developer':
        return '⚙️';
      case 'frontend_developer':
        return '🎨';
      case 'devops':
        return '🚀';
      default:
        return '🤖';
    }
  };

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold">Tasks</h2>
        <Button variant="outline" size="sm">
          Refresh
        </Button>
      </div>
      
      {isLoading ? (
        <div className="flex justify-center items-center h-64">
          <div className="text-center">
            <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
            <p>Loading tasks...</p>
          </div>
        </div>
      ) : tasks.length === 0 ? (
        <Card>
          <CardContent className="flex justify-center items-center h-64">
            <div className="text-center">
              <p className="mb-2">No tasks found</p>
              <Button variant="outline" size="sm">
                Create New Task
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {tasks.map((task) => (
            <Card key={task.id} className={`task-card ${getStatusClass(task.status)}`}>
              <CardHeader className="pb-2">
                <div className="flex justify-between items-start">
                  <div className="flex items-center gap-2">
                    <span className="text-lg">{getAgentEmoji(task.agent_type)}</span>
                    <div>
                      <CardTitle className="text-xl">
                        {task.sequence}. {task.title}
                      </CardTitle>
                      <p className="text-sm text-muted-foreground capitalize">
                        {task.agent_type.replace('_', ' ')} Agent
                      </p>
                    </div>
                  </div>
                  <div className="rounded-full px-2 py-1 text-xs font-medium">
                    {getStatusLabel(task.status)}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">{task.description}</p>
                
                {task.dependencies && task.dependencies.length > 0 && (
                  <div className="mb-3">
                    <p className="text-xs font-medium text-muted-foreground mb-1">Dependencies:</p>
                    <div className="flex gap-1">
                      {task.dependencies.map((depId) => (
                        <span key={depId} className="text-xs bg-muted px-2 py-1 rounded">
                          Task {depId}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {task.result && (
                  <div className="mb-3 p-2 bg-green-50 border border-green-200 rounded">
                    <p className="text-xs font-medium text-green-800 mb-1">Result:</p>
                    <p className="text-sm text-green-700">{task.result}</p>
                  </div>
                )}
                
                <div className="flex justify-between items-center text-xs text-muted-foreground">
                  <div>Created: {new Date(task.created_at).toLocaleString()}</div>
                  <div>Updated: {task.updated_at ? new Date(task.updated_at).toLocaleString() : 'N/A'}</div>
                </div>
                
                <div className="flex justify-end mt-4 space-x-2">
                  <Button variant="outline" size="sm">
                    View Details
                  </Button>
                  {task.status !== 'completed' && task.status !== 'failed' && task.status !== 'done' && (
                    <Button size="sm">
                      {task.status === 'pending' ? 'Start' : 'Complete'}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default TaskView;
