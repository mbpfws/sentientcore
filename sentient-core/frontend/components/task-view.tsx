'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface Task {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
}

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
        created_at: '2025-07-02T10:00:00Z',
        updated_at: '2025-07-02T11:30:00Z'
      },
      {
        id: '2',
        title: 'Design system architecture',
        description: 'Create high-level system architecture',
        status: 'in_progress',
        created_at: '2025-07-02T12:00:00Z',
        updated_at: '2025-07-02T14:15:00Z'
      },
      {
        id: '3',
        title: 'Implement backend API',
        description: 'Develop RESTful API endpoints for the application',
        status: 'pending',
        created_at: '2025-07-02T15:00:00Z',
        updated_at: '2025-07-02T15:00:00Z'
      },
      {
        id: '4',
        title: 'Set up CI/CD pipeline',
        description: 'Configure continuous integration and deployment',
        status: 'failed',
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
      case 'failed':
        return 'Failed';
      default:
        return status;
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
                  <CardTitle className="text-xl">{task.title}</CardTitle>
                  <div className="rounded-full px-2 py-1 text-xs font-medium">
                    {getStatusLabel(task.status)}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">{task.description}</p>
                <div className="flex justify-between items-center text-xs text-muted-foreground">
                  <div>Created: {new Date(task.created_at).toLocaleString()}</div>
                  <div>Updated: {new Date(task.updated_at).toLocaleString()}</div>
                </div>
                <div className="flex justify-end mt-4 space-x-2">
                  <Button variant="outline" size="sm">
                    View Details
                  </Button>
                  {task.status !== 'completed' && task.status !== 'failed' && (
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
