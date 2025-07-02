import { ApiResponse, Message } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SendMessageRequest {
  message: string;
  workflow_mode?: string;
  research_mode?: string;
  task_id?: string;
}

export interface ChatHistory {
  messages: Message[];
  workflow_mode: string;
  task_id?: string;
}

export const ChatService = {
  /**
   * Send a message to the chat endpoint
   */
  sendMessage: async (request: SendMessageRequest): Promise<Message> => {
    try {
      const response = await fetch(`${API_URL}/api/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });
      
      if (!response.ok) {
        throw new Error(`Error sending message: ${response.statusText}`);
      }
      
      const data: ApiResponse<Message> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in sendMessage:', error);
      throw error;
    }
  },

  /**
   * Get chat history
   */
  getChatHistory: async (workflow_mode?: string, task_id?: string): Promise<ChatHistory> => {
    try {
      let url = `${API_URL}/api/chat/history`;
      
      // Add query params if provided
      const params = new URLSearchParams();
      if (workflow_mode) params.append('workflow_mode', workflow_mode);
      if (task_id) params.append('task_id', task_id);
      
      if (params.toString()) {
        url += `?${params.toString()}`;
      }
      
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Error fetching chat history: ${response.statusText}`);
      }
      
      const data: ApiResponse<ChatHistory> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in getChatHistory:', error);
      throw error;
    }
  },
  
  /**
   * Clear chat history
   */
  clearChatHistory: async (workflow_mode?: string): Promise<{ success: boolean }> => {
    try {
      let url = `${API_URL}/api/chat/history`;
      
      // Add query params if provided
      if (workflow_mode) {
        url += `?workflow_mode=${encodeURIComponent(workflow_mode)}`;
      }
      
      const response = await fetch(url, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        throw new Error(`Error clearing chat history: ${response.statusText}`);
      }
      
      const data: ApiResponse<{ success: boolean }> = await response.json();
      return data.data;
    } catch (error) {
      console.error('Error in clearChatHistory:', error);
      throw error;
    }
  },
};
