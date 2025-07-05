import { ApiResponse, Message } from './types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface SendMessageRequest {
  message: string;
  workflow_mode?: string;
  research_mode?: string;
  task_id?: string;
  image_data?: Uint8Array;
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
      let body: string | FormData;
      let headers: Record<string, string> = {};
      let endpoint = `${API_URL}/api/chat/message`;
      
      if (request.image_data) {
        // Use FormData for requests with images
        const formData = new FormData();
        formData.append('message', request.message);
        if (request.workflow_mode) formData.append('workflow_mode', request.workflow_mode);
        if (request.research_mode) formData.append('research_mode', request.research_mode);
        if (request.task_id) formData.append('task_id', request.task_id);
        
        // Convert Uint8Array to Blob for FormData
        const imageBlob = new Blob([request.image_data], { type: 'image/jpeg' });
        formData.append('image_data', imageBlob, 'image.jpg');
        
        body = formData;
        // Don't set Content-Type header for FormData, let browser set it with boundary
      } else {
        // Use JSON endpoint for text-only requests
        endpoint = `${API_URL}/api/chat/message/json`;
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify(request);
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers,
        body,
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
