const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export interface StreamRequest {
  model: string;
  system_prompt: string;
  user_prompt: string;
  temperature?: number;
}

// This function now returns the EventSource instance so the caller can close it.
export const getLLMStream = (
  request: StreamRequest,
  onMessage: (data: any) => void,
  onError: (error: Event) => void,
  onOpen: (event: Event) => void,
  onClose: () => void
): EventSource => {
  // Using native EventSource which doesn't support POST directly.
  // We can pass parameters via URL encoding.
  const params = new URLSearchParams({
      model: request.model,
      system_prompt: request.system_prompt,
      user_prompt: request.user_prompt,
      temperature: request.temperature?.toString() || '0.7',
  });

  // This requires the backend to be updated to handle GET requests with query params.
  const url = `${API_URL}/llm/stream?${params.toString()}`;

  const eventSource = new EventSource(url, { withCredentials: true });

  eventSource.onopen = (event) => {
    console.log('SSE connection opened.');
    onOpen(event);
  };

  eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
      console.log('SSE stream finished.');
      eventSource.close();
      onClose();
      return;
    }
    try {
      const parsedData = JSON.parse(event.data);
      onMessage(parsedData);
    } catch (error) {
      console.error('Failed to parse SSE message data:', event.data, error);
      // Since we can't pass a custom error object, we'll just call the generic handler
      onError(new Event('parsingerror')); 
    }
  };

  eventSource.onerror = (error) => {
    console.error('SSE error:', error);
    onError(error);
    eventSource.close();
    onClose();
  };

  return eventSource;
};
