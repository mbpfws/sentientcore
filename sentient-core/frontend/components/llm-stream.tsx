"use client";

import { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { getLLMStream, StreamRequest } from '@/lib/api/llm';

export function LLMStream() {
  const [prompt, setPrompt] = useState('Explain the importance of low-latency LLMs in three sentences.');
  const [streamingResponse, setStreamingResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const handleStream = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setIsLoading(true);
    setStreamingResponse('');
    setError(null);

    const request: StreamRequest = {
      model: 'llama3-8b-8192',
      system_prompt: 'You are a helpful assistant.',
      user_prompt: prompt,
      temperature: 0.7,
    };

    const eventSource = getLLMStream(
      request,
      (data) => {
        if (data.content) {
          setStreamingResponse((prev) => prev + data.content);
        }
        if (data.error) {
          setError(data.error);
          console.error('Stream error from backend:', data.error);
        }
      },
      (error) => {
        setError('Failed to connect to the streaming service.');
        console.error('SSE connection error:', error);
        setIsLoading(false);
      },
      () => { // onOpen
        console.log('Stream connection established.');
      },
      () => { // onClose
        setIsLoading(false);
        console.log('Stream connection closed.');
      }
    );
    eventSourceRef.current = eventSource;
  };

  return (
    <div className="w-full max-w-2xl mx-auto p-4">
      <div className="flex flex-col gap-4">
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt here..."
          className="min-h-[100px] text-base"
        />
        <Button onClick={handleStream} disabled={isLoading}>
          {isLoading ? 'Streaming...' : 'Start Stream'}
        </Button>
      </div>

      {error && (
        <div className="mt-6 p-4 bg-destructive/10 border border-destructive/50 text-destructive rounded-md">
          <h3 className="font-bold">Error</h3>
          <p>{error}</p>
        </div>
      )}

      {streamingResponse && (
        <div className="mt-6 p-4 bg-muted/50 rounded-md border">
          <h3 className="font-bold mb-2">Streaming Response:</h3>
          <p className="whitespace-pre-wrap">{streamingResponse}</p>
        </div>
      )}
    </div>
  );
}
