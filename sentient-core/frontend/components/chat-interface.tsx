'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import TextareaAutosize from 'react-textarea-autosize';
import ReactMarkdown from 'react-markdown';

interface Message {
  sender: 'user' | 'assistant';
  content: string;
  hasImage?: boolean;
}

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [researchMode, setResearchMode] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages whenever they change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    
    // Add user message
    const userMessage: Message = { sender: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    try {
      // In a real implementation, this would call the backend API
      // const response = await fetch('/api/chat', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({
      //     message: input,
      //     workflow_mode: 'intelligent',
      //     research_mode: researchMode
      //   })
      // });
      // const data = await response.json();
      
      // Simulate API response for now
      setTimeout(() => {
        const assistantMessage: Message = {
          sender: 'assistant',
          content: `I received your message: "${input}". This is a placeholder response. In the real implementation, this would be the response from the Multi-Agent RAG system.`
        };
        setMessages(prev => [...prev, assistantMessage]);
        setIsLoading(false);
        setResearchMode(null); // Reset research mode after use
      }, 1000);
      
    } catch (error) {
      console.error('Error sending message:', error);
      setIsLoading(false);
    }
  };

  const setResearchModeAndNotify = (mode: string) => {
    setResearchMode(mode);
    // Show a notification or some UI indicator that the mode is selected
  };

  return (
    <div className="flex flex-col h-[calc(100vh-200px)]">
      {/* Research mode buttons */}
      <div className="grid grid-cols-3 gap-2 mb-4">
        <Button 
          variant={researchMode === 'knowledge' ? 'default' : 'outline'}
          onClick={() => setResearchModeAndNotify('knowledge')}
        >
          📚 Knowledge Research
        </Button>
        <Button 
          variant={researchMode === 'deep' ? 'default' : 'outline'}
          onClick={() => setResearchModeAndNotify('deep')}
        >
          🧠 Deep Research
        </Button>
        <Button 
          variant={researchMode === 'best_in_class' ? 'default' : 'outline'}
          onClick={() => setResearchModeAndNotify('best_in_class')}
        >
          🏆 Best-in-Class
        </Button>
      </div>

      {/* Chat messages area */}
      <div className="flex-grow overflow-y-auto border rounded-md p-4 mb-4">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-muted-foreground">
            <div>
              <h3 className="text-lg font-semibold mb-2">Welcome to Sentient Core</h3>
              <p>Send a message to start building with the Multi-Agent RAG System</p>
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div 
              key={index}
              className={`mb-4 ${message.sender === 'user' ? 'flex justify-end' : ''}`}
            >
              <Card className={`p-3 max-w-[80%] ${message.sender === 'user' ? 'bg-primary text-primary-foreground ml-auto' : 'bg-muted'}`}>
                <ReactMarkdown>
                  {message.content}
                </ReactMarkdown>
              </Card>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat input area */}
      <div className="flex items-end border rounded-md p-2">
        <TextareaAutosize
          className="flex-grow resize-none bg-background focus:outline-none px-3 py-2"
          placeholder="What do you want to build today?"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          maxRows={5}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSendMessage();
            }
          }}
        />
        <Button 
          onClick={handleSendMessage} 
          disabled={isLoading || !input.trim()}
          className="ml-2"
        >
          {isLoading ? (
            <span className="animate-pulse">...</span>
          ) : (
            'Send'
          )}
        </Button>
      </div>
    </div>
  );
};

export default ChatInterface;
