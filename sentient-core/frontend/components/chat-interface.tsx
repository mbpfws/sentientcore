'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import TextareaAutosize from 'react-textarea-autosize';
import ReactMarkdown from 'react-markdown';
import { ChatService, Message as ApiMessage } from '@/lib/api';
import { useAppContext } from '@/lib/context/app-context';

// Extended from API Message type with UI-specific fields
interface Message extends Partial<ApiMessage> {
  sender: 'user' | 'assistant';
  content: string;
  hasImage?: boolean;
  isLoading?: boolean;
}

const ChatInterface = () => {
  const { activeWorkflow } = useAppContext();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [researchMode, setResearchMode] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom of messages whenever they change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);
  
  // Load chat history when component mounts or workflow changes
  useEffect(() => {
    const loadChatHistory = async () => {
      if (!activeWorkflow) return;
      
      try {
        setIsLoading(true);
        const history = await ChatService.getChatHistory(activeWorkflow);
        
        if (history.messages && history.messages.length > 0) {
          const formattedMessages: Message[] = history.messages.map(msg => ({
            id: msg.id,
            sender: msg.sender,
            content: msg.content,
            created_at: msg.created_at
          }));
          
          setMessages(formattedMessages);
        }
      } catch (error) {
        console.error('Failed to load chat history:', error);
        // Could add UI toast notification here
      } finally {
        setIsLoading(false);
      }
    };
    
    loadChatHistory();
  }, [activeWorkflow]);

  const handleSendMessage = async () => {
    if (!input.trim() || !activeWorkflow) return;
    
    // Create a temporary ID for the user message
    const tempUserId = `user_${Date.now()}`;
    
    // Add user message to UI immediately
    const userMessage: Message = { 
      id: tempUserId,
      sender: 'user', 
      content: input,
      created_at: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    
    // Add a loading message placeholder
    const loadingMessage: Message = {
      id: `loading_${Date.now()}`,
      sender: 'assistant',
      content: '...',
      isLoading: true,
      created_at: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, loadingMessage]);
    
    try {
      // Send the message to the backend
      const response = await ChatService.sendMessage({
        message: input,
        workflow_mode: activeWorkflow,
        research_mode: researchMode || undefined
      });
      
      // Replace the loading message with the actual response
      setMessages(prev => prev.map(msg => 
        msg.isLoading ? {
          id: response.id,
          sender: 'assistant',
          content: response.content,
          created_at: response.created_at
        } : msg
      ));
      
      // Reset research mode after use
      setResearchMode(null);
      
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Replace loading message with error message
      setMessages(prev => prev.map(msg => 
        msg.isLoading ? {
          id: `error_${Date.now()}`,
          sender: 'assistant',
          content: 'Sorry, there was an error processing your request. Please try again.',
          created_at: new Date().toISOString()
        } : msg
      ));
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
          disabled={isLoading}
        >
          📚 Knowledge Research
        </Button>
        <Button 
          variant={researchMode === 'deep' ? 'default' : 'outline'}
          onClick={() => setResearchModeAndNotify('deep')}
          disabled={isLoading}
        >
          🧠 Deep Research
        </Button>
        <Button 
          variant={researchMode === 'best_in_class' ? 'default' : 'outline'}
          onClick={() => setResearchModeAndNotify('best_in_class')}
          disabled={isLoading}
        >
          🏆 Best-in-Class
        </Button>
      </div>

      {/* Chat messages area */}
      <div className="flex-grow overflow-y-auto border rounded-md p-4 mb-4">
        {isLoading && messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-muted-foreground">
            <div className="text-center">
              <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full mx-auto mb-2"></div>
              <p>Loading conversation history...</p>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-muted-foreground">
            <div>
              <h3 className="text-lg font-semibold mb-2">Welcome to Sentient Core</h3>
              <p>Send a message to start building with the Multi-Agent RAG System</p>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div 
              key={message.id || `msg_${Math.random()}`}
              className={`mb-4 ${message.sender === 'user' ? 'flex justify-end' : ''}`}
            >
              <Card 
                className={`p-3 max-w-[80%] ${message.sender === 'user' 
                  ? 'bg-primary text-primary-foreground ml-auto' 
                  : message.isLoading 
                    ? 'bg-muted animate-pulse' 
                    : 'bg-muted'}`}
              >
                {message.isLoading ? (
                  <div className="flex items-center gap-1">
                    <span className="animate-bounce">●</span>
                    <span className="animate-bounce delay-100">●</span>
                    <span className="animate-bounce delay-200">●</span>
                  </div>
                ) : (
                  <ReactMarkdown>
                    {message.content}
                  </ReactMarkdown>
                )}
                {message.created_at && !message.isLoading && (
                  <div className="text-xs mt-2 opacity-70">
                    {new Date(message.created_at).toLocaleTimeString()}
                  </div>
                )}
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
          disabled={isLoading || !input.trim() || !activeWorkflow}
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
