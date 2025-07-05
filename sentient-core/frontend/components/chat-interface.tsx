'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import TextareaAutosize from 'react-textarea-autosize';
import ReactMarkdown from 'react-markdown';
import { ChatService, Message as ApiMessage } from '@/lib/api';
import { useAppContext } from '@/lib/context/app-context';
import { Paperclip, X, Image as ImageIcon } from 'lucide-react';

// Extended from API Message type with UI-specific fields
interface Message extends Partial<ApiMessage> {
  sender: 'user' | 'assistant';
  content: string;
  hasImage?: boolean;
  imageUrl?: string;
  isLoading?: boolean;
}

const ChatInterface = () => {
  const { activeWorkflow } = useAppContext();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [researchMode, setResearchMode] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
    if ((!input.trim() && !selectedImage) || !activeWorkflow) return;
    
    // Create a temporary ID for the user message
    const tempUserId = `user_${Date.now()}`;
    
    // Add user message to UI immediately
    const userMessage: Message = { 
      id: tempUserId,
      sender: 'user', 
      content: input || (selectedImage ? '[Image attached]' : ''),
      hasImage: !!selectedImage,
      imageUrl: imagePreview || undefined,
      created_at: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    
    // Store current input and image for API call
    const currentInput = input;
    const currentImage = selectedImage;
    
    // Clear input and image
    setInput('');
    handleImageRemove();
    
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
      // Prepare image data if present
      let imageData: Uint8Array | undefined;
      if (currentImage) {
        imageData = await fileToBytes(currentImage);
      }
      
      // Send the message to the backend
      const response = await ChatService.sendMessage({
        message: currentInput,
        workflow_mode: activeWorkflow,
        research_mode: researchMode || undefined,
        image_data: imageData
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

  // Handle image file selection
  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      
      // Create preview URL
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Remove selected image
  const handleImageRemove = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Convert file to base64 bytes
  const fileToBytes = (file: File): Promise<Uint8Array> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          resolve(new Uint8Array(reader.result));
        } else {
          reject(new Error('Failed to read file as ArrayBuffer'));
        }
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
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
                  <div>
                    {message.hasImage && message.imageUrl && (
                      <div className="mb-2">
                        <img 
                          src={message.imageUrl} 
                          alt="Attached image" 
                          className="max-w-xs max-h-48 rounded-md object-cover"
                        />
                      </div>
                    )}
                    <ReactMarkdown>
                      {message.content}
                    </ReactMarkdown>
                  </div>
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

      {/* Image preview */}
      {imagePreview && (
        <div className="mb-4 p-3 border rounded-md bg-muted/50">
          <div className="flex items-start gap-3">
            <div className="relative">
              <img 
                src={imagePreview} 
                alt="Preview" 
                className="max-w-32 max-h-32 rounded-md object-cover"
              />
              <Button
                size="sm"
                variant="destructive"
                className="absolute -top-2 -right-2 h-6 w-6 rounded-full p-0"
                onClick={handleImageRemove}
              >
                <X className="h-3 w-3" />
              </Button>
            </div>
            <div className="flex-1">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <ImageIcon className="h-4 w-4" />
                <span>{selectedImage?.name}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Image will be sent with your message
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Chat input area */}
      <div className="flex items-end border rounded-md p-2">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleImageSelect}
          className="hidden"
        />
        <Button
          variant="ghost"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          disabled={isLoading}
          className="mr-2 p-2"
          title="Attach image"
        >
          <Paperclip className="h-4 w-4" />
        </Button>
        <TextareaAutosize
          className="flex-grow resize-none bg-background focus:outline-none px-3 py-2"
          placeholder={selectedImage ? "Add a message (optional)" : "What do you want to build today?"}
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
          disabled={isLoading || (!input.trim() && !selectedImage) || !activeWorkflow}
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
