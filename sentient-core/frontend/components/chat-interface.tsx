'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import TextareaAutosize from 'react-textarea-autosize';
import ReactMarkdown from 'react-markdown';
import { ChatService, Message as ApiMessage } from '@/lib/api';
import { useAppContext } from '@/lib/context/app-context';
import { Paperclip, X, Image as ImageIcon, History, Settings, FileText, Zap, Search, BookOpen } from 'lucide-react';
import { ResearchResults } from './research-results';
import { VerboseFeedback } from './verbose-feedback';
import { researchService } from '@/lib/api/research-service';
import ImplementationWorkflow from './implementation/implementation-workflow';
import ImplementationProgress from './implementation/implementation-progress';

// Extended from API Message type with UI-specific fields
interface Message {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp?: string;
  created_at?: string;
  images?: {
    file: File;
    preview: string;
  }[];
  researchMode?: string;
  isError?: boolean;
  isLoading?: boolean;
  hasImage?: boolean;
  imageUrl?: string;
  researchData?: any;
}

interface VerboseStep {
  id: string;
  type: 'search' | 'synthesis' | 'error' | 'completion';
  title: string;
  description: string;
  status: 'running' | 'completed' | 'error';
  timestamp: string;
  duration?: number;
  details?: {
    input?: any;
    output?: any;
    sources?: string[];
    tokens_used?: number;
    model_used?: string;
    progress?: number;
  };
  substeps?: VerboseStep[];
}

const ChatInterface = () => {
  const { activeWorkflow } = useAppContext();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImages, setSelectedImages] = useState<File[]>([]);
  const [researchMode, setResearchMode] = useState<'knowledge' | 'deep' | 'best_in_class'>('knowledge');
  const [activeTab, setActiveTab] = useState<'chat' | 'research' | 'verbose'>('chat');
  const [verboseSteps, setVerboseSteps] = useState<VerboseStep[]>([]);
  const [showVerbose, setShowVerbose] = useState<boolean>(false);
  
  // Implementation workflow state
  const [implementationMode, setImplementationMode] = useState<'workflow' | 'progress' | null>(null);
  const [activeImplementationId, setActiveImplementationId] = useState<string | null>(null);
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
    setVerboseSteps([]);
    
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
      // Check if this is a research query
      const isResearchQuery = researchMode && (currentInput.toLowerCase().includes('research') || 
        currentInput.toLowerCase().includes('search') || 
        currentInput.toLowerCase().includes('find') ||
        currentInput.toLowerCase().includes('analyze'));

      if (isResearchQuery && activeWorkflow) {
         // Start research with verbose feedback
         setShowVerbose(true);
         setActiveTab('verbose');
         
         // Add initial verbose step
         const initialStep: VerboseStep = {
           id: 'init-' + Date.now(),
           type: 'search',
           title: 'Initializing Research',
           description: `Starting ${researchMode} research for: "${currentInput}"`,
           status: 'running',
           timestamp: new Date().toISOString()
         };
         setVerboseSteps([initialStep]);

         try {
           // Start research
           const researchResult = await researchService.startResearch({
             query: currentInput,
             mode: researchMode,
             workflow_id: activeWorkflow
           });

           // Subscribe to research updates
           researchService.subscribeToUpdates(researchResult.data.id, (updatedResult) => {
             // Update verbose steps based on research progress
             const progressStep: VerboseStep = {
               id: 'progress-' + Date.now(),
               type: updatedResult.status === 'searching' ? 'search' : 'synthesis',
               title: updatedResult.status === 'searching' ? 'Searching Sources' : 'Synthesizing Results',
               description: `Research progress: ${updatedResult.progress || 0}%`,
               status: updatedResult.status === 'completed' ? 'completed' : 'running',
               timestamp: new Date().toISOString(),
               details: {
                 progress: updatedResult.progress,
                 model_used: 'compound-beta'
               }
             };
             
             setVerboseSteps(prev => {
               const filtered = prev.filter(step => !step.id.startsWith('progress-'));
               return [...filtered, progressStep];
             });

             // If completed, add final message
             if (updatedResult.status === 'completed' && updatedResult.results) {
               setMessages(prev => prev.map(msg => 
                 msg.isLoading ? {
                   id: updatedResult.id,
                   sender: 'assistant' as const,
                   content: updatedResult.results.summary,
                   created_at: new Date().toISOString(),
                   researchData: updatedResult
                 } : msg
               ));
               setActiveTab('research');
             }
           });
         } catch (error) {
           console.error('Research failed:', error);
           // Add error step
           const errorStep: VerboseStep = {
             id: 'error-' + Date.now(),
             type: 'error',
             title: 'Research Failed',
             description: 'Failed to start research process',
             status: 'error',
             timestamp: new Date().toISOString()
           };
           setVerboseSteps(prev => [...prev, errorStep]);
         }
       } else {
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
      }
      
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
      
      // Add error to verbose steps if active
       if (showVerbose) {
         const errorStep: VerboseStep = {
           id: 'error-' + Date.now(),
           type: 'error',
           title: 'Error Occurred',
           description: 'An error occurred during processing',
           status: 'error',
           timestamp: new Date().toISOString()
         };
         setVerboseSteps(prev => [...prev, errorStep]);
       }
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
      {/* Mode selection buttons */}
      <div className="space-y-2 mb-4">
        {/* Research mode buttons */}
        <div className="grid grid-cols-3 gap-2">
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
        
        {/* Implementation mode buttons */}
        <div className="grid grid-cols-2 gap-2">
          <Button 
            variant={implementationMode === 'workflow' ? 'default' : 'outline'}
            onClick={() => {
              setImplementationMode('workflow');
              setActiveTab('implementation');
            }}
            disabled={isLoading}
          >
            🔧 Start Implementation
          </Button>
          <Button 
            variant={implementationMode === 'progress' ? 'default' : 'outline'}
            onClick={() => {
              setImplementationMode('progress');
              setActiveTab('implementation');
            }}
            disabled={isLoading || !activeImplementationId}
          >
            📊 View Progress
          </Button>
        </div>
      </div>

      {/* Tab navigation and content areas */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="grid w-full grid-cols-4 mb-4">
          <TabsTrigger value="chat">Chat</TabsTrigger>
          <TabsTrigger value="research">Research Results</TabsTrigger>
          <TabsTrigger value="implementation">Implementation</TabsTrigger>
          <TabsTrigger value="verbose">Verbose Feedback</TabsTrigger>
        </TabsList>
        
        <TabsContent value="chat" className="flex-grow overflow-y-auto border rounded-md p-4 mb-4">
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
        </TabsContent>
        
        <TabsContent value="research" className="flex-1 overflow-hidden">
          <ResearchResults />
        </TabsContent>
        
        <TabsContent value="implementation" className="flex-1 overflow-hidden">
          {implementationMode === 'workflow' ? (
            <ImplementationWorkflow 
              onImplementationStart={(implementationId) => {
                setActiveImplementationId(implementationId);
                setImplementationMode('progress');
              }}
            />
          ) : implementationMode === 'progress' && activeImplementationId ? (
            <ImplementationProgress 
              implementationId={activeImplementationId}
              onBack={() => setImplementationMode('workflow')}
            />
          ) : (
            <div className="flex h-full items-center justify-center text-center text-muted-foreground">
              <div>
                <h3 className="text-lg font-semibold mb-2">Implementation Workflow</h3>
                <p>Click "Start Implementation" to begin a new feature implementation workflow</p>
              </div>
            </div>
          )}
        </TabsContent>
        
        <TabsContent value="verbose" className="flex-1 overflow-hidden">
          <VerboseFeedback steps={verboseSteps} />
        </TabsContent>
      </Tabs>

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
