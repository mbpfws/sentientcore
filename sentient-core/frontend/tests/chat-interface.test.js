import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatInterface from '../components/chat-interface';
import { ChatService } from '../lib/api';
import { AppContextProvider } from '../lib/context/app-context';

// Mock the API service
jest.mock('../lib/api', () => ({
  ChatService: {
    sendMessage: jest.fn(),
    getChatHistory: jest.fn(),
  },
}));

// Sample test data
const mockChatHistory = {
  messages: [
    {
      id: 'user_1',
      sender: 'user',
      content: 'Hello, can you help me with a question?',
      created_at: '2023-06-15T10:30:00Z',
    },
    {
      id: 'assistant_1',
      sender: 'assistant',
      content: "I'd be happy to help! What's your question?",
      created_at: '2023-06-15T10:30:05Z',
    },
  ],
};

const mockResponse = {
  id: 'assistant_2',
  sender: 'assistant',
  content: 'Here is the answer to your question.',
  created_at: '2023-06-15T10:35:00Z',
};

// Setup wrapper with context
const renderWithContext = (component) => {
  return render(
    <AppContextProvider initialWorkflow="intelligent">
      {component}
    </AppContextProvider>
  );
};

describe('ChatInterface Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
    
    // Setup default mock implementations
    ChatService.getChatHistory.mockResolvedValue(mockChatHistory);
    ChatService.sendMessage.mockResolvedValue(mockResponse);
  });

  test('renders chat interface correctly', async () => {
    renderWithContext(<ChatInterface />);
    
    // Check if chat history is loaded
    await waitFor(() => {
      expect(ChatService.getChatHistory).toHaveBeenCalledWith('intelligent');
    });
    
    // Check if messages are displayed
    expect(screen.getByText('Hello, can you help me with a question?')).toBeInTheDocument();
    expect(screen.getByText("I'd be happy to help! What's your question?")).toBeInTheDocument();
  });

  test('allows sending a new message', async () => {
    renderWithContext(<ChatInterface />);
    
    // Wait for initial chat history to load
    await waitFor(() => {
      expect(ChatService.getChatHistory).toHaveBeenCalled();
    });
    
    // Type a new message
    const inputField = screen.getByPlaceholderText('Type your message here...');
    fireEvent.change(inputField, { target: { value: 'Can you explain RAG systems?' } });
    
    // Send the message
    const sendButton = screen.getByText('Send');
    fireEvent.click(sendButton);
    
    // Check if the API was called with correct parameters
    expect(ChatService.sendMessage).toHaveBeenCalledWith({
      message: 'Can you explain RAG systems?',
      workflow_mode: 'intelligent',
      research_mode: undefined,
    });
    
    // Check if the user message appears
    expect(screen.getByText('Can you explain RAG systems?')).toBeInTheDocument();
    
    // Wait for assistant response to appear
    await waitFor(() => {
      expect(screen.getByText('Here is the answer to your question.')).toBeInTheDocument();
    });
  });

  test('handles API error gracefully', async () => {
    // Mock API error
    ChatService.sendMessage.mockRejectedValue(new Error('API Error'));
    
    renderWithContext(<ChatInterface />);
    
    // Wait for initial chat history to load
    await waitFor(() => {
      expect(ChatService.getChatHistory).toHaveBeenCalled();
    });
    
    // Type and send a message
    const inputField = screen.getByPlaceholderText('Type your message here...');
    fireEvent.change(inputField, { target: { value: 'Will this fail?' } });
    fireEvent.click(screen.getByText('Send'));
    
    // Check if error message appears
    await waitFor(() => {
      expect(screen.getByText('Sorry, there was an error processing your request. Please try again.')).toBeInTheDocument();
    });
  });

  test('toggles research mode correctly', async () => {
    renderWithContext(<ChatInterface />);
    
    // Click on a research mode button
    const deepResearchButton = screen.getByText('🧠 Deep Research');
    fireEvent.click(deepResearchButton);
    
    // Send a message
    const inputField = screen.getByPlaceholderText('Type your message here...');
    fireEvent.change(inputField, { target: { value: 'Research this topic' } });
    fireEvent.click(screen.getByText('Send'));
    
    // Check if the API was called with the research mode
    expect(ChatService.sendMessage).toHaveBeenCalledWith({
      message: 'Research this topic',
      workflow_mode: 'intelligent',
      research_mode: 'deep',
    });
    
    // Research mode should reset after sending
    await waitFor(() => {
      // Send another message
      fireEvent.change(inputField, { target: { value: 'Another question' } });
      fireEvent.click(screen.getByText('Send'));
      
      // Check that research mode is no longer included
      expect(ChatService.sendMessage).toHaveBeenCalledWith({
        message: 'Another question',
        workflow_mode: 'intelligent',
        research_mode: undefined,
      });
    });
  });
});
