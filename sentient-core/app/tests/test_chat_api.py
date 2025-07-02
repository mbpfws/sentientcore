import pytest
from fastapi.testclient import TestClient
from api.app import app
from unittest.mock import patch, MagicMock

# Initialize test client
client = TestClient(app)

# Sample test data
sample_message = {
    "message": "Hello, can you help me with a question?",
    "workflow_mode": "intelligent",
    "research_mode": None
}

sample_chat_history = {
    "messages": [
        {
            "id": "msg_1",
            "sender": "user",
            "content": "Hello, can you help me with a question?",
            "created_at": "2023-06-15T10:30:00Z"
        },
        {
            "id": "msg_2",
            "sender": "assistant",
            "content": "I'd be happy to help! What's your question?",
            "created_at": "2023-06-15T10:30:05Z"
        }
    ]
}

@pytest.fixture
def mock_chat_processor():
    """Create a mock for the chat processor functions."""
    with patch("api.routers.chat.process_message") as mock_processor:
        mock_processor.return_value = {
            "id": "resp_1",
            "content": "I'd be happy to help! What's your question?",
            "created_at": "2023-06-15T10:30:05Z"
        }
        yield mock_processor

@pytest.fixture
def mock_chat_history():
    """Create a mock for the chat history retrieval."""
    with patch("api.routers.chat.get_chat_history") as mock_history:
        mock_history.return_value = sample_chat_history
        yield mock_history

def test_send_message(mock_chat_processor):
    """Test sending a message to the chat endpoint."""
    response = client.post("/chat/message", json=sample_message)
    
    assert response.status_code == 200
    assert "id" in response.json()
    assert "content" in response.json()
    assert "created_at" in response.json()
    
    # Verify chat processor was called with correct arguments
    mock_chat_processor.assert_called_once()
    args, kwargs = mock_chat_processor.call_args
    assert kwargs["message"] == sample_message["message"]
    assert kwargs["workflow_mode"] == sample_message["workflow_mode"]

def test_get_chat_history(mock_chat_history):
    """Test retrieving chat history."""
    response = client.get("/chat/history/intelligent")
    
    assert response.status_code == 200
    assert "messages" in response.json()
    assert len(response.json()["messages"]) == 2
    
    # Verify message structure
    messages = response.json()["messages"]
    assert messages[0]["sender"] == "user"
    assert messages[1]["sender"] == "assistant"
    
    # Verify the history retrieval was called with correct workflow
    mock_chat_history.assert_called_once_with("intelligent")

def test_invalid_workflow_mode():
    """Test sending a message with an invalid workflow mode."""
    invalid_message = sample_message.copy()
    invalid_message["workflow_mode"] = "nonexistent_mode"
    
    response = client.post("/chat/message", json=invalid_message)
    
    assert response.status_code == 422  # Validation error

def test_empty_message():
    """Test sending an empty message."""
    empty_message = sample_message.copy()
    empty_message["message"] = ""
    
    response = client.post("/chat/message", json=empty_message)
    
    assert response.status_code == 422  # Validation error
