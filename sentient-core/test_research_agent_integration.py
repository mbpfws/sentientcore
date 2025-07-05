"""
Integration test for the Research Agent with Groq integration and persistent memory.

This script tests the end-to-end functionality of the research agent, including:
- Agentic tooling with Groq models
- Persistent memory for research results
- Streaming progress updates
- Result consolidation and downloads
"""

import os
import asyncio
import json
from datetime import datetime
from fastapi.testclient import TestClient

# Ensure we're in the right working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import after setting working directory
from core.agents.research_agent import ResearchAgent
from core.services.memory_service import MemoryService
from app import app

# Create a FastAPI test client
client = TestClient(app)

async def test_research_agent_direct():
    """Test the Research Agent directly without going through the API endpoints."""
    print("\n--- Testing Research Agent Directly ---")
    
    # Initialize the research agent
    agent = ResearchAgent()
    
    # Test each research mode
    for mode in ['knowledge', 'deep', 'best_in_class']:
        print(f"\nTesting {mode.upper()} research mode:")
        
        query = f"What are the latest advancements in Groq's LLM technology? (Mode: {mode})"
        print(f"Query: {query}")
        
        # Define a progress callback to monitor streaming
        def progress_callback(progress_data):
            progress_percent = progress_data.get('progress', 0)
            step = progress_data.get('step', '')
            print(f"Progress: {progress_percent:.1f}% - {step}")
        
        # Execute the search with persistence enabled
        result = await agent.execute_search(
            query=query,
            research_mode=mode,
            persist_to_memory=True,
            progress_callback=progress_callback
        )
        
        print(f"Search completed with {len(result.get('sources', []))} sources")
        print(f"Summary length: {len(result.get('summary', ''))}")
        
        # Verify the structure of the result
        assert 'sources' in result, "Result should contain sources"
        assert 'summary' in result, "Result should contain summary"
        assert 'query' in result, "Result should contain the original query"
        assert 'research_mode' in result, "Result should contain the research mode"
        
        # Validate that each source has the expected fields
        for i, source in enumerate(result.get('sources', [])):
            assert 'url' in source, f"Source {i} missing URL"
            assert 'title' in source, f"Source {i} missing title"
            assert 'snippet' in source, f"Source {i} missing snippet"
            
        print(f"{mode.upper()} research test passed!")

async def test_api_integration():
    """Test the Research Agent through the API endpoints."""
    print("\n--- Testing Research Agent API Integration ---")
    
    # 1. Start a new research
    print("\nStarting a new research...")
    response = client.post(
        "/api/research/start",
        json={
            "query": "What are the advantages of using Groq for LLM inference?",
            "mode": "knowledge",
            "persist_to_memory": True
        }
    )
    assert response.status_code == 200, f"Failed to start research: {response.text}"
    
    research_data = response.json()["data"]
    research_id = research_data["id"]
    print(f"Research started with ID: {research_id}")
    
    # 2. Retrieve research results (may not be complete yet)
    print("\nRetrieving results...")
    response = client.get(f"/api/research/results/{research_id}")
    assert response.status_code == 200, f"Failed to get research results: {response.text}"
    
    status_data = response.json()["data"]
    print(f"Current status: {status_data.get('status')}")
    print(f"Progress: {status_data.get('progress')}%")
    
    # 3. Wait and poll for completion (in a real test we'd use SSE streaming)
    print("\nWaiting for research to complete...")
    max_attempts = 15
    attempt = 0
    
    while attempt < max_attempts:
        response = client.get(f"/api/research/results/{research_id}")
        status_data = response.json()["data"]
        
        if status_data.get("status") == "completed":
            print("Research completed!")
            break
        
        print(f"Progress: {status_data.get('progress')}%")
        await asyncio.sleep(10)  # Wait 10 seconds between checks
        attempt += 1
    
    assert status_data.get("status") == "completed", "Research did not complete in time"
    
    # 4. Get markdown download
    print("\nDownloading markdown content...")
    response = client.get(f"/api/research/download/markdown/{research_id}")
    assert response.status_code == 200, "Failed to download markdown content"
    markdown_content = response.text
    print(f"Markdown content length: {len(markdown_content)} chars")
    
    # 5. Attempt PDF download
    print("\nDownloading PDF content...")
    response = client.get(f"/api/research/download/pdf/{research_id}")
    assert response.status_code == 200, "Failed to download PDF content"
    pdf_content = response.content
    print(f"PDF content size: {len(pdf_content)} bytes")
    
    # 6. Validate persistence by checking memory
    print("\nVerifying research persistence in memory...")
    memory_service = MemoryService()
    memory_items = await memory_service.query_items(
        filter_criteria={"type": "research_finding", "metadata.research_id": research_id},
        limit=10
    )
    
    assert len(memory_items) > 0, "Research results not found in persistent memory"
    print(f"Found {len(memory_items)} persistent memory items for this research")
    
    print("\nAPI Integration test passed!")

async def test_batch_research():
    """Test the batch research functionality."""
    print("\n--- Testing Batch Research Functionality ---")
    
    # Start multiple research queries at once
    print("\nStarting batch research...")
    response = client.post(
        "/api/research/batch",
        json={
            "requests": [
                {
                    "query": "Comparison between Groq and NVIDIA for LLM inference",
                    "mode": "deep",
                    "persist_to_memory": True
                },
                {
                    "query": "Best practices for optimizing LLM prompts with Groq",
                    "mode": "best_in_class",
                    "persist_to_memory": True
                }
            ]
        }
    )
    
    assert response.status_code == 200, f"Failed to start batch research: {response.text}"
    
    research_ids = response.json()["data"]["research_ids"]
    print(f"Started {len(research_ids)} research queries")
    
    # Get all current research results
    print("\nRetrieving all research results...")
    response = client.get("/api/research/results")
    assert response.status_code == 200, f"Failed to get all research results: {response.text}"
    
    all_results = response.json()["data"]
    print(f"Found {len(all_results)} total research results")
    
    # Verify our batch research IDs are in the results
    found_ids = [r["id"] for r in all_results]
    for rid in research_ids:
        assert rid in found_ids, f"Research ID {rid} not found in results"
    
    print("Batch research test passed!")

async def main():
    """Run all integration tests."""
    try:
        print("\n============================================")
        print("RESEARCH AGENT INTEGRATION TEST")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("============================================")
        
        await test_research_agent_direct()
        await test_api_integration()
        await test_batch_research()
        
        print("\n============================================")
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("============================================")
    except Exception as e:
        print("\n============================================")
        print(f"❌ TEST FAILED: {str(e)}")
        print("============================================")
        raise

if __name__ == "__main__":
    # Run the async tests
    asyncio.run(main())
