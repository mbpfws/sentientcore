#!/usr/bin/env python3
"""
Test script for Build 2 research functionality
"""

import requests
import json
import time

def test_research_functionality():
    """Test the Build 2 research delegation functionality"""
    base_url = "http://127.0.0.1:8001"
    
    print("Testing Build 2 Research Functionality")
    print("=" * 50)
    
    # Test 1: Basic server connectivity
    print("\n1. Testing server connectivity...")
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        if response.status_code == 200:
            print("✓ Server is accessible")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Failed to connect to server: {e}")
        return False
    
    # Test 2: Knowledge Research Mode
    print("\n2. Testing Knowledge Research mode...")
    try:
        payload = {
            "message": "What are the latest developments in quantum computing?",
            "research_mode": "knowledge"
        }
        response = requests.post(f"{base_url}/test/research", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Knowledge research request processed")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Research mode: {result.get('research_mode')}")
            print(f"  Message count: {result.get('message_count', 0)}")
            
            if result.get('success'):
                response_text = result.get('response', '')
                print(f"  Response preview: {response_text[:200]}...")
                
                # Check for research delegation keywords
                research_keywords = ['research', 'investigate', 'analyze', 'study', 'explore']
                found_keywords = [kw for kw in research_keywords if kw.lower() in response_text.lower()]
                if found_keywords:
                    print(f"  ✓ Research delegation detected (keywords: {found_keywords})")
                else:
                    print("  ⚠ No clear research delegation detected")
            else:
                print(f"  ✗ Request failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"  Response: {response.text}")
    except Exception as e:
        print(f"✗ Knowledge research test failed: {e}")
    
    # Test 3: Deep Research Mode
    print("\n3. Testing Deep Research mode...")
    try:
        payload = {
            "message": "Analyze the impact of AI on software development",
            "research_mode": "deep"
        }
        response = requests.post(f"{base_url}/test/research", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Deep research request processed")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Research mode: {result.get('research_mode')}")
            
            if result.get('success'):
                response_text = result.get('response', '')
                print(f"  Response preview: {response_text[:200]}...")
            else:
                print(f"  ✗ Request failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"✗ Request failed with status {response.status_code}")
    except Exception as e:
        print(f"✗ Deep research test failed: {e}")
    
    # Test 4: Best-in-Class Research Mode
    print("\n4. Testing Best-in-Class Research mode...")
    try:
        payload = {
            "message": "Find the best practices for microservices architecture",
            "research_mode": "best_in_class"
        }
        response = requests.post(f"{base_url}/test/research", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Best-in-class research request processed")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Research mode: {result.get('research_mode')}")
            
            if result.get('success'):
                response_text = result.get('response', '')
                print(f"  Response preview: {response_text[:200]}...")
            else:
                print(f"  ✗ Request failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"✗ Request failed with status {response.status_code}")
    except Exception as e:
        print(f"✗ Best-in-class research test failed: {e}")
    
    # Test 5: Regular mode (no research)
    print("\n5. Testing regular mode (no research)...")
    try:
        payload = {
            "message": "Hello, how are you?"
        }
        response = requests.post(f"{base_url}/test/research", json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Regular request processed")
            print(f"  Success: {result.get('success', False)}")
            print(f"  Research mode: {result.get('research_mode')}")
            
            if result.get('success'):
                response_text = result.get('response', '')
                print(f"  Response preview: {response_text[:200]}...")
            else:
                print(f"  ✗ Request failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"✗ Request failed with status {response.status_code}")
    except Exception as e:
        print(f"✗ Regular mode test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Build 2 Research Functionality Test Complete")
    return True

if __name__ == "__main__":
    test_research_functionality()