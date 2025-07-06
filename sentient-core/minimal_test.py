#!/usr/bin/env python3
"""
Minimal test script to verify core functionality without server dependencies
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if core modules can be imported"""
    print("Testing core imports...")
    
    try:
        from core.models import AppState, Message, EnhancedTask
        print("✓ Core models imported successfully")
    except Exception as e:
        print(f"✗ Core models import failed: {e}")
        return False
    
    try:
        from core.agents.research_agent import ResearchAgent
        print("✓ Research agent imported successfully")
    except Exception as e:
        print(f"✗ Research agent import failed: {e}")
        return False
    
    try:
        from core.agents.architect_planner_agent import ArchitectPlannerAgent
        print("✓ Architect planner agent imported successfully")
    except Exception as e:
        print(f"✗ Architect planner agent import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic agent functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from core.services.llm_service import EnhancedLLMService
        llm_service = EnhancedLLMService()
        print("✓ LLM service instantiated successfully")
    except Exception as e:
        print(f"✗ LLM service instantiation failed: {e}")
        return False
    
    try:
        from core.agents.research_agent import ResearchAgent
        agent = ResearchAgent(llm_service)
        print("✓ Research agent instantiated successfully")
    except Exception as e:
        print(f"✗ Research agent instantiation failed: {e}")
        return False
    
    try:
        from core.agents.architect_planner_agent import ArchitectPlannerAgent
        planner = ArchitectPlannerAgent(llm_service)
        print("✓ Architect planner agent instantiated successfully")
    except Exception as e:
        print(f"✗ Architect planner agent instantiation failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("\nTesting environment...")
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if env_file.exists():
        print("✓ .env file found")
    else:
        print("⚠ .env file not found")
    
    # Check Python version
    print(f"✓ Python version: {sys.version}")
    
    # Check key directories
    core_dir = project_root / "core"
    if core_dir.exists():
        print("✓ Core directory exists")
    else:
        print("✗ Core directory missing")
        return False
    
    app_dir = project_root / "app"
    if app_dir.exists():
        print("✓ App directory exists")
    else:
        print("✗ App directory missing")
        return False
    
    return True

def main():
    """Run all tests"""
    print("=== Sentient Core Minimal Test Suite ===")
    print(f"Project root: {project_root}")
    
    # Test environment
    env_ok = test_environment()
    
    # Test imports
    imports_ok = test_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    print("\n=== Test Results ===")
    print(f"Environment: {'✓ PASS' if env_ok else '✗ FAIL'}")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Functionality: {'✓ PASS' if functionality_ok else '✗ FAIL'}")
    
    overall_status = env_ok and imports_ok and functionality_ok
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if overall_status else '✗ SOME TESTS FAILED'}")
    
    return overall_status

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)