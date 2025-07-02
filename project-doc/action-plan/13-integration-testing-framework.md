# 13 - Integration Testing Framework

## Overview

The Integration Testing Framework provides comprehensive end-to-end testing capabilities for the multi-agent system. It ensures seamless integration between all components, validates workflow execution, and maintains system reliability through automated testing scenarios. This framework supports both automated testing and human-in-the-loop validation processes.

## Current State Analysis

### Testing Requirements
- End-to-end workflow testing
- Agent interaction validation
- Frontend-backend integration testing
- Performance and load testing
- User acceptance testing scenarios
- Regression testing automation

### Integration Points
- Multi-agent workflow orchestration
- Real-time communication between agents
- Frontend state synchronization
- External service integrations
- Database consistency validation

## Implementation Tasks

### Task 13.1: Core Testing Framework

**File**: `tests/integration/framework.py`

**Integration Test Framework**:
```python
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import pytest
from dataclasses import dataclass

@dataclass
class TestScenario:
    name: str
    description: str
    steps: List[Dict[str, Any]]
    expected_outcomes: List[Dict[str, Any]]
    timeout: int = 300
    prerequisites: List[str] = None

class IntegrationTestFramework:
    def __init__(self):
        self.test_scenarios = {}
        self.test_results = []
        self.active_sessions = {}
        self.mock_services = {}
        
    async def register_test_scenario(self, scenario: TestScenario):
        """Register a new test scenario"""
        self.test_scenarios[scenario.name] = scenario
        
    async def execute_test_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Execute a complete test scenario"""
        scenario = self.test_scenarios.get(scenario_name)
        if not scenario:
            raise ValueError(f"Test scenario '{scenario_name}' not found")
            
        test_session = {
            'scenario_name': scenario_name,
            'start_time': datetime.utcnow(),
            'status': 'running',
            'steps_completed': [],
            'errors': [],
            'results': {}
        }
        
        session_id = f"test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.active_sessions[session_id] = test_session
        
        try:
            # Execute test steps
            for step_index, step in enumerate(scenario.steps):
                step_result = await self._execute_test_step(session_id, step_index, step)
                test_session['steps_completed'].append(step_result)
                
                if not step_result.get('success', False):
                    test_session['status'] = 'failed'
                    break
            
            # Validate outcomes
            if test_session['status'] != 'failed':
                validation_result = await self._validate_outcomes(session_id, scenario.expected_outcomes)
                test_session['validation'] = validation_result
                test_session['status'] = 'passed' if validation_result.get('all_passed', False) else 'failed'
            
            test_session['end_time'] = datetime.utcnow()
            test_session['duration'] = (test_session['end_time'] - test_session['start_time']).total_seconds()
            
            # Store results
            self.test_results.append(test_session)
            
            return test_session
            
        except Exception as e:
            test_session['status'] = 'error'
            test_session['error'] = str(e)
            test_session['end_time'] = datetime.utcnow()
            return test_session
    
    async def _execute_test_step(self, session_id: str, step_index: int, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual test step"""
        step_type = step.get('type')
        step_params = step.get('params', {})
        
        step_result = {
            'step_index': step_index,
            'step_type': step_type,
            'start_time': datetime.utcnow(),
            'success': False
        }
        
        try:
            if step_type == 'agent_task':
                result = await self._test_agent_task(step_params)
            elif step_type == 'workflow_execution':
                result = await self._test_workflow_execution(step_params)
            elif step_type == 'frontend_interaction':
                result = await self._test_frontend_interaction(step_params)
            elif step_type == 'api_call':
                result = await self._test_api_call(step_params)
            elif step_type == 'database_validation':
                result = await self._test_database_validation(step_params)
            elif step_type == 'performance_check':
                result = await self._test_performance_check(step_params)
            else:
                raise ValueError(f"Unknown step type: {step_type}")
            
            step_result['result'] = result
            step_result['success'] = result.get('success', False)
            
        except Exception as e:
            step_result['error'] = str(e)
            step_result['success'] = False
        
        step_result['end_time'] = datetime.utcnow()
        step_result['duration'] = (step_result['end_time'] - step_result['start_time']).total_seconds()
        
        return step_result
```

### Task 13.2: Agent Integration Testing

**File**: `tests/integration/test_agent_integration.py`

**Agent Integration Tests**:
```python
import pytest
from tests.integration.framework import IntegrationTestFramework, TestScenario

class TestAgentIntegration:
    def __init__(self):
        self.framework = IntegrationTestFramework()
        self.setup_agent_test_scenarios()
    
    def setup_agent_test_scenarios(self):
        """Setup agent integration test scenarios"""
        
        # Multi-agent collaboration scenario
        collaboration_scenario = TestScenario(
            name="multi_agent_collaboration",
            description="Test collaboration between Research, Architect, and Developer agents",
            steps=[
                {
                    'type': 'agent_task',
                    'params': {
                        'agent_type': 'research',
                        'task': {
                            'type': 'research_request',
                            'query': 'React component best practices',
                            'scope': 'comprehensive'
                        }
                    }
                },
                {
                    'type': 'agent_task',
                    'params': {
                        'agent_type': 'architect_planner',
                        'task': {
                            'type': 'architecture_planning',
                            'project_name': 'Test Component Library',
                            'requirements': ['reusable components', 'TypeScript support']
                        }
                    }
                },
                {
                    'type': 'agent_task',
                    'params': {
                        'agent_type': 'frontend_developer',
                        'task': {
                            'type': 'component_development',
                            'component_spec': {
                                'name': 'TestButton',
                                'description': 'Reusable button component'
                            }
                        }
                    }
                }
            ],
            expected_outcomes=[
                {
                    'type': 'research_completion',
                    'criteria': {
                        'confidence_score': 0.8,
                        'knowledge_nodes_count': 5
                    }
                },
                {
                    'type': 'architecture_completion',
                    'criteria': {
                        'architecture_type': 'defined',
                        'tech_stack': 'recommended'
                    }
                },
                {
                    'type': 'component_completion',
                    'criteria': {
                        'component_code': 'generated',
                        'tests_included': True
                    }
                }
            ],
            timeout=600
        )
        
        self.framework.register_test_scenario(collaboration_scenario)
        
        # Workflow orchestration scenario
        orchestration_scenario = TestScenario(
            name="workflow_orchestration",
            description="Test Ultra Orchestrator managing complete development workflow",
            steps=[
                {
                    'type': 'workflow_execution',
                    'params': {
                        'workflow_type': 'development_request',
                        'request': {
                            'description': 'Build a simple todo application',
                            'requirements': ['CRUD operations', 'responsive design'],
                            'tech_stack': {'frontend': 'React', 'backend': 'FastAPI'}
                        }
                    }
                }
            ],
            expected_outcomes=[
                {
                    'type': 'workflow_completion',
                    'criteria': {
                        'status': 'completed',
                        'all_phases_executed': True,
                        'artifacts_generated': True
                    }
                }
            ],
            timeout=1800
        )
        
        self.framework.register_test_scenario(orchestration_scenario)
    
    async def test_multi_agent_collaboration(self):
        """Test multi-agent collaboration scenario"""
        result = await self.framework.execute_test_scenario("multi_agent_collaboration")
        assert result['status'] == 'passed', f"Test failed: {result.get('error', 'Unknown error')}"
    
    async def test_workflow_orchestration(self):
        """Test workflow orchestration scenario"""
        result = await self.framework.execute_test_scenario("workflow_orchestration")
        assert result['status'] == 'passed', f"Test failed: {result.get('error', 'Unknown error')}"
```

### Task 13.3: Frontend-Backend Integration Testing

**File**: `tests/integration/test_frontend_backend.py`

**Frontend-Backend Integration Tests**:
```python
import pytest
import asyncio
from playwright.async_api import async_playwright
from tests.integration.framework import IntegrationTestFramework, TestScenario

class TestFrontendBackendIntegration:
    def __init__(self):
        self.framework = IntegrationTestFramework()
        self.setup_frontend_backend_scenarios()
    
    def setup_frontend_backend_scenarios(self):
        """Setup frontend-backend integration scenarios"""
        
        # Real-time agent monitoring scenario
        monitoring_scenario = TestScenario(
            name="realtime_agent_monitoring",
            description="Test real-time agent status updates in frontend",
            steps=[
                {
                    'type': 'frontend_interaction',
                    'params': {
                        'action': 'navigate_to_dashboard',
                        'url': 'http://localhost:3000'
                    }
                },
                {
                    'type': 'agent_task',
                    'params': {
                        'agent_type': 'research',
                        'task': {
                            'type': 'research_request',
                            'query': 'Test query for monitoring'
                        }
                    }
                },
                {
                    'type': 'frontend_interaction',
                    'params': {
                        'action': 'verify_agent_status_update',
                        'expected_status': 'working'
                    }
                }
            ],
            expected_outcomes=[
                {
                    'type': 'frontend_update',
                    'criteria': {
                        'agent_status_displayed': True,
                        'real_time_updates': True
                    }
                }
            ]
        )
        
        self.framework.register_test_scenario(monitoring_scenario)
        
        # Chat interface scenario
        chat_scenario = TestScenario(
            name="chat_interface_interaction",
            description="Test chat interface with agent responses",
            steps=[
                {
                    'type': 'frontend_interaction',
                    'params': {
                        'action': 'open_chat_interface'
                    }
                },
                {
                    'type': 'frontend_interaction',
                    'params': {
                        'action': 'send_message',
                        'message': 'Create a simple React component'
                    }
                },
                {
                    'type': 'frontend_interaction',
                    'params': {
                        'action': 'wait_for_response',
                        'timeout': 30
                    }
                }
            ],
            expected_outcomes=[
                {
                    'type': 'chat_response',
                    'criteria': {
                        'response_received': True,
                        'agent_workflow_initiated': True
                    }
                }
            ]
        )
        
        self.framework.register_test_scenario(chat_scenario)
    
    async def _test_frontend_interaction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute frontend interaction test"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                action = params.get('action')
                
                if action == 'navigate_to_dashboard':
                    await page.goto(params.get('url'))
                    await page.wait_for_load_state('networkidle')
                    return {'success': True, 'action': 'navigation_completed'}
                
                elif action == 'verify_agent_status_update':
                    # Wait for agent status element to appear
                    status_selector = '[data-testid="agent-status"]'
                    await page.wait_for_selector(status_selector, timeout=10000)
                    
                    status_text = await page.text_content(status_selector)
                    expected_status = params.get('expected_status')
                    
                    return {
                        'success': expected_status.lower() in status_text.lower(),
                        'actual_status': status_text,
                        'expected_status': expected_status
                    }
                
                elif action == 'open_chat_interface':
                    chat_button = '[data-testid="chat-button"]'
                    await page.click(chat_button)
                    await page.wait_for_selector('[data-testid="chat-input"]')
                    return {'success': True, 'action': 'chat_opened'}
                
                elif action == 'send_message':
                    message = params.get('message')
                    await page.fill('[data-testid="chat-input"]', message)
                    await page.click('[data-testid="send-button"]')
                    return {'success': True, 'message_sent': message}
                
                elif action == 'wait_for_response':
                    timeout = params.get('timeout', 30) * 1000
                    await page.wait_for_selector('[data-testid="agent-response"]', timeout=timeout)
                    response_text = await page.text_content('[data-testid="agent-response"]')
                    return {'success': True, 'response': response_text}
                
                else:
                    return {'success': False, 'error': f'Unknown action: {action}'}
                    
            except Exception as e:
                return {'success': False, 'error': str(e)}
            
            finally:
                await browser.close()
```

### Task 13.4: Performance Integration Testing

**File**: `tests/integration/test_performance.py`

**Performance Integration Tests**:
```python
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from tests.integration.framework import IntegrationTestFramework, TestScenario

class TestPerformanceIntegration:
    def __init__(self):
        self.framework = IntegrationTestFramework()
        self.setup_performance_scenarios()
    
    def setup_performance_scenarios(self):
        """Setup performance testing scenarios"""
        
        # Concurrent agent execution scenario
        concurrent_scenario = TestScenario(
            name="concurrent_agent_execution",
            description="Test system performance with multiple concurrent agents",
            steps=[
                {
                    'type': 'performance_check',
                    'params': {
                        'test_type': 'concurrent_agents',
                        'agent_count': 5,
                        'task_complexity': 'medium',
                        'duration': 60
                    }
                }
            ],
            expected_outcomes=[
                {
                    'type': 'performance_metrics',
                    'criteria': {
                        'average_response_time': 5.0,  # seconds
                        'memory_usage_mb': 500,
                        'cpu_usage_percent': 80
                    }
                }
            ]
        )
        
        self.framework.register_test_scenario(concurrent_scenario)
        
        # Load testing scenario
        load_scenario = TestScenario(
            name="system_load_testing",
            description="Test system under high load conditions",
            steps=[
                {
                    'type': 'performance_check',
                    'params': {
                        'test_type': 'load_testing',
                        'concurrent_users': 10,
                        'requests_per_second': 50,
                        'duration': 120
                    }
                }
            ],
            expected_outcomes=[
                {
                    'type': 'load_metrics',
                    'criteria': {
                        'success_rate': 0.95,
                        'average_response_time': 3.0,
                        'error_rate': 0.05
                    }
                }
            ]
        )
        
        self.framework.register_test_scenario(load_scenario)
    
    async def _test_performance_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance check"""
        test_type = params.get('test_type')
        
        if test_type == 'concurrent_agents':
            return await self._test_concurrent_agents(params)
        elif test_type == 'load_testing':
            return await self._test_system_load(params)
        else:
            return {'success': False, 'error': f'Unknown performance test type: {test_type}'}
    
    async def _test_concurrent_agents(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Test concurrent agent execution"""
        agent_count = params.get('agent_count', 5)
        duration = params.get('duration', 60)
        
        start_time = time.time()
        response_times = []
        errors = []
        
        async def run_agent_task(agent_id: int):
            task_start = time.time()
            try:
                # Simulate agent task
                result = await self._simulate_agent_task(f'agent_{agent_id}')
                task_end = time.time()
                response_times.append(task_end - task_start)
                return result
            except Exception as e:
                errors.append(str(e))
                return None
        
        # Run concurrent agent tasks
        tasks = [run_agent_task(i) for i in range(agent_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Calculate metrics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        success_rate = (len(results) - len(errors)) / len(results) if results else 0
        
        return {
            'success': success_rate > 0.8,
            'metrics': {
                'total_duration': total_duration,
                'average_response_time': avg_response_time,
                'success_rate': success_rate,
                'error_count': len(errors),
                'agent_count': agent_count
            }
        }
    
    async def _simulate_agent_task(self, agent_id: str) -> Dict[str, Any]:
        """Simulate an agent task for performance testing"""
        # Simulate processing time
        await asyncio.sleep(0.5 + (hash(agent_id) % 100) / 100)  # 0.5-1.5 seconds
        
        return {
            'agent_id': agent_id,
            'status': 'completed',
            'result': f'Task completed by {agent_id}'
        }
```

### Task 13.5: Human-in-the-Loop Testing Interface

**File**: `frontend/components/testing/test-interface.tsx`

**Testing Interface Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

interface TestScenario {
  name: string;
  description: string;
  status: 'pending' | 'running' | 'passed' | 'failed';
  progress: number;
  results?: any;
}

interface TestInterfaceProps {
  onRunTest: (scenarioName: string) => void;
  onValidateResult: (scenarioName: string, validation: any) => void;
}

export const TestInterface: React.FC<TestInterfaceProps> = ({
  onRunTest,
  onValidateResult
}) => {
  const [scenarios, setScenarios] = useState<TestScenario[]>([]);
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [validationMode, setValidationMode] = useState(false);

  useEffect(() => {
    // Load test scenarios
    fetchTestScenarios();
  }, []);

  const fetchTestScenarios = async () => {
    try {
      const response = await fetch('/api/testing/scenarios');
      const data = await response.json();
      setScenarios(data.scenarios);
    } catch (error) {
      console.error('Failed to fetch test scenarios:', error);
    }
  };

  const handleRunTest = async (scenarioName: string) => {
    setSelectedScenario(scenarioName);
    onRunTest(scenarioName);
  };

  const handleValidateResult = (validation: any) => {
    if (selectedScenario) {
      onValidateResult(selectedScenario, validation);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      case 'running': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Integration Testing Dashboard</h2>
        <Button
          onClick={() => setValidationMode(!validationMode)}
          variant={validationMode ? 'default' : 'outline'}
        >
          {validationMode ? 'Exit Validation Mode' : 'Enter Validation Mode'}
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {scenarios.map((scenario) => (
          <Card key={scenario.name} className="relative">
            <CardHeader>
              <div className="flex justify-between items-start">
                <CardTitle className="text-lg">{scenario.name}</CardTitle>
                <Badge className={getStatusColor(scenario.status)}>
                  {scenario.status}
                </Badge>
              </div>
              <p className="text-sm text-gray-600">{scenario.description}</p>
            </CardHeader>
            <CardContent>
              {scenario.status === 'running' && (
                <div className="mb-4">
                  <Progress value={scenario.progress} className="w-full" />
                  <p className="text-xs text-gray-500 mt-1">
                    Progress: {scenario.progress}%
                  </p>
                </div>
              )}
              
              <div className="flex gap-2">
                <Button
                  onClick={() => handleRunTest(scenario.name)}
                  disabled={scenario.status === 'running'}
                  size="sm"
                >
                  {scenario.status === 'running' ? 'Running...' : 'Run Test'}
                </Button>
                
                {validationMode && scenario.status === 'passed' && (
                  <Button
                    onClick={() => handleValidateResult({ validated: true })}
                    variant="outline"
                    size="sm"
                  >
                    Validate
                  </Button>
                )}
              </div>
              
              {scenario.results && (
                <div className="mt-4 p-2 bg-gray-50 rounded text-xs">
                  <pre>{JSON.stringify(scenario.results, null, 2)}</pre>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {validationMode && (
        <Card>
          <CardHeader>
            <CardTitle>Human Validation Guidelines</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <p>• Verify that all UI components render correctly</p>
              <p>• Check real-time updates and agent status changes</p>
              <p>• Validate chat interface responsiveness</p>
              <p>• Confirm workflow execution produces expected results</p>
              <p>• Test error handling and recovery mechanisms</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
```

## Testing API Endpoints

### Task 13.6: Testing API Integration

**File**: `app/api/testing.py`

**Testing API Endpoints**:
```python
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
from tests.integration.framework import IntegrationTestFramework

router = APIRouter(prefix="/api/testing", tags=["testing"])
test_framework = IntegrationTestFramework()

@router.get("/scenarios")
async def get_test_scenarios():
    """Get all available test scenarios"""
    scenarios = []
    for name, scenario in test_framework.test_scenarios.items():
        scenarios.append({
            'name': name,
            'description': scenario.description,
            'status': 'pending',
            'progress': 0
        })
    
    return {'scenarios': scenarios}

@router.post("/scenarios/{scenario_name}/run")
async def run_test_scenario(scenario_name: str):
    """Run a specific test scenario"""
    try:
        result = await test_framework.execute_test_scenario(scenario_name)
        return {'result': result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_test_results():
    """Get all test results"""
    return {'results': test_framework.test_results}

@router.post("/validate/{scenario_name}")
async def validate_test_result(scenario_name: str, validation: Dict[str, Any]):
    """Submit human validation for test result"""
    # Store validation result
    for result in test_framework.test_results:
        if result.get('scenario_name') == scenario_name:
            result['human_validation'] = validation
            break
    
    return {'status': 'validation_recorded'}
```

## Validation Criteria

### Backend Validation
- [ ] Integration test framework initializes correctly
- [ ] Test scenarios execute without errors
- [ ] Agent integration tests pass
- [ ] Performance tests meet criteria
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Testing interface displays scenarios correctly
- [ ] Test execution triggers properly
- [ ] Real-time updates show test progress
- [ ] Human validation workflow functions
- [ ] Results display accurately

### Integration Validation
- [ ] End-to-end workflows complete successfully
- [ ] Frontend-backend communication validated
- [ ] Performance metrics within acceptable ranges
- [ ] Error handling works across all components
- [ ] Human-in-the-loop validation effective

## Human Testing Scenarios

1. **Multi-Agent Workflow Test**: Execute complete development workflow and validate results
2. **Real-time Monitoring Test**: Monitor agent status updates in real-time
3. **Chat Interface Test**: Interact with agents through chat interface
4. **Performance Validation Test**: Verify system performance under load
5. **Error Recovery Test**: Test system recovery from various error conditions

## Next Steps

After successful validation of the integration testing framework, proceed to **14-external-service-integrations.md** for implementing external service integrations and API connections.

---

**Dependencies**: This phase requires all previous agent implementations to be functional and builds the foundation for comprehensive system validation.