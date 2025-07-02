# 12 - Deployment Agent Implementation

## Overview

The Deployment Agent serves as the automated deployment and infrastructure management hub for the multi-agent system. It provides comprehensive deployment orchestration, environment management, containerization, CI/CD pipeline integration, infrastructure provisioning, and automated scaling capabilities. This agent ensures reliable, scalable, and efficient deployment of applications across various environments.

## Current State Analysis

### Existing File
- `core/agents/deployment_agent.py` - Basic deployment functionality

### Enhancement Requirements
- Multi-environment deployment management (dev, staging, production)
- Container orchestration (Docker, Kubernetes)
- CI/CD pipeline integration
- Infrastructure as Code (IaC) support
- Automated scaling and load balancing
- Blue-green and canary deployments
- Environment configuration management
- Deployment monitoring and rollback capabilities
- Security scanning and compliance checks
- Resource optimization and cost management

## Implementation Tasks

### Task 12.1: Enhanced Deployment Agent

**File**: `core/agents/deployment_agent.py` (Complete Rewrite)

**Deployment Agent Implementation**:
```python
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
from enum import Enum
import json
import yaml
import docker
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentStatus
from ..services.llm_service import LLMService
from ..services.memory_service import MemoryService
from ..models import DeploymentTask, DeploymentConfig, EnvironmentConfig

class DeploymentStrategy(Enum):
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"

class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DeploymentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentMetrics:
    deployment_id: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    success_rate: float
    error_count: int
    resource_usage: Dict[str, float]
    performance_metrics: Dict[str, Any]

@dataclass
class ContainerConfig:
    image: str
    tag: str
    ports: List[Dict[str, int]]
    environment_vars: Dict[str, str]
    volumes: List[Dict[str, str]]
    resources: Dict[str, Any]
    health_check: Dict[str, Any]

class DeploymentAgent(BaseAgent):
    def __init__(self, agent_id: str = "deployment_agent"):
        super().__init__(
            agent_id=agent_id,
            name="Deployment Agent",
            description="Advanced deployment and infrastructure management agent"
        )
        self.capabilities = [
            "container_deployment",
            "kubernetes_orchestration",
            "ci_cd_integration",
            "infrastructure_provisioning",
            "environment_management",
            "automated_scaling",
            "blue_green_deployment",
            "canary_deployment",
            "rollback_management",
            "security_scanning",
            "performance_monitoring",
            "cost_optimization"
        ]
        
        self.deployment_sessions = {}
        self.environments = {}
        self.deployment_configs = {}
        self.active_deployments = {}
        self.deployment_history = []
        self.docker_client = None
        self.kubernetes_client = None
        
    async def initialize(self, llm_service: LLMService, memory_service: MemoryService):
        """Initialize deployment agent"""
        self.llm_service = llm_service
        self.memory_service = memory_service
        await self._setup_docker_client()
        await self._setup_kubernetes_client()
        await self._load_deployment_configs()
        await self._setup_default_environments()
        await self.update_status(AgentStatus.IDLE)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process deployment task"""
        try:
            await self.update_status(AgentStatus.THINKING)
            
            # Parse deployment task
            deployment_task = self._parse_deployment_task(task)
            
            # Create deployment session
            session_id = f"deployment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            self.deployment_sessions[session_id] = {
                'task': deployment_task,
                'start_time': datetime.utcnow(),
                'status': DeploymentStatus.PENDING,
                'steps_completed': [],
                'artifacts_created': []
            }
            
            await self.update_status(AgentStatus.WORKING)
            
            # Execute deployment workflow
            deployment_result = await self._execute_deployment_workflow(session_id, deployment_task)
            
            # Generate deployment report
            deployment_report = await self._generate_deployment_report(session_id, deployment_result)
            
            # Store deployment data
            await self._store_deployment_data(session_id, deployment_result, deployment_report)
            
            await self.update_status(AgentStatus.COMPLETED)
            
            return {
                'session_id': session_id,
                'deployment_strategy': deployment_task.strategy.value,
                'environment': deployment_task.environment.value,
                'deployment_status': deployment_result.get('status'),
                'deployed_services': deployment_result.get('deployed_services', []),
                'deployment_url': deployment_result.get('deployment_url'),
                'metrics': deployment_result.get('metrics'),
                'deployment_report': deployment_report,
                'confidence_score': deployment_result.get('confidence_score', 0.0)
            }
            
        except Exception as e:
            await self.update_status(AgentStatus.ERROR, str(e))
            raise
    
    async def can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Determine if agent can handle deployment task"""
        return task.get('type') in [
            'container_deployment',
            'kubernetes_deployment',
            'application_deployment',
            'infrastructure_provisioning',
            'environment_setup',
            'scaling_operation',
            'rollback_deployment',
            'blue_green_deployment',
            'canary_deployment'
        ]
    
    async def _execute_deployment_workflow(self, session_id: str, task: DeploymentTask) -> Dict[str, Any]:
        """Execute complete deployment workflow"""
        workflow_result = {
            'status': DeploymentStatus.IN_PROGRESS,
            'deployed_services': [],
            'deployment_url': None,
            'metrics': {},
            'confidence_score': 0.0
        }
        
        try:
            # Step 1: Pre-deployment validation
            await self.log_activity(f"Starting pre-deployment validation for session {session_id}")
            validation_result = await self._validate_deployment_requirements(task)
            if not validation_result['valid']:
                raise Exception(f"Deployment validation failed: {validation_result['errors']}")
            
            # Step 2: Environment preparation
            await self.log_activity("Preparing deployment environment")
            env_result = await self._prepare_deployment_environment(task)
            workflow_result['environment_prepared'] = env_result
            
            # Step 3: Build and package application
            await self.log_activity("Building and packaging application")
            build_result = await self._build_and_package_application(task)
            workflow_result['build_artifacts'] = build_result
            
            # Step 4: Security scanning
            await self.log_activity("Performing security scanning")
            security_result = await self._perform_security_scanning(task, build_result)
            workflow_result['security_scan'] = security_result
            
            # Step 5: Deploy application
            await self.log_activity("Deploying application")
            deploy_result = await self._deploy_application(task, build_result)
            workflow_result['deployed_services'] = deploy_result['services']
            workflow_result['deployment_url'] = deploy_result.get('url')
            
            # Step 6: Post-deployment verification
            await self.log_activity("Performing post-deployment verification")
            verification_result = await self._verify_deployment(task, deploy_result)
            workflow_result['verification'] = verification_result
            
            # Step 7: Configure monitoring and alerting
            await self.log_activity("Configuring monitoring and alerting")
            monitoring_result = await self._setup_deployment_monitoring(task, deploy_result)
            workflow_result['monitoring_configured'] = monitoring_result
            
            # Step 8: Performance testing
            await self.log_activity("Running performance tests")
            performance_result = await self._run_performance_tests(task, deploy_result)
            workflow_result['performance_metrics'] = performance_result
            
            # Step 9: Generate deployment metrics
            metrics = await self._collect_deployment_metrics(session_id, workflow_result)
            workflow_result['metrics'] = metrics
            
            # Step 10: Assess deployment quality
            quality_assessment = await self._assess_deployment_quality(workflow_result)
            workflow_result['confidence_score'] = quality_assessment['confidence_score']
            
            workflow_result['status'] = DeploymentStatus.COMPLETED
            await self.log_activity(f"Deployment workflow completed successfully for session {session_id}")
            
        except Exception as e:
            workflow_result['status'] = DeploymentStatus.FAILED
            workflow_result['error'] = str(e)
            await self.log_activity(f"Deployment workflow failed for session {session_id}: {str(e)}")
            
            # Attempt rollback if deployment partially succeeded
            if workflow_result.get('deployed_services'):
                await self.log_activity("Attempting automatic rollback")
                rollback_result = await self._rollback_deployment(session_id, task)
                workflow_result['rollback_result'] = rollback_result
        
        return workflow_result
    
    async def _validate_deployment_requirements(self, task: DeploymentTask) -> Dict[str, Any]:
        """Validate deployment requirements and prerequisites"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate environment configuration
            if task.environment not in self.environments:
                validation_result['errors'].append(f"Environment {task.environment.value} not configured")
            
            # Validate deployment strategy
            if task.strategy == DeploymentStrategy.BLUE_GREEN:
                if not await self._validate_blue_green_requirements(task):
                    validation_result['errors'].append("Blue-green deployment requirements not met")
            
            # Validate resource requirements
            resource_check = await self._validate_resource_requirements(task)
            if not resource_check['sufficient']:
                validation_result['errors'].append(f"Insufficient resources: {resource_check['missing']}")
            
            # Validate dependencies
            dependency_check = await self._validate_dependencies(task)
            if not dependency_check['satisfied']:
                validation_result['errors'].append(f"Unsatisfied dependencies: {dependency_check['missing']}")
            
            # Validate security requirements
            security_check = await self._validate_security_requirements(task)
            if not security_check['compliant']:
                validation_result['errors'].append(f"Security requirements not met: {security_check['issues']}")
            
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    async def _prepare_deployment_environment(self, task: DeploymentTask) -> Dict[str, Any]:
        """Prepare deployment environment"""
        env_result = {
            'environment_ready': False,
            'configurations_applied': [],
            'resources_provisioned': []
        }
        
        try:
            # Get environment configuration
            env_config = self.environments.get(task.environment)
            if not env_config:
                raise Exception(f"Environment configuration not found: {task.environment.value}")
            
            # Provision infrastructure if needed
            if task.provision_infrastructure:
                infra_result = await self._provision_infrastructure(task, env_config)
                env_result['infrastructure_provisioned'] = infra_result
            
            # Setup networking
            network_result = await self._setup_networking(task, env_config)
            env_result['networking_configured'] = network_result
            
            # Configure load balancers
            if task.enable_load_balancing:
                lb_result = await self._configure_load_balancers(task, env_config)
                env_result['load_balancers_configured'] = lb_result
            
            # Setup databases and storage
            storage_result = await self._setup_storage_and_databases(task, env_config)
            env_result['storage_configured'] = storage_result
            
            # Apply environment-specific configurations
            config_result = await self._apply_environment_configurations(task, env_config)
            env_result['configurations_applied'] = config_result
            
            env_result['environment_ready'] = True
            
        except Exception as e:
            env_result['error'] = str(e)
            raise
        
        return env_result
    
    async def _build_and_package_application(self, task: DeploymentTask) -> Dict[str, Any]:
        """Build and package application for deployment"""
        build_result = {
            'build_successful': False,
            'artifacts': [],
            'container_images': [],
            'build_logs': []
        }
        
        try:
            # Build application based on type
            if task.deployment_type == 'container':
                container_result = await self._build_container_images(task)
                build_result['container_images'] = container_result['images']
                build_result['build_logs'].extend(container_result['logs'])
            
            elif task.deployment_type == 'kubernetes':
                k8s_result = await self._build_kubernetes_manifests(task)
                build_result['kubernetes_manifests'] = k8s_result['manifests']
                build_result['artifacts'].extend(k8s_result['artifacts'])
            
            elif task.deployment_type == 'serverless':
                serverless_result = await self._build_serverless_package(task)
                build_result['serverless_package'] = serverless_result['package']
                build_result['artifacts'].append(serverless_result['package'])
            
            # Generate deployment scripts
            scripts_result = await self._generate_deployment_scripts(task)
            build_result['deployment_scripts'] = scripts_result
            
            # Create configuration files
            config_result = await self._generate_configuration_files(task)
            build_result['configuration_files'] = config_result
            
            build_result['build_successful'] = True
            
        except Exception as e:
            build_result['error'] = str(e)
            raise
        
        return build_result
    
    async def _perform_security_scanning(self, task: DeploymentTask, build_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform security scanning on build artifacts"""
        security_result = {
            'scan_completed': False,
            'vulnerabilities_found': [],
            'compliance_status': {},
            'security_score': 0.0
        }
        
        try:
            # Container image scanning
            if build_result.get('container_images'):
                for image in build_result['container_images']:
                    image_scan = await self._scan_container_image(image)
                    security_result['vulnerabilities_found'].extend(image_scan['vulnerabilities'])
            
            # Static code analysis
            if task.source_code_path:
                sast_result = await self._perform_static_analysis(task.source_code_path)
                security_result['static_analysis'] = sast_result
            
            # Dependency scanning
            dependency_scan = await self._scan_dependencies(task)
            security_result['dependency_vulnerabilities'] = dependency_scan
            
            # Compliance checking
            compliance_result = await self._check_compliance(task, build_result)
            security_result['compliance_status'] = compliance_result
            
            # Calculate security score
            security_score = await self._calculate_security_score(security_result)
            security_result['security_score'] = security_score
            
            # Check if security requirements are met
            if security_score < task.minimum_security_score:
                raise Exception(f"Security score {security_score} below minimum required {task.minimum_security_score}")
            
            security_result['scan_completed'] = True
            
        except Exception as e:
            security_result['error'] = str(e)
            raise
        
        return security_result
    
    async def _deploy_application(self, task: DeploymentTask, build_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy application using specified strategy"""
        deploy_result = {
            'deployment_successful': False,
            'services': [],
            'endpoints': [],
            'url': None
        }
        
        try:
            if task.strategy == DeploymentStrategy.ROLLING:
                result = await self._rolling_deployment(task, build_result)
            elif task.strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._blue_green_deployment(task, build_result)
            elif task.strategy == DeploymentStrategy.CANARY:
                result = await self._canary_deployment(task, build_result)
            else:
                result = await self._recreate_deployment(task, build_result)
            
            deploy_result.update(result)
            deploy_result['deployment_successful'] = True
            
        except Exception as e:
            deploy_result['error'] = str(e)
            raise
        
        return deploy_result
    
    async def _verify_deployment(self, task: DeploymentTask, deploy_result: Dict[str, Any]) -> Dict[str, Any]:
        """Verify deployment success and health"""
        verification_result = {
            'verification_successful': False,
            'health_checks': [],
            'connectivity_tests': [],
            'functional_tests': []
        }
        
        try:
            # Health checks
            for service in deploy_result['services']:
                health_check = await self._perform_health_check(service)
                verification_result['health_checks'].append(health_check)
            
            # Connectivity tests
            for endpoint in deploy_result['endpoints']:
                connectivity_test = await self._test_connectivity(endpoint)
                verification_result['connectivity_tests'].append(connectivity_test)
            
            # Functional tests
            if task.run_functional_tests:
                functional_result = await self._run_functional_tests(task, deploy_result)
                verification_result['functional_tests'] = functional_result
            
            # Smoke tests
            smoke_test_result = await self._run_smoke_tests(task, deploy_result)
            verification_result['smoke_tests'] = smoke_test_result
            
            # Check if all verifications passed
            all_passed = all([
                all(check['status'] == 'healthy' for check in verification_result['health_checks']),
                all(test['status'] == 'passed' for test in verification_result['connectivity_tests']),
                smoke_test_result.get('all_passed', True)
            ])
            
            verification_result['verification_successful'] = all_passed
            
        except Exception as e:
            verification_result['error'] = str(e)
            raise
        
        return verification_result
    
    async def _setup_deployment_monitoring(self, task: DeploymentTask, deploy_result: Dict[str, Any]) -> Dict[str, Any]:
        """Setup monitoring and alerting for deployed services"""
        monitoring_result = {
            'monitoring_configured': False,
            'metrics_endpoints': [],
            'alerts_configured': [],
            'dashboards_created': []
        }
        
        try:
            # Configure application metrics
            for service in deploy_result['services']:
                metrics_config = await self._configure_service_metrics(service, task)
                monitoring_result['metrics_endpoints'].append(metrics_config)
            
            # Setup alerting rules
            alerting_result = await self._configure_alerting_rules(task, deploy_result)
            monitoring_result['alerts_configured'] = alerting_result
            
            # Create monitoring dashboards
            dashboard_result = await self._create_monitoring_dashboards(task, deploy_result)
            monitoring_result['dashboards_created'] = dashboard_result
            
            # Configure log aggregation
            logging_result = await self._configure_log_aggregation(task, deploy_result)
            monitoring_result['logging_configured'] = logging_result
            
            monitoring_result['monitoring_configured'] = True
            
        except Exception as e:
            monitoring_result['error'] = str(e)
            # Don't raise exception for monitoring setup failures
        
        return monitoring_result
    
    async def _run_performance_tests(self, task: DeploymentTask, deploy_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run performance tests on deployed application"""
        performance_result = {
            'tests_completed': False,
            'load_test_results': {},
            'stress_test_results': {},
            'performance_metrics': {}
        }
        
        try:
            if not task.run_performance_tests:
                performance_result['tests_completed'] = True
                return performance_result
            
            # Load testing
            if deploy_result.get('url'):
                load_test = await self._run_load_tests(deploy_result['url'], task.performance_config)
                performance_result['load_test_results'] = load_test
            
            # Stress testing
            stress_test = await self._run_stress_tests(deploy_result, task.performance_config)
            performance_result['stress_test_results'] = stress_test
            
            # Collect performance metrics
            metrics = await self._collect_performance_metrics(deploy_result)
            performance_result['performance_metrics'] = metrics
            
            performance_result['tests_completed'] = True
            
        except Exception as e:
            performance_result['error'] = str(e)
            # Don't raise exception for performance test failures
        
        return performance_result
    
    async def _parse_deployment_task(self, task: Dict[str, Any]) -> DeploymentTask:
        """Parse and validate deployment task"""
        return DeploymentTask(
            deployment_type=task.get('deployment_type', 'container'),
            strategy=DeploymentStrategy(task.get('strategy', 'rolling')),
            environment=EnvironmentType(task.get('environment', 'development')),
            application_name=task.get('application_name', 'app'),
            source_code_path=task.get('source_code_path'),
            container_config=task.get('container_config', {}),
            kubernetes_config=task.get('kubernetes_config', {}),
            environment_variables=task.get('environment_variables', {}),
            resource_requirements=task.get('resource_requirements', {}),
            scaling_config=task.get('scaling_config', {}),
            security_config=task.get('security_config', {}),
            monitoring_config=task.get('monitoring_config', {}),
            performance_config=task.get('performance_config', {}),
            provision_infrastructure=task.get('provision_infrastructure', False),
            enable_load_balancing=task.get('enable_load_balancing', True),
            run_functional_tests=task.get('run_functional_tests', True),
            run_performance_tests=task.get('run_performance_tests', False),
            minimum_security_score=task.get('minimum_security_score', 0.8)
        )
    
    # Additional helper methods would be implemented here...
    # (Container operations, Kubernetes operations, etc.)
```

### Task 12.2: Deployment Service

**File**: `core/services/deployment_service.py`

```python
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import json

from .database_service import DatabaseService
from ..models import DeploymentTask, DeploymentConfig, EnvironmentConfig

class DeploymentService:
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        
    async def store_deployment_session(self, session_id: str, session_data: Dict[str, Any]):
        """Store deployment session data"""
        await self.db_service.execute(
            """
            INSERT INTO deployment_sessions 
            (session_id, task_data, start_time, status, result_data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                json.dumps(session_data.get('task', {})),
                session_data['start_time'],
                session_data.get('status', 'pending'),
                json.dumps(session_data.get('result', {}))
            )
        )
    
    async def get_deployment_history(self, environment: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get deployment history"""
        query = """
            SELECT session_id, task_data, start_time, status, result_data
            FROM deployment_sessions
        """
        params = []
        
        if environment:
            query += " WHERE JSON_EXTRACT(task_data, '$.environment') = ?"
            params.append(environment)
        
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        
        rows = await self.db_service.fetch_all(query, params)
        
        return [
            {
                'session_id': row[0],
                'task_data': json.loads(row[1]),
                'start_time': row[2],
                'status': row[3],
                'result_data': json.loads(row[4])
            }
            for row in rows
        ]
    
    async def get_active_deployments(self) -> List[Dict[str, Any]]:
        """Get currently active deployments"""
        rows = await self.db_service.fetch_all(
            """
            SELECT session_id, task_data, start_time, status
            FROM deployment_sessions
            WHERE status IN ('pending', 'in_progress')
            ORDER BY start_time DESC
            """
        )
        
        return [
            {
                'session_id': row[0],
                'task_data': json.loads(row[1]),
                'start_time': row[2],
                'status': row[3]
            }
            for row in rows
        ]
    
    async def store_deployment_metrics(self, session_id: str, metrics: Dict[str, Any]):
        """Store deployment metrics"""
        await self.db_service.execute(
            """
            INSERT INTO deployment_metrics
            (session_id, metric_type, metric_data, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (
                session_id,
                metrics.get('type', 'general'),
                json.dumps(metrics),
                datetime.utcnow()
            )
        )
    
    async def get_environment_status(self, environment: str) -> Dict[str, Any]:
        """Get environment status and health"""
        # Get recent deployments for environment
        recent_deployments = await self.get_deployment_history(environment, 10)
        
        # Calculate environment health metrics
        total_deployments = len(recent_deployments)
        successful_deployments = len([d for d in recent_deployments if d['status'] == 'completed'])
        success_rate = successful_deployments / total_deployments if total_deployments > 0 else 0
        
        return {
            'environment': environment,
            'total_deployments': total_deployments,
            'successful_deployments': successful_deployments,
            'success_rate': success_rate,
            'last_deployment': recent_deployments[0] if recent_deployments else None,
            'health_status': 'healthy' if success_rate > 0.8 else 'degraded' if success_rate > 0.5 else 'unhealthy'
        }
```

### Task 12.3: Database Schema

**File**: `core/database/deployment_schema.sql`

```sql
-- Deployment Sessions Table
CREATE TABLE IF NOT EXISTS deployment_sessions (
    session_id TEXT PRIMARY KEY,
    task_data TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'pending',
    result_data TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Deployment Metrics Table
CREATE TABLE IF NOT EXISTS deployment_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    metric_data TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES deployment_sessions(session_id)
);

-- Environment Configurations Table
CREATE TABLE IF NOT EXISTS environment_configs (
    environment_name TEXT PRIMARY KEY,
    config_data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Deployment Artifacts Table
CREATE TABLE IF NOT EXISTS deployment_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES deployment_sessions(session_id)
);

-- Service Deployments Table
CREATE TABLE IF NOT EXISTS service_deployments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    service_name TEXT NOT NULL,
    service_version TEXT NOT NULL,
    deployment_url TEXT,
    status TEXT NOT NULL,
    health_check_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES deployment_sessions(session_id)
);

-- Deployment Rollbacks Table
CREATE TABLE IF NOT EXISTS deployment_rollbacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_session_id TEXT NOT NULL,
    rollback_session_id TEXT NOT NULL,
    rollback_reason TEXT,
    rollback_status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (original_session_id) REFERENCES deployment_sessions(session_id),
    FOREIGN KEY (rollback_session_id) REFERENCES deployment_sessions(session_id)
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_deployment_sessions_status ON deployment_sessions(status);
CREATE INDEX IF NOT EXISTS idx_deployment_sessions_start_time ON deployment_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_deployment_metrics_session_id ON deployment_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_deployment_metrics_timestamp ON deployment_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_service_deployments_session_id ON service_deployments(session_id);
CREATE INDEX IF NOT EXISTS idx_service_deployments_status ON service_deployments(status);
```

## Frontend Implementation

### Task 12.4: Deployment Dashboard Component

**File**: `frontend/components/deployment-dashboard.tsx`

```tsx
"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Play, Square, RotateCcw, CheckCircle, XCircle, Clock, Zap } from 'lucide-react';

interface DeploymentTask {
  deployment_type: string;
  strategy: string;
  environment: string;
  application_name: string;
  source_code_path?: string;
  container_config?: any;
  kubernetes_config?: any;
  environment_variables?: Record<string, string>;
  resource_requirements?: any;
  scaling_config?: any;
  security_config?: any;
  monitoring_config?: any;
  performance_config?: any;
  provision_infrastructure?: boolean;
  enable_load_balancing?: boolean;
  run_functional_tests?: boolean;
  run_performance_tests?: boolean;
  minimum_security_score?: number;
}

interface DeploymentResult {
  session_id: string;
  deployment_strategy: string;
  environment: string;
  deployment_status: string;
  deployed_services: any[];
  deployment_url?: string;
  metrics: any;
  deployment_report: any;
  confidence_score: number;
}

interface DeploymentSession {
  session_id: string;
  task_data: DeploymentTask;
  start_time: string;
  status: string;
  result_data?: DeploymentResult;
}

const DeploymentDashboard: React.FC = () => {
  const [task, setTask] = useState<DeploymentTask>({
    deployment_type: 'container',
    strategy: 'rolling',
    environment: 'development',
    application_name: '',
    environment_variables: {},
    provision_infrastructure: false,
    enable_load_balancing: true,
    run_functional_tests: true,
    run_performance_tests: false,
    minimum_security_score: 0.8
  });
  
  const [isDeploying, setIsDeploying] = useState(false);
  const [currentResult, setCurrentResult] = useState<DeploymentResult | null>(null);
  const [deploymentHistory, setDeploymentHistory] = useState<DeploymentSession[]>([]);
  const [environmentStatus, setEnvironmentStatus] = useState<any>({});
  const [activeTab, setActiveTab] = useState('deploy');
  
  useEffect(() => {
    fetchDeploymentHistory();
    fetchEnvironmentStatus();
  }, []);
  
  const fetchDeploymentHistory = async () => {
    try {
      const response = await fetch('/api/deployment/history');
      if (response.ok) {
        const history = await response.json();
        setDeploymentHistory(history);
      }
    } catch (error) {
      console.error('Failed to fetch deployment history:', error);
    }
  };
  
  const fetchEnvironmentStatus = async () => {
    try {
      const environments = ['development', 'staging', 'production'];
      const statusPromises = environments.map(async (env) => {
        const response = await fetch(`/api/deployment/environment/${env}/status`);
        if (response.ok) {
          return { [env]: await response.json() };
        }
        return { [env]: null };
      });
      
      const statuses = await Promise.all(statusPromises);
      const combinedStatus = statuses.reduce((acc, status) => ({ ...acc, ...status }), {});
      setEnvironmentStatus(combinedStatus);
    } catch (error) {
      console.error('Failed to fetch environment status:', error);
    }
  };
  
  const handleDeploy = async () => {
    if (!task.application_name.trim()) {
      alert('Please enter an application name');
      return;
    }
    
    setIsDeploying(true);
    setCurrentResult(null);
    
    try {
      const response = await fetch('/api/deployment/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          type: 'application_deployment',
          ...task
        }),
      });
      
      if (response.ok) {
        const result = await response.json();
        setCurrentResult(result.result);
        await fetchDeploymentHistory();
        await fetchEnvironmentStatus();
      } else {
        const error = await response.json();
        alert(`Deployment failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Deployment error:', error);
      alert('Deployment failed. Please try again.');
    } finally {
      setIsDeploying(false);
    }
  };
  
  const handleRollback = async (sessionId: string) => {
    try {
      const response = await fetch(`/api/deployment/rollback/${sessionId}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        await fetchDeploymentHistory();
        await fetchEnvironmentStatus();
        alert('Rollback initiated successfully');
      } else {
        const error = await response.json();
        alert(`Rollback failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Rollback error:', error);
      alert('Rollback failed. Please try again.');
    }
  };
  
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'in_progress':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'default';
      case 'failed':
        return 'destructive';
      case 'in_progress':
        return 'secondary';
      default:
        return 'outline';
    }
  };
  
  const getEnvironmentHealthColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'default';
      case 'degraded':
        return 'secondary';
      case 'unhealthy':
        return 'destructive';
      default:
        return 'outline';
    }
  };
  
  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Deployment Dashboard</h1>
        <div className="flex items-center gap-2">
          <Zap className="h-6 w-6 text-blue-500" />
          <span className="text-sm text-muted-foreground">Automated Deployment & Infrastructure</span>
        </div>
      </div>
      
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="deploy">Deploy</TabsTrigger>
          <TabsTrigger value="environments">Environments</TabsTrigger>
          <TabsTrigger value="history">History</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>
        
        <TabsContent value="deploy" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Application Deployment</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="app-name">Application Name</Label>
                  <Input
                    id="app-name"
                    value={task.application_name}
                    onChange={(e) => setTask({ ...task, application_name: e.target.value })}
                    placeholder="Enter application name"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="deployment-type">Deployment Type</Label>
                  <Select
                    value={task.deployment_type}
                    onValueChange={(value) => setTask({ ...task, deployment_type: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="container">Container</SelectItem>
                      <SelectItem value="kubernetes">Kubernetes</SelectItem>
                      <SelectItem value="serverless">Serverless</SelectItem>
                      <SelectItem value="traditional">Traditional</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="strategy">Deployment Strategy</Label>
                  <Select
                    value={task.strategy}
                    onValueChange={(value) => setTask({ ...task, strategy: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rolling">Rolling Update</SelectItem>
                      <SelectItem value="blue_green">Blue-Green</SelectItem>
                      <SelectItem value="canary">Canary</SelectItem>
                      <SelectItem value="recreate">Recreate</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="environment">Environment</Label>
                  <Select
                    value={task.environment}
                    onValueChange={(value) => setTask({ ...task, environment: value })}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="development">Development</SelectItem>
                      <SelectItem value="staging">Staging</SelectItem>
                      <SelectItem value="production">Production</SelectItem>
                      <SelectItem value="testing">Testing</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="source-path">Source Code Path (Optional)</Label>
                <Input
                  id="source-path"
                  value={task.source_code_path || ''}
                  onChange={(e) => setTask({ ...task, source_code_path: e.target.value })}
                  placeholder="Path to source code repository"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="min-security-score">Minimum Security Score</Label>
                  <Input
                    id="min-security-score"
                    type="number"
                    min="0"
                    max="1"
                    step="0.1"
                    value={task.minimum_security_score}
                    onChange={(e) => setTask({ ...task, minimum_security_score: parseFloat(e.target.value) })}
                  />
                </div>
                
                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="provision-infra"
                      checked={task.provision_infrastructure}
                      onChange={(e) => setTask({ ...task, provision_infrastructure: e.target.checked })}
                    />
                    <Label htmlFor="provision-infra">Provision Infrastructure</Label>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="load-balancing"
                      checked={task.enable_load_balancing}
                      onChange={(e) => setTask({ ...task, enable_load_balancing: e.target.checked })}
                    />
                    <Label htmlFor="load-balancing">Enable Load Balancing</Label>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="functional-tests"
                      checked={task.run_functional_tests}
                      onChange={(e) => setTask({ ...task, run_functional_tests: e.target.checked })}
                    />
                    <Label htmlFor="functional-tests">Run Functional Tests</Label>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      id="performance-tests"
                      checked={task.run_performance_tests}
                      onChange={(e) => setTask({ ...task, run_performance_tests: e.target.checked })}
                    />
                    <Label htmlFor="performance-tests">Run Performance Tests</Label>
                  </div>
                </div>
              </div>
              
              <Button 
                onClick={handleDeploy} 
                disabled={isDeploying || !task.application_name.trim()}
                className="w-full"
              >
                {isDeploying ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Deploying...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Deploy Application
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
          
          {currentResult && (
            <Card>
              <CardHeader>
                <CardTitle>Deployment Result</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>Session ID</Label>
                    <p className="text-sm font-mono">{currentResult.session_id}</p>
                  </div>
                  <div>
                    <Label>Status</Label>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(currentResult.deployment_status)}
                      <Badge variant={getStatusColor(currentResult.deployment_status)}>
                        {currentResult.deployment_status}
                      </Badge>
                    </div>
                  </div>
                  <div>
                    <Label>Environment</Label>
                    <p className="text-sm">{currentResult.environment}</p>
                  </div>
                  <div>
                    <Label>Strategy</Label>
                    <p className="text-sm">{currentResult.deployment_strategy}</p>
                  </div>
                </div>
                
                {currentResult.deployment_url && (
                  <div>
                    <Label>Deployment URL</Label>
                    <a 
                      href={currentResult.deployment_url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-blue-500 hover:underline"
                    >
                      {currentResult.deployment_url}
                    </a>
                  </div>
                )}
                
                {currentResult.deployed_services && currentResult.deployed_services.length > 0 && (
                  <div>
                    <Label>Deployed Services</Label>
                    <div className="space-y-2">
                      {currentResult.deployed_services.map((service, index) => (
                        <div key={index} className="border rounded p-2">
                          <p className="font-medium">{service.name}</p>
                          <p className="text-sm text-muted-foreground">{service.version}</p>
                          {service.url && (
                            <a 
                              href={service.url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-blue-500 hover:underline text-sm"
                            >
                              {service.url}
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                <div>
                  <Label>Confidence Score</Label>
                  <div className="flex items-center gap-2">
                    <Progress value={currentResult.confidence_score * 100} className="flex-1" />
                    <span className="text-sm">{(currentResult.confidence_score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        <TabsContent value="environments" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(environmentStatus).map(([env, status]: [string, any]) => (
              <Card key={env}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span className="capitalize">{env}</span>
                    {status && (
                      <Badge variant={getEnvironmentHealthColor(status.health_status)}>
                        {status.health_status}
                      </Badge>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {status ? (
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Total Deployments:</span>
                        <span className="text-sm font-medium">{status.total_deployments}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Success Rate:</span>
                        <span className="text-sm font-medium">{(status.success_rate * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={status.success_rate * 100} className="mt-2" />
                      {status.last_deployment && (
                        <div className="mt-4 pt-2 border-t">
                          <p className="text-xs text-muted-foreground">Last Deployment:</p>
                          <p className="text-sm font-medium">{status.last_deployment.task_data.application_name}</p>
                          <p className="text-xs text-muted-foreground">
                            {new Date(status.last_deployment.start_time).toLocaleString()}
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">No data available</p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
        
        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Deployment History</CardTitle>
            </CardHeader>
            <CardContent>
              {deploymentHistory.length === 0 ? (
                <p className="text-muted-foreground">No deployment history available</p>
              ) : (
                <div className="space-y-3">
                  {deploymentHistory.map((session) => (
                    <div key={session.session_id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(session.status)}
                          <h4 className="font-semibold">{session.task_data.application_name}</h4>
                          <Badge variant={getStatusColor(session.status)}>
                            {session.status}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2">
                          {session.status === 'completed' && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleRollback(session.session_id)}
                            >
                              <RotateCcw className="h-4 w-4 mr-1" />
                              Rollback
                            </Button>
                          )}
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Environment:</span>
                          <p className="font-medium">{session.task_data.environment}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Strategy:</span>
                          <p className="font-medium">{session.task_data.strategy}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Type:</span>
                          <p className="font-medium">{session.task_data.deployment_type}</p>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Started:</span>
                          <p className="font-medium">{new Date(session.start_time).toLocaleString()}</p>
                        </div>
                      </div>
                      
                      {session.result_data && session.result_data.deployment_url && (
                        <div className="mt-2">
                          <span className="text-muted-foreground text-sm">URL:</span>
                          <a 
                            href={session.result_data.deployment_url} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-500 hover:underline ml-2"
                          >
                            {session.result_data.deployment_url}
                          </a>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="monitoring" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Deployment Monitoring</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">
                Deployment monitoring and metrics will be displayed here.
                This includes real-time deployment status, resource usage,
                performance metrics, and health checks.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default DeploymentDashboard;
```

## Backend API Integration

### Task 12.5: Deployment API Endpoints

**File**: `app/api/deployment.py`

```python
from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from core.agents.deployment_agent import DeploymentAgent
from core.services.deployment_service import DeploymentService
from core.models import DeploymentTask

router = APIRouter(prefix="/api/deployment", tags=["deployment"])

# Global deployment agent instance
deployment_agent: Optional[DeploymentAgent] = None
deployment_service: Optional[DeploymentService] = None

@router.post("/start")
async def start_deployment(task: Dict[str, Any]):
    """Start deployment task"""
    try:
        if not deployment_agent:
            raise HTTPException(status_code=500, detail="Deployment agent not initialized")
        
        result = await deployment_agent.process_task(task)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_deployment_history(
    environment: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """Get deployment history"""
    try:
        if not deployment_service:
            raise HTTPException(status_code=500, detail="Deployment service not initialized")
        
        history = await deployment_service.get_deployment_history(environment, limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/active")
async def get_active_deployments():
    """Get currently active deployments"""
    try:
        if not deployment_service:
            raise HTTPException(status_code=500, detail="Deployment service not initialized")
        
        active_deployments = await deployment_service.get_active_deployments()
        return active_deployments
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environment/{environment}/status")
async def get_environment_status(environment: str = Path(...)):
    """Get environment status and health"""
    try:
        if not deployment_service:
            raise HTTPException(status_code=500, detail="Deployment service not initialized")
        
        status = await deployment_service.get_environment_status(environment)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback/{session_id}")
async def rollback_deployment(session_id: str = Path(...)):
    """Rollback a deployment"""
    try:
        if not deployment_agent:
            raise HTTPException(status_code=500, detail="Deployment agent not initialized")
        
        rollback_task = {
            "type": "rollback_deployment",
            "session_id": session_id
        }
        
        result = await deployment_agent.process_task(rollback_task)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}")
async def get_deployment_session(session_id: str = Path(...)):
    """Get deployment session details"""
    try:
        if not deployment_agent:
            raise HTTPException(status_code=500, detail="Deployment agent not initialized")
        
        session = deployment_agent.deployment_sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/environments")
async def get_supported_environments():
    """Get supported deployment environments"""
    return {
        "environments": [
            {"value": "development", "label": "Development"},
            {"value": "staging", "label": "Staging"},
            {"value": "production", "label": "Production"},
            {"value": "testing", "label": "Testing"}
        ]
    }

@router.get("/strategies")
async def get_deployment_strategies():
    """Get supported deployment strategies"""
    return {
        "strategies": [
            {"value": "rolling", "label": "Rolling Update", "description": "Gradually replace instances"},
            {"value": "blue_green", "label": "Blue-Green", "description": "Switch between two environments"},
            {"value": "canary", "label": "Canary", "description": "Gradual rollout to subset of users"},
            {"value": "recreate", "label": "Recreate", "description": "Stop all instances and create new ones"}
        ]
    }

@router.get("/types")
async def get_deployment_types():
    """Get supported deployment types"""
    return {
        "types": [
            {"value": "container", "label": "Container", "description": "Docker container deployment"},
            {"value": "kubernetes", "label": "Kubernetes", "description": "Kubernetes orchestration"},
            {"value": "serverless", "label": "Serverless", "description": "Function-as-a-Service deployment"},
            {"value": "traditional", "label": "Traditional", "description": "Traditional server deployment"}
        ]
    }
```

### Task 12.6: Enhanced Pydantic Models

**File**: `core/models.py` (Add to existing models)

```python
# Deployment Models
class DeploymentTask(BaseModel):
    deployment_type: str = Field(..., description="Type of deployment (container, kubernetes, serverless, traditional)")
    strategy: str = Field(..., description="Deployment strategy (rolling, blue_green, canary, recreate)")
    environment: str = Field(..., description="Target environment (development, staging, production, testing)")
    application_name: str = Field(..., description="Name of the application to deploy")
    source_code_path: Optional[str] = Field(None, description="Path to source code repository")
    container_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Container configuration")
    kubernetes_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Kubernetes configuration")
    environment_variables: Optional[Dict[str, str]] = Field(default_factory=dict, description="Environment variables")
    resource_requirements: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Resource requirements")
    scaling_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Auto-scaling configuration")
    security_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Security configuration")
    monitoring_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Monitoring configuration")
    performance_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance testing configuration")
    provision_infrastructure: bool = Field(False, description="Whether to provision infrastructure")
    enable_load_balancing: bool = Field(True, description="Whether to enable load balancing")
    run_functional_tests: bool = Field(True, description="Whether to run functional tests")
    run_performance_tests: bool = Field(False, description="Whether to run performance tests")
    minimum_security_score: float = Field(0.8, description="Minimum required security score")

class DeploymentConfig(BaseModel):
    environment: str = Field(..., description="Environment name")
    infrastructure_config: Dict[str, Any] = Field(default_factory=dict, description="Infrastructure configuration")
    networking_config: Dict[str, Any] = Field(default_factory=dict, description="Networking configuration")
    security_policies: List[str] = Field(default_factory=list, description="Security policies")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource limits")
    monitoring_endpoints: List[str] = Field(default_factory=list, description="Monitoring endpoints")
    backup_config: Optional[Dict[str, Any]] = Field(None, description="Backup configuration")

class EnvironmentConfig(BaseModel):
    name: str = Field(..., description="Environment name")
    type: str = Field(..., description="Environment type")
    region: str = Field(..., description="Deployment region")
    vpc_config: Optional[Dict[str, Any]] = Field(None, description="VPC configuration")
    database_config: Optional[Dict[str, Any]] = Field(None, description="Database configuration")
    storage_config: Optional[Dict[str, Any]] = Field(None, description="Storage configuration")
    load_balancer_config: Optional[Dict[str, Any]] = Field(None, description="Load balancer configuration")
    ssl_config: Optional[Dict[str, Any]] = Field(None, description="SSL configuration")
    domain_config: Optional[Dict[str, Any]] = Field(None, description="Domain configuration")

class DeploymentResult(BaseModel):
    session_id: str = Field(..., description="Deployment session ID")
    deployment_strategy: str = Field(..., description="Deployment strategy used")
    environment: str = Field(..., description="Target environment")
    deployment_status: str = Field(..., description="Deployment status")
    deployed_services: List[Dict[str, Any]] = Field(default_factory=list, description="Deployed services")
    deployment_url: Optional[str] = Field(None, description="Deployment URL")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Deployment metrics")
    deployment_report: Dict[str, Any] = Field(default_factory=dict, description="Detailed deployment report")
    confidence_score: float = Field(0.0, description="Deployment confidence score")
    artifacts_created: List[str] = Field(default_factory=list, description="Created artifacts")
    rollback_available: bool = Field(False, description="Whether rollback is available")

class DeploymentMetrics(BaseModel):
    deployment_id: str = Field(..., description="Deployment ID")
    start_time: datetime = Field(..., description="Deployment start time")
    end_time: Optional[datetime] = Field(None, description="Deployment end time")
    duration: Optional[float] = Field(None, description="Deployment duration in seconds")
    success_rate: float = Field(0.0, description="Deployment success rate")
    error_count: int = Field(0, description="Number of errors")
    resource_usage: Dict[str, float] = Field(default_factory=dict, description="Resource usage metrics")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    security_scan_results: Dict[str, Any] = Field(default_factory=dict, description="Security scan results")
    compliance_status: Dict[str, bool] = Field(default_factory=dict, description="Compliance status")

class RollbackConfig(BaseModel):
    original_session_id: str = Field(..., description="Original deployment session ID")
    rollback_strategy: str = Field("immediate", description="Rollback strategy")
    preserve_data: bool = Field(True, description="Whether to preserve data during rollback")
    notification_config: Optional[Dict[str, Any]] = Field(None, description="Notification configuration")
    verification_tests: List[str] = Field(default_factory=list, description="Tests to run after rollback")
```

## Testing Strategy

### Task 12.7: Unit Tests for Deployment Agent

**File**: `tests/test_deployment_agent.py`

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from core.agents.deployment_agent import DeploymentAgent
from core.models import DeploymentTask

@pytest.fixture
def deployment_agent():
    return DeploymentAgent()

@pytest.fixture
def sample_deployment_task():
    return {
        "deployment_type": "container",
        "strategy": "rolling",
        "environment": "staging",
        "application_name": "test-app",
        "source_code_path": "/path/to/source",
        "container_config": {
            "image": "test-app:latest",
            "ports": [8080]
        },
        "environment_variables": {
            "NODE_ENV": "staging"
        },
        "provision_infrastructure": True,
        "enable_load_balancing": True,
        "run_functional_tests": True
    }

class TestDeploymentAgent:
    @pytest.mark.asyncio
    async def test_initialization(self, deployment_agent):
        """Test deployment agent initialization"""
        assert deployment_agent.deployment_sessions == {}
        assert deployment_agent.active_deployments == {}
        assert deployment_agent.environment_configs == {}
        assert deployment_agent.deployment_history == []

    @pytest.mark.asyncio
    async def test_process_deployment_task(self, deployment_agent, sample_deployment_task):
        """Test processing deployment task"""
        with patch.object(deployment_agent, '_execute_deployment_workflow', new_callable=AsyncMock) as mock_workflow:
            mock_workflow.return_value = {
                "session_id": "test-session-123",
                "deployment_status": "completed",
                "deployment_url": "https://test-app.staging.example.com",
                "confidence_score": 0.95
            }
            
            result = await deployment_agent.process_task(sample_deployment_task)
            
            assert result["deployment_status"] == "completed"
            assert "session_id" in result
            assert result["confidence_score"] > 0.9
            mock_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_deployment_config(self, deployment_agent, sample_deployment_task):
        """Test deployment configuration validation"""
        # Test valid configuration
        is_valid, errors = await deployment_agent._validate_deployment_config(sample_deployment_task)
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid configuration
        invalid_task = sample_deployment_task.copy()
        invalid_task["deployment_type"] = "invalid_type"
        
        is_valid, errors = await deployment_agent._validate_deployment_config(invalid_task)
        assert not is_valid
        assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_prepare_environment(self, deployment_agent, sample_deployment_task):
        """Test environment preparation"""
        with patch.object(deployment_agent, '_provision_infrastructure', new_callable=AsyncMock) as mock_provision:
            mock_provision.return_value = True
            
            result = await deployment_agent._prepare_environment(sample_deployment_task)
            
            assert result is True
            mock_provision.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_and_package(self, deployment_agent, sample_deployment_task):
        """Test build and package process"""
        with patch.object(deployment_agent, '_build_container_image', new_callable=AsyncMock) as mock_build:
            mock_build.return_value = {
                "image_id": "test-app:latest",
                "build_status": "success"
            }
            
            result = await deployment_agent._build_and_package(sample_deployment_task)
            
            assert result["build_status"] == "success"
            assert "image_id" in result
            mock_build.assert_called_once()

    @pytest.mark.asyncio
    async def test_security_scanning(self, deployment_agent, sample_deployment_task):
        """Test security scanning"""
        with patch.object(deployment_agent, '_run_security_scan', new_callable=AsyncMock) as mock_scan:
            mock_scan.return_value = {
                "security_score": 0.9,
                "vulnerabilities": [],
                "compliance_status": "passed"
            }
            
            result = await deployment_agent._run_security_scan(sample_deployment_task)
            
            assert result["security_score"] >= 0.8
            assert result["compliance_status"] == "passed"
            mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_application(self, deployment_agent, sample_deployment_task):
        """Test application deployment"""
        with patch.object(deployment_agent, '_execute_deployment_strategy', new_callable=AsyncMock) as mock_deploy:
            mock_deploy.return_value = {
                "deployment_id": "deploy-123",
                "status": "deployed",
                "endpoints": ["https://test-app.staging.example.com"]
            }
            
            result = await deployment_agent._deploy_application(sample_deployment_task)
            
            assert result["status"] == "deployed"
            assert len(result["endpoints"]) > 0
            mock_deploy.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_deployment(self, deployment_agent, sample_deployment_task):
        """Test deployment verification"""
        with patch.object(deployment_agent, '_run_health_checks', new_callable=AsyncMock) as mock_health:
            mock_health.return_value = {
                "health_status": "healthy",
                "response_time": 150,
                "availability": 100.0
            }
            
            result = await deployment_agent._verify_deployment(sample_deployment_task)
            
            assert result["health_status"] == "healthy"
            assert result["availability"] == 100.0
            mock_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_rollback_deployment(self, deployment_agent):
        """Test deployment rollback"""
        rollback_config = {
            "original_session_id": "session-123",
            "rollback_strategy": "immediate",
            "preserve_data": True
        }
        
        with patch.object(deployment_agent, '_execute_rollback', new_callable=AsyncMock) as mock_rollback:
            mock_rollback.return_value = {
                "rollback_status": "completed",
                "rollback_time": datetime.now().isoformat()
            }
            
            result = await deployment_agent._rollback_deployment(rollback_config)
            
            assert result["rollback_status"] == "completed"
            mock_rollback.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_testing(self, deployment_agent, sample_deployment_task):
        """Test performance testing"""
        with patch.object(deployment_agent, '_run_performance_tests', new_callable=AsyncMock) as mock_perf:
            mock_perf.return_value = {
                "response_time_avg": 120,
                "throughput": 1000,
                "error_rate": 0.01,
                "performance_score": 0.95
            }
            
            result = await deployment_agent._run_performance_tests(sample_deployment_task)
            
            assert result["performance_score"] > 0.9
            assert result["error_rate"] < 0.05
            mock_perf.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_setup(self, deployment_agent, sample_deployment_task):
        """Test monitoring setup"""
        with patch.object(deployment_agent, '_setup_monitoring', new_callable=AsyncMock) as mock_monitoring:
            mock_monitoring.return_value = {
                "monitoring_enabled": True,
                "dashboards_created": ["main-dashboard", "performance-dashboard"],
                "alerts_configured": 5
            }
            
            result = await deployment_agent._setup_monitoring(sample_deployment_task)
            
            assert result["monitoring_enabled"] is True
            assert len(result["dashboards_created"]) > 0
            mock_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling(self, deployment_agent, sample_deployment_task):
        """Test error handling during deployment"""
        with patch.object(deployment_agent, '_execute_deployment_workflow', new_callable=AsyncMock) as mock_workflow:
            mock_workflow.side_effect = Exception("Deployment failed")
            
            result = await deployment_agent.process_task(sample_deployment_task)
            
            assert "error" in result
            assert result["deployment_status"] == "failed"
```

### Task 12.8: Integration Tests for Deployment API

**File**: `tests/test_deployment_api.py`

```python
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from app.main import app
from core.agents.deployment_agent import DeploymentAgent
from core.services.deployment_service import DeploymentService

client = TestClient(app)

@pytest.fixture
def mock_deployment_agent():
    agent = Mock(spec=DeploymentAgent)
    agent.process_task = AsyncMock()
    agent.deployment_sessions = {}
    return agent

@pytest.fixture
def mock_deployment_service():
    service = Mock(spec=DeploymentService)
    service.get_deployment_history = AsyncMock()
    service.get_active_deployments = AsyncMock()
    service.get_environment_status = AsyncMock()
    return service

@pytest.fixture
def sample_deployment_request():
    return {
        "deployment_type": "container",
        "strategy": "rolling",
        "environment": "staging",
        "application_name": "test-app",
        "source_code_path": "/path/to/source",
        "container_config": {
            "image": "test-app:latest",
            "ports": [8080]
        },
        "provision_infrastructure": True,
        "enable_load_balancing": True
    }

class TestDeploymentAPI:
    def test_start_deployment_success(self, mock_deployment_agent, sample_deployment_request):
        """Test successful deployment start"""
        mock_deployment_agent.process_task.return_value = {
            "session_id": "test-session-123",
            "deployment_status": "in_progress",
            "confidence_score": 0.9
        }
        
        with patch('app.api.deployment.deployment_agent', mock_deployment_agent):
            response = client.post("/api/deployment/start", json=sample_deployment_request)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "result" in data
            assert data["result"]["deployment_status"] == "in_progress"

    def test_start_deployment_agent_not_initialized(self, sample_deployment_request):
        """Test deployment start when agent is not initialized"""
        with patch('app.api.deployment.deployment_agent', None):
            response = client.post("/api/deployment/start", json=sample_deployment_request)
            
            assert response.status_code == 500
            assert "Deployment agent not initialized" in response.json()["detail"]

    def test_get_deployment_history_success(self, mock_deployment_service):
        """Test successful deployment history retrieval"""
        mock_deployment_service.get_deployment_history.return_value = [
            {
                "session_id": "session-1",
                "application_name": "app-1",
                "environment": "staging",
                "status": "completed",
                "start_time": "2024-01-01T10:00:00Z"
            }
        ]
        
        with patch('app.api.deployment.deployment_service', mock_deployment_service):
            response = client.get("/api/deployment/history")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["session_id"] == "session-1"

    def test_get_deployment_history_with_filters(self, mock_deployment_service):
        """Test deployment history with environment filter"""
        mock_deployment_service.get_deployment_history.return_value = []
        
        with patch('app.api.deployment.deployment_service', mock_deployment_service):
            response = client.get("/api/deployment/history?environment=production&limit=10")
            
            assert response.status_code == 200
            mock_deployment_service.get_deployment_history.assert_called_with("production", 10)

    def test_get_active_deployments_success(self, mock_deployment_service):
        """Test successful active deployments retrieval"""
        mock_deployment_service.get_active_deployments.return_value = [
            {
                "session_id": "active-session-1",
                "application_name": "active-app",
                "status": "deploying",
                "progress": 75
            }
        ]
        
        with patch('app.api.deployment.deployment_service', mock_deployment_service):
            response = client.get("/api/deployment/active")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["status"] == "deploying"

    def test_get_environment_status_success(self, mock_deployment_service):
        """Test successful environment status retrieval"""
        mock_deployment_service.get_environment_status.return_value = {
            "environment": "staging",
            "health_status": "healthy",
            "total_deployments": 25,
            "success_rate": 0.96,
            "last_deployment": {
                "application_name": "test-app",
                "status": "completed"
            }
        }
        
        with patch('app.api.deployment.deployment_service', mock_deployment_service):
            response = client.get("/api/deployment/environment/staging/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["environment"] == "staging"
            assert data["health_status"] == "healthy"

    def test_rollback_deployment_success(self, mock_deployment_agent):
        """Test successful deployment rollback"""
        mock_deployment_agent.process_task.return_value = {
            "rollback_status": "completed",
            "rollback_time": "2024-01-01T11:00:00Z"
        }
        
        with patch('app.api.deployment.deployment_agent', mock_deployment_agent):
            response = client.post("/api/deployment/rollback/session-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["result"]["rollback_status"] == "completed"

    def test_get_deployment_session_success(self, mock_deployment_agent):
        """Test successful deployment session retrieval"""
        mock_deployment_agent.deployment_sessions = {
            "session-123": {
                "session_id": "session-123",
                "status": "completed",
                "application_name": "test-app"
            }
        }
        
        with patch('app.api.deployment.deployment_agent', mock_deployment_agent):
            response = client.get("/api/deployment/session/session-123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "session-123"
            assert data["status"] == "completed"

    def test_get_deployment_session_not_found(self, mock_deployment_agent):
        """Test deployment session not found"""
        mock_deployment_agent.deployment_sessions = {}
        
        with patch('app.api.deployment.deployment_agent', mock_deployment_agent):
            response = client.get("/api/deployment/session/nonexistent")
            
            assert response.status_code == 404
            assert "Session not found" in response.json()["detail"]

    def test_get_supported_environments(self):
        """Test getting supported environments"""
        response = client.get("/api/deployment/environments")
        
        assert response.status_code == 200
        data = response.json()
        assert "environments" in data
        assert len(data["environments"]) > 0
        assert any(env["value"] == "production" for env in data["environments"])

    def test_get_deployment_strategies(self):
        """Test getting deployment strategies"""
        response = client.get("/api/deployment/strategies")
        
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert len(data["strategies"]) > 0
        assert any(strategy["value"] == "blue_green" for strategy in data["strategies"])

    def test_get_deployment_types(self):
        """Test getting deployment types"""
        response = client.get("/api/deployment/types")
        
        assert response.status_code == 200
        data = response.json()
        assert "types" in data
        assert len(data["types"]) > 0
        assert any(dtype["value"] == "kubernetes" for dtype in data["types"])
```

### Task 12.9: Frontend Component Tests

**File**: `frontend/__tests__/deployment-dashboard.test.tsx`

```typescript
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { jest } from '@jest/globals';
import DeploymentDashboard from '../components/deployment-dashboard';

// Mock fetch
global.fetch = jest.fn();

const mockFetch = fetch as jest.MockedFunction<typeof fetch>;

describe('DeploymentDashboard', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  test('renders deployment dashboard with all tabs', () => {
    render(<DeploymentDashboard />);
    
    expect(screen.getByText('Deployment Agent')).toBeInTheDocument();
    expect(screen.getByText('Deploy')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
    expect(screen.getByText('Environments')).toBeInTheDocument();
    expect(screen.getByText('History')).toBeInTheDocument();
    expect(screen.getByText('Monitoring')).toBeInTheDocument();
  });

  test('loads deployment options on mount', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          environments: [{ value: 'staging', label: 'Staging' }]
        })
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          strategies: [{ value: 'rolling', label: 'Rolling Update' }]
        })
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          types: [{ value: 'container', label: 'Container' }]
        })
      } as Response);

    render(<DeploymentDashboard />);
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });
  });

  test('submits deployment form with correct data', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ environments: [] })
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ strategies: [] })
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ types: [] })
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          status: 'success',
          result: { session_id: 'test-123' }
        })
      } as Response);

    render(<DeploymentDashboard />);
    
    // Fill form
    fireEvent.change(screen.getByLabelText(/Application Name/i), {
      target: { value: 'test-app' }
    });
    
    // Submit form
    fireEvent.click(screen.getByText('Start Deployment'));
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith('/api/deployment/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: expect.stringContaining('test-app')
      });
    });
  });

  test('displays deployment progress correctly', async () => {
    const mockSession = {
      session_id: 'test-123',
      status: 'in_progress',
      progress: 50,
      task_data: { application_name: 'test-app' }
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => mockSession
    } as Response);

    render(<DeploymentDashboard />);
    
    // Simulate having an active session
    fireEvent.click(screen.getByText('Active'));
    
    await waitFor(() => {
      expect(screen.getByText('50%')).toBeInTheDocument();
    });
  });

  test('handles deployment errors gracefully', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));

    render(<DeploymentDashboard />);
    
    fireEvent.change(screen.getByLabelText(/Application Name/i), {
      target: { value: 'test-app' }
    });
    
    fireEvent.click(screen.getByText('Start Deployment'));
    
    await waitFor(() => {
      expect(screen.getByText(/Error starting deployment/i)).toBeInTheDocument();
    });
  });

  test('displays deployment history correctly', async () => {
    const mockHistory = [
      {
        session_id: 'session-1',
        task_data: { application_name: 'app-1', environment: 'staging' },
        status: 'completed',
        start_time: '2024-01-01T10:00:00Z'
      }
    ];

    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => mockHistory
    } as Response);

    render(<DeploymentDashboard />);
    
    fireEvent.click(screen.getByText('History'));
    
    await waitFor(() => {
      expect(screen.getByText('app-1')).toBeInTheDocument();
      expect(screen.getByText('staging')).toBeInTheDocument();
      expect(screen.getByText('completed')).toBeInTheDocument();
    });
  });

  test('handles rollback functionality', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ status: 'success' })
    } as Response);

    render(<DeploymentDashboard />);
    
    // Mock a completed deployment in history
    const rollbackButton = screen.getByText('Rollback');
    fireEvent.click(rollbackButton);
    
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/rollback/'),
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  test('validates required form fields', () => {
    render(<DeploymentDashboard />);
    
    // Try to submit without required fields
    fireEvent.click(screen.getByText('Start Deployment'));
    
    expect(screen.getByText(/Application name is required/i)).toBeInTheDocument();
  });

  test('toggles advanced configuration sections', () => {
    render(<DeploymentDashboard />);
    
    // Check container configuration toggle
    const containerToggle = screen.getByLabelText(/Enable Container Configuration/i);
    fireEvent.click(containerToggle);
    
    expect(screen.getByLabelText(/Container Image/i)).toBeInTheDocument();
  });

  test('displays environment status correctly', async () => {
    const mockStatus = {
      staging: {
        health_status: 'healthy',
        total_deployments: 25,
        success_rate: 0.96
      }
    };

    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => mockStatus.staging
    } as Response);

    render(<DeploymentDashboard />);
    
    fireEvent.click(screen.getByText('Environments'));
    
    await waitFor(() => {
      expect(screen.getByText('healthy')).toBeInTheDocument();
      expect(screen.getByText('96.0%')).toBeInTheDocument();
    });
  });
});
```

## Validation Criteria

### Backend Validation

1. **Deployment Agent Core Functionality**:
   - ✅ Multi-environment deployment support (development, staging, production, testing)
   - ✅ Multiple deployment strategies (rolling, blue-green, canary, recreate)
   - ✅ Container and Kubernetes orchestration
   - ✅ Infrastructure provisioning and configuration
   - ✅ Security scanning and compliance validation
   - ✅ Automated testing integration (functional, performance)
   - ✅ Monitoring and alerting setup
   - ✅ Rollback capabilities with data preservation
   - ✅ Resource optimization and cost management

2. **API Endpoints**:
   - ✅ POST `/api/deployment/start` - Start deployment tasks
   - ✅ GET `/api/deployment/history` - Retrieve deployment history with filters
   - ✅ GET `/api/deployment/active` - Get active deployments
   - ✅ GET `/api/deployment/environment/{environment}/status` - Environment health
   - ✅ POST `/api/deployment/rollback/{session_id}` - Rollback deployments
   - ✅ GET `/api/deployment/session/{session_id}` - Session details
   - ✅ GET `/api/deployment/environments` - Supported environments
   - ✅ GET `/api/deployment/strategies` - Deployment strategies
   - ✅ GET `/api/deployment/types` - Deployment types

3. **Database Integration**:
   - ✅ Deployment sessions tracking
   - ✅ Metrics and performance data storage
   - ✅ Environment configurations management
   - ✅ Deployment artifacts tracking
   - ✅ Service deployments monitoring
   - ✅ Rollback history maintenance

### Frontend Validation

1. **User Interface Components**:
   - ✅ Deployment configuration form with validation
   - ✅ Real-time deployment progress tracking
   - ✅ Active deployments monitoring dashboard
   - ✅ Environment status and health indicators
   - ✅ Deployment history with filtering
   - ✅ Rollback functionality with confirmation
   - ✅ Advanced configuration options (container, Kubernetes, security)
   - ✅ Performance and monitoring metrics display

2. **State Management**:
   - ✅ Form state management with validation
   - ✅ Real-time session polling for progress updates
   - ✅ Error handling and user feedback
   - ✅ Loading states and progress indicators
   - ✅ Data persistence and caching

3. **User Experience**:
   - ✅ Intuitive deployment workflow
   - ✅ Clear status indicators and progress feedback
   - ✅ Responsive design for different screen sizes
   - ✅ Accessible form controls and navigation
   - ✅ Comprehensive error messages and guidance

## Human Testing Scenarios

### Scenario 1: Container Application Deployment
**Objective**: Deploy a containerized web application to staging environment

**Steps**:
1. Navigate to Deployment Dashboard
2. Select "Deploy" tab
3. Configure deployment:
   - Application Name: "web-app-v2"
   - Deployment Type: "Container"
   - Strategy: "Rolling Update"
   - Environment: "Staging"
   - Enable container configuration
   - Set container image: "web-app:v2.1.0"
   - Configure ports: 8080, 8443
   - Set environment variables: NODE_ENV=staging
4. Enable infrastructure provisioning
5. Enable load balancing
6. Enable functional tests
7. Start deployment
8. Monitor progress in "Active" tab
9. Verify deployment success and access URL
10. Check deployment metrics and logs

**Expected Results**:
- Deployment completes successfully with >95% confidence score
- Application is accessible via provided URL
- Health checks pass with "healthy" status
- Performance metrics show acceptable response times
- Security scan passes with score >0.8

### Scenario 2: Kubernetes Microservice Deployment
**Objective**: Deploy a microservice architecture to production using Kubernetes

**Steps**:
1. Navigate to Deployment Dashboard
2. Configure deployment:
   - Application Name: "user-service"
   - Deployment Type: "Kubernetes"
   - Strategy: "Blue-Green"
   - Environment: "Production"
   - Enable Kubernetes configuration
   - Set namespace: "production"
   - Configure replicas: 3
   - Set resource limits: CPU 500m, Memory 1Gi
   - Enable auto-scaling: min 2, max 10
3. Configure security settings
4. Enable performance testing
5. Start deployment
6. Monitor blue-green deployment progress
7. Verify traffic switching
8. Check service mesh integration
9. Validate monitoring and alerting setup

**Expected Results**:
- Blue-green deployment executes without downtime
- All replicas are healthy and ready
- Auto-scaling configuration is active
- Service mesh integration is successful
- Monitoring dashboards are created and populated
- Performance tests pass with acceptable metrics

### Scenario 3: Serverless Function Deployment
**Objective**: Deploy serverless functions with automated scaling

**Steps**:
1. Navigate to Deployment Dashboard
2. Configure deployment:
   - Application Name: "data-processor"
   - Deployment Type: "Serverless"
   - Strategy: "Canary"
   - Environment: "Production"
   - Configure function settings
   - Set memory allocation: 512MB
   - Set timeout: 30 seconds
   - Configure triggers: HTTP, S3 events
3. Enable security scanning
4. Configure monitoring and alerting
5. Start canary deployment
6. Monitor traffic distribution (10% -> 50% -> 100%)
7. Verify function execution and performance
8. Check cost optimization recommendations

**Expected Results**:
- Canary deployment progresses through stages successfully
- Function executes correctly with all triggers
- Performance metrics meet SLA requirements
- Cost optimization suggestions are provided
- Security scan passes without critical vulnerabilities
- Monitoring and alerting are properly configured

### Scenario 4: Deployment Rollback and Recovery
**Objective**: Test rollback functionality and disaster recovery

**Steps**:
1. Navigate to "History" tab
2. Identify a completed deployment
3. Click "Rollback" button
4. Confirm rollback operation
5. Monitor rollback progress
6. Verify application functionality after rollback
7. Check data integrity and preservation
8. Review rollback metrics and logs
9. Test environment status and health
10. Validate monitoring and alerting during rollback

**Expected Results**:
- Rollback completes successfully within expected timeframe
- Application functionality is fully restored
- Data integrity is maintained (no data loss)
- Environment health returns to "healthy" status
- Rollback metrics are accurately recorded
- Monitoring and alerting function correctly during rollback
- Users experience minimal service disruption

---

**Next Action Plan**: `13-security-agent-implementation.md`

**Dependencies**: 
- Core agent framework (Task 3)
- Database services (Task 2)
- API infrastructure (Task 2)
- Frontend components framework

**Estimated Completion**: 3-4 development cycles with comprehensive testing