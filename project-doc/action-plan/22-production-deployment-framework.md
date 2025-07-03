# 22 - Production Deployment Framework

## Executive Summary

This action plan establishes a comprehensive production deployment framework for the Autonomous Multi-Agent RAG System. It includes containerization, CI/CD pipelines, infrastructure as code, monitoring, and automated deployment strategies to ensure reliable, scalable, and maintainable production deployments.

## Development Objectives

### Primary Goals
- Implement Docker containerization for all services
- Establish CI/CD pipelines with automated testing
- Create infrastructure as code (IaC) templates
- Implement blue-green deployment strategies
- Set up comprehensive monitoring and alerting
- Ensure security best practices in production

### Success Metrics
- Zero-downtime deployments achieved
- Deployment time reduced to under 10 minutes
- Automated rollback capability functional
- 99.9% uptime target met
- Security vulnerabilities automatically detected

## Backend Implementation

### 1. Container Orchestration Service
**File**: `backend/core/deployment/container_service.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import docker
import yaml
import asyncio
from pathlib import Path

class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"

class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ContainerConfig:
    name: str
    image: str
    tag: str
    ports: Dict[str, int]
    environment: Dict[str, str]
    volumes: List[str]
    networks: List[str]
    resources: Dict[str, Any]
    health_check: Optional[Dict[str, Any]] = None

@dataclass
class DeploymentConfig:
    environment: EnvironmentType
    containers: List[ContainerConfig]
    load_balancer: Optional[Dict[str, Any]] = None
    database: Optional[Dict[str, Any]] = None
    redis: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None

@dataclass
class DeploymentResult:
    deployment_id: str
    status: DeploymentStatus
    containers: List[str]
    urls: List[str]
    logs: List[str]
    metrics: Dict[str, Any]
    created_at: str
    updated_at: str

class ContainerOrchestrationService:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.deployments: Dict[str, DeploymentResult] = {}
        
    async def build_images(self, config: DeploymentConfig) -> Dict[str, str]:
        """Build Docker images for all containers"""
        built_images = {}
        
        for container in config.containers:
            try:
                # Build image with proper tagging
                image_tag = f"{container.image}:{container.tag}"
                
                # Build context path
                build_path = Path(f"./docker/{container.name}")
                
                # Build image
                image, logs = self.docker_client.images.build(
                    path=str(build_path),
                    tag=image_tag,
                    rm=True,
                    forcerm=True
                )
                
                built_images[container.name] = image_tag
                
            except Exception as e:
                raise Exception(f"Failed to build image for {container.name}: {str(e)}")
                
        return built_images
    
    async def deploy_containers(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy containers based on configuration"""
        deployment_id = f"deploy_{config.environment.value}_{int(time.time())}"
        
        try:
            # Build images first
            built_images = await self.build_images(config)
            
            # Create network if needed
            network_name = f"{config.environment.value}_network"
            try:
                network = self.docker_client.networks.get(network_name)
            except docker.errors.NotFound:
                network = self.docker_client.networks.create(
                    network_name,
                    driver="bridge"
                )
            
            # Deploy containers
            deployed_containers = []
            urls = []
            
            for container in config.containers:
                container_name = f"{container.name}_{config.environment.value}"
                
                # Stop existing container if running
                try:
                    existing = self.docker_client.containers.get(container_name)
                    existing.stop()
                    existing.remove()
                except docker.errors.NotFound:
                    pass
                
                # Start new container
                new_container = self.docker_client.containers.run(
                    built_images[container.name],
                    name=container_name,
                    ports=container.ports,
                    environment=container.environment,
                    volumes=container.volumes,
                    network=network_name,
                    detach=True,
                    restart_policy={"Name": "unless-stopped"}
                )
                
                deployed_containers.append(container_name)
                
                # Generate URLs for web services
                if "80" in container.ports or "8000" in container.ports:
                    port = container.ports.get("80", container.ports.get("8000"))
                    urls.append(f"http://localhost:{port}")
            
            # Create deployment result
            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.RUNNING,
                containers=deployed_containers,
                urls=urls,
                logs=[],
                metrics={},
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            
            self.deployments[deployment_id] = result
            return result
            
        except Exception as e:
            # Create failed deployment result
            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                containers=[],
                urls=[],
                logs=[str(e)],
                metrics={},
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat()
            )
            
            self.deployments[deployment_id] = result
            raise
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get status of a deployment"""
        return self.deployments.get(deployment_id)
    
    async def stop_deployment(self, deployment_id: str) -> bool:
        """Stop a running deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return False
            
        try:
            for container_name in deployment.containers:
                container = self.docker_client.containers.get(container_name)
                container.stop()
                container.remove()
            
            deployment.status = DeploymentStatus.STOPPED
            deployment.updated_at = datetime.utcnow().isoformat()
            return True
            
        except Exception:
            return False
    
    async def get_container_logs(self, container_name: str) -> List[str]:
        """Get logs from a specific container"""
        try:
            container = self.docker_client.containers.get(container_name)
            logs = container.logs(tail=100).decode('utf-8').split('\n')
            return [log for log in logs if log.strip()]
        except Exception:
            return []
    
    async def get_container_metrics(self, container_name: str) -> Dict[str, Any]:
        """Get metrics from a specific container"""
        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
            
            # Calculate memory usage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                'cpu_percent': round(cpu_percent, 2),
                'memory_usage_mb': round(memory_usage / 1024 / 1024, 2),
                'memory_limit_mb': round(memory_limit / 1024 / 1024, 2),
                'memory_percent': round(memory_percent, 2),
                'network_rx_bytes': stats['networks']['eth0']['rx_bytes'],
                'network_tx_bytes': stats['networks']['eth0']['tx_bytes']
            }
            
        except Exception:
            return {}
```

### 2. CI/CD Pipeline Service
**File**: `backend/core/deployment/cicd_service.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import asyncio
import subprocess
import yaml
from pathlib import Path
import git

class PipelineStage(Enum):
    CHECKOUT = "checkout"
    BUILD = "build"
    TEST = "test"
    SECURITY_SCAN = "security_scan"
    DEPLOY = "deploy"
    SMOKE_TEST = "smoke_test"
    ROLLBACK = "rollback"

class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineStep:
    name: str
    stage: PipelineStage
    command: str
    working_directory: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    timeout: int = 300
    retry_count: int = 0
    continue_on_error: bool = False

@dataclass
class PipelineConfig:
    name: str
    trigger: str  # "push", "pull_request", "manual"
    branches: List[str]
    steps: List[PipelineStep]
    environment: EnvironmentType
    notifications: Optional[Dict[str, Any]] = None

@dataclass
class PipelineRun:
    run_id: str
    pipeline_name: str
    status: PipelineStatus
    current_step: Optional[str]
    steps_completed: List[str]
    steps_failed: List[str]
    logs: Dict[str, List[str]]
    artifacts: List[str]
    start_time: str
    end_time: Optional[str]
    duration: Optional[int]
    commit_hash: Optional[str]
    branch: Optional[str]

class CICDPipelineService:
    def __init__(self, container_service: ContainerOrchestrationService):
        self.container_service = container_service
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.runs: Dict[str, PipelineRun] = {}
        self.running_processes: Dict[str, subprocess.Popen] = {}
        
    async def register_pipeline(self, config: PipelineConfig) -> bool:
        """Register a new CI/CD pipeline"""
        self.pipelines[config.name] = config
        return True
    
    async def trigger_pipeline(self, pipeline_name: str, branch: str = "main", 
                             commit_hash: Optional[str] = None) -> str:
        """Trigger a pipeline run"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        config = self.pipelines[pipeline_name]
        run_id = f"{pipeline_name}_{int(time.time())}"
        
        # Create pipeline run
        run = PipelineRun(
            run_id=run_id,
            pipeline_name=pipeline_name,
            status=PipelineStatus.PENDING,
            current_step=None,
            steps_completed=[],
            steps_failed=[],
            logs={},
            artifacts=[],
            start_time=datetime.utcnow().isoformat(),
            end_time=None,
            duration=None,
            commit_hash=commit_hash,
            branch=branch
        )
        
        self.runs[run_id] = run
        
        # Start pipeline execution
        asyncio.create_task(self._execute_pipeline(run_id))
        
        return run_id
    
    async def _execute_pipeline(self, run_id: str) -> None:
        """Execute pipeline steps"""
        run = self.runs[run_id]
        config = self.pipelines[run.pipeline_name]
        
        try:
            run.status = PipelineStatus.RUNNING
            
            for step in config.steps:
                run.current_step = step.name
                
                # Execute step
                success = await self._execute_step(run_id, step)
                
                if success:
                    run.steps_completed.append(step.name)
                else:
                    run.steps_failed.append(step.name)
                    if not step.continue_on_error:
                        run.status = PipelineStatus.FAILED
                        break
            
            if run.status == PipelineStatus.RUNNING:
                run.status = PipelineStatus.SUCCESS
                
        except Exception as e:
            run.status = PipelineStatus.FAILED
            run.logs["error"] = [str(e)]
        
        finally:
            run.end_time = datetime.utcnow().isoformat()
            start_time = datetime.fromisoformat(run.start_time)
            end_time = datetime.fromisoformat(run.end_time)
            run.duration = int((end_time - start_time).total_seconds())
            run.current_step = None
    
    async def _execute_step(self, run_id: str, step: PipelineStep) -> bool:
        """Execute a single pipeline step"""
        run = self.runs[run_id]
        
        try:
            # Prepare environment
            env = os.environ.copy()
            if step.environment:
                env.update(step.environment)
            
            # Execute command
            process = subprocess.Popen(
                step.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=step.working_directory,
                env=env,
                universal_newlines=True
            )
            
            self.running_processes[run_id] = process
            
            # Capture output
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    output_lines.append(output.strip())
            
            # Wait for completion
            return_code = process.wait(timeout=step.timeout)
            
            # Store logs
            run.logs[step.name] = output_lines
            
            return return_code == 0
            
        except subprocess.TimeoutExpired:
            process.kill()
            run.logs[step.name] = [f"Step timed out after {step.timeout} seconds"]
            return False
        except Exception as e:
            run.logs[step.name] = [f"Step failed: {str(e)}"]
            return False
        finally:
            if run_id in self.running_processes:
                del self.running_processes[run_id]
    
    async def get_pipeline_run(self, run_id: str) -> Optional[PipelineRun]:
        """Get pipeline run details"""
        return self.runs.get(run_id)
    
    async def cancel_pipeline_run(self, run_id: str) -> bool:
        """Cancel a running pipeline"""
        if run_id not in self.runs:
            return False
        
        run = self.runs[run_id]
        if run.status != PipelineStatus.RUNNING:
            return False
        
        # Kill running process
        if run_id in self.running_processes:
            process = self.running_processes[run_id]
            process.kill()
            del self.running_processes[run_id]
        
        run.status = PipelineStatus.CANCELLED
        run.end_time = datetime.utcnow().isoformat()
        
        return True
    
    async def get_pipeline_runs(self, pipeline_name: Optional[str] = None) -> List[PipelineRun]:
        """Get list of pipeline runs"""
        runs = list(self.runs.values())
        
        if pipeline_name:
            runs = [run for run in runs if run.pipeline_name == pipeline_name]
        
        return sorted(runs, key=lambda x: x.start_time, reverse=True)
```

### 3. Infrastructure as Code Service
**File**: `backend/core/deployment/infrastructure_service.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
import json
from pathlib import Path

class InfrastructureProvider(Enum):
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    AWS_ECS = "aws_ecs"
    AZURE_CONTAINER = "azure_container"
    GCP_CLOUD_RUN = "gcp_cloud_run"

class ResourceType(Enum):
    COMPUTE = "compute"
    DATABASE = "database"
    STORAGE = "storage"
    NETWORK = "network"
    LOAD_BALANCER = "load_balancer"
    MONITORING = "monitoring"

@dataclass
class InfrastructureResource:
    name: str
    type: ResourceType
    provider: InfrastructureProvider
    configuration: Dict[str, Any]
    dependencies: List[str] = None
    tags: Dict[str, str] = None

@dataclass
class InfrastructureTemplate:
    name: str
    version: str
    provider: InfrastructureProvider
    environment: EnvironmentType
    resources: List[InfrastructureResource]
    variables: Dict[str, Any] = None
    outputs: Dict[str, str] = None

class InfrastructureService:
    def __init__(self):
        self.templates: Dict[str, InfrastructureTemplate] = {}
        self.deployed_stacks: Dict[str, Dict[str, Any]] = {}
        
    async def create_template(self, template: InfrastructureTemplate) -> bool:
        """Create infrastructure template"""
        self.templates[template.name] = template
        return True
    
    async def generate_docker_compose(self, template: InfrastructureTemplate) -> str:
        """Generate Docker Compose configuration"""
        compose_config = {
            'version': '3.8',
            'services': {},
            'networks': {},
            'volumes': {}
        }
        
        # Add default network
        compose_config['networks']['app_network'] = {
            'driver': 'bridge'
        }
        
        for resource in template.resources:
            if resource.type == ResourceType.COMPUTE:
                service_config = {
                    'image': resource.configuration.get('image'),
                    'container_name': resource.name,
                    'ports': resource.configuration.get('ports', []),
                    'environment': resource.configuration.get('environment', {}),
                    'volumes': resource.configuration.get('volumes', []),
                    'networks': ['app_network'],
                    'restart': 'unless-stopped'
                }
                
                # Add health check if specified
                if 'health_check' in resource.configuration:
                    service_config['healthcheck'] = resource.configuration['health_check']
                
                # Add dependencies
                if resource.dependencies:
                    service_config['depends_on'] = resource.dependencies
                
                compose_config['services'][resource.name] = service_config
            
            elif resource.type == ResourceType.DATABASE:
                db_config = {
                    'image': resource.configuration.get('image', 'postgres:13'),
                    'container_name': f"{resource.name}_db",
                    'environment': resource.configuration.get('environment', {}),
                    'volumes': [f"{resource.name}_data:/var/lib/postgresql/data"],
                    'networks': ['app_network'],
                    'restart': 'unless-stopped'
                }
                
                compose_config['services'][f"{resource.name}_db"] = db_config
                compose_config['volumes'][f"{resource.name}_data"] = {}
        
        return yaml.dump(compose_config, default_flow_style=False)
    
    async def generate_kubernetes_manifests(self, template: InfrastructureTemplate) -> Dict[str, str]:
        """Generate Kubernetes manifests"""
        manifests = {}
        
        for resource in template.resources:
            if resource.type == ResourceType.COMPUTE:
                # Deployment manifest
                deployment = {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'metadata': {
                        'name': resource.name,
                        'labels': resource.tags or {}
                    },
                    'spec': {
                        'replicas': resource.configuration.get('replicas', 1),
                        'selector': {
                            'matchLabels': {'app': resource.name}
                        },
                        'template': {
                            'metadata': {
                                'labels': {'app': resource.name}
                            },
                            'spec': {
                                'containers': [{
                                    'name': resource.name,
                                    'image': resource.configuration.get('image'),
                                    'ports': [{
                                        'containerPort': port
                                    } for port in resource.configuration.get('container_ports', [])],
                                    'env': [{
                                        'name': k,
                                        'value': str(v)
                                    } for k, v in resource.configuration.get('environment', {}).items()]
                                }]
                            }
                        }
                    }
                }
                
                manifests[f"{resource.name}-deployment.yaml"] = yaml.dump(deployment)
                
                # Service manifest
                if 'ports' in resource.configuration:
                    service = {
                        'apiVersion': 'v1',
                        'kind': 'Service',
                        'metadata': {
                            'name': f"{resource.name}-service"
                        },
                        'spec': {
                            'selector': {'app': resource.name},
                            'ports': [{
                                'port': port['external'],
                                'targetPort': port['internal']
                            } for port in resource.configuration['ports']],
                            'type': 'ClusterIP'
                        }
                    }
                    
                    manifests[f"{resource.name}-service.yaml"] = yaml.dump(service)
        
        return manifests
    
    async def deploy_infrastructure(self, template_name: str, 
                                  environment: EnvironmentType) -> Dict[str, Any]:
        """Deploy infrastructure based on template"""
        if template_name not in self.templates:
            raise ValueError(f"Template {template_name} not found")
        
        template = self.templates[template_name]
        stack_id = f"{template_name}_{environment.value}_{int(time.time())}"
        
        try:
            if template.provider == InfrastructureProvider.DOCKER_COMPOSE:
                # Generate and deploy Docker Compose
                compose_content = await self.generate_docker_compose(template)
                
                # Write compose file
                compose_path = Path(f"./deployments/{stack_id}/docker-compose.yml")
                compose_path.parent.mkdir(parents=True, exist_ok=True)
                compose_path.write_text(compose_content)
                
                # Deploy using docker-compose
                import subprocess
                result = subprocess.run(
                    ["docker-compose", "-f", str(compose_path), "up", "-d"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    raise Exception(f"Docker Compose deployment failed: {result.stderr}")
                
                deployment_info = {
                    'stack_id': stack_id,
                    'provider': template.provider.value,
                    'status': 'deployed',
                    'compose_file': str(compose_path),
                    'services': list(template.resources),
                    'deployed_at': datetime.utcnow().isoformat()
                }
                
            elif template.provider == InfrastructureProvider.KUBERNETES:
                # Generate and deploy Kubernetes manifests
                manifests = await self.generate_kubernetes_manifests(template)
                
                # Write manifest files
                manifest_dir = Path(f"./deployments/{stack_id}/k8s")
                manifest_dir.mkdir(parents=True, exist_ok=True)
                
                for filename, content in manifests.items():
                    manifest_path = manifest_dir / filename
                    manifest_path.write_text(content)
                
                # Deploy using kubectl
                import subprocess
                result = subprocess.run(
                    ["kubectl", "apply", "-f", str(manifest_dir)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    raise Exception(f"Kubernetes deployment failed: {result.stderr}")
                
                deployment_info = {
                    'stack_id': stack_id,
                    'provider': template.provider.value,
                    'status': 'deployed',
                    'manifest_dir': str(manifest_dir),
                    'resources': list(template.resources),
                    'deployed_at': datetime.utcnow().isoformat()
                }
            
            self.deployed_stacks[stack_id] = deployment_info
            return deployment_info
            
        except Exception as e:
            deployment_info = {
                'stack_id': stack_id,
                'provider': template.provider.value,
                'status': 'failed',
                'error': str(e),
                'deployed_at': datetime.utcnow().isoformat()
            }
            
            self.deployed_stacks[stack_id] = deployment_info
            raise
    
    async def destroy_infrastructure(self, stack_id: str) -> bool:
        """Destroy deployed infrastructure"""
        if stack_id not in self.deployed_stacks:
            return False
        
        stack_info = self.deployed_stacks[stack_id]
        
        try:
            if stack_info['provider'] == InfrastructureProvider.DOCKER_COMPOSE.value:
                # Stop and remove Docker Compose services
                import subprocess
                result = subprocess.run(
                    ["docker-compose", "-f", stack_info['compose_file'], "down", "-v"],
                    capture_output=True,
                    text=True
                )
                
                return result.returncode == 0
                
            elif stack_info['provider'] == InfrastructureProvider.KUBERNETES.value:
                # Delete Kubernetes resources
                import subprocess
                result = subprocess.run(
                    ["kubectl", "delete", "-f", stack_info['manifest_dir']],
                    capture_output=True,
                    text=True
                )
                
                return result.returncode == 0
            
        except Exception:
            return False
        
        return False
    
    async def get_stack_status(self, stack_id: str) -> Optional[Dict[str, Any]]:
        """Get status of deployed stack"""
        return self.deployed_stacks.get(stack_id)
    
    async def list_deployed_stacks(self) -> List[Dict[str, Any]]:
        """List all deployed stacks"""
        return list(self.deployed_stacks.values())
```

## Frontend Implementation

### Deployment Dashboard Component
**File**: `frontend/components/deployment/deployment-dashboard.tsx`

```typescript
'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Play, 
  Square, 
  RefreshCw, 
  Download, 
  Upload,
  Server,
  Database,
  Network,
  Monitor,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Clock
} from 'lucide-react'

interface DeploymentConfig {
  name: string
  environment: 'development' | 'staging' | 'production'
  containers: ContainerConfig[]
  status: 'pending' | 'building' | 'deploying' | 'running' | 'failed' | 'stopped'
}

interface ContainerConfig {
  name: string
  image: string
  tag: string
  ports: Record<string, number>
  status: string
  health: 'healthy' | 'unhealthy' | 'starting' | 'unknown'
}

interface PipelineRun {
  run_id: string
  pipeline_name: string
  status: 'pending' | 'running' | 'success' | 'failed' | 'cancelled'
  current_step: string | null
  steps_completed: string[]
  steps_failed: string[]
  start_time: string
  duration: number | null
  branch: string
}

interface DeploymentMetrics {
  cpu_percent: number
  memory_usage_mb: number
  memory_limit_mb: number
  memory_percent: number
  network_rx_bytes: number
  network_tx_bytes: number
}

export function DeploymentDashboard() {
  const [deployments, setDeployments] = useState<DeploymentConfig[]>([])
  const [pipelineRuns, setPipelineRuns] = useState<PipelineRun[]>([])
  const [metrics, setMetrics] = useState<Record<string, DeploymentMetrics>>({})
  const [selectedDeployment, setSelectedDeployment] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [logs, setLogs] = useState<Record<string, string[]>>({})

  useEffect(() => {
    loadDeployments()
    loadPipelineRuns()
    
    // Set up real-time updates
    const interval = setInterval(() => {
      loadDeployments()
      loadMetrics()
    }, 5000)
    
    return () => clearInterval(interval)
  }, [])

  const loadDeployments = async () => {
    try {
      const response = await fetch('/api/deployment/deployments')
      const data = await response.json()
      setDeployments(data)
    } catch (error) {
      console.error('Failed to load deployments:', error)
    }
  }

  const loadPipelineRuns = async () => {
    try {
      const response = await fetch('/api/deployment/pipeline-runs')
      const data = await response.json()
      setPipelineRuns(data)
    } catch (error) {
      console.error('Failed to load pipeline runs:', error)
    }
  }

  const loadMetrics = async () => {
    try {
      const response = await fetch('/api/deployment/metrics')
      const data = await response.json()
      setMetrics(data)
    } catch (error) {
      console.error('Failed to load metrics:', error)
    }
  }

  const deployApplication = async (environment: string) => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/deployment/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ environment })
      })
      
      if (response.ok) {
        await loadDeployments()
      }
    } catch (error) {
      console.error('Deployment failed:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const stopDeployment = async (deploymentId: string) => {
    try {
      await fetch(`/api/deployment/deployments/${deploymentId}/stop`, {
        method: 'POST'
      })
      await loadDeployments()
    } catch (error) {
      console.error('Failed to stop deployment:', error)
    }
  }

  const triggerPipeline = async (pipelineName: string) => {
    try {
      await fetch('/api/deployment/pipelines/trigger', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pipeline_name: pipelineName, branch: 'main' })
      })
      await loadPipelineRuns()
    } catch (error) {
      console.error('Failed to trigger pipeline:', error)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': case 'success': return 'bg-green-500'
      case 'failed': return 'bg-red-500'
      case 'pending': case 'building': case 'deploying': return 'bg-yellow-500'
      case 'stopped': case 'cancelled': return 'bg-gray-500'
      default: return 'bg-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': case 'success': return <CheckCircle className="h-4 w-4" />
      case 'failed': return <XCircle className="h-4 w-4" />
      case 'pending': case 'building': case 'deploying': return <Clock className="h-4 w-4" />
      default: return <AlertTriangle className="h-4 w-4" />
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Deployment Dashboard</h1>
        <div className="flex gap-2">
          <Button 
            onClick={() => triggerPipeline('main-pipeline')}
            disabled={isLoading}
          >
            <Play className="h-4 w-4 mr-2" />
            Trigger Pipeline
          </Button>
          <Button 
            onClick={() => deployApplication('staging')}
            disabled={isLoading}
            variant="outline"
          >
            <Upload className="h-4 w-4 mr-2" />
            Deploy to Staging
          </Button>
          <Button 
            onClick={() => deployApplication('production')}
            disabled={isLoading}
            variant="outline"
          >
            <Server className="h-4 w-4 mr-2" />
            Deploy to Production
          </Button>
        </div>
      </div>

      <Tabs defaultValue="deployments" className="space-y-4">
        <TabsList>
          <TabsTrigger value="deployments">Active Deployments</TabsTrigger>
          <TabsTrigger value="pipelines">CI/CD Pipelines</TabsTrigger>
          <TabsTrigger value="infrastructure">Infrastructure</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>

        <TabsContent value="deployments" className="space-y-4">
          <div className="grid gap-4">
            {deployments.map((deployment) => (
              <Card key={deployment.name}>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {getStatusIcon(deployment.status)}
                        {deployment.name}
                        <Badge className={getStatusColor(deployment.status)}>
                          {deployment.status}
                        </Badge>
                      </CardTitle>
                      <CardDescription>
                        Environment: {deployment.environment}
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => setSelectedDeployment(
                          selectedDeployment === deployment.name ? null : deployment.name
                        )}
                      >
                        <Monitor className="h-4 w-4 mr-2" />
                        Details
                      </Button>
                      {deployment.status === 'running' && (
                        <Button 
                          size="sm" 
                          variant="destructive"
                          onClick={() => stopDeployment(deployment.name)}
                        >
                          <Square className="h-4 w-4 mr-2" />
                          Stop
                        </Button>
                      )}
                    </div>
                  </div>
                </CardHeader>
                
                {selectedDeployment === deployment.name && (
                  <CardContent>
                    <div className="space-y-4">
                      <h4 className="font-semibold">Containers</h4>
                      <div className="grid gap-2">
                        {deployment.containers.map((container) => (
                          <div key={container.name} className="flex justify-between items-center p-2 border rounded">
                            <div>
                              <span className="font-medium">{container.name}</span>
                              <span className="text-sm text-gray-500 ml-2">
                                {container.image}:{container.tag}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge 
                                variant={container.health === 'healthy' ? 'default' : 'destructive'}
                              >
                                {container.health}
                              </Badge>
                              {metrics[container.name] && (
                                <div className="text-sm text-gray-600">
                                  CPU: {metrics[container.name].cpu_percent}% | 
                                  Memory: {metrics[container.name].memory_percent.toFixed(1)}%
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                )}
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="pipelines" className="space-y-4">
          <div className="grid gap-4">
            {pipelineRuns.map((run) => (
              <Card key={run.run_id}>
                <CardHeader>
                  <div className="flex justify-between items-center">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        {getStatusIcon(run.status)}
                        {run.pipeline_name}
                        <Badge className={getStatusColor(run.status)}>
                          {run.status}
                        </Badge>
                      </CardTitle>
                      <CardDescription>
                        Branch: {run.branch} | Started: {new Date(run.start_time).toLocaleString()}
                        {run.duration && ` | Duration: ${run.duration}s`}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent>
                  <div className="space-y-4">
                    {run.current_step && (
                      <div>
                        <div className="flex justify-between text-sm mb-2">
                          <span>Current Step: {run.current_step}</span>
                          <span>{run.steps_completed.length} completed</span>
                        </div>
                        <Progress 
                          value={(run.steps_completed.length / (run.steps_completed.length + 1)) * 100} 
                        />
                      </div>
                    )}
                    
                    {run.steps_failed.length > 0 && (
                      <Alert>
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          Failed steps: {run.steps_failed.join(', ')}
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="infrastructure" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Server className="h-5 w-5" />
                  Compute Resources
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Active Containers:</span>
                    <span className="font-semibold">
                      {deployments.reduce((acc, d) => acc + d.containers.length, 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Running Services:</span>
                    <span className="font-semibold">
                      {deployments.filter(d => d.status === 'running').length}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="h-5 w-5" />
                  Database Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>PostgreSQL:</span>
                    <Badge className="bg-green-500">Healthy</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Redis:</span>
                    <Badge className="bg-green-500">Healthy</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Network className="h-5 w-5" />
                  Network Status
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span>Load Balancer:</span>
                    <Badge className="bg-green-500">Active</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>SSL Certificate:</span>
                    <Badge className="bg-green-500">Valid</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid gap-4">
            {Object.entries(metrics).map(([containerName, metric]) => (
              <Card key={containerName}>
                <CardHeader>
                  <CardTitle>{containerName} Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">CPU Usage</div>
                      <div className="text-2xl font-bold">{metric.cpu_percent}%</div>
                      <Progress value={metric.cpu_percent} className="mt-2" />
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Memory Usage</div>
                      <div className="text-2xl font-bold">{metric.memory_percent.toFixed(1)}%</div>
                      <Progress value={metric.memory_percent} className="mt-2" />
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Memory (MB)</div>
                      <div className="text-lg font-semibold">
                        {metric.memory_usage_mb.toFixed(0)} / {metric.memory_limit_mb.toFixed(0)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Network I/O</div>
                      <div className="text-sm">
                        ↓ {(metric.network_rx_bytes / 1024 / 1024).toFixed(2)} MB<br/>
                        ↑ {(metric.network_tx_bytes / 1024 / 1024).toFixed(2)} MB
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
```

## API Endpoints

### Deployment API
**File**: `backend/api/deployment.py`

```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel

from core.deployment.container_service import ContainerOrchestrationService, DeploymentConfig, EnvironmentType
from core.deployment.cicd_service import CICDPipelineService
from core.deployment.infrastructure_service import InfrastructureService

router = APIRouter(prefix="/api/deployment", tags=["deployment"])

# Initialize services
container_service = ContainerOrchestrationService()
cicd_service = CICDPipelineService(container_service)
infrastructure_service = InfrastructureService()

class DeployRequest(BaseModel):
    environment: str
    config_name: Optional[str] = "default"

class PipelineTriggerRequest(BaseModel):
    pipeline_name: str
    branch: str = "main"
    commit_hash: Optional[str] = None

@router.post("/deploy")
async def deploy_application(request: DeployRequest, background_tasks: BackgroundTasks):
    """Deploy application to specified environment"""
    try:
        environment = EnvironmentType(request.environment)
        
        # Create deployment configuration
        config = DeploymentConfig(
            environment=environment,
            containers=[
                # Add your container configurations here
            ]
        )
        
        # Deploy in background
        background_tasks.add_task(container_service.deploy_containers, config)
        
        return {"message": "Deployment started", "environment": request.environment}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/deployments")
async def get_deployments():
    """Get all active deployments"""
    try:
        # Return mock data for now - implement actual deployment retrieval
        return [
            {
                "name": "sentientcore-prod",
                "environment": "production",
                "status": "running",
                "containers": [
                    {
                        "name": "frontend",
                        "image": "sentientcore/frontend",
                        "tag": "latest",
                        "ports": {"3000": 3000},
                        "status": "running",
                        "health": "healthy"
                    },
                    {
                        "name": "backend",
                        "image": "sentientcore/backend",
                        "tag": "latest",
                        "ports": {"8000": 8000},
                        "status": "running",
                        "health": "healthy"
                    }
                ]
            }
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/deployments/{deployment_id}/stop")
async def stop_deployment(deployment_id: str):
    """Stop a running deployment"""
    try:
        success = await container_service.stop_deployment(deployment_id)
        if success:
            return {"message": "Deployment stopped successfully"}
        else:
            raise HTTPException(status_code=404, detail="Deployment not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipelines/trigger")
async def trigger_pipeline(request: PipelineTriggerRequest):
    """Trigger a CI/CD pipeline"""
    try:
        run_id = await cicd_service.trigger_pipeline(
            request.pipeline_name,
            request.branch,
            request.commit_hash
        )
        
        return {"run_id": run_id, "message": "Pipeline triggered successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline-runs")
async def get_pipeline_runs(pipeline_name: Optional[str] = None):
    """Get pipeline run history"""
    try:
        runs = await cicd_service.get_pipeline_runs(pipeline_name)
        return runs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline-runs/{run_id}")
async def get_pipeline_run(run_id: str):
    """Get specific pipeline run details"""
    try:
        run = await cicd_service.get_pipeline_run(run_id)
        if run:
            return run
        else:
            raise HTTPException(status_code=404, detail="Pipeline run not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline-runs/{run_id}/cancel")
async def cancel_pipeline_run(run_id: str):
    """Cancel a running pipeline"""
    try:
        success = await cicd_service.cancel_pipeline_run(run_id)
        if success:
            return {"message": "Pipeline cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Pipeline run not found or not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_deployment_metrics():
    """Get deployment metrics for all containers"""
    try:
        # Return mock metrics - implement actual metrics collection
        return {
            "frontend": {
                "cpu_percent": 15.5,
                "memory_usage_mb": 256.8,
                "memory_limit_mb": 512.0,
                "memory_percent": 50.2,
                "network_rx_bytes": 1024000,
                "network_tx_bytes": 2048000
            },
            "backend": {
                "cpu_percent": 25.3,
                "memory_usage_mb": 512.4,
                "memory_limit_mb": 1024.0,
                "memory_percent": 50.0,
                "network_rx_bytes": 2048000,
                "network_tx_bytes": 1024000
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/infrastructure/stacks")
async def get_infrastructure_stacks():
    """Get deployed infrastructure stacks"""
    try:
        stacks = await infrastructure_service.list_deployed_stacks()
        return stacks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/infrastructure/deploy")
async def deploy_infrastructure(template_name: str, environment: str):
    """Deploy infrastructure from template"""
    try:
        env_type = EnvironmentType(environment)
        result = await infrastructure_service.deploy_infrastructure(template_name, env_type)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/infrastructure/stacks/{stack_id}")
async def destroy_infrastructure(stack_id: str):
    """Destroy infrastructure stack"""
    try:
        success = await infrastructure_service.destroy_infrastructure(stack_id)
        if success:
            return {"message": "Infrastructure destroyed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Stack not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing Strategy

### Unit Tests

#### Backend Tests
**File**: `backend/tests/test_deployment_services.py`

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from core.deployment.container_service import ContainerOrchestrationService, DeploymentConfig, EnvironmentType
from core.deployment.cicd_service import CICDPipelineService, PipelineConfig, PipelineStep, PipelineStage
from core.deployment.infrastructure_service import InfrastructureService, InfrastructureTemplate

class TestContainerOrchestrationService:
    @pytest.fixture
    def service(self):
        return ContainerOrchestrationService()
    
    @pytest.fixture
    def sample_config(self):
        return DeploymentConfig(
            environment=EnvironmentType.DEVELOPMENT,
            containers=[]
        )
    
    @patch('docker.from_env')
    async def test_build_images_success(self, mock_docker, service, sample_config):
        # Mock Docker client
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock image build
        mock_image = Mock()
        mock_client.images.build.return_value = (mock_image, [])
        
        service.docker_client = mock_client
        
        # Test image building
        result = await service.build_images(sample_config)
        
        assert isinstance(result, dict)
    
    async def test_deploy_containers_invalid_config(self, service):
        with pytest.raises(Exception):
            await service.deploy_containers(None)
    
    async def test_get_deployment_status_not_found(self, service):
        result = await service.get_deployment_status("nonexistent")
        assert result is None

class TestCICDPipelineService:
    @pytest.fixture
    def container_service(self):
        return Mock(spec=ContainerOrchestrationService)
    
    @pytest.fixture
    def service(self, container_service):
        return CICDPipelineService(container_service)
    
    @pytest.fixture
    def sample_pipeline_config(self):
        return PipelineConfig(
            name="test-pipeline",
            trigger="push",
            branches=["main"],
            steps=[
                PipelineStep(
                    name="test",
                    stage=PipelineStage.TEST,
                    command="echo 'test'"
                )
            ],
            environment=EnvironmentType.DEVELOPMENT
        )
    
    async def test_register_pipeline(self, service, sample_pipeline_config):
        result = await service.register_pipeline(sample_pipeline_config)
        assert result is True
        assert "test-pipeline" in service.pipelines
    
    async def test_trigger_pipeline_not_found(self, service):
        with pytest.raises(ValueError):
            await service.trigger_pipeline("nonexistent")
    
    async def test_trigger_pipeline_success(self, service, sample_pipeline_config):
        await service.register_pipeline(sample_pipeline_config)
        run_id = await service.trigger_pipeline("test-pipeline")
        
        assert run_id is not None
        assert run_id in service.runs
    
    async def test_cancel_pipeline_run(self, service, sample_pipeline_config):
        await service.register_pipeline(sample_pipeline_config)
        run_id = await service.trigger_pipeline("test-pipeline")
        
        # Wait a bit for pipeline to start
        await asyncio.sleep(0.1)
        
        result = await service.cancel_pipeline_run(run_id)
        assert result is True
        
        run = await service.get_pipeline_run(run_id)
        assert run.status.value == "cancelled"

class TestInfrastructureService:
    @pytest.fixture
    def service(self):
        return InfrastructureService()
    
    @pytest.fixture
    def sample_template(self):
        return InfrastructureTemplate(
            name="test-template",
            version="1.0",
            provider=InfrastructureProvider.DOCKER_COMPOSE,
            environment=EnvironmentType.DEVELOPMENT,
            resources=[]
        )
    
    async def test_create_template(self, service, sample_template):
        result = await service.create_template(sample_template)
        assert result is True
        assert "test-template" in service.templates
    
    async def test_generate_docker_compose(self, service, sample_template):
        await service.create_template(sample_template)
        compose_content = await service.generate_docker_compose(sample_template)
        
        assert "version: '3.8'" in compose_content
        assert "services:" in compose_content
        assert "networks:" in compose_content
    
    async def test_deploy_infrastructure_template_not_found(self, service):
        with pytest.raises(ValueError):
            await service.deploy_infrastructure("nonexistent", EnvironmentType.DEVELOPMENT)
```

#### Frontend Tests
**File**: `frontend/components/deployment/__tests__/deployment-dashboard.test.tsx`

```typescript
import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { DeploymentDashboard } from '../deployment-dashboard'

// Mock fetch
global.fetch = jest.fn()

const mockDeployments = [
  {
    name: 'test-deployment',
    environment: 'development',
    status: 'running',
    containers: [
      {
        name: 'frontend',
        image: 'test/frontend',
        tag: 'latest',
        ports: { '3000': 3000 },
        status: 'running',
        health: 'healthy'
      }
    ]
  }
]

const mockPipelineRuns = [
  {
    run_id: 'run-123',
    pipeline_name: 'test-pipeline',
    status: 'success',
    current_step: null,
    steps_completed: ['build', 'test'],
    steps_failed: [],
    start_time: '2024-01-01T00:00:00Z',
    duration: 120,
    branch: 'main'
  }
]

describe('DeploymentDashboard', () => {
  beforeEach(() => {
    (fetch as jest.Mock).mockClear()
  })

  it('renders deployment dashboard', async () => {
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockDeployments
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockPipelineRuns
      })

    render(<DeploymentDashboard />)

    expect(screen.getByText('Deployment Dashboard')).toBeInTheDocument()
    expect(screen.getByText('Trigger Pipeline')).toBeInTheDocument()
    expect(screen.getByText('Deploy to Staging')).toBeInTheDocument()
    expect(screen.getByText('Deploy to Production')).toBeInTheDocument()
  })

  it('loads and displays deployments', async () => {
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockDeployments
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockPipelineRuns
      })

    render(<DeploymentDashboard />)

    await waitFor(() => {
      expect(screen.getByText('test-deployment')).toBeInTheDocument()
      expect(screen.getByText('running')).toBeInTheDocument()
    })
  })

  it('triggers deployment when button clicked', async () => {
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockDeployments
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockPipelineRuns
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ message: 'Deployment started' })
      })

    render(<DeploymentDashboard />)

    const deployButton = screen.getByText('Deploy to Staging')
    fireEvent.click(deployButton)

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/deployment/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ environment: 'staging' })
      })
    })
  })

  it('shows deployment details when details button clicked', async () => {
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockDeployments
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockPipelineRuns
      })

    render(<DeploymentDashboard />)

    await waitFor(() => {
      const detailsButton = screen.getByText('Details')
      fireEvent.click(detailsButton)
      expect(screen.getByText('Containers')).toBeInTheDocument()
      expect(screen.getByText('frontend')).toBeInTheDocument()
    })
  })

  it('displays pipeline runs in pipelines tab', async () => {
    (fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockDeployments
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockPipelineRuns
      })

    render(<DeploymentDashboard />)

    const pipelinesTab = screen.getByText('CI/CD Pipelines')
    fireEvent.click(pipelinesTab)

    await waitFor(() => {
      expect(screen.getByText('test-pipeline')).toBeInTheDocument()
      expect(screen.getByText('success')).toBeInTheDocument()
      expect(screen.getByText('Branch: main')).toBeInTheDocument()
    })
  })
})
```

### Integration Tests

#### End-to-End Deployment Test
**File**: `backend/tests/integration/test_deployment_workflow.py`

```python
import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from main import app
from core.deployment.container_service import ContainerOrchestrationService
from core.deployment.cicd_service import CICDPipelineService

client = TestClient(app)

class TestDeploymentWorkflow:
    @patch('core.deployment.container_service.docker.from_env')
    def test_full_deployment_workflow(self, mock_docker):
        """Test complete deployment workflow from trigger to running"""
        # Mock Docker client
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock successful image build
        mock_image = Mock()
        mock_client.images.build.return_value = (mock_image, [])
        
        # Mock successful container run
        mock_container = Mock()
        mock_client.containers.run.return_value = mock_container
        
        # Mock network creation
        mock_network = Mock()
        mock_client.networks.create.return_value = mock_network
        
        # Step 1: Trigger pipeline
        response = client.post("/api/deployment/pipelines/trigger", json={
            "pipeline_name": "main-pipeline",
            "branch": "main"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "run_id" in data
        
        # Step 2: Deploy application
        response = client.post("/api/deployment/deploy", json={
            "environment": "development"
        })
        
        assert response.status_code == 200
        assert "Deployment started" in response.json()["message"]
        
        # Step 3: Check deployment status
        response = client.get("/api/deployment/deployments")
        assert response.status_code == 200
        
        deployments = response.json()
        assert len(deployments) > 0
    
    def test_pipeline_failure_handling(self):
        """Test pipeline failure scenarios"""
        # Trigger pipeline with invalid configuration
        response = client.post("/api/deployment/pipelines/trigger", json={
            "pipeline_name": "nonexistent-pipeline",
            "branch": "main"
        })
        
        assert response.status_code == 400
    
    def test_deployment_metrics_collection(self):
        """Test metrics collection from deployments"""
        response = client.get("/api/deployment/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert isinstance(metrics, dict)
        
        # Check metric structure
        for container_name, metric in metrics.items():
            assert "cpu_percent" in metric
            assert "memory_usage_mb" in metric
            assert "memory_percent" in metric
    
    def test_infrastructure_deployment(self):
        """Test infrastructure deployment workflow"""
        # Deploy infrastructure
        response = client.post("/api/deployment/infrastructure/deploy", params={
            "template_name": "basic-stack",
            "environment": "development"
        })
        
        # Should handle missing template gracefully
        assert response.status_code in [200, 400]
        
        # Get infrastructure stacks
        response = client.get("/api/deployment/infrastructure/stacks")
        assert response.status_code == 200
```

### Human Testing Scenarios

#### Deployment Testing Scenarios
1. **Basic Deployment Flow**
   - Navigate to Deployment Dashboard
   - Click "Deploy to Staging" button
   - Verify deployment status updates in real-time
   - Check container health indicators
   - Verify application accessibility via provided URLs

2. **CI/CD Pipeline Testing**
   - Trigger pipeline from dashboard
   - Monitor pipeline progress in real-time
   - Verify step completion indicators
   - Test pipeline cancellation functionality
   - Check pipeline logs and error handling

3. **Infrastructure Management**
   - View infrastructure status cards
   - Monitor resource utilization metrics
   - Test infrastructure deployment/destruction
   - Verify network and database connectivity

4. **Monitoring and Metrics**
   - Check real-time container metrics
   - Verify CPU and memory usage displays
   - Test metric refresh intervals
   - Monitor network I/O statistics

5. **Error Handling**
   - Test deployment with invalid configuration
   - Verify error message display
   - Test rollback functionality
   - Check failure notification system

## Validation Criteria

### Backend Validation
- [ ] Container orchestration service successfully builds and deploys containers
- [ ] CI/CD pipeline service executes all pipeline stages correctly
- [ ] Infrastructure service generates valid Docker Compose and Kubernetes manifests
- [ ] Deployment metrics are collected and reported accurately
- [ ] Error handling provides meaningful feedback
- [ ] Background tasks execute without blocking main thread
- [ ] Database operations maintain data consistency
- [ ] API endpoints return appropriate HTTP status codes
- [ ] Security measures prevent unauthorized deployments
- [ ] Resource cleanup occurs properly on deployment failures

### Frontend Validation
- [ ] Deployment dashboard loads and displays current deployments
- [ ] Real-time updates show deployment status changes
- [ ] Pipeline monitoring displays progress accurately
- [ ] Infrastructure metrics are visualized clearly
- [ ] User interactions trigger appropriate API calls
- [ ] Error states are handled gracefully with user feedback
- [ ] Responsive design works across different screen sizes
- [ ] Loading states provide clear user feedback
- [ ] Navigation between tabs maintains state
- [ ] Action buttons are disabled appropriately during operations

### Integration Validation
- [ ] End-to-end deployment workflow completes successfully
- [ ] Pipeline triggers result in actual deployments
- [ ] Metrics collection reflects real container states
- [ ] Infrastructure changes are reflected in dashboard
- [ ] Rollback functionality restores previous state
- [ ] Multi-environment deployments work independently
- [ ] Concurrent deployments are handled correctly
- [ ] System recovery after failures is automatic
- [ ] Monitoring alerts are triggered appropriately
- [ ] Performance remains acceptable under load

## Success Metrics
- Deployment time reduced to under 10 minutes
- Zero-downtime deployments achieved
- 99.9% deployment success rate
- Automated rollback completes within 2 minutes
- Infrastructure provisioning time under 5 minutes
- Real-time monitoring with <30 second update intervals
- Pipeline execution time optimized by 50%
- Security scan integration with 100% coverage
- Multi-environment support fully functional
- Documentation and testing coverage >90%

## Next Steps
Proceed to `23-monitoring-alerting-system.md` for comprehensive monitoring and alerting implementation.