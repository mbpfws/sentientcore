# 14 - External Service Integrations

## Overview

The External Service Integrations module provides comprehensive connectivity to external APIs, services, and third-party tools. This includes AI model providers, development tools, cloud services, databases, and other external systems that enhance the multi-agent platform's capabilities.

## Current State Analysis

### Integration Requirements
- AI model provider integrations (OpenAI, Anthropic, Google, etc.)
- Development tool integrations (GitHub, GitLab, Jira, etc.)
- Cloud service integrations (AWS, Azure, GCP)
- Database service integrations (MongoDB, PostgreSQL, Redis)
- Communication service integrations (Slack, Discord, Email)
- Monitoring and analytics integrations

### Security Considerations
- API key management and rotation
- OAuth 2.0 and authentication flows
- Rate limiting and quota management
- Data privacy and compliance
- Secure credential storage

## Implementation Tasks

### Task 14.1: Core Integration Framework

**File**: `core/integrations/base.py`

**Base Integration Framework**:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
import json
from enum import Enum

class IntegrationStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    UNAUTHORIZED = "unauthorized"

@dataclass
class IntegrationConfig:
    name: str
    service_type: str
    base_url: str
    api_version: str
    authentication: Dict[str, Any]
    rate_limits: Dict[str, int]
    timeout: int = 30
    retry_attempts: int = 3
    enabled: bool = True

@dataclass
class IntegrationMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    rate_limit_hits: int = 0
    quota_usage: Dict[str, Any] = None

class BaseIntegration(ABC):
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = IntegrationStatus.INACTIVE
        self.metrics = IntegrationMetrics()
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter(config.rate_limits)
        
    async def initialize(self):
        """Initialize the integration"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            await self.authenticate()
            await self.validate_connection()
            self.status = IntegrationStatus.ACTIVE
        except Exception as e:
            self.status = IntegrationStatus.ERROR
            raise e
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def authenticate(self):
        """Authenticate with the external service"""
        pass
    
    @abstractmethod
    async def validate_connection(self):
        """Validate connection to the service"""
        pass
    
    async def make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make authenticated request to external service"""
        if not self.session:
            raise RuntimeError("Integration not initialized")
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        url = f"{self.config.base_url}/{endpoint.lstrip('/')}"
        request_headers = await self.get_auth_headers()
        if headers:
            request_headers.update(headers)
        
        start_time = datetime.utcnow()
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                headers=request_headers
            ) as response:
                response_data = await response.json()
                
                # Update metrics
                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds()
                self.update_metrics(True, response_time)
                
                if response.status == 429:  # Rate limited
                    self.status = IntegrationStatus.RATE_LIMITED
                    self.metrics.rate_limit_hits += 1
                    raise RateLimitError("Rate limit exceeded")
                
                response.raise_for_status()
                return response_data
                
        except Exception as e:
            self.update_metrics(False, 0)
            if "unauthorized" in str(e).lower():
                self.status = IntegrationStatus.UNAUTHORIZED
            raise e
    
    @abstractmethod
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests"""
        pass
    
    def update_metrics(self, success: bool, response_time: float):
        """Update integration metrics"""
        self.metrics.total_requests += 1
        self.metrics.last_request_time = datetime.utcnow()
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        if self.metrics.total_requests == 1:
            self.metrics.average_response_time = response_time
        else:
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_requests - 1) + response_time) 
                / self.metrics.total_requests
            )

class RateLimiter:
    def __init__(self, limits: Dict[str, int]):
        self.limits = limits
        self.requests = {}
    
    async def acquire(self):
        """Acquire rate limit token"""
        current_time = datetime.utcnow()
        
        # Simple rate limiting implementation
        for period, limit in self.limits.items():
            if period not in self.requests:
                self.requests[period] = []
            
            # Remove old requests
            cutoff_time = current_time - self.get_period_timedelta(period)
            self.requests[period] = [
                req_time for req_time in self.requests[period] 
                if req_time > cutoff_time
            ]
            
            # Check if we can make a request
            if len(self.requests[period]) >= limit:
                sleep_time = (self.requests[period][0] - cutoff_time).total_seconds()
                await asyncio.sleep(sleep_time)
            
            self.requests[period].append(current_time)
    
    def get_period_timedelta(self, period: str):
        """Convert period string to timedelta"""
        from datetime import timedelta
        
        if period == "minute":
            return timedelta(minutes=1)
        elif period == "hour":
            return timedelta(hours=1)
        elif period == "day":
            return timedelta(days=1)
        else:
            return timedelta(minutes=1)

class RateLimitError(Exception):
    pass
```

### Task 14.2: AI Model Provider Integrations

**File**: `core/integrations/ai_providers.py`

**AI Provider Integrations**:
```python
from core.integrations.base import BaseIntegration, IntegrationConfig
from typing import Dict, Any, List, Optional
import json

class OpenAIIntegration(BaseIntegration):
    def __init__(self, api_key: str, organization: Optional[str] = None):
        config = IntegrationConfig(
            name="openai",
            service_type="ai_provider",
            base_url="https://api.openai.com/v1",
            api_version="v1",
            authentication={"api_key": api_key, "organization": organization},
            rate_limits={"minute": 60, "hour": 3600}
        )
        super().__init__(config)
        self.api_key = api_key
        self.organization = organization
    
    async def authenticate(self):
        """Authenticate with OpenAI API"""
        # OpenAI uses API key authentication
        pass
    
    async def validate_connection(self):
        """Validate connection to OpenAI"""
        try:
            await self.make_request("GET", "models")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to OpenAI: {e}")
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get OpenAI authentication headers"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers
    
    async def create_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Create chat completion"""
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        return await self.make_request("POST", "chat/completions", data)
    
    async def create_embedding(
        self, 
        text: str, 
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """Create text embedding"""
        data = {
            "input": text,
            "model": model
        }
        return await self.make_request("POST", "embeddings", data)
    
    async def list_models(self) -> Dict[str, Any]:
        """List available models"""
        return await self.make_request("GET", "models")

class AnthropicIntegration(BaseIntegration):
    def __init__(self, api_key: str):
        config = IntegrationConfig(
            name="anthropic",
            service_type="ai_provider",
            base_url="https://api.anthropic.com/v1",
            api_version="v1",
            authentication={"api_key": api_key},
            rate_limits={"minute": 60, "hour": 1000}
        )
        super().__init__(config)
        self.api_key = api_key
    
    async def authenticate(self):
        """Authenticate with Anthropic API"""
        pass
    
    async def validate_connection(self):
        """Validate connection to Anthropic"""
        # Anthropic doesn't have a simple health check endpoint
        # We'll validate during first actual request
        pass
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get Anthropic authentication headers"""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def create_message(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Create message with Claude"""
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            **kwargs
        }
        return await self.make_request("POST", "messages", data)

class GoogleAIIntegration(BaseIntegration):
    def __init__(self, api_key: str):
        config = IntegrationConfig(
            name="google_ai",
            service_type="ai_provider",
            base_url="https://generativelanguage.googleapis.com/v1",
            api_version="v1",
            authentication={"api_key": api_key},
            rate_limits={"minute": 60, "hour": 1500}
        )
        super().__init__(config)
        self.api_key = api_key
    
    async def authenticate(self):
        """Authenticate with Google AI API"""
        pass
    
    async def validate_connection(self):
        """Validate connection to Google AI"""
        try:
            await self.make_request("GET", "models")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Google AI: {e}")
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get Google AI authentication headers"""
        return {
            "Content-Type": "application/json"
        }
    
    async def generate_content(
        self, 
        model: str, 
        prompt: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate content with Gemini"""
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            **kwargs
        }
        endpoint = f"models/{model}:generateContent?key={self.api_key}"
        return await self.make_request("POST", endpoint, data)
```

### Task 14.3: Development Tool Integrations

**File**: `core/integrations/dev_tools.py`

**Development Tool Integrations**:
```python
from core.integrations.base import BaseIntegration, IntegrationConfig
from typing import Dict, Any, List, Optional
import base64

class GitHubIntegration(BaseIntegration):
    def __init__(self, token: str, username: Optional[str] = None):
        config = IntegrationConfig(
            name="github",
            service_type="dev_tool",
            base_url="https://api.github.com",
            api_version="v3",
            authentication={"token": token, "username": username},
            rate_limits={"hour": 5000}
        )
        super().__init__(config)
        self.token = token
        self.username = username
    
    async def authenticate(self):
        """Authenticate with GitHub API"""
        pass
    
    async def validate_connection(self):
        """Validate connection to GitHub"""
        try:
            await self.make_request("GET", "user")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GitHub: {e}")
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get GitHub authentication headers"""
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SentientCore-Agent"
        }
    
    async def create_repository(
        self, 
        name: str, 
        description: str = "", 
        private: bool = False
    ) -> Dict[str, Any]:
        """Create a new repository"""
        data = {
            "name": name,
            "description": description,
            "private": private
        }
        return await self.make_request("POST", "user/repos", data)
    
    async def create_file(
        self, 
        repo: str, 
        path: str, 
        content: str, 
        message: str,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """Create or update a file in repository"""
        encoded_content = base64.b64encode(content.encode()).decode()
        data = {
            "message": message,
            "content": encoded_content,
            "branch": branch
        }
        endpoint = f"repos/{repo}/contents/{path}"
        return await self.make_request("PUT", endpoint, data)
    
    async def create_pull_request(
        self, 
        repo: str, 
        title: str, 
        head: str, 
        base: str, 
        body: str = ""
    ) -> Dict[str, Any]:
        """Create a pull request"""
        data = {
            "title": title,
            "head": head,
            "base": base,
            "body": body
        }
        endpoint = f"repos/{repo}/pulls"
        return await self.make_request("POST", endpoint, data)
    
    async def create_issue(
        self, 
        repo: str, 
        title: str, 
        body: str = "", 
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """Create an issue"""
        data = {
            "title": title,
            "body": body
        }
        if labels:
            data["labels"] = labels
        
        endpoint = f"repos/{repo}/issues"
        return await self.make_request("POST", endpoint, data)

class JiraIntegration(BaseIntegration):
    def __init__(self, base_url: str, username: str, api_token: str):
        config = IntegrationConfig(
            name="jira",
            service_type="dev_tool",
            base_url=f"{base_url}/rest/api/3",
            api_version="3",
            authentication={"username": username, "api_token": api_token},
            rate_limits={"minute": 100}
        )
        super().__init__(config)
        self.username = username
        self.api_token = api_token
    
    async def authenticate(self):
        """Authenticate with Jira API"""
        pass
    
    async def validate_connection(self):
        """Validate connection to Jira"""
        try:
            await self.make_request("GET", "myself")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Jira: {e}")
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get Jira authentication headers"""
        auth_string = f"{self.username}:{self.api_token}"
        encoded_auth = base64.b64encode(auth_string.encode()).decode()
        return {
            "Authorization": f"Basic {encoded_auth}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    async def create_issue(
        self, 
        project_key: str, 
        summary: str, 
        description: str, 
        issue_type: str = "Task"
    ) -> Dict[str, Any]:
        """Create a Jira issue"""
        data = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": description}
                            ]
                        }
                    ]
                },
                "issuetype": {"name": issue_type}
            }
        }
        return await self.make_request("POST", "issue", data)
    
    async def update_issue(
        self, 
        issue_key: str, 
        fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update a Jira issue"""
        data = {"fields": fields}
        endpoint = f"issue/{issue_key}"
        return await self.make_request("PUT", endpoint, data)
```

### Task 14.4: Cloud Service Integrations

**File**: `core/integrations/cloud_services.py`

**Cloud Service Integrations**:
```python
from core.integrations.base import BaseIntegration, IntegrationConfig
from typing import Dict, Any, List, Optional
import boto3
from azure.identity import DefaultAzureCredential
from google.cloud import storage as gcs

class AWSIntegration(BaseIntegration):
    def __init__(self, access_key: str, secret_key: str, region: str = "us-east-1"):
        config = IntegrationConfig(
            name="aws",
            service_type="cloud_provider",
            base_url=f"https://s3.{region}.amazonaws.com",
            api_version="v4",
            authentication={
                "access_key": access_key,
                "secret_key": secret_key,
                "region": region
            },
            rate_limits={"second": 100}
        )
        super().__init__(config)
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.s3_client = None
        self.ec2_client = None
    
    async def authenticate(self):
        """Authenticate with AWS"""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
        self.ec2_client = boto3.client(
            'ec2',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
    
    async def validate_connection(self):
        """Validate connection to AWS"""
        try:
            self.s3_client.list_buckets()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to AWS: {e}")
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get AWS authentication headers"""
        # AWS uses signature-based authentication
        return {}
    
    async def create_s3_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """Create S3 bucket"""
        try:
            if self.region != 'us-east-1':
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            else:
                self.s3_client.create_bucket(Bucket=bucket_name)
            return {'status': 'success', 'bucket': bucket_name}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def upload_file_to_s3(
        self, 
        bucket_name: str, 
        file_path: str, 
        object_key: str
    ) -> Dict[str, Any]:
        """Upload file to S3"""
        try:
            self.s3_client.upload_file(file_path, bucket_name, object_key)
            return {'status': 'success', 'object_key': object_key}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def launch_ec2_instance(
        self, 
        image_id: str, 
        instance_type: str = "t2.micro",
        key_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Launch EC2 instance"""
        try:
            params = {
                'ImageId': image_id,
                'MinCount': 1,
                'MaxCount': 1,
                'InstanceType': instance_type
            }
            if key_name:
                params['KeyName'] = key_name
            
            response = self.ec2_client.run_instances(**params)
            return {
                'status': 'success', 
                'instance_id': response['Instances'][0]['InstanceId']
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

class AzureIntegration(BaseIntegration):
    def __init__(self, subscription_id: str, resource_group: str):
        config = IntegrationConfig(
            name="azure",
            service_type="cloud_provider",
            base_url="https://management.azure.com",
            api_version="2021-04-01",
            authentication={
                "subscription_id": subscription_id,
                "resource_group": resource_group
            },
            rate_limits={"minute": 1000}
        )
        super().__init__(config)
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.credential = None
    
    async def authenticate(self):
        """Authenticate with Azure"""
        self.credential = DefaultAzureCredential()
    
    async def validate_connection(self):
        """Validate connection to Azure"""
        # Azure validation would require specific service calls
        pass
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get Azure authentication headers"""
        token = self.credential.get_token("https://management.azure.com/.default")
        return {
            "Authorization": f"Bearer {token.token}",
            "Content-Type": "application/json"
        }

class GCPIntegration(BaseIntegration):
    def __init__(self, project_id: str, credentials_path: str):
        config = IntegrationConfig(
            name="gcp",
            service_type="cloud_provider",
            base_url="https://storage.googleapis.com/storage/v1",
            api_version="v1",
            authentication={
                "project_id": project_id,
                "credentials_path": credentials_path
            },
            rate_limits={"second": 100}
        )
        super().__init__(config)
        self.project_id = project_id
        self.credentials_path = credentials_path
        self.storage_client = None
    
    async def authenticate(self):
        """Authenticate with GCP"""
        self.storage_client = gcs.Client.from_service_account_json(
            self.credentials_path,
            project=self.project_id
        )
    
    async def validate_connection(self):
        """Validate connection to GCP"""
        try:
            list(self.storage_client.list_buckets())
        except Exception as e:
            raise ConnectionError(f"Failed to connect to GCP: {e}")
    
    async def get_auth_headers(self) -> Dict[str, str]:
        """Get GCP authentication headers"""
        # GCP uses service account authentication
        return {}
    
    async def create_storage_bucket(self, bucket_name: str) -> Dict[str, Any]:
        """Create GCS bucket"""
        try:
            bucket = self.storage_client.create_bucket(bucket_name)
            return {'status': 'success', 'bucket': bucket.name}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
```

### Task 14.5: Integration Management Service

**File**: `core/services/integration_service.py`

**Integration Management Service**:
```python
from typing import Dict, Any, List, Optional
from core.integrations.base import BaseIntegration, IntegrationConfig, IntegrationStatus
from core.integrations.ai_providers import OpenAIIntegration, AnthropicIntegration, GoogleAIIntegration
from core.integrations.dev_tools import GitHubIntegration, JiraIntegration
from core.integrations.cloud_services import AWSIntegration, AzureIntegration, GCPIntegration
import asyncio
from datetime import datetime

class IntegrationService:
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.integration_configs: Dict[str, IntegrationConfig] = {}
        self.health_check_interval = 300  # 5 minutes
        self.health_check_task = None
    
    async def initialize(self):
        """Initialize the integration service"""
        # Load integration configurations
        await self.load_configurations()
        
        # Start health check task
        self.health_check_task = asyncio.create_task(self.health_check_loop())
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.health_check_task:
            self.health_check_task.cancel()
        
        for integration in self.integrations.values():
            await integration.cleanup()
    
    async def load_configurations(self):
        """Load integration configurations from database/config"""
        # This would typically load from database or config files
        # For now, we'll use default configurations
        pass
    
    async def register_integration(
        self, 
        integration_type: str, 
        config: Dict[str, Any]
    ) -> str:
        """Register a new integration"""
        integration_name = config.get('name', integration_type)
        
        try:
            if integration_type == 'openai':
                integration = OpenAIIntegration(
                    api_key=config['api_key'],
                    organization=config.get('organization')
                )
            elif integration_type == 'anthropic':
                integration = AnthropicIntegration(api_key=config['api_key'])
            elif integration_type == 'google_ai':
                integration = GoogleAIIntegration(api_key=config['api_key'])
            elif integration_type == 'github':
                integration = GitHubIntegration(
                    token=config['token'],
                    username=config.get('username')
                )
            elif integration_type == 'jira':
                integration = JiraIntegration(
                    base_url=config['base_url'],
                    username=config['username'],
                    api_token=config['api_token']
                )
            elif integration_type == 'aws':
                integration = AWSIntegration(
                    access_key=config['access_key'],
                    secret_key=config['secret_key'],
                    region=config.get('region', 'us-east-1')
                )
            elif integration_type == 'azure':
                integration = AzureIntegration(
                    subscription_id=config['subscription_id'],
                    resource_group=config['resource_group']
                )
            elif integration_type == 'gcp':
                integration = GCPIntegration(
                    project_id=config['project_id'],
                    credentials_path=config['credentials_path']
                )
            else:
                raise ValueError(f"Unknown integration type: {integration_type}")
            
            await integration.initialize()
            self.integrations[integration_name] = integration
            self.integration_configs[integration_name] = integration.config
            
            return integration_name
            
        except Exception as e:
            raise RuntimeError(f"Failed to register integration {integration_name}: {e}")
    
    async def get_integration(self, name: str) -> Optional[BaseIntegration]:
        """Get integration by name"""
        return self.integrations.get(name)
    
    async def list_integrations(self) -> List[Dict[str, Any]]:
        """List all registered integrations"""
        integrations = []
        for name, integration in self.integrations.items():
            integrations.append({
                'name': name,
                'type': integration.config.service_type,
                'status': integration.status.value,
                'metrics': {
                    'total_requests': integration.metrics.total_requests,
                    'success_rate': (
                        integration.metrics.successful_requests / integration.metrics.total_requests
                        if integration.metrics.total_requests > 0 else 0
                    ),
                    'average_response_time': integration.metrics.average_response_time,
                    'last_request': integration.metrics.last_request_time.isoformat() 
                        if integration.metrics.last_request_time else None
                }
            })
        return integrations
    
    async def remove_integration(self, name: str) -> bool:
        """Remove an integration"""
        if name in self.integrations:
            await self.integrations[name].cleanup()
            del self.integrations[name]
            del self.integration_configs[name]
            return True
        return False
    
    async def health_check_loop(self):
        """Periodic health check for all integrations"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health check error: {e}")
    
    async def perform_health_checks(self):
        """Perform health checks on all integrations"""
        for name, integration in self.integrations.items():
            try:
                await integration.validate_connection()
                if integration.status == IntegrationStatus.ERROR:
                    integration.status = IntegrationStatus.ACTIVE
            except Exception as e:
                integration.status = IntegrationStatus.ERROR
                print(f"Health check failed for {name}: {e}")
    
    async def get_integration_metrics(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for an integration"""
        integration = self.integrations.get(name)
        if not integration:
            return None
        
        return {
            'name': name,
            'status': integration.status.value,
            'total_requests': integration.metrics.total_requests,
            'successful_requests': integration.metrics.successful_requests,
            'failed_requests': integration.metrics.failed_requests,
            'success_rate': (
                integration.metrics.successful_requests / integration.metrics.total_requests
                if integration.metrics.total_requests > 0 else 0
            ),
            'average_response_time': integration.metrics.average_response_time,
            'rate_limit_hits': integration.metrics.rate_limit_hits,
            'last_request_time': integration.metrics.last_request_time.isoformat() 
                if integration.metrics.last_request_time else None,
            'quota_usage': integration.metrics.quota_usage
        }
```

### Task 14.6: Frontend Integration Dashboard

**File**: `frontend/components/integrations/integration-dashboard.tsx`

**Integration Dashboard Component**:
```typescript
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AlertCircle, CheckCircle, Clock, XCircle } from 'lucide-react';

interface Integration {
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error' | 'rate_limited' | 'unauthorized';
  metrics: {
    total_requests: number;
    success_rate: number;
    average_response_time: number;
    last_request: string | null;
  };
}

interface IntegrationDashboardProps {
  onAddIntegration: (type: string, config: any) => void;
  onRemoveIntegration: (name: string) => void;
  onTestIntegration: (name: string) => void;
}

export const IntegrationDashboard: React.FC<IntegrationDashboardProps> = ({
  onAddIntegration,
  onRemoveIntegration,
  onTestIntegration
}) => {
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [selectedType, setSelectedType] = useState<string>('');
  const [config, setConfig] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchIntegrations();
    const interval = setInterval(fetchIntegrations, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchIntegrations = async () => {
    try {
      const response = await fetch('/api/integrations');
      const data = await response.json();
      setIntegrations(data.integrations);
    } catch (error) {
      console.error('Failed to fetch integrations:', error);
    }
  };

  const handleAddIntegration = async () => {
    if (!selectedType) return;
    
    setLoading(true);
    try {
      await onAddIntegration(selectedType, config);
      setConfig({});
      setSelectedType('');
      await fetchIntegrations();
    } catch (error) {
      console.error('Failed to add integration:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'error': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'rate_limited': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'unauthorized': return <AlertCircle className="h-4 w-4 text-orange-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      case 'rate_limited': return 'bg-yellow-500';
      case 'unauthorized': return 'bg-orange-500';
      default: return 'bg-gray-500';
    }
  };

  const integrationTypes = [
    { value: 'openai', label: 'OpenAI', fields: ['api_key', 'organization'] },
    { value: 'anthropic', label: 'Anthropic', fields: ['api_key'] },
    { value: 'google_ai', label: 'Google AI', fields: ['api_key'] },
    { value: 'github', label: 'GitHub', fields: ['token', 'username'] },
    { value: 'jira', label: 'Jira', fields: ['base_url', 'username', 'api_token'] },
    { value: 'aws', label: 'AWS', fields: ['access_key', 'secret_key', 'region'] },
    { value: 'azure', label: 'Azure', fields: ['subscription_id', 'resource_group'] },
    { value: 'gcp', label: 'Google Cloud', fields: ['project_id', 'credentials_path'] }
  ];

  const selectedIntegrationType = integrationTypes.find(t => t.value === selectedType);

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">External Integrations</h2>
        <Badge variant="outline">
          {integrations.length} Active Integrations
        </Badge>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="add">Add Integration</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {integrations.map((integration) => (
              <Card key={integration.name}>
                <CardHeader className="pb-2">
                  <div className="flex justify-between items-start">
                    <CardTitle className="text-lg">{integration.name}</CardTitle>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(integration.status)}
                      <Badge className={getStatusColor(integration.status)}>
                        {integration.status}
                      </Badge>
                    </div>
                  </div>
                  <p className="text-sm text-gray-600 capitalize">{integration.type}</p>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Requests:</span>
                      <span>{integration.metrics.total_requests}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Success Rate:</span>
                      <span>{(integration.metrics.success_rate * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Avg Response:</span>
                      <span>{integration.metrics.average_response_time.toFixed(2)}s</span>
                    </div>
                    {integration.metrics.last_request && (
                      <div className="flex justify-between">
                        <span>Last Request:</span>
                        <span>{new Date(integration.metrics.last_request).toLocaleTimeString()}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="flex gap-2 mt-4">
                    <Button
                      onClick={() => onTestIntegration(integration.name)}
                      size="sm"
                      variant="outline"
                    >
                      Test
                    </Button>
                    <Button
                      onClick={() => onRemoveIntegration(integration.name)}
                      size="sm"
                      variant="destructive"
                    >
                      Remove
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="add" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Add New Integration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="integration-type">Integration Type</Label>
                <Select value={selectedType} onValueChange={setSelectedType}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select integration type" />
                  </SelectTrigger>
                  <SelectContent>
                    {integrationTypes.map((type) => (
                      <SelectItem key={type.value} value={type.value}>
                        {type.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {selectedIntegrationType && (
                <div className="space-y-3">
                  {selectedIntegrationType.fields.map((field) => (
                    <div key={field}>
                      <Label htmlFor={field}>
                        {field.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                        {field.includes('key') || field.includes('token') ? ' *' : ''}
                      </Label>
                      <Input
                        id={field}
                        type={field.includes('key') || field.includes('token') ? 'password' : 'text'}
                        value={config[field] || ''}
                        onChange={(e) => setConfig(prev => ({ ...prev, [field]: e.target.value }))}
                        placeholder={`Enter ${field.replace('_', ' ')}`}
                      />
                    </div>
                  ))}
                </div>
              )}

              <Button
                onClick={handleAddIntegration}
                disabled={!selectedType || loading}
                className="w-full"
              >
                {loading ? 'Adding...' : 'Add Integration'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Request Volume</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {integrations.map((integration) => (
                    <div key={integration.name} className="flex justify-between items-center">
                      <span className="text-sm">{integration.name}</span>
                      <Badge variant="outline">{integration.metrics.total_requests}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Success Rates</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {integrations.map((integration) => (
                    <div key={integration.name} className="flex justify-between items-center">
                      <span className="text-sm">{integration.name}</span>
                      <Badge 
                        variant={integration.metrics.success_rate > 0.9 ? "default" : "destructive"}
                      >
                        {(integration.metrics.success_rate * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};
```

## API Endpoints

### Task 14.7: Integration API

**File**: `app/api/integrations.py`

**Integration API Endpoints**:
```python
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from pydantic import BaseModel
from core.services.integration_service import IntegrationService

router = APIRouter(prefix="/api/integrations", tags=["integrations"])

# Global integration service instance
integration_service = IntegrationService()

class IntegrationConfigRequest(BaseModel):
    type: str
    config: Dict[str, Any]

class IntegrationTestRequest(BaseModel):
    name: str
    test_type: str = "connection"

@router.on_event("startup")
async def startup_integrations():
    await integration_service.initialize()

@router.on_event("shutdown")
async def shutdown_integrations():
    await integration_service.cleanup()

@router.get("")
async def list_integrations():
    """List all registered integrations"""
    integrations = await integration_service.list_integrations()
    return {"integrations": integrations}

@router.post("/register")
async def register_integration(request: IntegrationConfigRequest):
    """Register a new integration"""
    try:
        integration_name = await integration_service.register_integration(
            request.type, 
            request.config
        )
        return {
            "status": "success", 
            "integration_name": integration_name,
            "message": f"Integration '{integration_name}' registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{integration_name}")
async def remove_integration(integration_name: str):
    """Remove an integration"""
    success = await integration_service.remove_integration(integration_name)
    if success:
        return {"status": "success", "message": f"Integration '{integration_name}' removed"}
    else:
        raise HTTPException(status_code=404, detail="Integration not found")

@router.get("/{integration_name}/metrics")
async def get_integration_metrics(integration_name: str):
    """Get detailed metrics for an integration"""
    metrics = await integration_service.get_integration_metrics(integration_name)
    if metrics:
        return {"metrics": metrics}
    else:
        raise HTTPException(status_code=404, detail="Integration not found")

@router.post("/test")
async def test_integration(request: IntegrationTestRequest):
    """Test an integration"""
    integration = await integration_service.get_integration(request.name)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    
    try:
        if request.test_type == "connection":
            await integration.validate_connection()
            return {"status": "success", "message": "Connection test passed"}
        else:
            raise HTTPException(status_code=400, detail="Unknown test type")
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/health")
async def integration_health_check():
    """Get health status of all integrations"""
    await integration_service.perform_health_checks()
    integrations = await integration_service.list_integrations()
    
    healthy_count = sum(1 for i in integrations if i['status'] == 'active')
    total_count = len(integrations)
    
    return {
        "status": "healthy" if healthy_count == total_count else "degraded",
        "healthy_integrations": healthy_count,
        "total_integrations": total_count,
        "integrations": integrations
    }
```

## Validation Criteria

### Backend Validation
- [ ] Base integration framework handles authentication and rate limiting
- [ ] AI provider integrations connect and make requests successfully
- [ ] Development tool integrations perform CRUD operations
- [ ] Cloud service integrations authenticate and access resources
- [ ] Integration service manages multiple integrations
- [ ] Health checks monitor integration status
- [ ] API endpoints respond correctly

### Frontend Validation
- [ ] Integration dashboard displays all integrations
- [ ] Add integration form works for all types
- [ ] Real-time status updates function
- [ ] Metrics display accurately
- [ ] Test integration functionality works
- [ ] Remove integration functionality works

### Integration Validation
- [ ] External API calls succeed with proper authentication
- [ ] Rate limiting prevents API quota exhaustion
- [ ] Error handling gracefully manages failures
- [ ] Metrics tracking provides accurate data
- [ ] Health monitoring detects issues

## Human Testing Scenarios

1. **AI Provider Integration Test**: Add OpenAI integration and test completion generation
2. **Development Tool Test**: Connect GitHub and create repository/issues
3. **Cloud Service Test**: Connect AWS and perform S3 operations
4. **Health Monitoring Test**: Monitor integration health and error recovery
5. **Rate Limiting Test**: Verify rate limiting prevents quota exhaustion

## Next Steps

After successful validation of external service integrations, proceed to **15-security-authentication-framework.md** for implementing comprehensive security and authentication systems.

---

**Dependencies**: This phase requires core services and agent framework to be functional and provides external connectivity for enhanced agent capabilities.