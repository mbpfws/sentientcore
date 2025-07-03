# 15 - Security & Authentication Framework

## Overview

The Security & Authentication Framework provides comprehensive security measures for the multi-agent platform, including user authentication, authorization, API security, data encryption, audit logging, and compliance management. This framework ensures secure access control and protects sensitive data across all system components.

## Current State Analysis

### Security Requirements
- Multi-factor authentication (MFA)
- Role-based access control (RBAC)
- JWT token management
- API rate limiting and security
- Data encryption at rest and in transit
- Audit logging and compliance
- Session management
- Password policies and security

### Compliance Considerations
- GDPR compliance for data protection
- SOC 2 Type II requirements
- OWASP security guidelines
- Industry-specific regulations

## Implementation Tasks

### Task 15.1: Core Security Framework

**File**: `core/security/base.py`

**Base Security Framework**:
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
import bcrypt
import jwt
from cryptography.fernet import Fernet
import logging

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuthenticationMethod(Enum):
    PASSWORD = "password"
    MFA = "mfa"
    API_KEY = "api_key"
    OAUTH = "oauth"
    SSO = "sso"

@dataclass
class SecurityConfig:
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True
    mfa_enabled: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    encryption_key: Optional[str] = None
    audit_enabled: bool = True

@dataclass
class User:
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    permissions: List[str]
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    is_active: bool = True

@dataclass
class Session:
    id: str
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True

@dataclass
class AuditLog:
    id: str
    user_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    timestamp: datetime
    security_level: SecurityLevel
    success: bool

class SecurityManager:
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize encryption
        if config.encryption_key:
            self.cipher = Fernet(config.encryption_key.encode())
        else:
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            self.config.encryption_key = key.decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password strength"""
        issues = []
        
        if len(password) < self.config.password_min_length:
            issues.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one number")
        
        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one special character")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength": self.calculate_password_strength(password)
        }
    
    def calculate_password_strength(self, password: str) -> str:
        """Calculate password strength score"""
        score = 0
        
        # Length bonus
        score += min(len(password) * 2, 20)
        
        # Character variety bonus
        if any(c.islower() for c in password):
            score += 5
        if any(c.isupper() for c in password):
            score += 5
        if any(c.isdigit() for c in password):
            score += 5
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 10
        
        # Uniqueness bonus
        unique_chars = len(set(password))
        score += min(unique_chars * 2, 20)
        
        if score >= 80:
            return "very_strong"
        elif score >= 60:
            return "strong"
        elif score >= 40:
            return "medium"
        elif score >= 20:
            return "weak"
        else:
            return "very_weak"
    
    def generate_jwt_token(self, user: User, additional_claims: Dict[str, Any] = None) -> str:
        """Generate JWT token for user"""
        now = datetime.utcnow()
        expiration = now + timedelta(hours=self.config.jwt_expiration_hours)
        
        payload = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "roles": user.roles,
            "permissions": user.permissions,
            "iat": now,
            "exp": expiration,
            "jti": secrets.token_urlsafe(32)  # JWT ID for token revocation
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret, 
                algorithms=[self.config.jwt_algorithm]
            )
            return {"valid": True, "payload": payload}
        except jwt.ExpiredSignatureError:
            return {"valid": False, "error": "Token has expired"}
        except jwt.InvalidTokenError as e:
            return {"valid": False, "error": str(e)}
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def generate_api_key(self, user_id: str, name: str) -> str:
        """Generate API key for user"""
        key_data = f"{user_id}:{name}:{secrets.token_urlsafe(32)}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def generate_mfa_secret(self) -> str:
        """Generate MFA secret for TOTP"""
        return secrets.token_urlsafe(32)
    
    def verify_mfa_token(self, secret: str, token: str) -> bool:
        """Verify MFA TOTP token"""
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    def log_security_event(
        self, 
        action: str, 
        user_id: Optional[str] = None,
        resource: str = "",
        details: Dict[str, Any] = None,
        ip_address: str = "",
        user_agent: str = "",
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        success: bool = True
    ):
        """Log security event for audit"""
        if not self.config.audit_enabled:
            return
        
        audit_log = AuditLog(
            id=secrets.token_urlsafe(16),
            user_id=user_id,
            action=action,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow(),
            security_level=security_level,
            success=success
        )
        
        # Log to file/database
        self.logger.info(
            f"Security Event: {action} | User: {user_id} | Success: {success} | IP: {ip_address}"
        )
```

### Task 15.2: Authentication Service

**File**: `core/services/auth_service.py`

**Authentication Service**:
```python
from typing import Dict, Any, Optional, List
from core.security.base import SecurityManager, SecurityConfig, User, Session, AuthenticationMethod
from datetime import datetime, timedelta
import secrets
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from core.database.models import UserModel, SessionModel, AuditLogModel

class AuthenticationService:
    def __init__(self, security_manager: SecurityManager, db_session: AsyncSession):
        self.security_manager = security_manager
        self.db_session = db_session
        self.active_sessions: Dict[str, Session] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
    
    async def register_user(
        self, 
        username: str, 
        email: str, 
        password: str,
        roles: List[str] = None,
        ip_address: str = "",
        user_agent: str = ""
    ) -> Dict[str, Any]:
        """Register a new user"""
        try:
            # Validate password strength
            password_validation = self.security_manager.validate_password_strength(password)
            if not password_validation["valid"]:
                return {
                    "success": False,
                    "error": "Password does not meet requirements",
                    "details": password_validation["issues"]
                }
            
            # Check if user already exists
            existing_user = await self.get_user_by_username(username)
            if existing_user:
                return {"success": False, "error": "Username already exists"}
            
            existing_email = await self.get_user_by_email(email)
            if existing_email:
                return {"success": False, "error": "Email already registered"}
            
            # Create user
            password_hash = self.security_manager.hash_password(password)
            user_id = secrets.token_urlsafe(16)
            
            user = User(
                id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles or ["user"],
                permissions=self.get_default_permissions(roles or ["user"]),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Save to database
            await self.save_user(user)
            
            # Log security event
            self.security_manager.log_security_event(
                action="user_registration",
                user_id=user_id,
                resource="user_account",
                details={"username": username, "email": email},
                ip_address=ip_address,
                user_agent=user_agent,
                success=True
            )
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "User registered successfully"
            }
            
        except Exception as e:
            self.security_manager.log_security_event(
                action="user_registration",
                resource="user_account",
                details={"username": username, "error": str(e)},
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return {"success": False, "error": str(e)}
    
    async def authenticate_user(
        self, 
        username: str, 
        password: str,
        mfa_token: Optional[str] = None,
        ip_address: str = "",
        user_agent: str = ""
    ) -> Dict[str, Any]:
        """Authenticate user with username/password and optional MFA"""
        try:
            # Check for account lockout
            if await self.is_account_locked(username):
                return {
                    "success": False,
                    "error": "Account is temporarily locked due to too many failed attempts"
                }
            
            # Get user
            user = await self.get_user_by_username(username)
            if not user or not user.is_active:
                await self.record_failed_attempt(username, ip_address, user_agent)
                return {"success": False, "error": "Invalid credentials"}
            
            # Verify password
            if not self.security_manager.verify_password(password, user.password_hash):
                await self.record_failed_attempt(username, ip_address, user_agent)
                return {"success": False, "error": "Invalid credentials"}
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_token:
                    return {
                        "success": False,
                        "error": "MFA token required",
                        "requires_mfa": True
                    }
                
                if not self.security_manager.verify_mfa_token(user.mfa_secret, mfa_token):
                    await self.record_failed_attempt(username, ip_address, user_agent)
                    return {"success": False, "error": "Invalid MFA token"}
            
            # Clear failed attempts
            await self.clear_failed_attempts(username)
            
            # Create session
            session = await self.create_session(user, ip_address, user_agent)
            
            # Generate JWT token
            jwt_token = self.security_manager.generate_jwt_token(user)
            
            # Update last login
            user.last_login = datetime.utcnow()
            await self.save_user(user)
            
            # Log successful login
            self.security_manager.log_security_event(
                action="user_login",
                user_id=user.id,
                resource="user_session",
                details={"session_id": session.id},
                ip_address=ip_address,
                user_agent=user_agent,
                success=True
            )
            
            return {
                "success": True,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "roles": user.roles,
                    "permissions": user.permissions
                },
                "session_id": session.id,
                "token": jwt_token,
                "expires_at": session.expires_at.isoformat()
            }
            
        except Exception as e:
            self.security_manager.log_security_event(
                action="user_login",
                resource="user_session",
                details={"username": username, "error": str(e)},
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return {"success": False, "error": str(e)}
    
    async def create_session(
        self, 
        user: User, 
        ip_address: str, 
        user_agent: str
    ) -> Session:
        """Create user session"""
        session_id = secrets.token_urlsafe(32)
        session_token = secrets.token_urlsafe(64)
        
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=self.security_manager.config.session_timeout_minutes)
        
        session = Session(
            id=session_id,
            user_id=user.id,
            token=session_token,
            created_at=now,
            expires_at=expires_at,
            last_activity=now,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Store in memory and database
        self.active_sessions[session_id] = session
        await self.save_session(session)
        
        return session
    
    async def validate_session(self, session_id: str, token: str) -> Optional[User]:
        """Validate user session"""
        session = self.active_sessions.get(session_id)
        if not session:
            session = await self.get_session_from_db(session_id)
        
        if not session or not session.is_active:
            return None
        
        if session.token != token:
            return None
        
        if datetime.utcnow() > session.expires_at:
            await self.invalidate_session(session_id)
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        await self.save_session(session)
        
        # Get user
        user = await self.get_user_by_id(session.user_id)
        return user
    
    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate user session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.is_active = False
            await self.save_session(session)
            del self.active_sessions[session_id]
            return True
        return False
    
    async def enable_mfa(self, user_id: str) -> Dict[str, Any]:
        """Enable MFA for user"""
        user = await self.get_user_by_id(user_id)
        if not user:
            return {"success": False, "error": "User not found"}
        
        mfa_secret = self.security_manager.generate_mfa_secret()
        user.mfa_secret = mfa_secret
        user.mfa_enabled = True
        await self.save_user(user)
        
        # Generate QR code data for TOTP setup
        import pyotp
        totp = pyotp.TOTP(mfa_secret)
        qr_code_url = totp.provisioning_uri(
            name=user.email,
            issuer_name="SentientCore"
        )
        
        return {
            "success": True,
            "secret": mfa_secret,
            "qr_code_url": qr_code_url
        }
    
    async def is_account_locked(self, username: str) -> bool:
        """Check if account is locked"""
        user = await self.get_user_by_username(username)
        if not user:
            return False
        
        if user.locked_until and datetime.utcnow() < user.locked_until:
            return True
        
        return False
    
    async def record_failed_attempt(
        self, 
        username: str, 
        ip_address: str, 
        user_agent: str
    ):
        """Record failed login attempt"""
        user = await self.get_user_by_username(username)
        if user:
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.security_manager.config.max_login_attempts:
                lockout_duration = timedelta(minutes=self.security_manager.config.lockout_duration_minutes)
                user.locked_until = datetime.utcnow() + lockout_duration
            
            await self.save_user(user)
        
        # Log failed attempt
        self.security_manager.log_security_event(
            action="failed_login_attempt",
            user_id=user.id if user else None,
            resource="user_session",
            details={"username": username},
            ip_address=ip_address,
            user_agent=user_agent,
            success=False
        )
    
    async def clear_failed_attempts(self, username: str):
        """Clear failed login attempts"""
        user = await self.get_user_by_username(username)
        if user:
            user.failed_login_attempts = 0
            user.locked_until = None
            await self.save_user(user)
    
    def get_default_permissions(self, roles: List[str]) -> List[str]:
        """Get default permissions for roles"""
        permission_map = {
            "admin": [
                "user.create", "user.read", "user.update", "user.delete",
                "agent.create", "agent.read", "agent.update", "agent.delete",
                "system.configure", "system.monitor", "audit.read"
            ],
            "developer": [
                "agent.create", "agent.read", "agent.update",
                "project.create", "project.read", "project.update",
                "deployment.create", "deployment.read"
            ],
            "user": [
                "agent.read", "project.read", "task.create", "task.read", "task.update"
            ]
        }
        
        permissions = set()
        for role in roles:
            permissions.update(permission_map.get(role, []))
        
        return list(permissions)
    
    # Database operations (to be implemented with actual ORM)
    async def save_user(self, user: User):
        """Save user to database"""
        # Implementation depends on chosen ORM
        pass
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        # Implementation depends on chosen ORM
        pass
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        # Implementation depends on chosen ORM
        pass
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        # Implementation depends on chosen ORM
        pass
    
    async def save_session(self, session: Session):
        """Save session to database"""
        # Implementation depends on chosen ORM
        pass
    
    async def get_session_from_db(self, session_id: str) -> Optional[Session]:
        """Get session from database"""
        # Implementation depends on chosen ORM
        pass
```

### Task 15.3: Authorization & RBAC

**File**: `core/security/authorization.py`

**Role-Based Access Control**:
```python
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from core.security.base import User

class Permission(Enum):
    # User permissions
    USER_CREATE = "user.create"
    USER_READ = "user.read"
    USER_UPDATE = "user.update"
    USER_DELETE = "user.delete"
    
    # Agent permissions
    AGENT_CREATE = "agent.create"
    AGENT_READ = "agent.read"
    AGENT_UPDATE = "agent.update"
    AGENT_DELETE = "agent.delete"
    AGENT_EXECUTE = "agent.execute"
    
    # Project permissions
    PROJECT_CREATE = "project.create"
    PROJECT_READ = "project.read"
    PROJECT_UPDATE = "project.update"
    PROJECT_DELETE = "project.delete"
    
    # Task permissions
    TASK_CREATE = "task.create"
    TASK_READ = "task.read"
    TASK_UPDATE = "task.update"
    TASK_DELETE = "task.delete"
    
    # System permissions
    SYSTEM_CONFIGURE = "system.configure"
    SYSTEM_MONITOR = "system.monitor"
    SYSTEM_BACKUP = "system.backup"
    
    # Audit permissions
    AUDIT_READ = "audit.read"
    AUDIT_EXPORT = "audit.export"
    
    # Deployment permissions
    DEPLOYMENT_CREATE = "deployment.create"
    DEPLOYMENT_READ = "deployment.read"
    DEPLOYMENT_EXECUTE = "deployment.execute"
    DEPLOYMENT_ROLLBACK = "deployment.rollback"

@dataclass
class Role:
    name: str
    description: str
    permissions: Set[Permission]
    is_system_role: bool = False

@dataclass
class Resource:
    type: str
    id: Optional[str] = None
    attributes: Dict[str, Any] = None

@dataclass
class AccessRequest:
    user: User
    permission: Permission
    resource: Resource
    context: Dict[str, Any] = None

class AuthorizationService:
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.resource_policies: Dict[str, List[callable]] = {}
        self.initialize_default_roles()
    
    def initialize_default_roles(self):
        """Initialize default system roles"""
        # Super Admin role
        self.roles["super_admin"] = Role(
            name="super_admin",
            description="Super administrator with all permissions",
            permissions=set(Permission),
            is_system_role=True
        )
        
        # Admin role
        self.roles["admin"] = Role(
            name="admin",
            description="Administrator with most permissions",
            permissions={
                Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE, Permission.USER_DELETE,
                Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_DELETE,
                Permission.PROJECT_CREATE, Permission.PROJECT_READ, Permission.PROJECT_UPDATE, Permission.PROJECT_DELETE,
                Permission.SYSTEM_CONFIGURE, Permission.SYSTEM_MONITOR,
                Permission.AUDIT_READ, Permission.AUDIT_EXPORT,
                Permission.DEPLOYMENT_CREATE, Permission.DEPLOYMENT_READ, Permission.DEPLOYMENT_EXECUTE
            }
        )
        
        # Developer role
        self.roles["developer"] = Role(
            name="developer",
            description="Developer with agent and project permissions",
            permissions={
                Permission.AGENT_CREATE, Permission.AGENT_READ, Permission.AGENT_UPDATE, Permission.AGENT_EXECUTE,
                Permission.PROJECT_CREATE, Permission.PROJECT_READ, Permission.PROJECT_UPDATE,
                Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_UPDATE,
                Permission.DEPLOYMENT_CREATE, Permission.DEPLOYMENT_READ
            }
        )
        
        # User role
        self.roles["user"] = Role(
            name="user",
            description="Basic user with limited permissions",
            permissions={
                Permission.AGENT_READ, Permission.AGENT_EXECUTE,
                Permission.PROJECT_READ,
                Permission.TASK_CREATE, Permission.TASK_READ, Permission.TASK_UPDATE
            }
        )
        
        # Viewer role
        self.roles["viewer"] = Role(
            name="viewer",
            description="Read-only access",
            permissions={
                Permission.AGENT_READ,
                Permission.PROJECT_READ,
                Permission.TASK_READ
            }
        )
    
    def check_permission(
        self, 
        user: User, 
        permission: Permission, 
        resource: Resource = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Check if user has permission for resource"""
        # Check if user is active
        if not user.is_active:
            return False
        
        # Check direct permissions
        if permission.value in user.permissions:
            return self.check_resource_policy(user, permission, resource, context)
        
        # Check role-based permissions
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role and permission in role.permissions:
                return self.check_resource_policy(user, permission, resource, context)
        
        return False
    
    def check_resource_policy(
        self, 
        user: User, 
        permission: Permission, 
        resource: Resource = None,
        context: Dict[str, Any] = None
    ) -> bool:
        """Check resource-specific policies"""
        if not resource:
            return True
        
        # Get policies for resource type
        policies = self.resource_policies.get(resource.type, [])
        
        # Apply all policies
        for policy in policies:
            if not policy(user, permission, resource, context):
                return False
        
        return True
    
    def add_role(self, role: Role):
        """Add a new role"""
        self.roles[role.name] = role
    
    def remove_role(self, role_name: str) -> bool:
        """Remove a role"""
        if role_name in self.roles and not self.roles[role_name].is_system_role:
            del self.roles[role_name]
            return True
        return False
    
    def assign_role_to_user(self, user: User, role_name: str) -> bool:
        """Assign role to user"""
        if role_name in self.roles:
            if role_name not in user.roles:
                user.roles.append(role_name)
                # Update permissions
                self.update_user_permissions(user)
                return True
        return False
    
    def remove_role_from_user(self, user: User, role_name: str) -> bool:
        """Remove role from user"""
        if role_name in user.roles:
            user.roles.remove(role_name)
            # Update permissions
            self.update_user_permissions(user)
            return True
        return False
    
    def update_user_permissions(self, user: User):
        """Update user permissions based on roles"""
        permissions = set()
        
        # Add permissions from roles
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(perm.value for perm in role.permissions)
        
        user.permissions = list(permissions)
    
    def add_resource_policy(self, resource_type: str, policy: callable):
        """Add resource-specific policy"""
        if resource_type not in self.resource_policies:
            self.resource_policies[resource_type] = []
        self.resource_policies[resource_type].append(policy)
    
    def get_user_permissions(self, user: User) -> List[str]:
        """Get all permissions for user"""
        permissions = set(user.permissions)
        
        # Add permissions from roles
        for role_name in user.roles:
            role = self.roles.get(role_name)
            if role:
                permissions.update(perm.value for perm in role.permissions)
        
        return list(permissions)
    
    def can_access_resource(
        self, 
        user: User, 
        resource_type: str, 
        resource_id: str = None,
        action: str = "read"
    ) -> bool:
        """Check if user can access specific resource"""
        permission_map = {
            "create": "create",
            "read": "read",
            "update": "update",
            "delete": "delete",
            "execute": "execute"
        }
        
        permission_name = f"{resource_type}.{permission_map.get(action, action)}"
        
        try:
            permission = Permission(permission_name)
            resource = Resource(type=resource_type, id=resource_id)
            return self.check_permission(user, permission, resource)
        except ValueError:
            return False

# Example resource policies
def owner_only_policy(user: User, permission: Permission, resource: Resource, context: Dict[str, Any] = None) -> bool:
    """Policy that allows access only to resource owner"""
    if resource.attributes and "owner_id" in resource.attributes:
        return resource.attributes["owner_id"] == user.id
    return True

def team_member_policy(user: User, permission: Permission, resource: Resource, context: Dict[str, Any] = None) -> bool:
    """Policy that allows access to team members"""
    if resource.attributes and "team_members" in resource.attributes:
        return user.id in resource.attributes["team_members"]
    return True

def business_hours_policy(user: User, permission: Permission, resource: Resource, context: Dict[str, Any] = None) -> bool:
    """Policy that restricts access to business hours"""
    from datetime import datetime
    now = datetime.now()
    return 9 <= now.hour <= 17  # 9 AM to 5 PM
```

### Task 15.4: API Security Middleware

**File**: `core/security/middleware.py`

**API Security Middleware**:
```python
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from core.services.auth_service import AuthenticationService
from core.security.authorization import AuthorizationService, Permission, Resource
from core.security.base import User
import time
from collections import defaultdict
import asyncio

class SecurityMiddleware:
    def __init__(
        self, 
        auth_service: AuthenticationService,
        authz_service: AuthorizationService
    ):
        self.auth_service = auth_service
        self.authz_service = authz_service
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": [],
            "blocked_until": 0
        })
        self.security_bearer = HTTPBearer(auto_error=False)
    
    async def authenticate_request(self, request: Request) -> Optional[User]:
        """Authenticate request and return user"""
        # Try JWT token first
        authorization = request.headers.get("Authorization")
        if authorization and authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            token_data = self.auth_service.security_manager.verify_jwt_token(token)
            if token_data["valid"]:
                user_id = token_data["payload"]["user_id"]
                return await self.auth_service.get_user_by_id(user_id)
        
        # Try session authentication
        session_id = request.headers.get("X-Session-ID")
        session_token = request.headers.get("X-Session-Token")
        if session_id and session_token:
            return await self.auth_service.validate_session(session_id, session_token)
        
        # Try API key authentication
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self.authenticate_api_key(api_key)
        
        return None
    
    async def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate using API key"""
        # Implementation depends on API key storage
        # This would typically query the database for the API key
        pass
    
    def check_rate_limit(self, request: Request, user: Optional[User] = None) -> bool:
        """Check rate limiting for request"""
        # Determine rate limit key (IP or user ID)
        if user:
            rate_limit_key = f"user:{user.id}"
            requests_per_minute = 100  # Higher limit for authenticated users
        else:
            rate_limit_key = f"ip:{request.client.host}"
            requests_per_minute = 20   # Lower limit for anonymous users
        
        current_time = time.time()
        user_data = self.rate_limits[rate_limit_key]
        
        # Check if currently blocked
        if current_time < user_data["blocked_until"]:
            return False
        
        # Clean old requests (older than 1 minute)
        user_data["requests"] = [
            req_time for req_time in user_data["requests"]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(user_data["requests"]) >= requests_per_minute:
            # Block for 5 minutes
            user_data["blocked_until"] = current_time + 300
            return False
        
        # Add current request
        user_data["requests"].append(current_time)
        return True
    
    def require_permission(self, permission: Permission, resource_type: str = None):
        """Decorator to require specific permission"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                request = kwargs.get('request') or args[0]
                user = await self.authenticate_request(request)
                
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                # Check permission
                resource = Resource(type=resource_type) if resource_type else None
                if not self.authz_service.check_permission(user, permission, resource):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Add user to kwargs for use in endpoint
                kwargs['current_user'] = user
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, required_roles: list):
        """Decorator to require specific roles"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                request = kwargs.get('request') or args[0]
                user = await self.authenticate_request(request)
                
                if not user:
                    raise HTTPException(status_code=401, detail="Authentication required")
                
                # Check roles
                if not any(role in user.roles for role in required_roles):
                    raise HTTPException(status_code=403, detail="Insufficient role permissions")
                
                kwargs['current_user'] = user
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    async def security_headers_middleware(self, request: Request, call_next):
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response
    
    async def rate_limit_middleware(self, request: Request, call_next):
        """Rate limiting middleware"""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)
        
        user = await self.authenticate_request(request)
        
        if not self.check_rate_limit(request, user):
            raise HTTPException(
                status_code=429, 
                detail="Rate limit exceeded. Please try again later."
            )
        
        return await call_next(request)
    
    async def audit_middleware(self, request: Request, call_next):
        """Audit logging middleware"""
        start_time = time.time()
        user = await self.authenticate_request(request)
        
        try:
            response = await call_next(request)
            
            # Log successful request
            self.auth_service.security_manager.log_security_event(
                action=f"{request.method} {request.url.path}",
                user_id=user.id if user else None,
                resource="api_endpoint",
                details={
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time
                },
                ip_address=request.client.host,
                user_agent=request.headers.get("User-Agent", ""),
                success=True
            )
            
            return response
            
        except Exception as e:
            # Log failed request
            self.auth_service.security_manager.log_security_event(
                action=f"{request.method} {request.url.path}",
                user_id=user.id if user else None,
                resource="api_endpoint",
                details={
                    "error": str(e),
                    "response_time": time.time() - start_time
                },
                ip_address=request.client.host,
                user_agent=request.headers.get("User-Agent", ""),
                success=False
            )
            raise

# Dependency functions for FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    auth_service: AuthenticationService = Depends()
) -> User:
    """Get current authenticated user"""
    token_data = auth_service.security_manager.verify_jwt_token(credentials.credentials)
    if not token_data["valid"]:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = token_data["payload"]["user_id"]
    user = await auth_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def require_permissions(*permissions: Permission):
    """Dependency to require specific permissions"""
    async def check_permissions(
        current_user: User = Depends(get_current_active_user),
        authz_service: AuthorizationService = Depends()
    ):
        for permission in permissions:
            if not authz_service.check_permission(current_user, permission):
                raise HTTPException(
                    status_code=403, 
                    detail=f"Permission {permission.value} required"
                )
        return current_user
    
    return check_permissions
```

### Task 15.5: Frontend Authentication Components

**File**: `frontend/components/auth/login-form.tsx`

**Login Form Component**:
```typescript
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Eye, EyeOff, Shield } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';

interface LoginFormProps {
  onSuccess?: () => void;
  onRegisterClick?: () => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSuccess, onRegisterClick }) => {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    mfaToken: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [requiresMfa, setRequiresMfa] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const result = await login({
        username: formData.username,
        password: formData.password,
        mfaToken: requiresMfa ? formData.mfaToken : undefined
      });

      if (result.success) {
        onSuccess?.();
      } else if (result.requires_mfa) {
        setRequiresMfa(true);
        setError('Please enter your MFA token');
      } else {
        setError(result.error || 'Login failed');
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (error) setError('');
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="flex justify-center mb-4">
          <Shield className="h-12 w-12 text-blue-600" />
        </div>
        <CardTitle className="text-2xl">Sign In</CardTitle>
        <p className="text-gray-600">Access your SentientCore account</p>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="username">Username or Email</Label>
            <Input
              id="username"
              type="text"
              value={formData.username}
              onChange={(e) => handleInputChange('username', e.target.value)}
              placeholder="Enter your username or email"
              required
              disabled={loading}
            />
          </div>

          <div>
            <Label htmlFor="password">Password</Label>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={(e) => handleInputChange('password', e.target.value)}
                placeholder="Enter your password"
                required
                disabled={loading}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                onClick={() => setShowPassword(!showPassword)}
                disabled={loading}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </Button>
            </div>
            
            {/* Password Strength Indicator */}
            {formData.password && passwordStrength && (
              <div className="mt-2">
                <div className="flex justify-between text-sm mb-1">
                  <span>Password Strength</span>
                  <span className={passwordStrength.score >= 60 ? 'text-green-600' : 'text-red-600'}>
                    {passwordStrength.score >= 80 ? 'Very Strong' :
                     passwordStrength.score >= 60 ? 'Strong' :
                     passwordStrength.score >= 40 ? 'Medium' : 'Weak'}
                  </span>
                </div>
                <Progress value={passwordStrength.score} className="h-2" />
                {passwordStrength.feedback.length > 0 && (
                  <div className="mt-2">
                    <p className="text-sm text-gray-600 mb-1">Requirements:</p>
                    <ul className="text-xs space-y-1">
                      {passwordStrength.feedback.map((item, index) => (
                        <li key={index} className="flex items-center text-red-600">
                          <X className="h-3 w-3 mr-1" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>

          <div>
            <Label htmlFor="confirmPassword">Confirm Password</Label>
            <div className="relative">
              <Input
                id="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                value={formData.confirmPassword}
                onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                placeholder="Confirm your password"
                required
                disabled={loading}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                disabled={loading}
              >
                {showConfirmPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </Button>
            </div>
            
            {/* Password Match Indicator */}
            {formData.confirmPassword && (
              <div className="mt-1 flex items-center text-sm">
                {formData.password === formData.confirmPassword ? (
                  <>
                    <Check className="h-4 w-4 text-green-600 mr-1" />
                    <span className="text-green-600">Passwords match</span>
                  </>
                ) : (
                  <>
                    <X className="h-4 w-4 text-red-600 mr-1" />
                    <span className="text-red-600">Passwords do not match</span>
                  </>
                )}
              </div>
            )}
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert className="border-green-200 bg-green-50">
              <Check className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-800">{success}</AlertDescription>
            </Alert>
          )}

          <Button
            type="submit"
            className="w-full"
            disabled={loading || !isFormValid()}
          >
            {loading ? 'Creating Account...' : 'Create Account'}
          </Button>

          <div className="text-center">
            {onLoginClick && (
              <div className="text-sm text-gray-600">
                Already have an account?{' '}
                <Button
                  type="button"
                  variant="link"
                  className="p-0 h-auto font-medium"
                  onClick={onLoginClick}
                >
                  Sign in
                </Button>
              </div>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  );
};
```

**File**: `frontend/hooks/useAuth.ts`

**Authentication Hook**:
```typescript
import { useState, useEffect, createContext, useContext } from 'react';
import { useRouter } from 'next/navigation';

interface User {
  id: string;
  username: string;
  email: string;
  roles: string[];
  permissions: string[];
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

interface LoginData {
  username: string;
  password: string;
  mfaToken?: string;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
}

interface AuthContextType extends AuthState {
  login: (data: LoginData) => Promise<any>;
  register: (data: RegisterData) => Promise<any>;
  logout: () => void;
  refreshToken: () => Promise<boolean>;
  hasPermission: (permission: string) => boolean;
  hasRole: (role: string) => boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    token: null,
    isAuthenticated: false,
    isLoading: true
  });
  
  const router = useRouter();

  useEffect(() => {
    // Check for existing token on mount
    const token = localStorage.getItem('auth_token');
    const userData = localStorage.getItem('user_data');
    
    if (token && userData) {
      try {
        const user = JSON.parse(userData);
        setAuthState({
          user,
          token,
          isAuthenticated: true,
          isLoading: false
        });
      } catch (error) {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_data');
        setAuthState(prev => ({ ...prev, isLoading: false }));
      }
    } else {
      setAuthState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  const login = async (data: LoginData) => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.success) {
        const { user, token } = result;
        
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user_data', JSON.stringify(user));
        
        setAuthState({
          user,
          token,
          isAuthenticated: true,
          isLoading: false
        });
      }

      return result;
    } catch (error) {
      return { success: false, error: 'Network error' };
    }
  };

  const register = async (data: RegisterData) => {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      return await response.json();
    } catch (error) {
      return { success: false, error: 'Network error' };
    }
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
    
    setAuthState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false
    });
    
    router.push('/login');
  };

  const refreshToken = async (): Promise<boolean> => {
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authState.token}`,
        },
      });

      if (response.ok) {
        const result = await response.json();
        const { token, user } = result;
        
        localStorage.setItem('auth_token', token);
        localStorage.setItem('user_data', JSON.stringify(user));
        
        setAuthState(prev => ({
          ...prev,
          user,
          token
        }));
        
        return true;
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
    }
    
    logout();
    return false;
  };

  const hasPermission = (permission: string): boolean => {
    return authState.user?.permissions.includes(permission) || false;
  };

  const hasRole = (role: string): boolean => {
    return authState.user?.roles.includes(role) || false;
  };

  const contextValue: AuthContextType = {
    ...authState,
    login,
    register,
    logout,
    refreshToken,
    hasPermission,
    hasRole
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};
```-0 h-full px-3 py-2 hover:bg-transparent"
                onClick={() => setShowPassword(!showPassword)}
                disabled={loading}
              >
                {showPassword ? (
                  <EyeOff className="h-4 w-4" />
                ) : (
                  <Eye className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>

          {requiresMfa && (
            <div>
              <Label htmlFor="mfaToken">MFA Token</Label>
              <Input
                id="mfaToken"
                type="text"
                value={formData.mfaToken}
                onChange={(e) => handleInputChange('mfaToken', e.target.value)}
                placeholder="Enter 6-digit MFA code"
                maxLength={6}
                required
                disabled={loading}
              />
              <p className="text-sm text-gray-600 mt-1">
                Enter the 6-digit code from your authenticator app
              </p>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Button
            type="submit"
            className="w-full"
            disabled={loading || !formData.username || !formData.password || (requiresMfa && !formData.mfaToken)}
          >
            {loading ? 'Signing in...' : 'Sign In'}
          </Button>

          <div className="text-center space-y-2">
            <Button
              type="button"
              variant="link"
              className="text-sm"
              onClick={() => {/* Handle forgot password */}}
            >
              Forgot your password?
            </Button>
            
            {onRegisterClick && (
              <div className="text-sm text-gray-600">
                Don't have an account?{' '}
                <Button
                  type="button"
                  variant="link"
                  className="p-0 h-auto font-medium"
                  onClick={onRegisterClick}
                >
                  Sign up
                </Button>
              </div>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  );
};
```

**File**: `frontend/components/auth/register-form.tsx`

**Registration Form Component**:
```typescript
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Eye, EyeOff, UserPlus, Check, X } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';

interface RegisterFormProps {
  onSuccess?: () => void;
  onLoginClick?: () => void;
}

interface PasswordStrength {
  score: number;
  feedback: string[];
  color: string;
}

export const RegisterForm: React.FC<RegisterFormProps> = ({ onSuccess, onLoginClick }) => {
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [passwordStrength, setPasswordStrength] = useState<PasswordStrength | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  const { register } = useAuth();

  const calculatePasswordStrength = (password: string): PasswordStrength => {
    let score = 0;
    const feedback: string[] = [];
    
    if (password.length >= 8) {
      score += 20;
    } else {
      feedback.push('At least 8 characters');
    }
    
    if (/[a-z]/.test(password)) {
      score += 10;
    } else {
      feedback.push('Lowercase letter');
    }
    
    if (/[A-Z]/.test(password)) {
      score += 10;
    } else {
      feedback.push('Uppercase letter');
    }
    
    if (/\d/.test(password)) {
      score += 10;
    } else {
      feedback.push('Number');
    }
    
    if (/[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(password)) {
      score += 20;
    } else {
      feedback.push('Special character');
    }
    
    if (password.length >= 12) {
      score += 10;
    }
    
    if (new Set(password).size >= password.length * 0.7) {
      score += 20;
    }
    
    let color = 'bg-red-500';
    if (score >= 80) color = 'bg-green-500';
    else if (score >= 60) color = 'bg-yellow-500';
    else if (score >= 40) color = 'bg-orange-500';
    
    return { score, feedback, color };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    // Validate passwords match
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      setLoading(false);
      return;
    }

    // Validate password strength
    if (passwordStrength && passwordStrength.score < 60) {
      setError('Password is too weak. Please choose a stronger password.');
      setLoading(false);
      return;
    }

    try {
      const result = await register({
        username: formData.username,
        email: formData.email,
        password: formData.password
      });

      if (result.success) {
        setSuccess('Account created successfully! You can now sign in.');
        setTimeout(() => {
          onSuccess?.();
        }, 2000);
      } else {
        setError(result.error || 'Registration failed');
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    
    if (field === 'password') {
      setPasswordStrength(calculatePasswordStrength(value));
    }
    
    if (error) setError('');
    if (success) setSuccess('');
  };

  const isFormValid = () => {
    return (
      formData.username &&
      formData.email &&
      formData.password &&
      formData.confirmPassword &&
      formData.password === formData.confirmPassword &&
      passwordStrength &&
      passwordStrength.score >= 60
    );
  };

  return (
    <Card className="w-full max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="flex justify-center mb-4">
          <UserPlus className="h-12 w-12 text-blue-600" />
        </div>
        <CardTitle className="text-2xl">Create Account</CardTitle>
        <p className="text-gray-600">Join SentientCore today</p>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <Label htmlFor="username">Username</Label>
            <Input
              id="username"
              type="text"
              value={formData.username}
              onChange={(e) => handleInputChange('username', e.target.value)}
              placeholder="Choose a username"
              required
              disabled={loading}
            />
          </div>

          <div>
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              value={formData.email}
              onChange={(e) => handleInputChange('email', e.target.value)}
              placeholder="Enter your email"
              required
              disabled={loading}
            />
          </div>

          <div>
            <Label htmlFor="password">Password</Label>
            <div className="relative">
              <Input
                id="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={(e) => handleInputChange('password', e.target.value)}
                placeholder="Create a strong password"
                required
                disabled={loading}
              />
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="absolute right-0 top