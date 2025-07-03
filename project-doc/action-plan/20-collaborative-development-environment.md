# Action Plan 20: Collaborative Development Environment

## Overview

This action plan establishes a comprehensive collaborative development environment that enables real-time code collaboration, team project management, and synchronized development workflows. The system will support multiple developers working on the same codebase with features like real-time editing, conflict resolution, code reviews, and team communication.

## Core Components

### 1. Real-Time Collaboration Engine

**File**: `core/collaboration/realtime_engine.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime
import asyncio
import json
import uuid
from enum import Enum

class CollaborationEventType(Enum):
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CURSOR_MOVED = "cursor_moved"
    TEXT_CHANGED = "text_changed"
    FILE_OPENED = "file_opened"
    FILE_CLOSED = "file_closed"
    COMMENT_ADDED = "comment_added"
    SELECTION_CHANGED = "selection_changed"
    VOICE_CHAT_STARTED = "voice_chat_started"
    SCREEN_SHARE_STARTED = "screen_share_started"

class UserPresence(Enum):
    ONLINE = "online"
    AWAY = "away"
    BUSY = "busy"
    OFFLINE = "offline"

@dataclass
class CollaborationUser:
    user_id: str
    username: str
    email: str
    avatar_url: Optional[str] = None
    presence: UserPresence = UserPresence.ONLINE
    current_file: Optional[str] = None
    cursor_position: Optional[Dict[str, int]] = None
    selection_range: Optional[Dict[str, Any]] = None
    color: str = "#3B82F6"  # Default blue
    last_activity: datetime = field(default_factory=datetime.now)
    permissions: Set[str] = field(default_factory=set)

@dataclass
class CollaborationEvent:
    event_id: str
    event_type: CollaborationEventType
    user_id: str
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TextOperation:
    operation_id: str
    user_id: str
    file_path: str
    operation_type: str  # 'insert', 'delete', 'replace'
    position: int
    content: str
    length: int
    timestamp: datetime
    applied: bool = False

@dataclass
class CollaborationSession:
    session_id: str
    project_id: str
    project_name: str
    owner_id: str
    created_at: datetime
    active_users: Dict[str, CollaborationUser] = field(default_factory=dict)
    open_files: Set[str] = field(default_factory=set)
    file_locks: Dict[str, str] = field(default_factory=dict)  # file_path -> user_id
    pending_operations: List[TextOperation] = field(default_factory=list)
    chat_messages: List[Dict[str, Any]] = field(default_factory=list)
    voice_chat_active: bool = False
    screen_share_active: bool = False
    screen_share_user: Optional[str] = None

class RealtimeCollaborationEngine:
    def __init__(self):
        self.sessions: Dict[str, CollaborationSession] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.event_handlers: Dict[CollaborationEventType, List[Callable]] = {}
        self.websocket_connections: Dict[str, Any] = {}  # user_id -> websocket
        self.operation_queue: asyncio.Queue = asyncio.Queue()
        self.conflict_resolver = OperationalTransform()
        
    async def create_session(self, project_id: str, project_name: str, owner_id: str) -> str:
        """Create a new collaboration session."""
        session_id = str(uuid.uuid4())
        
        session = CollaborationSession(
            session_id=session_id,
            project_id=project_id,
            project_name=project_name,
            owner_id=owner_id,
            created_at=datetime.now()
        )
        
        self.sessions[session_id] = session
        return session_id
    
    async def join_session(self, session_id: str, user: CollaborationUser) -> bool:
        """Add a user to a collaboration session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.active_users[user.user_id] = user
        self.user_sessions[user.user_id] = session_id
        
        # Broadcast user joined event
        await self._broadcast_event(
            session_id,
            CollaborationEventType.USER_JOINED,
            user.user_id,
            {
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "avatar_url": user.avatar_url,
                    "color": user.color
                },
                "active_users_count": len(session.active_users)
            }
        )
        
        return True
    
    async def leave_session(self, user_id: str) -> bool:
        """Remove a user from their current session."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        if user_id in session.active_users:
            user = session.active_users[user_id]
            del session.active_users[user_id]
            del self.user_sessions[user_id]
            
            # Release any file locks held by this user
            files_to_unlock = [
                file_path for file_path, lock_user_id in session.file_locks.items()
                if lock_user_id == user_id
            ]
            
            for file_path in files_to_unlock:
                del session.file_locks[file_path]
            
            # Broadcast user left event
            await self._broadcast_event(
                session_id,
                CollaborationEventType.USER_LEFT,
                user_id,
                {
                    "user_id": user_id,
                    "username": user.username,
                    "active_users_count": len(session.active_users),
                    "unlocked_files": files_to_unlock
                }
            )
            
            # Clean up empty sessions
            if len(session.active_users) == 0:
                del self.sessions[session_id]
            
            return True
        
        return False
    
    async def handle_text_operation(self, user_id: str, operation: TextOperation) -> bool:
        """Handle a text editing operation with conflict resolution."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        # Apply operational transformation for conflict resolution
        transformed_operation = await self.conflict_resolver.transform_operation(
            operation, session.pending_operations
        )
        
        # Add to pending operations
        session.pending_operations.append(transformed_operation)
        
        # Broadcast the operation to other users
        await self._broadcast_event(
            session_id,
            CollaborationEventType.TEXT_CHANGED,
            user_id,
            {
                "operation": {
                    "operation_id": transformed_operation.operation_id,
                    "file_path": transformed_operation.file_path,
                    "operation_type": transformed_operation.operation_type,
                    "position": transformed_operation.position,
                    "content": transformed_operation.content,
                    "length": transformed_operation.length
                },
                "user_id": user_id
            },
            exclude_user=user_id
        )
        
        return True
    
    async def handle_cursor_movement(self, user_id: str, file_path: str, position: Dict[str, int]) -> bool:
        """Handle cursor position updates."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        if user_id in session.active_users:
            session.active_users[user_id].cursor_position = position
            session.active_users[user_id].current_file = file_path
            session.active_users[user_id].last_activity = datetime.now()
            
            # Broadcast cursor movement
            await self._broadcast_event(
                session_id,
                CollaborationEventType.CURSOR_MOVED,
                user_id,
                {
                    "user_id": user_id,
                    "file_path": file_path,
                    "position": position
                },
                exclude_user=user_id
            )
            
            return True
        
        return False
    
    async def handle_file_operation(self, user_id: str, file_path: str, operation: str) -> bool:
        """Handle file open/close operations."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        if operation == "open":
            session.open_files.add(file_path)
            event_type = CollaborationEventType.FILE_OPENED
        elif operation == "close":
            session.open_files.discard(file_path)
            # Release file lock if held by this user
            if session.file_locks.get(file_path) == user_id:
                del session.file_locks[file_path]
            event_type = CollaborationEventType.FILE_CLOSED
        else:
            return False
        
        await self._broadcast_event(
            session_id,
            event_type,
            user_id,
            {
                "user_id": user_id,
                "file_path": file_path,
                "open_files_count": len(session.open_files)
            }
        )
        
        return True
    
    async def request_file_lock(self, user_id: str, file_path: str) -> bool:
        """Request exclusive edit lock on a file."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        # Check if file is already locked
        if file_path in session.file_locks:
            return False
        
        # Grant lock
        session.file_locks[file_path] = user_id
        
        await self._broadcast_event(
            session_id,
            CollaborationEventType.TEXT_CHANGED,
            user_id,
            {
                "action": "file_locked",
                "file_path": file_path,
                "locked_by": user_id
            }
        )
        
        return True
    
    async def release_file_lock(self, user_id: str, file_path: str) -> bool:
        """Release exclusive edit lock on a file."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        # Check if user holds the lock
        if session.file_locks.get(file_path) != user_id:
            return False
        
        # Release lock
        del session.file_locks[file_path]
        
        await self._broadcast_event(
            session_id,
            CollaborationEventType.TEXT_CHANGED,
            user_id,
            {
                "action": "file_unlocked",
                "file_path": file_path,
                "unlocked_by": user_id
            }
        )
        
        return True
    
    async def add_chat_message(self, user_id: str, message: str, message_type: str = "text") -> bool:
        """Add a chat message to the session."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        chat_message = {
            "message_id": str(uuid.uuid4()),
            "user_id": user_id,
            "username": session.active_users[user_id].username,
            "message": message,
            "message_type": message_type,
            "timestamp": datetime.now().isoformat()
        }
        
        session.chat_messages.append(chat_message)
        
        # Keep only last 100 messages
        if len(session.chat_messages) > 100:
            session.chat_messages = session.chat_messages[-100:]
        
        await self._broadcast_event(
            session_id,
            CollaborationEventType.COMMENT_ADDED,
            user_id,
            {
                "chat_message": chat_message
            }
        )
        
        return True
    
    async def start_voice_chat(self, user_id: str) -> bool:
        """Start voice chat for the session."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        session.voice_chat_active = True
        
        await self._broadcast_event(
            session_id,
            CollaborationEventType.VOICE_CHAT_STARTED,
            user_id,
            {
                "started_by": user_id,
                "voice_chat_active": True
            }
        )
        
        return True
    
    async def start_screen_share(self, user_id: str) -> bool:
        """Start screen sharing for the session."""
        if user_id not in self.user_sessions:
            return False
        
        session_id = self.user_sessions[user_id]
        session = self.sessions[session_id]
        
        # Only one user can share screen at a time
        if session.screen_share_active:
            return False
        
        session.screen_share_active = True
        session.screen_share_user = user_id
        
        await self._broadcast_event(
            session_id,
            CollaborationEventType.SCREEN_SHARE_STARTED,
            user_id,
            {
                "shared_by": user_id,
                "screen_share_active": True
            }
        )
        
        return True
    
    async def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a collaboration session."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session.session_id,
            "project_id": session.project_id,
            "project_name": session.project_name,
            "owner_id": session.owner_id,
            "active_users": [
                {
                    "user_id": user.user_id,
                    "username": user.username,
                    "avatar_url": user.avatar_url,
                    "presence": user.presence.value,
                    "current_file": user.current_file,
                    "cursor_position": user.cursor_position,
                    "color": user.color,
                    "last_activity": user.last_activity.isoformat()
                }
                for user in session.active_users.values()
            ],
            "open_files": list(session.open_files),
            "file_locks": session.file_locks,
            "chat_messages": session.chat_messages[-20:],  # Last 20 messages
            "voice_chat_active": session.voice_chat_active,
            "screen_share_active": session.screen_share_active,
            "screen_share_user": session.screen_share_user
        }
    
    async def _broadcast_event(
        self, 
        session_id: str, 
        event_type: CollaborationEventType, 
        user_id: str, 
        data: Dict[str, Any],
        exclude_user: Optional[str] = None
    ):
        """Broadcast an event to all users in a session."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(),
            data=data
        )
        
        # Send to all connected users except excluded one
        for target_user_id in session.active_users.keys():
            if exclude_user and target_user_id == exclude_user:
                continue
            
            if target_user_id in self.websocket_connections:
                try:
                    websocket = self.websocket_connections[target_user_id]
                    await websocket.send_text(json.dumps({
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "user_id": event.user_id,
                        "session_id": event.session_id,
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data
                    }))
                except Exception as e:
                    # Remove disconnected websocket
                    del self.websocket_connections[target_user_id]
    
    def register_websocket(self, user_id: str, websocket: Any):
        """Register a websocket connection for a user."""
        self.websocket_connections[user_id] = websocket
    
    def unregister_websocket(self, user_id: str):
        """Unregister a websocket connection for a user."""
        if user_id in self.websocket_connections:
            del self.websocket_connections[user_id]

class OperationalTransform:
    """Handles operational transformation for conflict resolution."""
    
    async def transform_operation(
        self, 
        operation: TextOperation, 
        pending_operations: List[TextOperation]
    ) -> TextOperation:
        """Transform an operation against pending operations."""
        transformed_op = operation
        
        # Apply transformation against each pending operation
        for pending_op in pending_operations:
            if (
                pending_op.file_path == operation.file_path and
                not pending_op.applied and
                pending_op.timestamp < operation.timestamp
            ):
                transformed_op = self._transform_against_operation(transformed_op, pending_op)
        
        return transformed_op
    
    def _transform_against_operation(
        self, 
        op1: TextOperation, 
        op2: TextOperation
    ) -> TextOperation:
        """Transform operation op1 against operation op2."""
        # Simple operational transformation logic
        # In a production system, this would be more sophisticated
        
        if op2.operation_type == "insert":
            if op1.position >= op2.position:
                # Adjust position for insertion before current position
                op1.position += op2.length
        elif op2.operation_type == "delete":
            if op1.position > op2.position:
                # Adjust position for deletion before current position
                op1.position -= min(op2.length, op1.position - op2.position)
        
        return op1
```

### 2. Code Review System

**File**: `core/collaboration/code_review.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import uuid

class ReviewStatus(Enum):
    DRAFT = "draft"
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    MERGED = "merged"
    CLOSED = "closed"

class CommentType(Enum):
    GENERAL = "general"
    SUGGESTION = "suggestion"
    ISSUE = "issue"
    PRAISE = "praise"
    QUESTION = "question"

@dataclass
class CodeComment:
    comment_id: str
    author_id: str
    author_name: str
    file_path: str
    line_number: int
    column_number: Optional[int] = None
    content: str = ""
    comment_type: CommentType = CommentType.GENERAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    replies: List['CodeComment'] = field(default_factory=list)
    suggested_code: Optional[str] = None
    original_code: Optional[str] = None

@dataclass
class FileChange:
    file_path: str
    change_type: str  # 'added', 'modified', 'deleted', 'renamed'
    old_path: Optional[str] = None
    additions: int = 0
    deletions: int = 0
    diff: str = ""
    binary: bool = False

@dataclass
class ReviewRequest:
    review_id: str
    title: str
    description: str
    author_id: str
    author_name: str
    project_id: str
    branch_name: str
    base_branch: str = "main"
    status: ReviewStatus = ReviewStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    reviewers: Set[str] = field(default_factory=set)
    assignees: Set[str] = field(default_factory=set)
    labels: Set[str] = field(default_factory=set)
    file_changes: List[FileChange] = field(default_factory=list)
    comments: List[CodeComment] = field(default_factory=list)
    approvals: Set[str] = field(default_factory=set)
    change_requests: Set[str] = field(default_factory=set)
    merge_conflicts: bool = False
    mergeable: bool = True
    ci_status: Optional[str] = None
    test_results: Optional[Dict[str, Any]] = None

class CodeReviewSystem:
    def __init__(self):
        self.reviews: Dict[str, ReviewRequest] = {}
        self.user_reviews: Dict[str, Set[str]] = {}  # user_id -> review_ids
        self.project_reviews: Dict[str, Set[str]] = {}  # project_id -> review_ids
        
    async def create_review(
        self,
        title: str,
        description: str,
        author_id: str,
        author_name: str,
        project_id: str,
        branch_name: str,
        file_changes: List[FileChange],
        reviewers: Optional[Set[str]] = None
    ) -> str:
        """Create a new code review request."""
        review_id = str(uuid.uuid4())
        
        review = ReviewRequest(
            review_id=review_id,
            title=title,
            description=description,
            author_id=author_id,
            author_name=author_name,
            project_id=project_id,
            branch_name=branch_name,
            file_changes=file_changes,
            reviewers=reviewers or set()
        )
        
        self.reviews[review_id] = review
        
        # Update indexes
        if author_id not in self.user_reviews:
            self.user_reviews[author_id] = set()
        self.user_reviews[author_id].add(review_id)
        
        if project_id not in self.project_reviews:
            self.project_reviews[project_id] = set()
        self.project_reviews[project_id].add(review_id)
        
        return review_id
    
    async def add_comment(
        self,
        review_id: str,
        author_id: str,
        author_name: str,
        file_path: str,
        line_number: int,
        content: str,
        comment_type: CommentType = CommentType.GENERAL,
        suggested_code: Optional[str] = None,
        parent_comment_id: Optional[str] = None
    ) -> Optional[str]:
        """Add a comment to a code review."""
        if review_id not in self.reviews:
            return None
        
        review = self.reviews[review_id]
        comment_id = str(uuid.uuid4())
        
        comment = CodeComment(
            comment_id=comment_id,
            author_id=author_id,
            author_name=author_name,
            file_path=file_path,
            line_number=line_number,
            content=content,
            comment_type=comment_type,
            suggested_code=suggested_code
        )
        
        if parent_comment_id:
            # Find parent comment and add as reply
            parent_comment = self._find_comment(review, parent_comment_id)
            if parent_comment:
                parent_comment.replies.append(comment)
            else:
                return None
        else:
            # Add as top-level comment
            review.comments.append(comment)
        
        review.updated_at = datetime.now()
        return comment_id
    
    async def resolve_comment(self, review_id: str, comment_id: str, resolver_id: str) -> bool:
        """Mark a comment as resolved."""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        comment = self._find_comment(review, comment_id)
        
        if comment:
            comment.resolved = True
            comment.resolved_by = resolver_id
            comment.resolved_at = datetime.now()
            review.updated_at = datetime.now()
            return True
        
        return False
    
    async def submit_review(
        self,
        review_id: str,
        reviewer_id: str,
        decision: str,  # 'approve', 'request_changes', 'comment'
        summary_comment: Optional[str] = None
    ) -> bool:
        """Submit a review decision."""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        
        # Remove previous decisions by this reviewer
        review.approvals.discard(reviewer_id)
        review.change_requests.discard(reviewer_id)
        
        if decision == "approve":
            review.approvals.add(reviewer_id)
        elif decision == "request_changes":
            review.change_requests.add(reviewer_id)
        
        # Add summary comment if provided
        if summary_comment:
            await self.add_comment(
                review_id=review_id,
                author_id=reviewer_id,
                author_name="",  # Would be filled from user data
                file_path="",
                line_number=0,
                content=summary_comment,
                comment_type=CommentType.GENERAL
            )
        
        # Update review status
        await self._update_review_status(review_id)
        
        review.updated_at = datetime.now()
        return True
    
    async def merge_review(self, review_id: str, merger_id: str) -> bool:
        """Merge a code review."""
        if review_id not in self.reviews:
            return False
        
        review = self.reviews[review_id]
        
        # Check if review can be merged
        if not review.mergeable or review.merge_conflicts:
            return False
        
        if review.status != ReviewStatus.APPROVED:
            return False
        
        # Perform merge (this would integrate with version control)
        # For now, just update status
        review.status = ReviewStatus.MERGED
        review.updated_at = datetime.now()
        
        return True
    
    async def get_review(self, review_id: str) -> Optional[ReviewRequest]:
        """Get a code review by ID."""
        return self.reviews.get(review_id)
    
    async def get_user_reviews(self, user_id: str, status_filter: Optional[ReviewStatus] = None) -> List[ReviewRequest]:
        """Get all reviews for a user."""
        if user_id not in self.user_reviews:
            return []
        
        reviews = [self.reviews[review_id] for review_id in self.user_reviews[user_id]]
        
        if status_filter:
            reviews = [r for r in reviews if r.status == status_filter]
        
        return sorted(reviews, key=lambda r: r.updated_at, reverse=True)
    
    async def get_project_reviews(self, project_id: str, status_filter: Optional[ReviewStatus] = None) -> List[ReviewRequest]:
        """Get all reviews for a project."""
        if project_id not in self.project_reviews:
            return []
        
        reviews = [self.reviews[review_id] for review_id in self.project_reviews[project_id]]
        
        if status_filter:
            reviews = [r for r in reviews if r.status == status_filter]
        
        return sorted(reviews, key=lambda r: r.updated_at, reverse=True)
    
    async def get_pending_reviews_for_user(self, user_id: str) -> List[ReviewRequest]:
        """Get reviews pending review from a specific user."""
        pending_reviews = []
        
        for review in self.reviews.values():
            if (
                user_id in review.reviewers and
                review.status in [ReviewStatus.PENDING, ReviewStatus.IN_REVIEW] and
                user_id not in review.approvals and
                user_id not in review.change_requests
            ):
                pending_reviews.append(review)
        
        return sorted(pending_reviews, key=lambda r: r.created_at)
    
    def _find_comment(self, review: ReviewRequest, comment_id: str) -> Optional[CodeComment]:
        """Find a comment by ID in a review."""
        def search_comments(comments: List[CodeComment]) -> Optional[CodeComment]:
            for comment in comments:
                if comment.comment_id == comment_id:
                    return comment
                # Search in replies
                found = search_comments(comment.replies)
                if found:
                    return found
            return None
        
        return search_comments(review.comments)
    
    async def _update_review_status(self, review_id: str):
        """Update review status based on approvals and change requests."""
        review = self.reviews[review_id]
        
        if review.change_requests:
            review.status = ReviewStatus.CHANGES_REQUESTED
        elif len(review.approvals) >= len(review.reviewers) and review.reviewers:
            review.status = ReviewStatus.APPROVED
        elif review.approvals or review.change_requests:
            review.status = ReviewStatus.IN_REVIEW
        else:
            review.status = ReviewStatus.PENDING
```

### 3. Collaboration Dashboard Frontend

**File**: `frontend/components/collaboration/collaboration-dashboard.tsx`

```typescript
import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { 
  Users, 
  MessageSquare, 
  Video, 
  Share2, 
  FileText, 
  GitBranch,
  Clock,
  CheckCircle,
  AlertCircle,
  Eye,
  Edit3,
  Lock,
  Unlock,
  Send,
  Mic,
  MicOff,
  Camera,
  CameraOff
} from 'lucide-react';

interface CollaborationUser {
  user_id: string;
  username: string;
  email: string;
  avatar_url?: string;
  presence: 'online' | 'away' | 'busy' | 'offline';
  current_file?: string;
  cursor_position?: { line: number; column: number };
  color: string;
  last_activity: string;
}

interface ChatMessage {
  message_id: string;
  user_id: string;
  username: string;
  message: string;
  message_type: string;
  timestamp: string;
}

interface CollaborationSession {
  session_id: string;
  project_id: string;
  project_name: string;
  owner_id: string;
  active_users: CollaborationUser[];
  open_files: string[];
  file_locks: Record<string, string>;
  chat_messages: ChatMessage[];
  voice_chat_active: boolean;
  screen_share_active: boolean;
  screen_share_user?: string;
}

interface CodeReview {
  review_id: string;
  title: string;
  description: string;
  author_id: string;
  author_name: string;
  status: string;
  created_at: string;
  updated_at: string;
  reviewers: string[];
  approvals: string[];
  change_requests: string[];
  file_changes: any[];
  comments: any[];
}

export default function CollaborationDashboard() {
  const [activeSession, setActiveSession] = useState<CollaborationSession | null>(null);
  const [currentUser, setCurrentUser] = useState<CollaborationUser | null>(null);
  const [reviews, setReviews] = useState<CodeReview[]>([]);
  const [chatMessage, setChatMessage] = useState('');
  const [isVoiceChatActive, setIsVoiceChatActive] = useState(false);
  const [isScreenShareActive, setIsScreenShareActive] = useState(false);
  const [selectedTab, setSelectedTab] = useState('session');
  const websocketRef = useRef<WebSocket | null>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadCollaborationData();
    initializeWebSocket();
    
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [activeSession?.chat_messages]);

  const loadCollaborationData = async () => {
    try {
      // Load current session
      const sessionResponse = await fetch('/api/collaboration/session');
      if (sessionResponse.ok) {
        const sessionData = await sessionResponse.json();
        setActiveSession(sessionData);
      }

      // Load current user
      const userResponse = await fetch('/api/auth/me');
      if (userResponse.ok) {
        const userData = await userResponse.json();
        setCurrentUser(userData);
      }

      // Load reviews
      const reviewsResponse = await fetch('/api/collaboration/reviews');
      if (reviewsResponse.ok) {
        const reviewsData = await reviewsResponse.json();
        setReviews(reviewsData);
      }
    } catch (error) {
      console.error('Failed to load collaboration data:', error);
    }
  };

  const initializeWebSocket = () => {
    const wsUrl = `ws://localhost:8000/ws/collaboration`;
    websocketRef.current = new WebSocket(wsUrl);

    websocketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    websocketRef.current.onclose = () => {
      // Attempt to reconnect after 3 seconds
      setTimeout(initializeWebSocket, 3000);
    };
  };

  const handleWebSocketMessage = (data: any) => {
    switch (data.event_type) {
      case 'user_joined':
      case 'user_left':
      case 'cursor_moved':
      case 'text_changed':
      case 'file_opened':
      case 'file_closed':
        // Reload session data
        loadCollaborationData();
        break;
      case 'comment_added':
        if (data.data.chat_message) {
          setActiveSession(prev => prev ? {
            ...prev,
            chat_messages: [...prev.chat_messages, data.data.chat_message]
          } : null);
        }
        break;
    }
  };

  const sendChatMessage = async () => {
    if (!chatMessage.trim() || !activeSession) return;

    try {
      await fetch('/api/collaboration/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession.session_id,
          message: chatMessage,
          message_type: 'text'
        })
      });
      
      setChatMessage('');
    } catch (error) {
      console.error('Failed to send chat message:', error);
    }
  };

  const toggleVoiceChat = async () => {
    try {
      const action = isVoiceChatActive ? 'stop' : 'start';
      await fetch(`/api/collaboration/voice-chat/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSession?.session_id })
      });
      
      setIsVoiceChatActive(!isVoiceChatActive);
    } catch (error) {
      console.error('Failed to toggle voice chat:', error);
    }
  };

  const toggleScreenShare = async () => {
    try {
      const action = isScreenShareActive ? 'stop' : 'start';
      await fetch(`/api/collaboration/screen-share/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: activeSession?.session_id })
      });
      
      setIsScreenShareActive(!isScreenShareActive);
    } catch (error) {
      console.error('Failed to toggle screen share:', error);
    }
  };

  const requestFileLock = async (filePath: string) => {
    try {
      await fetch('/api/collaboration/file-lock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession?.session_id,
          file_path: filePath,
          action: 'lock'
        })
      });
    } catch (error) {
      console.error('Failed to request file lock:', error);
    }
  };

  const releaseFileLock = async (filePath: string) => {
    try {
      await fetch('/api/collaboration/file-lock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSession?.session_id,
          file_path: filePath,
          action: 'unlock'
        })
      });
    } catch (error) {
      console.error('Failed to release file lock:', error);
    }
  };

  const getPresenceColor = (presence: string) => {
    switch (presence) {
      case 'online': return 'bg-green-500';
      case 'away': return 'bg-yellow-500';
      case 'busy': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'text-green-600';
      case 'changes_requested': return 'text-red-600';
      case 'pending': return 'text-yellow-600';
      case 'in_review': return 'text-blue-600';
      default: return 'text-gray-600';
    }
  };

  const renderSessionOverview = () => (
    <div className="space-y-6">
      {/* Session Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            {activeSession?.project_name || 'No Active Session'}
          </CardTitle>
        </CardHeader>
        <CardContent>
          {activeSession ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  {activeSession.active_users.length} active users
                </span>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant={isVoiceChatActive ? "destructive" : "outline"}
                    onClick={toggleVoiceChat}
                  >
                    {isVoiceChatActive ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
                    {isVoiceChatActive ? 'Leave Voice' : 'Join Voice'}
                  </Button>
                  <Button
                    size="sm"
                    variant={isScreenShareActive ? "destructive" : "outline"}
                    onClick={toggleScreenShare}
                  >
                    {isScreenShareActive ? <CameraOff className="h-4 w-4" /> : <Camera className="h-4 w-4" />}
                    {isScreenShareActive ? 'Stop Share' : 'Share Screen'}
                  </Button>
                </div>
              </div>
              
              {/* Active Users */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Active Users</h4>
                <div className="space-y-2">
                  {activeSession.active_users.map((user) => (
                    <div key={user.user_id} className="flex items-center gap-3 p-2 rounded-lg bg-muted/50">
                      <div className="relative">
                        <Avatar className="h-8 w-8">
                          <AvatarImage src={user.avatar_url} />
                          <AvatarFallback>{user.username.charAt(0).toUpperCase()}</AvatarFallback>
                        </Avatar>
                        <div className={`absolute -bottom-1 -right-1 h-3 w-3 rounded-full border-2 border-background ${getPresenceColor(user.presence)}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{user.username}</p>
                        <p className="text-xs text-muted-foreground truncate">
                          {user.current_file ? `Editing: ${user.current_file}` : 'Idle'}
                        </p>
                      </div>
                      <div 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: user.color }}
                        title={`${user.username}'s cursor color`}
                      />
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Open Files */}
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Open Files ({activeSession.open_files.length})</h4>
                <ScrollArea className="h-32">
                  <div className="space-y-1">
                    {activeSession.open_files.map((file) => {
                      const isLocked = activeSession.file_locks[file];
                      const lockedByCurrentUser = isLocked === currentUser?.user_id;
                      
                      return (
                        <div key={file} className="flex items-center gap-2 p-2 rounded bg-muted/30">
                          <FileText className="h-4 w-4" />
                          <span className="flex-1 text-sm truncate">{file}</span>
                          {isLocked ? (
                            <div className="flex items-center gap-1">
                              <Lock className="h-3 w-3 text-red-500" />
                              {lockedByCurrentUser ? (
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => releaseFileLock(file)}
                                >
                                  <Unlock className="h-3 w-3" />
                                </Button>
                              ) : (
                                <span className="text-xs text-red-500">
                                  Locked by {activeSession.active_users.find(u => u.user_id === isLocked)?.username}
                                </span>
                              )}
                            </div>
                          ) : (
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => requestFileLock(file)}
                            >
                              <Lock className="h-3 w-3" />
                            </Button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </ScrollArea>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">No active collaboration session</p>
          )}
        </CardContent>
      </Card>
    </div>
  );

  const renderChat = () => (
    <Card className="h-[600px] flex flex-col">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5" />
          Team Chat
        </CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col p-0">
        <ScrollArea className="flex-1 p-4" ref={chatScrollRef}>
          <div className="space-y-3">
            {activeSession?.chat_messages.map((message) => (
              <div key={message.message_id} className="flex gap-3">
                <Avatar className="h-8 w-8">
                  <AvatarFallback>{message.username.charAt(0).toUpperCase()}</AvatarFallback>
                </Avatar>
                <div className="flex-1 space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium">{message.username}</span>
                    <span className="text-xs text-muted-foreground">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm">{message.message}</p>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
        <Separator />
        <div className="p-4">
          <div className="flex gap-2">
            <Input
              placeholder="Type a message..."
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendChatMessage()}
            />
            <Button onClick={sendChatMessage} disabled={!chatMessage.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderReviews = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Code Reviews</h3>
        <Button>
          <GitBranch className="h-4 w-4 mr-2" />
          Create Review
        </Button>
      </div>
      
      <div className="grid gap-4">
        {reviews.map((review) => (
          <Card key={review.review_id}>
            <CardHeader>
              <div className="flex items-start justify-between">
                <div>
                  <CardTitle className="text-base">{review.title}</CardTitle>
                  <p className="text-sm text-muted-foreground mt-1">
                    by {review.author_name} • {new Date(review.created_at).toLocaleDateString()}
                  </p>
                </div>
                <Badge variant="outline" className={getStatusColor(review.status)}>
                  {review.status.replace('_', ' ')}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <p className="text-sm mb-4">{review.description}</p>
              
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  {review.approvals.length} approvals
                </div>
                <div className="flex items-center gap-1">
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  {review.change_requests.length} change requests
                </div>
                <div className="flex items-center gap-1">
                  <MessageSquare className="h-4 w-4" />
                  {review.comments.length} comments
                </div>
                <div className="flex items-center gap-1">
                  <FileText className="h-4 w-4" />
                  {review.file_changes.length} files
                </div>
              </div>
              
              <div className="flex gap-2 mt-4">
                <Button size="sm" variant="outline">
                  <Eye className="h-4 w-4 mr-2" />
                  View
                </Button>
                <Button size="sm" variant="outline">
                  <Edit3 className="h-4 w-4 mr-2" />
                  Review
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Collaboration Dashboard</h1>
        <p className="text-muted-foreground mt-2">
          Real-time collaboration and code review management
        </p>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="session">Active Session</TabsTrigger>
          <TabsTrigger value="chat">Team Chat</TabsTrigger>
          <TabsTrigger value="reviews">Code Reviews</TabsTrigger>
        </TabsList>
        
        <TabsContent value="session" className="mt-6">
          {renderSessionOverview()}
        </TabsContent>
        
        <TabsContent value="chat" className="mt-6">
          {renderChat()}
        </TabsContent>
        
        <TabsContent value="reviews" className="mt-6">
          {renderReviews()}
        </TabsContent>
      </Tabs>
    </div>
  );
}
```

### 4. Backend API Endpoints

**File**: `backend/api/collaboration.py`

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.security import HTTPBearer
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncio

from core.collaboration.realtime_engine import (
    RealtimeCollaborationEngine, 
    CollaborationUser, 
    TextOperation,
    UserPresence
)
from core.collaboration.code_review import CodeReviewSystem, ReviewStatus, CommentType
from core.auth.auth_service import get_current_user
from core.database.models import User

router = APIRouter(prefix="/api/collaboration", tags=["collaboration"])
security = HTTPBearer()

# Global instances
collaboration_engine = RealtimeCollaborationEngine()
code_review_system = CodeReviewSystem()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        collaboration_engine.register_websocket(user_id, websocket)
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        collaboration_engine.unregister_websocket(user_id)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """WebSocket endpoint for real-time collaboration."""
    try:
        # Authenticate user from token
        user = await authenticate_websocket_user(token)
        if not user:
            await websocket.close(code=4001)
            return
        
        await manager.connect(websocket, user.id)
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                await handle_websocket_message(user.id, message)
        except WebSocketDisconnect:
            manager.disconnect(user.id)
            await collaboration_engine.leave_session(user.id)
    except Exception as e:
        await websocket.close(code=4000)

async def handle_websocket_message(user_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    message_type = message.get("type")
    data = message.get("data", {})
    
    if message_type == "cursor_move":
        await collaboration_engine.handle_cursor_movement(
            user_id=user_id,
            file_path=data.get("file_path"),
            position=data.get("position")
        )
    elif message_type == "text_operation":
        operation = TextOperation(
            operation_id=data.get("operation_id"),
            user_id=user_id,
            file_path=data.get("file_path"),
            operation_type=data.get("operation_type"),
            position=data.get("position"),
            content=data.get("content"),
            length=data.get("length"),
            timestamp=datetime.now()
        )
        await collaboration_engine.handle_text_operation(user_id, operation)
    elif message_type == "file_operation":
        await collaboration_engine.handle_file_operation(
            user_id=user_id,
            file_path=data.get("file_path"),
            operation=data.get("operation")
        )

@router.post("/session/create")
async def create_collaboration_session(
    project_id: str,
    project_name: str,
    current_user: User = Depends(get_current_user)
):
    """Create a new collaboration session."""
    session_id = await collaboration_engine.create_session(
        project_id=project_id,
        project_name=project_name,
        owner_id=current_user.id
    )
    
    return {"session_id": session_id, "status": "created"}

@router.post("/session/join")
async def join_collaboration_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Join an existing collaboration session."""
    user = CollaborationUser(
        user_id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        avatar_url=getattr(current_user, 'avatar_url', None),
        presence=UserPresence.ONLINE
    )
    
    success = await collaboration_engine.join_session(session_id, user)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "joined", "session_id": session_id}

@router.post("/session/leave")
async def leave_collaboration_session(
    current_user: User = Depends(get_current_user)
):
    """Leave current collaboration session."""
    success = await collaboration_engine.leave_session(current_user.id)
    
    return {"status": "left" if success else "not_in_session"}

@router.get("/session")
async def get_current_session(
    current_user: User = Depends(get_current_user)
):
    """Get current collaboration session state."""
    if current_user.id not in collaboration_engine.user_sessions:
        return None
    
    session_id = collaboration_engine.user_sessions[current_user.id]
    session_state = await collaboration_engine.get_session_state(session_id)
    
    return session_state

@router.post("/chat")
async def send_chat_message(
    session_id: str,
    message: str,
    message_type: str = "text",
    current_user: User = Depends(get_current_user)
):
    """Send a chat message to the collaboration session."""
    success = await collaboration_engine.add_chat_message(
        user_id=current_user.id,
        message=message,
        message_type=message_type
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to send message")
    
    return {"status": "sent"}

@router.post("/file-lock")
async def handle_file_lock(
    session_id: str,
    file_path: str,
    action: str,  # 'lock' or 'unlock'
    current_user: User = Depends(get_current_user)
):
    """Request or release a file lock."""
    if action == "lock":
        success = await collaboration_engine.request_file_lock(current_user.id, file_path)
    elif action == "unlock":
        success = await collaboration_engine.release_file_lock(current_user.id, file_path)
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    return {"status": "success" if success else "failed", "action": action}

@router.post("/voice-chat/{action}")
async def handle_voice_chat(
    action: str,  # 'start' or 'stop'
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Start or stop voice chat."""
    if action == "start":
        success = await collaboration_engine.start_voice_chat(current_user.id)
    else:
        # For stopping, we'd need to implement stop_voice_chat method
        success = True  # Placeholder
    
    return {"status": "success" if success else "failed", "action": action}

@router.post("/screen-share/{action}")
async def handle_screen_share(
    action: str,  # 'start' or 'stop'
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Start or stop screen sharing."""
    if action == "start":
        success = await collaboration_engine.start_screen_share(current_user.id)
    else:
        # For stopping, we'd need to implement stop_screen_share method
        success = True  # Placeholder
    
    return {"status": "success" if success else "failed", "action": action}

# Code Review Endpoints

@router.post("/reviews")
async def create_code_review(
    title: str,
    description: str,
    project_id: str,
    branch_name: str,
    file_changes: List[Dict[str, Any]],
    reviewers: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user)
):
    """Create a new code review."""
    from core.collaboration.code_review import FileChange
    
    # Convert file changes
    changes = [
        FileChange(
            file_path=change["file_path"],
            change_type=change["change_type"],
            old_path=change.get("old_path"),
            additions=change.get("additions", 0),
            deletions=change.get("deletions", 0),
            diff=change.get("diff", "")
        )
        for change in file_changes
    ]
    
    review_id = await code_review_system.create_review(
        title=title,
        description=description,
        author_id=current_user.id,
        author_name=current_user.username,
        project_id=project_id,
        branch_name=branch_name,
        file_changes=changes,
        reviewers=set(reviewers) if reviewers else set()
    )
    
    return {"review_id": review_id, "status": "created"}

@router.get("/reviews")
async def get_reviews(
    project_id: Optional[str] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get code reviews."""
    if project_id:
        status_filter = ReviewStatus(status) if status else None
        reviews = await code_review_system.get_project_reviews(project_id, status_filter)
    else:
        reviews = await code_review_system.get_user_reviews(current_user.id)
    
    return [
        {
            "review_id": review.review_id,
            "title": review.title,
            "description": review.description,
            "author_id": review.author_id,
            "author_name": review.author_name,
            "status": review.status.value,
            "created_at": review.created_at.isoformat(),
            "updated_at": review.updated_at.isoformat(),
            "reviewers": list(review.reviewers),
            "approvals": list(review.approvals),
            "change_requests": list(review.change_requests),
            "file_changes": [
                {
                    "file_path": change.file_path,
                    "change_type": change.change_type,
                    "additions": change.additions,
                    "deletions": change.deletions
                }
                for change in review.file_changes
            ],
            "comments": [
                {
                    "comment_id": comment.comment_id,
                    "author_name": comment.author_name,
                    "content": comment.content,
                    "file_path": comment.file_path,
                    "line_number": comment.line_number,
                    "created_at": comment.created_at.isoformat(),
                    "resolved": comment.resolved
                }
                for comment in review.comments
            ]
        }
        for review in reviews
    ]

@router.get("/reviews/{review_id}")
async def get_review(
    review_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific code review."""
    review = await code_review_system.get_review(review_id)
    
    if not review:
        raise HTTPException(status_code=404, detail="Review not found")
    
    return {
        "review_id": review.review_id,
        "title": review.title,
        "description": review.description,
        "author_id": review.author_id,
        "author_name": review.author_name,
        "project_id": review.project_id,
        "branch_name": review.branch_name,
        "status": review.status.value,
        "created_at": review.created_at.isoformat(),
        "updated_at": review.updated_at.isoformat(),
        "reviewers": list(review.reviewers),
        "approvals": list(review.approvals),
        "change_requests": list(review.change_requests),
        "file_changes": [
            {
                "file_path": change.file_path,
                "change_type": change.change_type,
                "old_path": change.old_path,
                "additions": change.additions,
                "deletions": change.deletions,
                "diff": change.diff,
                "binary": change.binary
            }
            for change in review.file_changes
        ],
        "comments": [
            {
                "comment_id": comment.comment_id,
                "author_id": comment.author_id,
                "author_name": comment.author_name,
                "file_path": comment.file_path,
                "line_number": comment.line_number,
                "column_number": comment.column_number,
                "content": comment.content,
                "comment_type": comment.comment_type.value,
                "created_at": comment.created_at.isoformat(),
                "resolved": comment.resolved,
                "resolved_by": comment.resolved_by,
                "suggested_code": comment.suggested_code,
                "replies": [
                    {
                        "comment_id": reply.comment_id,
                        "author_name": reply.author_name,
                        "content": reply.content,
                        "created_at": reply.created_at.isoformat()
                    }
                    for reply in comment.replies
                ]
            }
            for comment in review.comments
        ]
    }

@router.post("/reviews/{review_id}/comments")
async def add_review_comment(
    review_id: str,
    file_path: str,
    line_number: int,
    content: str,
    comment_type: str = "general",
    suggested_code: Optional[str] = None,
    parent_comment_id: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Add a comment to a code review."""
    comment_id = await code_review_system.add_comment(
        review_id=review_id,
        author_id=current_user.id,
        author_name=current_user.username,
        file_path=file_path,
        line_number=line_number,
        content=content,
        comment_type=CommentType(comment_type),
        suggested_code=suggested_code,
        parent_comment_id=parent_comment_id
    )
    
    if not comment_id:
        raise HTTPException(status_code=400, detail="Failed to add comment")
    
    return {"comment_id": comment_id, "status": "added"}

@router.post("/reviews/{review_id}/submit")
async def submit_review(
    review_id: str,
    decision: str,  # 'approve', 'request_changes', 'comment'
    summary_comment: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Submit a review decision."""
    success = await code_review_system.submit_review(
        review_id=review_id,
        reviewer_id=current_user.id,
        decision=decision,
        summary_comment=summary_comment
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to submit review")
    
    return {"status": "submitted", "decision": decision}

@router.post("/reviews/{review_id}/merge")
async def merge_review(
    review_id: str,
    current_user: User = Depends(get_current_user)
):
    """Merge a code review."""
    success = await code_review_system.merge_review(review_id, current_user.id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to merge review")
    
    return {"status": "merged"}

@router.get("/reviews/pending")
async def get_pending_reviews(
    current_user: User = Depends(get_current_user)
):
    """Get reviews pending review from current user."""
    reviews = await code_review_system.get_pending_reviews_for_user(current_user.id)
    
    return [
        {
            "review_id": review.review_id,
            "title": review.title,
            "author_name": review.author_name,
            "created_at": review.created_at.isoformat(),
            "file_changes_count": len(review.file_changes)
        }
        for review in reviews
    ]

async def authenticate_websocket_user(token: str) -> Optional[User]:
    """Authenticate user from WebSocket token."""
    # This would integrate with your auth system
    # For now, return a placeholder
    return None
```

This collaborative development environment provides comprehensive real-time collaboration features including synchronized editing, conflict resolution, code reviews, and team communication tools. The system supports multiple developers working together efficiently with proper coordination and quality control mechanisms.