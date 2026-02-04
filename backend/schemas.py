from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import enum

# Enums
class UserRole(str, enum.Enum):
    DEV = "DEV"
    TESTER = "TESTER"
    PO = "PO"
    PM = "PM"
    RE = "RE"

class TaskPriority(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class TaskStatus(str, enum.Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    IN_REVIEW = "IN_REVIEW"
    DONE = "DONE"

# User Schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr
    role: UserRole

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_verified: bool = False
    # Gamification fields
    xp: int = 0
    level: int = 1
    total_tasks_completed: int = 0
    
    class Config:
        orm_mode = True

class EmailVerification(BaseModel):
    email: EmailStr
    token: str
    created_at: datetime
    expires_at: datetime

# Project Schemas
class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectCreate(ProjectBase):
    pass

class Project(ProjectBase):
    id: int
    created_at: datetime
    creator_id: int

    class Config:
        orm_mode = True

# Task Schemas
class TaskBase(BaseModel):
    title: str
    description: str
    priority: Optional[TaskPriority] = None # Optional because AI might set it
    status: TaskStatus = TaskStatus.TODO
    estimated_days: float = 0.0
    spent_days: float = 0.0
    project_id: Optional[int] = None
    dependencies: Optional[List[int]] = []  # List of task IDs this task depends on
    parent_id: Optional[int] = None  # Parent task ID for subtask hierarchy
    labels: Optional[List[int]] = []  # List of label IDs
    team_id: Optional[int] = None  # Team assignment

class TaskCreate(TaskBase):
    assignee_id: Optional[int] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None
    assignee_id: Optional[int] = None
    estimated_days: Optional[float] = None
    spent_days: Optional[float] = None
    project_id: Optional[int] = None
    dependencies: Optional[List[int]] = None  # Update dependencies
    parent_id: Optional[int] = None  # Update parent task
    labels: Optional[List[int]] = None  # Update labels
    team_id: Optional[int] = None  # Update team

class Task(TaskBase):
    id: int
    created_at: datetime
    creator_id: int
    assignee_id: Optional[int] = None
    dependencies: Optional[List[int]] = []  # List of dependency task IDs
    parent_id: Optional[int] = None
    labels: Optional[List[int]] = []
    team_id: Optional[int] = None
    
    class Config:
        orm_mode = True

# Token Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

# Comment Schemas
class CommentBase(BaseModel):
    content: str
    attachments: Optional[List[str]] = [] # List of file URLs/paths

class CommentCreate(CommentBase):
    pass

class Comment(CommentBase):
    id: int
    task_id: int
    user_id: int
    username: str # Denormalized for easier display
    created_at: datetime

    class Config:
        orm_mode = True

# Codebase RAG Query Schema
class CodebaseQuery(BaseModel):
    question: str

# Label Schemas
class LabelBase(BaseModel):
    name: str
    color: str  # Hex color like "#FF5733"

class LabelCreate(LabelBase):
    pass

class Label(LabelBase):
    id: int
    
    class Config:
        orm_mode = True

# Team Schemas
class TeamBase(BaseModel):
    name: str
    description: Optional[str] = None

class TeamCreate(TeamBase):
    pass

class Team(TeamBase):
    id: int
    created_at: datetime
    creator_id: int
    
    class Config:
        orm_mode = True
