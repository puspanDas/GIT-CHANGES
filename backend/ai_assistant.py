"""
AI Pair Programming Assistant Service
Provides code suggestions, implementation hints, and complexity analysis for tasks
"""
from typing import Dict, List, Any, Optional, Set
import json_storage
import re

# ==================== CONSTANTS ====================
# Using constants instead of magic numbers for maintainability
MAX_SUGGESTIONS = 6
MAX_CODE_HINTS = 3
MAX_RESOURCES = 4
MAX_SUBTASKS = 6
MAX_COMPLEXITY_SCORE = 10

# Complexity score weights
HIGH_COMPLEXITY_WEIGHT = 3
MEDIUM_COMPLEXITY_WEIGHT = 2
LOW_COMPLEXITY_WEIGHT = 1

# Complexity thresholds
HIGH_COMPLEXITY_THRESHOLD = 6
MEDIUM_COMPLEXITY_THRESHOLD = 3
HIGH_COUNT_THRESHOLD = 2
MEDIUM_COUNT_THRESHOLD = 2

# Keywords that indicate different types of development tasks
TASK_PATTERNS = {
    "authentication": {
        "keywords": ["auth", "login", "logout", "jwt", "token", "session", "password", "oauth"],
        "suggestions": [
            "Use bcrypt or argon2 for password hashing",
            "Implement JWT with refresh tokens for stateless authentication",
            "Add rate limiting to prevent brute force attacks",
            "Store tokens securely using httpOnly cookies"
        ],
        "code_hints": [
            {
                "language": "python",
                "title": "JWT Token Generation",
                "code": """from jose import jwt
from datetime import datetime, timedelta

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")"""
            }
        ],
        "resources": [
            {"title": "OWASP Authentication Cheatsheet", "url": "https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html"},
            {"title": "JWT Best Practices", "url": "https://auth0.com/blog/a-look-at-the-latest-draft-for-jwt-bcp/"}
        ]
    },
    "database": {
        "keywords": ["database", "db", "sql", "query", "migration", "orm", "model", "schema", "table"],
        "suggestions": [
            "Use parameterized queries to prevent SQL injection",
            "Implement database migrations for schema changes",
            "Add indexes on frequently queried columns",
            "Consider connection pooling for better performance"
        ],
        "code_hints": [
            {
                "language": "python",
                "title": "SQLAlchemy Model Example",
                "code": """from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)"""
            }
        ],
        "resources": [
            {"title": "SQLAlchemy Documentation", "url": "https://docs.sqlalchemy.org/"},
            {"title": "Database Design Best Practices", "url": "https://www.postgresql.org/docs/current/ddl.html"}
        ]
    },
    "api": {
        "keywords": ["api", "endpoint", "rest", "graphql", "route", "request", "response", "http"],
        "suggestions": [
            "Follow RESTful naming conventions (nouns, not verbs)",
            "Implement proper HTTP status codes",
            "Add request validation and error handling",
            "Document endpoints with OpenAPI/Swagger"
        ],
        "code_hints": [
            {
                "language": "python",
                "title": "FastAPI Endpoint Example",
                "code": """from fastapi import APIRouter, HTTPException, Depends
from typing import List

router = APIRouter(prefix="/api/v1", tags=["items"])

@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item"""
            }
        ],
        "resources": [
            {"title": "REST API Design Guide", "url": "https://restfulapi.net/"},
            {"title": "FastAPI Documentation", "url": "https://fastapi.tiangolo.com/"}
        ]
    },
    "frontend": {
        "keywords": ["ui", "component", "react", "vue", "angular", "css", "style", "button", "form", "page"],
        "suggestions": [
            "Break down UI into reusable components",
            "Implement proper loading and error states",
            "Use CSS-in-JS or CSS modules for scoped styling",
            "Add accessibility attributes (ARIA labels)"
        ],
        "code_hints": [
            {
                "language": "javascript",
                "title": "React Component Pattern",
                "code": """import React, { useState, useEffect } from 'react';

const DataComponent = ({ id }) => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchData(id)
            .then(setData)
            .catch(setError)
            .finally(() => setLoading(false));
    }, [id]);

    if (loading) return <LoadingSpinner />;
    if (error) return <ErrorMessage error={error} />;
    return <DataDisplay data={data} />;
};"""
            }
        ],
        "resources": [
            {"title": "React Documentation", "url": "https://react.dev/"},
            {"title": "Web Accessibility Guidelines", "url": "https://www.w3.org/WAI/WCAG21/quickref/"}
        ]
    },
    "testing": {
        "keywords": ["test", "unit", "integration", "mock", "assert", "coverage", "pytest", "jest"],
        "suggestions": [
            "Follow AAA pattern: Arrange, Act, Assert",
            "Mock external dependencies in unit tests",
            "Aim for high coverage on critical paths",
            "Use descriptive test names that explain the scenario"
        ],
        "code_hints": [
            {
                "language": "python",
                "title": "Pytest Test Example",
                "code": """import pytest
from unittest.mock import Mock, patch

class TestUserService:
    def test_create_user_with_valid_data(self, mock_db):
        # Arrange
        user_data = {"username": "test", "email": "test@example.com"}
        
        # Act
        result = UserService.create_user(user_data)
        
        # Assert
        assert result.username == "test"
        mock_db.add.assert_called_once()"""
            }
        ],
        "resources": [
            {"title": "Pytest Documentation", "url": "https://docs.pytest.org/"},
            {"title": "Testing Best Practices", "url": "https://martinfowler.com/articles/practical-test-pyramid.html"}
        ]
    },
    "performance": {
        "keywords": ["performance", "optimize", "speed", "cache", "slow", "fast", "efficient", "memory"],
        "suggestions": [
            "Profile before optimizing to find actual bottlenecks",
            "Implement caching for expensive operations",
            "Use pagination for large data sets",
            "Consider lazy loading for non-critical resources"
        ],
        "code_hints": [
            {
                "language": "python",
                "title": "Caching with Redis",
                "code": """from functools import lru_cache
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def get_cached_data(key: str, fetch_func, ttl: int = 300):
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    data = fetch_func()
    redis_client.setex(key, ttl, json.dumps(data))
    return data"""
            }
        ],
        "resources": [
            {"title": "Python Performance Tips", "url": "https://wiki.python.org/moin/PythonSpeed/PerformanceTips"},
            {"title": "Redis Caching Guide", "url": "https://redis.io/docs/manual/patterns/"}
        ]
    },
    "security": {
        "keywords": ["security", "vulnerability", "xss", "csrf", "injection", "encrypt", "secure", "sanitize"],
        "suggestions": [
            "Never trust user input - validate and sanitize everything",
            "Use HTTPS for all communications",
            "Implement CORS properly for API endpoints",
            "Keep dependencies updated to patch vulnerabilities"
        ],
        "code_hints": [
            {
                "language": "python",
                "title": "Input Validation",
                "code": """from pydantic import BaseModel, validator, EmailStr
import bleach

class UserInput(BaseModel):
    username: str
    email: EmailStr
    content: str

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v

    @validator('content')
    def sanitize_content(cls, v):
        return bleach.clean(v, strip=True)"""
            }
        ],
        "resources": [
            {"title": "OWASP Top 10", "url": "https://owasp.org/www-project-top-ten/"},
            {"title": "Security Headers Guide", "url": "https://securityheaders.com/"}
        ]
    }
}

# Complexity indicators
COMPLEXITY_INDICATORS = {
    "high": ["complex", "refactor", "redesign", "architecture", "migration", "integration", "multiple", "system"],
    "medium": ["implement", "create", "build", "add", "update", "modify", "feature"],
    "low": ["fix", "change", "update", "simple", "minor", "typo", "rename", "move"]
}


def analyze_task_keywords(text: str) -> List[str]:
    """
    Extract relevant keywords from task text using optimized Set-based matching.
    Time Complexity: O(k) where k = total keywords across all categories
    """
    text_lower = text.lower()
    # Use set for O(1) membership check when building result
    matched_categories: Set[str] = set()
    
    for category, data in TASK_PATTERNS.items():
        # Convert to set for faster lookup if category has many keywords
        keywords = data["keywords"]
        for keyword in keywords:
            if keyword in text_lower:
                matched_categories.add(category)
                break  # Early exit on first match
    
    return list(matched_categories)


def estimate_complexity(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate task complexity based on description and title.
    Optimized to use Set intersection for O(min(n,m)) complexity.
    """
    text = f"{task.get('title', '')} {task.get('description', '')}".lower()
    
    # Use set intersection for efficient counting - O(n) single pass
    text_words = set(text.split())
    
    # Convert indicator lists to sets for O(1) intersection
    high_set = set(COMPLEXITY_INDICATORS["high"])
    medium_set = set(COMPLEXITY_INDICATORS["medium"])
    low_set = set(COMPLEXITY_INDICATORS["low"])
    
    # Count matches using set intersection
    high_count = len(text_words & high_set)
    medium_count = len(text_words & medium_set)
    low_count = len(text_words & low_set)
    
    # Also check for partial matches (substring) for compound words
    for word in text_words:
        for indicator in high_set:
            if indicator in word and indicator != word:
                high_count += 1
                break
    
    # Calculate complexity score using constants
    score = (
        (high_count * HIGH_COMPLEXITY_WEIGHT) + 
        (medium_count * MEDIUM_COMPLEXITY_WEIGHT) + 
        (low_count * LOW_COMPLEXITY_WEIGHT)
    )
    
    # Determine level using constants
    if high_count >= HIGH_COUNT_THRESHOLD or score >= HIGH_COMPLEXITY_THRESHOLD:
        level = "high"
        estimated_hours = "8-16 hours"
        color = "red"
    elif medium_count >= MEDIUM_COUNT_THRESHOLD or score >= MEDIUM_COMPLEXITY_THRESHOLD:
        level = "medium"
        estimated_hours = "4-8 hours"
        color = "yellow"
    else:
        level = "low"
        estimated_hours = "1-4 hours"
        color = "green"
    
    # Get matched categories for additional context
    categories = analyze_task_keywords(text)
    
    return {
        "level": level,
        "score": min(score, MAX_COMPLEXITY_SCORE),
        "estimated_hours": estimated_hours,
        "color": color,
        "categories": categories,
        "reasoning": f"Found {high_count} high-complexity, {medium_count} medium-complexity, and {low_count} low-complexity indicators"
    }


def generate_code_suggestions(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate AI code suggestions for a task.
    Uses pattern matching to provide relevant suggestions, code hints, and resources.
    """
    text = f"{task.get('title', '')} {task.get('description', '')}".lower()
    categories = analyze_task_keywords(text)
    
    suggestions: List[str] = []
    code_hints: List[Dict] = []
    resources: List[Dict] = []
    
    # Gather relevant suggestions from matched categories
    for category in categories:
        if category in TASK_PATTERNS:
            pattern = TASK_PATTERNS[category]
            suggestions.extend(pattern.get("suggestions", []))
            code_hints.extend(pattern.get("code_hints", []))
            resources.extend(pattern.get("resources", []))
    
    # Add general suggestions if no specific matches
    if not suggestions:
        suggestions = [
            "Break down the task into smaller subtasks",
            "Write tests before implementing (TDD approach)",
            "Document your implementation decisions",
            "Consider edge cases and error handling"
        ]
    
    # Deduplicate using dict.fromkeys (preserves order) and apply limits
    suggestions = list(dict.fromkeys(suggestions))[:MAX_SUGGESTIONS]
    code_hints = code_hints[:MAX_CODE_HINTS]
    resources = resources[:MAX_RESOURCES]
    
    return {
        "task_id": task.get("id"),
        "task_title": task.get("title"),
        "matched_categories": categories,
        "suggestions": suggestions,
        "code_hints": code_hints,
        "resources": resources,
        "complexity": estimate_complexity(task)
    }


def get_task_analysis(task_id: int) -> Optional[Dict[str, Any]]:
    """Get full AI analysis for a specific task"""
    tasks = json_storage.get_all_tasks()
    task = next((t for t in tasks if t.get("id") == task_id), None)
    
    if not task:
        return None
    
    return generate_code_suggestions(task)


def suggest_subtasks(task: Dict[str, Any]) -> List[Dict[str, str]]:
    """Suggest subtasks for breaking down a complex task"""
    text = f"{task.get('title', '')} {task.get('description', '')}".lower()
    categories = analyze_task_keywords(text)
    
    subtasks = []
    
    # Generate subtasks based on categories
    if "database" in categories:
        subtasks.extend([
            {"title": "Design database schema", "priority": "HIGH"},
            {"title": "Create migration scripts", "priority": "MEDIUM"},
            {"title": "Add indexes for optimization", "priority": "LOW"}
        ])
    
    if "api" in categories:
        subtasks.extend([
            {"title": "Define API contract/schema", "priority": "HIGH"},
            {"title": "Implement endpoint handlers", "priority": "HIGH"},
            {"title": "Add input validation", "priority": "MEDIUM"},
            {"title": "Write API documentation", "priority": "LOW"}
        ])
    
    if "authentication" in categories:
        subtasks.extend([
            {"title": "Set up authentication middleware", "priority": "HIGH"},
            {"title": "Implement token generation", "priority": "HIGH"},
            {"title": "Add password hashing", "priority": "HIGH"},
            {"title": "Create login/logout endpoints", "priority": "MEDIUM"}
        ])
    
    if "frontend" in categories:
        subtasks.extend([
            {"title": "Create component structure", "priority": "HIGH"},
            {"title": "Implement UI layout", "priority": "MEDIUM"},
            {"title": "Add state management", "priority": "MEDIUM"},
            {"title": "Style components", "priority": "LOW"}
        ])
    
    if "testing" in categories:
        subtasks.extend([
            {"title": "Set up test framework", "priority": "HIGH"},
            {"title": "Write unit tests", "priority": "HIGH"},
            {"title": "Add integration tests", "priority": "MEDIUM"},
            {"title": "Configure CI pipeline", "priority": "LOW"}
        ])
    
    # Default subtasks if no categories matched
    if not subtasks:
        subtasks = [
            {"title": "Analyze requirements", "priority": "HIGH"},
            {"title": "Plan implementation approach", "priority": "HIGH"},
            {"title": "Implement core functionality", "priority": "HIGH"},
            {"title": "Add error handling", "priority": "MEDIUM"},
            {"title": "Write tests", "priority": "MEDIUM"},
            {"title": "Update documentation", "priority": "LOW"}
        ]
    
    return subtasks[:6]  # Return max 6 subtasks
