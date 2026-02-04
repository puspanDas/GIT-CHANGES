"""
JSON Storage Module - Optimized Version with Numba
Uses caching, hash tables, orjson for fast I/O, and efficient data structures.
"""
import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

# Use orjson for faster JSON parsing if available
try:
    import orjson
    def json_loads(s): return orjson.loads(s)
    def json_dumps(d): return orjson.dumps(d, option=orjson.OPT_INDENT_2).decode()
    HAS_ORJSON = True
except ImportError:
    import json
    def json_loads(s): return json.loads(s)
    def json_dumps(d): return json.dumps(d, indent=2, default=str)
    HAS_ORJSON = False

# Try to import performance cache
try:
    from performance_core import get_cache, FastIndexCache
    HAS_PERF_CACHE = True
except ImportError:
    HAS_PERF_CACHE = False

DATA_FILE = "data.json"

# ==================== CONSTANTS ====================
# Default values for schema migrations - using dict for single-pass updates
DATA_DEFAULTS = {
    "users": [],
    "tasks": [],
    "comments": [],
    "projects": [],
    "labels": [],
    "teams": [],
    "verification_tokens": [],
    "next_user_id": 1,
    "next_task_id": 1,
    "next_comment_id": 1,
    "next_project_id": 1,
    "next_label_id": 1,
    "next_team_id": 1
}

USER_DEFAULTS = {"xp": 0, "level": 1, "total_tasks_completed": 0, "is_verified": False}

# ==================== CACHING ====================
# In-memory cache to avoid repeated file reads
_cache: Dict[str, Any] = {"data": None, "timestamp": 0}
CACHE_TTL = 0.5  # seconds - short TTL for consistency

# Performance index cache
_perf_cache = get_cache() if HAS_PERF_CACHE else None
_perf_cache_version = 0

def _invalidate_cache():
    """Invalidate the cache when data is modified"""
    global _perf_cache_version
    _cache["data"] = None
    _cache["timestamp"] = 0
    if _perf_cache:
        _perf_cache.invalidate()
    _perf_cache_version += 1


def load_data() -> Dict[str, Any]:
    """
    Load data from JSON file with caching.
    Uses orjson for fast parsing when available.
    """
    global _perf_cache_version
    
    # Check cache first
    if _cache["data"] is not None and (time.time() - _cache["timestamp"]) < CACHE_TTL:
        return _cache["data"]
    
    # Load from file
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb' if HAS_ORJSON else 'r') as f:
            content = f.read()
            data = json_loads(content)
        
        # Single-pass defaults using setdefault
        for key, default in DATA_DEFAULTS.items():
            data.setdefault(key, default if not isinstance(default, list) else default.copy())
        
        # Migrate gamification fields
        for user in data.get("users", []):
            for key, default in USER_DEFAULTS.items():
                user.setdefault(key, default)
    else:
        # Create new data structure
        data = {k: (v.copy() if isinstance(v, list) else v) for k, v in DATA_DEFAULTS.items()}
    
    # Update cache
    _cache["data"] = data
    _cache["timestamp"] = time.time()
    
    # Rebuild performance indexes
    if _perf_cache:
        _perf_cache.build_user_indexes(data.get("users", []))
        _perf_cache.build_task_index(data.get("tasks", []))
    
    return data


def save_data(data: Dict[str, Any]):
    """Save data to JSON file using orjson for speed and invalidate cache"""
    content = json_dumps(data)
    with open(DATA_FILE, 'wb' if HAS_ORJSON else 'w') as f:
        if HAS_ORJSON:
            f.write(content.encode() if isinstance(content, str) else content)
        else:
            f.write(content)
    _invalidate_cache()


# ==================== INDEXED LOOKUPS ====================
# These functions use performance cache for O(1) lookups

def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username - O(1) with cache, O(n) fallback"""
    data = load_data()
    
    if _perf_cache:
        return _perf_cache.get_user_by_username(username)
    
    # Fallback
    username_index = {u["username"]: u for u in data["users"]}
    return username_index.get(username)


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user by email - O(1) with cache, O(n) fallback"""
    data = load_data()
    
    if _perf_cache:
        return _perf_cache.get_user_by_email(email)
    
    # Fallback
    email_index = {u["email"]: u for u in data["users"]}
    return email_index.get(email)


def get_user_by_id(user_id: int) -> Optional[Dict]:
    """Get user by ID - O(1) with cache"""
    data = load_data()
    
    if _perf_cache:
        return _perf_cache.get_user_by_id(user_id)
    
    # Fallback
    id_index = {u["id"]: u for u in data["users"]}
    return id_index.get(user_id)


def get_task_by_id(task_id: int) -> Optional[Dict]:
    """Get task by ID - O(1) with cache"""
    data = load_data()
    
    if _perf_cache:
        return _perf_cache.get_task_by_id(task_id)
    
    # Fallback
    task_index = {t["id"]: t for t in data["tasks"]}
    return task_index.get(task_id)


def create_user(username: str, email: str, hashed_password: str, role: str) -> Optional[Dict]:
    """Create a new user with gamification fields"""
    data = load_data()
    
    # Use indexed check for existence - single pass to build both indexes
    users = data["users"]
    existing = {u["username"] for u in users} | {u["email"] for u in users}
    
    if username in existing or email in existing:
        return None
    
    user = {
        "id": data["next_user_id"],
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "role": role,
        **USER_DEFAULTS  # Spread gamification defaults
    }
    data["users"].append(user)
    data["next_user_id"] += 1
    save_data(data)
    return user


def get_all_tasks() -> List[Dict]:
    """Get all tasks"""
    return load_data()["tasks"]


def get_all_users() -> List[Dict]:
    """Get all users"""
    return load_data()["users"]


def get_all_projects() -> List[Dict]:
    """Get all projects"""
    return load_data()["projects"]


def create_project(name: str, description: str, creator_id: int) -> Dict:
    """Create a new project"""
    data = load_data()
    
    project = {
        "id": data["next_project_id"],
        "name": name,
        "description": description,
        "creator_id": creator_id,
        "created_at": datetime.utcnow().isoformat()
    }
    data["projects"].append(project)
    data["next_project_id"] += 1
    save_data(data)
    return project


def create_task(
    title: str, 
    description: str, 
    priority: str, 
    status: str, 
    creator_id: int, 
    assignee_id: Optional[int] = None, 
    estimated_days: float = 0.0, 
    spent_days: float = 0.0, 
    project_id: Optional[int] = None, 
    dependencies: Optional[List[int]] = None,
    parent_id: Optional[int] = None,
    labels: Optional[List[int]] = None,
    team_id: Optional[int] = None
) -> Dict:
    """Create a new task"""
    data = load_data()
    
    task = {
        "id": data["next_task_id"],
        "title": title,
        "description": description,
        "priority": priority,
        "status": status,
        "creator_id": creator_id,
        "assignee_id": assignee_id,
        "estimated_days": estimated_days,
        "spent_days": spent_days,
        "project_id": project_id,
        "dependencies": dependencies or [],
        "parent_id": parent_id,
        "labels": labels or [],
        "team_id": team_id,
        "created_at": datetime.utcnow().isoformat()
    }
    data["tasks"].append(task)
    data["next_task_id"] += 1
    save_data(data)
    return task


def update_task(task_id: int, updates: dict) -> Optional[Dict]:
    """Update a task - O(n) to find, O(k) to update fields"""
    data = load_data()
    
    # Build index for O(1) lookup
    task_index = {i: t for i, t in enumerate(data["tasks"]) if t["id"] == task_id}
    
    if not task_index:
        return None
    
    idx = next(iter(task_index.keys()))
    task = data["tasks"][idx]
    
    # Update only non-None values using dict update pattern
    task.update({k: v for k, v in updates.items() if v is not None})
    
    save_data(data)
    return task


def update_user(user_id: int, updates: dict) -> Optional[Dict]:
    """Update a user - useful for XP updates"""
    data = load_data()
    
    for i, user in enumerate(data["users"]):
        if user["id"] == user_id:
            user.update({k: v for k, v in updates.items() if v is not None})
            save_data(data)
            return user
    return None


def create_comment(task_id: int, user_id: int, username: str, content: str, attachments: List[str]) -> Dict:
    """Create a new comment"""
    data = load_data()
    
    comment = {
        "id": data.get("next_comment_id", 1),
        "task_id": task_id,
        "user_id": user_id,
        "username": username,
        "content": content,
        "attachments": attachments,
        "created_at": datetime.utcnow().isoformat()
    }
    
    data["comments"].append(comment)
    data["next_comment_id"] = data.get("next_comment_id", 1) + 1
    save_data(data)
    return comment


def get_comments_by_task(task_id: int) -> List[Dict]:
    """Get comments for a specific task - uses list comprehension with sorted"""
    data = load_data()
    comments = data.get("comments", [])
    
    # Single pass: filter and will sort after
    task_comments = [c for c in comments if c["task_id"] == task_id]
    
    # Sort by created_at descending (newest first)
    return sorted(task_comments, key=lambda x: x.get("created_at", ""), reverse=True)


# ==================== LABELS ====================

def get_all_labels() -> List[Dict]:
    """Get all labels"""
    return load_data().get("labels", [])


def get_label_by_id(label_id: int) -> Optional[Dict]:
    """Get label by ID"""
    data = load_data()
    labels = data.get("labels", [])
    label_index = {l["id"]: l for l in labels}
    return label_index.get(label_id)


def create_label(name: str, color: str) -> Dict:
    """Create a new label"""
    data = load_data()
    
    label = {
        "id": data.get("next_label_id", 1),
        "name": name,
        "color": color
    }
    
    if "labels" not in data:
        data["labels"] = []
    
    data["labels"].append(label)
    data["next_label_id"] = data.get("next_label_id", 1) + 1
    save_data(data)
    return label


def delete_label(label_id: int) -> bool:
    """Delete a label by ID"""
    data = load_data()
    labels = data.get("labels", [])
    
    for i, label in enumerate(labels):
        if label["id"] == label_id:
            data["labels"].pop(i)
            # Remove this label from all tasks
            for task in data.get("tasks", []):
                if "labels" in task and label_id in task["labels"]:
                    task["labels"].remove(label_id)
            save_data(data)
            return True
    return False


# ==================== TEAMS ====================

def get_all_teams() -> List[Dict]:
    """Get all teams"""
    return load_data().get("teams", [])


def get_team_by_id(team_id: int) -> Optional[Dict]:
    """Get team by ID"""
    data = load_data()
    teams = data.get("teams", [])
    team_index = {t["id"]: t for t in teams}
    return team_index.get(team_id)


def create_team(name: str, description: str, creator_id: int) -> Dict:
    """Create a new team"""
    data = load_data()
    
    team = {
        "id": data.get("next_team_id", 1),
        "name": name,
        "description": description,
        "creator_id": creator_id,
        "created_at": datetime.utcnow().isoformat()
    }
    
    if "teams" not in data:
        data["teams"] = []
    
    data["teams"].append(team)
    data["next_team_id"] = data.get("next_team_id", 1) + 1
    save_data(data)
    return team


def delete_team(team_id: int) -> bool:
    """Delete a team by ID"""
    data = load_data()
    teams = data.get("teams", [])
    
    for i, team in enumerate(teams):
        if team["id"] == team_id:
            data["teams"].pop(i)
            # Remove team assignment from all tasks
            for task in data.get("tasks", []):
                if task.get("team_id") == team_id:
                    task["team_id"] = None
            save_data(data)
            return True
    return False


# ==================== VERIFICATION TOKENS ====================
def create_verification_token(email: str, token: str, expires_hours: int = 24) -> Dict:
    """Create a verification token for email confirmation"""
    from datetime import timedelta
    data = load_data()
    
    # Remove any existing tokens for this email
    if "verification_tokens" not in data:
        data["verification_tokens"] = []
    data["verification_tokens"] = [t for t in data["verification_tokens"] if t["email"] != email]
    
    token_record = {
        "email": email,
        "token": token,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(hours=expires_hours)).isoformat()
    }
    
    data["verification_tokens"].append(token_record)
    save_data(data)
    return token_record


def get_verification_token(token: str) -> Optional[Dict]:
    """Get verification token record"""
    data = load_data()
    tokens = data.get("verification_tokens", [])
    
    for t in tokens:
        if t["token"] == token:
            # Check if expired
            expires_at = datetime.fromisoformat(t["expires_at"])
            if datetime.utcnow() < expires_at:
                return t
            else:
                # Token expired, remove it
                delete_verification_token(token)
                return None
    return None


def delete_verification_token(token: str) -> bool:
    """Delete a verification token"""
    data = load_data()
    tokens = data.get("verification_tokens", [])
    
    for i, t in enumerate(tokens):
        if t["token"] == token:
            data["verification_tokens"].pop(i)
            save_data(data)
            return True
    return False


def verify_user_email(email: str) -> bool:
    """Mark a user as verified"""
    data = load_data()
    users = data.get("users", [])
    
    for user in users:
        if user["email"] == email:
            user["is_verified"] = True
            save_data(data)
            return True
    return False


def get_user_by_email_for_verification(email: str) -> Optional[Dict]:
    """Get user by email for resending verification"""
    data = load_data()
    users = data.get("users", [])
    
    for user in users:
        if user["email"] == email:
            return user
    return None

