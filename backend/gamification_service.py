"""
Gamification Service - Optimized Version with Numba JIT
Manages XP system, leveling, and leaderboard for developer productivity gamification.
Uses Numba JIT compilation for faster calculations.
"""
from typing import Dict, List, Optional, Tuple
import bisect
import json_storage

# Try to import Numba-optimized functions, fallback to pure Python
try:
    import performance_core
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ==================== CONSTANTS ====================
# XP Configuration - using dict for O(1) lookup
XP_REWARDS = {
    "HIGH": 100,
    "CRITICAL": 100,
    "BUG": 50,
    "MEDIUM": 35,
    "LOW": 25,
    "DEFAULT": 25
}

# Level thresholds - XP required to reach each level (sorted for binary search)
LEVEL_THRESHOLDS = [0, 100, 300, 600, 1000, 1500, 2200, 3000, 4000, 5000]
MAX_LEVEL = 10

# Level colors and titles - using dict for O(1) lookup
LEVEL_CONFIG = {
    1: {"color": "#6b7280", "glow": "rgba(107, 114, 128, 0.4)", "title": "Novice", "intensity": 1},
    2: {"color": "#6b7280", "glow": "rgba(107, 114, 128, 0.5)", "title": "Novice", "intensity": 1.2},
    3: {"color": "#3b82f6", "glow": "rgba(59, 130, 246, 0.5)", "title": "Apprentice", "intensity": 1.5},
    4: {"color": "#3b82f6", "glow": "rgba(59, 130, 246, 0.6)", "title": "Apprentice", "intensity": 1.8},
    5: {"color": "#8b5cf6", "glow": "rgba(139, 92, 246, 0.6)", "title": "Expert", "intensity": 2},
    6: {"color": "#8b5cf6", "glow": "rgba(139, 92, 246, 0.7)", "title": "Expert", "intensity": 2.3},
    7: {"color": "#f59e0b", "glow": "rgba(245, 158, 11, 0.7)", "title": "Master", "intensity": 2.5},
    8: {"color": "#f59e0b", "glow": "rgba(245, 158, 11, 0.8)", "title": "Master", "intensity": 2.8},
    9: {"color": "#eab308", "glow": "rgba(234, 179, 8, 0.8)", "title": "Legend", "intensity": 3},
    10: {"color": "#ef4444", "glow": "rgba(239, 68, 68, 0.9)", "title": "Champion", "intensity": 3.5},
}

# Keywords for bug detection - frozenset for O(1) lookup
BUG_KEYWORDS = frozenset({"bug", "fix"})

# Default gamification stats for new users
USER_DEFAULTS = {"xp": 0, "level": 1, "total_tasks_completed": 0}


def calculate_level(xp: int) -> int:
    """
    Calculate user level based on XP.
    Uses Numba JIT when available for faster calculation.
    """
    if HAS_NUMBA:
        return performance_core.calculate_level(xp)
    else:
        # Fallback to bisect
        level = bisect.bisect_right(LEVEL_THRESHOLDS, xp)
        return min(level, MAX_LEVEL)


def calculate_xp_to_next_level(xp: int, current_level: int) -> Tuple[int, int]:
    """
    Calculate XP needed for next level and current progress.
    Returns: (xp_for_next_level, xp_progress_in_current_level)
    """
    if current_level >= MAX_LEVEL:
        return (0, 0)
    
    current_threshold = LEVEL_THRESHOLDS[current_level - 1]
    next_threshold = LEVEL_THRESHOLDS[current_level]
    
    return (next_threshold - current_threshold, xp - current_threshold)


def get_xp_for_task(priority: str, description: str = "") -> int:
    """
    Calculate XP reward for completing a task based on priority and type.
    Uses frozenset for O(1) keyword lookup.
    """
    description_lower = description.lower() if description else ""
    
    # Check if it's a bug using frozenset intersection
    if BUG_KEYWORDS & set(description_lower.split()):
        return XP_REWARDS["BUG"]
    
    # Direct dict lookup - O(1)
    return XP_REWARDS.get(priority, XP_REWARDS["DEFAULT"])


def get_level_config(level: int) -> Dict:
    """Get color and glow configuration for a level - O(1) dict lookup"""
    clamped_level = max(1, min(level, MAX_LEVEL))
    return LEVEL_CONFIG.get(clamped_level, LEVEL_CONFIG[1])


def award_xp(user_id: int, xp_amount: int) -> Optional[Dict]:
    """
    Award XP to a user and recalculate their level.
    Returns updated gamification stats.
    Uses setdefault for cleaner field initialization.
    """
    data = json_storage.load_data()
    
    # Build user index for O(1) lookup
    user_index = {u["id"]: (i, u) for i, u in enumerate(data["users"])}
    
    if user_id not in user_index:
        return None
    
    idx, user = user_index[user_id]
    
    # Initialize fields using setdefault - cleaner than if/else
    for key, default in USER_DEFAULTS.items():
        user.setdefault(key, default)
    
    old_level = user["level"]
    
    # Award XP and increment tasks
    user["xp"] += xp_amount
    user["total_tasks_completed"] += 1
    
    # Recalculate level using binary search
    new_level = calculate_level(user["xp"])
    user["level"] = new_level
    
    json_storage.save_data(data)
    
    level_config = get_level_config(new_level)
    xp_needed, xp_progress = calculate_xp_to_next_level(user["xp"], new_level)
    
    return {
        "user_id": user_id,
        "xp": user["xp"],
        "xp_gained": xp_amount,
        "level": new_level,
        "level_up": new_level > old_level,
        "old_level": old_level,
        "level_config": level_config,
        "xp_to_next_level": xp_needed,
        "xp_progress": xp_progress,
        "total_tasks_completed": user["total_tasks_completed"]
    }


def get_user_gamification_stats(user_id: int) -> Optional[Dict]:
    """Get gamification stats for a user - uses dict index for O(1) lookup"""
    data = json_storage.load_data()
    
    # Build user index - O(n) once, then O(1) lookup
    user_index = {u["id"]: u for u in data["users"]}
    user = user_index.get(user_id)
    
    if not user:
        return None
    
    xp = user.get("xp", 0)
    level = calculate_level(xp)  # Binary search - O(log n)
    
    level_config = get_level_config(level)
    xp_needed, xp_progress = calculate_xp_to_next_level(xp, level)
    
    return {
        "user_id": user_id,
        "username": user["username"],
        "xp": xp,
        "level": level,
        "level_config": level_config,
        "xp_to_next_level": xp_needed,
        "xp_progress": xp_progress,
        "xp_progress_percent": (xp_progress / xp_needed * 100) if xp_needed > 0 else 100,
        "total_tasks_completed": user.get("total_tasks_completed", 0)
    }


def get_leaderboard(limit: int = 10) -> List[Dict]:
    """
    Get top users by XP for leaderboard.
    Uses Numba-optimized batch processing when available.
    """
    data = json_storage.load_data()
    users = data.get("users", [])
    
    if not users:
        return []
    
    # Use Numba-optimized version if available
    if HAS_NUMBA:
        leaderboard = performance_core.build_leaderboard_fast(users, limit)
        # Add level_config to each entry
        for entry in leaderboard:
            entry["level_config"] = get_level_config(entry["level"])
        return leaderboard
    
    # Fallback: Python implementation
    leaderboard = [
        {
            "user_id": user["id"],
            "username": user["username"],
            "xp": user.get("xp", 0),
            "level": calculate_level(user.get("xp", 0)),
            "level_config": get_level_config(calculate_level(user.get("xp", 0))),
            "total_tasks_completed": user.get("total_tasks_completed", 0)
        }
        for user in users
    ]
    
    # Sort by XP descending and take top N
    leaderboard.sort(key=lambda x: x["xp"], reverse=True)
    top_users = leaderboard[:limit]
    
    # Add rank using enumerate
    for rank, user in enumerate(top_users, 1):
        user["rank"] = rank
    
    return top_users


def get_all_level_colors() -> Dict:
    """Return all level color configurations for frontend"""
    return LEVEL_CONFIG
