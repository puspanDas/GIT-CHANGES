"""
Performance Core Module - Optimized Python with Numba JIT Compilation

This module provides high-performance implementations for:
- Level calculations with Numba JIT
- XP calculations
- Leaderboard sorting
- Dependency graph cycle detection
- KPI metrics calculations

To uninstall: pip uninstall numba llvmlite numpy
"""
import numpy as np
from numba import jit, prange
from typing import List, Dict, Tuple, Optional, Set
from functools import lru_cache
import bisect

# ==================== CONSTANTS ====================
# Level thresholds as numpy array for vectorized operations
LEVEL_THRESHOLDS = np.array([0, 100, 300, 600, 1000, 1500, 2200, 3000, 4000, 5000], dtype=np.int64)
MAX_LEVEL = 10

# XP rewards
XP_REWARDS = {
    "HIGH": 100,
    "CRITICAL": 100,
    "BUG": 50,
    "MEDIUM": 35,
    "LOW": 25,
    "DEFAULT": 25
}

# Bug keywords for detection
BUG_KEYWORDS = frozenset({"bug", "fix", "bugfix", "hotfix", "patch"})


# ==================== NUMBA JIT FUNCTIONS ====================

@jit(nopython=True, cache=True)
def _calculate_level_jit(xp: int) -> int:
    """
    Calculate user level from XP using binary search - O(log n)
    JIT compiled for maximum performance
    """
    thresholds = np.array([0, 100, 300, 600, 1000, 1500, 2200, 3000, 4000, 5000], dtype=np.int64)
    max_level = 10
    
    # Binary search
    left, right = 0, len(thresholds)
    while left < right:
        mid = (left + right) // 2
        if thresholds[mid] <= xp:
            left = mid + 1
        else:
            right = mid
    
    return min(left, max_level)


@jit(nopython=True, cache=True)
def _calculate_xp_progress_jit(xp: int, level: int) -> Tuple[int, int]:
    """
    Calculate XP needed for next level and current progress
    Returns: (xp_for_next_level, xp_progress_in_current_level)
    """
    thresholds = np.array([0, 100, 300, 600, 1000, 1500, 2200, 3000, 4000, 5000], dtype=np.int64)
    max_level = 10
    
    if level >= max_level:
        return (0, 0)
    
    current_idx = max(0, level - 1)
    next_idx = level
    
    if next_idx >= len(thresholds):
        return (0, 0)
    
    current_threshold = thresholds[current_idx]
    next_threshold = thresholds[next_idx]
    
    return (next_threshold - current_threshold, xp - current_threshold)


@jit(nopython=True, parallel=True, cache=True)
def _batch_calculate_levels(xp_array: np.ndarray) -> np.ndarray:
    """
    Calculate levels for many users at once using parallel processing
    Much faster than calling calculate_level in a loop
    """
    thresholds = np.array([0, 100, 300, 600, 1000, 1500, 2200, 3000, 4000, 5000], dtype=np.int64)
    max_level = 10
    n = len(xp_array)
    levels = np.empty(n, dtype=np.int64)
    
    for i in prange(n):
        xp = xp_array[i]
        # Binary search
        left, right = 0, len(thresholds)
        while left < right:
            mid = (left + right) // 2
            if thresholds[mid] <= xp:
                left = mid + 1
            else:
                right = mid
        levels[i] = min(left, max_level)
    
    return levels


@jit(nopython=True, cache=True)
def _detect_cycle_dfs(edges: np.ndarray, n_nodes: int, new_from: int, new_to: int) -> bool:
    """
    Detect if adding a new edge would create a cycle using DFS
    edges: Nx2 array of (from, to) pairs
    """
    # Build adjacency list as a simple array structure
    # For simplicity, we'll use a matrix representation
    adj_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    
    for i in range(len(edges)):
        from_node = edges[i, 0]
        to_node = edges[i, 1]
        if from_node < n_nodes and to_node < n_nodes:
            adj_matrix[from_node, to_node] = 1
    
    # Add the new edge
    if new_from < n_nodes and new_to < n_nodes:
        adj_matrix[new_from, new_to] = 1
    
    # DFS for cycle detection
    visited = np.zeros(n_nodes, dtype=np.int8)
    rec_stack = np.zeros(n_nodes, dtype=np.int8)
    
    for start in range(n_nodes):
        if visited[start] == 0:
            # DFS from this node
            stack = [start]
            path = []
            
            while len(stack) > 0:
                node = stack[-1]
                
                if visited[node] == 0:
                    visited[node] = 1
                    rec_stack[node] = 1
                    path.append(node)
                
                found_unvisited = False
                for neighbor in range(n_nodes):
                    if adj_matrix[node, neighbor] == 1:
                        if visited[neighbor] == 0:
                            stack.append(neighbor)
                            found_unvisited = True
                            break
                        elif rec_stack[neighbor] == 1:
                            return True  # Cycle found
                
                if not found_unvisited:
                    stack.pop()
                    rec_stack[node] = 0
    
    return False


@jit(nopython=True, parallel=True, cache=True)
def _calculate_velocity_metrics(
    statuses: np.ndarray,
    estimated_days: np.ndarray,
    spent_days: np.ndarray,
    done_status: int
) -> Tuple[int, float, float]:
    """
    Calculate velocity metrics in a single parallel pass
    Returns: (completed_count, total_time, avg_accuracy)
    """
    n = len(statuses)
    completed = 0
    total_time = 0.0
    total_accuracy = 0.0
    accuracy_count = 0
    
    for i in prange(n):
        if statuses[i] == done_status:
            completed += 1
            total_time += spent_days[i]
            
            if estimated_days[i] > 0 and spent_days[i] > 0:
                accuracy = 1.0 - min(1.0, abs(estimated_days[i] - spent_days[i]) / estimated_days[i])
                total_accuracy += accuracy
                accuracy_count += 1
    
    avg_accuracy = total_accuracy / accuracy_count if accuracy_count > 0 else 0.0
    
    return (completed, total_time, avg_accuracy)


# ==================== PYTHON WRAPPER FUNCTIONS ====================

def calculate_level(xp: int) -> int:
    """Calculate user level from XP - JIT optimized"""
    return int(_calculate_level_jit(xp))


def calculate_xp_to_next_level(xp: int, current_level: int) -> Tuple[int, int]:
    """Calculate XP needed for next level"""
    return _calculate_xp_progress_jit(xp, current_level)


def get_xp_for_task(priority: str, description: str = "") -> int:
    """
    Calculate XP reward for completing a task based on priority
    """
    if description:
        desc_lower = description.lower()
        # Check for bug keywords
        for word in desc_lower.split():
            if word in BUG_KEYWORDS:
                return XP_REWARDS["BUG"]
    
    return XP_REWARDS.get(priority.upper(), XP_REWARDS["DEFAULT"])


def batch_calculate_levels(users: List[Dict]) -> List[int]:
    """
    Calculate levels for many users at once - parallel processing
    """
    if not users:
        return []
    
    xp_array = np.array([u.get("xp", 0) for u in users], dtype=np.int64)
    levels = _batch_calculate_levels(xp_array)
    return levels.tolist()


def build_leaderboard_fast(users: List[Dict], limit: int = 10) -> List[Dict]:
    """
    Build leaderboard efficiently using numpy for sorting
    """
    if not users:
        return []
    
    # Extract data into numpy arrays for fast sorting
    n = len(users)
    xp_array = np.array([u.get("xp", 0) for u in users], dtype=np.int64)
    
    # Get sorted indices by XP descending
    sorted_indices = np.argsort(-xp_array)[:limit]
    
    # Calculate levels in batch
    levels = _batch_calculate_levels(xp_array[sorted_indices])
    
    # Build result
    leaderboard = []
    for rank, (idx, level) in enumerate(zip(sorted_indices, levels), 1):
        user = users[idx]
        xp = int(xp_array[idx])
        xp_needed, xp_progress = calculate_xp_to_next_level(xp, int(level))
        
        leaderboard.append({
            "rank": rank,
            "user_id": user.get("id"),
            "username": user.get("username", ""),
            "xp": xp,
            "level": int(level),
            "xp_to_next_level": xp_needed,
            "xp_progress": xp_progress,
            "total_tasks_completed": user.get("total_tasks_completed", 0)
        })
    
    return leaderboard


def detect_cycle(tasks: List[Dict], new_from: int, new_to: int) -> bool:
    """
    Detect if adding a dependency would create a cycle
    """
    if not tasks:
        return False
    
    # Build node mapping
    all_ids = set()
    for task in tasks:
        all_ids.add(task.get("id", 0))
        for dep in task.get("dependencies", []):
            all_ids.add(dep)
    all_ids.add(new_from)
    all_ids.add(new_to)
    
    id_to_idx = {id_: idx for idx, id_ in enumerate(sorted(all_ids))}
    n_nodes = len(id_to_idx)
    
    # Build edges array
    edges_list = []
    for task in tasks:
        task_id = task.get("id", 0)
        for dep_id in task.get("dependencies", []):
            edges_list.append([id_to_idx.get(task_id, 0), id_to_idx.get(dep_id, 0)])
    
    if not edges_list:
        edges = np.zeros((0, 2), dtype=np.int64)
    else:
        edges = np.array(edges_list, dtype=np.int64)
    
    return _detect_cycle_dfs(
        edges, 
        n_nodes, 
        id_to_idx.get(new_from, 0), 
        id_to_idx.get(new_to, 0)
    )


def calculate_velocity(tasks: List[Dict], cutoff_days: int = 7) -> Dict:
    """
    Calculate velocity metrics using JIT-optimized computation
    """
    if not tasks:
        return {
            "completed_count": 0,
            "total_time": 0.0,
            "avg_accuracy": 0.0
        }
    
    # Convert to numpy arrays
    status_map = {"DONE": 1, "IN_PROGRESS": 2, "TODO": 3, "BACKLOG": 4}
    
    statuses = np.array([status_map.get(t.get("status", ""), 0) for t in tasks], dtype=np.int64)
    estimated = np.array([t.get("estimated_days", 0.0) for t in tasks], dtype=np.float64)
    spent = np.array([t.get("spent_days", 0.0) for t in tasks], dtype=np.float64)
    
    completed, total_time, avg_accuracy = _calculate_velocity_metrics(
        statuses, estimated, spent, done_status=1
    )
    
    return {
        "completed_count": int(completed),
        "total_time": float(total_time),
        "avg_accuracy": float(avg_accuracy),
        "avg_completion_time": float(total_time / completed) if completed > 0 else 0.0
    }


# ==================== CACHING UTILITIES ====================

class FastIndexCache:
    """
    In-memory index cache for O(1) lookups
    Automatically invalidates on data changes
    """
    
    def __init__(self):
        self._user_id_index: Dict[int, Dict] = {}
        self._user_username_index: Dict[str, Dict] = {}
        self._user_email_index: Dict[str, Dict] = {}
        self._task_id_index: Dict[int, Dict] = {}
        self._version = 0
    
    def build_user_indexes(self, users: List[Dict]) -> None:
        """Build all user indexes in a single pass - O(n)"""
        self._user_id_index.clear()
        self._user_username_index.clear()
        self._user_email_index.clear()
        
        for user in users:
            user_id = user.get("id")
            if user_id is not None:
                self._user_id_index[user_id] = user
            
            username = user.get("username")
            if username:
                self._user_username_index[username] = user
            
            email = user.get("email")
            if email:
                self._user_email_index[email] = user
        
        self._version += 1
    
    def build_task_index(self, tasks: List[Dict]) -> None:
        """Build task index - O(n)"""
        self._task_id_index = {t.get("id"): t for t in tasks if t.get("id") is not None}
        self._version += 1
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """O(1) lookup"""
        return self._user_id_index.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """O(1) lookup"""
        return self._user_username_index.get(username)
    
    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """O(1) lookup"""
        return self._user_email_index.get(email)
    
    def get_task_by_id(self, task_id: int) -> Optional[Dict]:
        """O(1) lookup"""
        return self._task_id_index.get(task_id)
    
    def invalidate(self) -> None:
        """Clear all caches"""
        self._user_id_index.clear()
        self._user_username_index.clear()
        self._user_email_index.clear()
        self._task_id_index.clear()
        self._version += 1


# Global cache instance
_cache = FastIndexCache()


def get_cache() -> FastIndexCache:
    """Get the global cache instance"""
    return _cache


# ==================== BENCHMARK UTILITY ====================

def benchmark_performance():
    """Run a quick benchmark to verify speedup"""
    import time
    
    # Generate test data
    n_users = 10000
    test_xps = np.random.randint(0, 10000, n_users)
    test_users = [{"id": i, "xp": int(xp), "username": f"user{i}"} for i, xp in enumerate(test_xps)]
    
    # Warm up JIT
    _calculate_level_jit(500)
    _batch_calculate_levels(np.array([100, 200, 300]))
    
    # Benchmark single level calculation
    start = time.perf_counter()
    for xp in test_xps[:1000]:
        calculate_level(int(xp))
    jit_time = time.perf_counter() - start
    
    # Benchmark batch level calculation
    start = time.perf_counter()
    batch_calculate_levels(test_users)
    batch_time = time.perf_counter() - start
    
    # Benchmark leaderboard
    start = time.perf_counter()
    build_leaderboard_fast(test_users, 100)
    leaderboard_time = time.perf_counter() - start
    
    print(f"Performance Results ({n_users} users):")
    print(f"  Single level calc (1000x): {jit_time*1000:.2f}ms")
    print(f"  Batch level calc: {batch_time*1000:.2f}ms")
    print(f"  Leaderboard build: {leaderboard_time*1000:.2f}ms")
    
    return {
        "single_calc_ms": jit_time * 1000,
        "batch_calc_ms": batch_time * 1000,
        "leaderboard_ms": leaderboard_time * 1000
    }


if __name__ == "__main__":
    benchmark_performance()
