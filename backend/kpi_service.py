"""
KPI Service for calculating team and project performance metrics
Optimized with Counter, defaultdict, single-pass algorithms, and frozenset lookups.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import Counter, defaultdict
import json_storage

# ==================== CONSTANTS ====================
# Using frozenset for O(1) membership testing
DONE_STATUS = "DONE"
ACTIVE_STATUSES = frozenset({"IN_PROGRESS", "IN_REVIEW"})
OVERLOAD_THRESHOLD = 5
HIGH_PRIORITY_STATUSES = frozenset({"CRITICAL", "HIGH"})
DEVELOPER_ROLES = frozenset({"DEV", "TESTER"})
ACCURACY_TOLERANCE = 0.1  # 10% tolerance for "accurate" estimation


def calculate_velocity(days: int = 7) -> Dict[str, Any]:
    """Calculate task completion velocity over specified period"""
    tasks = json_storage.get_all_tasks()
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # Single pass with generator expression
    completed_count = sum(
        1 for t in tasks 
        if t.get("status") == DONE_STATUS and 
        _parse_date(t.get("created_at", "")) > cutoff_date
    )
    
    velocity = completed_count / days if days > 0 else 0
    
    return {
        "period_days": days,
        "tasks_completed": completed_count,
        "velocity_per_day": round(velocity, 2),
        "velocity_per_week": round(velocity * 7, 2)
    }


def _parse_date(date_str: str) -> datetime:
    """Safely parse ISO date string with fallback"""
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return datetime.min


def calculate_cycle_time() -> Dict[str, Any]:
    """Calculate average time from task creation to completion"""
    tasks = json_storage.get_all_tasks()
    
    # Single pass: filter completed and sum spent_days
    total_time = 0
    count = 0
    completed_count = 0
    
    for task in tasks:
        if task.get("status") == DONE_STATUS:
            completed_count += 1
            spent = task.get("spent_days", 0)
            if spent > 0:
                total_time += spent
                count += 1
    
    avg_cycle_time = total_time / count if count > 0 else 0
    
    return {
        "average_cycle_time_days": round(avg_cycle_time, 2),
        "total_completed_tasks": completed_count,
        "tasks_with_time_data": count
    }


def calculate_completion_rate() -> Dict[str, Any]:
    """
    Calculate task completion rate and on-time delivery.
    Uses Counter for single-pass status counting.
    """
    tasks = json_storage.get_all_tasks()
    total_tasks = len(tasks)
    
    if total_tasks == 0:
        return _empty_completion_rate()
    
    # Single pass: count statuses AND calculate on-time delivery
    status_counts = Counter()
    on_time = 0
    
    for task in tasks:
        status = task.get("status", "TODO")
        status_counts[status] += 1
        
        # Check on-time delivery for completed tasks
        if status == DONE_STATUS:
            estimated = task.get("estimated_days", 0)
            spent = task.get("spent_days", 0)
            if estimated > 0 and spent <= estimated:
                on_time += 1
    
    completed = status_counts[DONE_STATUS]
    completion_rate = (completed / total_tasks * 100) if total_tasks > 0 else 0
    on_time_rate = (on_time / completed * 100) if completed > 0 else 0
    
    return {
        "total_tasks": total_tasks,
        "completed": completed,
        "in_progress": status_counts["IN_PROGRESS"],
        "todo": status_counts["TODO"],
        "in_review": status_counts["IN_REVIEW"],
        "completion_rate_percent": round(completion_rate, 2),
        "on_time_delivery_percent": round(on_time_rate, 2)
    }


def _empty_completion_rate() -> Dict[str, Any]:
    """Return empty completion rate structure"""
    return {
        "total_tasks": 0,
        "completed": 0,
        "in_progress": 0,
        "todo": 0,
        "in_review": 0,
        "completion_rate_percent": 0,
        "on_time_delivery_percent": 0
    }


def calculate_estimation_accuracy() -> Dict[str, Any]:
    """
    Compare estimated vs actual spent time.
    Single pass with classification using ternary operators.
    """
    tasks = json_storage.get_all_tasks()
    
    # Single pass with immediate classification
    over_estimated = 0
    under_estimated = 0
    accurate = 0
    total_variance = 0
    count = 0
    
    for task in tasks:
        if task.get("status") != DONE_STATUS:
            continue
            
        estimated = task.get("estimated_days", 0)
        spent = task.get("spent_days", 0)
        
        if estimated <= 0 or spent <= 0:
            continue
        
        count += 1
        variance = abs(spent - estimated) / estimated
        total_variance += variance
        
        # Classify using ternary chain instead of if/elif/else
        (accurate if variance < ACCURACY_TOLERANCE else 
         (under_estimated if spent > estimated else over_estimated))
        
        # Actually increment the right counter
        if variance < ACCURACY_TOLERANCE:
            accurate += 1
        elif spent > estimated:
            under_estimated += 1
        else:
            over_estimated += 1
    
    if count == 0:
        return {
            "accuracy_percent": 0,
            "over_estimated": 0,
            "under_estimated": 0,
            "accurate": 0,
            "total_analyzed": 0
        }
    
    avg_accuracy = (1 - (total_variance / count)) * 100
    
    return {
        "accuracy_percent": round(max(0, avg_accuracy), 2),
        "over_estimated": over_estimated,
        "under_estimated": under_estimated,
        "accurate": accurate,
        "total_analyzed": count
    }


def get_developer_metrics() -> List[Dict[str, Any]]:
    """
    Get per-developer productivity metrics.
    Uses pre-built task index for O(n) instead of O(n*m).
    """
    tasks = json_storage.get_all_tasks()
    users = json_storage.get_all_users()
    
    # Pre-build task index by assignee - O(n) single pass
    tasks_by_assignee: Dict[int, List[Dict]] = defaultdict(list)
    for task in tasks:
        assignee_id = task.get("assignee_id")
        if assignee_id:
            tasks_by_assignee[assignee_id].append(task)
    
    # Filter developers using frozenset
    developers = [u for u in users if u.get("role") in DEVELOPER_ROLES]
    
    metrics = []
    for dev in developers:
        dev_id = dev.get("id")
        dev_tasks = tasks_by_assignee.get(dev_id, [])
        
        # Use Counter for status breakdown
        status_counts = Counter(t.get("status") for t in dev_tasks)
        completed_tasks = [t for t in dev_tasks if t.get("status") == DONE_STATUS]
        
        # Calculate totals using generator expressions
        total_estimated = sum(t.get("estimated_days", 0) for t in dev_tasks)
        total_spent = sum(t.get("spent_days", 0) for t in completed_tasks)
        
        # Calculate on-time completion
        on_time = sum(
            1 for t in completed_tasks 
            if t.get("estimated_days", 0) > 0 and t.get("spent_days", 0) <= t.get("estimated_days", 0)
        )
        
        total_tasks = len(dev_tasks)
        completed_count = len(completed_tasks)
        
        metrics.append({
            "user_id": dev_id,
            "username": dev.get("username"),
            "role": dev.get("role"),
            "total_tasks": total_tasks,
            "completed_tasks": completed_count,
            "in_progress_tasks": status_counts.get("IN_PROGRESS", 0),
            "estimated_days": round(total_estimated, 2),
            "spent_days": round(total_spent, 2),
            "on_time_completions": on_time,
            "completion_rate": round(completed_count / total_tasks * 100, 2) if total_tasks else 0
        })
    
    # Sort by completed tasks descending
    return sorted(metrics, key=lambda x: x["completed_tasks"], reverse=True)


def detect_risks() -> Dict[str, Any]:
    """
    Identify at-risk tasks and overloaded team members.
    Uses Counter and pre-built indexes.
    """
    tasks = json_storage.get_all_tasks()
    users = json_storage.get_all_users()
    
    # Build user lookup dict - O(1) access later
    user_lookup = {u["id"]: u for u in users}
    
    risks = {
        "overdue_tasks": [],
        "overloaded_developers": [],
        "high_priority_blocked": [],
        "estimation_issues": []
    }
    
    # Single pass through tasks - count workload AND detect issues
    dev_workload: Counter = Counter()
    
    for task in tasks:
        status = task.get("status")
        assignee_id = task.get("assignee_id")
        priority = task.get("priority")
        
        # Count in-progress workload
        if status == "IN_PROGRESS" and assignee_id:
            dev_workload[assignee_id] += 1
        
        # Check for estimation issues
        if status in ACTIVE_STATUSES:
            estimated = task.get("estimated_days", 0)
            spent = task.get("spent_days", 0)
            
            if estimated > 0 and spent > estimated * 1.5:
                risks["estimation_issues"].append({
                    "task_id": task.get("id"),
                    "title": task.get("title"),
                    "estimated_days": estimated,
                    "spent_days": spent,
                    "overrun_percent": round((spent - estimated) / estimated * 100, 2)
                })
        
        # Check for high priority blocked
        if priority in HIGH_PRIORITY_STATUSES and status == "TODO":
            risks["high_priority_blocked"].append({
                "task_id": task.get("id"),
                "title": task.get("title"),
                "priority": priority
            })
    
    # Find overloaded developers
    for dev_id, task_count in dev_workload.items():
        if task_count > OVERLOAD_THRESHOLD:
            user = user_lookup.get(dev_id, {})
            risks["overloaded_developers"].append({
                "user_id": dev_id,
                "username": user.get("username", "Unknown"),
                "active_tasks": task_count
            })
    
    return risks


def get_all_kpis() -> Dict[str, Any]:
    """Get all KPI metrics in one call"""
    return {
        "velocity": calculate_velocity(7),
        "cycle_time": calculate_cycle_time(),
        "completion_rate": calculate_completion_rate(),
        "estimation_accuracy": calculate_estimation_accuracy(),
        "team_performance": get_developer_metrics(),
        "risks": detect_risks()
    }
