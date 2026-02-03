"""
AI Insights Service for generating predictions and recommendations
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Set
from collections import defaultdict
import json_storage
import kpi_service

# ==================== CONSTANTS ====================
# Using constants instead of magic numbers for maintainability
MAX_TASKS_PER_DEVELOPER = 5
ROLE_MATCH_BONUS = 1
DEFAULT_SPRINT_DAYS = 14

# Priority ordering for sorting
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# Status categories
ACTIVE_STATUSES = frozenset({"TODO", "IN_PROGRESS", "IN_REVIEW"})
ASSIGNABLE_STATUSES = frozenset({"TODO", "IN_PROGRESS"})

# Keywords for task type detection
TEST_KEYWORDS = frozenset({"test", "qa", "verify", "validation", "testing"})

# Developer roles
DEVELOPER_ROLES = frozenset({"DEV", "TESTER"})

def predict_sprint_completion(sprint_days: int = DEFAULT_SPRINT_DAYS) -> Dict[str, Any]:
    """Predict if current sprint will complete on time based on velocity"""
    velocity_data = kpi_service.calculate_velocity(7)
    tasks = json_storage.get_all_tasks()
    
    # Count remaining tasks using frozenset for O(1) lookup
    remaining_tasks = [
        t for t in tasks 
        if t.get("status") in ACTIVE_STATUSES
    ]
    
    velocity_per_day = velocity_data.get("velocity_per_day", 0)
    
    if velocity_per_day > 0:
        days_needed = len(remaining_tasks) / velocity_per_day
        will_complete = days_needed <= sprint_days
    else:
        will_complete = False
        days_needed = 0
    
    return {
        "remaining_tasks": len(remaining_tasks),
        "sprint_days_left": sprint_days,
        "current_velocity_per_day": velocity_per_day,
        "estimated_days_needed": round(days_needed, 2),
        "will_complete_on_time": will_complete,
        "confidence": "high" if velocity_per_day > 1 else "medium" if velocity_per_day > 0.5 else "low"
    }

def suggest_task_reassignment() -> List[Dict[str, Any]]:
    """Recommend task reassignments based on workload balance"""
    dev_metrics = kpi_service.get_developer_metrics()
    risks = kpi_service.detect_risks()
    
    suggestions = []
    
    # Find overloaded developers
    overloaded = risks.get("overloaded_developers", [])
    
    if not overloaded:
        return []
    
    # Find underutilized developers
    avg_tasks = sum(d["in_progress_tasks"] for d in dev_metrics) / len(dev_metrics) if dev_metrics else 0
    underutilized = [
        d for d in dev_metrics 
        if d["in_progress_tasks"] < avg_tasks * 0.5
    ]
    
    for overloaded_dev in overloaded:
        for available_dev in underutilized:
            suggestions.append({
                "from_developer": overloaded_dev["username"],
                "to_developer": available_dev["username"],
                "from_workload": overloaded_dev["active_tasks"],
                "to_workload": available_dev["in_progress_tasks"],
                "recommendation": f"Consider reassigning 2-3 tasks from {overloaded_dev['username']} to {available_dev['username']}"
            })
    
    return suggestions

def identify_bottlenecks() -> Dict[str, Any]:
    """Find bottlenecks in the workflow"""
    tasks = json_storage.get_all_tasks()
    completion_data = kpi_service.calculate_completion_rate()
    
    bottlenecks = {
        "status_bottlenecks": [],
        "priority_bottlenecks": [],
        "recommendations": []
    }
    
    # Check for status bottlenecks
    in_review_count = completion_data.get("in_review", 0)
    in_progress_count = completion_data.get("in_progress", 0)
    
    if in_review_count > in_progress_count * 0.5:
        bottlenecks["status_bottlenecks"].append({
            "status": "IN_REVIEW",
            "count": in_review_count,
            "issue": "High number of tasks stuck in review"
        })
        bottlenecks["recommendations"].append(
            "Consider dedicating more resources to code reviews or implementing pair programming"
        )
    
    # Check for priority bottlenecks
    high_priority_todo = [
        t for t in tasks 
        if t.get("priority") in ["CRITICAL", "HIGH"] and t.get("status") == "TODO"
    ]
    
    if len(high_priority_todo) > 3:
        bottlenecks["priority_bottlenecks"].append({
            "priority": "HIGH/CRITICAL",
            "count": len(high_priority_todo),
            "issue": "Multiple high-priority tasks not started"
        })
        bottlenecks["recommendations"].append(
            "Prioritize high-priority tasks and assign them to available developers immediately"
        )
    
    return bottlenecks

def forecast_delivery_date(project_id: int = None) -> Dict[str, Any]:
    """Predict project completion date based on current velocity"""
    tasks = json_storage.get_all_tasks()
    
    if project_id:
        tasks = [t for t in tasks if t.get("project_id") == project_id]
    
    remaining_tasks = [
        t for t in tasks 
        if t.get("status") in ["TODO", "IN_PROGRESS", "IN_REVIEW"]
    ]
    
    velocity_data = kpi_service.calculate_velocity(14)  # 2-week velocity
    velocity_per_day = velocity_data.get("velocity_per_day", 0)
    
    if velocity_per_day > 0:
        days_needed = len(remaining_tasks) / velocity_per_day
        delivery_date = datetime.now() + timedelta(days=days_needed)
    else:
        days_needed = 0
        delivery_date = None
    
    return {
        "total_tasks": len(tasks),
        "remaining_tasks": len(remaining_tasks),
        "current_velocity": velocity_per_day,
        "estimated_days_to_completion": round(days_needed, 2),
        "forecasted_delivery_date": delivery_date.isoformat() if delivery_date else None,
        "confidence": "high" if velocity_per_day > 1 else "medium" if velocity_per_day > 0.3 else "low"
    }

def analyze_team_health() -> Dict[str, Any]:
    """Generate overall team health score and insights"""
    completion_data = kpi_service.calculate_completion_rate()
    estimation_data = kpi_service.calculate_estimation_accuracy()
    risks = kpi_service.detect_risks()
    velocity_data = kpi_service.calculate_velocity(7)
    
    # Calculate health score (0-100)
    health_factors = []
    
    # Factor 1: Completion rate (0-25 points)
    completion_rate = completion_data.get("completion_rate_percent", 0)
    health_factors.append(min(25, completion_rate / 4))
    
    # Factor 2: On-time delivery (0-25 points)
    on_time_rate = completion_data.get("on_time_delivery_percent", 0)
    health_factors.append(min(25, on_time_rate / 4))
    
    # Factor 3: Estimation accuracy (0-25 points)
    accuracy = estimation_data.get("accuracy_percent", 0)
    health_factors.append(min(25, accuracy / 4))
    
    # Factor 4: Low risks (0-25 points)
    risk_count = (
        len(risks.get("overloaded_developers", [])) +
        len(risks.get("estimation_issues", [])) +
        len(risks.get("high_priority_blocked", []))
    )
    risk_score = max(0, 25 - (risk_count * 3))
    health_factors.append(risk_score)
    
    health_score = sum(health_factors)
    
    # Determine health status
    if health_score >= 80:
        status = "excellent"
        color = "green"
    elif health_score >= 60:
        status = "good"
        color = "blue"
    elif health_score >= 40:
        status = "fair"
        color = "yellow"
    else:
        status = "needs attention"
        color = "red"
    
    # Generate insights
    insights = []
    
    if velocity_data.get("velocity_per_day", 0) < 0.5:
        insights.append("Team velocity is low. Consider reducing task complexity or addressing blockers.")
    
    if len(risks.get("overloaded_developers", [])) > 0:
        insights.append("Some team members are overloaded. Consider redistributing tasks.")
    
    if on_time_rate < 70:
        insights.append("Many tasks are completing late. Review estimation process and task planning.")
    
    if accuracy < 70:
        insights.append("Estimation accuracy needs improvement. Consider using historical data for better estimates.")
    
    return {
        "health_score": round(health_score, 2),
        "status": status,
        "color": color,
        "factors": {
            "completion_rate": round(health_factors[0], 2),
            "on_time_delivery": round(health_factors[1], 2),
            "estimation_accuracy": round(health_factors[2], 2),
            "risk_management": round(health_factors[3], 2)
        },
        "insights": insights,
        "velocity": velocity_data,
        "risk_count": risk_count
    }

def optimize_sprint_assignments(sprint_days: int = DEFAULT_SPRINT_DAYS) -> Dict[str, Any]:
    """
    AI-powered sprint planning - suggests optimal task assignments
    based on workload, skills, and task requirements.
    
    Optimized with:
    - Pre-computed task-by-assignee map (single pass)
    - frozenset for O(1) status lookups
    - Constants for maintainability
    """
    tasks = json_storage.get_all_tasks()
    users = json_storage.get_all_users()
    dev_metrics = kpi_service.get_developer_metrics()
    
    # Pre-compute task counts by assignee in single pass - O(n)
    tasks_by_assignee: Dict[int, int] = defaultdict(int)
    unassigned_tasks = []
    
    for task in tasks:
        assignee_id = task.get("assignee_id")
        status = task.get("status")
        
        if status == "TODO" and not assignee_id:
            unassigned_tasks.append(task)
        elif assignee_id and status in ASSIGNABLE_STATUSES:
            tasks_by_assignee[assignee_id] += 1
    
    # Build developer capacity map using pre-computed counts - O(m)
    dev_capacity = {}
    for user in users:
        if user.get("role") in DEVELOPER_ROLES:
            user_id = user.get("id")
            current_tasks = tasks_by_assignee.get(user_id, 0)
            capacity = max(0, MAX_TASKS_PER_DEVELOPER - current_tasks)
            dev_capacity[user_id] = {
                "username": user.get("username"),
                "role": user.get("role"),
                "current_tasks": current_tasks,
                "available_capacity": capacity
            }
    
    # Generate assignment suggestions
    suggestions = []
    remaining_unassigned = []
    
    # Sort tasks by priority using constant
    sorted_tasks = sorted(
        unassigned_tasks, 
        key=lambda t: PRIORITY_ORDER.get(t.get("priority", "LOW"), 3)
    )
    
    for task in sorted_tasks:
        # Find best available developer
        best_dev = None
        best_capacity = 0
        
        for dev_id, dev_info in dev_capacity.items():
            if dev_info["available_capacity"] > best_capacity:
                # Prefer DEV for non-testing tasks, TESTER for testing tasks
                task_title = task.get("title", "").lower()
                is_test_task = any(kw in task_title for kw in ["test", "qa", "verify", "validation"])
                
                if is_test_task and dev_info["role"] == "TESTER":
                    best_dev = dev_id
                    best_capacity = dev_info["available_capacity"] + 1  # Bonus for role match
                elif not is_test_task and dev_info["role"] == "DEV":
                    best_dev = dev_id
                    best_capacity = dev_info["available_capacity"] + 1
                elif dev_info["available_capacity"] > best_capacity:
                    best_dev = dev_id
                    best_capacity = dev_info["available_capacity"]
        
        if best_dev and dev_capacity[best_dev]["available_capacity"] > 0:
            suggestions.append({
                "task_id": task.get("id"),
                "task_title": task.get("title"),
                "task_priority": task.get("priority"),
                "suggested_assignee_id": best_dev,
                "suggested_assignee": dev_capacity[best_dev]["username"],
                "reason": f"Has capacity ({dev_capacity[best_dev]['available_capacity']} slots) and matches role"
            })
            dev_capacity[best_dev]["available_capacity"] -= 1
        else:
            remaining_unassigned.append({
                "task_id": task.get("id"),
                "task_title": task.get("title"),
                "reason": "No available developer with capacity"
            })
    
    # Calculate workload distribution after suggestions
    workload_distribution = []
    for dev_id, dev_info in dev_capacity.items():
        assigned_in_plan = len([s for s in suggestions if s["suggested_assignee_id"] == dev_id])
        workload_distribution.append({
            "username": dev_info["username"],
            "current_tasks": dev_info["current_tasks"],
            "suggested_new_tasks": assigned_in_plan,
            "total_after_plan": dev_info["current_tasks"] + assigned_in_plan,
            "remaining_capacity": dev_info["available_capacity"]
        })
    
    return {
        "sprint_days": sprint_days,
        "total_unassigned_tasks": len(unassigned_tasks),
        "tasks_assigned_in_plan": len(suggestions),
        "remaining_unassigned": len(remaining_unassigned),
        "assignment_suggestions": suggestions,
        "could_not_assign": remaining_unassigned,
        "workload_distribution": workload_distribution,
        "recommendation": _generate_sprint_recommendation(suggestions, remaining_unassigned, workload_distribution)
    }


def _generate_sprint_recommendation(suggestions: List, remaining: List, distribution: List) -> str:
    """Generate a human-readable recommendation for the sprint plan"""
    if not suggestions and not remaining:
        return "All tasks are already assigned. Sprint is ready to begin!"
    
    if not remaining:
        return f"Optimal plan found! All {len(suggestions)} unassigned tasks can be distributed across the team."
    
    overloaded = [d for d in distribution if d["total_after_plan"] >= 5]
    if overloaded:
        names = ", ".join([d["username"] for d in overloaded])
        return f"Plan has {len(suggestions)} assignments, but {len(remaining)} tasks couldn't be assigned. Consider redistributing work from: {names}"
    
    return f"Partial plan available. {len(suggestions)} tasks assigned, {len(remaining)} remain unassigned due to capacity limits."


def generate_sprint_plan(sprint_days: int = 14) -> Dict[str, Any]:
    """
    Generate a comprehensive sprint plan with timeline and milestones
    """
    optimization = optimize_sprint_assignments(sprint_days)
    velocity_data = kpi_service.calculate_velocity(7)
    
    # Calculate timeline
    tasks_per_day = velocity_data.get("velocity_per_day", 1)
    total_planned_tasks = optimization["tasks_assigned_in_plan"]
    
    estimated_completion_days = total_planned_tasks / tasks_per_day if tasks_per_day > 0 else sprint_days
    
    # Generate milestones
    milestones = []
    if total_planned_tasks > 0:
        milestone_interval = max(1, sprint_days // 3)
        tasks_per_milestone = total_planned_tasks // 3
        
        milestones = [
            {"day": milestone_interval, "target": f"Complete {tasks_per_milestone} critical/high priority tasks"},
            {"day": milestone_interval * 2, "target": f"Complete {tasks_per_milestone * 2} total tasks, begin testing"},
            {"day": sprint_days, "target": "Sprint completion, all tasks done or carried over"}
        ]
    
    return {
        "sprint_duration_days": sprint_days,
        "optimization": optimization,
        "velocity": velocity_data,
        "estimated_completion_days": round(estimated_completion_days, 1),
        "on_track": estimated_completion_days <= sprint_days,
        "milestones": milestones,
        "generated_at": datetime.now().isoformat()
    }


def get_all_insights() -> Dict[str, Any]:
    """Get all AI insights in one call"""
    return {
        "sprint_prediction": predict_sprint_completion(),
        "reassignment_suggestions": suggest_task_reassignment(),
        "bottlenecks": identify_bottlenecks(),
        "delivery_forecast": forecast_delivery_date(),
        "team_health": analyze_team_health(),
        "sprint_plan": optimize_sprint_assignments()
    }
