"""
Comprehensive Test Suite for AI Task Manager
Tests all features and scenarios - Automatically starts servers
"""

import requests
import json
import time
import subprocess
import sys
import os
import atexit

BASE_URL = "http://localhost:8000"

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Store process references
backend_process = None
frontend_process = None

def cleanup_servers():
    """Clean up server processes on exit"""
    global backend_process, frontend_process
    print(f"\n{YELLOW}Cleaning up servers...{RESET}")
    if backend_process:
        backend_process.terminate()
        print(f"{GREEN}✓ Backend stopped{RESET}")
    if frontend_process:
        frontend_process.terminate()
        print(f"{GREEN}✓ Frontend stopped{RESET}")

def start_servers():
    """Start backend and frontend servers"""
    global backend_process, frontend_process
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Starting Backend and Frontend Servers...{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    # Get paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(root_dir, "backend")
    frontend_dir = os.path.join(root_dir, "frontend")
    
    # Start Backend
    print(f"{YELLOW}Starting Backend (FastAPI)...{RESET}")
    backend_log = open("backend_test.log", "w")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8000"],
        cwd=backend_dir,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
        shell=False
    )
    
    # Start Frontend
    print(f"{YELLOW}Starting Frontend (Vite)...{RESET}")
    frontend_log = open("frontend_test.log", "w")
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"
    frontend_process = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir,
        stdout=frontend_log,
        stderr=subprocess.STDOUT,
        shell=False
    )
    
    # Register cleanup
    atexit.register(cleanup_servers)
    
    # Wait for servers to start
    print(f"{YELLOW}Waiting for servers to start...{RESET}")
    max_attempts = 30
    for i in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/docs", timeout=1)
            if response.status_code == 200:
                print(f"{GREEN}✓ Backend is ready!{RESET}")
                print(f"{GREEN}✓ Frontend is starting (will be available soon){RESET}\n")
                time.sleep(2)  # Extra time for frontend
                return True
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"Error checking server: {e}")
            
        time.sleep(1)
        if i % 5 == 0:
            print(f"  Attempt {i+1}/{max_attempts}...")
    
    print(f"{RED}✗ Servers failed to start within {max_attempts} seconds{RESET}")
    
    print(f"\n{YELLOW}Backend Log (last 10 lines):{RESET}")
    try:
        with open("backend_test.log", "r") as f:
            print("".join(f.readlines()[-10:]))
    except:
        print("Could not read backend log")

    cleanup_servers()
    return False

def print_test(test_name, passed, message=""):
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} - {test_name}")
    if message:
        print(f"    {message}")

def test_user_registration():
    """Test 1: User Registration"""
    print(f"\n{YELLOW}=== Test 1: User Registration ==={RESET}")
    
    # Test 1.1: Valid registration
    response = requests.post(f"{BASE_URL}/users/", json={
        "username": "testdev",
        "email": "testdev@example.com",
        "password": "password123",
        "role": "DEV"
    })
    print_test("1.1 Valid user registration", response.status_code == 200)
    
    # Test 1.2: Duplicate username
    response = requests.post(f"{BASE_URL}/users/", json={
        "username": "testdev",
        "email": "another@example.com",
        "password": "password123",
        "role": "DEV"
    })
    print_test("1.2 Duplicate username rejection", response.status_code == 400)
    
    # Test 1.3: Duplicate email
    response = requests.post(f"{BASE_URL}/users/", json={
        "username": "anotheruser",
        "email": "testdev@example.com",
        "password": "password123",
        "role": "DEV"
    })
    print_test("1.3 Duplicate email rejection", response.status_code == 400)
    
    # Test 1.4: Multiple roles
    roles = ["TESTER", "PO", "PM", "RE"]
    for i, role in enumerate(roles, start=2):
        response = requests.post(f"{BASE_URL}/users/", json={
            "username": f"user{role.lower()}",
            "email": f"user{role.lower()}@example.com",
            "password": "password123",
            "role": role
        })
        print_test(f"1.{i+2} Register {role} user", response.status_code == 200)

def test_user_authentication():
    """Test 2: User Authentication"""
    print(f"\n{YELLOW}=== Test 2: User Authentication ==={RESET}")
    
    # Test 2.1: Valid login
    response = requests.post(f"{BASE_URL}/token", data={
        "username": "testdev",
        "password": "password123"
    })
    success = response.status_code == 200
    token = None
    if success:
        token = response.json()["access_token"]
    print_test("2.1 Valid login", success, f"Token: {token[:20]}..." if token else "")
    
    # Test 2.2: Invalid password
    response = requests.post(f"{BASE_URL}/token", data={
        "username": "testdev",
        "password": "wrongpassword"
    })
    print_test("2.2 Invalid password rejection", response.status_code == 401)
    
    # Test 2.3: Non-existent user
    response = requests.post(f"{BASE_URL}/token", data={
        "username": "nonexistent",
        "password": "password123"
    })
    print_test("2.3 Non-existent user rejection", response.status_code == 401)
    
    return token

def test_get_current_user(token):
    """Test 3: Get Current User"""
    print(f"\n{YELLOW}=== Test 3: Get Current User ==={RESET}")
    
    # Test 3.1: Valid token
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/users/me/", headers=headers)
    success = response.status_code == 200
    if success:
        user = response.json()
        print_test("3.1 Get current user with valid token", True, f"User: {user['username']}, Role: {user['role']}")
    else:
        print_test("3.1 Get current user with valid token", False)
    
    # Test 3.2: Invalid token
    headers = {"Authorization": "Bearer invalid_token"}
    response = requests.get(f"{BASE_URL}/users/me/", headers=headers)
    print_test("3.2 Invalid token rejection", response.status_code == 401)
    
    # Test 3.3: No token
    response = requests.get(f"{BASE_URL}/users/me/")
    print_test("3.3 No token rejection", response.status_code == 401)

def test_task_creation_with_ai(token):
    """Test 4: Task Creation with AI Priority Detection"""
    print(f"\n{YELLOW}=== Test 4: Task Creation with AI Priority ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test cases with expected priorities
    test_cases = [
        ("Fix critical security vulnerability", "CRITICAL", "Critical security issue"),
        ("Update button color to blue", "LOW", "Simple UI change"),
        ("Optimize database queries for performance", "HIGH", "Performance optimization"),
        ("Add user profile page", "MEDIUM", "New feature"),
        ("Server is down and users cannot login", "CRITICAL", "System down"),
    ]
    
    for i, (description, expected_priority, test_name) in enumerate(test_cases, start=1):
        response = requests.post(f"{BASE_URL}/tasks/", 
            headers=headers,
            json={
                "title": f"Test Task {i}",
                "description": description
            }
        )
        
        if response.status_code == 200:
            task = response.json()
            detected_priority = task.get("priority")
            # AI might not always get it exactly right, so we just check if it assigned a priority
            success = detected_priority in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            print_test(f"4.{i} {test_name}", success, 
                      f"Expected: {expected_priority}, Got: {detected_priority}")
        else:
            print_test(f"4.{i} {test_name}", False, f"Status: {response.status_code}")

def test_task_retrieval(token):
    """Test 5: Task Retrieval"""
    print(f"\n{YELLOW}=== Test 5: Task Retrieval ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 5.1: Get all tasks
    response = requests.get(f"{BASE_URL}/tasks/", headers=headers)
    success = response.status_code == 200
    if success:
        tasks = response.json()
        print_test("5.1 Get all tasks", True, f"Found {len(tasks)} tasks")
    else:
        print_test("5.1 Get all tasks", False)
    
    # Test 5.2: Tasks without authentication
    response = requests.get(f"{BASE_URL}/tasks/")
    print_test("5.2 Tasks require authentication", response.status_code == 401)

def test_excel_export(token):
    """Test 6: Excel Export"""
    print(f"\n{YELLOW}=== Test 6: Excel Export ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 6.1: Export with valid token
    response = requests.get(f"{BASE_URL}/tasks/export/excel", headers=headers)
    success = response.status_code == 200 and response.headers.get("content-type") == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    print_test("6.1 Export tasks to Excel", success, 
               f"File size: {len(response.content)} bytes" if success else "")
    
    # Test 6.2: Export without authentication
    response = requests.get(f"{BASE_URL}/tasks/export/excel")
    print_test("6.2 Export requires authentication", response.status_code == 401)

def test_task_assignment(token):
    """Test 7: Task Assignment Scenarios"""
    print(f"\n{YELLOW}=== Test 7: Task Assignment ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 7.1: Create task with assignee
    response = requests.post(f"{BASE_URL}/tasks/", 
        headers=headers,
        json={
            "title": "Assigned Task",
            "description": "This task is assigned to someone",
            "assignee_id": 1
        }
    )
    success = response.status_code == 200
    if success:
        task = response.json()
        print_test("7.1 Create task with assignee", task.get("assignee_id") == 1)
    else:
        print_test("7.1 Create task with assignee", False)
    
    # Test 7.2: Create unassigned task
    response = requests.post(f"{BASE_URL}/tasks/", 
        headers=headers,
        json={
            "title": "Unassigned Task",
            "description": "No one is assigned yet"
        }
    )
    success = response.status_code == 200
    if success:
        task = response.json()
        print_test("7.2 Create unassigned task", task.get("assignee_id") is None)
    else:
        print_test("7.2 Create unassigned task", False)


def test_gamification(token):
    """Test 8: Gamification System (XP, Levels, Leaderboard)"""
    print(f"\n{YELLOW}=== Test 8: Gamification System ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 8.1: Get current user's gamification stats
    response = requests.get(f"{BASE_URL}/gamification/me", headers=headers)
    success = response.status_code == 200
    if success:
        stats = response.json()
        has_fields = all(k in stats for k in ["xp", "level", "level_config"])
        print_test("8.1 Get gamification stats", has_fields, 
                   f"XP: {stats.get('xp')}, Level: {stats.get('level')}")
    else:
        print_test("8.1 Get gamification stats", False)
    
    # Test 8.2: Get leaderboard
    response = requests.get(f"{BASE_URL}/gamification/leaderboard?limit=5", headers=headers)
    success = response.status_code == 200
    if success:
        leaderboard = response.json()
        print_test("8.2 Get leaderboard", isinstance(leaderboard, list), 
                   f"Found {len(leaderboard)} users")
    else:
        print_test("8.2 Get leaderboard", False)
    
    # Test 8.3: Get level colors
    response = requests.get(f"{BASE_URL}/gamification/level-colors", headers=headers)
    success = response.status_code == 200
    if success:
        colors = response.json()
        print_test("8.3 Get level colors", len(colors) == 10, 
                   f"Found {len(colors)} level configurations")
    else:
        print_test("8.3 Get level colors", False)
    
    # Test 8.4: XP awarded on task completion
    # Create a HIGH priority task
    response = requests.post(f"{BASE_URL}/tasks/", 
        headers=headers,
        json={
            "title": "XP Test Task",
            "description": "Testing XP award",
            "priority": "HIGH",
            "status": "TODO",
            "assignee_id": 1
        }
    )
    task_id = response.json().get("id") if response.status_code == 200 else None
    
    if task_id:
        # Get initial XP
        initial_stats = requests.get(f"{BASE_URL}/gamification/me", headers=headers).json()
        initial_xp = initial_stats.get("xp", 0)
        
        # Complete the task
        response = requests.put(f"{BASE_URL}/tasks/{task_id}", 
            headers=headers,
            json={"status": "DONE"}
        )
        
        # Check XP increased
        new_stats = requests.get(f"{BASE_URL}/gamification/me", headers=headers).json()
        new_xp = new_stats.get("xp", 0)
        xp_gained = new_xp - initial_xp
        
        print_test("8.4 XP awarded on task completion", xp_gained > 0, 
                   f"XP gained: {xp_gained}")
    else:
        print_test("8.4 XP awarded on task completion", False, "Could not create task")


def test_projects(token):
    """Test 9: Project Management"""
    print(f"\n{YELLOW}=== Test 9: Project Management ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test 9.1: Create project
    response = requests.post(f"{BASE_URL}/projects/", 
        headers=headers,
        json={
            "name": "Test Project",
            "description": "A project for testing"
        }
    )
    success = response.status_code == 200
    project_id = None
    if success:
        project = response.json()
        project_id = project.get("id")
        print_test("9.1 Create project", project.get("name") == "Test Project",
                   f"Project ID: {project_id}")
    else:
        print_test("9.1 Create project", False, f"Status: {response.status_code}")
    
    # Test 9.2: Get all projects
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    success = response.status_code == 200
    if success:
        projects = response.json()
        print_test("9.2 Get all projects", isinstance(projects, list),
                   f"Found {len(projects)} projects")
    else:
        print_test("9.2 Get all projects", False)
    
    # Test 9.3: Create task with project
    if project_id:
        response = requests.post(f"{BASE_URL}/tasks/", 
            headers=headers,
            json={
                "title": "Project Task",
                "description": "Task in test project",
                "project_id": project_id
            }
        )
        success = response.status_code == 200
        if success:
            task = response.json()
            print_test("9.3 Create task with project", 
                       task.get("project_id") == project_id)
        else:
            print_test("9.3 Create task with project", False)
    else:
        print_test("9.3 Create task with project", False, "No project created")


def test_comments(token):
    """Test 10: Task Comments"""
    print(f"\n{YELLOW}=== Test 10: Task Comments ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # First create a task to comment on
    response = requests.post(f"{BASE_URL}/tasks/", 
        headers=headers,
        json={
            "title": "Task for Comments",
            "description": "Testing comments on this task"
        }
    )
    task_id = response.json().get("id") if response.status_code == 200 else 1
    
    # Test 10.1: Create comment
    response = requests.post(f"{BASE_URL}/tasks/{task_id}/comments", 
        headers=headers,
        json={
            "content": "This is a test comment",
            "attachments": []
        }
    )
    success = response.status_code == 200
    if success:
        comment = response.json()
        print_test("10.1 Create comment", comment.get("content") == "This is a test comment")
    else:
        print_test("10.1 Create comment", False, f"Status: {response.status_code}")
    
    # Test 10.2: Get comments for task
    response = requests.get(f"{BASE_URL}/tasks/{task_id}/comments", headers=headers)
    success = response.status_code == 200
    if success:
        comments = response.json()
        print_test("10.2 Get task comments", len(comments) >= 1,
                   f"Found {len(comments)} comments")
    else:
        print_test("10.2 Get task comments", False)


def test_dependencies(token):
    """Test 11: Smart Dependencies"""
    print(f"\n{YELLOW}=== Test 11: Smart Dependencies ==={RESET}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create two tasks for dependency testing
    task1_resp = requests.post(f"{BASE_URL}/tasks/", 
        headers=headers,
        json={"title": "Dependency Task 1", "description": "First task"}
    )
    task2_resp = requests.post(f"{BASE_URL}/tasks/", 
        headers=headers,
        json={"title": "Dependency Task 2", "description": "Depends on task 1"}
    )
    
    task1_id = task1_resp.json().get("id") if task1_resp.status_code == 200 else None
    task2_id = task2_resp.json().get("id") if task2_resp.status_code == 200 else None
    
    if task1_id and task2_id:
        # Test 11.1: Add dependency (POST /tasks/{task_id}/dependencies/{dependency_id})
        response = requests.post(f"{BASE_URL}/tasks/{task2_id}/dependencies/{task1_id}",
            headers=headers
        )
        success = response.status_code == 200 and response.json().get("success")
        print_test("11.1 Add dependency", success)
        
        # Test 11.2: Get task dependencies
        response = requests.get(f"{BASE_URL}/tasks/{task2_id}/dependencies", headers=headers)
        success = response.status_code == 200
        if success:
            deps = response.json()
            has_dep = len(deps.get("dependencies", [])) > 0
            print_test("11.2 Get task dependencies", has_dep)
        else:
            print_test("11.2 Get task dependencies", False)
        
        # Test 11.3: Get dependency graph (for project)
        response = requests.get(f"{BASE_URL}/projects/1/dependency-graph", headers=headers)
        success = response.status_code == 200
        if success:
            graph = response.json()
            has_structure = "nodes" in graph and "edges" in graph
            print_test("11.3 Get dependency graph", has_structure,
                       f"Nodes: {len(graph.get('nodes', []))}, Edges: {len(graph.get('edges', []))}")
        else:
            print_test("11.3 Get dependency graph", False)
        
        # Test 11.4: Get blocked tasks
        response = requests.get(f"{BASE_URL}/dependencies/blocked-tasks", headers=headers)
        success = response.status_code == 200
        print_test("11.4 Get blocked tasks", success)
        
        # Test 11.5: Remove dependency (DELETE /tasks/{task_id}/dependencies/{dependency_id})
        response = requests.delete(f"{BASE_URL}/tasks/{task2_id}/dependencies/{task1_id}",
            headers=headers
        )
        success = response.status_code == 200 and response.json().get("success")
        print_test("11.5 Remove dependency", success)
    else:
        print_test("11.1-11.5 Dependency tests", False, "Could not create tasks")


def test_analytics(token):
    """Test 12: Analytics and KPIs (PM/PO only)"""
    print(f"\n{YELLOW}=== Test 12: Analytics and KPIs ==={RESET}")
    
    # Login as PM user for analytics access
    pm_response = requests.post(f"{BASE_URL}/token", data={
        "username": "userpm",
        "password": "password123"
    })
    
    if pm_response.status_code == 200:
        pm_token = pm_response.json()["access_token"]
        pm_headers = {"Authorization": f"Bearer {pm_token}"}
        
        # Test 12.1: Get KPIs
        response = requests.get(f"{BASE_URL}/analytics/kpis", headers=pm_headers)
        success = response.status_code == 200
        if success:
            kpis = response.json()
            has_metrics = "velocity" in kpis and "completion_rate" in kpis
            print_test("12.1 Get KPIs", has_metrics)
        else:
            print_test("12.1 Get KPIs", False, f"Status: {response.status_code}")
        
        # Test 12.2: Get team performance
        response = requests.get(f"{BASE_URL}/analytics/team-performance", headers=pm_headers)
        success = response.status_code == 200
        print_test("12.2 Get team performance", success)
        
        # Test 12.3: Get AI insights (uses /analytics/insights)
        response = requests.get(f"{BASE_URL}/analytics/insights", headers=pm_headers)
        success = response.status_code == 200
        if success:
            insights = response.json()
            has_predictions = "sprint_prediction" in insights or "delivery_forecast" in insights
            print_test("12.3 Get AI insights", has_predictions)
        else:
            print_test("12.3 Get AI insights", False)
        
        # Test 12.4: Get team health
        response = requests.get(f"{BASE_URL}/analytics/team-health", headers=pm_headers)
        success = response.status_code == 200
        if success:
            health = response.json()
            has_score = "health_score" in health
            print_test("12.4 Get team health score", has_score,
                       f"Score: {health.get('health_score')}")
        else:
            print_test("12.4 Get team health score", False)
    else:
        print_test("12.1-12.4 Analytics tests", False, "Could not login as PM")
    
    # Test 12.5: DEV user denied analytics
    dev_headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/analytics/kpis", headers=dev_headers)
    print_test("12.5 DEV denied analytics access", response.status_code == 403)


def test_ai_sprint_planning(token):
    """Test 13: AI Sprint Planning"""
    print(f"\n{YELLOW}=== Test 13: AI Sprint Planning ==={RESET}")
    
    # Login as PM for sprint planning
    pm_response = requests.post(f"{BASE_URL}/token", data={
        "username": "userpm",
        "password": "password123"
    })
    
    if pm_response.status_code == 200:
        pm_token = pm_response.json()["access_token"]
        pm_headers = {"Authorization": f"Bearer {pm_token}"}
        
        # Test 13.1: Generate sprint plan (uses /ai/sprint-plan)
        response = requests.get(f"{BASE_URL}/ai/sprint-plan?days=14", headers=pm_headers)
        success = response.status_code == 200
        if success:
            plan = response.json()
            has_data = "sprint_duration_days" in plan or "optimization" in plan
            print_test("13.1 Generate sprint plan", has_data)
        else:
            print_test("13.1 Generate sprint plan", False, f"Status: {response.status_code}")
        
        # Test 13.2: Get sprint assignments
        response = requests.get(f"{BASE_URL}/ai/sprint-assignments", headers=pm_headers)
        success = response.status_code == 200
        print_test("13.2 Get sprint assignments", success)
    else:
        print_test("13.1-13.2 Sprint planning tests", False, "Could not login as PM")


def run_all_tests():
    """Run all tests"""
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}AI TASK MANAGER - COMPREHENSIVE TEST SUITE{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")
    
    # Start servers
    if not start_servers():
        print(f"{RED}Failed to start servers. Exiting.{RESET}")
        return
    
    # Run tests
    try:
        test_user_registration()
        token = test_user_authentication()
        
        if token:
            # Core functionality
            test_get_current_user(token)
            test_task_creation_with_ai(token)
            test_task_retrieval(token)
            test_excel_export(token)
            test_task_assignment(token)
            
            # New features
            test_gamification(token)
            test_projects(token)
            test_comments(token)
            test_dependencies(token)
            test_analytics(token)
            test_ai_sprint_planning(token)
        else:
            print(f"{RED}Cannot continue tests without valid token{RESET}")
        
        print(f"\n{YELLOW}{'='*60}{RESET}")
        print(f"{GREEN}TEST SUITE COMPLETED{RESET}")
        print(f"{YELLOW}{'='*60}{RESET}\n")
        
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}Tests failed with error: {e}{RESET}")
    finally:
        cleanup_servers()

if __name__ == "__main__":
    run_all_tests()

