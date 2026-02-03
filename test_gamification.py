"""
Comprehensive Feature Test Suite
Tests all features including new Gamification system
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

test_results = []

def print_test(test_name, passed, message=""):
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"{status} - {test_name}")
    if message:
        print(f"    {message}")
    test_results.append({"test": test_name, "passed": passed, "message": message})
    return passed

def test_server_health():
    """Test 1: Server Health Check"""
    print(f"\n{YELLOW}=== Test 1: Server Health ==={RESET}")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        return print_test("1.1 Server is running", response.status_code == 200, f"Status: {response.status_code}")
    except Exception as e:
        print_test("1.1 Server is running", False, str(e))
        return False

def test_authentication():
    """Test 2: Authentication (Existing Feature)"""
    print(f"\n{YELLOW}=== Test 2: Authentication ==={RESET}")
    
    # Test login with existing user
    response = requests.post(f"{BASE_URL}/token", data={
        "username": "puspan das",
        "password": "Dev@123"  # Common test password
    })
    
    if response.status_code != 200:
        # Try another common password
        response = requests.post(f"{BASE_URL}/token", data={
            "username": "qatest",
            "password": "test123"
        })
    
    if response.status_code == 200:
        token = response.json().get("access_token")
        print_test("2.1 User login", True, f"Token obtained")
        return token
    else:
        # Create new test user
        print(f"{YELLOW}    Creating test user...{RESET}")
        reg_response = requests.post(f"{BASE_URL}/users/", json={
            "username": "gamification_test",
            "email": "gamification_test@test.com",
            "password": "test123",
            "role": "DEV"
        })
        
        if reg_response.status_code == 200:
            print_test("2.1a User registration", True, "New user created")
            # Login with new user
            response = requests.post(f"{BASE_URL}/token", data={
                "username": "gamification_test",
                "password": "test123"
            })
            if response.status_code == 200:
                token = response.json().get("access_token")
                print_test("2.1b User login", True, "Token obtained")
                return token
        
        print_test("2.1 User login", False, f"Status: {response.status_code}")
        return None

def test_existing_features(token):
    """Test 3: Existing Features (Tasks, Projects, Comments)"""
    print(f"\n{YELLOW}=== Test 3: Existing Features ==={RESET}")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test get tasks
    response = requests.get(f"{BASE_URL}/tasks/", headers=headers)
    print_test("3.1 Get all tasks", response.status_code == 200, f"Found {len(response.json())} tasks")
    
    # Test get projects
    response = requests.get(f"{BASE_URL}/projects/", headers=headers)
    print_test("3.2 Get all projects", response.status_code == 200, f"Found {len(response.json())} projects")
    
    # Test get users
    response = requests.get(f"{BASE_URL}/users/", headers=headers)
    users = response.json()
    print_test("3.3 Get all users", response.status_code == 200, f"Found {len(users)} users")
    
    # Test get current user
    response = requests.get(f"{BASE_URL}/users/me/", headers=headers)
    if response.status_code == 200:
        user = response.json()
        print_test("3.4 Get current user", True, f"User: {user.get('username')}")
        # Check gamification fields exist
        has_xp = 'xp' in user
        has_level = 'level' in user
        print_test("3.5 User has gamification fields", has_xp and has_level, 
                   f"XP: {user.get('xp', 'N/A')}, Level: {user.get('level', 'N/A')}")
    else:
        print_test("3.4 Get current user", False)

def test_gamification_endpoints(token):
    """Test 4: New Gamification Features"""
    print(f"\n{YELLOW}=== Test 4: Gamification Endpoints (NEW) ==={RESET}")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test get my gamification stats
    response = requests.get(f"{BASE_URL}/gamification/me", headers=headers)
    if response.status_code == 200:
        stats = response.json()
        print_test("4.1 Get my gamification stats", True, 
                   f"XP: {stats.get('xp')}, Level: {stats.get('level')}, Title: {stats.get('level_config', {}).get('title')}")
        
        # Check stats structure
        has_fields = all(k in stats for k in ['xp', 'level', 'level_config', 'xp_to_next_level'])
        print_test("4.2 Stats have required fields", has_fields)
    else:
        print_test("4.1 Get my gamification stats", False, f"Status: {response.status_code}")
    
    # Test get leaderboard
    response = requests.get(f"{BASE_URL}/gamification/leaderboard", headers=headers)
    if response.status_code == 200:
        leaderboard = response.json()
        print_test("4.3 Get leaderboard", True, f"Found {len(leaderboard)} users")
        if len(leaderboard) > 0:
            first = leaderboard[0]
            has_rank = 'rank' in first
            has_level_config = 'level_config' in first
            print_test("4.4 Leaderboard has required fields", has_rank and has_level_config)
    else:
        print_test("4.3 Get leaderboard", False, f"Status: {response.status_code}")
    
    # Test get level colors
    response = requests.get(f"{BASE_URL}/gamification/level-colors", headers=headers)
    if response.status_code == 200:
        colors = response.json()
        print_test("4.5 Get level colors", True, f"Got {len(colors)} level configs")
        # Check level 5 config (Expert)
        level5 = colors.get('5', {})
        print_test("4.6 Level 5 is Expert (Purple)", 
                   level5.get('title') == 'Expert' and '#8b5cf6' in level5.get('color', ''),
                   f"Title: {level5.get('title')}, Color: {level5.get('color')}")
    else:
        print_test("4.5 Get level colors", False, f"Status: {response.status_code}")

def test_xp_awarding(token):
    """Test 5: XP Awarding on Task Completion"""
    print(f"\n{YELLOW}=== Test 5: XP Awarding ==={RESET}")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Get current stats
    response = requests.get(f"{BASE_URL}/gamification/me", headers=headers)
    initial_xp = response.json().get('xp', 0) if response.status_code == 200 else 0
    initial_tasks = response.json().get('total_tasks_completed', 0) if response.status_code == 200 else 0
    print(f"    Initial XP: {initial_xp}, Tasks Completed: {initial_tasks}")
    
    # Get current user to use as assignee
    response = requests.get(f"{BASE_URL}/users/me/", headers=headers)
    current_user_id = response.json().get('id') if response.status_code == 200 else None
    
    if not current_user_id:
        print_test("5.1 Create task for XP test", False, "Could not get user ID")
        return
    
    # Create a HIGH priority task
    response = requests.post(f"{BASE_URL}/tasks/", headers=headers, json={
        "title": "XP Test Task",
        "description": "Testing XP awarding system",
        "priority": "HIGH",
        "status": "TODO",
        "assignee_id": current_user_id
    })
    
    if response.status_code == 200:
        task = response.json()
        task_id = task.get('id')
        print_test("5.1 Create HIGH priority task", True, f"Task ID: {task_id}")
        
        # Move task to DONE - should award 100 XP
        response = requests.put(f"{BASE_URL}/tasks/{task_id}", headers=headers, json={
            "status": "DONE"
        })
        
        if response.status_code == 200:
            print_test("5.2 Move task to DONE", True)
            
            # Check if XP was awarded
            response = requests.get(f"{BASE_URL}/gamification/me", headers=headers)
            if response.status_code == 200:
                new_xp = response.json().get('xp', 0)
                new_tasks = response.json().get('total_tasks_completed', 0)
                xp_gained = new_xp - initial_xp
                print_test("5.3 XP awarded for HIGH task", xp_gained == 100, 
                           f"Expected +100 XP, Got +{xp_gained} XP (Total: {new_xp})")
                print_test("5.4 Tasks completed incremented", new_tasks == initial_tasks + 1,
                           f"Before: {initial_tasks}, After: {new_tasks}")
        else:
            print_test("5.2 Move task to DONE", False, f"Status: {response.status_code}")
    else:
        print_test("5.1 Create HIGH priority task", False, f"Status: {response.status_code}")

def test_analytics_endpoints(token):
    """Test 6: Analytics Endpoints (Manager-only)"""
    print(f"\n{YELLOW}=== Test 6: Analytics Endpoints ==={RESET}")
    headers = {"Authorization": f"Bearer {token}"}
    
    # These may fail if user is not PM/PO - that's expected
    response = requests.get(f"{BASE_URL}/analytics/kpis", headers=headers)
    if response.status_code == 200:
        print_test("6.1 Get KPIs (PM/PO only)", True)
    else:
        print_test("6.1 Get KPIs (PM/PO only)", response.status_code == 403, 
                   f"Status: {response.status_code} (403 expected for non-managers)")

def test_dependency_endpoints(token):
    """Test 7: Dependency Endpoints"""
    print(f"\n{YELLOW}=== Test 7: Dependency Endpoints ==={RESET}")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/dependencies/blocked-tasks", headers=headers)
    print_test("7.1 Get blocked tasks", response.status_code == 200, f"Status: {response.status_code}")

def test_collaboration_endpoints(token):
    """Test 8: Collaboration Endpoints"""
    print(f"\n{YELLOW}=== Test 8: Collaboration Endpoints ==={RESET}")
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/collaboration/active-users", headers=headers)
    print_test("8.1 Get active users", response.status_code == 200)
    
    response = requests.get(f"{BASE_URL}/collaboration/stats", headers=headers)
    print_test("8.2 Get collaboration stats", response.status_code == 200)

def generate_report():
    """Generate test report"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    passed = sum(1 for t in test_results if t['passed'])
    failed = sum(1 for t in test_results if not t['passed'])
    total = len(test_results)
    
    print(f"\n{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")
    print(f"Total: {total}")
    print(f"Pass Rate: {(passed/total*100):.1f}%")
    
    if failed > 0:
        print(f"\n{RED}Failed Tests:{RESET}")
        for t in test_results:
            if not t['passed']:
                print(f"  - {t['test']}: {t['message']}")
    
    return passed, failed

def run_all_tests():
    """Run all tests"""
    print(f"\n{YELLOW}{'='*60}{RESET}")
    print(f"{YELLOW}COMPREHENSIVE FEATURE TEST SUITE{RESET}")
    print(f"{YELLOW}Testing Gamification + All Existing Features{RESET}")
    print(f"{YELLOW}{'='*60}{RESET}")
    
    # Test 1: Server Health
    if not test_server_health():
        print(f"{RED}Server not running! Start with: python -m uvicorn main:app{RESET}")
        return
    
    # Test 2: Authentication
    token = test_authentication()
    if not token:
        print(f"{RED}Cannot continue without authentication{RESET}")
        return
    
    # Test 3: Existing Features
    test_existing_features(token)
    
    # Test 4: Gamification Endpoints
    test_gamification_endpoints(token)
    
    # Test 5: XP Awarding
    test_xp_awarding(token)
    
    # Test 6: Analytics
    test_analytics_endpoints(token)
    
    # Test 7: Dependencies
    test_dependency_endpoints(token)
    
    # Test 8: Collaboration
    test_collaboration_endpoints(token)
    
    # Generate Report
    passed, failed = generate_report()
    
    print(f"\n{GREEN}All tests completed!{RESET}\n")
    return passed, failed

if __name__ == "__main__":
    run_all_tests()
