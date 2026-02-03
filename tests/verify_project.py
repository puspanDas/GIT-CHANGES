import requests
import json

BASE_URL = "http://localhost:8000"

def login():
    response = requests.post(f"{BASE_URL}/token", data={"username": "puspan das", "password": "password123"}) # Assuming password from data.json hash or known dev password. 
    # Wait, I don't know the password. I can create a new user or use existing token if I can get it. 
    # Or I can just disable auth for testing? No, that's bad.
    # I'll try to create a new user first.
    return response.json()

def create_user():
    user = {
        "username": "testuser_project",
        "email": "test_project@example.com",
        "password": "testpassword",
        "role": "DEV"
    }
    response = requests.post(f"{BASE_URL}/users/", json=user)
    if response.status_code == 400: # Already exists
        return login_user("testuser_project", "testpassword")
    return login_user("testuser_project", "testpassword")

def login_user(username, password):
    response = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
    return response.json()["access_token"]

def verify():
    try:
        token = create_user()
        headers = {"Authorization": f"Bearer {token}"}
        
        # 1. Create Project
        print("Creating Project...")
        project_data = {"name": "API Test Project", "description": "Testing API"}
        response = requests.post(f"{BASE_URL}/projects/", json=project_data, headers=headers)
        print(response.status_code, response.json())
        project_id = response.json()["id"]
        
        # 2. List Projects
        print("Listing Projects...")
        response = requests.get(f"{BASE_URL}/projects/", headers=headers)
        print(response.json())
        
        # 3. Create Task in Project
        print("Creating Task in Project...")
        task_data = {
            "title": "Project Task",
            "description": "Task in project",
            "project_id": project_id
        }
        response = requests.post(f"{BASE_URL}/tasks/", json=task_data, headers=headers)
        print(response.json())
        
        # 4. List Tasks in Project
        print("Listing Tasks in Project...")
        response = requests.get(f"{BASE_URL}/tasks/?project_id={project_id}", headers=headers)
        tasks = response.json()
        print(f"Tasks in project {project_id}: {len(tasks)}")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Project Task"
        
        # 5. List Tasks Global (or other project)
        print("Listing Tasks Global (no project_id)...")
        response = requests.get(f"{BASE_URL}/tasks/", headers=headers)
        all_tasks = response.json()
        # Should include the project task? 
        # My implementation of read_tasks:
        # if project_id: filter
        # else: return all
        # So it should be included.
        print(f"Total tasks: {len(all_tasks)}")
        
        print("Verification Successful!")
        
    except Exception as e:
        print(f"Verification Failed: {e}")

if __name__ == "__main__":
    verify()
