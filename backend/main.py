from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import schemas, auth, json_storage, ml_service, kpi_service, ai_insights, ai_assistant, dependency_service, collaboration_service, codebase_rag_service, gamification_service
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid

app = FastAPI(title="AI Task Manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Mount uploads directory to serve files
# Mount uploads directory to serve files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def read_root():
    return {"message": "Welcome to TaskFlow API. Visit /docs for documentation."}

@app.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = json_storage.get_user_by_username(form_data.username)
    
    if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = auth.timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user["username"], "role": user["role"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate):
    if json_storage.get_user_by_username(user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    if json_storage.get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = auth.get_password_hash(user.password)
    db_user = json_storage.create_user(user.username, user.email, hashed_password, user.role)
    
    if not db_user:
        raise HTTPException(status_code=500, detail="Failed to create user")
    
    return db_user

@app.get("/users/me/", response_model=schemas.User)
async def read_users_me(current_user: dict = Depends(auth.get_current_user_json)):
    return current_user

@app.get("/users/", response_model=List[schemas.User])
def read_users(current_user: dict = Depends(auth.get_current_user_json)):
    return json_storage.get_all_users()

@app.post("/projects/", response_model=schemas.Project)
def create_project(project: schemas.ProjectCreate, current_user: dict = Depends(auth.get_current_user_json)):
    return json_storage.create_project(project.name, project.description, current_user["id"])

@app.get("/projects/", response_model=List[schemas.Project])
def read_projects(current_user: dict = Depends(auth.get_current_user_json)):
    return json_storage.get_all_projects()

@app.post("/tasks/", response_model=schemas.Task)
def create_task(task: schemas.TaskCreate, current_user: dict = Depends(auth.get_current_user_json)):
    # AI Priority Detection
    priority = task.priority
    if not priority:
        priority = ml_service.predictor.predict(task.description)
    
    db_task = json_storage.create_task(
        title=task.title,
        description=task.description,
        priority=priority,
        status=task.status or "TODO",
        creator_id=current_user["id"],
        assignee_id=task.assignee_id,
        estimated_days=task.estimated_days,
        spent_days=task.spent_days,
        project_id=task.project_id,
        dependencies=task.dependencies or [],
        parent_id=task.parent_id,
        labels=task.labels or [],
        team_id=task.team_id
    )
    return db_task

@app.put("/tasks/{task_id}", response_model=schemas.Task)
def update_task(task_id: int, task_update: schemas.TaskUpdate, current_user: dict = Depends(auth.get_current_user_json)):
    # Get current task state before update
    tasks = json_storage.get_all_tasks()
    current_task = next((t for t in tasks if t.get("id") == task_id), None)
    
    if not current_task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    old_status = current_task.get("status")
    new_status = task_update.status
    
    updated_task = json_storage.update_task(task_id, task_update.dict(exclude_unset=True))
    if not updated_task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Award XP if task was moved to DONE and assignee exists
    if new_status == "DONE" and old_status != "DONE":
        assignee_id = updated_task.get("assignee_id")
        if assignee_id:
            priority = updated_task.get("priority", "MEDIUM")
            description = updated_task.get("description", "")
            xp_amount = gamification_service.get_xp_for_task(priority, description)
            gamification_service.award_xp(assignee_id, xp_amount)
    
    return updated_task

@app.get("/tasks/", response_model=List[schemas.Task])
def read_tasks(project_id: int = None, current_user: dict = Depends(auth.get_current_user_json)):
    tasks = json_storage.get_all_tasks()
    if project_id:
        tasks = [t for t in tasks if t.get("project_id") == project_id]
    return tasks

@app.get("/tasks/export/excel")
def export_tasks_excel(current_user: dict = Depends(auth.get_current_user_json)):
    tasks = json_storage.get_all_tasks()
    data = []
    
    # Get users for lookup
    all_data = json_storage.load_data()
    users_dict = {u["id"]: u["username"] for u in all_data["users"]}
    
    for t in tasks:
        data.append({
            "ID": t["id"],
            "Title": t["title"],
            "Description": t["description"],
            "Priority": t["priority"],
            "Status": t["status"],
            "Est. Days": t.get("estimated_days", 0),
            "Spent Days": t.get("spent_days", 0),
            "Creator": users_dict.get(t["creator_id"], "Unknown"),
            "Assignee": users_dict.get(t["assignee_id"], "Unassigned") if t.get("assignee_id") else "Unassigned"
        })
    
    import pandas as pd
    from fastapi.responses import StreamingResponse
    import io
    
    df = pd.DataFrame(data)
    stream = io.BytesIO()
    with pd.ExcelWriter(stream, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    stream.seek(0)
    
    return StreamingResponse(
        stream, 
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=tasks.xlsx"}
    )

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), current_user: dict = Depends(auth.get_current_user_json)):
    # Generate unique filename
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"filename": file.filename, "url": f"/uploads/{unique_filename}"}

@app.post("/tasks/{task_id}/comments", response_model=schemas.Comment)
def create_comment(task_id: int, comment: schemas.CommentCreate, current_user: dict = Depends(auth.get_current_user_json)):
    # Verify task exists
    tasks = json_storage.get_all_tasks()
    if not any(t["id"] == task_id for t in tasks):
        raise HTTPException(status_code=404, detail="Task not found")
        
    db_comment = json_storage.create_comment(
        task_id=task_id,
        user_id=current_user["id"],
        username=current_user["username"],
        content=comment.content,
        attachments=comment.attachments
    )
    return db_comment

@app.get("/tasks/{task_id}/comments", response_model=List[schemas.Comment])
def read_comments(task_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    return json_storage.get_comments_by_task(task_id)

# Dependency Management Endpoints
@app.get("/tasks/{task_id}/dependencies")
def get_task_dependencies(task_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get dependencies for a specific task"""
    tasks = json_storage.get_all_tasks()
    task = next((t for t in tasks if t.get("id") == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    dep_ids = task.get("dependencies", [])
    dependencies = [t for t in tasks if t.get("id") in dep_ids]
    dependents = dependency_service.get_task_dependents(task_id)
    
    return {
        "task_id": task_id,
        "dependencies": dependencies,
        "dependents": dependents,
        "is_blocked": any(t.get("status") != "DONE" for t in dependencies)
    }

@app.post("/tasks/{task_id}/dependencies/{dependency_id}")
def add_task_dependency(task_id: int, dependency_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Add a dependency to a task"""
    result = dependency_service.add_dependency(task_id, dependency_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.delete("/tasks/{task_id}/dependencies/{dependency_id}")
def remove_task_dependency(task_id: int, dependency_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Remove a dependency from a task"""
    result = dependency_service.remove_dependency(task_id, dependency_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/tasks/{task_id}/suggested-dependencies")
def get_suggested_dependencies(task_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get AI-suggested dependencies for a task"""
    result = dependency_service.suggest_dependencies(task_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/projects/{project_id}/dependency-graph")
def get_project_dependency_graph(project_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get the complete dependency graph for a project"""
    return dependency_service.get_dependency_graph(project_id)

@app.get("/dependencies/blocked-tasks")
def get_blocked_tasks(current_user: dict = Depends(auth.get_current_user_json)):
    """Get all tasks that are currently blocked by dependencies"""
    return dependency_service.get_blocked_tasks()

# Analytics Endpoints (Manager-only)
@app.get("/analytics/kpis")
def get_kpis(current_user: dict = Depends(auth.require_manager_role)):
    """Get all KPI metrics - PM/PO only"""
    return kpi_service.get_all_kpis()

@app.get("/analytics/velocity")
def get_velocity(days: int = 7, current_user: dict = Depends(auth.require_manager_role)):
    """Get task completion velocity - PM/PO only"""
    return kpi_service.calculate_velocity(days)

@app.get("/analytics/team-performance")
def get_team_performance(current_user: dict = Depends(auth.require_manager_role)):
    """Get per-developer performance metrics - PM/PO only"""
    return kpi_service.get_developer_metrics()

@app.get("/analytics/risks")
def get_risks(current_user: dict = Depends(auth.require_manager_role)):
    """Get risk detection results - PM/PO only"""
    return kpi_service.detect_risks()

@app.get("/analytics/insights")
def get_insights(current_user: dict = Depends(auth.require_manager_role)):
    """Get AI-generated insights and predictions - PM/PO only"""
    return ai_insights.get_all_insights()

@app.get("/analytics/sprint-prediction")
def get_sprint_prediction(sprint_days: int = 14, current_user: dict = Depends(auth.require_manager_role)):
    """Predict sprint completion - PM/PO only"""
    return ai_insights.predict_sprint_completion(sprint_days)

@app.get("/analytics/team-health")
def get_team_health(current_user: dict = Depends(auth.require_manager_role)):
    """Get overall team health score - PM/PO only"""
    return ai_insights.analyze_team_health()

# AI Assistant Endpoints (all authenticated users)
@app.get("/ai/code-suggestions/{task_id}")
def get_code_suggestions(task_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get AI code suggestions for a specific task"""
    result = ai_assistant.get_task_analysis(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return result

@app.get("/ai/task-analysis/{task_id}")
def get_task_analysis(task_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get AI complexity analysis for a task"""
    result = ai_assistant.get_task_analysis(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "complexity": result.get("complexity"),
        "categories": result.get("matched_categories")
    }

@app.get("/ai/subtask-suggestions/{task_id}")
def get_subtask_suggestions(task_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get AI-suggested subtasks for breaking down a complex task"""
    tasks = json_storage.get_all_tasks()
    task = next((t for t in tasks if t.get("id") == task_id), None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "suggested_subtasks": ai_assistant.suggest_subtasks(task)}

# Sprint Planning Endpoints (Manager-only)
@app.get("/ai/sprint-plan")
def get_sprint_plan(sprint_days: int = 14, current_user: dict = Depends(auth.require_manager_role)):
    """Get AI-generated sprint plan with optimal assignments - PM/PO only"""
    return ai_insights.generate_sprint_plan(sprint_days)

@app.get("/ai/sprint-assignments")
def get_sprint_assignments(sprint_days: int = 14, current_user: dict = Depends(auth.require_manager_role)):
    """Get AI-suggested task assignments for sprint - PM/PO only"""
    return ai_insights.optimize_sprint_assignments(sprint_days)

@app.post("/ai/apply-assignments")
def apply_sprint_assignments(assignments: List[dict], current_user: dict = Depends(auth.require_manager_role)):
    """Apply suggested assignments to tasks - PM/PO only"""
    applied = []
    errors = []
    
    for assignment in assignments:
        task_id = assignment.get("task_id")
        assignee_id = assignment.get("assignee_id")
        
        if not task_id or not assignee_id:
            errors.append({"task_id": task_id, "error": "Missing task_id or assignee_id"})
            continue
        
        updated = json_storage.update_task(task_id, {"assignee_id": assignee_id})
        if updated:
            applied.append({"task_id": task_id, "assignee_id": assignee_id})
        else:
            errors.append({"task_id": task_id, "error": "Task not found"})
    
    return {"applied": applied, "errors": errors, "total_applied": len(applied)}

# Real-time Collaboration Endpoints
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time collaboration"""
    # Get token from query params for authentication
    token = websocket.query_params.get('token')
    
    if not token:
        await websocket.close(code=4001)
        return
    
    try:
        # Verify token and get user
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if not username:
            await websocket.close(code=4001)
            return
        
        user = json_storage.get_user_by_username(username)
        if not user:
            await websocket.close(code=4001)
            return
        
        user_data = {
            'id': user['id'],
            'username': user['username'],
            'role': user['role']
        }
        
    except Exception:
        await websocket.close(code=4001)
        return
    
    # Connect and handle messages
    await collaboration_service.manager.connect(websocket, client_id, user_data)
    
    try:
        # Broadcast user joined
        await collaboration_service.manager.broadcast({
            'type': 'user_online',
            'data': {
                'user_id': user['id'],
                'username': user['username']
            }
        }, exclude_client=client_id)
        
        while True:
            data = await websocket.receive_json()
            await collaboration_service.handle_message(
                websocket, client_id, user['id'], data
            )
    except WebSocketDisconnect:
        collaboration_service.manager.disconnect(client_id)
        await collaboration_service.manager.broadcast({
            'type': 'user_offline',
            'data': {'user_id': user['id']}
        })
    except Exception as e:
        collaboration_service.manager.disconnect(client_id)

@app.get("/collaboration/active-users")
def get_active_users(current_user: dict = Depends(auth.get_current_user_json)):
    """Get currently active users"""
    return collaboration_service.get_active_users_list()

@app.get("/collaboration/task/{task_id}/viewers")
def get_task_viewers(task_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get users currently viewing a task"""
    return collaboration_service.get_task_viewers_list(task_id)

@app.get("/collaboration/stats")
def get_collaboration_stats(current_user: dict = Depends(auth.get_current_user_json)):
    """Get overall collaboration statistics"""
    return collaboration_service.get_collaboration_stats()

# Gamification Endpoints
@app.get("/gamification/me")
def get_my_gamification_stats(current_user: dict = Depends(auth.get_current_user_json)):
    """Get current user's gamification stats (XP, level, progress)"""
    stats = gamification_service.get_user_gamification_stats(current_user["id"])
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")
    return stats

@app.get("/gamification/user/{user_id}")
def get_user_gamification_stats(user_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get specific user's gamification stats"""
    stats = gamification_service.get_user_gamification_stats(user_id)
    if not stats:
        raise HTTPException(status_code=404, detail="User not found")
    return stats

@app.get("/gamification/leaderboard")
def get_leaderboard(limit: int = 10, current_user: dict = Depends(auth.get_current_user_json)):
    """Get top users by XP for Sprint Champion leaderboard"""
    return gamification_service.get_leaderboard(limit)

@app.get("/gamification/level-colors")
def get_level_colors(current_user: dict = Depends(auth.get_current_user_json)):
    """Get all level color configurations for cursor glow effects"""
    return gamification_service.get_all_level_colors()

# RAG Codebase Chat Endpoints
@app.post("/rag/index")
def rebuild_codebase_index(current_user: dict = Depends(auth.get_current_user_json)):
    """Rebuild the vector index from Python source files"""
    return codebase_rag_service.index_codebase()

@app.get("/rag/status")
def get_rag_index_status(current_user: dict = Depends(auth.get_current_user_json)):
    """Check RAG index status"""
    return codebase_rag_service.get_index_status()

@app.post("/rag/chat")
def chat_with_codebase(query: schemas.CodebaseQuery, current_user: dict = Depends(auth.get_current_user_json)):
    """Query the codebase using natural language"""
    if not query.question or len(query.question.strip()) < 3:
        raise HTTPException(status_code=400, detail="Question must be at least 3 characters")
    return codebase_rag_service.query_codebase(query.question)

# Labels Management Endpoints
@app.get("/labels/", response_model=List[schemas.Label])
def get_labels(current_user: dict = Depends(auth.get_current_user_json)):
    """Get all labels"""
    return json_storage.get_all_labels()

@app.post("/labels/", response_model=schemas.Label)
def create_label(label: schemas.LabelCreate, current_user: dict = Depends(auth.get_current_user_json)):
    """Create a new label"""
    if not label.name or not label.name.strip():
        raise HTTPException(status_code=400, detail="Label name is required")
    return json_storage.create_label(label.name.strip(), label.color)

@app.delete("/labels/{label_id}")
def delete_label(label_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Delete a label"""
    success = json_storage.delete_label(label_id)
    if not success:
        raise HTTPException(status_code=404, detail="Label not found")
    return {"message": "Label deleted successfully"}

# Teams Management Endpoints
@app.get("/teams/", response_model=List[schemas.Team])
def get_teams(current_user: dict = Depends(auth.get_current_user_json)):
    """Get all teams"""
    return json_storage.get_all_teams()

@app.post("/teams/", response_model=schemas.Team)
def create_team(team: schemas.TeamCreate, current_user: dict = Depends(auth.get_current_user_json)):
    """Create a new team"""
    if not team.name or not team.name.strip():
        raise HTTPException(status_code=400, detail="Team name is required")
    return json_storage.create_team(team.name.strip(), team.description or "", current_user["id"])

@app.get("/teams/{team_id}", response_model=schemas.Team)
def get_team(team_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Get a specific team"""
    team = json_storage.get_team_by_id(team_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team

@app.delete("/teams/{team_id}")
def delete_team(team_id: int, current_user: dict = Depends(auth.get_current_user_json)):
    """Delete a team"""
    success = json_storage.delete_team(team_id)
    if not success:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"message": "Team deleted successfully"}
