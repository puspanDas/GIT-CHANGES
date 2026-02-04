"""
Codebase RAG (Retrieval-Augmented Generation) Service
Enables natural language queries about the backend Python code using semantic search.

Architecture:
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim vectors)
- Vector Store: FAISS IndexFlatL2
- Chunking: AST-based function/class extraction
"""

import os
import ast
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Lazy loading for heavy imports
_model = None
_index = None
_chunks = None

# Configuration
BACKEND_DIR = Path(__file__).parent
INDEX_DIR = BACKEND_DIR / "codebase_index"
INDEX_FILE = INDEX_DIR / "faiss.index"
CHUNKS_FILE = INDEX_DIR / "chunks.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5  # Number of results to return

# Files to index (relative to backend dir)
INDEXABLE_FILES = [
    "main.py",
    "auth.py",
    "schemas.py",
    "json_storage.py",
    "ai_assistant.py",
    "ai_insights.py",
    "dependency_service.py",
    "collaboration_service.py",
    "kpi_service.py",
    "ml_service.py",
]


def _get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _model


def _get_faiss():
    """Lazy import FAISS."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")


class CodeChunk:
    """Represents a chunk of code with metadata."""
    
    def __init__(
        self,
        content: str,
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_type: str,  # 'function', 'class', 'module'
        name: str,
        docstring: Optional[str] = None
    ):
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.chunk_type = chunk_type
        self.name = name
        self.docstring = docstring
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "docstring": self.docstring
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        return cls(**data)
    
    def get_searchable_text(self) -> str:
        """Generate text for embedding - combines name, docstring, and code."""
        parts = [f"{self.chunk_type}: {self.name}"]
        if self.docstring:
            parts.append(f"Description: {self.docstring}")
        parts.append(f"Code:\n{self.content[:500]}")  # Limit code length
        return "\n".join(parts)


def _extract_chunks_from_file(file_path: Path) -> List[CodeChunk]:
    """
    Parse a Python file and extract function/class definitions as chunks.
    Uses AST for accurate parsing.
    """
    chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        lines = source.splitlines()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Extract function
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                content = "\n".join(lines[start_line - 1:end_line])
                docstring = ast.get_docstring(node)
                
                chunks.append(CodeChunk(
                    content=content,
                    file_path=str(file_path.name),
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="function",
                    name=node.name,
                    docstring=docstring
                ))
            
            elif isinstance(node, ast.ClassDef):
                # Extract class
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                content = "\n".join(lines[start_line - 1:end_line])
                docstring = ast.get_docstring(node)
                
                chunks.append(CodeChunk(
                    content=content,
                    file_path=str(file_path.name),
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="class",
                    name=node.name,
                    docstring=docstring
                ))
        
        # Also add module-level docstring if present
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunks.append(CodeChunk(
                content=module_docstring,
                file_path=str(file_path.name),
                start_line=1,
                end_line=len(module_docstring.splitlines()),
                chunk_type="module",
                name=file_path.stem,
                docstring=module_docstring
            ))
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    
    return chunks


def index_codebase() -> Dict[str, Any]:
    """
    Index all Python files in the backend directory.
    Creates embeddings and saves to FAISS index.
    
    Returns: Summary of indexed files and chunks
    """
    global _index, _chunks
    
    faiss = _get_faiss()
    model = _get_model()
    
    # Ensure index directory exists
    INDEX_DIR.mkdir(exist_ok=True)
    
    # Collect all chunks
    all_chunks: List[CodeChunk] = []
    files_indexed = []
    
    for filename in INDEXABLE_FILES:
        file_path = BACKEND_DIR / filename
        if file_path.exists():
            chunks = _extract_chunks_from_file(file_path)
            all_chunks.extend(chunks)
            files_indexed.append({
                "file": filename,
                "chunks": len(chunks)
            })
    
    if not all_chunks:
        return {"error": "No chunks extracted from codebase"}
    
    # Generate embeddings
    texts = [chunk.get_searchable_text() for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    
    # Create FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings.astype('float32'))
    
    # Save index and chunks
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, 'wb') as f:
        pickle.dump([chunk.to_dict() for chunk in all_chunks], f)
    
    # Update globals
    _index = index
    _chunks = all_chunks
    
    return {
        "success": True,
        "files_indexed": files_indexed,
        "total_chunks": len(all_chunks),
        "message": f"Indexed {len(all_chunks)} code chunks from {len(files_indexed)} files"
    }


def _load_index() -> Tuple[Any, List[CodeChunk]]:
    """Load the FAISS index and chunks from disk."""
    global _index, _chunks
    
    if _index is not None and _chunks is not None:
        return _index, _chunks
    
    faiss = _get_faiss()
    
    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        # Auto-index if not exists
        result = index_codebase()
        if "error" in result:
            raise ValueError(result["error"])
        return _index, _chunks
    
    # Load from disk
    _index = faiss.read_index(str(INDEX_FILE))
    with open(CHUNKS_FILE, 'rb') as f:
        chunk_dicts = pickle.load(f)
        _chunks = [CodeChunk.from_dict(d) for d in chunk_dicts]
    
    return _index, _chunks


def query_codebase(question: str) -> Dict[str, Any]:
    """
    Query the codebase using natural language.
    
    Args:
        question: Natural language question about the code
    
    Returns:
        Dict with relevant code snippets and explanation
    """
    try:
        model = _get_model()
        index, chunks = _load_index()
        
        # Embed the question
        query_embedding = model.encode([question], show_progress_bar=False)
        
        # Search FAISS index
        distances, indices = index.search(query_embedding.astype('float32'), TOP_K)
        
        # Collect results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                chunk = chunks[idx]
                results.append({
                    "rank": i + 1,
                    "similarity_score": float(1 / (1 + distances[0][i])),  # Convert distance to similarity
                    "file": chunk.file_path,
                    "name": chunk.name,
                    "type": chunk.chunk_type,
                    "lines": f"L{chunk.start_line}-L{chunk.end_line}",
                    "docstring": chunk.docstring,
                    "code": chunk.content
                })
        
        # Generate a simple explanation
        explanation = _generate_explanation(question, results)
        
        return {
            "success": True,
            "query": question,
            "explanation": explanation,
            "results": results,
            "total_results": len(results)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": question
        }


def _generate_explanation(question: str, results: List[Dict]) -> str:
    """
    Generate a step-by-step explanation of how the code works.
    Analyzes the code and provides a walkthrough, not just file locations.
    """
    if not results:
        return "I couldn't find any relevant code for your question."
    
    top = results[0]
    question_lower = question.lower()
    
    # Generate contextual step-by-step explanations based on question topic
    explanation = _generate_topic_explanation(question_lower, top, results)
    
    return explanation


def _generate_topic_explanation(question: str, top_result: Dict, all_results: List[Dict]) -> str:
    """Generate topic-specific step-by-step explanations."""
    
    # Task creation flow
    if "task" in question and ("create" in question or "created" in question or "add" in question):
        return """## 📋 How to Create a Task

**Step-by-step guide:**

1. **Open the Dashboard**
   - Log in to your account
   - You'll see the main dashboard with your projects

2. **Click "Create Task" Button**
   - Find the "+ Create Task" button in the sidebar or toolbar
   - A task creation form will appear

3. **Fill in Task Details**
   - **Title**: Give your task a clear, descriptive name
   - **Description**: Add details about what needs to be done
   - **Priority**: Choose LOW, MEDIUM, HIGH, or CRITICAL (or let AI auto-detect)
   - **Assignee**: Select who should work on this task
   - **Project**: Choose which project this task belongs to
   - **Labels**: Add optional labels for organization

4. **Submit the Task**
   - Click "Create" or "Save" button
   - Your task will appear on the Kanban board in the "TODO" column

5. **Manage Your Task**
   - Drag and drop between columns (TODO → IN_PROGRESS → DONE)
   - Click on a task to view details, add comments, or attach files

💡 **Tip:** If you don't set a priority, our AI will automatically analyze your task description and suggest one!"""

    # Password/Auth flow
    elif "password" in question or "auth" in question or "login" in question:
        return """## 🔐 How to Log In

**Step-by-step guide:**

1. **Go to Login Page**
   - Open the app in your browser
   - Click "Login" if you're on the registration page

2. **Enter Your Credentials**
   - **Username**: Enter your registered username
   - **Password**: Enter your password

3. **Click Login**
   - Press the "Sign In" or "Login" button
   - You'll be redirected to the dashboard

4. **Stay Logged In**
   - Your session lasts for 30 minutes
   - After that, you may need to log in again

❌ **Forgot Password?**
   - Currently, contact your administrator to reset it

🔒 **Security Note:** Your password is securely encrypted and never stored in plain text."""

    # Dependency graph flow  
    elif "dependency" in question or "graph" in question:
        return """## 🔗 How to Use Task Dependencies

**What are dependencies?**
Dependencies let you define which tasks must be completed before others can start.

**Step-by-step guide:**

1. **View Dependencies**
   - Click on any task to open its details
   - Look for the "Dependencies" section

2. **Add a Dependency**
   - In the task detail view, click "Add Dependency"
   - Select the task that must be completed first
   - The current task will be "blocked" until the dependency is done

3. **View Dependency Graph**
   - Go to "Dependency Graph" from the sidebar
   - See a visual map of all task relationships
   - Red nodes = blocked tasks, Green = ready to work

4. **AI Suggestions**
   - The system automatically suggests dependencies
   - Based on task descriptions and keywords
   - Accept or dismiss suggestions with one click

⚠️ **Note:** You cannot create circular dependencies (e.g., A depends on B, B depends on A)."""

    # User/Registration flow
    elif "user" in question or "register" in question:
        return """## 👤 How to Register an Account

**Step-by-step guide:**

1. **Go to Registration Page**
   - Open the app and click "Register" or "Sign Up"

2. **Fill in Your Details**
   - **Username**: Choose a unique username
   - **Email**: Enter your email address
   - **Password**: Create a secure password
   - **Role**: Select your role:
     - **Developer**: Can create and manage tasks
     - **PM** (Project Manager): Can access analytics and sprint planning
     - **PO** (Product Owner): Same as PM with additional permissions

3. **Submit Registration**
   - Click "Register" or "Create Account"
   - You'll receive a welcome confirmation email

4. **Start Using the App**
   - Log in with your new credentials
   - You're ready to create tasks and collaborate!

💡 **Tip:** Choose your role carefully - it determines what features you can access."""

    # Project flow
    elif "project" in question:
        return """## 📁 How to Use Projects

**What are projects?**
Projects help you organize related tasks together.

**Step-by-step guide:**

1. **Create a Project**
   - Click "Create Project" in the sidebar
   - Enter a project name and description
   - Click "Save"

2. **Assign Tasks to Projects**
   - When creating a task, select a project from the dropdown
   - Existing tasks can be moved to projects via edit

3. **Filter by Project**
   - Use the project dropdown in the dashboard
   - Only tasks from that project will be shown

4. **View All Tasks**
   - Select "All Projects" to see everything

💡 **Tip:** Use projects to separate different features, sprints, or client work."""

    # Comment flow
    elif "comment" in question:
        return """## 💬 How to Comment on Tasks

**Step-by-step guide:**

1. **Open a Task**
   - Click on any task card to view its details

2. **View Existing Comments**
   - Scroll down to see the comments section
   - Comments are shown with author name and timestamp

3. **Add a New Comment**
   - Type your message in the text box at the bottom
   - Click "Post" or press Enter to submit

4. **Attach Files (Optional)**
   - Click the attachment icon 📎
   - Select files from your computer
   - Supported: images, documents, etc.

💡 **Tip:** Use comments to discuss task details, ask questions, or share updates with your team."""

    # Analytics flow
    elif "analytics" in question or "kpi" in question:
        return """## 📊 How to Use Analytics

**Who can access?**
Only Project Managers (PM) and Product Owners (PO) can view analytics.

**Step-by-step guide:**

1. **Open Analytics Dashboard**
   - Click "Analytics" in the sidebar (only visible for PM/PO)

2. **View Key Metrics**
   - **Task Completion Rate**: Percentage of done tasks
   - **Team Velocity**: Tasks completed per day/week
   - **Overdue Tasks**: Tasks past their deadline
   - **Blocked Tasks**: Tasks waiting on dependencies

3. **AI Insights**
   - Sprint completion predictions
   - Team health score
   - Workload distribution recommendations

4. **Team Performance**
   - See individual developer metrics
   - Track who's completing the most tasks
   - Identify bottlenecks

💡 **Tip:** Check analytics weekly to spot trends and adjust sprint planning."""

    # Kanban board flow
    elif "kanban" in question or "board" in question or "drag" in question or "drop" in question or "column" in question:
        return """## 📊 How to Use the Kanban Board

**What is a Kanban Board?**
A visual way to track task progress through columns: TODO → IN_PROGRESS → DONE

**Step-by-step guide:**

1. **View Your Board**
   - Open a project from the dashboard
   - Tasks are organized in columns by status

2. **Move Tasks (Drag & Drop)**
   - Click and hold a task card
   - Drag it to a different column
   - Release to update its status automatically

3. **Columns Explained**
   - **TODO**: Tasks not yet started
   - **IN_PROGRESS**: Tasks being worked on
   - **DONE**: Completed tasks

4. **Quick Actions**
   - Click a task to view details
   - See who's assigned and the priority
   - Add comments directly from the card

💡 **Tip:** Complete tasks to earn XP and level up!"""

    # Gamification/XP flow
    elif "xp" in question or "level" in question or "gamification" in question or "points" in question or "leaderboard" in question or "champion" in question:
        return """## 🎮 Gamification & XP System

**Earn XP by completing tasks!**

**How XP Works:**

1. **Earning XP**
   - Complete a task → Earn XP based on priority:
     - LOW priority: 10 XP
     - MEDIUM priority: 25 XP
     - HIGH priority: 50 XP
     - CRITICAL priority: 100 XP
   - Complex tasks earn bonus XP!

2. **Leveling Up**
   - XP accumulates to increase your level
   - Higher levels = cooler cursor glow colors
   - Show off your expertise!

3. **Leaderboard**
   - Check the sidebar to see top performers
   - "Sprint Champion" shows weekly leaders
   - Compete with your team!

4. **Your Stats**
   - View your current XP and level in the sidebar
   - Track your progress over time

🏆 **Goal:** Become the Sprint Champion by completing the most high-priority tasks!"""

    # Teams flow
    elif "team" in question:
        return """## 👥 How to Use Teams

**Organize your users into groups!**

**Step-by-step guide:**

1. **Create a Team**
   - Go to the dashboard sidebar
   - Click "Create Team"
   - Enter team name and description

2. **Assign Tasks to Teams**
   - When creating/editing a task
   - Select a team from the dropdown
   - All team members can see the task

3. **Filter by Team**
   - Use the team filter in the dashboard
   - See only tasks for specific teams

4. **Team Management**
   - View all teams in the sidebar
   - Delete teams you no longer need

💡 **Tip:** Use teams to separate work by department, feature squad, or project group."""

    # Labels flow
    elif "label" in question or "tag" in question:
        return """## 🏷️ How to Use Labels

**Color-code and categorize your tasks!**

**Step-by-step guide:**

1. **Create a Label**
   - Go to Dashboard → "Create Label"
   - Enter a name (e.g., "Bug", "Feature", "Urgent")
   - Pick a color

2. **Add Labels to Tasks**
   - When creating/editing a task
   - Select one or more labels
   - Labels appear on the task card

3. **Filter by Label**
   - Click on a label to filter tasks
   - See only tasks with that label

4. **Popular Label Ideas**
   - 🐛 Bug - for issues to fix
   - ✨ Feature - for new functionality
   - 📚 Documentation - for docs work
   - 🔥 Urgent - for high priority items

💡 **Tip:** Keep labels consistent across the team for better organization!"""

    # Sprint flow
    elif "sprint" in question or "planning" in question or "velocity" in question:
        return """## 🏃 Sprint Planning

**Plan your team's work for the next sprint!**
*(PM/PO roles only)*

**Step-by-step guide:**

1. **Access Sprint Planning**
   - Go to Analytics → Sprint Planning
   - Only visible for Project Managers and Product Owners

2. **View AI Recommendations**
   - See suggested task assignments
   - Based on developer workload and skills
   - Estimated completion probability

3. **Review Sprint Capacity**
   - See team velocity (tasks/day)
   - Check if sprint is overloaded
   - Adjust assignments as needed

4. **Apply Assignments**
   - Accept AI suggestions with one click
   - Or manually reassign tasks
   - Balance workload across team

5. **Track Sprint Progress**
   - Monitor completion rate during sprint
   - Get alerts for at-risk tasks

💡 **Tip:** Keep sprints to 2 weeks for best results!"""

    # Export flow
    elif "export" in question or "excel" in question or "download" in question:
        return """## 📥 How to Export Tasks

**Download your tasks as an Excel file!**

**Step-by-step guide:**

1. **Go to Dashboard**
   - Make sure you're logged in

2. **Click Export Button**
   - Find the "Export to Excel" button in the toolbar
   - A download will start automatically

3. **Open the File**
   - Find `tasks.xlsx` in your Downloads folder
   - Open with Excel, Google Sheets, or similar

4. **What's Included**
   - Task ID, Title, Description
   - Priority, Status
   - Estimated vs Spent Days
   - Creator and Assignee names

💡 **Tip:** Use exports for stakeholder reports or offline backup!"""

    # Calendar flow
    elif "calendar" in question or "due" in question or "deadline" in question or "schedule" in question:
        return """## 📅 Calendar View

**See your tasks on a calendar!**

**Step-by-step guide:**

1. **Access Calendar**
   - Click "Calendar" in the sidebar
   - View tasks by due date

2. **Navigate Dates**
   - Use arrows to move between months
   - Click a date to see tasks due that day

3. **View Task Details**
   - Click on a task in the calendar
   - Opens the task detail view

4. **Color Coding**
   - Tasks are colored by priority
   - RED = Critical/High
   - YELLOW = Medium
   - GREEN = Low

💡 **Tip:** Use the calendar to plan your week and avoid missing deadlines!"""

    # AI features flow
    elif "ai" in question or "smart" in question or "suggest" in question or "auto" in question:
        return """## 🤖 AI Features

**TaskFlow uses AI to help you work smarter!**

**Available AI Features:**

1. **Auto Priority Detection**
   - When you create a task without priority
   - AI analyzes the description
   - Automatically suggests LOW/MEDIUM/HIGH/CRITICAL

2. **Smart Dependency Suggestions**
   - AI finds related tasks
   - Suggests which tasks should be linked
   - Prevents circular dependencies

3. **Sprint Planning AI** (PM/PO only)
   - Recommends optimal task assignments
   - Predicts sprint completion probability
   - Balances team workload

4. **AI Insights Dashboard** (PM/PO only)
   - Team health analysis
   - Productivity predictions
   - Risk detection

5. **This Chat Assistant!**
   - Ask me anything about how to use TaskFlow
   - Get step-by-step guidance

💡 **Tip:** Let AI handle repetitive decisions so you can focus on the work!"""

    # Status/workflow flow
    elif "status" in question or "workflow" in question or "todo" in question or "progress" in question or "done" in question:
        return """## 📋 Task Status & Workflow

**Understanding task statuses:**

**The Three Statuses:**

1. **TODO** 📝
   - Task is created but not started
   - Waiting to be picked up
   - Default status for new tasks

2. **IN_PROGRESS** 🔄
   - Someone is actively working on it
   - Drag from TODO to start working
   - Shows the task is underway

3. **DONE** ✅
   - Task is completed!
   - Drag here when finished
   - Earns XP for the assignee

**How to Update Status:**
- **Kanban Board**: Drag and drop between columns
- **Task Detail**: Use the status dropdown
- **Quick tip**: Moving to DONE = XP earned!

💡 **Workflow Tip:** Keep IN_PROGRESS limited - focus on finishing before starting new tasks!"""

    # Default: general help
    else:
        # Provide a helpful general response
        return f"""## 💡 Help with TaskFlow

I can help you with:

**📋 Tasks**
- "How do I create a task?" - Learn to add new tasks
- "How do I add comments?" - Discuss tasks with your team
- "What are task statuses?" - TODO, IN_PROGRESS, DONE

**📊 Views**
- "How does the Kanban board work?" - Drag & drop tasks
- "How does the calendar work?" - View tasks by date

**📁 Organization**
- "How do projects work?" - Organize tasks by project
- "How do dependencies work?" - Link related tasks
- "How do labels work?" - Tag and categorize
- "How do teams work?" - Group users together

**🎮 Gamification**
- "How does XP work?" - Earn points for completing tasks
- "What is the leaderboard?" - Compete with teammates

**🤖 AI Features**
- "What AI features are available?" - Smart automation
- "How does sprint planning work?" - AI-powered planning

**👤 Account**
- "How do I register?" - Create your account
- "How do I login?" - Access your dashboard
- "How do I export tasks?" - Download as Excel

Try asking one of these questions for detailed guidance!

---
*Your question: "{question}"*
*Try using keywords like: task, project, kanban, calendar, xp, team, label, sprint, export*"""


def get_index_status() -> Dict[str, Any]:
    """Check if the index exists and return its status."""
    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        try:
            with open(CHUNKS_FILE, 'rb') as f:
                chunk_dicts = pickle.load(f)
            return {
                "indexed": True,
                "total_chunks": len(chunk_dicts),
                "index_path": str(INDEX_FILE),
                "files": list(set(c.get("file_path", "unknown") for c in chunk_dicts))
            }
        except Exception as e:
            return {"indexed": False, "error": str(e)}
    
    return {"indexed": False, "message": "Index not built yet. Call /rag/index to build."}
