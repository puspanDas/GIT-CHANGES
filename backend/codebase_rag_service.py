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
        return """## 📋 How Tasks Are Created

**Step-by-step process:**

1. **Frontend Request**
   - User fills out the task form in `Dashboard.jsx`
   - Clicks "Create Task" → calls `createTask()` in `api.js`

2. **API Endpoint** (`main.py`)
   - `POST /tasks/` receives the request
   - Validates user authentication via `auth.get_current_user_json`

3. **AI Priority Detection** (`ml_service.py`)
   - If no priority specified, AI analyzes the description
   - Automatically assigns LOW/MEDIUM/HIGH/CRITICAL

4. **Database Storage** (`json_storage.py`)
   - `create_task()` generates a unique task ID
   - Saves to `data.json` with all task properties
   - Returns the created task object

5. **Response**
   - Task data returned to frontend
   - UI updates to show the new task

**Key files:** `main.py` (endpoint), `json_storage.py` (storage), `ml_service.py` (AI priority)"""

    # Password/Auth flow
    elif "password" in question or "auth" in question or "login" in question:
        return """## 🔐 How Authentication Works

**Step-by-step process:**

1. **User Login** (Frontend)
   - User enters credentials in `Login.jsx`
   - Calls `POST /token` endpoint

2. **Password Verification** (`auth.py`)
   - `verify_password()` extracts salt from stored hash
   - Re-hashes input with same salt
   - Compares hashes (SHA256 with salt)

3. **Token Generation** (`auth.py`)
   - `create_access_token()` creates JWT
   - Contains username, role, expiration (30 min)
   - Signed with SECRET_KEY

4. **Subsequent Requests**
   - Frontend stores token in localStorage
   - Sends `Authorization: Bearer <token>` header
   - `get_current_user_json()` validates each request

**Key files:** `auth.py` (password + JWT), `main.py` (login endpoint)"""

    # Dependency graph flow  
    elif "dependency" in question or "graph" in question:
        return """## 🔗 How the Dependency Graph Works

**Step-by-step process:**

1. **Dependency Storage**
   - Each task has a `dependencies` field (list of task IDs)
   - Stored in `data.json` via `json_storage.py`

2. **Graph Construction** (`dependency_service.py`)
   - `get_dependency_graph()` builds nodes & edges
   - Each task = node, each dependency = edge

3. **AI Suggestions**
   - `suggest_dependencies()` analyzes task descriptions
   - Uses keyword matching to find related tasks
   - Detects patterns like "depends on #123"

4. **Circular Reference Check**
   - `validate_dependency()` prevents cycles
   - Uses DFS to detect if adding creates a loop

5. **Frontend Visualization** (`DependencyGraph.jsx`)
   - Renders interactive graph with vis.js
   - Shows blocked vs. ready tasks

**Key files:** `dependency_service.py` (logic), `DependencyGraph.jsx` (UI)"""

    # User/Registration flow
    elif "user" in question or "register" in question:
        return """## 👤 How User Registration Works

**Step-by-step process:**

1. **Registration Form** (`Register.jsx`)
   - User enters username, email, password, role
   - Calls `POST /users/` endpoint

2. **Validation** (`main.py`)
   - Checks if username already exists
   - Checks if email already registered
   - Returns 400 error if duplicate found

3. **Password Hashing** (`auth.py`)
   - `get_password_hash()` generates random salt
   - Creates SHA256 hash of (password + salt)
   - Stores as "salt:hash" format for later verification

4. **Storage** (`json_storage.py`)
   - `create_user()` assigns unique user ID
   - Saves to `data.json` users array

**Key files:** `auth.py` (hashing), `json_storage.py` (storage), `schemas.py` (validation)"""

    # Project flow
    elif "project" in question:
        return """## 📁 How Projects Work

**Step-by-step process:**

1. **Create Project**
   - `POST /projects/` with name & description
   - `json_storage.create_project()` saves it

2. **Task Association**
   - Tasks have optional `project_id` field
   - Filter tasks by project via `GET /tasks/?project_id=X`

3. **Project Selection** (Frontend)
   - `Dashboard.jsx` shows project dropdown
   - Selecting a project filters tasks view

**Key files:** `main.py` (endpoints), `json_storage.py` (storage)"""

    # Comment flow
    elif "comment" in question:
        return """## 💬 How Comments Work

**Step-by-step process:**

1. **View Task Details**
   - Click task → opens `TaskDetailView.jsx`
   - Fetches comments via `GET /tasks/{id}/comments`

2. **Add Comment**
   - User types comment, optionally attaches files
   - Files uploaded via `POST /upload/` first
   - Comment saved via `POST /tasks/{id}/comments`

3. **Storage** (`json_storage.py`)
   - `create_comment()` stores content, user, attachments
   - Comments sorted by created_at (newest first)

**Key files:** `main.py` (endpoints), `json_storage.py` (storage), `TaskDetailView.jsx` (UI)"""

    # Analytics flow
    elif "analytics" in question or "kpi" in question:
        return """## 📊 How Analytics Work

**Step-by-step process:**

1. **Access Control**
   - Only PM/PO roles can access analytics
   - `require_manager_role()` checks user role

2. **KPI Calculation** (`kpi_service.py`)
   - Analyzes all tasks for completion rates
   - Calculates velocity (tasks/day)
   - Identifies overdue and blocked tasks

3. **AI Insights** (`ai_insights.py`)
   - Predicts sprint completion probability
   - Detects team health issues
   - Suggests workload balancing

**Key files:** `kpi_service.py`, `ai_insights.py`, `Analytics.jsx`"""

    # Default: code-focused explanation
    else:
        return f"""## 🔍 Code Search Results

Based on your question about **"{question}"**:

**Primary match:** `{top_result['name']}` in `{top_result['file']}`
- Location: lines {top_result['lines']}
- Type: {top_result['type']}

{f"> {top_result['docstring']}" if top_result.get('docstring') else ""}

**How to explore further:**
1. Open `{top_result['file']}` and go to line {top_result['lines'].split('-')[0][1:]}
2. Read the function/class docstring for purpose
3. Trace the callers to understand the flow

**Related code in:** {', '.join(f"`{r['file']}`" for r in all_results[1:4])}"""


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
