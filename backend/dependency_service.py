"""
Smart Dependency Detection Service
Auto-detects task dependencies from descriptions, code references, and keywords
"""
from typing import Dict, List, Any, Optional, Set, Tuple
import re
import json_storage


# Dependency detection patterns
EXPLICIT_DEPENDENCY_PATTERNS = [
    r'depends\s+on\s+#?(\d+)',           # "depends on #123" or "depends on 123"
    r'blocked\s+by\s+#?(\d+)',            # "blocked by #45"
    r'requires\s+#?(\d+)',                # "requires #12"
    r'after\s+#?(\d+)',                   # "after #67"
    r'prerequisite:\s*#?(\d+)',           # "prerequisite: #89"
    r'needs\s+task\s+#?(\d+)',            # "needs task #34"
    r'waiting\s+on\s+#?(\d+)',            # "waiting on #56"
]

KEYWORD_DEPENDENCY_PATTERNS = [
    r'depends\s+on\s+["\']?([^"\'#\d][^"\']+)["\']?',   # "depends on Login feature"
    r'requires\s+["\']?([^"\'#\d][^"\']+)["\']?',       # "requires Authentication"
    r'after\s+["\']?([^"\'#\d][^"\']+)["\']?',          # "after User setup"
    r'blocked\s+by\s+["\']?([^"\'#\d][^"\']+)["\']?',   # "blocked by API design"
    r'needs\s+["\']?([^"\'#\d][^"\']+)["\']?',          # "needs Database schema"
]

# Code/file reference patterns
CODE_REFERENCE_PATTERNS = [
    r'(?:modify|update|change|edit)\s+(\w+\.(?:py|js|jsx|ts|tsx|css|html))',  # "modify auth.py"
    r'(?:uses?|needs?|requires?)\s+(\w+(?:Service|Controller|Model|Component))',  # "uses AuthService"
    r'(?:import|from)\s+["\']?(\w+)["\']?',  # "import from utils"
    r'(?:in|at)\s+(\w+\.(?:py|js|jsx|ts|tsx))',  # "in schemas.py"
]

# Blocking status indicators
BLOCKING_KEYWORDS = {
    'blocking': ['blocker', 'critical dependency', 'must complete first', 'prerequisite'],
    'soft': ['related to', 'similar to', 'see also', 'reference']
}


def extract_explicit_dependencies(text: str) -> List[int]:
    """
    Extract explicit task ID dependencies from text.
    Patterns like "depends on #123", "blocked by 45", etc.
    
    Returns: List of task IDs
    """
    dependencies = set()
    text_lower = text.lower()
    
    for pattern in EXPLICIT_DEPENDENCY_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            try:
                task_id = int(match)
                dependencies.add(task_id)
            except ValueError:
                continue
    
    return list(dependencies)


def extract_keyword_dependencies(text: str, all_tasks: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract dependencies based on keyword matching with task titles.
    
    Returns: List of potential dependency matches with confidence scores
    """
    text_lower = text.lower()
    potential_deps = []
    
    for pattern in KEYWORD_DEPENDENCY_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            match_clean = match.strip().lower()
            if len(match_clean) < 3:  # Skip very short matches
                continue
            
            # Find tasks that match this reference
            for task in all_tasks:
                task_title_lower = task.get('title', '').lower()
                task_desc_lower = task.get('description', '').lower()
                
                # Calculate match confidence
                confidence = 0
                if match_clean in task_title_lower:
                    confidence = 0.9 if match_clean == task_title_lower else 0.7
                elif match_clean in task_desc_lower:
                    confidence = 0.5
                elif any(word in task_title_lower for word in match_clean.split()):
                    confidence = 0.4
                
                if confidence > 0.3:
                    potential_deps.append({
                        'task_id': task['id'],
                        'task_title': task.get('title', ''),
                        'matched_text': match,
                        'confidence': confidence,
                        'match_type': 'keyword'
                    })
    
    # Remove duplicates, keeping highest confidence
    unique_deps = {}
    for dep in potential_deps:
        task_id = dep['task_id']
        if task_id not in unique_deps or dep['confidence'] > unique_deps[task_id]['confidence']:
            unique_deps[task_id] = dep
    
    return list(unique_deps.values())


def extract_code_references(text: str) -> List[str]:
    """
    Extract code/file references from task text.
    
    Returns: List of referenced files/modules/components
    """
    references = set()
    
    for pattern in CODE_REFERENCE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        references.update(matches)
    
    return list(references)


def find_tasks_by_code_reference(reference: str, all_tasks: List[Dict]) -> List[Dict[str, Any]]:
    """
    Find tasks that also reference the same code/file.
    This suggests a potential dependency relationship.
    """
    ref_lower = reference.lower()
    related_tasks = []
    
    for task in all_tasks:
        task_text = f"{task.get('title', '')} {task.get('description', '')}".lower()
        if ref_lower in task_text:
            related_tasks.append({
                'task_id': task['id'],
                'task_title': task.get('title', ''),
                'shared_reference': reference,
                'match_type': 'code_reference'
            })
    
    return related_tasks


def analyze_task_dependencies(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive dependency analysis for a task.
    Combines multiple detection strategies.
    
    Returns: Analysis result with detected dependencies
    """
    text = f"{task.get('title', '')} {task.get('description', '')}"
    all_tasks = json_storage.get_all_tasks()
    
    # Don't create self-dependencies
    other_tasks = [t for t in all_tasks if t['id'] != task.get('id')]
    
    analysis = {
        'task_id': task.get('id'),
        'explicit_dependencies': [],
        'suggested_dependencies': [],
        'code_references': [],
        'related_tasks': [],
        'blocking_indicators': []
    }
    
    # 1. Extract explicit task ID references
    explicit_ids = extract_explicit_dependencies(text)
    valid_task_ids = {t['id'] for t in all_tasks}
    analysis['explicit_dependencies'] = [
        {
            'task_id': tid,
            'exists': tid in valid_task_ids,
            'task_title': next((t['title'] for t in all_tasks if t['id'] == tid), None)
        }
        for tid in explicit_ids if tid != task.get('id')
    ]
    
    # 2. Extract keyword-based dependencies
    keyword_deps = extract_keyword_dependencies(text, other_tasks)
    analysis['suggested_dependencies'] = sorted(
        keyword_deps, 
        key=lambda x: x['confidence'], 
        reverse=True
    )[:5]  # Top 5 suggestions
    
    # 3. Extract code references
    code_refs = extract_code_references(text)
    analysis['code_references'] = code_refs
    
    # 4. Find tasks with shared code references
    for ref in code_refs:
        related = find_tasks_by_code_reference(ref, other_tasks)
        for r in related:
            if r not in analysis['related_tasks']:
                analysis['related_tasks'].append(r)
    
    # 5. Detect blocking keywords
    text_lower = text.lower()
    for severity, keywords in BLOCKING_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                analysis['blocking_indicators'].append({
                    'keyword': keyword,
                    'severity': severity
                })
    
    return analysis


def suggest_dependencies(task_id: int) -> Dict[str, Any]:
    """
    Get AI-powered dependency suggestions for a task.
    
    Returns: Suggested dependencies with reasoning
    """
    all_tasks = json_storage.get_all_tasks()
    task = next((t for t in all_tasks if t['id'] == task_id), None)
    
    if not task:
        return {'error': 'Task not found', 'suggestions': []}
    
    analysis = analyze_task_dependencies(task)
    
    # Compile all suggestions
    suggestions = []
    
    # Add explicit dependencies (highest priority)
    for dep in analysis['explicit_dependencies']:
        if dep['exists']:
            suggestions.append({
                'task_id': dep['task_id'],
                'task_title': dep['task_title'],
                'confidence': 1.0,
                'reason': 'Explicitly mentioned in task description',
                'auto_add': True
            })
    
    # Add keyword-matched dependencies
    for dep in analysis['suggested_dependencies']:
        suggestions.append({
            'task_id': dep['task_id'],
            'task_title': dep['task_title'],
            'confidence': dep['confidence'],
            'reason': f'Keyword match: "{dep["matched_text"]}"',
            'auto_add': dep['confidence'] >= 0.8
        })
    
    # Add code-reference related tasks
    for rel in analysis['related_tasks'][:3]:  # Limit to 3
        if not any(s['task_id'] == rel['task_id'] for s in suggestions):
            suggestions.append({
                'task_id': rel['task_id'],
                'task_title': rel['task_title'],
                'confidence': 0.5,
                'reason': f'Shares code reference: {rel["shared_reference"]}',
                'auto_add': False
            })
    
    return {
        'task_id': task_id,
        'task_title': task.get('title'),
        'suggestions': suggestions,
        'code_references': analysis['code_references'],
        'blocking_indicators': analysis['blocking_indicators']
    }


def get_dependency_graph(project_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Build a complete dependency graph for visualization.
    
    Returns: Graph structure with nodes and edges
    """
    all_tasks = json_storage.get_all_tasks()
    
    # Filter by project if specified
    if project_id:
        tasks = [t for t in all_tasks if t.get('project_id') == project_id]
    else:
        tasks = all_tasks
    
    # Build nodes
    nodes = []
    for task in tasks:
        nodes.append({
            'id': task['id'],
            'title': task.get('title', ''),
            'status': task.get('status', 'TODO'),
            'priority': task.get('priority', 'MEDIUM'),
            'assignee_id': task.get('assignee_id'),
            'dependencies': task.get('dependencies', [])
        })
    
    # Build edges
    edges = []
    for task in tasks:
        for dep_id in task.get('dependencies', []):
            edges.append({
                'from': dep_id,
                'to': task['id'],
                'type': 'dependency'
            })
    
    # Calculate blocked status
    task_statuses = {t['id']: t.get('status', 'TODO') for t in tasks}
    blocked_tasks = []
    
    for task in tasks:
        deps = task.get('dependencies', [])
        if deps:
            # Task is blocked if any dependency is not DONE
            blocking_deps = [
                dep_id for dep_id in deps 
                if task_statuses.get(dep_id, 'TODO') != 'DONE'
            ]
            if blocking_deps:
                blocked_tasks.append({
                    'task_id': task['id'],
                    'blocked_by': blocking_deps
                })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'blocked_tasks': blocked_tasks,
        'total_tasks': len(tasks),
        'total_dependencies': len(edges)
    }


def validate_dependency(task_id: int, dependency_id: int) -> Dict[str, Any]:
    """
    Validate that adding a dependency won't create a circular reference.
    Uses task index for O(1) lookups instead of repeated next() calls.
    
    Returns: Validation result
    """
    all_tasks = json_storage.get_all_tasks()
    
    # Build task index once - O(n), then O(1) lookups
    task_index = {t['id']: t for t in all_tasks}
    
    # Check tasks exist using O(1) dict lookup
    task = task_index.get(task_id)
    dep_task = task_index.get(dependency_id)
    
    if not task:
        return {'valid': False, 'error': 'Task not found'}
    if not dep_task:
        return {'valid': False, 'error': 'Dependency task not found'}
    if task_id == dependency_id:
        return {'valid': False, 'error': 'Cannot depend on self'}
    
    # Check for circular dependencies using DFS with task index
    def has_path(start_id: int, end_id: int, visited: Set[int]) -> bool:
        if start_id == end_id:
            return True
        if start_id in visited:
            return False
        visited.add(start_id)
        
        # O(1) lookup using index instead of next()
        start_task = task_index.get(start_id)
        if not start_task:
            return False
        
        for dep in start_task.get('dependencies', []):
            if has_path(dep, end_id, visited):
                return True
        return False
    
    # Check if dependency_id eventually depends on task_id (would create cycle)
    if has_path(dependency_id, task_id, set()):
        return {
            'valid': False, 
            'error': 'Would create circular dependency'
        }
    
    return {'valid': True, 'dependency_task': dep_task}


def add_dependency(task_id: int, dependency_id: int) -> Dict[str, Any]:
    """
    Add a dependency to a task after validation.
    Uses set for O(1) membership check.
    
    Returns: Updated task or error
    """
    validation = validate_dependency(task_id, dependency_id)
    if not validation['valid']:
        return {'error': validation['error']}
    
    # Get task using the new json_storage function
    task = json_storage.get_task_by_id(task_id)
    
    current_deps = task.get('dependencies', [])
    # Use set for O(1) membership check
    if dependency_id in set(current_deps):
        return {'error': 'Dependency already exists'}
    
    new_deps = current_deps + [dependency_id]
    updated_task = json_storage.update_task(task_id, {'dependencies': new_deps})
    
    return {
        'success': True,
        'task': updated_task,
        'dependency_added': dependency_id
    }


def remove_dependency(task_id: int, dependency_id: int) -> Dict[str, Any]:
    """
    Remove a dependency from a task.
    Uses O(1) lookup with get_task_by_id.
    
    Returns: Updated task or error
    """
    task = json_storage.get_task_by_id(task_id)
    
    if not task:
        return {'error': 'Task not found'}
    
    current_deps = set(task.get('dependencies', []))
    if dependency_id not in current_deps:
        return {'error': 'Dependency not found'}
    
    # Use set difference for cleaner removal
    new_deps = list(current_deps - {dependency_id})
    updated_task = json_storage.update_task(task_id, {'dependencies': new_deps})
    
    return {
        'success': True,
        'task': updated_task,
        'dependency_removed': dependency_id
    }


def get_blocked_tasks() -> List[Dict[str, Any]]:
    """
    Get all tasks that are currently blocked by dependencies.
    
    Returns: List of blocked tasks with blocking details
    """
    graph = get_dependency_graph()
    return graph['blocked_tasks']


def get_task_dependents(task_id: int) -> List[Dict[str, Any]]:
    """
    Get all tasks that depend on this task.
    Uses list comprehension for cleaner iteration.
    
    Returns: List of dependent tasks
    """
    all_tasks = json_storage.get_all_tasks()
    
    # List comprehension with set membership check for each task
    return [
        {
            'id': task['id'],
            'title': task.get('title', ''),
            'status': task.get('status', 'TODO')
        }
        for task in all_tasks
        if task_id in set(task.get('dependencies', []))
    ]
