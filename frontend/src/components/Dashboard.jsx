import React, { useState, useEffect } from 'react';
import { getTasks, createTask, getUsers, updateTask, getCurrentUser, getComments, addComment, uploadFile, getProjects, createProject, getLabels, createLabel as createLabelApi, getTeams, createTeam as createTeamApi } from '../api';
import KanbanBoard from './KanbanBoard';
import ListView from './ListView';
import CalendarView from './CalendarView';
import TaskDetailView from './TaskDetailView';
import Analytics from './Analytics';
import SprintPlanner from './SprintPlanner';
import CodebaseChat from './CodebaseChat';
import Leaderboard from './Leaderboard';
import XPDisplay from './XPDisplay';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
    LayoutDashboard,
    Table,
    Plus,
    LogOut,
    Search,
    Bell,
    Settings,
    ChevronDown,
    Loader2,
    FileSpreadsheet,
    Layout,
    Download,
    X,
    MessageSquare,
    Paperclip,
    Send,
    File,
    ChevronRight,
    Users,
    Calendar,
    List,
    MoreHorizontal,
    Share2,
    Zap,
    Star,
    Filter,
    Activity,
    BarChart3,
    Sparkles,
    Trophy
} from 'lucide-react';

const Dashboard = ({ onLogout }) => {
    const [tasks, setTasks] = useState([]);
    const [users, setUsers] = useState([]);
    const [currentUser, setCurrentUser] = useState(null);
    const [view, setView] = useState('LIST'); // LIST, BOARD, CALENDAR, ANALYTICS
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [selectedTask, setSelectedTask] = useState(null);
    const [comments, setComments] = useState([]);
    const [newComment, setNewComment] = useState("");
    const [attachments, setAttachments] = useState([]);
    const [uploading, setUploading] = useState(false);
    const [activeTab, setActiveTab] = useState("details"); // details or activity
    const [newTask, setNewTask] = useState({ title: '', description: '', estimated_days: 0, assignee_id: null });
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);

    // Project State
    const [projects, setProjects] = useState([]);
    const [selectedProject, setSelectedProject] = useState(null);
    const [showCreateProjectModal, setShowCreateProjectModal] = useState(false);
    const [showProjectDropdown, setShowProjectDropdown] = useState(false);
    const [newProject, setNewProject] = useState({ name: '', description: '' });
    const [showSprintPlanner, setShowSprintPlanner] = useState(false);

    // Labels and Teams State
    const [labels, setLabels] = useState([]);
    const [teams, setTeams] = useState([]);

    // Filter State
    const [showAssigneeFilter, setShowAssigneeFilter] = useState(false);
    const [showTypeFilter, setShowTypeFilter] = useState(false);
    const [showStatusFilter, setShowStatusFilter] = useState(false);
    const [showMoreFilters, setShowMoreFilters] = useState(false);
    const [filters, setFilters] = useState({
        assignees: [],
        statuses: [],
        priorities: []
    });

    const navigate = useNavigate();

    useEffect(() => {
        fetchCurrentUser();
        fetchUsers();
        fetchProjects();
        fetchLabels();
        fetchTeams();
    }, []);

    useEffect(() => {
        if (selectedProject) {
            fetchTasks(selectedProject.id);
        } else if (projects.length > 0) {
            // Select first project by default if available
            setSelectedProject(projects[0]);
        } else {
            // If no projects, fetch all tasks (or handle empty state)
            fetchTasks();
        }
    }, [selectedProject, projects]);

    const fetchCurrentUser = async () => {
        try {
            const data = await getCurrentUser();
            setCurrentUser(data);
        } catch (err) {
            console.error("Failed to fetch current user", err);
        }
    };

    const fetchUsers = async () => {
        try {
            const data = await getUsers();
            setUsers(data);
        } catch (err) {
            console.error("Failed to fetch users", err);
        }
    };

    const fetchProjects = async () => {
        try {
            const data = await getProjects();
            setProjects(data);
            if (data.length > 0 && !selectedProject) {
                setSelectedProject(data[0]);
            }
        } catch (err) {
            console.error("Failed to fetch projects", err);
        }
    };

    const fetchLabels = async () => {
        try {
            const data = await getLabels();
            setLabels(data);
        } catch (err) {
            console.error("Failed to fetch labels", err);
        }
    };

    const fetchTeams = async () => {
        try {
            const data = await getTeams();
            setTeams(data);
        } catch (err) {
            console.error("Failed to fetch teams", err);
        }
    };

    const handleCreateLabel = async (name, color) => {
        try {
            const newLabel = await createLabelApi(name, color);
            setLabels([...labels, newLabel]);
            return newLabel;
        } catch (err) {
            console.error("Failed to create label", err);
        }
    };

    const handleCreateTeam = async (name) => {
        try {
            const newTeam = await createTeamApi(name, '');
            setTeams([...teams, newTeam]);
            return newTeam;
        } catch (err) {
            console.error("Failed to create team", err);
        }
    };

    const fetchTasks = async (projectId = null) => {
        setLoading(true);
        try {
            const data = await getTasks(projectId);
            setTasks(data);
        } catch (err) {
            console.error("Failed to fetch tasks", err);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateProject = async (e) => {
        e.preventDefault();
        try {
            const project = await createProject(newProject.name, newProject.description);
            setProjects([...projects, project]);
            setSelectedProject(project);
            setShowCreateProjectModal(false);
            setNewProject({ name: '', description: '' });
        } catch (err) {
            console.error("Failed to create project", err);
        }
    };

    const handleCreateTask = async (e) => {
        e.preventDefault();
        setCreating(true);
        try {
            await createTask({ ...newTask, project_id: selectedProject ? selectedProject.id : null });
            setShowCreateModal(false);
            setNewTask({ title: '', description: '', estimated_days: 0, assignee_id: null });
            fetchTasks(selectedProject ? selectedProject.id : null);
        } catch (err) {
            console.error("Failed to create task", err);
        } finally {
            setCreating(false);
        }
    };

    const handleUpdateTask = async (e) => {
        e.preventDefault();
        if (!selectedTask) return;
        setCreating(true);
        try {
            await updateTask(selectedTask.id, {
                title: selectedTask.title,
                description: selectedTask.description,
                estimated_days: selectedTask.estimated_days,
                assignee_id: selectedTask.assignee_id,
                status: selectedTask.status,
                priority: selectedTask.priority
            });
            setSelectedTask(null);
            fetchTasks(selectedProject ? selectedProject.id : null);
        } catch (err) {
            console.error("Failed to update task", err);
        } finally {
            setCreating(false);
        }
    };

    const fetchComments = async (taskId) => {
        try {
            const data = await getComments(taskId);
            setComments(data);
        } catch (err) {
            console.error("Failed to fetch comments", err);
        }
    };

    const handleAddComment = async (e) => {
        e.preventDefault();
        if (!newComment.trim() && attachments.length === 0) return;

        try {
            await addComment(selectedTask.id, newComment, attachments);
            setNewComment("");
            setAttachments([]);
            fetchComments(selectedTask.id);
        } catch (err) {
            console.error("Failed to add comment", err);
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setUploading(true);
        try {
            const data = await uploadFile(file);
            setAttachments([...attachments, data.url]);
        } catch (err) {
            console.error("Failed to upload file", err);
        } finally {
            setUploading(false);
        }
    };

    useEffect(() => {
        if (selectedTask) {
            fetchComments(selectedTask.id);
            setActiveTab("details");
        }
    }, [selectedTask]);

    const handleExportExcel = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch('http://localhost:8000/tasks/export/excel', {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) throw new Error('Export failed');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'tasks.xlsx';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            console.error("Export error:", err);
            alert("Failed to export tasks. Please try again.");
        }
    };

    // Filter tasks based on selected filters
    const filteredTasks = tasks.filter(task => {
        // Filter by assignees
        if (filters.assignees.length > 0) {
            if (!filters.assignees.includes(task.assignee_id)) {
                return false;
            }
        }

        // Filter by statuses
        if (filters.statuses.length > 0) {
            if (!filters.statuses.includes(task.status)) {
                return false;
            }
        }

        // Filter by priorities
        if (filters.priorities.length > 0) {
            if (!filters.priorities.includes(task.priority)) {
                return false;
            }
        }

        return true;
    });

    // Toggle filter selection
    const toggleFilter = (filterType, value) => {
        setFilters(prev => {
            const currentValues = prev[filterType];
            const isSelected = currentValues.includes(value);

            return {
                ...prev,
                [filterType]: isSelected
                    ? currentValues.filter(v => v !== value)
                    : [...currentValues, value]
            };
        });
    };

    // Clear all filters
    const clearAllFilters = () => {
        setFilters({
            assignees: [],
            statuses: [],
            priorities: []
        });
    };

    return (
        <div className="min-h-screen flex overflow-hidden font-sans relative">
            {/* Floating Orbs Background */}
            <div className="floating-orb orb-1"></div>
            <div className="floating-orb orb-2"></div>
            <div className="floating-orb orb-3"></div>

            {/* Sidebar */}
            <aside className="w-64 glass-sidebar flex flex-col z-20">
                <div className="p-4 flex items-center space-x-2 border-b border-white/10 mb-2">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white font-bold text-xs shadow-lg glow-primary">
                        TF
                    </div>
                    <span className="font-bold text-sm">TaskFlow</span>
                </div>

                <div className="flex-1 overflow-y-auto py-2">
                    <div className="px-4 py-1">
                        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 px-2">Menu</div>
                        <button
                            onClick={() => setView('BOARD')}
                            className={`w-full flex items-center space-x-2 px-3 py-2.5 rounded-lg cursor-pointer font-medium text-sm mb-1 transition-all duration-200 ${view === 'BOARD' ? 'bg-indigo-500/20 text-indigo-300 shadow-lg shadow-indigo-500/10' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'}`}
                        >
                            <Layout className="w-4 h-4" />
                            <span>Board</span>
                        </button>
                        <button
                            onClick={() => setView('LIST')}
                            className={`w-full flex items-center space-x-2 px-3 py-2.5 rounded-lg cursor-pointer font-medium text-sm mb-1 transition-all duration-200 ${view === 'LIST' ? 'bg-indigo-500/20 text-indigo-300 shadow-lg shadow-indigo-500/10' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'}`}
                        >
                            <List className="w-4 h-4" />
                            <span>List</span>
                        </button>
                        <button
                            onClick={() => setView('CALENDAR')}
                            className={`w-full flex items-center space-x-2 px-3 py-2.5 rounded-lg cursor-pointer font-medium text-sm mb-1 transition-all duration-200 ${view === 'CALENDAR' ? 'bg-indigo-500/20 text-indigo-300 shadow-lg shadow-indigo-500/10' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'}`}
                        >
                            <Calendar className="w-4 h-4" />
                            <span>Calendar</span>
                        </button>
                        {currentUser && (currentUser.role === 'PM' || currentUser.role === 'PO') && (
                            <button
                                onClick={() => setView('ANALYTICS')}
                                className={`w-full flex items-center space-x-2 px-3 py-2.5 rounded-lg cursor-pointer font-medium text-sm mb-1 transition-all duration-200 ${view === 'ANALYTICS' ? 'bg-indigo-500/20 text-indigo-300 shadow-lg shadow-indigo-500/10' : 'text-slate-400 hover:bg-white/5 hover:text-slate-200'}`}
                            >
                                <BarChart3 className="w-4 h-4" />
                                <span>Analytics</span>
                            </button>
                        )}
                        {currentUser && (currentUser.role === 'PM' || currentUser.role === 'PO') && (
                            <button
                                onClick={() => setShowSprintPlanner(true)}
                                className="w-full flex items-center space-x-2 px-3 py-2.5 rounded-lg cursor-pointer font-medium text-sm mb-1 transition-all duration-200 text-purple-400 hover:bg-purple-500/10 hover:text-purple-300 border border-purple-500/20"
                            >
                                <Sparkles className="w-4 h-4" />
                                <span>AI Sprint Planner</span>
                            </button>
                        )}
                    </div>
                </div>

                {/* Gamification Section */}
                <div className="px-4 py-2 border-t border-white/10">
                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 px-2 flex items-center gap-1">
                        <Trophy className="w-3 h-3" />
                        Sprint Champions
                    </div>
                    <Leaderboard compact={true} />
                </div>

                <div className="p-4 border-t border-white/10">
                    {/* XP Display */}
                    {currentUser && (
                        <div className="mb-3 relative">
                            <XPDisplay compact={true} />
                        </div>
                    )}

                    {currentUser && (
                        <div className="flex items-center space-x-3 mb-4 px-2">
                            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500/30 to-purple-500/30 flex items-center justify-center font-bold border border-indigo-400/30 text-xs text-indigo-300">
                                {currentUser.username.substring(0, 2).toUpperCase()}
                            </div>
                            <div className="overflow-hidden">
                                <p className="text-sm font-medium truncate">{currentUser.username}</p>
                                <p className="text-xs text-slate-400 truncate">{currentUser.role}</p>
                            </div>
                        </div>
                    )}
                    <button
                        onClick={onLogout}
                        className="w-full flex items-center space-x-3 px-2 py-2 rounded-lg text-slate-400 hover:bg-red-500/10 hover:text-red-400 transition-all duration-200 font-medium text-sm"
                    >
                        <LogOut className="w-4 h-4" />
                        <span>Sign Out</span>
                    </button>
                </div>
            </aside>

            <main className="flex-1 flex flex-col min-w-0 relative z-10">

                {/* Header */}
                <div className="px-6 pt-4 pb-0">
                    <div className="text-xs text-muted mb-2">Projects / TaskFlow</div>
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                            <div className="w-8 h-8 bg-primary rounded flex items-center justify-center text-white font-bold">
                                {selectedProject ? selectedProject.name.substring(0, 2).toUpperCase() : 'TF'}
                            </div>
                            <div className="relative">
                                <h1
                                    className="text-lg font-semibold text-foreground flex items-center gap-2 cursor-pointer hover:text-primary transition-colors select-none"
                                    onClick={() => setShowProjectDropdown(!showProjectDropdown)}
                                >
                                    {selectedProject ? selectedProject.name : 'All Tasks'}
                                    <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${showProjectDropdown ? 'rotate-180' : ''}`} />
                                </h1>
                                {/* Project Dropdown */}
                                {showProjectDropdown && (
                                    <>
                                        <div
                                            className="fixed inset-0 z-40"
                                            onClick={() => setShowProjectDropdown(false)}
                                        ></div>
                                        <div className="absolute top-full left-0 mt-1 w-64 bg-surface border border-border rounded shadow-lg z-50 animate-in fade-in zoom-in-95 duration-100">
                                            <div className="max-h-64 overflow-y-auto">
                                                {projects.map(p => (
                                                    <div
                                                        key={p.id}
                                                        className={`px-4 py-2 hover:bg-gray-100 cursor-pointer text-sm flex items-center justify-between ${selectedProject && selectedProject.id === p.id ? 'bg-primary/5 text-primary font-medium' : ''}`}
                                                        onClick={() => {
                                                            setSelectedProject(p);
                                                            setShowProjectDropdown(false);
                                                        }}
                                                    >
                                                        <span>{p.name}</span>
                                                        {selectedProject && selectedProject.id === p.id && <div className="w-2 h-2 rounded-full bg-primary"></div>}
                                                    </div>
                                                ))}
                                            </div>
                                            <div className="border-t border-border mt-1 pt-1">
                                                <button
                                                    className="w-full text-left px-4 py-2 text-primary text-sm hover:bg-gray-100 flex items-center"
                                                    onClick={() => {
                                                        setShowCreateProjectModal(true);
                                                        setShowProjectDropdown(false);
                                                    }}
                                                >
                                                    <Plus className="w-3 h-3 mr-2" /> New Project
                                                </button>
                                            </div>
                                        </div>
                                    </>
                                )}
                                {selectedProject && <p className="text-xs text-muted font-normal">{selectedProject.description}</p>}
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">

                            <button
                                onClick={() => setShowCreateModal(true)}
                                className="bg-primary hover:bg-primary-hover text-white px-3 py-1.5 rounded font-medium text-sm flex items-center space-x-1 mr-2 shadow-sm"
                            >
                                <Plus className="w-4 h-4" />
                                <span>Create Task</span>
                            </button>
                            <button onClick={() => alert("Share feature coming soon!")} className="p-2 hover:bg-surface rounded text-muted"><Share2 className="w-4 h-4" /></button>
                            <button onClick={() => alert("Settings feature coming soon!")} className="p-2 hover:bg-surface rounded text-muted"><Settings className="w-4 h-4" /></button>
                        </div>
                    </div>

                    {/* Tabs */}
                    <div className="flex items-center space-x-6 border-b border-border">
                        {['List', 'Board', 'Calendar'].map((tab) => (
                            <button
                                key={tab}
                                onClick={() => setView(tab.toUpperCase())}
                                className={`pb-2 text-sm font-medium border-b-2 transition-colors ${view === tab.toUpperCase() ? 'border-primary text-primary' : 'border-transparent text-muted hover:text-foreground hover:border-gray-300'}`}
                            >
                                {tab === 'List' && <List className="w-4 h-4 inline-block mr-1.5 mb-0.5" />}
                                {tab === 'Board' && <Layout className="w-4 h-4 inline-block mr-1.5 mb-0.5" />}
                                {tab === 'Calendar' && <Calendar className="w-4 h-4 inline-block mr-1.5 mb-0.5" />}
                                {tab}
                            </button>
                        ))}
                        {currentUser && (currentUser.role === 'PM' || currentUser.role === 'PO') && (
                            <button
                                onClick={() => setView('ANALYTICS')}
                                className={`pb-2 text-sm font-medium border-b-2 transition-colors ${view === 'ANALYTICS' ? 'border-primary text-primary' : 'border-transparent text-muted hover:text-foreground hover:border-gray-300'}`}
                            >
                                <BarChart3 className="w-4 h-4 inline-block mr-1.5 mb-0.5" />
                                Analytics
                            </button>
                        )}
                    </div>

                    {/* Filter Bar */}
                    <div className="py-4 flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                            <div className="flex bg-surface rounded p-0.5 border border-border">
                                <button className="px-3 py-1 bg-white shadow-sm rounded-sm text-xs font-medium text-primary">Basic</button>
                                <button className="px-3 py-1 text-xs font-medium text-muted hover:text-foreground">JQL</button>
                            </div>
                            <div className="relative">
                                <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted" />
                                <input type="text" placeholder="Search work" className="pl-7 pr-3 py-1.5 text-sm border border-border rounded hover:bg-surface focus:bg-white focus:ring-2 focus:ring-primary outline-none w-40 transition-all" />
                            </div>
                            <div className="h-6 w-px bg-border mx-2"></div>

                            {/* Assignee Filter */}
                            <div className="relative">
                                <button
                                    onClick={() => {
                                        setShowAssigneeFilter(!showAssigneeFilter);
                                        setShowTypeFilter(false);
                                        setShowStatusFilter(false);
                                        setShowMoreFilters(false);
                                    }}
                                    className={`flex items-center space-x-1 px-3 py-1.5 hover:bg-surface rounded text-sm font-medium ${filters.assignees.length > 0 ? 'bg-primary/10 text-primary' : 'text-foreground'}`}
                                >
                                    <span>Assignee</span>
                                    {filters.assignees.length > 0 && (
                                        <span className="ml-1 px-1.5 py-0.5 bg-primary text-white rounded-full text-xs">{filters.assignees.length}</span>
                                    )}
                                    <ChevronDown className="w-3 h-3 text-muted" />
                                </button>
                                {showAssigneeFilter && (
                                    <>
                                        <div className="fixed inset-0 z-40" onClick={() => setShowAssigneeFilter(false)}></div>
                                        <div className="absolute top-full left-0 mt-1 w-56 bg-surface border border-border rounded shadow-lg z-50 py-2 max-h-64 overflow-y-auto">
                                            <div className="px-3 py-2 text-xs font-semibold text-muted uppercase border-b border-border">Select Assignees</div>
                                            {users.map(user => (
                                                <label key={user.id} className="flex items-center px-3 py-2 hover:bg-gray-100 cursor-pointer">
                                                    <input
                                                        type="checkbox"
                                                        checked={filters.assignees.includes(user.id)}
                                                        onChange={() => toggleFilter('assignees', user.id)}
                                                        className="w-4 h-4 text-primary rounded border-border focus:ring-primary"
                                                    />
                                                    <div className="ml-2 flex items-center space-x-2">
                                                        <div className="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-[10px] font-bold text-gray-600">
                                                            {user.username.substring(0, 2).toUpperCase()}
                                                        </div>
                                                        <span className="text-sm">{user.username}</span>
                                                    </div>
                                                </label>
                                            ))}
                                            <div className="px-3 py-2 border-t border-border mt-1">
                                                <button
                                                    onClick={() => {
                                                        setFilters(prev => ({ ...prev, assignees: [] }));
                                                        setShowAssigneeFilter(false);
                                                    }}
                                                    className="text-xs text-primary hover:underline"
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* Type Filter - Placeholder for now */}
                            <button className="flex items-center space-x-1 px-3 py-1.5 hover:bg-surface rounded text-sm font-medium text-foreground">
                                <span>Type</span>
                                <ChevronDown className="w-3 h-3 text-muted" />
                            </button>

                            {/* Status Filter */}
                            <div className="relative">
                                <button
                                    onClick={() => {
                                        setShowStatusFilter(!showStatusFilter);
                                        setShowAssigneeFilter(false);
                                        setShowTypeFilter(false);
                                        setShowMoreFilters(false);
                                    }}
                                    className={`flex items-center space-x-1 px-3 py-1.5 hover:bg-surface rounded text-sm font-medium ${filters.statuses.length > 0 ? 'bg-primary/10 text-primary' : 'text-foreground'}`}
                                >
                                    <span>Status</span>
                                    {filters.statuses.length > 0 && (
                                        <span className="ml-1 px-1.5 py-0.5 bg-primary text-white rounded-full text-xs">{filters.statuses.length}</span>
                                    )}
                                    <ChevronDown className="w-3 h-3 text-muted" />
                                </button>
                                {showStatusFilter && (
                                    <>
                                        <div className="fixed inset-0 z-40" onClick={() => setShowStatusFilter(false)}></div>
                                        <div className="absolute top-full left-0 mt-1 w-56 bg-surface border border-border rounded shadow-lg z-50 py-2">
                                            <div className="px-3 py-2 text-xs font-semibold text-muted uppercase border-b border-border">Select Statuses</div>
                                            {['TODO', 'IN_PROGRESS', 'IN_REVIEW', 'DONE'].map(status => (
                                                <label key={status} className="flex items-center px-3 py-2 hover:bg-gray-100 cursor-pointer">
                                                    <input
                                                        type="checkbox"
                                                        checked={filters.statuses.includes(status)}
                                                        onChange={() => toggleFilter('statuses', status)}
                                                        className="w-4 h-4 text-primary rounded border-border focus:ring-primary"
                                                    />
                                                    <span className="ml-2 text-sm">{status.replace('_', ' ')}</span>
                                                </label>
                                            ))}
                                            <div className="px-3 py-2 border-t border-border mt-1">
                                                <button
                                                    onClick={() => {
                                                        setFilters(prev => ({ ...prev, statuses: [] }));
                                                        setShowStatusFilter(false);
                                                    }}
                                                    className="text-xs text-primary hover:underline"
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>

                            {/* More Filters (Priority) */}
                            <div className="relative">
                                <button
                                    onClick={() => {
                                        setShowMoreFilters(!showMoreFilters);
                                        setShowAssigneeFilter(false);
                                        setShowTypeFilter(false);
                                        setShowStatusFilter(false);
                                    }}
                                    className={`flex items-center space-x-1 px-3 py-1.5 hover:bg-surface rounded text-sm font-medium ${filters.priorities.length > 0 ? 'bg-primary/10 text-primary' : 'text-foreground'}`}
                                >
                                    <span>More filters</span>
                                    {filters.priorities.length > 0 && (
                                        <span className="ml-1 px-1.5 py-0.5 bg-primary text-white rounded-full text-xs">{filters.priorities.length}</span>
                                    )}
                                    <ChevronDown className="w-3 h-3 text-muted" />
                                </button>
                                {showMoreFilters && (
                                    <>
                                        <div className="fixed inset-0 z-40" onClick={() => setShowMoreFilters(false)}></div>
                                        <div className="absolute top-full left-0 mt-1 w-56 bg-surface border border-border rounded shadow-lg z-50 py-2">
                                            <div className="px-3 py-2 text-xs font-semibold text-muted uppercase border-b border-border">Select Priorities</div>
                                            {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(priority => (
                                                <label key={priority} className="flex items-center px-3 py-2 hover:bg-gray-100 cursor-pointer">
                                                    <input
                                                        type="checkbox"
                                                        checked={filters.priorities.includes(priority)}
                                                        onChange={() => toggleFilter('priorities', priority)}
                                                        className="w-4 h-4 text-primary rounded border-border focus:ring-primary"
                                                    />
                                                    <span className="ml-2 text-sm">{priority}</span>
                                                </label>
                                            ))}
                                            <div className="px-3 py-2 border-t border-border mt-1">
                                                <button
                                                    onClick={() => {
                                                        setFilters(prev => ({ ...prev, priorities: [] }));
                                                        setShowMoreFilters(false);
                                                    }}
                                                    className="text-xs text-primary hover:underline"
                                                >
                                                    Clear
                                                </button>
                                            </div>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>
                        <div className="flex items-center space-x-2">
                            {(filters.assignees.length > 0 || filters.statuses.length > 0 || filters.priorities.length > 0) && (
                                <button
                                    onClick={clearAllFilters}
                                    className="text-sm font-medium text-primary hover:underline px-2"
                                >
                                    Clear all filters
                                </button>
                            )}
                            <button className="text-sm font-medium text-muted hover:text-foreground px-3 py-1.5 hover:bg-surface rounded">Saved filters</button>
                            <button className="text-sm font-medium text-muted hover:text-foreground px-3 py-1.5 hover:bg-surface rounded flex items-center space-x-1">
                                <span>Group</span>
                                <ChevronDown className="w-3 h-3" />
                            </button>
                            <button onClick={handleExportExcel} className="p-1.5 hover:bg-surface rounded text-muted"><Download className="w-4 h-4" /></button>
                            <button className="p-1.5 hover:bg-surface rounded text-muted"><Share2 className="w-4 h-4" /></button>
                        </div>
                    </div>
                </div>

                <div className="flex-1 overflow-hidden p-6 relative z-0">
                    {loading ? (
                        <div className="flex items-center justify-center h-full">
                            <Loader2 className="w-8 h-8 animate-spin text-primary" />
                        </div>
                    ) : (
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4 }}
                            className="h-full"
                        >
                            {view === 'BOARD' && (
                                <KanbanBoard tasks={filteredTasks} onTaskUpdate={fetchTasks} onTaskClick={setSelectedTask} />
                            )}
                            {view === 'LIST' && (
                                <div className="flex flex-col h-full">
                                    <ListView tasks={filteredTasks} users={users} onTaskClick={setSelectedTask} />
                                    <button
                                        onClick={() => setShowCreateModal(true)}
                                        className="mt-4 flex items-center space-x-2 text-muted hover:text-foreground px-2 py-1 hover:bg-surface rounded w-max"
                                    >
                                        <Plus className="w-4 h-4" />
                                        <span className="text-sm font-medium">Create</span>
                                    </button>
                                </div>
                            )}
                            {view === 'CALENDAR' && (
                                <CalendarView tasks={filteredTasks} onTaskClick={setSelectedTask} />
                            )}
                            {view === 'ANALYTICS' && (
                                <Analytics />
                            )}
                            {(view !== 'BOARD' && view !== 'LIST' && view !== 'CALENDAR' && view !== 'ANALYTICS') && (
                                <div className="flex flex-col items-center justify-center h-full text-muted">
                                    <Layout className="w-12 h-12 mb-4 opacity-20" />
                                    <p>This view is under construction.</p>
                                </div>
                            )}
                        </motion.div>
                    )}
                </div>
            </main>

            {/* Create Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <motion.div
                        initial={{ scale: 0.95, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="card p-6 w-full max-w-lg bg-surface"
                    >
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold mb-4 text-foreground">Create New Task</h2>
                            <button onClick={() => setShowCreateModal(false)} className="text-muted hover:text-foreground">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <form onSubmit={handleCreateTask} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium mb-1 text-muted">Title</label>
                                <input
                                    type="text"
                                    value={newTask.title}
                                    onChange={(e) => setNewTask({ ...newTask, title: e.target.value })}
                                    className="input-field"
                                    placeholder="Task title"
                                    required
                                />
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-medium mb-1 text-muted">Estimated Days</label>
                                    <input
                                        type="number"
                                        min="0"
                                        step="0.5"
                                        value={newTask.estimated_days}
                                        onChange={(e) => setNewTask({ ...newTask, estimated_days: parseFloat(e.target.value) })}
                                        className="input-field"
                                        placeholder="e.g. 3.5"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium mb-1 text-muted">Assignee</label>
                                    <select
                                        value={newTask.assignee_id || ""}
                                        onChange={(e) => setNewTask({ ...newTask, assignee_id: e.target.value ? parseInt(e.target.value) : null })}
                                        className="input-field"
                                    >
                                        <option value="">Unassigned</option>
                                        {users.map(user => (
                                            <option key={user.id} value={user.id}>{user.username} ({user.role})</option>
                                        ))}
                                    </select>
                                </div>
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-1 text-muted">Description</label>
                                <textarea
                                    value={newTask.description}
                                    onChange={(e) => setNewTask({ ...newTask, description: e.target.value })}
                                    className="input-field h-32 resize-none"
                                    placeholder="Describe the task... AI will detect priority!"
                                    required
                                />
                            </div>

                            <div className="flex justify-end space-x-3 mt-6">
                                <button
                                    type="button"
                                    onClick={() => setShowCreateModal(false)}
                                    className="btn-secondary"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={creating}
                                    className="btn-primary"
                                >
                                    {creating && <Loader2 className="w-4 h-4 animate-spin" />}
                                    <span>Create Task</span>
                                </button>
                            </div>
                        </form>
                    </motion.div>
                </div>
            )}

            {/* Task Details Modal */}
            {selectedTask && (
                <TaskDetailView
                    task={selectedTask}
                    onClose={() => setSelectedTask(null)}
                    onUpdate={(updatedTask) => {
                        // Optimistic update
                        setSelectedTask(updatedTask);
                        // Call API to update (include new fields)
                        updateTask(updatedTask.id, {
                            title: updatedTask.title,
                            description: updatedTask.description,
                            estimated_days: updatedTask.estimated_days,
                            assignee_id: updatedTask.assignee_id,
                            status: updatedTask.status,
                            priority: updatedTask.priority,
                            parent_id: updatedTask.parent_id,
                            labels: updatedTask.labels,
                            team_id: updatedTask.team_id
                        }).then(() => fetchTasks(selectedProject ? selectedProject.id : null));
                    }}
                    users={users}
                    comments={comments}
                    onAddComment={async (content) => {
                        try {
                            await addComment(selectedTask.id, content, []);
                            fetchComments(selectedTask.id);
                        } catch (err) {
                            console.error("Failed to add comment", err);
                        }
                    }}
                    currentUser={currentUser}
                    tasks={tasks}
                    labels={labels}
                    teams={teams}
                    onCreateLabel={handleCreateLabel}
                    onCreateTeam={handleCreateTeam}
                />
            )}
            {/* Create Project Modal */}
            {showCreateProjectModal && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <motion.div
                        initial={{ scale: 0.95, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        className="card p-6 w-full max-w-md bg-surface"
                    >
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold mb-4 text-foreground">Create New Project</h2>
                            <button onClick={() => setShowCreateProjectModal(false)} className="text-muted hover:text-foreground">
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        <form onSubmit={handleCreateProject} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium mb-1 text-muted">Project Name</label>
                                <input
                                    type="text"
                                    value={newProject.name}
                                    onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                                    className="input-field"
                                    placeholder="Project Name"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1 text-muted">Description</label>
                                <textarea
                                    value={newProject.description}
                                    onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                                    className="input-field h-24 resize-none"
                                    placeholder="Project Description"
                                />
                            </div>
                            <div className="flex justify-end space-x-3 mt-6">
                                <button
                                    type="button"
                                    onClick={() => setShowCreateProjectModal(false)}
                                    className="btn-secondary"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="btn-primary"
                                >
                                    Create Project
                                </button>
                            </div>
                        </form>
                    </motion.div>
                </div>
            )}

            {/* Sprint Planner Modal */}
            {showSprintPlanner && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                    <SprintPlanner onClose={() => setShowSprintPlanner(false)} />
                </div>
            )}

            {/* Codebase Chat (RAG) */}
            <CodebaseChat />
        </div>
    );
};

export default Dashboard;
