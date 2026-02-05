import axios from 'axios';

// Use environment variable for production, fallback to localhost for local dev
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
});

api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

export const login = async (username, password) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    const response = await api.post('/token', formData);
    return response.data;
};

export const register = async (username, email, password, role) => {
    const response = await api.post('/users/', { username, email, password, role });
    return response.data;
};

export const verifyEmail = async (token) => {
    const response = await api.get(`/verify/${token}`);
    return response.data;
};

export const resendVerification = async (email) => {
    const response = await api.post(`/resend-verification?email=${encodeURIComponent(email)}`);
    return response.data;
};

export const getCurrentUser = async () => {
    const response = await api.get('/users/me/');
    return response.data;
};

export const getUsers = async () => {
    const response = await api.get('/users/');
    return response.data;
};

export const getProjects = async () => {
    const response = await api.get('/projects/');
    return response.data;
};

export const createProject = async (name, description) => {
    const response = await api.post('/projects/', { name, description });
    return response.data;
};

export const getTasks = async (projectId = null) => {
    const url = projectId ? `/tasks/?project_id=${projectId}` : '/tasks/';
    const response = await api.get(url);
    return response.data;
};

export const createTask = async (task) => {
    const response = await api.post('/tasks/', task);
    return response.data;
};

export const updateTask = async (taskId, updates) => {
    const response = await api.put(`/tasks/${taskId}`, updates);
    return response.data;
};

export const getComments = async (taskId) => {
    const response = await api.get(`/tasks/${taskId}/comments`);
    return response.data;
};

export const addComment = async (taskId, content, attachments = []) => {
    const response = await api.post(`/tasks/${taskId}/comments`, { content, attachments });
    return response.data;
};

export const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/upload/', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });
    return response.data;
};

// Analytics API (Manager-only)
export const getKPIs = async () => {
    const response = await api.get('/analytics/kpis');
    return response.data;
};

export const getVelocity = async (days = 7) => {
    const response = await api.get(`/analytics/velocity?days=${days}`);
    return response.data;
};

export const getTeamPerformance = async () => {
    const response = await api.get('/analytics/team-performance');
    return response.data;
};

export const getRisks = async () => {
    const response = await api.get('/analytics/risks');
    return response.data;
};

export const getAIInsights = async () => {
    const response = await api.get('/analytics/insights');
    return response.data;
};

export const getTeamHealth = async () => {
    const response = await api.get('/analytics/team-health');
    return response.data;
};

// AI Assistant API
export const getCodeSuggestions = async (taskId) => {
    const response = await api.get(`/ai/code-suggestions/${taskId}`);
    return response.data;
};

export const getTaskAnalysis = async (taskId) => {
    const response = await api.get(`/ai/task-analysis/${taskId}`);
    return response.data;
};

export const getSubtaskSuggestions = async (taskId) => {
    const response = await api.get(`/ai/subtask-suggestions/${taskId}`);
    return response.data;
};

// Sprint Planning API (Manager-only)
export const getSprintPlan = async (sprintDays = 14) => {
    const response = await api.get(`/ai/sprint-plan?sprint_days=${sprintDays}`);
    return response.data;
};

export const getSprintAssignments = async (sprintDays = 14) => {
    const response = await api.get(`/ai/sprint-assignments?sprint_days=${sprintDays}`);
    return response.data;
};

export const applySprintAssignments = async (assignments) => {
    const response = await api.post('/ai/apply-assignments', assignments);
    return response.data;
};

// Dependency Management API
export const getTaskDependencies = async (taskId) => {
    const response = await api.get(`/tasks/${taskId}/dependencies`);
    return response.data;
};

export const addTaskDependency = async (taskId, dependencyId) => {
    const response = await api.post(`/tasks/${taskId}/dependencies/${dependencyId}`);
    return response.data;
};

export const removeTaskDependency = async (taskId, dependencyId) => {
    const response = await api.delete(`/tasks/${taskId}/dependencies/${dependencyId}`);
    return response.data;
};

export const getSuggestedDependencies = async (taskId) => {
    const response = await api.get(`/tasks/${taskId}/suggested-dependencies`);
    return response.data;
};

export const getDependencyGraph = async (projectId) => {
    const response = await api.get(`/projects/${projectId}/dependency-graph`);
    return response.data;
};

export const getBlockedTasks = async () => {
    const response = await api.get('/dependencies/blocked-tasks');
    return response.data;
};

// Collaboration API
export const getActiveUsers = async () => {
    const response = await api.get('/collaboration/active-users');
    return response.data;
};

export const getTaskViewers = async (taskId) => {
    const response = await api.get(`/collaboration/task/${taskId}/viewers`);
    return response.data;
};

export const getCollaborationStats = async () => {
    const response = await api.get('/collaboration/stats');
    return response.data;
};

// Gamification API
export const getGamificationStats = async () => {
    const response = await api.get('/gamification/me');
    return response.data;
};

export const getUserGamificationStats = async (userId) => {
    const response = await api.get(`/gamification/user/${userId}`);
    return response.data;
};

export const getLeaderboard = async (limit = 10) => {
    const response = await api.get(`/gamification/leaderboard?limit=${limit}`);
    return response.data;
};

export const getLevelColors = async () => {
    const response = await api.get('/gamification/level-colors');
    return response.data;
};

// Codebase Chat (RAG) API
export const indexCodebase = async () => {
    const response = await api.post('/rag/index');
    return response.data;
};

export const getRagStatus = async () => {
    const response = await api.get('/rag/status');
    return response.data;
};

export const chatWithCodebase = async (question) => {
    const response = await api.post('/rag/chat', { question });
    return response.data;
};

// Labels API
export const getLabels = async () => {
    const response = await api.get('/labels/');
    return response.data;
};

export const createLabel = async (name, color) => {
    const response = await api.post('/labels/', { name, color });
    return response.data;
};

export const deleteLabel = async (labelId) => {
    const response = await api.delete(`/labels/${labelId}`);
    return response.data;
};

// Teams API
export const getTeams = async () => {
    const response = await api.get('/teams/');
    return response.data;
};

export const createTeam = async (name, description = '') => {
    const response = await api.post('/teams/', { name, description });
    return response.data;
};

export const deleteTeam = async (teamId) => {
    const response = await api.delete(`/teams/${teamId}`);
    return response.data;
};

export default api;
