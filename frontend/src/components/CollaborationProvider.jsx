import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';

const CollaborationContext = createContext(null);

export const useCollaboration = () => {
    const context = useContext(CollaborationContext);
    if (!context) {
        throw new Error('useCollaboration must be used within a CollaborationProvider');
    }
    return context;
};

export const CollaborationProvider = ({ children }) => {
    const [isConnected, setIsConnected] = useState(false);
    const [activeUsers, setActiveUsers] = useState([]);
    const [taskViewers, setTaskViewers] = useState({});
    const [cursors, setCursors] = useState({});
    const [typingUsers, setTypingUsers] = useState({});
    const [currentTaskId, setCurrentTaskId] = useState(null);
    const [lockStatus, setLockStatus] = useState(null);

    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const clientIdRef = useRef(`client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

    // Generate unique colors for users
    const userColors = useRef({});
    const getColorForUser = useCallback((userId) => {
        if (!userColors.current[userId]) {
            const colors = [
                '#ef4444', '#f97316', '#eab308', '#22c55e', '#14b8a6',
                '#3b82f6', '#6366f1', '#8b5cf6', '#ec4899', '#f43f5e'
            ];
            userColors.current[userId] = colors[Object.keys(userColors.current).length % colors.length];
        }
        return userColors.current[userId];
    }, []);

    const connect = useCallback(() => {
        const token = localStorage.getItem('token');
        if (!token) return;

        // Use ngrok URL if available, otherwise localhost
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const wsProtocol = apiUrl.startsWith('https') ? 'wss' : 'ws';
        const wsHost = apiUrl.replace(/^https?:\/\//, '');
        const wsUrl = `${wsProtocol}://${wsHost}/ws/${clientIdRef.current}?token=${token}`;

        try {
            wsRef.current = new WebSocket(wsUrl);

            wsRef.current.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
            };

            wsRef.current.onclose = (event) => {
                console.log('WebSocket disconnected', event.code);
                setIsConnected(false);

                // Attempt to reconnect after 3 seconds
                if (event.code !== 4001) { // Don't reconnect on auth failure
                    reconnectTimeoutRef.current = setTimeout(() => {
                        connect();
                    }, 3000);
                }
            };

            wsRef.current.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            wsRef.current.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    handleMessage(message);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
        }
    }, []);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    const sendMessage = useCallback((type, data) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type, data }));
        }
    }, []);

    const handleMessage = useCallback((message) => {
        const { type, data } = message;

        switch (type) {
            case 'user_online':
                setActiveUsers(prev => {
                    if (!prev.find(u => u.user_id === data.user_id)) {
                        return [...prev, { user_id: data.user_id, username: data.username }];
                    }
                    return prev;
                });
                break;

            case 'user_offline':
                setActiveUsers(prev => prev.filter(u => u.user_id !== data.user_id));
                setCursors(prev => {
                    const updated = { ...prev };
                    Object.keys(updated).forEach(taskId => {
                        delete updated[taskId][data.user_id];
                    });
                    return updated;
                });
                break;

            case 'user_joined':
                setTaskViewers(prev => ({
                    ...prev,
                    [data.task_id]: [...(prev[data.task_id] || []), {
                        user_id: data.user_id,
                        username: data.username
                    }]
                }));
                break;

            case 'user_left':
                setTaskViewers(prev => ({
                    ...prev,
                    [data.task_id]: (prev[data.task_id] || []).filter(u => u.user_id !== data.user_id)
                }));
                break;

            case 'current_viewers':
                setTaskViewers(prev => ({
                    ...prev,
                    [data.task_id]: data.viewers
                }));
                break;

            case 'cursor_update':
                setCursors(prev => ({
                    ...prev,
                    [currentTaskId]: {
                        ...(prev[currentTaskId] || {}),
                        [data.user_id]: {
                            ...data.cursor,
                            username: data.username,
                            color: getColorForUser(data.user_id)
                        }
                    }
                }));
                break;

            case 'typing_indicator':
                if (data.is_typing) {
                    setTypingUsers(prev => ({
                        ...prev,
                        [currentTaskId]: {
                            ...(prev[currentTaskId] || {}),
                            [data.user_id]: data.username
                        }
                    }));
                } else {
                    setTypingUsers(prev => {
                        const updated = { ...prev };
                        if (updated[currentTaskId]) {
                            delete updated[currentTaskId][data.user_id];
                        }
                        return updated;
                    });
                }
                break;

            case 'task_field_update':
                // Dispatch custom event for task components to listen
                window.dispatchEvent(new CustomEvent('realtime_task_update', { detail: data }));
                break;

            case 'lock_result':
                setLockStatus(data);
                break;

            case 'task_locked':
                setLockStatus({
                    locked: true,
                    user_id: data.locked_by,
                    username: data.username
                });
                break;

            case 'task_unlocked':
                setLockStatus(null);
                break;

            case 'pong':
                // Keep-alive response
                break;

            default:
                console.log('Unknown message type:', type);
        }
    }, [currentTaskId, getColorForUser]);

    // Connect on mount
    useEffect(() => {
        connect();
        return () => disconnect();
    }, [connect, disconnect]);

    // Keep-alive ping every 30 seconds
    useEffect(() => {
        const pingInterval = setInterval(() => {
            sendMessage('ping', {});
        }, 30000);

        return () => clearInterval(pingInterval);
    }, [sendMessage]);

    // Public API
    const joinTask = useCallback((taskId) => {
        setCurrentTaskId(taskId);
        sendMessage('join_task', { task_id: taskId });
    }, [sendMessage]);

    const leaveTask = useCallback((taskId) => {
        sendMessage('leave_task', { task_id: taskId });
        setCurrentTaskId(null);
        setCursors(prev => {
            const updated = { ...prev };
            delete updated[taskId];
            return updated;
        });
    }, [sendMessage]);

    const moveCursor = useCallback((x, y, elementId = null) => {
        if (!currentTaskId) return;
        sendMessage('cursor_move', {
            task_id: currentTaskId,
            cursor: { x, y, element_id: elementId }
        });
    }, [currentTaskId, sendMessage]);

    const startTyping = useCallback(() => {
        if (!currentTaskId) return;
        sendMessage('typing_start', { task_id: currentTaskId });
    }, [currentTaskId, sendMessage]);

    const stopTyping = useCallback(() => {
        if (!currentTaskId) return;
        sendMessage('typing_stop', { task_id: currentTaskId });
    }, [currentTaskId, sendMessage]);

    const sendTaskUpdate = useCallback((field, value) => {
        if (!currentTaskId) return;
        sendMessage('task_update', {
            task_id: currentTaskId,
            field,
            value
        });
    }, [currentTaskId, sendMessage]);

    const acquireLock = useCallback((taskId) => {
        sendMessage('acquire_lock', { task_id: taskId });
    }, [sendMessage]);

    const releaseLock = useCallback((taskId) => {
        sendMessage('release_lock', { task_id: taskId });
        setLockStatus(null);
    }, [sendMessage]);

    const value = {
        isConnected,
        activeUsers,
        taskViewers: taskViewers[currentTaskId] || [],
        cursors: cursors[currentTaskId] || {},
        typingUsers: Object.values(typingUsers[currentTaskId] || {}),
        lockStatus,
        currentTaskId,
        getColorForUser,
        joinTask,
        leaveTask,
        moveCursor,
        startTyping,
        stopTyping,
        sendTaskUpdate,
        acquireLock,
        releaseLock
    };

    return (
        <CollaborationContext.Provider value={value}>
            {children}
        </CollaborationContext.Provider>
    );
};

export default CollaborationProvider;
