import React from 'react';
import { useCollaboration } from './CollaborationProvider';
import { Users, Wifi, WifiOff } from 'lucide-react';

// Avatar component for user presence
const UserAvatar = ({ username, color, size = 'md', showStatus = false, status = 'online' }) => {
    const sizes = {
        sm: 'w-6 h-6 text-xs',
        md: 'w-8 h-8 text-sm',
        lg: 'w-10 h-10 text-base'
    };

    const statusColors = {
        online: 'bg-green-500',
        away: 'bg-yellow-500',
        offline: 'bg-gray-500'
    };

    const initial = username ? username.charAt(0).toUpperCase() : '?';

    return (
        <div className="relative">
            <div
                className={`${sizes[size]} rounded-full flex items-center justify-center font-medium text-white ring-2 ring-white/20`}
                style={{ backgroundColor: color }}
                title={username}
            >
                {initial}
            </div>
            {showStatus && (
                <div
                    className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-gray-900 ${statusColors[status]}`}
                />
            )}
        </div>
    );
};

// Task viewer avatars stack
export const TaskViewerStack = ({ maxVisible = 3 }) => {
    const { taskViewers, getColorForUser, isConnected } = useCollaboration();

    if (!isConnected || taskViewers.length === 0) return null;

    const visibleViewers = taskViewers.slice(0, maxVisible);
    const remainingCount = taskViewers.length - maxVisible;

    return (
        <div className="flex items-center gap-2">
            <div className="flex -space-x-2">
                {visibleViewers.map((viewer, idx) => (
                    <div key={viewer.user_id} style={{ zIndex: maxVisible - idx }}>
                        <UserAvatar
                            username={viewer.username}
                            color={getColorForUser(viewer.user_id)}
                            size="sm"
                        />
                    </div>
                ))}
                {remainingCount > 0 && (
                    <div
                        className="w-6 h-6 rounded-full bg-gray-600 flex items-center justify-center text-xs text-white ring-2 ring-white/20"
                        style={{ zIndex: 0 }}
                    >
                        +{remainingCount}
                    </div>
                )}
            </div>
            <span className="text-xs text-white/60">
                {taskViewers.length} viewing
            </span>
        </div>
    );
};

// Connection status indicator
export const ConnectionStatus = () => {
    const { isConnected } = useCollaboration();

    return (
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs ${isConnected
                ? 'bg-green-500/20 text-green-400'
                : 'bg-red-500/20 text-red-400'
            }`}>
            {isConnected ? (
                <>
                    <Wifi size={14} />
                    <span>Live</span>
                </>
            ) : (
                <>
                    <WifiOff size={14} />
                    <span>Offline</span>
                </>
            )}
        </div>
    );
};

// Active users panel
export const ActiveUsersPanel = () => {
    const { activeUsers, getColorForUser, isConnected } = useCollaboration();

    return (
        <div className="glass-card p-4">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Users size={18} className="text-purple-400" />
                    <h3 className="text-sm font-medium text-white">Active Users</h3>
                </div>
                <ConnectionStatus />
            </div>

            {activeUsers.length === 0 ? (
                <p className="text-sm text-white/50">No other users online</p>
            ) : (
                <div className="space-y-2">
                    {activeUsers.map(user => (
                        <div
                            key={user.user_id}
                            className="flex items-center gap-3 p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
                        >
                            <UserAvatar
                                username={user.username}
                                color={getColorForUser(user.user_id)}
                                size="md"
                                showStatus
                                status="online"
                            />
                            <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium text-white truncate">
                                    {user.username}
                                </p>
                                {user.task_id && (
                                    <p className="text-xs text-white/50 truncate">
                                        Viewing task #{user.task_id}
                                    </p>
                                )}
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

// Typing indicator
export const TypingIndicator = () => {
    const { typingUsers } = useCollaboration();

    if (typingUsers.length === 0) return null;

    const text = typingUsers.length === 1
        ? `${typingUsers[0]} is typing...`
        : typingUsers.length === 2
            ? `${typingUsers[0]} and ${typingUsers[1]} are typing...`
            : `${typingUsers[0]} and ${typingUsers.length - 1} others are typing...`;

    return (
        <div className="flex items-center gap-2 text-sm text-white/60 animate-pulse">
            <div className="typing-dots flex gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
            <span>{text}</span>
        </div>
    );
};

// Main presence indicator for task cards
const PresenceIndicator = ({ taskId, compact = false }) => {
    const { taskViewers, getColorForUser, isConnected } = useCollaboration();

    // Filter viewers for this specific task
    const viewers = taskViewers.filter(v => v.task_id === taskId);

    if (!isConnected || viewers.length === 0) return null;

    if (compact) {
        return (
            <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-white/60">{viewers.length}</span>
            </div>
        );
    }

    return (
        <div className="flex items-center gap-2">
            <div className="flex -space-x-2">
                {viewers.slice(0, 3).map((viewer, idx) => (
                    <div key={viewer.user_id} style={{ zIndex: 3 - idx }}>
                        <UserAvatar
                            username={viewer.username}
                            color={getColorForUser(viewer.user_id)}
                            size="sm"
                        />
                    </div>
                ))}
            </div>
            {viewers.length > 3 && (
                <span className="text-xs text-white/60">+{viewers.length - 3}</span>
            )}
        </div>
    );
};

export default PresenceIndicator;
