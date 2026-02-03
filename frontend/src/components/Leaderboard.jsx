import React, { useState, useEffect } from 'react';
import { getLeaderboard } from '../api';
import { Trophy, Medal, Star, Zap, TrendingUp, X, Crown, Flame, Award } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const Leaderboard = ({ compact = false }) => {
    const [leaderboard, setLeaderboard] = useState([]);
    const [loading, setLoading] = useState(true);
    const [expanded, setExpanded] = useState(false);
    const [selectedUser, setSelectedUser] = useState(null);

    useEffect(() => {
        fetchLeaderboard();
        // Poll for updates every 10 seconds
        const interval = setInterval(fetchLeaderboard, 10000);
        return () => clearInterval(interval);
    }, [expanded]);

    const fetchLeaderboard = async () => {
        try {
            const data = await getLeaderboard(expanded ? 10 : (compact ? 3 : 5));
            setLeaderboard(data);
        } catch (err) {
            console.error("Failed to fetch leaderboard:", err);
        } finally {
            setLoading(false);
        }
    };

    const getRankIcon = (rank) => {
        switch (rank) {
            case 1:
                return (
                    <motion.span
                        className="text-xl"
                        animate={{ rotate: [0, 10, -10, 0] }}
                        transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 2 }}
                    >
                        🥇
                    </motion.span>
                );
            case 2:
                return <span className="text-xl">🥈</span>;
            case 3:
                return <span className="text-xl">🥉</span>;
            default:
                return <span className="text-sm font-bold text-slate-500">#{rank}</span>;
        }
    };

    const getRankGlow = (rank, levelConfig) => {
        if (rank === 1) return `0 0 20px ${levelConfig?.glow || 'rgba(234, 179, 8, 0.6)'}`;
        if (rank === 2) return `0 0 15px rgba(192, 192, 192, 0.4)`;
        if (rank === 3) return `0 0 12px rgba(205, 127, 50, 0.4)`;
        return 'none';
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center p-4">
                <div className="animate-spin w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full"></div>
            </div>
        );
    }

    if (leaderboard.length === 0) {
        return (
            <div className="text-center py-4 text-slate-400 text-sm">
                <Trophy className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p>No champions yet!</p>
                <p className="text-xs">Complete tasks to earn XP</p>
            </div>
        );
    }

    const LeaderboardItem = ({ user, index }) => (
        <motion.div
            key={user.user_id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`flex items-center gap-3 p-2 rounded-lg transition-all duration-200 cursor-pointer ${user.rank === 1
                ? 'bg-gradient-to-r from-yellow-500/20 to-orange-500/10 border border-yellow-500/30'
                : user.rank <= 3
                    ? 'bg-white/5 border border-white/10'
                    : 'hover:bg-white/5'
                }`}
            style={{
                boxShadow: getRankGlow(user.rank, user.level_config)
            }}
            whileHover={{ scale: 1.02, x: 5 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => setSelectedUser(user)}
        >
            {/* Rank */}
            <div className="w-8 flex justify-center">
                {getRankIcon(user.rank)}
            </div>

            {/* Avatar with level glow */}
            <motion.div
                className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold border-2 transition-all"
                style={{
                    backgroundColor: user.level_config?.color + '30',
                    borderColor: user.level_config?.color,
                    boxShadow: `0 0 ${6 + user.level * 2}px ${user.level_config?.glow || 'transparent'}`
                }}
                whileHover={{
                    boxShadow: `0 0 ${12 + user.level * 2}px ${user.level_config?.glow || 'transparent'}`
                }}
            >
                {user.username.substring(0, 2).toUpperCase()}
            </motion.div>

            {/* User info */}
            <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                    <span className="font-medium text-sm text-white truncate">
                        {user.username}
                    </span>
                    <span
                        className="text-xs px-1.5 py-0.5 rounded-full"
                        style={{
                            backgroundColor: user.level_config?.color + '30',
                            color: user.level_config?.color
                        }}
                    >
                        Lv.{user.level}
                    </span>
                </div>
                <div className="text-xs text-slate-400">
                    {user.level_config?.title}
                </div>
            </div>

            {/* XP */}
            <div className="text-right">
                <motion.div
                    className="flex items-center gap-1 text-sm font-bold"
                    style={{ color: user.level_config?.color }}
                    whileHover={{ scale: 1.1 }}
                >
                    <Zap className="w-3 h-3" />
                    {user.xp.toLocaleString()}
                </motion.div>
                <div className="text-xs text-slate-500">
                    {user.total_tasks_completed} tasks
                </div>
            </div>
        </motion.div>
    );

    return (
        <div className={compact ? 'relative' : 'p-4'}>
            {!compact && (
                <div className="flex items-center gap-2 mb-4">
                    <Trophy className="w-5 h-5 text-yellow-500" />
                    <h3 className="text-lg font-bold text-white">Sprint Champions</h3>
                </div>
            )}

            <motion.div
                className="space-y-2 cursor-pointer"
                onClick={compact ? () => setExpanded(true) : undefined}
            >
                {leaderboard.map((user, index) => (
                    <LeaderboardItem key={user.user_id} user={user} index={index} />
                ))}
            </motion.div>

            {compact && leaderboard.length > 0 && (
                <motion.button
                    className="w-full mt-2 py-1.5 text-xs text-center text-indigo-400 hover:text-indigo-300 hover:bg-indigo-500/10 rounded-lg transition-all"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setExpanded(true)}
                >
                    View Full Leaderboard →
                </motion.button>
            )}

            {!compact && leaderboard.length > 0 && (
                <div className="mt-4 pt-4 border-t border-white/10 text-center">
                    <p className="text-xs text-slate-400">
                        Complete tasks to climb the leaderboard! 🚀
                    </p>
                </div>
            )}

            {/* Expanded Leaderboard Modal */}
            <AnimatePresence>
                {expanded && (
                    <motion.div
                        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setExpanded(false)}
                    >
                        <motion.div
                            className="w-full max-w-md mx-4 p-6 rounded-2xl bg-slate-900/95 border border-white/10 shadow-2xl"
                            initial={{ scale: 0.9, y: 50 }}
                            animate={{ scale: 1, y: 0 }}
                            exit={{ scale: 0.9, y: 50 }}
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="flex items-center justify-between mb-6">
                                <div className="flex items-center gap-3">
                                    <motion.div
                                        animate={{ rotate: [0, 10, -10, 0] }}
                                        transition={{ duration: 0.5, repeat: Infinity, repeatDelay: 3 }}
                                    >
                                        <Trophy className="w-8 h-8 text-yellow-500" />
                                    </motion.div>
                                    <div>
                                        <h2 className="text-xl font-bold text-white">Sprint Champions</h2>
                                        <p className="text-sm text-slate-400">Top performers this sprint</p>
                                    </div>
                                </div>
                                <button
                                    onClick={() => setExpanded(false)}
                                    className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                                >
                                    <X className="w-5 h-5 text-slate-400" />
                                </button>
                            </div>

                            <div className="space-y-2 max-h-96 overflow-y-auto">
                                {leaderboard.map((user, index) => (
                                    <LeaderboardItem key={user.user_id} user={user} index={index} />
                                ))}
                            </div>

                            <div className="mt-6 p-4 rounded-xl bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/20">
                                <div className="flex items-center gap-2 mb-2">
                                    <Flame className="w-5 h-5 text-orange-500" />
                                    <span className="font-semibold text-white">How to Climb</span>
                                </div>
                                <ul className="text-xs text-slate-300 space-y-1">
                                    <li>🎯 Complete HIGH priority tasks = +100 XP</li>
                                    <li>🐛 Fix bugs = +50 XP</li>
                                    <li>📋 Complete MEDIUM tasks = +35 XP</li>
                                    <li>✅ Complete LOW tasks = +25 XP</li>
                                </ul>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* User Detail Modal */}
            <AnimatePresence>
                {selectedUser && (
                    <motion.div
                        className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={() => setSelectedUser(null)}
                    >
                        <motion.div
                            className="w-full max-w-sm mx-4 p-6 rounded-2xl text-center"
                            style={{
                                background: `linear-gradient(135deg, ${selectedUser.level_config?.color}10, ${selectedUser.level_config?.color}30)`,
                                border: `2px solid ${selectedUser.level_config?.color}`,
                                boxShadow: `0 0 40px ${selectedUser.level_config?.glow || selectedUser.level_config?.color}`
                            }}
                            initial={{ scale: 0.9, y: 50 }}
                            animate={{ scale: 1, y: 0 }}
                            exit={{ scale: 0.9, y: 50 }}
                            onClick={(e) => e.stopPropagation()}
                        >
                            {/* Rank Badge */}
                            <div className="text-4xl mb-4">
                                {getRankIcon(selectedUser.rank)}
                            </div>

                            {/* Avatar */}
                            <motion.div
                                className="w-20 h-20 mx-auto rounded-full flex items-center justify-center text-2xl font-bold mb-4"
                                style={{
                                    backgroundColor: selectedUser.level_config?.color + '30',
                                    border: `3px solid ${selectedUser.level_config?.color}`,
                                    boxShadow: `0 0 30px ${selectedUser.level_config?.glow || 'transparent'}`
                                }}
                                animate={{
                                    boxShadow: [
                                        `0 0 20px ${selectedUser.level_config?.glow || 'transparent'}`,
                                        `0 0 40px ${selectedUser.level_config?.glow || 'transparent'}`,
                                        `0 0 20px ${selectedUser.level_config?.glow || 'transparent'}`
                                    ]
                                }}
                                transition={{ duration: 2, repeat: Infinity }}
                            >
                                {selectedUser.username.substring(0, 2).toUpperCase()}
                            </motion.div>

                            <h3 className="text-xl font-bold text-white mb-1">{selectedUser.username}</h3>
                            <div
                                className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium mb-4"
                                style={{
                                    backgroundColor: selectedUser.level_config?.color + '30',
                                    color: selectedUser.level_config?.color
                                }}
                            >
                                <Crown className="w-4 h-4" />
                                Level {selectedUser.level} • {selectedUser.level_config?.title}
                            </div>

                            <div className="grid grid-cols-2 gap-4 mb-4">
                                <div className="p-3 rounded-lg bg-white/5">
                                    <div className="text-2xl font-bold text-yellow-500 flex items-center justify-center gap-1">
                                        <Zap className="w-5 h-5" />
                                        {selectedUser.xp.toLocaleString()}
                                    </div>
                                    <div className="text-xs text-slate-400">Total XP</div>
                                </div>
                                <div className="p-3 rounded-lg bg-white/5">
                                    <div className="text-2xl font-bold text-indigo-400 flex items-center justify-center gap-1">
                                        <Award className="w-5 h-5" />
                                        {selectedUser.total_tasks_completed}
                                    </div>
                                    <div className="text-xs text-slate-400">Tasks Done</div>
                                </div>
                            </div>

                            <button
                                onClick={() => setSelectedUser(null)}
                                className="w-full py-2 px-4 rounded-lg bg-white/10 hover:bg-white/20 text-white font-medium transition-colors"
                            >
                                Close
                            </button>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Leaderboard;
