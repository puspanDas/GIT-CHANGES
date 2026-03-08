import React, { useState, useEffect } from 'react';
import { getGamificationStats } from '../api';
import { Zap, TrendingUp, Star, Award, ChevronUp, Target, Flame, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const XPDisplay = ({ compact = false, onXPChange }) => {
    const [stats, setStats] = useState(null);
    const [prevStats, setPrevStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [expanded, setExpanded] = useState(false);
    const [showXPGain, setShowXPGain] = useState(false);
    const [xpGained, setXpGained] = useState(0);

    useEffect(() => {
        fetchStats();
        // Poll for updates every 5 seconds
        const interval = setInterval(fetchStats, 5000);
        return () => clearInterval(interval);
    }, []);

    // Detect XP changes
    useEffect(() => {
        if (prevStats && stats && stats.xp > prevStats.xp) {
            const gained = stats.xp - prevStats.xp;
            setXpGained(gained);
            setShowXPGain(true);
            onXPChange?.(gained);
            setTimeout(() => setShowXPGain(false), 2000);
        }
    }, [stats, prevStats, onXPChange]);

    const fetchStats = async () => {
        try {
            const data = await getGamificationStats();
            setPrevStats(stats);
            setStats(data);
        } catch (err) {
            console.error("Failed to fetch gamification stats:", err);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="animate-pulse bg-white/5 rounded-lg h-16"></div>
        );
    }

    if (!stats) {
        return null;
    }

    const progressPercent = stats.xp_progress_percent || 0;

    // Calculate streak color based on level
    const getStreakColor = () => {
        if (progressPercent > 80) return '#10B981'; // Green - almost there!
        if (progressPercent > 50) return stats.level_config?.color;
        if (progressPercent > 25) return '#F59E0B'; // Orange
        return '#6B7280'; // Gray
    };

    if (compact) {
        return (
            <motion.div
                className="cursor-pointer"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setExpanded(!expanded)}
            >
                {/* XP Gain Popup */}
                <AnimatePresence>
                    {showXPGain && (
                        <motion.div
                            className="absolute -top-8 left-1/2 transform -translate-x-1/2 flex items-center gap-1 px-3 py-1 rounded-full bg-gradient-to-r from-yellow-500 to-orange-500 text-white font-bold text-sm shadow-lg z-50"
                            initial={{ opacity: 0, y: 20, scale: 0.5 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: -20 }}
                        >
                            <Zap className="w-4 h-4" />
                            +{xpGained} XP
                            <Flame className="w-4 h-4 text-yellow-200" />
                        </motion.div>
                    )}
                </AnimatePresence>

                <div className="flex items-center gap-2 px-2 py-1 relative">
                    <motion.div
                        className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold relative"
                        style={{
                            backgroundColor: stats.level_config?.color + '30',
                            borderColor: stats.level_config?.color,
                            border: `2px solid ${stats.level_config?.color}`,
                            boxShadow: `0 0 ${8 + stats.level * 2}px ${stats.level_config?.glow || 'transparent'}`
                        }}
                        animate={{
                            boxShadow: [
                                `0 0 ${8 + stats.level * 2}px ${stats.level_config?.glow || 'transparent'}`,
                                `0 0 ${12 + stats.level * 2}px ${stats.level_config?.glow || 'transparent'}`,
                                `0 0 ${8 + stats.level * 2}px ${stats.level_config?.glow || 'transparent'}`
                            ]
                        }}
                        transition={{ duration: 2, repeat: Infinity }}
                    >
                        {stats.level}
                    </motion.div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1 text-xs">
                            <Zap className="w-3 h-3" style={{ color: stats.level_config?.color }} />
                            <span className="font-medium" style={{ color: stats.level_config?.color }}>
                                {stats.xp.toLocaleString()} XP
                            </span>
                        </div>
                        <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                            <motion.div
                                className="h-full rounded-full relative"
                                style={{
                                    backgroundColor: getStreakColor(),
                                    boxShadow: `0 0 8px ${getStreakColor()}`
                                }}
                                initial={{ width: 0 }}
                                animate={{ width: `${progressPercent}%` }}
                                transition={{ duration: 0.5 }}
                            >
                                {/* Shimmer effect */}
                                <motion.div
                                    className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
                                    animate={{ x: ['-100%', '200%'] }}
                                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                                />
                            </motion.div>
                        </div>
                        <div className="text-[10px] text-muted mt-0.5">
                            {stats.xp_progress}/{stats.xp_to_next_level} to Lv.{stats.level + 1}
                        </div>
                    </div>
                </div>

                {/* Expanded View */}
                <AnimatePresence>
                    {expanded && (
                        <motion.div
                            className="absolute bottom-full left-0 right-0 mb-2 p-4 rounded-xl bg-background/95 backdrop-blur-sm border border-white/10 shadow-xl z-50"
                            initial={{ opacity: 0, y: 10, scale: 0.95 }}
                            animate={{ opacity: 1, y: 0, scale: 1 }}
                            exit={{ opacity: 0, y: 10, scale: 0.95 }}
                        >
                            <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-2">
                                    <Star className="w-5 h-5" style={{ color: stats.level_config?.color }} />
                                    <span className="font-bold text-foreground">Your Stats</span>
                                </div>
                                <button onClick={(e) => { e.stopPropagation(); setExpanded(false); }}>
                                    <X className="w-4 h-4 text-muted hover:text-foreground" />
                                </button>
                            </div>

                            <div className="grid grid-cols-2 gap-3">
                                <motion.div
                                    className="p-3 rounded-lg bg-white/5 border border-white/10"
                                    whileHover={{ scale: 1.02 }}
                                >
                                    <div className="text-2xl font-bold" style={{ color: stats.level_config?.color }}>
                                        Lv.{stats.level}
                                    </div>
                                    <div className="text-xs text-muted">{stats.level_config?.title}</div>
                                </motion.div>

                                <motion.div
                                    className="p-3 rounded-lg bg-white/5 border border-white/10"
                                    whileHover={{ scale: 1.02 }}
                                >
                                    <div className="text-2xl font-bold text-yellow-500 flex items-center gap-1">
                                        <Zap className="w-5 h-5" />
                                        {stats.xp.toLocaleString()}
                                    </div>
                                    <div className="text-xs text-muted">Total XP</div>
                                </motion.div>

                                <motion.div
                                    className="p-3 rounded-lg bg-white/5 border border-white/10"
                                    whileHover={{ scale: 1.02 }}
                                >
                                    <div className="text-2xl font-bold text-indigo-400 flex items-center gap-1">
                                        <Award className="w-5 h-5" />
                                        {stats.total_tasks_completed}
                                    </div>
                                    <div className="text-xs text-muted">Tasks Done</div>
                                </motion.div>

                                <motion.div
                                    className="p-3 rounded-lg bg-white/5 border border-white/10"
                                    whileHover={{ scale: 1.02 }}
                                >
                                    <div className="text-2xl font-bold text-green-400 flex items-center gap-1">
                                        <Target className="w-5 h-5" />
                                        {Math.round(progressPercent)}%
                                    </div>
                                    <div className="text-xs text-muted">To Next Level</div>
                                </motion.div>
                            </div>

                            {/* XP Tips */}
                            <div className="mt-3 p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20">
                                <p className="text-xs text-indigo-300">
                                    💡 Complete HIGH/CRITICAL tasks for +100 XP!
                                </p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>
        );
    }

    // Full-size display
    return (
        <motion.div
            className="glass-card p-4 rounded-xl cursor-pointer"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.01 }}
            onClick={() => setExpanded(!expanded)}
        >
            {/* XP Gain Popup */}
            <AnimatePresence>
                {showXPGain && (
                    <motion.div
                        className="absolute -top-4 left-1/2 transform -translate-x-1/2 flex items-center gap-1 px-4 py-2 rounded-full bg-gradient-to-r from-yellow-500 to-orange-500 text-white font-bold shadow-lg shadow-yellow-500/50 z-50"
                        initial={{ opacity: 0, y: 20, scale: 0.5 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -30 }}
                    >
                        <motion.div animate={{ rotate: [0, 15, -15, 0] }} transition={{ duration: 0.3, repeat: 2 }}>
                            <Zap className="w-5 h-5" />
                        </motion.div>
                        +{xpGained} XP!
                        <Flame className="w-5 h-5 text-yellow-200" />
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Level Badge */}
            <div className="flex items-center gap-3 mb-3">
                <motion.div
                    className="w-12 h-12 rounded-xl flex items-center justify-center text-xl font-bold"
                    style={{
                        backgroundColor: stats.level_config?.color + '20',
                        border: `2px solid ${stats.level_config?.color}`,
                        boxShadow: `0 0 ${10 + stats.level * 3}px ${stats.level_config?.glow || 'transparent'}`
                    }}
                    animate={{
                        boxShadow: [
                            `0 0 ${10 + stats.level * 3}px ${stats.level_config?.glow || 'transparent'}`,
                            `0 0 ${15 + stats.level * 3}px ${stats.level_config?.glow || 'transparent'}`,
                            `0 0 ${10 + stats.level * 3}px ${stats.level_config?.glow || 'transparent'}`
                        ]
                    }}
                    transition={{ duration: 2, repeat: Infinity }}
                    whileHover={{ scale: 1.1, rotate: [0, -5, 5, 0] }}
                >
                    <span style={{ color: stats.level_config?.color }}>{stats.level}</span>
                </motion.div>
                <div>
                    <div className="text-sm text-muted">Level {stats.level}</div>
                    <div
                        className="text-lg font-bold"
                        style={{ color: stats.level_config?.color }}
                    >
                        {stats.level_config?.title}
                    </div>
                </div>
                <motion.div
                    className="ml-auto"
                    animate={{ y: [0, -3, 0] }}
                    transition={{ duration: 1, repeat: Infinity }}
                >
                    <ChevronUp className="w-5 h-5 text-muted" />
                </motion.div>
            </div>

            {/* XP Progress */}
            <div className="mb-2">
                <div className="flex justify-between text-xs mb-1">
                    <span className="text-muted">XP Progress</span>
                    <span style={{ color: stats.level_config?.color }}>
                        {stats.xp_progress} / {stats.xp_to_next_level}
                    </span>
                </div>
                <div className="w-full h-3 bg-white/10 rounded-full overflow-hidden">
                    <motion.div
                        className="h-full rounded-full relative"
                        style={{
                            backgroundColor: getStreakColor(),
                            boxShadow: `0 0 10px ${stats.level_config?.glow}`
                        }}
                        initial={{ width: 0 }}
                        animate={{ width: `${progressPercent}%` }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                    >
                        {/* Shimmer effect */}
                        <motion.div
                            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent"
                            animate={{ x: ['-100%', '200%'] }}
                            transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
                        />
                    </motion.div>
                </div>
            </div>

            {/* Stats Row */}
            <div className="flex items-center justify-between text-sm pt-2 border-t border-white/10">
                <motion.div
                    className="flex items-center gap-1 cursor-pointer"
                    whileHover={{ scale: 1.05 }}
                >
                    <Zap className="w-4 h-4 text-yellow-500" />
                    <span className="text-white font-medium">{stats.xp.toLocaleString()}</span>
                    <span className="text-muted">XP</span>
                </motion.div>
                <motion.div
                    className="flex items-center gap-1 cursor-pointer"
                    whileHover={{ scale: 1.05 }}
                >
                    <Award className="w-4 h-4 text-indigo-400" />
                    <span className="text-white font-medium">{stats.total_tasks_completed}</span>
                    <span className="text-muted">tasks</span>
                </motion.div>
            </div>
        </motion.div>
    );
};

export default XPDisplay;
