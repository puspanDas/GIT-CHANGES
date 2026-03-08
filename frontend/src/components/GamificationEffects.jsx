import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Trophy, Star, Sparkles, ArrowUp } from 'lucide-react';

// XP Gain Notification Component
export const XPNotification = ({ amount, onComplete }) => {
    useEffect(() => {
        const timer = setTimeout(() => {
            onComplete?.();
        }, 2000);
        return () => clearTimeout(timer);
    }, [onComplete]);

    return (
        <motion.div
            className="fixed top-20 right-8 z-50 pointer-events-none"
            initial={{ opacity: 0, y: 50, scale: 0.5 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -50, scale: 0.5 }}
        >
            <div className="flex items-center gap-2 px-6 py-4 rounded-xl bg-gradient-to-r from-yellow-500 to-orange-500 text-white font-bold text-xl shadow-lg shadow-yellow-500/50">
                <motion.div
                    animate={{ rotate: [0, 15, -15, 0] }}
                    transition={{ duration: 0.5, repeat: 2 }}
                >
                    <Zap className="w-8 h-8" />
                </motion.div>
                <span>+{amount} XP</span>
                <motion.div
                    className="absolute -top-2 -right-2"
                    initial={{ scale: 0 }}
                    animate={{ scale: [0, 1.2, 1] }}
                    transition={{ delay: 0.2 }}
                >
                    <Sparkles className="w-6 h-6 text-yellow-200" />
                </motion.div>
            </div>
        </motion.div>
    );
};

// Level Up Celebration Modal
export const LevelUpModal = ({ level, levelConfig, onClose }) => {
    useEffect(() => {
        // Auto-close after 4 seconds
        const timer = setTimeout(onClose, 4000);
        return () => clearTimeout(timer);
    }, [onClose]);

    return (
        <motion.div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
        >
            <motion.div
                className="relative p-8 rounded-2xl text-center"
                style={{
                    background: `linear-gradient(135deg, ${levelConfig?.color}20, ${levelConfig?.color}40)`,
                    border: `3px solid ${levelConfig?.color}`,
                    boxShadow: `0 0 60px ${levelConfig?.glow || levelConfig?.color}`
                }}
                initial={{ scale: 0, rotate: -180 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: "spring", damping: 10 }}
            >
                {/* Particles */}
                {[...Array(12)].map((_, i) => (
                    <motion.div
                        key={i}
                        className="absolute w-3 h-3 rounded-full"
                        style={{ backgroundColor: levelConfig?.color }}
                        initial={{
                            x: 0,
                            y: 0,
                            opacity: 1
                        }}
                        animate={{
                            x: Math.cos(i * 30 * Math.PI / 180) * 150,
                            y: Math.sin(i * 30 * Math.PI / 180) * 150,
                            opacity: 0
                        }}
                        transition={{ duration: 1, ease: "easeOut" }}
                    />
                ))}

                <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ duration: 0.5, repeat: Infinity }}
                >
                    <Trophy className="w-20 h-20 mx-auto mb-4" style={{ color: levelConfig?.color }} />
                </motion.div>

                <motion.h2
                    className="text-3xl font-bold text-white mb-2"
                    initial={{ y: 20, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                >
                    LEVEL UP!
                </motion.h2>

                <motion.div
                    className="text-6xl font-black mb-2"
                    style={{ color: levelConfig?.color }}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.5, type: "spring" }}
                >
                    {level}
                </motion.div>

                <motion.p
                    className="text-xl font-semibold mb-4"
                    style={{ color: levelConfig?.color }}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.7 }}
                >
                    {levelConfig?.title}
                </motion.p>

                <motion.p
                    className="text-foreground/70 text-sm"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.9 }}
                >
                    Keep completing tasks to level up! 🚀
                </motion.p>
            </motion.div>
        </motion.div>
    );
};

// Gamification Context for global state
const GamificationContext = React.createContext();

export const GamificationProvider = ({ children }) => {
    const [notifications, setNotifications] = useState([]);
    const [levelUp, setLevelUp] = useState(null);
    const [stats, setStats] = useState(null);

    const showXPGain = useCallback((amount) => {
        const id = Date.now();
        setNotifications(prev => [...prev, { id, amount }]);
    }, []);

    const removeNotification = useCallback((id) => {
        setNotifications(prev => prev.filter(n => n.id !== id));
    }, []);

    const showLevelUp = useCallback((level, levelConfig) => {
        setLevelUp({ level, levelConfig });
    }, []);

    const hideLevelUp = useCallback(() => {
        setLevelUp(null);
    }, []);

    const updateStats = useCallback((newStats) => {
        if (stats && newStats.level > stats.level) {
            // User leveled up!
            showLevelUp(newStats.level, newStats.level_config);
        }
        if (stats && newStats.xp > stats.xp) {
            // XP gained
            showXPGain(newStats.xp - stats.xp);
        }
        setStats(newStats);
    }, [stats, showLevelUp, showXPGain]);

    return (
        <GamificationContext.Provider value={{
            stats,
            updateStats,
            showXPGain,
            showLevelUp
        }}>
            {children}

            {/* XP Notifications */}
            <AnimatePresence>
                {notifications.map((notif, index) => (
                    <motion.div
                        key={notif.id}
                        style={{ top: `${80 + index * 70}px` }}
                        initial={{ opacity: 0, x: 100 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 100 }}
                    >
                        <XPNotification
                            amount={notif.amount}
                            onComplete={() => removeNotification(notif.id)}
                        />
                    </motion.div>
                ))}
            </AnimatePresence>

            {/* Level Up Modal */}
            <AnimatePresence>
                {levelUp && (
                    <LevelUpModal
                        level={levelUp.level}
                        levelConfig={levelUp.levelConfig}
                        onClose={hideLevelUp}
                    />
                )}
            </AnimatePresence>
        </GamificationContext.Provider>
    );
};

export const useGamification = () => {
    const context = React.useContext(GamificationContext);
    if (!context) {
        throw new Error('useGamification must be used within GamificationProvider');
    }
    return context;
};

// Clickable Stats Card Component
export const StatsCard = ({ icon: Icon, label, value, color, onClick }) => {
    return (
        <motion.div
            className="flex items-center gap-3 p-3 rounded-lg bg-white/5 border border-white/10 cursor-pointer hover:bg-white/10 transition-all"
            whileHover={{ scale: 1.02, y: -2 }}
            whileTap={{ scale: 0.98 }}
            onClick={onClick}
        >
            <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: color + '30' }}
            >
                <Icon className="w-5 h-5" style={{ color }} />
            </div>
            <div>
                <div className="text-xs text-muted">{label}</div>
                <div className="text-lg font-bold text-foreground">{value}</div>
            </div>
        </motion.div>
    );
};

export default GamificationProvider;
