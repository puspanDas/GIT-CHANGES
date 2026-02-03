import React, { useEffect, useState } from 'react';
import { useCollaboration } from './CollaborationProvider';

// Level-based glow configuration (synced with backend gamification_service)
const LEVEL_GLOW_CONFIG = {
    1: { color: '#6b7280', glow: 'rgba(107, 114, 128, 0.4)', intensity: 1 },
    2: { color: '#6b7280', glow: 'rgba(107, 114, 128, 0.5)', intensity: 1.2 },
    3: { color: '#3b82f6', glow: 'rgba(59, 130, 246, 0.5)', intensity: 1.5 },
    4: { color: '#3b82f6', glow: 'rgba(59, 130, 246, 0.6)', intensity: 1.8 },
    5: { color: '#8b5cf6', glow: 'rgba(139, 92, 246, 0.6)', intensity: 2 },
    6: { color: '#8b5cf6', glow: 'rgba(139, 92, 246, 0.7)', intensity: 2.3 },
    7: { color: '#f59e0b', glow: 'rgba(245, 158, 11, 0.7)', intensity: 2.5 },
    8: { color: '#f59e0b', glow: 'rgba(245, 158, 11, 0.8)', intensity: 2.8 },
    9: { color: '#eab308', glow: 'rgba(234, 179, 8, 0.8)', intensity: 3 },
    10: { color: '#ef4444', glow: 'rgba(239, 68, 68, 0.9)', intensity: 3.5 },
};

const getLevelConfig = (level) => {
    const clampedLevel = Math.min(Math.max(level || 1, 1), 10);
    return LEVEL_GLOW_CONFIG[clampedLevel] || LEVEL_GLOW_CONFIG[1];
};

const LiveCursors = ({ containerRef }) => {
    const { cursors, isConnected } = useCollaboration();
    const [containerBounds, setContainerBounds] = useState(null);

    useEffect(() => {
        if (containerRef?.current) {
            const updateBounds = () => {
                const bounds = containerRef.current.getBoundingClientRect();
                setContainerBounds(bounds);
            };

            updateBounds();
            window.addEventListener('resize', updateBounds);

            return () => window.removeEventListener('resize', updateBounds);
        }
    }, [containerRef]);

    if (!isConnected || Object.keys(cursors).length === 0) return null;

    return (
        <div className="live-cursors-container pointer-events-none fixed inset-0 z-50">
            {Object.entries(cursors).map(([userId, cursor]) => {
                // Get level-based glow configuration
                const level = cursor.level || 1;
                const levelConfig = getLevelConfig(level);
                const cursorColor = levelConfig.color;
                const glowIntensity = levelConfig.intensity;
                const glowColor = levelConfig.glow;

                return (
                    <div
                        key={userId}
                        className="live-cursor absolute transition-all duration-75 ease-out"
                        style={{
                            left: cursor.x,
                            top: cursor.y,
                            transform: 'translate(-2px, -2px)'
                        }}
                    >
                        {/* Cursor Glow Effect (level-based) */}
                        <div
                            className="absolute -inset-2 rounded-full animate-pulse"
                            style={{
                                background: `radial-gradient(circle, ${glowColor} 0%, transparent 70%)`,
                                transform: `scale(${glowIntensity})`,
                                opacity: 0.6,
                                filter: `blur(${2 * glowIntensity}px)`
                            }}
                        />

                        {/* Cursor SVG with level-based color */}
                        <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            style={{
                                filter: `drop-shadow(0 0 ${4 * glowIntensity}px ${glowColor}) drop-shadow(0 2px 4px rgba(0,0,0,0.3))`,
                                position: 'relative',
                                zIndex: 1
                            }}
                        >
                            <path
                                d="M5.65 2.128c-.8-.333-1.62.434-1.367 1.274l4.98 16.54c.3.996 1.645 1.11 2.1.177l2.67-5.477c.13-.267.348-.485.615-.616l5.477-2.67c.933-.455.82-1.8-.177-2.1L3.508 4.276l2.142-2.148z"
                                fill={cursorColor}
                                stroke="white"
                                strokeWidth="1.5"
                            />
                        </svg>

                        {/* Username label with level badge */}
                        <div
                            className="cursor-label absolute left-5 top-4 px-2 py-1 rounded text-xs text-white whitespace-nowrap flex items-center gap-1"
                            style={{
                                backgroundColor: cursorColor,
                                boxShadow: `0 0 ${6 * glowIntensity}px ${glowColor}, 0 2px 8px rgba(0,0,0,0.3)`
                            }}
                        >
                            <span>{cursor.username}</span>
                            {level > 1 && (
                                <span
                                    className="text-[10px] px-1 rounded-full bg-white/20"
                                    style={{ textShadow: '0 0 2px rgba(0,0,0,0.5)' }}
                                >
                                    Lv.{level}
                                </span>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
    );
};

export default LiveCursors;

