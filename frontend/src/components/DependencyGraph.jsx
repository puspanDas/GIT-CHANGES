import React, { useState, useEffect, useRef } from 'react';
import { getDependencyGraph } from '../api';
import {
    GitBranch,
    AlertTriangle,
    CheckCircle2,
    Clock,
    Zap,
    Filter,
    ZoomIn,
    ZoomOut,
    Maximize2
} from 'lucide-react';

const DependencyGraph = ({ projectId, tasks = [], onTaskClick }) => {
    const [graphData, setGraphData] = useState({ nodes: [], edges: [], blocked_tasks: [] });
    const [loading, setLoading] = useState(false);
    const [zoom, setZoom] = useState(1);
    const [filter, setFilter] = useState('all'); // all, blocked, has-dependencies
    const svgRef = useRef(null);
    const containerRef = useRef(null);

    useEffect(() => {
        if (projectId) {
            fetchGraph();
        } else if (tasks.length > 0) {
            // Build graph from provided tasks
            buildLocalGraph();
        }
    }, [projectId, tasks]);

    const fetchGraph = async () => {
        setLoading(true);
        try {
            const data = await getDependencyGraph(projectId);
            setGraphData(data);
        } catch (error) {
            console.error('Failed to fetch dependency graph:', error);
        }
        setLoading(false);
    };

    const buildLocalGraph = () => {
        const nodes = tasks.map(t => ({
            id: t.id,
            title: t.title,
            status: t.status,
            priority: t.priority,
            dependencies: t.dependencies || []
        }));

        const edges = [];
        tasks.forEach(task => {
            (task.dependencies || []).forEach(depId => {
                edges.push({ from: depId, to: task.id, type: 'dependency' });
            });
        });

        // Calculate blocked tasks
        const taskStatuses = {};
        tasks.forEach(t => { taskStatuses[t.id] = t.status; });

        const blocked_tasks = [];
        tasks.forEach(task => {
            const deps = task.dependencies || [];
            if (deps.length > 0) {
                const blockingDeps = deps.filter(depId => taskStatuses[depId] !== 'DONE');
                if (blockingDeps.length > 0) {
                    blocked_tasks.push({ task_id: task.id, blocked_by: blockingDeps });
                }
            }
        });

        setGraphData({ nodes, edges, blocked_tasks });
    };

    // Calculate node positions using a simple layered layout
    const calculateLayout = () => {
        const { nodes, edges } = graphData;
        if (nodes.length === 0) return { nodePositions: {}, width: 400, height: 300 };

        // Build adjacency for topological sort
        const inDegree = {};
        const adjacency = {};
        nodes.forEach(n => {
            inDegree[n.id] = 0;
            adjacency[n.id] = [];
        });

        edges.forEach(e => {
            if (inDegree[e.to] !== undefined) {
                inDegree[e.to]++;
            }
            if (adjacency[e.from]) {
                adjacency[e.from].push(e.to);
            }
        });

        // Layer assignment using BFS
        const layers = [];
        const nodeLayer = {};
        const queue = nodes.filter(n => inDegree[n.id] === 0).map(n => n.id);

        if (queue.length === 0) {
            // Handle cycles - just put all in layer 0
            nodes.forEach(n => { nodeLayer[n.id] = 0; });
            layers[0] = nodes.map(n => n.id);
        } else {
            let layer = 0;
            while (queue.length > 0) {
                const layerNodes = [...queue];
                layers[layer] = layerNodes;
                layerNodes.forEach(id => { nodeLayer[id] = layer; });
                queue.length = 0;

                layerNodes.forEach(id => {
                    adjacency[id].forEach(child => {
                        inDegree[child]--;
                        if (inDegree[child] === 0) {
                            queue.push(child);
                        }
                    });
                });
                layer++;
            }

            // Add remaining nodes (cycles) to last layer
            nodes.forEach(n => {
                if (nodeLayer[n.id] === undefined) {
                    nodeLayer[n.id] = layer;
                    if (!layers[layer]) layers[layer] = [];
                    layers[layer].push(n.id);
                }
            });
        }

        // Calculate positions
        const nodeWidth = 180;
        const nodeHeight = 70;
        const layerGap = 150;
        const nodeGap = 30;

        const nodePositions = {};
        layers.forEach((layerNodes, layerIdx) => {
            const layerHeight = layerNodes.length * (nodeHeight + nodeGap) - nodeGap;
            const startY = -layerHeight / 2;

            layerNodes.forEach((nodeId, nodeIdx) => {
                nodePositions[nodeId] = {
                    x: layerIdx * (nodeWidth + layerGap) + 50,
                    y: startY + nodeIdx * (nodeHeight + nodeGap) + 200
                };
            });
        });

        const maxX = Math.max(...Object.values(nodePositions).map(p => p.x)) + nodeWidth + 50;
        const maxY = Math.max(...Object.values(nodePositions).map(p => p.y)) + nodeHeight + 50;
        const minY = Math.min(...Object.values(nodePositions).map(p => p.y)) - 50;

        // Adjust Y to make all positive
        Object.values(nodePositions).forEach(pos => {
            pos.y -= minY;
        });

        return {
            nodePositions,
            width: Math.max(400, maxX),
            height: Math.max(300, maxY - minY + 100)
        };
    };

    const getStatusColor = (status) => {
        switch (status) {
            case 'DONE': return '#22c55e';
            case 'IN_PROGRESS': return '#3b82f6';
            case 'IN_REVIEW': return '#f59e0b';
            default: return '#6b7280';
        }
    };

    const getPriorityBorder = (priority) => {
        switch (priority) {
            case 'CRITICAL': return '3px solid #ef4444';
            case 'HIGH': return '3px solid #f97316';
            case 'MEDIUM': return '3px solid #eab308';
            default: return '2px solid rgba(255,255,255,0.2)';
        }
    };

    const isBlocked = (taskId) => {
        return graphData.blocked_tasks.some(b => b.task_id === taskId);
    };

    const getBlockingInfo = (taskId) => {
        const blocked = graphData.blocked_tasks.find(b => b.task_id === taskId);
        if (!blocked) return null;
        return blocked.blocked_by;
    };

    const filteredNodes = () => {
        switch (filter) {
            case 'blocked':
                return graphData.nodes.filter(n => isBlocked(n.id));
            case 'has-dependencies':
                return graphData.nodes.filter(n => n.dependencies?.length > 0);
            default:
                return graphData.nodes;
        }
    };

    const { nodePositions, width, height } = calculateLayout();

    if (loading) {
        return (
            <div className="glass-card p-6 flex items-center justify-center">
                <div className="animate-spin rounded-full h-8 w-8 border-2 border-purple-500 border-t-transparent"></div>
            </div>
        );
    }

    const displayNodes = filteredNodes();
    const displayNodeIds = new Set(displayNodes.map(n => n.id));
    const displayEdges = graphData.edges.filter(e =>
        displayNodeIds.has(e.from) && displayNodeIds.has(e.to)
    );

    return (
        <div className="glass-card p-4">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                    <GitBranch className="text-purple-400" size={20} />
                    <h3 className="text-lg font-semibold text-white">Dependency Graph</h3>
                </div>

                <div className="flex items-center gap-3">
                    {/* Filter */}
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        className="bg-white/10 border border-white/20 rounded-lg px-3 py-1.5 text-sm text-white"
                    >
                        <option value="all">All Tasks</option>
                        <option value="blocked">Blocked Only</option>
                        <option value="has-dependencies">With Dependencies</option>
                    </select>

                    {/* Zoom controls */}
                    <div className="flex items-center gap-1 bg-white/10 rounded-lg">
                        <button
                            onClick={() => setZoom(z => Math.max(0.5, z - 0.1))}
                            className="p-1.5 hover:bg-white/10 rounded-l-lg transition-colors"
                        >
                            <ZoomOut size={16} className="text-white/70" />
                        </button>
                        <span className="text-xs text-white/70 px-2">{Math.round(zoom * 100)}%</span>
                        <button
                            onClick={() => setZoom(z => Math.min(2, z + 0.1))}
                            className="p-1.5 hover:bg-white/10 rounded-r-lg transition-colors"
                        >
                            <ZoomIn size={16} className="text-white/70" />
                        </button>
                    </div>
                </div>
            </div>

            {/* Stats */}
            <div className="flex gap-4 mb-4">
                <div className="flex items-center gap-2 text-sm">
                    <div className="w-3 h-3 rounded-full bg-gray-500"></div>
                    <span className="text-white/60">Total: {graphData.nodes.length}</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <span className="text-white/60">Blocked: {graphData.blocked_tasks.length}</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                    <GitBranch size={14} className="text-purple-400" />
                    <span className="text-white/60">Dependencies: {graphData.edges.length}</span>
                </div>
            </div>

            {/* Graph Container */}
            <div
                ref={containerRef}
                className="relative overflow-auto bg-black/20 rounded-lg"
                style={{ height: '400px' }}
            >
                <svg
                    ref={svgRef}
                    width={width * zoom}
                    height={height * zoom}
                    className="cursor-move"
                    style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}
                >
                    <defs>
                        {/* Arrow marker */}
                        <marker
                            id="arrowhead"
                            markerWidth="10"
                            markerHeight="7"
                            refX="9"
                            refY="3.5"
                            orient="auto"
                        >
                            <polygon points="0 0, 10 3.5, 0 7" fill="#8b5cf6" />
                        </marker>

                        {/* Blocked arrow marker */}
                        <marker
                            id="arrowhead-blocked"
                            markerWidth="10"
                            markerHeight="7"
                            refX="9"
                            refY="3.5"
                            orient="auto"
                        >
                            <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
                        </marker>

                        {/* Glow filter */}
                        <filter id="glow">
                            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                            <feMerge>
                                <feMergeNode in="coloredBlur" />
                                <feMergeNode in="SourceGraphic" />
                            </feMerge>
                        </filter>
                    </defs>

                    {/* Edges */}
                    {displayEdges.map((edge, idx) => {
                        const fromPos = nodePositions[edge.from];
                        const toPos = nodePositions[edge.to];
                        if (!fromPos || !toPos) return null;

                        const isBlockingEdge = isBlocked(edge.to);
                        const fromTask = graphData.nodes.find(n => n.id === edge.from);
                        const isFromIncomplete = fromTask && fromTask.status !== 'DONE';

                        return (
                            <g key={`edge-${idx}`}>
                                <line
                                    x1={fromPos.x + 180}
                                    y1={fromPos.y + 35}
                                    x2={toPos.x}
                                    y2={toPos.y + 35}
                                    stroke={isBlockingEdge && isFromIncomplete ? '#ef4444' : '#8b5cf6'}
                                    strokeWidth="2"
                                    strokeDasharray={isBlockingEdge && isFromIncomplete ? "5,5" : "none"}
                                    markerEnd={isBlockingEdge && isFromIncomplete ? "url(#arrowhead-blocked)" : "url(#arrowhead)"}
                                    opacity="0.7"
                                />
                            </g>
                        );
                    })}

                    {/* Nodes */}
                    {displayNodes.map(node => {
                        const pos = nodePositions[node.id];
                        if (!pos) return null;

                        const blocked = isBlocked(node.id);
                        const blockingIds = getBlockingInfo(node.id);

                        return (
                            <g
                                key={node.id}
                                transform={`translate(${pos.x}, ${pos.y})`}
                                onClick={() => onTaskClick && onTaskClick(node)}
                                style={{ cursor: 'pointer' }}
                            >
                                {/* Node background */}
                                <rect
                                    width="180"
                                    height="70"
                                    rx="8"
                                    fill={blocked ? "rgba(239, 68, 68, 0.2)" : "rgba(255, 255, 255, 0.1)"}
                                    stroke={blocked ? "#ef4444" : "rgba(255, 255, 255, 0.2)"}
                                    strokeWidth="2"
                                    filter={blocked ? "url(#glow)" : "none"}
                                />

                                {/* Status indicator */}
                                <circle
                                    cx="15"
                                    cy="20"
                                    r="6"
                                    fill={getStatusColor(node.status)}
                                />

                                {/* Task title */}
                                <text
                                    x="30"
                                    y="25"
                                    fill="white"
                                    fontSize="12"
                                    fontWeight="500"
                                >
                                    {node.title?.substring(0, 18)}{node.title?.length > 18 ? '...' : ''}
                                </text>

                                {/* Task ID */}
                                <text
                                    x="15"
                                    y="45"
                                    fill="rgba(255,255,255,0.5)"
                                    fontSize="10"
                                >
                                    #{node.id}
                                </text>

                                {/* Dependency count */}
                                {node.dependencies?.length > 0 && (
                                    <g transform="translate(140, 35)">
                                        <rect
                                            x="0"
                                            y="0"
                                            width="30"
                                            height="20"
                                            rx="4"
                                            fill="rgba(139, 92, 246, 0.3)"
                                        />
                                        <text
                                            x="15"
                                            y="14"
                                            fill="#c4b5fd"
                                            fontSize="10"
                                            textAnchor="middle"
                                        >
                                            {node.dependencies.length} dep
                                        </text>
                                    </g>
                                )}

                                {/* Blocked indicator */}
                                {blocked && (
                                    <g transform="translate(155, 5)">
                                        <AlertTriangle size={16} color="#ef4444" />
                                    </g>
                                )}

                                {/* Priority indicator */}
                                <rect
                                    x="0"
                                    y="60"
                                    width="180"
                                    height="10"
                                    rx="0 0 8 8"
                                    fill={
                                        node.priority === 'CRITICAL' ? 'rgba(239, 68, 68, 0.5)' :
                                            node.priority === 'HIGH' ? 'rgba(249, 115, 22, 0.5)' :
                                                node.priority === 'MEDIUM' ? 'rgba(234, 179, 8, 0.5)' :
                                                    'rgba(107, 114, 128, 0.3)'
                                    }
                                />
                            </g>
                        );
                    })}
                </svg>
            </div>

            {/* Legend */}
            <div className="flex flex-wrap gap-4 mt-4 text-xs text-white/60">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-500"></div>
                    <span>Done</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                    <span>In Progress</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                    <span>In Review</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-gray-500"></div>
                    <span>Todo</span>
                </div>
                <div className="flex items-center gap-2">
                    <AlertTriangle size={12} className="text-red-500" />
                    <span>Blocked</span>
                </div>
            </div>
        </div>
    );
};

export default DependencyGraph;
