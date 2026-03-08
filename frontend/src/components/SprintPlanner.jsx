import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
    Calendar, Users, Zap, Target, AlertTriangle, CheckCircle2,
    ChevronRight, Loader2, RefreshCw, Play, BarChart3, Clock,
    User, ArrowRight, TrendingUp
} from 'lucide-react';
import { getSprintPlan, applySprintAssignments } from '../api';

// Constants for maintainability
const MAX_WORKLOAD = 5;
const REFRESH_DELAY_MS = 3000;

const SprintPlanner = ({ onClose }) => {
    const [sprintPlan, setSprintPlan] = useState(null);
    const [loading, setLoading] = useState(true);
    const [applying, setApplying] = useState(false);
    const [error, setError] = useState(null);
    const [sprintDays, setSprintDays] = useState(14);
    const [selectedAssignments, setSelectedAssignments] = useState([]);
    const [applyResult, setApplyResult] = useState(null);

    // Memoize selected task IDs as Set for O(1) lookup
    const selectedTaskIds = useMemo(() =>
        new Set(selectedAssignments.map(a => a.task_id)),
        [selectedAssignments]
    );

    const fetchSprintPlan = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getSprintPlan(sprintDays);
            setSprintPlan(data);
            // Pre-select all assignments
            if (data?.optimization?.assignment_suggestions) {
                setSelectedAssignments(
                    data.optimization.assignment_suggestions.map(s => ({
                        task_id: s.task_id,
                        assignee_id: s.suggested_assignee_id
                    }))
                );
            }
        } catch (err) {
            setError('Failed to load sprint plan');
            console.error(err);
        } finally {
            setLoading(false);
        }
    }, [sprintDays]);

    useEffect(() => {
        fetchSprintPlan();
    }, [fetchSprintPlan]);

    const toggleAssignment = useCallback((taskId, assigneeId) => {
        setSelectedAssignments(prev => {
            const exists = prev.some(a => a.task_id === taskId);
            if (exists) {
                return prev.filter(a => a.task_id !== taskId);
            }
            return [...prev, { task_id: taskId, assignee_id: assigneeId }];
        });
    }, []);

    const handleApply = useCallback(async () => {
        if (selectedAssignments.length === 0) return;

        setApplying(true);
        try {
            const result = await applySprintAssignments(selectedAssignments);
            setApplyResult(result);
            // Refresh plan after applying
            setTimeout(() => {
                fetchSprintPlan();
                setApplyResult(null);
            }, REFRESH_DELAY_MS);
        } catch (err) {
            setError('Failed to apply assignments');
        } finally {
            setApplying(false);
        }
    }, [selectedAssignments, fetchSprintPlan]);

    // Memoize color functions to avoid recreation on each render
    const getPriorityColor = useCallback((priority) => {
        switch (priority) {
            case 'CRITICAL': return 'text-red-400 bg-red-400/10 border-red-400/30';
            case 'HIGH': return 'text-orange-400 bg-orange-400/10 border-orange-400/30';
            case 'MEDIUM': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30';
            default: return 'text-green-400 bg-green-400/10 border-green-400/30';
        }
    }, []);

    const getWorkloadColor = useCallback((total) => {
        if (total >= MAX_WORKLOAD) return 'bg-red-500';
        if (total >= 4) return 'bg-orange-500';
        if (total >= 2) return 'bg-yellow-500';
        return 'bg-green-500';
    }, []);

    return (
        <div className="glass-modal w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="px-6 py-4 border-b border-white/10 bg-gradient-to-r from-indigo-500/10 to-purple-500/10">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-indigo-500/20">
                            <Calendar className="w-5 h-5 text-indigo-400" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-foreground">AI Sprint Planner</h2>
                            <p className="text-sm text-muted">Optimize task assignments with AI</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <select
                            value={sprintDays}
                            onChange={(e) => setSprintDays(Number(e.target.value))}
                            className="glass-input px-3 py-1.5 text-sm"
                        >
                            <option value={7}>1 Week</option>
                            <option value={14}>2 Weeks</option>
                            <option value={21}>3 Weeks</option>
                        </select>
                        <button
                            onClick={fetchSprintPlan}
                            className="p-2 rounded-lg hover:bg-white/5 transition-colors"
                        >
                            <RefreshCw className={`w-4 h-4 text-muted ${loading ? 'animate-spin' : ''}`} />
                        </button>
                        <button
                            onClick={onClose}
                            className="text-muted hover:text-white transition-colors text-xl"
                        >
                            ×
                        </button>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {loading && (
                    <div className="flex items-center justify-center py-16">
                        <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
                        <span className="ml-3 text-muted">Generating optimal sprint plan...</span>
                    </div>
                )}

                {error && (
                    <div className="flex items-center gap-2 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                        <span className="text-red-400">{error}</span>
                    </div>
                )}

                {applyResult && (
                    <div className="flex items-center gap-2 p-4 rounded-lg bg-green-500/10 border border-green-500/20">
                        <CheckCircle2 className="w-5 h-5 text-green-400" />
                        <span className="text-green-400">
                            Applied {applyResult.total_applied} assignments successfully!
                        </span>
                    </div>
                )}

                {sprintPlan && !loading && (
                    <>
                        {/* Summary Cards */}
                        <div className="grid grid-cols-4 gap-4">
                            <div className="glass-card p-4">
                                <div className="flex items-center gap-2 text-muted text-sm mb-1">
                                    <Target className="w-4 h-4" />
                                    Sprint Duration
                                </div>
                                <div className="text-2xl font-bold text-foreground">{sprintDays} days</div>
                            </div>
                            <div className="glass-card p-4">
                                <div className="flex items-center gap-2 text-muted text-sm mb-1">
                                    <Zap className="w-4 h-4" />
                                    Tasks to Assign
                                </div>
                                <div className="text-2xl font-bold text-indigo-400">
                                    {sprintPlan.optimization?.tasks_assigned_in_plan || 0}
                                </div>
                            </div>
                            <div className="glass-card p-4">
                                <div className="flex items-center gap-2 text-muted text-sm mb-1">
                                    <TrendingUp className="w-4 h-4" />
                                    Velocity
                                </div>
                                <div className="text-2xl font-bold text-green-400">
                                    {sprintPlan.velocity?.velocity_per_day?.toFixed(1) || 0}/day
                                </div>
                            </div>
                            <div className="glass-card p-4">
                                <div className="flex items-center gap-2 text-muted text-sm mb-1">
                                    <Clock className="w-4 h-4" />
                                    Est. Completion
                                </div>
                                <div className={`text-2xl font-bold ${sprintPlan.on_track ? 'text-green-400' : 'text-red-400'}`}>
                                    {sprintPlan.estimated_completion_days} days
                                </div>
                            </div>
                        </div>

                        {/* Recommendation */}
                        {sprintPlan.optimization?.recommendation && (
                            <div className="glass-card p-4 border-l-4 border-indigo-500">
                                <div className="flex items-start gap-3">
                                    <Zap className="w-5 h-5 text-indigo-400 mt-0.5" />
                                    <div>
                                        <div className="text-sm font-medium text-foreground mb-1">AI Recommendation</div>
                                        <div className="text-sm text-muted">{sprintPlan.optimization.recommendation}</div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Workload Distribution */}
                        {sprintPlan.optimization?.workload_distribution?.length > 0 && (
                            <div className="glass-card p-4">
                                <div className="flex items-center gap-2 mb-4">
                                    <BarChart3 className="w-5 h-5 text-purple-400" />
                                    <span className="font-medium text-foreground">Workload Distribution</span>
                                </div>
                                <div className="space-y-3">
                                    {sprintPlan.optimization.workload_distribution.map((dev, idx) => (
                                        <div key={idx} className="flex items-center gap-4">
                                            <div className="w-32 flex items-center gap-2">
                                                <User className="w-4 h-4 text-muted" />
                                                <span className="text-sm text-muted truncate">{dev.username}</span>
                                            </div>
                                            <div className="flex-1 h-6 bg-white/5 rounded-full overflow-hidden flex">
                                                <div
                                                    className={`h-full ${getWorkloadColor(dev.current_tasks)} opacity-60`}
                                                    style={{ width: `${(dev.current_tasks / 5) * 100}%` }}
                                                />
                                                <div
                                                    className={`h-full ${getWorkloadColor(dev.total_after_plan)}`}
                                                    style={{ width: `${(dev.suggested_new_tasks / 5) * 100}%` }}
                                                />
                                            </div>
                                            <div className="w-20 text-right">
                                                <span className="text-sm text-muted">{dev.current_tasks}</span>
                                                <span className="text-muted mx-1">→</span>
                                                <span className="text-sm text-white font-medium">{dev.total_after_plan}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Assignment Suggestions */}
                        {sprintPlan.optimization?.assignment_suggestions?.length > 0 && (
                            <div className="glass-card p-4">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-2">
                                        <Users className="w-5 h-5 text-cyan-400" />
                                        <span className="font-medium text-foreground">Suggested Assignments</span>
                                    </div>
                                    <span className="text-sm text-muted">
                                        {selectedAssignments.length} selected
                                    </span>
                                </div>
                                <div className="space-y-2 max-h-64 overflow-y-auto">
                                    {sprintPlan.optimization.assignment_suggestions.map((suggestion, idx) => {
                                        const isSelected = selectedAssignments.some(a => a.task_id === suggestion.task_id);
                                        return (
                                            <div
                                                key={idx}
                                                onClick={() => toggleAssignment(suggestion.task_id, suggestion.suggested_assignee_id)}
                                                className={`flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all ${isSelected
                                                    ? 'bg-indigo-500/20 border border-indigo-500/30'
                                                    : 'bg-white/5 border border-transparent hover:bg-white/10'
                                                    }`}
                                            >
                                                <div className={`w-4 h-4 rounded border-2 flex items-center justify-center ${isSelected ? 'bg-indigo-500 border-indigo-500' : 'border-gray-500'
                                                    }`}>
                                                    {isSelected && <CheckCircle2 className="w-3 h-3 text-foreground" />}
                                                </div>
                                                <div className="flex-1 min-w-0">
                                                    <div className="text-sm text-foreground truncate">{suggestion.task_title}</div>
                                                    <div className="text-xs text-muted">{suggestion.reason}</div>
                                                </div>
                                                <span className={`px-2 py-0.5 text-xs rounded border ${getPriorityColor(suggestion.task_priority)}`}>
                                                    {suggestion.task_priority}
                                                </span>
                                                <ArrowRight className="w-4 h-4 text-muted" />
                                                <div className="flex items-center gap-1.5">
                                                    <User className="w-4 h-4 text-indigo-400" />
                                                    <span className="text-sm text-indigo-400">{suggestion.suggested_assignee}</span>
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Milestones */}
                        {sprintPlan.milestones?.length > 0 && (
                            <div className="glass-card p-4">
                                <div className="flex items-center gap-2 mb-4">
                                    <Target className="w-5 h-5 text-yellow-400" />
                                    <span className="font-medium text-foreground">Sprint Milestones</span>
                                </div>
                                <div className="space-y-3">
                                    {sprintPlan.milestones.map((milestone, idx) => (
                                        <div key={idx} className="flex items-center gap-4">
                                            <div className="w-16 text-center">
                                                <div className="text-lg font-bold text-foreground">Day {milestone.day}</div>
                                            </div>
                                            <ChevronRight className="w-4 h-4 text-muted" />
                                            <div className="text-sm text-muted">{milestone.target}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>

            {/* Footer */}
            <div className="px-6 py-4 border-t border-white/10 flex items-center justify-between">
                <button
                    onClick={onClose}
                    className="btn-secondary"
                >
                    Cancel
                </button>
                <button
                    onClick={handleApply}
                    disabled={selectedAssignments.length === 0 || applying}
                    className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {applying ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Applying...
                        </>
                    ) : (
                        <>
                            <Play className="w-4 h-4" />
                            Apply {selectedAssignments.length} Assignments
                        </>
                    )}
                </button>
            </div>
        </div>
    );
};

export default SprintPlanner;
