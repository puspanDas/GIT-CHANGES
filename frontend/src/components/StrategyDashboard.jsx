import React, { useState, useEffect } from 'react';
import { getAllPRDs, generatePRD, getOKRs, updatePRDStatus, deletePRD, decomposePRD, createTasksFromPRD, simulateABTest } from '../api';
import PRDBuilder from './PRDBuilder';
import { motion, AnimatePresence } from 'framer-motion';
import {
    FileText, Plus, Target, Sparkles, ChevronRight, Clock, CheckCircle2, Rocket,
    Trash2, ArrowRight, Layers, Loader2, X, BarChart3, TrendingUp, TrendingDown,
    AlertTriangle, Zap, Users, Eye, Check, FlaskConical
} from 'lucide-react';

const STATUS_CONFIG = {
    DRAFT: { label: 'Draft', color: '#6366f1', bg: 'rgba(99, 102, 241, 0.15)', icon: FileText },
    APPROVED: { label: 'Approved', color: '#10b981', bg: 'rgba(16, 185, 129, 0.15)', icon: CheckCircle2 },
    IN_PROGRESS: { label: 'In Progress', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)', icon: Clock },
    SHIPPED: { label: 'Shipped', color: '#06b6d4', bg: 'rgba(6, 182, 212, 0.15)', icon: Rocket }
};

const StrategyDashboard = () => {
    const [prds, setPrds] = useState([]);
    const [okrs, setOkrs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showBuilder, setShowBuilder] = useState(false);
    const [selectedPRD, setSelectedPRD] = useState(null);
    const [decomposedTasks, setDecomposedTasks] = useState(null);
    const [decomposing, setDecomposing] = useState(false);
    const [creatingTasks, setCreatingTasks] = useState(false);
    const [abTestData, setAbTestData] = useState(null);
    const [loadingABTest, setLoadingABTest] = useState(false);
    const [activeView, setActiveView] = useState('prds'); // prds, okrs

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const [prdsData, okrsData] = await Promise.all([getAllPRDs(), getOKRs()]);
            setPrds(prdsData);
            setOkrs(okrsData);
        } catch (err) {
            console.error('Failed to fetch strategy data', err);
        } finally {
            setLoading(false);
        }
    };

    const handleGeneratePRD = async (idea, okrId) => {
        try {
            const prd = await generatePRD(idea, okrId);
            setPrds(prev => [prd, ...prev]);
            setSelectedPRD(prd);
            setShowBuilder(false);
        } catch (err) {
            console.error('Failed to generate PRD', err);
        }
    };

    const handleStatusChange = async (prdId, newStatus) => {
        try {
            const updated = await updatePRDStatus(prdId, newStatus);
            setPrds(prev => prev.map(p => p.id === prdId ? updated : p));
            if (selectedPRD?.id === prdId) setSelectedPRD(updated);
        } catch (err) {
            console.error('Failed to update status', err);
        }
    };

    const handleDelete = async (prdId) => {
        try {
            await deletePRD(prdId);
            setPrds(prev => prev.filter(p => p.id !== prdId));
            if (selectedPRD?.id === prdId) {
                setSelectedPRD(null);
                setDecomposedTasks(null);
            }
        } catch (err) {
            console.error('Failed to delete PRD', err);
        }
    };

    const handleDecompose = async (prdId) => {
        setDecomposing(true);
        try {
            const result = await decomposePRD(prdId);
            setDecomposedTasks(result);
        } catch (err) {
            console.error('Failed to decompose PRD', err);
        } finally {
            setDecomposing(false);
        }
    };

    const handleCreateTasks = async (prdId) => {
        setCreatingTasks(true);
        try {
            await createTasksFromPRD(prdId);
            await fetchData();
            setDecomposedTasks(null);
        } catch (err) {
            console.error('Failed to create tasks', err);
        } finally {
            setCreatingTasks(false);
        }
    };

    const handleABTest = async (prd) => {
        setLoadingABTest(true);
        try {
            const result = await simulateABTest(prd.id, prd.title);
            setAbTestData(result);
        } catch (err) {
            console.error('Failed to simulate A/B test', err);
        } finally {
            setLoadingABTest(false);
        }
    };

    // ==================== RENDER HELPERS ====================

    const renderStatusBadge = (status) => {
        const config = STATUS_CONFIG[status] || STATUS_CONFIG.DRAFT;
        const Icon = config.icon;
        return (
            <span
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold"
                style={{ background: config.bg, color: config.color }}
            >
                <Icon className="w-3 h-3" />
                {config.label}
            </span>
        );
    };

    const renderStatusActions = (prd) => {
        const statusFlow = ['DRAFT', 'APPROVED', 'IN_PROGRESS', 'SHIPPED'];
        const currentIdx = statusFlow.indexOf(prd.status);
        const nextStatus = currentIdx < statusFlow.length - 1 ? statusFlow[currentIdx + 1] : null;

        return (
            <div className="flex items-center gap-2 flex-wrap">
                {nextStatus && (
                    <button
                        onClick={() => handleStatusChange(prd.id, nextStatus)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200"
                        style={{
                            background: STATUS_CONFIG[nextStatus].bg,
                            color: STATUS_CONFIG[nextStatus].color,
                            border: `1px solid ${STATUS_CONFIG[nextStatus].color}30`
                        }}
                    >
                        <ArrowRight className="w-3 h-3" />
                        Move to {STATUS_CONFIG[nextStatus].label}
                    </button>
                )}
                {prd.status === 'APPROVED' && (
                    <button
                        onClick={() => handleDecompose(prd.id)}
                        disabled={decomposing}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-purple-500/15 text-purple-400 border border-purple-500/30 hover:bg-purple-500/25 transition-all duration-200"
                    >
                        {decomposing ? <Loader2 className="w-3 h-3 animate-spin" /> : <Layers className="w-3 h-3" />}
                        Decompose into Tasks
                    </button>
                )}
                {prd.status === 'SHIPPED' && (
                    <button
                        onClick={() => handleABTest(prd)}
                        disabled={loadingABTest}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-cyan-500/15 text-cyan-400 border border-cyan-500/30 hover:bg-cyan-500/25 transition-all duration-200"
                    >
                        {loadingABTest ? <Loader2 className="w-3 h-3 animate-spin" /> : <FlaskConical className="w-3 h-3" />}
                        View A/B Test Results
                    </button>
                )}
                <button
                    onClick={() => handleDelete(prd.id)}
                    className="flex items-center gap-1 px-2 py-1.5 rounded-lg text-xs text-red-400/60 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200"
                >
                    <Trash2 className="w-3 h-3" />
                </button>
            </div>
        );
    };

    // ==================== SUB-VIEWS ====================

    const renderOKRsView = () => (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {okrs.map((okr, idx) => (
                <motion.div
                    key={okr.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.08 }}
                    className="glass-card rounded-xl p-5 border border-border hover:border-indigo-500/30 transition-all duration-300"
                >
                    <div className="flex items-start justify-between mb-3">
                        <span
                            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold"
                            style={{ background: `${okr.color}20`, color: okr.color }}
                        >
                            <Target className="w-3 h-3" />
                            {okr.category}
                        </span>
                        <span className="text-xs text-muted">{okr.quarter}</span>
                    </div>
                    <h3 className="text-sm font-semibold text-foreground mb-3">{okr.objective}</h3>
                    <div className="space-y-2">
                        {okr.key_results.map((kr, i) => (
                            <div key={i} className="flex items-start gap-2 text-xs text-muted">
                                <div className="w-4 h-4 rounded-full border border-border flex items-center justify-center flex-shrink-0 mt-0.5">
                                    <span className="text-[10px]">{i + 1}</span>
                                </div>
                                <span>{kr}</span>
                            </div>
                        ))}
                    </div>
                    <div className="mt-4 pt-3 border-t border-border">
                        <div className="flex items-center justify-between text-xs text-muted">
                            <span>{prds.filter(p => p.okr?.id === okr.id).length} PRDs aligned</span>
                            <div className="flex -space-x-1">
                                {prds.filter(p => p.okr?.id === okr.id).slice(0, 3).map((p, i) => (
                                    <div key={i} className="w-5 h-5 rounded-full bg-surface border border-border flex items-center justify-center">
                                        <FileText className="w-2.5 h-2.5 text-muted" />
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </motion.div>
            ))}
        </div>
    );

    const renderPRDDetail = () => {
        if (!selectedPRD) return null;
        const prd = selectedPRD;

        return (
            <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="glass-card rounded-xl border border-border overflow-hidden"
            >
                {/* Header */}
                <div className="p-5 border-b border-border">
                    <div className="flex items-start justify-between mb-3">
                        <div className="flex-1">
                            <div className="flex items-center gap-2 mb-2">
                                {renderStatusBadge(prd.status)}
                                {prd.okr && (
                                    <span
                                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium"
                                        style={{ background: `${prd.okr.color}15`, color: prd.okr.color }}
                                    >
                                        <Target className="w-3 h-3" />
                                        {prd.okr.category}
                                    </span>
                                )}
                            </div>
                            <h2 className="text-lg font-bold text-foreground">{prd.title}</h2>
                            <p className="text-xs text-muted mt-1">by {prd.author} • {new Date(prd.created_at).toLocaleDateString()}</p>
                        </div>
                        <button onClick={() => { setSelectedPRD(null); setDecomposedTasks(null); setAbTestData(null); }} className="p-1.5 hover:bg-surface rounded-lg transition-colors">
                            <X className="w-4 h-4 text-muted" />
                        </button>
                    </div>
                    {renderStatusActions(prd)}
                </div>

                {/* PRD Content */}
                <div className="p-5 space-y-5 max-h-[calc(100vh-300px)] overflow-y-auto">
                    {/* OKR Alignment */}
                    {prd.okr && (
                        <div className="rounded-lg p-4" style={{ background: `${prd.okr.color}08`, border: `1px solid ${prd.okr.color}20` }}>
                            <div className="flex items-center gap-2 mb-2">
                                <Target className="w-4 h-4" style={{ color: prd.okr.color }} />
                                <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: prd.okr.color }}>Strategic Alignment</span>
                            </div>
                            <p className="text-sm font-medium text-foreground">{prd.okr.objective}</p>
                            <p className="text-xs text-muted mt-1">{prd.okr.quarter}</p>
                        </div>
                    )}

                    {/* Problem Statement */}
                    <section>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                            <AlertTriangle className="w-3.5 h-3.5 text-amber-400" />
                            Problem Statement
                        </h3>
                        <p className="text-sm text-foreground/80 leading-relaxed">{prd.problem_statement}</p>
                    </section>

                    {/* Target Personas */}
                    <section>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                            <Users className="w-3.5 h-3.5 text-indigo-400" />
                            Target Personas
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {prd.target_personas?.map((persona, i) => (
                                <div key={i} className="bg-surface/50 rounded-lg p-3 border border-border">
                                    <p className="text-sm font-medium text-foreground">{persona.name}</p>
                                    <p className="text-xs text-muted mt-1">{persona.description}</p>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Success Metrics */}
                    <section>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                            <TrendingUp className="w-3.5 h-3.5 text-green-400" />
                            Success Metrics (KPIs)
                        </h3>
                        <div className="space-y-2">
                            {prd.success_metrics?.map((metric, i) => (
                                <div key={i} className="flex items-center gap-2 text-sm text-foreground/80">
                                    <Check className="w-3.5 h-3.5 text-green-400 flex-shrink-0" />
                                    <span>{metric}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Out of Scope */}
                    <section>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                            <X className="w-3.5 h-3.5 text-red-400" />
                            Out of Scope
                        </h3>
                        <div className="space-y-1.5">
                            {prd.out_of_scope?.map((item, i) => (
                                <div key={i} className="flex items-center gap-2 text-sm text-foreground/60">
                                    <div className="w-1.5 h-1.5 rounded-full bg-red-400/40" />
                                    <span>{item}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Edge Cases */}
                    <section>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                            <Eye className="w-3.5 h-3.5 text-cyan-400" />
                            Edge Cases
                        </h3>
                        <div className="space-y-1.5">
                            {prd.edge_cases?.map((item, i) => (
                                <div key={i} className="flex items-center gap-2 text-sm text-foreground/70">
                                    <div className="w-1.5 h-1.5 rounded-full bg-cyan-400/40" />
                                    <span>{item}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Risks */}
                    <section>
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                            <AlertTriangle className="w-3.5 h-3.5 text-orange-400" />
                            Risks
                        </h3>
                        <div className="space-y-1.5">
                            {prd.risks?.map((risk, i) => (
                                <div key={i} className="flex items-start gap-2 text-sm text-foreground/70 bg-orange-500/5 rounded-lg p-2.5 border border-orange-500/10">
                                    <AlertTriangle className="w-3.5 h-3.5 text-orange-400 flex-shrink-0 mt-0.5" />
                                    <span>{risk}</span>
                                </div>
                            ))}
                        </div>
                    </section>

                    {/* Competitive Analysis */}
                    {prd.competitive_analysis && (
                        <section>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                                <BarChart3 className="w-3.5 h-3.5 text-violet-400" />
                                Competitive Analysis
                            </h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-xs">
                                    <thead>
                                        <tr className="border-b border-border">
                                            <th className="text-left py-2 px-3 text-muted font-semibold">Competitor</th>
                                            <th className="text-left py-2 px-3 text-muted font-semibold">Has Feature</th>
                                            <th className="text-left py-2 px-3 text-muted font-semibold">Quality</th>
                                            <th className="text-left py-2 px-3 text-muted font-semibold">Notes</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {prd.competitive_analysis.map((comp, i) => (
                                            <tr key={i} className="border-b border-border/50">
                                                <td className="py-2 px-3 text-foreground font-medium">{comp.name}</td>
                                                <td className="py-2 px-3">
                                                    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold ${comp.has_feature ? 'bg-green-500/15 text-green-400' : 'bg-red-500/15 text-red-400'}`}>
                                                        {comp.has_feature ? 'Yes' : 'No'}
                                                    </span>
                                                </td>
                                                <td className="py-2 px-3 text-muted">{comp.quality}</td>
                                                <td className="py-2 px-3 text-muted/70">{comp.notes}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </section>
                    )}

                    {/* Implementation Phases */}
                    {prd.implementation_phases && (
                        <section>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                                <Layers className="w-3.5 h-3.5 text-indigo-400" />
                                Implementation Phases
                            </h3>
                            <div className="space-y-2">
                                {prd.implementation_phases.map((phase, i) => (
                                    <div key={i} className="bg-surface/50 rounded-lg p-3 border border-border flex items-center justify-between">
                                        <div>
                                            <p className="text-sm font-medium text-foreground">{phase.name}</p>
                                            <p className="text-xs text-muted mt-0.5">{phase.scope}</p>
                                        </div>
                                        <div className="text-right flex-shrink-0 ml-3">
                                            <p className="text-xs text-muted">{phase.duration}</p>
                                            <p className="text-xs font-semibold text-indigo-400">{phase.points} pts</p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="mt-3 flex items-center justify-between p-3 bg-indigo-500/10 rounded-lg border border-indigo-500/20">
                                <span className="text-xs font-medium text-indigo-300">Total Estimated Story Points</span>
                                <span className="text-sm font-bold text-indigo-400">{prd.estimated_story_points}</span>
                            </div>
                        </section>
                    )}

                    {/* Decomposed Tasks */}
                    {decomposedTasks && (
                        <section>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-2 flex items-center gap-1.5">
                                <Zap className="w-3.5 h-3.5 text-amber-400" />
                                Decomposed Engineering Tasks ({decomposedTasks.total_story_points} pts)
                            </h3>
                            <div className="space-y-2">
                                {decomposedTasks.tasks.map((task, i) => (
                                    <div key={i} className="bg-surface/50 rounded-lg p-3 border border-border">
                                        <div className="flex items-start justify-between mb-1">
                                            <p className="text-sm font-medium text-foreground">{task.title}</p>
                                            <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                                                <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${task.priority === 'CRITICAL' ? 'bg-red-500/15 text-red-400' : task.priority === 'HIGH' ? 'bg-orange-500/15 text-orange-400' : task.priority === 'MEDIUM' ? 'bg-amber-500/15 text-amber-400' : 'bg-green-500/15 text-green-400'}`}>
                                                    {task.priority}
                                                </span>
                                                <span className="text-xs font-bold text-indigo-400">{task.story_points}pt</span>
                                            </div>
                                        </div>
                                        <p className="text-xs text-muted mb-2">{task.description}</p>
                                        <div className="flex items-center gap-3 text-[10px] text-muted">
                                            <span className="px-1.5 py-0.5 rounded bg-surface border border-border">{task.type}</span>
                                            <span>{task.estimated_days}d est.</span>
                                            <span>{task.phase}</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <button
                                onClick={() => handleCreateTasks(prd.id)}
                                disabled={creatingTasks}
                                className="mt-3 w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium bg-gradient-to-r from-indigo-500 to-purple-600 text-white hover:from-indigo-600 hover:to-purple-700 transition-all duration-200 shadow-lg shadow-indigo-500/20"
                            >
                                {creatingTasks ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                                {creatingTasks ? 'Creating Tasks...' : 'Create All Tasks in TaskFlow'}
                            </button>
                        </section>
                    )}

                    {/* A/B Test Results */}
                    {abTestData && (
                        <section>
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3 flex items-center gap-1.5">
                                <FlaskConical className="w-3.5 h-3.5 text-cyan-400" />
                                A/B Test Results — {abTestData.test_duration_days} Day Test
                            </h3>

                            {/* Recommendation Banner */}
                            <div className={`rounded-lg p-4 mb-4 border ${abTestData.recommendation === 'SHIP_IT' ? 'bg-green-500/10 border-green-500/20' : abTestData.recommendation === 'ITERATE' ? 'bg-orange-500/10 border-orange-500/20' : 'bg-blue-500/10 border-blue-500/20'}`}>
                                <div className="flex items-center gap-2 mb-1">
                                    {abTestData.recommendation === 'SHIP_IT' ? <Rocket className="w-4 h-4 text-green-400" /> : abTestData.recommendation === 'ITERATE' ? <AlertTriangle className="w-4 h-4 text-orange-400" /> : <Clock className="w-4 h-4 text-blue-400" />}
                                    <span className={`text-sm font-bold ${abTestData.recommendation === 'SHIP_IT' ? 'text-green-400' : abTestData.recommendation === 'ITERATE' ? 'text-orange-400' : 'text-blue-400'}`}>
                                        {abTestData.recommendation === 'SHIP_IT' ? '🚀 Ship It!' : abTestData.recommendation === 'ITERATE' ? '🔄 Iterate' : '⏳ Extend Test'}
                                    </span>
                                </div>
                                <p className="text-xs text-foreground/70">{abTestData.recommendation_text}</p>
                            </div>

                            {/* Key Metrics */}
                            <div className="grid grid-cols-3 gap-3 mb-4">
                                <div className="bg-surface/50 rounded-lg p-3 border border-border text-center">
                                    <p className="text-xs text-muted mb-1">Conversion Lift</p>
                                    <p className={`text-lg font-bold ${abTestData.lift_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {abTestData.lift_pct > 0 ? '+' : ''}{abTestData.lift_pct}%
                                    </p>
                                </div>
                                <div className="bg-surface/50 rounded-lg p-3 border border-border text-center">
                                    <p className="text-xs text-muted mb-1">Confidence</p>
                                    <p className={`text-lg font-bold ${abTestData.is_significant ? 'text-green-400' : 'text-amber-400'}`}>
                                        {abTestData.confidence_pct}%
                                    </p>
                                </div>
                                <div className="bg-surface/50 rounded-lg p-3 border border-border text-center">
                                    <p className="text-xs text-muted mb-1">Adoption</p>
                                    <p className="text-lg font-bold text-indigo-400">{abTestData.adoption_rate}%</p>
                                </div>
                            </div>

                            {/* Control vs Variant */}
                            <div className="grid grid-cols-2 gap-3 mb-4">
                                <div className="bg-surface/50 rounded-lg p-3 border border-border">
                                    <p className="text-xs text-muted mb-1">{abTestData.control.name}</p>
                                    <p className="text-sm font-bold text-foreground">{abTestData.control.conversion_rate}% conversion</p>
                                    <p className="text-[10px] text-muted">{abTestData.control.users.toLocaleString()} users</p>
                                </div>
                                <div className="bg-surface/50 rounded-lg p-3 border border-indigo-500/20">
                                    <p className="text-xs text-muted mb-1">{abTestData.variant.name}</p>
                                    <p className="text-sm font-bold text-indigo-400">{abTestData.variant.conversion_rate}% conversion</p>
                                    <p className="text-[10px] text-muted">{abTestData.variant.users.toLocaleString()} users</p>
                                </div>
                            </div>

                            {/* Adoption Chart (Visual Bar) */}
                            <div className="mb-4">
                                <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">Daily Adoption Trend</p>
                                <div className="flex items-end gap-1 h-20">
                                    {abTestData.daily_data.map((day, i) => (
                                        <div key={i} className="flex-1 flex flex-col items-center">
                                            <div
                                                className="w-full rounded-t bg-gradient-to-t from-indigo-500 to-purple-500 transition-all duration-300"
                                                style={{ height: `${Math.max(day.adoption_pct, 2)}%` }}
                                                title={`Day ${day.day}: ${day.adoption_pct}%`}
                                            />
                                            {i % 3 === 0 && <span className="text-[8px] text-muted mt-1">{day.date}</span>}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Segments */}
                            <div className="mb-4">
                                <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">User Segments</p>
                                <div className="space-y-2">
                                    {abTestData.segments.map((seg, i) => (
                                        <div key={i} className="flex items-center justify-between bg-surface/50 rounded-lg p-2.5 border border-border">
                                            <span className="text-xs text-foreground">{seg.name}</span>
                                            <div className="flex items-center gap-3 text-xs">
                                                <span className="text-indigo-400 font-medium">{seg.adoption}% adopted</span>
                                                <span className="text-amber-400 font-medium">★ {seg.satisfaction}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {/* Secondary Metrics */}
                            <div>
                                <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-2">Secondary Metrics</p>
                                <div className="grid grid-cols-3 gap-2">
                                    <div className="bg-surface/50 rounded-lg p-2.5 border border-border text-center">
                                        <p className="text-[10px] text-muted">Session Duration</p>
                                        <p className={`text-xs font-bold ${abTestData.secondary_metrics.avg_session_duration_change.startsWith('+') ? 'text-green-400' : 'text-red-400'}`}>
                                            {abTestData.secondary_metrics.avg_session_duration_change}
                                        </p>
                                    </div>
                                    <div className="bg-surface/50 rounded-lg p-2.5 border border-border text-center">
                                        <p className="text-[10px] text-muted">Support Tickets</p>
                                        <p className={`text-xs font-bold ${abTestData.secondary_metrics.support_tickets_change.startsWith('-') ? 'text-green-400' : 'text-red-400'}`}>
                                            {abTestData.secondary_metrics.support_tickets_change}
                                        </p>
                                    </div>
                                    <div className="bg-surface/50 rounded-lg p-2.5 border border-border text-center">
                                        <p className="text-[10px] text-muted">Load Impact</p>
                                        <p className={`text-xs font-bold ${abTestData.secondary_metrics.page_load_impact_ms <= 0 ? 'text-green-400' : 'text-amber-400'}`}>
                                            {abTestData.secondary_metrics.page_load_impact_ms > 0 ? '+' : ''}{abTestData.secondary_metrics.page_load_impact_ms}ms
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </section>
                    )}
                </div>
            </motion.div>
        );
    };

    // ==================== MAIN RENDER ====================

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <Loader2 className="w-6 h-6 animate-spin text-indigo-400" />
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col">
            {/* Header */}
            <div className="px-6 pt-4 pb-3 border-b border-border">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-lg font-bold text-foreground flex items-center gap-2">
                            <Sparkles className="w-5 h-5 text-indigo-400" />
                            Product Strategy
                        </h1>
                        <p className="text-xs text-muted mt-0.5">AI-powered PRD generation, OKR alignment, and feature validation</p>
                    </div>
                    <button
                        onClick={() => setShowBuilder(true)}
                        className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-gradient-to-r from-indigo-500 to-purple-600 text-white hover:from-indigo-600 hover:to-purple-700 transition-all duration-200 shadow-lg shadow-indigo-500/20"
                    >
                        <Plus className="w-4 h-4" />
                        New PRD
                    </button>
                </div>

                {/* View Tabs */}
                <div className="flex gap-1 mt-3">
                    <button
                        onClick={() => setActiveView('prds')}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${activeView === 'prds' ? 'bg-indigo-500/20 text-indigo-300' : 'text-muted hover:text-foreground hover:bg-surface'}`}
                    >
                        <FileText className="w-3.5 h-3.5 inline mr-1.5" />
                        PRDs ({prds.length})
                    </button>
                    <button
                        onClick={() => setActiveView('okrs')}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${activeView === 'okrs' ? 'bg-indigo-500/20 text-indigo-300' : 'text-muted hover:text-foreground hover:bg-surface'}`}
                    >
                        <Target className="w-3.5 h-3.5 inline mr-1.5" />
                        OKRs ({okrs.length})
                    </button>
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden p-4">
                {activeView === 'okrs' ? (
                    renderOKRsView()
                ) : (
                    <div className="flex gap-4 h-full">
                        {/* PRD List */}
                        <div className={`${selectedPRD ? 'w-80' : 'w-full'} flex-shrink-0 overflow-y-auto space-y-2 transition-all duration-300`}>
                            {prds.length === 0 ? (
                                <div className="flex flex-col items-center justify-center h-64 text-center">
                                    <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 flex items-center justify-center mb-4">
                                        <Sparkles className="w-8 h-8 text-indigo-400" />
                                    </div>
                                    <p className="text-sm font-medium text-foreground mb-1">No PRDs yet</p>
                                    <p className="text-xs text-muted mb-4">Start by generating a PRD from a feature idea</p>
                                    <button
                                        onClick={() => setShowBuilder(true)}
                                        className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/25 transition-all duration-200"
                                    >
                                        <Plus className="w-4 h-4" />
                                        Generate First PRD
                                    </button>
                                </div>
                            ) : (
                                prds.map((prd, idx) => (
                                    <motion.div
                                        key={prd.id}
                                        initial={{ opacity: 0, y: 10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: idx * 0.05 }}
                                        onClick={() => { setSelectedPRD(prd); setDecomposedTasks(null); setAbTestData(null); }}
                                        className={`glass-card rounded-xl p-4 border cursor-pointer transition-all duration-200 ${selectedPRD?.id === prd.id ? 'border-indigo-500/50 bg-indigo-500/5' : 'border-border hover:border-indigo-500/30'}`}
                                    >
                                        <div className="flex items-start justify-between mb-2">
                                            {renderStatusBadge(prd.status)}
                                            <ChevronRight className="w-4 h-4 text-muted" />
                                        </div>
                                        <h3 className="text-sm font-semibold text-foreground mb-1 line-clamp-2">{prd.title}</h3>
                                        <div className="flex items-center gap-2 text-[10px] text-muted">
                                            {prd.okr && (
                                                <span className="px-1.5 py-0.5 rounded-full" style={{ background: `${prd.okr.color}15`, color: prd.okr.color }}>
                                                    {prd.okr.category}
                                                </span>
                                            )}
                                            <span>{prd.estimated_story_points} pts</span>
                                            <span>•</span>
                                            <span>{new Date(prd.created_at).toLocaleDateString()}</span>
                                        </div>
                                    </motion.div>
                                ))
                            )}
                        </div>

                        {/* Detail Panel */}
                        {selectedPRD && (
                            <div className="flex-1 min-w-0 overflow-hidden">
                                {renderPRDDetail()}
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* PRD Builder Modal */}
            <AnimatePresence>
                {showBuilder && (
                    <PRDBuilder
                        okrs={okrs}
                        onGenerate={handleGeneratePRD}
                        onClose={() => setShowBuilder(false)}
                    />
                )}
            </AnimatePresence>
        </div>
    );
};

export default StrategyDashboard;
