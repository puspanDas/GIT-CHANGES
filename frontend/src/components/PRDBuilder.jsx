import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Target, X, Loader2, Lightbulb, ArrowRight } from 'lucide-react';

const EXAMPLE_IDEAS = [
    "Add dark mode to reduce eye strain",
    "Implement SSO login for enterprise customers",
    "Build a real-time notification system",
    "Create a mobile-responsive experience",
    "Add automated sprint report generation",
    "Implement AI-powered task estimation"
];

const PRDBuilder = ({ okrs, onGenerate, onClose }) => {
    const [idea, setIdea] = useState('');
    const [selectedOkr, setSelectedOkr] = useState(null);
    const [generating, setGenerating] = useState(false);
    const [step, setStep] = useState(1); // 1: idea, 2: okr alignment

    const handleGenerate = async () => {
        if (!idea.trim()) return;
        setGenerating(true);
        try {
            await onGenerate(idea.trim(), selectedOkr);
        } finally {
            setGenerating(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
            onClick={onClose}
        >
            <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 20 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 20 }}
                transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                className="w-full max-w-2xl bg-background border border-border rounded-2xl shadow-2xl overflow-hidden"
                onClick={e => e.stopPropagation()}
            >
                {/* Header */}
                <div className="px-6 py-4 border-b border-border flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                            <Sparkles className="w-5 h-5 text-white" />
                        </div>
                        <div>
                            <h2 className="text-sm font-bold text-foreground">AI PRD Generator</h2>
                            <p className="text-xs text-muted">Transform an idea into a structured product requirement</p>
                        </div>
                    </div>
                    <button onClick={onClose} className="p-1.5 hover:bg-surface rounded-lg transition-colors">
                        <X className="w-4 h-4 text-muted" />
                    </button>
                </div>

                {/* Step Indicator */}
                <div className="px-6 pt-4">
                    <div className="flex items-center gap-2">
                        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all ${step >= 1 ? 'bg-indigo-500/20 text-indigo-300' : 'bg-surface text-muted'}`}>
                            <Lightbulb className="w-3 h-3" />
                            1. Feature Idea
                        </div>
                        <ArrowRight className="w-3 h-3 text-muted" />
                        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium transition-all ${step >= 2 ? 'bg-indigo-500/20 text-indigo-300' : 'bg-surface text-muted'}`}>
                            <Target className="w-3 h-3" />
                            2. OKR Alignment
                        </div>
                    </div>
                </div>

                {/* Content */}
                <div className="p-6">
                    {step === 1 ? (
                        <div>
                            <label className="block text-xs font-semibold text-muted uppercase tracking-wider mb-2">
                                Describe your feature idea in one sentence
                            </label>
                            <textarea
                                value={idea}
                                onChange={(e) => setIdea(e.target.value)}
                                placeholder="e.g., Add dark mode to reduce eye strain for users working at night"
                                className="w-full h-28 px-4 py-3 bg-surface border border-border rounded-xl text-sm text-foreground placeholder:text-muted/50 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/20 resize-none transition-all"
                                autoFocus
                            />

                            {/* Example Ideas */}
                            <div className="mt-4">
                                <p className="text-xs text-muted mb-2 flex items-center gap-1">
                                    <Lightbulb className="w-3 h-3 text-amber-400" />
                                    Try one of these:
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    {EXAMPLE_IDEAS.map((ex, i) => (
                                        <button
                                            key={i}
                                            onClick={() => setIdea(ex)}
                                            className="px-2.5 py-1 rounded-lg text-xs text-muted hover:text-foreground bg-surface border border-border hover:border-indigo-500/30 transition-all duration-200"
                                        >
                                            {ex}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="flex justify-end mt-4">
                                <button
                                    onClick={() => setStep(2)}
                                    disabled={!idea.trim()}
                                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 hover:bg-indigo-500/25 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                                >
                                    Next: Align to OKR
                                    <ArrowRight className="w-3.5 h-3.5" />
                                </button>
                            </div>
                        </div>
                    ) : (
                        <div>
                            <label className="block text-xs font-semibold text-muted uppercase tracking-wider mb-2">
                                Link to a Business Objective (optional but recommended)
                            </label>
                            <p className="text-xs text-muted mb-3">
                                Aligning features to OKRs demonstrates outcome-driven product thinking.
                            </p>
                            <div className="space-y-2 max-h-56 overflow-y-auto mb-4">
                                {okrs.map((okr) => (
                                    <div
                                        key={okr.id}
                                        onClick={() => setSelectedOkr(selectedOkr === okr.id ? null : okr.id)}
                                        className={`p-3 rounded-xl border cursor-pointer transition-all duration-200 ${selectedOkr === okr.id ? 'border-indigo-500/50 bg-indigo-500/5' : 'border-border hover:border-indigo-500/30 bg-surface/50'}`}
                                    >
                                        <div className="flex items-center gap-2 mb-1">
                                            <span
                                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold"
                                                style={{ background: `${okr.color}20`, color: okr.color }}
                                            >
                                                {okr.category}
                                            </span>
                                            <span className="text-[10px] text-muted">{okr.quarter}</span>
                                            {selectedOkr === okr.id && (
                                                <span className="ml-auto text-[10px] font-semibold text-indigo-400">Selected</span>
                                            )}
                                        </div>
                                        <p className="text-sm font-medium text-foreground">{okr.objective}</p>
                                    </div>
                                ))}
                            </div>

                            <div className="flex items-center justify-between">
                                <button
                                    onClick={() => setStep(1)}
                                    className="px-3 py-1.5 rounded-lg text-xs text-muted hover:text-foreground transition-colors"
                                >
                                    ← Back
                                </button>
                                <button
                                    onClick={handleGenerate}
                                    disabled={generating || !idea.trim()}
                                    className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-medium bg-gradient-to-r from-indigo-500 to-purple-600 text-white hover:from-indigo-600 hover:to-purple-700 transition-all duration-200 shadow-lg shadow-indigo-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {generating ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Generating PRD...
                                        </>
                                    ) : (
                                        <>
                                            <Sparkles className="w-4 h-4" />
                                            Generate PRD
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </motion.div>
        </motion.div>
    );
};

export default PRDBuilder;
