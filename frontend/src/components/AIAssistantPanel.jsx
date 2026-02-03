import React, { useState, useEffect } from 'react';
import {
    Sparkles, Code, Lightbulb, BookOpen, ChevronDown, ChevronUp,
    Copy, Check, Zap, AlertTriangle, Loader2, ExternalLink
} from 'lucide-react';
import { getCodeSuggestions } from '../api';

const AIAssistantPanel = ({ taskId, isOpen, onToggle }) => {
    const [suggestions, setSuggestions] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [copiedIndex, setCopiedIndex] = useState(null);
    const [expandedHints, setExpandedHints] = useState({});

    useEffect(() => {
        if (isOpen && taskId && !suggestions) {
            fetchSuggestions();
        }
    }, [isOpen, taskId]);

    const fetchSuggestions = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await getCodeSuggestions(taskId);
            setSuggestions(data);
        } catch (err) {
            setError('Failed to load AI suggestions');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const copyToClipboard = (code, index) => {
        navigator.clipboard.writeText(code);
        setCopiedIndex(index);
        setTimeout(() => setCopiedIndex(null), 2000);
    };

    const toggleHint = (index) => {
        setExpandedHints(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    };

    const getComplexityColor = (level) => {
        switch (level) {
            case 'low': return 'text-green-400 bg-green-400/10';
            case 'medium': return 'text-yellow-400 bg-yellow-400/10';
            case 'high': return 'text-red-400 bg-red-400/10';
            default: return 'text-gray-400 bg-gray-400/10';
        }
    };

    if (!isOpen) {
        return (
            <button
                onClick={onToggle}
                className="w-full flex items-center gap-2 px-4 py-3 glass-card hover:border-purple-500/30 transition-all group"
            >
                <Sparkles className="w-5 h-5 text-purple-400 group-hover:animate-pulse" />
                <span className="text-sm font-medium text-gray-200">AI Assistant</span>
                <ChevronDown className="w-4 h-4 ml-auto text-gray-400" />
            </button>
        );
    }

    return (
        <div className="glass-card border-purple-500/20 overflow-hidden">
            {/* Header */}
            <button
                onClick={onToggle}
                className="w-full flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-purple-500/10 to-indigo-500/10 border-b border-white/5"
            >
                <Sparkles className="w-5 h-5 text-purple-400" />
                <span className="text-sm font-medium text-gray-200">AI Assistant</span>
                <ChevronUp className="w-4 h-4 ml-auto text-gray-400" />
            </button>

            {/* Content */}
            <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
                {loading && (
                    <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-6 h-6 text-purple-400 animate-spin" />
                        <span className="ml-2 text-sm text-gray-400">Analyzing task...</span>
                    </div>
                )}

                {error && (
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                        <AlertTriangle className="w-4 h-4 text-red-400" />
                        <span className="text-sm text-red-400">{error}</span>
                        <button
                            onClick={fetchSuggestions}
                            className="ml-auto text-xs text-red-400 hover:text-red-300 underline"
                        >
                            Retry
                        </button>
                    </div>
                )}

                {suggestions && !loading && (
                    <>
                        {/* Complexity Badge */}
                        {suggestions.complexity && (
                            <div className="flex items-center justify-between">
                                <span className="text-xs text-gray-400">Task Complexity</span>
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getComplexityColor(suggestions.complexity.level)}`}>
                                    {suggestions.complexity.level.toUpperCase()} • {suggestions.complexity.estimated_hours}
                                </span>
                            </div>
                        )}

                        {/* Categories */}
                        {suggestions.matched_categories?.length > 0 && (
                            <div className="flex flex-wrap gap-2">
                                {suggestions.matched_categories.map((cat, idx) => (
                                    <span
                                        key={idx}
                                        className="px-2 py-1 text-xs rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20"
                                    >
                                        {cat}
                                    </span>
                                ))}
                            </div>
                        )}

                        {/* Suggestions */}
                        {suggestions.suggestions?.length > 0 && (
                            <div className="space-y-2">
                                <div className="flex items-center gap-2 text-sm font-medium text-gray-300">
                                    <Lightbulb className="w-4 h-4 text-yellow-400" />
                                    Implementation Tips
                                </div>
                                <ul className="space-y-1">
                                    {suggestions.suggestions.map((tip, idx) => (
                                        <li key={idx} className="flex items-start gap-2 text-sm text-gray-400">
                                            <Zap className="w-3 h-3 mt-1 text-purple-400 flex-shrink-0" />
                                            {tip}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {/* Code Hints */}
                        {suggestions.code_hints?.length > 0 && (
                            <div className="space-y-2">
                                <div className="flex items-center gap-2 text-sm font-medium text-gray-300">
                                    <Code className="w-4 h-4 text-cyan-400" />
                                    Code Examples
                                </div>
                                {suggestions.code_hints.map((hint, idx) => (
                                    <div key={idx} className="rounded-lg bg-black/30 border border-white/5 overflow-hidden">
                                        <button
                                            onClick={() => toggleHint(idx)}
                                            className="w-full flex items-center justify-between px-3 py-2 hover:bg-white/5 transition-colors"
                                        >
                                            <span className="text-xs text-gray-300">{hint.title}</span>
                                            <span className="text-xs text-gray-500">{hint.language}</span>
                                        </button>
                                        {expandedHints[idx] && (
                                            <div className="relative">
                                                <pre className="p-3 text-xs overflow-x-auto text-gray-300 bg-black/20">
                                                    <code>{hint.code}</code>
                                                </pre>
                                                <button
                                                    onClick={() => copyToClipboard(hint.code, idx)}
                                                    className="absolute top-2 right-2 p-1.5 rounded-md bg-white/5 hover:bg-white/10 transition-colors"
                                                >
                                                    {copiedIndex === idx ? (
                                                        <Check className="w-3 h-3 text-green-400" />
                                                    ) : (
                                                        <Copy className="w-3 h-3 text-gray-400" />
                                                    )}
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Resources */}
                        {suggestions.resources?.length > 0 && (
                            <div className="space-y-2">
                                <div className="flex items-center gap-2 text-sm font-medium text-gray-300">
                                    <BookOpen className="w-4 h-4 text-blue-400" />
                                    Resources
                                </div>
                                <div className="space-y-1">
                                    {suggestions.resources.map((resource, idx) => (
                                        <a
                                            key={idx}
                                            href={resource.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group"
                                        >
                                            <span className="text-xs text-gray-400 group-hover:text-gray-300 flex-1 truncate">
                                                {resource.title}
                                            </span>
                                            <ExternalLink className="w-3 h-3 text-gray-500 group-hover:text-gray-400" />
                                        </a>
                                    ))}
                                </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    );
};

export default AIAssistantPanel;
