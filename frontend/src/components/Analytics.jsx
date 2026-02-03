import React, { useState, useEffect } from 'react';
import { getKPIs, getTeamPerformance, getAIInsights, getTeamHealth } from '../api';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, Users, AlertTriangle, CheckCircle, Clock, Target, Zap, Activity, Brain } from 'lucide-react';

const Analytics = () => {
    const [kpis, setKpis] = useState(null);
    const [teamPerformance, setTeamPerformance] = useState([]);
    const [insights, setInsights] = useState(null);
    const [teamHealth, setTeamHealth] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchAnalyticsData();
    }, []);

    const fetchAnalyticsData = async () => {
        setLoading(true);
        setError(null);
        try {
            const [kpiData, teamData, insightsData, healthData] = await Promise.all([
                getKPIs(),
                getTeamPerformance(),
                getAIInsights(),
                getTeamHealth()
            ]);

            setKpis(kpiData);
            setTeamPerformance(teamData);
            setInsights(insightsData);
            setTeamHealth(healthData);
        } catch (err) {
            console.error('Failed to fetch analytics:', err);
            setError(err.response?.data?.detail || 'Failed to load analytics data');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-full">
                <div className="text-center">
                    <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-muted">Loading analytics...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center h-full">
                <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
                    <AlertTriangle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                    <h3 className="text-lg font-semibold text-red-800 mb-2">Access Denied</h3>
                    <p className="text-red-600">{error}</p>
                </div>
            </div>
        );
    }

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

    // Prepare chart data
    const statusData = kpis?.completion_rate ? [
        { name: 'Completed', value: kpis.completion_rate.completed },
        { name: 'In Progress', value: kpis.completion_rate.in_progress },
        { name: 'To Do', value: kpis.completion_rate.todo },
        { name: 'In Review', value: kpis.completion_rate.in_review },
    ] : [];

    return (
        <div className="h-full overflow-y-auto bg-background">
            <div className="max-w-7xl mx-auto py-6 px-4">
                {/* Header */}
                <div className="mb-8">
                    <div className="flex items-center space-x-3 mb-2">
                        <Activity className="w-8 h-8 text-primary" />
                        <h1 className="text-3xl font-bold text-foreground">Analytics Dashboard</h1>
                    </div>
                    <p className="text-muted">AI-powered insights and team performance metrics</p>
                </div>

                {/* Team Health Score */}
                {teamHealth && (
                    <div className="mb-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-6 text-white shadow-lg">
                        <div className="flex items-center justify-between">
                            <div>
                                <h2 className="text-2xl font-bold mb-2">Team Health Score</h2>
                                <p className="text-white/90">Overall team performance and wellbeing</p>
                            </div>
                            <div className="text-center">
                                <div className="text-6xl font-bold mb-2">{teamHealth.health_score}</div>
                                <div className="text-lg uppercase font-semibold">{teamHealth.status}</div>
                            </div>
                        </div>
                        {teamHealth.insights && teamHealth.insights.length > 0 && (
                            <div className="mt-4 pt-4 border-t border-white/20">
                                <p className="font-semibold mb-2">Key Insights:</p>
                                <ul className="space-y-1">
                                    {teamHealth.insights.slice(0, 3).map((insight, idx) => (
                                        <li key={idx} className="flex items-start space-x-2">
                                            <span className="text-white/80">•</span>
                                            <span className="text-white/90">{insight}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}

                {/* KPI Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    {/* Velocity Card */}
                    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-4">
                            <TrendingUp className="w-8 h-8 text-blue-500" />
                            <span className="text-xs font-semibold text-muted uppercase">Velocity</span>
                        </div>
                        <div className="text-3xl font-bold text-foreground mb-1">
                            {kpis?.velocity?.velocity_per_week || 0}
                        </div>
                        <p className="text-sm text-muted">tasks/week</p>
                    </div>

                    {/* Completion Rate Card */}
                    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-4">
                            <CheckCircle className="w-8 h-8 text-green-500" />
                            <span className="text-xs font-semibold text-muted uppercase">Completion</span>
                        </div>
                        <div className="text-3xl font-bold text-foreground mb-1">
                            {kpis?.completion_rate?.completion_rate_percent || 0}%
                        </div>
                        <p className="text-sm text-muted">completion rate</p>
                    </div>

                    {/* Cycle Time Card */}
                    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-4">
                            <Clock className="w-8 h-8 text-orange-500" />
                            <span className="text-xs font-semibold text-muted uppercase">Cycle Time</span>
                        </div>
                        <div className="text-3xl font-bold text-foreground mb-1">
                            {kpis?.cycle_time?.average_cycle_time_days || 0}
                        </div>
                        <p className="text-sm text-muted">days average</p>
                    </div>

                    {/* Estimation Accuracy Card */}
                    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm hover:shadow-md transition-shadow">
                        <div className="flex items-center justify-between mb-4">
                            <Target className="w-8 h-8 text-purple-500" />
                            <span className="text-xs font-semibold text-muted uppercase">Accuracy</span>
                        </div>
                        <div className="text-3xl font-bold text-foreground mb-1">
                            {kpis?.estimation_accuracy?.accuracy_percent || 0}%
                        </div>
                        <p className="text-sm text-muted">estimation accuracy</p>
                    </div>
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    {/* Team Performance Chart */}
                    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm">
                        <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                            <Users className="w-5 h-5 text-primary" />
                            <span>Team Performance</span>
                        </h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={teamPerformance}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="username" />
                                <YAxis />
                                <Tooltip />
                                <Legend />
                                <Bar dataKey="completed_tasks" fill="#8884d8" name="Completed" />
                                <Bar dataKey="in_progress_tasks" fill="#82ca9d" name="In Progress" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Task Status Distribution */}
                    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm">
                        <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                            <Activity className="w-5 h-5 text-primary" />
                            <span>Task Status Distribution</span>
                        </h3>
                        <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                                <Pie
                                    data={statusData}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={(entry) => `${entry.name}: ${entry.value}`}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    dataKey="value"
                                >
                                    {statusData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* AI Insights Panel */}
                {insights && (
                    <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg p-6 border border-blue-200 shadow-sm mb-8">
                        <div className="flex items-center space-x-3 mb-4">
                            <Brain className="w-6 h-6 text-purple-600" />
                            <h3 className="text-xl font-semibold text-purple-900">AI-Powered Insights</h3>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Sprint Prediction */}
                            {insights.sprint_prediction && (
                                <div className="bg-white rounded-lg p-4 border border-blue-200">
                                    <h4 className="font-semibold text-blue-900 mb-2">Sprint Completion Forecast</h4>
                                    <div className="flex items-center space-x-2 mb-2">
                                        {insights.sprint_prediction.will_complete_on_time ? (
                                            <CheckCircle className="w-5 h-5 text-green-500" />
                                        ) : (
                                            <AlertTriangle className="w-5 h-5 text-orange-500" />
                                        )}
                                        <span className="font-medium">
                                            {insights.sprint_prediction.will_complete_on_time ? 'On Track' : 'At Risk'}
                                        </span>
                                    </div>
                                    <p className="text-sm text-gray-600">
                                        {insights.sprint_prediction.remaining_tasks} tasks remaining,
                                        estimated {insights.sprint_prediction.estimated_days_needed} days needed
                                    </p>
                                </div>
                            )}

                            {/* Delivery Forecast */}
                            {insights.delivery_forecast && (
                                <div className="bg-white rounded-lg p-4 border border-blue-200">
                                    <h4 className="font-semibold text-blue-900 mb-2">Delivery Forecast</h4>
                                    <div className="text-2xl font-bold text-purple-600 mb-1">
                                        {insights.delivery_forecast.estimated_days_to_completion} days
                                    </div>
                                    <p className="text-sm text-gray-600">
                                        {insights.delivery_forecast.remaining_tasks} tasks remaining
                                    </p>
                                    <p className="text-xs text-gray-500 mt-1">
                                        Confidence: {insights.delivery_forecast.confidence}
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Bottlenecks */}
                        {insights.bottlenecks?.recommendations?.length > 0 && (
                            <div className="mt-4 bg-yellow-50 rounded-lg p-4 border border-yellow-200">
                                <h4 className="font-semibold text-yellow-900 mb-2 flex items-center space-x-2">
                                    <AlertTriangle className="w-5 h-5" />
                                    <span>Recommendations</span>
                                </h4>
                                <ul className="space-y-1">
                                    {insights.bottlenecks.recommendations.map((rec, idx) => (
                                        <li key={idx} className="text-sm text-yellow-800">• {rec}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                )}

                {/* Risks Panel */}
                {kpis?.risks && (
                    <div className="bg-surface rounded-lg p-6 border border-border shadow-sm">
                        <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                            <AlertTriangle className="w-5 h-5 text-orange-500" />
                            <span>Risk Detection</span>
                        </h3>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Overloaded Developers */}
                            {kpis.risks.overloaded_developers?.length > 0 && (
                                <div className="bg-orange-50 rounded-lg p-4 border border-orange-200">
                                    <h4 className="font-semibold text-orange-900 mb-2">Overloaded Team Members</h4>
                                    <ul className="space-y-2">
                                        {kpis.risks.overloaded_developers.map((dev, idx) => (
                                            <li key={idx} className="text-sm text-orange-800">
                                                <span className="font-medium">{dev.username}</span>: {dev.active_tasks} active tasks
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {/* High Priority Blocked */}
                            {kpis.risks.high_priority_blocked?.length > 0 && (
                                <div className="bg-red-50 rounded-lg p-4 border border-red-200">
                                    <h4 className="font-semibold text-red-900 mb-2">High Priority Tasks Blocked</h4>
                                    <ul className="space-y-2">
                                        {kpis.risks.high_priority_blocked.slice(0, 3).map((task, idx) => (
                                            <li key={idx} className="text-sm text-red-800">
                                                <span className="font-medium">{task.title}</span> ({task.priority})
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Analytics;
