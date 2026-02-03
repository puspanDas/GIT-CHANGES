import React, { useState } from 'react';
import { ChevronLeft, ChevronRight, Calendar as CalendarIcon } from 'lucide-react';

const CalendarView = ({ tasks, onTaskClick }) => {
    const [currentDate, setCurrentDate] = useState(new Date());

    const getDaysInMonth = (date) => {
        const year = date.getFullYear();
        const month = date.getMonth();
        const days = new Date(year, month + 1, 0).getDate();
        return days;
    };

    const getFirstDayOfMonth = (date) => {
        const year = date.getFullYear();
        const month = date.getMonth();
        return new Date(year, month, 1).getDay();
    };

    const prevMonth = () => {
        setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() - 1, 1));
    };

    const nextMonth = () => {
        setCurrentDate(new Date(currentDate.getFullYear(), currentDate.getMonth() + 1, 1));
    };

    const renderCalendar = () => {
        const daysInMonth = getDaysInMonth(currentDate);
        const firstDay = getFirstDayOfMonth(currentDate);
        const days = [];

        // Empty cells for days before the first day of the month
        for (let i = 0; i < firstDay; i++) {
            days.push(<div key={`empty-${i}`} className="h-32 border border-border bg-surface/30"></div>);
        }

        // Days of the month
        for (let day = 1; day <= daysInMonth; day++) {
            const date = new Date(currentDate.getFullYear(), currentDate.getMonth(), day);
            const dateString = date.toISOString().split('T')[0];

            // Filter tasks for this day (using created_at as a proxy for now, or estimated_days logic if available)
            // Ideally tasks would have a 'due_date' or 'start_date'. For now, let's assume we show tasks created on this day
            // OR we can just distribute them for demo if no date field exists.
            // Let's check task structure. It has created_at.

            const dayTasks = tasks.filter(task => {
                if (!task.created_at) return false;
                const taskDate = new Date(task.created_at).toISOString().split('T')[0];
                return taskDate === dateString;
            });

            days.push(
                <div key={day} className="h-32 border border-border p-2 hover:bg-surface/50 transition-colors overflow-y-auto">
                    <div className="text-sm font-medium text-muted mb-1">{day}</div>
                    <div className="space-y-1">
                        {dayTasks.map(task => (
                            <div
                                key={task.id}
                                onClick={() => onTaskClick(task)}
                                className="text-xs bg-white border border-border p-1 rounded shadow-sm truncate cursor-pointer hover:border-primary"
                            >
                                <span className={`inline-block w-2 h-2 rounded-full mr-1 ${task.priority === 'HIGH' || task.priority === 'CRITICAL' ? 'bg-danger' :
                                    task.priority === 'MEDIUM' ? 'bg-warning' : 'bg-success'
                                    }`}></span>
                                {task.title}
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        return days;
    };

    const monthNames = ["January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ];

    return (
        <div className="flex flex-col h-full bg-white rounded-lg shadow-sm border border-border">
            <div className="flex items-center justify-between p-4 border-b border-border">
                <div className="flex items-center space-x-4">
                    <h2 className="text-lg font-semibold text-foreground flex items-center">
                        <CalendarIcon className="w-5 h-5 mr-2 text-primary" />
                        {monthNames[currentDate.getMonth()]} {currentDate.getFullYear()}
                    </h2>
                    <div className="flex items-center space-x-1 bg-surface rounded-md p-0.5 border border-border">
                        <button onClick={prevMonth} className="p-1 hover:bg-white rounded shadow-sm transition-all">
                            <ChevronLeft className="w-4 h-4 text-muted" />
                        </button>
                        <button onClick={nextMonth} className="p-1 hover:bg-white rounded shadow-sm transition-all">
                            <ChevronRight className="w-4 h-4 text-muted" />
                        </button>
                    </div>
                </div>
                <button
                    onClick={() => setCurrentDate(new Date())}
                    className="text-sm font-medium text-primary hover:underline"
                >
                    Today
                </button>
            </div>

            <div className="grid grid-cols-7 border-b border-border bg-surface/50">
                {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                    <div key={day} className="py-2 text-center text-xs font-semibold text-muted uppercase tracking-wider">
                        {day}
                    </div>
                ))}
            </div>

            <div className="grid grid-cols-7 flex-1 overflow-y-auto">
                {renderCalendar()}
            </div>
        </div>
    );
};

export default CalendarView;
