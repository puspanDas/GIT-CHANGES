import React from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { MoreHorizontal, Calendar, User as UserIcon } from 'lucide-react';

const KanbanBoard = ({ tasks, onTaskUpdate, onTaskClick }) => {
    const columns = {
        TODO: { id: 'TODO', title: 'To Do', color: 'bg-secondary', items: tasks.filter(t => t.status === 'TODO') },
        IN_PROGRESS: { id: 'IN_PROGRESS', title: 'In Progress', color: 'bg-primary', items: tasks.filter(t => t.status === 'IN_PROGRESS') },
        IN_REVIEW: { id: 'IN_REVIEW', title: 'In Review', color: 'bg-accent', items: tasks.filter(t => t.status === 'IN_REVIEW') },
        DONE: { id: 'DONE', title: 'Done', color: 'bg-success', items: tasks.filter(t => t.status === 'DONE') },
    };

    const onDragEnd = (result) => {
        if (!result.destination) return;
        // In a real app, update backend here
        console.log("Moved task", result.draggableId, "to", result.destination.droppableId);
    };

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'CRITICAL': return 'bg-danger/10 text-danger border-danger/20';
            case 'HIGH': return 'bg-warning/10 text-warning border-warning/20';
            case 'MEDIUM': return 'bg-primary/10 text-primary border-primary/20';
            case 'LOW': return 'bg-success/10 text-success border-success/20';
            default: return 'bg-secondary/10 text-muted border-secondary/20';
        }
    };

    return (
        <div className="flex h-full overflow-x-auto pb-4 space-x-6">
            <DragDropContext onDragEnd={onDragEnd}>
                {Object.entries(columns).map(([columnId, column]) => (
                    <div key={columnId} className="flex-shrink-0 w-80 flex flex-col h-full">
                        <div className="flex items-center justify-between mb-4 px-2">
                            <div className="flex items-center space-x-2">
                                <div className={`w-3 h-3 rounded-full ${column.color}`} />
                                <h3 className="font-semibold text-foreground">{column.title}</h3>
                                <span className="bg-surface border border-border px-2 py-0.5 rounded-full text-xs text-muted">{column.items.length}</span>
                            </div>
                            <button className="text-muted hover:text-foreground">
                                <MoreHorizontal className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="flex-1 bg-surface/50 rounded-xl p-3 border border-border/50">
                            <Droppable droppableId={columnId}>
                                {(provided) => (
                                    <div
                                        {...provided.droppableProps}
                                        ref={provided.innerRef}
                                        className="space-y-3 min-h-[200px]"
                                    >
                                        {column.items.map((task, index) => (
                                            <Draggable key={task.id} draggableId={String(task.id)} index={index}>
                                                {(provided, snapshot) => (
                                                    <div
                                                        ref={provided.innerRef}
                                                        {...provided.draggableProps}
                                                        {...provided.dragHandleProps}
                                                        onClick={() => onTaskClick(task)}
                                                        className={`card p-4 group hover:border-primary/50 transition-all cursor-pointer ${snapshot.isDragging ? 'shadow-2xl ring-2 ring-primary/20 rotate-2 bg-surface' : 'bg-surface'}`}
                                                    >
                                                        <div className="flex justify-between items-start mb-2">
                                                            <span className={`text-[10px] font-bold px-2 py-1 rounded border ${getPriorityColor(task.priority)}`}>
                                                                {task.priority}
                                                            </span>
                                                            <button className="opacity-0 group-hover:opacity-100 text-muted hover:text-foreground transition-opacity">
                                                                <MoreHorizontal className="w-4 h-4" />
                                                            </button>
                                                        </div>

                                                        <h4 className="font-medium text-foreground mb-2 line-clamp-2">{task.title}</h4>

                                                        <div className="flex items-center justify-between mt-4 pt-3 border-t border-border">
                                                            <div className="flex items-center space-x-2 text-muted text-xs">
                                                                <UserIcon className="w-4 h-4" />
                                                                <span>{task.assignee_id ? `User ${task.assignee_id}` : 'Unassigned'}</span>
                                                            </div>
                                                            {task.created_at && (
                                                                <div className="flex items-center space-x-1 text-muted text-xs">
                                                                    <Calendar className="w-3 h-3" />
                                                                    <span>{new Date(task.created_at).toLocaleDateString()}</span>
                                                                </div>
                                                            )}
                                                            {task.estimated_days > 0 && (
                                                                <div className="flex items-center space-x-1 text-primary text-xs font-medium bg-primary/10 px-2 py-0.5 rounded">
                                                                    <Calendar className="w-3 h-3" />
                                                                    <span>{task.estimated_days}d</span>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </div>
                                                )}
                                            </Draggable>
                                        ))}
                                        {provided.placeholder}
                                    </div>
                                )}
                            </Droppable>
                        </div>
                    </div>
                ))}
            </DragDropContext>
        </div>
    );
};

export default KanbanBoard;
