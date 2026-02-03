import React, { useState } from 'react';
import {
    X, Share2, MoreHorizontal, Link, Plus, ChevronDown, ChevronRight,
    CheckSquare, MessageSquare, History, Clock, Paperclip, Send,
    Layout, Calendar, User, Tag, Users, AlertCircle, CheckCircle2
} from 'lucide-react';
import AIAssistantPanel from './AIAssistantPanel';

const TaskDetailView = ({ task, onClose, onUpdate, users, comments, onAddComment, currentUser }) => {
    const [activeTab, setActiveTab] = useState("comments");
    const [newComment, setNewComment] = useState("");
    const [isDescriptionEditing, setIsDescriptionEditing] = useState(false);
    const [description, setDescription] = useState(task?.description || "");
    const [aiPanelOpen, setAiPanelOpen] = useState(false);
    const editorRef = React.useRef(null);

    const handleSaveDescription = () => {
        onUpdate({ ...task, description });
        setIsDescriptionEditing(false);
    };

    const priorityColors = {
        LOW: "bg-blue-100 text-blue-800",
        MEDIUM: "bg-orange-100 text-orange-800",
        HIGH: "bg-red-100 text-red-800",
        CRITICAL: "bg-red-200 text-red-900"
    };

    const statusColors = {
        TODO: "bg-gray-200 text-gray-700",
        IN_PROGRESS: "bg-blue-100 text-blue-700",
        IN_REVIEW: "bg-purple-100 text-purple-700",
        DONE: "bg-green-100 text-green-700"
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 overflow-y-auto py-10 select-none">
            <div className="bg-white w-full max-w-5xl h-[90vh] rounded-lg shadow-2xl flex flex-col overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white sticky top-0 z-10">
                    <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <CheckSquare className="w-4 h-4 text-blue-600" />
                        <span>KAN-{task.id}</span>
                        <span>/</span>
                        <span>{task.title}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                        <button className="p-2 hover:bg-gray-100 rounded text-gray-500"><Share2 className="w-4 h-4" /></button>
                        <button className="p-2 hover:bg-gray-100 rounded text-gray-500"><MoreHorizontal className="w-4 h-4" /></button>
                        <button onClick={onClose} className="p-2 hover:bg-gray-100 rounded text-gray-500 hover:text-red-500">
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                <div className="flex-1 flex overflow-hidden">
                    {/* Main Content (Left Column) */}
                    <div className="flex-1 overflow-y-auto p-8 scrollbar-thin scrollbar-thumb-gray-300">
                        <h1 className="text-2xl font-semibold text-gray-900 mb-6">{task.title}</h1>

                        {/* Description */}
                        <div className="mb-8 group">
                            <h3 className="text-sm font-semibold text-gray-900 mb-2">Description</h3>
                            {isDescriptionEditing ? (
                                <div className="space-y-2">
                                    <textarea
                                        value={description}
                                        onChange={(e) => setDescription(e.target.value)}
                                        className="w-full min-h-[100px] p-3 border border-blue-500 rounded-md focus:outline-none focus:ring-1 focus:ring-blue-500 text-sm"
                                        autoFocus
                                    />
                                    <div className="flex space-x-2">
                                        <button onClick={handleSaveDescription} className="px-3 py-1.5 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700">Save</button>
                                        <button onClick={() => setIsDescriptionEditing(false)} className="px-3 py-1.5 text-gray-600 hover:bg-gray-100 rounded text-sm font-medium">Cancel</button>
                                    </div>
                                </div>
                            ) : (
                                <div
                                    onClick={() => setIsDescriptionEditing(true)}
                                    className="min-h-[60px] p-2 -ml-2 rounded hover:bg-gray-100 cursor-text text-sm text-gray-700 whitespace-pre-wrap"
                                >
                                    {task.description || "Add a description..."}
                                </div>
                            )}
                        </div>

                        {/* Subtasks (Mocked) */}
                        <div className="mb-8">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="text-sm font-semibold text-gray-900">Subtasks</h3>
                                <div className="flex items-center space-x-2">
                                    <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                                        <div className="h-full bg-blue-600 w-1/3"></div>
                                    </div>
                                    <span className="text-xs text-gray-500">33% Done</span>
                                    <button className="p-1 hover:bg-gray-100 rounded"><Plus className="w-4 h-4 text-gray-500" /></button>
                                </div>
                            </div>
                            <div className="space-y-1">
                                <div className="flex items-center justify-between p-2 hover:bg-gray-50 rounded group border border-transparent hover:border-gray-200 transition-all">
                                    <div className="flex items-center space-x-3">
                                        <CheckSquare className="w-4 h-4 text-blue-400" />
                                        <span className="text-sm text-gray-500 line-through">KAN-{task.id + 1} Research requirements</span>
                                    </div>
                                    <div className="flex items-center space-x-4">
                                        <span className="px-2 py-0.5 bg-green-100 text-green-800 text-xs font-bold rounded uppercase">DONE</span>
                                    </div>
                                </div>
                                <div className="flex items-center justify-between p-2 hover:bg-gray-50 rounded group border border-transparent hover:border-gray-200 transition-all">
                                    <div className="flex items-center space-x-3">
                                        <CheckSquare className="w-4 h-4 text-blue-400" />
                                        <span className="text-sm text-gray-700">KAN-{task.id + 2} Draft implementation plan</span>
                                    </div>
                                    <div className="flex items-center space-x-4">
                                        <div className="w-6 h-6 rounded-full bg-orange-200 flex items-center justify-center text-xs font-bold text-orange-700">JD</div>
                                        <span className="px-2 py-0.5 bg-blue-100 text-blue-800 text-xs font-bold rounded uppercase">IN PROGRESS</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Linked Work Items (Mocked) */}
                        <div className="mb-8">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="text-sm font-semibold text-gray-900">Linked work items</h3>
                                <button className="p-1 hover:bg-gray-100 rounded"><Plus className="w-4 h-4 text-gray-500" /></button>
                            </div>
                            <div className="p-3 border border-dashed border-gray-300 rounded text-center">
                                <span className="text-sm text-gray-500 cursor-pointer hover:text-blue-600">+ Add linked work item</span>
                            </div>
                        </div>

                        {/* Activity */}
                        <div>
                            <div className="flex items-center space-x-4 border-b border-gray-200 mb-4">
                                <h3 className="text-sm font-semibold text-gray-900 pb-2">Activity</h3>
                                <div className="flex space-x-1">
                                    <button
                                        onClick={() => setActiveTab("comments")}
                                        className={`px-3 py-1 text-sm font-medium rounded-t border-b-2 transition-colors ${activeTab === "comments" ? "border-blue-600 text-blue-600 bg-gray-50" : "border-transparent text-gray-500 hover:bg-gray-50"}`}
                                    >
                                        Comments
                                    </button>
                                    <button
                                        onClick={() => setActiveTab("history")}
                                        className={`px-3 py-1 text-sm font-medium rounded-t border-b-2 transition-colors ${activeTab === "history" ? "border-blue-600 text-blue-600 bg-gray-50" : "border-transparent text-gray-500 hover:bg-gray-50"}`}
                                    >
                                        History
                                    </button>
                                </div>
                            </div>

                            <div className="flex space-x-3 mb-6">
                                <div className="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-white font-bold text-xs">
                                    {currentUser?.username?.substring(0, 2).toUpperCase() || "ME"}
                                </div>
                                <div className="flex-1">
                                    <div className="border border-gray-200 rounded-md overflow-hidden focus-within:ring-1 focus-within:ring-blue-500 focus-within:border-blue-500 transition-all bg-white">
                                        {/* Rich Text Toolbar */}
                                        <div className="flex items-center space-x-1 p-2 border-b border-gray-200 bg-gray-50">
                                            <button
                                                onClick={() => document.execCommand('bold', false, null)}
                                                className="p-1.5 hover:bg-gray-200 rounded text-gray-700 font-bold"
                                                title="Bold"
                                            >
                                                B
                                            </button>
                                            <button
                                                onClick={() => document.execCommand('italic', false, null)}
                                                className="p-1.5 hover:bg-gray-200 rounded text-gray-700 italic"
                                                title="Italic"
                                            >
                                                I
                                            </button>
                                            <div className="h-4 w-px bg-gray-300 mx-1"></div>
                                            <input
                                                type="color"
                                                onChange={(e) => document.execCommand('foreColor', false, e.target.value)}
                                                className="w-6 h-6 p-0 border-0 rounded cursor-pointer"
                                                title="Text Color"
                                            />
                                        </div>

                                        {/* Editor Area */}
                                        <div
                                            ref={editorRef}
                                            contentEditable={true}
                                            onInput={(e) => setNewComment(e.currentTarget.innerHTML)}
                                            className="w-full p-3 text-sm focus:outline-none min-h-[80px] text-gray-900"
                                            style={{ minHeight: '80px' }}
                                        />

                                        {newComment && (
                                            <div className="flex items-center justify-between px-2 py-2 bg-gray-50 border-t border-gray-200">
                                                <div className="flex space-x-1">
                                                    <button className="p-1.5 hover:bg-gray-200 rounded text-gray-500"><Paperclip className="w-4 h-4" /></button>
                                                </div>
                                                <button
                                                    onClick={() => {
                                                        onAddComment(newComment);
                                                        setNewComment("");
                                                        if (editorRef.current) {
                                                            editorRef.current.innerHTML = "";
                                                        }
                                                    }}
                                                    className="px-3 py-1 bg-blue-600 text-white text-xs font-medium rounded hover:bg-blue-700"
                                                >
                                                    Save
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                    <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
                                        <span className="font-medium text-gray-700">Pro tip:</span>
                                        <span>Use the toolbar to format your text</span>
                                    </div>
                                </div>
                            </div>

                            <div className="space-y-6">
                                {(comments || []).map(comment => (
                                    <div key={comment.id} className="flex space-x-3 group">
                                        <div className="w-8 h-8 rounded-full bg-gray-200 flex-shrink-0 flex items-center justify-center text-xs font-bold text-gray-600">
                                            {comment.username.substring(0, 2).toUpperCase()}
                                        </div>
                                        <div className="flex-1">
                                            <div className="flex items-center space-x-2 mb-1">
                                                <span className="font-semibold text-sm text-gray-900">{comment.username}</span>
                                                <span className="text-xs text-gray-500">{new Date(comment.created_at).toLocaleString()}</span>
                                            </div>
                                            <div className="text-sm text-gray-800 whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: comment.content }} />
                                            <div className="flex items-center space-x-3 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                <button className="text-xs text-gray-500 hover:underline">Edit</button>
                                                <button className="text-xs text-gray-500 hover:underline">Delete</button>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Sidebar (Right Column) */}
                    <div className="w-80 border-l border-gray-200 bg-gray-50/50 p-6 overflow-y-auto">
                        <div className="mb-6">
                            <select
                                value={task.status}
                                onChange={(e) => onUpdate({ ...task, status: e.target.value })}
                                className={`w-full p-2 rounded font-semibold text-sm border-transparent focus:ring-2 focus:ring-blue-500 cursor-pointer ${statusColors[task.status] || "bg-gray-200"}`}
                            >
                                <option value="TODO">To Do</option>
                                <option value="IN_PROGRESS">In Progress</option>
                                <option value="IN_REVIEW">In Review</option>
                                <option value="DONE">Done</option>
                            </select>
                        </div>

                        <div className="space-y-6">
                            <div className="border border-gray-200 rounded bg-white p-4 shadow-sm">
                                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">Details</h3>

                                <div className="space-y-4">
                                    <div className="grid grid-cols-3 gap-2 items-center">
                                        <span className="text-sm text-gray-500">Assignee</span>
                                        <div className="col-span-2">
                                            <div className="flex items-center space-x-2 p-1 hover:bg-gray-100 rounded cursor-pointer group relative">
                                                <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center text-blue-700 text-xs font-bold">
                                                    {users.find(u => u.id === task.assignee_id)?.username.substring(0, 2).toUpperCase() || <User className="w-4 h-4" />}
                                                </div>
                                                <select
                                                    value={task.assignee_id || ""}
                                                    onChange={(e) => onUpdate({ ...task, assignee_id: e.target.value ? parseInt(e.target.value) : null })}
                                                    className="absolute inset-0 opacity-0 cursor-pointer"
                                                >
                                                    <option value="">Unassigned</option>
                                                    {users.map(u => (
                                                        <option key={u.id} value={u.id}>{u.username}</option>
                                                    ))}
                                                </select>
                                                <span className="text-sm text-blue-600 group-hover:underline truncate">
                                                    {users.find(u => u.id === task.assignee_id)?.username || "Unassigned"}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 items-center">
                                        <span className="text-sm text-gray-500">Priority</span>
                                        <div className="col-span-2">
                                            <div className="flex items-center space-x-2 p-1 hover:bg-gray-100 rounded cursor-pointer relative">
                                                <select
                                                    value={task.priority}
                                                    onChange={(e) => onUpdate({ ...task, priority: e.target.value })}
                                                    className="absolute inset-0 opacity-0 cursor-pointer"
                                                >
                                                    <option value="LOW">Low</option>
                                                    <option value="MEDIUM">Medium</option>
                                                    <option value="HIGH">High</option>
                                                    <option value="CRITICAL">Critical</option>
                                                </select>
                                                <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${priorityColors[task.priority]}`}>
                                                    {task.priority}
                                                </span>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 items-center">
                                        <span className="text-sm text-gray-500">Parent</span>
                                        <div className="col-span-2 text-sm text-gray-400 hover:text-gray-600 cursor-pointer">
                                            Add parent
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 items-center">
                                        <span className="text-sm text-gray-500">Due date</span>
                                        <div className="col-span-2 text-sm text-gray-400 hover:text-gray-600 cursor-pointer flex items-center">
                                            <Calendar className="w-4 h-4 mr-2" />
                                            <span>Dec 15, 2025</span>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 items-center">
                                        <span className="text-sm text-gray-500">Labels</span>
                                        <div className="col-span-2 text-sm text-gray-400 hover:text-gray-600 cursor-pointer">
                                            Add labels
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 items-center">
                                        <span className="text-sm text-gray-500">Team</span>
                                        <div className="col-span-2 text-sm text-gray-400 hover:text-gray-600 cursor-pointer">
                                            Add team
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-3 gap-2 items-center">
                                        <span className="text-sm text-gray-500">Reporter</span>
                                        <div className="col-span-2 flex items-center space-x-2">
                                            <div className="w-6 h-6 rounded-full bg-gray-300 flex items-center justify-center text-xs font-bold">PD</div>
                                            <span className="text-sm text-gray-700">Puspan Das</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* AI Assistant Panel */}
                            <div className="mt-4">
                                <AIAssistantPanel
                                    taskId={task.id}
                                    isOpen={aiPanelOpen}
                                    onToggle={() => setAiPanelOpen(!aiPanelOpen)}
                                />
                            </div>

                            <div className="text-xs text-gray-400 pt-4 border-t border-gray-200">
                                <p>Created {new Date(task.created_at).toLocaleDateString()}</p>
                                <p>Updated just now</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TaskDetailView;
