import React, { useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-alpine.css"; // Use Alpine Light theme

const ListView = ({ tasks, users, onTaskClick }) => {
    // Helper to find user by ID
    const getUser = (id) => users.find(u => u.id === id);

    const columnDefs = useMemo(() => [
        {
            headerName: "",
            checkboxSelection: true,
            headerCheckboxSelection: true,
            width: 50,
            pinned: 'left',
            lockPosition: true
        },
        {
            field: "id",
            headerName: "Key",
            width: 100,
            cellRenderer: params => (
                <span className="text-primary hover:underline cursor-pointer font-medium">
                    KAN-{params.value}
                </span>
            )
        },
        {
            field: "title",
            headerName: "Summary",
            flex: 2,
            minWidth: 300,
            cellRenderer: params => (
                <div className="flex items-center space-x-2">
                    {/* Type Icon Placeholder */}
                    <div className="w-4 h-4 bg-blue-400 rounded-sm flex items-center justify-center text-[10px] text-white font-bold">✓</div>
                    <span className="text-foreground hover:text-primary cursor-pointer hover:underline">
                        {params.value}
                    </span>
                </div>
            )
        },
        {
            field: "assignee_id",
            headerName: "Assignee",
            width: 180,
            cellRenderer: params => {
                const user = getUser(params.value);
                return user ? (
                    <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 rounded-full bg-surface flex items-center justify-center text-[10px] font-bold text-muted">
                            {user.username.substring(0, 2).toUpperCase()}
                        </div>
                        <span>{user.username}</span>
                    </div>
                ) : (
                    <div className="flex items-center space-x-2 text-muted">
                        <div className="w-6 h-6 rounded-full bg-surface flex items-center justify-center">
                            <svg className="w-4 h-4 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
                        </div>
                        <span>Unassigned</span>
                    </div>
                );
            }
        },
        {
            field: "creator_id",
            headerName: "Reporter",
            width: 180,
            cellRenderer: params => {
                const user = getUser(params.value);
                return user ? (
                    <div className="flex items-center space-x-2">
                        <div className="w-6 h-6 rounded-full bg-orange-200 flex items-center justify-center text-[10px] font-bold text-orange-700">
                            {user.username.substring(0, 2).toUpperCase()}
                        </div>
                        <span>{user.username}</span>
                    </div>
                ) : <span>Unknown</span>;
            }
        },
        {
            field: "priority",
            headerName: "Priority",
            width: 120,
            cellRenderer: params => {
                const colors = {
                    'CRITICAL': 'text-red-600',
                    'HIGH': 'text-orange-600',
                    'MEDIUM': 'text-yellow-600',
                    'LOW': 'text-green-600'
                };
                // Simple icon placeholder
                return (
                    <div className="flex items-center space-x-2">
                        <span className={`font-medium ${colors[params.value] || 'text-muted'}`}>{params.value}</span>
                    </div>
                );
            }
        },
        {
            field: "status",
            headerName: "Status",
            width: 140,
            cellRenderer: params => {
                const styles = {
                    'TODO': 'bg-surface text-foreground',
                    'IN_PROGRESS': 'bg-blue-100 text-blue-700',
                    'IN_REVIEW': 'bg-purple-100 text-purple-700',
                    'DONE': 'bg-green-100 text-green-700'
                };
                return (
                    <span className={`px-2 py-0.5 rounded font-semibold text-xs uppercase ${styles[params.value] || 'bg-surface'}`}>
                        {params.value.replace('_', ' ')}
                    </span>
                );
            }
        },
        {
            headerName: "Resolution",
            width: 120,
            valueGetter: params => params.data.status === 'DONE' ? 'Done' : 'Unresolved'
        },
        {
            field: "created_at",
            headerName: "Created",
            width: 150,
            valueFormatter: params => new Date(params.value).toLocaleDateString() + ' ' + new Date(params.value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        },
        {
            field: "created_at", // Using created_at as updated for now since we don't track updated_at
            headerName: "Updated",
            width: 150,
            valueFormatter: params => new Date(params.value).toLocaleDateString() + ' ' + new Date(params.value).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        },
    ], [users]);

    const defaultColDef = useMemo(() => ({
        sortable: true,
        filter: true,
        resizable: true,
    }), []);

    return (
        <div className="ag-theme-alpine h-full w-full">
            <style>{`
                .ag-theme-alpine {
                    --ag-font-family: 'Inter', sans-serif;
                    --ag-font-size: 14px;
                    --ag-header-background-color: #F4F5F7;
                    --ag-header-foreground-color: #5E6C84;
                    --ag-border-color: #DFE1E6;
                    --ag-row-hover-color: #FAFBFC;
                    --ag-selected-row-background-color: #EBECF0;
                }
                .ag-header-cell-label {
                    font-weight: 600;
                }
            `}</style>
            <AgGridReact
                rowData={tasks}
                columnDefs={columnDefs}
                defaultColDef={defaultColDef}
                pagination={true}
                paginationPageSize={20}
                rowHeight={40}
                headerHeight={40}
                rowSelection="multiple"
                onRowClicked={(event) => onTaskClick(event.data)}
            />
        </div>
    );
};

export default ListView;
