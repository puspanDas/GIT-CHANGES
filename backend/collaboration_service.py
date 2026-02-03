"""
Real-time Collaboration Service
Manages WebSocket connections, presence tracking, cursor sharing, and live updates
"""
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import json
import asyncio


class ConnectionManager:
    """Manages WebSocket connections for real-time collaboration"""
    
    def __init__(self):
        # Active connections: {client_id: websocket}
        self.active_connections: Dict[str, Any] = {}
        
        # User presence: {user_id: {client_id, username, task_id, cursor_pos, last_active}}
        self.user_presence: Dict[int, Dict[str, Any]] = {}
        
        # Task viewers: {task_id: set of user_ids}
        self.task_viewers: Dict[int, Set[int]] = {}
        
        # Typing indicators: {task_id: {user_id: timestamp}}
        self.typing_indicators: Dict[int, Dict[int, datetime]] = {}
        
        # Edit locks: {task_id: {user_id, acquired_at}}
        self.edit_locks: Dict[int, Dict[str, Any]] = {}
        
        # Cursor positions: {task_id: {user_id: {x, y, element_id}}}
        self.cursor_positions: Dict[int, Dict[int, Dict[str, Any]]] = {}

    async def connect(self, websocket, client_id: str, user_data: Dict):
        """Register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        user_id = user_data.get('id')
        if user_id:
            self.user_presence[user_id] = {
                'client_id': client_id,
                'username': user_data.get('username', 'Anonymous'),
                'role': user_data.get('role', 'DEV'),
                'task_id': None,
                'cursor_pos': None,
                'last_active': datetime.utcnow().isoformat(),
                'status': 'online'
            }
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Find and remove user presence
        user_to_remove = None
        for user_id, presence in self.user_presence.items():
            if presence.get('client_id') == client_id:
                user_to_remove = user_id
                # Remove from task viewers
                task_id = presence.get('task_id')
                if task_id and task_id in self.task_viewers:
                    self.task_viewers[task_id].discard(user_id)
                break
        
        if user_to_remove:
            del self.user_presence[user_to_remove]
            # Release any locks held by this user
            self._release_user_locks(user_to_remove)
    
    def _release_user_locks(self, user_id: int):
        """Release all locks held by a user"""
        locks_to_remove = []
        for task_id, lock in self.edit_locks.items():
            if lock.get('user_id') == user_id:
                locks_to_remove.append(task_id)
        for task_id in locks_to_remove:
            del self.edit_locks[task_id]

    async def broadcast(self, message: Dict, exclude_client: str = None):
        """Send message to all connected clients"""
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            if client_id != exclude_client:
                try:
                    await websocket.send_json(message)
                except Exception:
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

    async def send_to_task_viewers(self, task_id: int, message: Dict, exclude_user: int = None):
        """Send message to all users viewing a specific task"""
        if task_id not in self.task_viewers:
            return
        
        for user_id in self.task_viewers[task_id]:
            if user_id != exclude_user:
                presence = self.user_presence.get(user_id)
                if presence:
                    client_id = presence.get('client_id')
                    if client_id in self.active_connections:
                        try:
                            await self.active_connections[client_id].send_json(message)
                        except Exception:
                            pass

    async def send_to_user(self, user_id: int, message: Dict):
        """Send message to a specific user"""
        presence = self.user_presence.get(user_id)
        if presence:
            client_id = presence.get('client_id')
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(message)
                except Exception:
                    pass

    # Presence Management
    def update_user_task(self, user_id: int, task_id: Optional[int]):
        """Update which task a user is currently viewing"""
        if user_id not in self.user_presence:
            return
        
        # Remove from old task
        old_task_id = self.user_presence[user_id].get('task_id')
        if old_task_id and old_task_id in self.task_viewers:
            self.task_viewers[old_task_id].discard(user_id)
        
        # Add to new task
        self.user_presence[user_id]['task_id'] = task_id
        self.user_presence[user_id]['last_active'] = datetime.utcnow().isoformat()
        
        if task_id:
            if task_id not in self.task_viewers:
                self.task_viewers[task_id] = set()
            self.task_viewers[task_id].add(user_id)
    
    def get_task_viewers(self, task_id: int) -> List[Dict]:
        """Get list of users viewing a specific task"""
        if task_id not in self.task_viewers:
            return []
        
        viewers = []
        for user_id in self.task_viewers[task_id]:
            presence = self.user_presence.get(user_id)
            if presence:
                viewers.append({
                    'user_id': user_id,
                    'username': presence.get('username'),
                    'role': presence.get('role'),
                    'cursor_pos': self.cursor_positions.get(task_id, {}).get(user_id)
                })
        return viewers
    
    def get_active_users(self) -> List[Dict]:
        """Get list of all active users"""
        return [
            {
                'user_id': user_id,
                'username': presence.get('username'),
                'role': presence.get('role'),
                'task_id': presence.get('task_id'),
                'last_active': presence.get('last_active'),
                'status': presence.get('status', 'online')
            }
            for user_id, presence in self.user_presence.items()
        ]

    # Cursor Tracking
    def update_cursor(self, user_id: int, task_id: int, cursor_data: Dict):
        """Update cursor position for a user on a task"""
        if task_id not in self.cursor_positions:
            self.cursor_positions[task_id] = {}
        
        self.cursor_positions[task_id][user_id] = {
            'x': cursor_data.get('x', 0),
            'y': cursor_data.get('y', 0),
            'element_id': cursor_data.get('element_id'),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Update last active
        if user_id in self.user_presence:
            self.user_presence[user_id]['last_active'] = datetime.utcnow().isoformat()
    
    def get_cursors(self, task_id: int, exclude_user: int = None) -> Dict[int, Dict]:
        """Get all cursor positions for a task"""
        if task_id not in self.cursor_positions:
            return {}
        
        cursors = {}
        for user_id, pos in self.cursor_positions[task_id].items():
            if user_id != exclude_user:
                presence = self.user_presence.get(user_id)
                cursors[user_id] = {
                    **pos,
                    'username': presence.get('username') if presence else 'Unknown'
                }
        return cursors

    # Typing Indicators
    def set_typing(self, user_id: int, task_id: int, is_typing: bool):
        """Set typing indicator for a user"""
        if task_id not in self.typing_indicators:
            self.typing_indicators[task_id] = {}
        
        if is_typing:
            self.typing_indicators[task_id][user_id] = datetime.utcnow()
        elif user_id in self.typing_indicators[task_id]:
            del self.typing_indicators[task_id][user_id]
    
    def get_typing_users(self, task_id: int) -> List[Dict]:
        """Get users currently typing on a task"""
        if task_id not in self.typing_indicators:
            return []
        
        typing_users = []
        cutoff = datetime.utcnow()
        
        for user_id, timestamp in list(self.typing_indicators[task_id].items()):
            # Remove stale typing indicators (older than 5 seconds)
            if (cutoff - timestamp).seconds > 5:
                del self.typing_indicators[task_id][user_id]
            else:
                presence = self.user_presence.get(user_id)
                if presence:
                    typing_users.append({
                        'user_id': user_id,
                        'username': presence.get('username')
                    })
        
        return typing_users

    # Edit Locks for Pair Programming
    def acquire_lock(self, user_id: int, task_id: int) -> Dict:
        """Try to acquire an edit lock on a task"""
        if task_id in self.edit_locks:
            existing_lock = self.edit_locks[task_id]
            if existing_lock['user_id'] != user_id:
                # Lock held by another user
                holder = self.user_presence.get(existing_lock['user_id'])
                return {
                    'success': False,
                    'locked_by': existing_lock['user_id'],
                    'locked_by_username': holder.get('username') if holder else 'Unknown',
                    'acquired_at': existing_lock['acquired_at']
                }
        
        # Acquire the lock
        self.edit_locks[task_id] = {
            'user_id': user_id,
            'acquired_at': datetime.utcnow().isoformat()
        }
        
        return {'success': True, 'task_id': task_id}
    
    def release_lock(self, user_id: int, task_id: int) -> Dict:
        """Release an edit lock on a task"""
        if task_id not in self.edit_locks:
            return {'success': True, 'message': 'No lock to release'}
        
        if self.edit_locks[task_id]['user_id'] != user_id:
            return {'success': False, 'error': 'Lock held by another user'}
        
        del self.edit_locks[task_id]
        return {'success': True, 'task_id': task_id}
    
    def get_lock_status(self, task_id: int) -> Optional[Dict]:
        """Get the lock status for a task"""
        if task_id not in self.edit_locks:
            return None
        
        lock = self.edit_locks[task_id]
        holder = self.user_presence.get(lock['user_id'])
        return {
            'locked': True,
            'user_id': lock['user_id'],
            'username': holder.get('username') if holder else 'Unknown',
            'acquired_at': lock['acquired_at']
        }


# Global connection manager instance
manager = ConnectionManager()


# Message handlers for WebSocket events
async def handle_message(websocket, client_id: str, user_id: int, message: Dict):
    """Process incoming WebSocket messages"""
    event_type = message.get('type')
    data = message.get('data', {})
    
    if event_type == 'join_task':
        # User started viewing a task
        task_id = data.get('task_id')
        manager.update_user_task(user_id, task_id)
        
        # Notify other viewers
        await manager.send_to_task_viewers(task_id, {
            'type': 'user_joined',
            'data': {
                'user_id': user_id,
                'username': manager.user_presence.get(user_id, {}).get('username'),
                'task_id': task_id
            }
        }, exclude_user=user_id)
        
        # Send current viewers to the joining user
        await manager.send_to_user(user_id, {
            'type': 'current_viewers',
            'data': {
                'task_id': task_id,
                'viewers': manager.get_task_viewers(task_id)
            }
        })
    
    elif event_type == 'leave_task':
        task_id = data.get('task_id')
        manager.update_user_task(user_id, None)
        
        # Notify other viewers
        await manager.send_to_task_viewers(task_id, {
            'type': 'user_left',
            'data': {
                'user_id': user_id,
                'task_id': task_id
            }
        })
    
    elif event_type == 'cursor_move':
        task_id = data.get('task_id')
        cursor_data = data.get('cursor', {})
        
        manager.update_cursor(user_id, task_id, cursor_data)
        
        # Broadcast cursor position to other viewers
        await manager.send_to_task_viewers(task_id, {
            'type': 'cursor_update',
            'data': {
                'user_id': user_id,
                'username': manager.user_presence.get(user_id, {}).get('username'),
                'cursor': cursor_data
            }
        }, exclude_user=user_id)
    
    elif event_type == 'typing_start':
        task_id = data.get('task_id')
        manager.set_typing(user_id, task_id, True)
        
        await manager.send_to_task_viewers(task_id, {
            'type': 'typing_indicator',
            'data': {
                'user_id': user_id,
                'username': manager.user_presence.get(user_id, {}).get('username'),
                'is_typing': True
            }
        }, exclude_user=user_id)
    
    elif event_type == 'typing_stop':
        task_id = data.get('task_id')
        manager.set_typing(user_id, task_id, False)
        
        await manager.send_to_task_viewers(task_id, {
            'type': 'typing_indicator',
            'data': {
                'user_id': user_id,
                'is_typing': False
            }
        }, exclude_user=user_id)
    
    elif event_type == 'task_update':
        # Real-time task field update
        task_id = data.get('task_id')
        field = data.get('field')
        value = data.get('value')
        
        await manager.send_to_task_viewers(task_id, {
            'type': 'task_field_update',
            'data': {
                'user_id': user_id,
                'username': manager.user_presence.get(user_id, {}).get('username'),
                'task_id': task_id,
                'field': field,
                'value': value
            }
        }, exclude_user=user_id)
    
    elif event_type == 'acquire_lock':
        task_id = data.get('task_id')
        result = manager.acquire_lock(user_id, task_id)
        
        await manager.send_to_user(user_id, {
            'type': 'lock_result',
            'data': result
        })
        
        if result['success']:
            # Notify others that task is now locked
            await manager.send_to_task_viewers(task_id, {
                'type': 'task_locked',
                'data': {
                    'task_id': task_id,
                    'locked_by': user_id,
                    'username': manager.user_presence.get(user_id, {}).get('username')
                }
            }, exclude_user=user_id)
    
    elif event_type == 'release_lock':
        task_id = data.get('task_id')
        result = manager.release_lock(user_id, task_id)
        
        if result['success']:
            await manager.send_to_task_viewers(task_id, {
                'type': 'task_unlocked',
                'data': {'task_id': task_id}
            })
    
    elif event_type == 'ping':
        # Keep-alive ping
        await manager.send_to_user(user_id, {'type': 'pong'})


# Helper functions for REST API integration
def get_active_users_list() -> List[Dict]:
    """Get list of currently active users"""
    return manager.get_active_users()


def get_task_viewers_list(task_id: int) -> List[Dict]:
    """Get list of users viewing a task"""
    return manager.get_task_viewers(task_id)


def get_collaboration_stats() -> Dict:
    """Get overall collaboration statistics"""
    return {
        'active_connections': len(manager.active_connections),
        'active_users': len(manager.user_presence),
        'active_tasks': len(manager.task_viewers),
        'active_locks': len(manager.edit_locks)
    }
