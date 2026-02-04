"""
Migration script to set is_verified=True for all existing users
"""
import sys
sys.path.insert(0, 'backend')
import json_storage

data = json_storage.load_data()
updated = 0

for user in data.get('users', []):
    if 'is_verified' not in user or user['is_verified'] != True:
        user['is_verified'] = True
        updated += 1

json_storage.save_data(data)
print(f'Migrated {updated} existing users to is_verified=True')
