import json
import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# User database file
USERS_DB_PATH = "users_db.json"

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> dict:
    """Load users from JSON file."""
    if not os.path.exists(USERS_DB_PATH):
        # Create default admin account
        default_users = {
            "admin": {
                "password": hash_password("administrator"),
                "role": "admin",
                "name": "Administrator",
                "email": "admin@nfs.local"
            }
        }
        save_users(default_users)
        logger.info("Created default admin account: admin/administrator")
        return default_users
    
    try:
        with open(USERS_DB_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading users: {str(e)}")
        return {}

def save_users(users: dict) -> bool:
    """Save users to JSON file."""
    try:
        with open(USERS_DB_PATH, 'w') as f:
            json.dump(users, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving users: {str(e)}")
        return False

def create_user(username: str, password: str, role: str, name: str, email: str) -> Tuple[bool, str]:
    """
    Create a new user account.
    
    Args:
        username: Unique username
        password: User password (will be hashed)
        role: Either 'teacher', 'student', or 'admin'
        name: Full name
        email: Email address
        
    Returns:
        Tuple of (success, message)
    """
    if role not in ['teacher', 'student', 'admin']:
        return False, "Invalid role. Must be 'teacher', 'student', or 'admin'."
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    
    users = load_users()
    
    if username in users:
        return False, "Username already exists."
    
    users[username] = {
        "password": hash_password(password),
        "role": role,
        "name": name,
        "email": email
    }
    
    if save_users(users):
        return True, "Account created successfully!"
    else:
        return False, "Error saving user data."

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[dict]]:
    """
    Authenticate a user.
    
    Returns:
        Tuple of (success, user_data)
        user_data contains: {'username', 'role', 'name', 'email'}
    """
    users = load_users()
    
    if username not in users:
        logger.warning(f"Login attempt with non-existent username: {username}")
        return False, None
    
    user = users[username]
    hashed_input = hash_password(password)
    
    if user['password'] == hashed_input:
        logger.info(f"Successful login: {username} ({user['role']})")
        return True, {
            'username': username,
            'role': user['role'],
            'name': user['name'],
            'email': user['email']
        }
    
    logger.warning(f"Failed login attempt for user: {username}")
    return False, None

def get_user_role(username: str) -> Optional[str]:
    """Get the role of a user."""
    users = load_users()
    if username in users:
        return users[username]['role']
    return None

def change_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """Change user password."""
    users = load_users()
    
    if username not in users:
        return False, "User not found."
    
    if users[username]['password'] != hash_password(old_password):
        return False, "Incorrect old password."
    
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters long."
    
    users[username]['password'] = hash_password(new_password)
    
    if save_users(users):
        return True, "Password changed successfully!"
    else:
        return False, "Error saving changes."

def get_all_users() -> list:
    """Get list of all users (admin only)."""
    users = load_users()
    return [
        {
            'username': username,
            'role': data['role'],
            'name': data['name'],
            'email': data['email']
        }
        for username, data in users.items()
    ]

def delete_user(username: str) -> Tuple[bool, str]:
    """Delete a user account (admin only)."""
    users = load_users()
    
    if username not in users:
        return False, "User not found."
    
    if username == "admin":
        return False, "Cannot delete admin account."
    
    del users[username]
    
    if save_users(users):
        logger.info(f"User deleted: {username}")
        return True, f"User '{username}' deleted successfully."
    else:
        return False, "Error saving changes."