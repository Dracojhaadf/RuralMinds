import hashlib
import logging
from typing import Optional, Tuple
from core.database import get_db_connection, ensure_migrated

logger = logging.getLogger(__name__)

# Ensure migrations on module load just in case (safe due to IF NOT EXISTS and .bak)
ensure_migrated()

# --- PASSWORD HASHING (bcrypt with SHA-256 fallback for legacy accounts) ---

try:
    import bcrypt
    _BCRYPT_AVAILABLE = True
except ImportError:
    _BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed — falling back to SHA-256 (INSECURE). Install: pip install bcrypt")


def hash_password(password: str) -> str:
    """Hash a password using bcrypt (preferred) or SHA-256 (fallback)."""
    if _BCRYPT_AVAILABLE:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')
    # Fallback: unsalted SHA-256 (NOT recommended for production)
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash. Handles both bcrypt and legacy SHA-256."""
    if _BCRYPT_AVAILABLE and stored_hash.startswith('$2b$'):
        # bcrypt hash
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
    # Legacy SHA-256 hash (64 hex chars)
    return hashlib.sha256(password.encode()).hexdigest() == stored_hash


def _upgrade_hash_if_needed(username: str, password: str, stored_hash: str):
    """Auto-upgrade legacy SHA-256 hash to bcrypt on successful login."""
    if _BCRYPT_AVAILABLE and not stored_hash.startswith('$2b$'):
        new_hash = hash_password(password)
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('UPDATE users SET password = ? WHERE username = ?', (new_hash, username))
            conn.commit()
            conn.close()
            logger.info(f"Upgraded password hash for user '{username}' from SHA-256 to bcrypt")
        except Exception as e:
            logger.warning(f"Failed to upgrade password hash for '{username}': {e}")


def _ensure_admin():
    """Ensure the default administrator exists in the SQLite database."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username FROM users WHERE username = ?', ('admin',))
    if not c.fetchone():
        c.execute('''
            INSERT INTO users (username, password, role, name, email) 
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', hash_password("administrator"), 'admin', 'Administrator', 'admin@ruralminds.local'))
        conn.commit()
    conn.close()

# Keep admin check upon startup
_ensure_admin()

def create_user(username: str, password: str, role: str, name: str, email: str) -> Tuple[bool, str]:
    """
    Create a new user account.
    """
    if role not in ['teacher', 'student', 'admin']:
        return False, "Invalid role. Must be 'teacher', 'student', or 'admin'."
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters long."
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if c.fetchone():
        conn.close()
        return False, "Username already exists."
    
    try:
        c.execute('''
            INSERT INTO users (username, password, role, name, email)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hash_password(password), role, name, email))
        conn.commit()
        success = True
        msg = "Account created successfully!"
    except Exception as e:
        logger.error(f"Error saving user data: {str(e)}")
        success = False
        msg = "Error saving user data."
    finally:
        conn.close()
    
    return success, msg

def authenticate_user(username: str, password: str) -> Tuple[bool, Optional[dict]]:
    """
    Authenticate a user via SQLite. Auto-upgrades legacy SHA-256 hashes to bcrypt.
    """
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    
    if not user:
        logger.warning(f"Login attempt with non-existent username: {username}")
        return False, None
    
    if _verify_password(password, user['password']):
        # Auto-upgrade legacy hash
        _upgrade_hash_if_needed(username, password, user['password'])
        logger.info(f"Successful login: {username} ({user['role']})")
        return True, {
            'username': user['username'],
            'role': user['role'],
            'name': user['name'],
            'email': user['email']
        }
    
    logger.warning(f"Failed login attempt for user: {username}")
    return False, None

def get_user_role(username: str) -> Optional[str]:
    """Get the role of a user."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT role FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    
    if row:
        return row['role']
    return None

def change_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """Change user password."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT password FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    
    if not user:
        conn.close()
        return False, "User not found."
    
    if not _verify_password(old_password, user['password']):
        conn.close()
        return False, "Incorrect old password."
    
    if len(new_password) < 6:
        conn.close()
        return False, "New password must be at least 6 characters long."
    
    try:
        c.execute('UPDATE users SET password = ? WHERE username = ?', (hash_password(new_password), username))
        conn.commit()
        success = True
        msg = "Password changed successfully!"
    except Exception as e:
        logger.error(f"Error saving changes: {str(e)}")
        success = False
        msg = "Error saving changes."
    finally:
        conn.close()
        
    return success, msg

def get_all_users() -> list:
    """Get list of all users (admin only)."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('SELECT username, role, name, email FROM users')
    rows = c.fetchall()
    conn.close()
    
    return [
        {
            'username': r['username'],
            'role': r['role'],
            'name': r['name'],
            'email': r['email']
        } for r in rows
    ]

def delete_user(username: str) -> Tuple[bool, str]:
    """Delete a user account (admin only)."""
    if username == "admin" or username == "administrator":
        return False, "Cannot delete primary admin account."
        
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT username FROM users WHERE username = ?', (username,))
    if not c.fetchone():
        conn.close()
        return False, "User not found."
    
    try:
        c.execute('DELETE FROM users WHERE username = ?', (username,))
        conn.commit()
        logger.info(f"User deleted: {username}")
        success = True
        msg = f"User '{username}' deleted successfully."
    except Exception as e:
        logger.error(f"Error saving changes: {str(e)}")
        success = False
        msg = "Error saving changes."
    finally:
        conn.close()
        
    return success, msg
