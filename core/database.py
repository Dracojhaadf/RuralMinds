import sqlite3
import json
import os
import logging
from datetime import datetime
from config.settings import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

DB_PATH = SQLITE_DB_PATH

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # 1. Users Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            name TEXT,
            email TEXT
        )
    ''')
    
    # 2. Forum Posts Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS forum_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            title TEXT,
            content TEXT,
            category TEXT,
            related_document TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            status TEXT,
            upvotes INTEGER DEFAULT 0,
            FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
        )
    ''')
    
    # 3. Forum Replies Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS forum_replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER,
            username TEXT,
            content TEXT,
            created_at TIMESTAMP,
            is_answer BOOLEAN,
            FOREIGN KEY(post_id) REFERENCES forum_posts(id) ON DELETE CASCADE,
            FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
        )
    ''')
    
    # 4. Documents Table for PDF tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            upload_path TEXT,
            uploaded_by TEXT,
            uploaded_at TIMESTAMP,
            FOREIGN KEY(uploaded_by) REFERENCES users(username) ON DELETE CASCADE
        )
    ''')
    
    # 5. Videos Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            name TEXT,
            video_path TEXT,
            caption_path TEXT,
            uploaded_by TEXT,
            uploaded_at TIMESTAMP,
            FOREIGN KEY(uploaded_by) REFERENCES users(username) ON DELETE CASCADE
        )
    ''')
    # Seed default 'teacher' user to satisfy foreign key constraints for automated scripts/defaults
    try:
        import bcrypt
        default_pwd = bcrypt.hashpw(b"placeholder", bcrypt.gensalt()).decode('utf-8')
    except Exception:
        import hashlib
        default_pwd = hashlib.sha256(b"placeholder").hexdigest()
        
    c.execute('''
        INSERT OR IGNORE INTO users (username, password, role, name, email)
        VALUES (?, ?, ?, ?, ?)
    ''', ('teacher', default_pwd, 'teacher', 'Default Teacher', 'teacher@ruralminds.local'))

    conn.commit()
    conn.close()

def _migrate_users():
    if os.path.exists("users_db.json"):
        logger.info("Migrating users_db.json to SQLite database...")
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            with open("users_db.json", "r") as f:
                users_data = json.load(f)
            
            for username, data in users_data.items():
                c.execute('''
                    INSERT OR IGNORE INTO users (username, password, role, name, email)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, data.get('password'), data.get('role'), data.get('name'), data.get('email')))
            
            conn.commit()
            
            # Rename the file so it doesn't migrate again
            os.rename("users_db.json", "users_db.json.bak")
            logger.info("Migration successful: users_db.json.bak created.")
        except Exception as e:
            logger.error(f"Failed to migrate users: {e}")
        finally:
            conn.close()

def _migrate_forum():
    if os.path.exists("forum_db.json"):
        logger.info("Migrating forum_db.json to SQLite database...")
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            with open("forum_db.json", "r", encoding='utf-8') as f:
                forum_data = json.load(f)
            
            posts = forum_data.get('posts', [])
            for post in posts:
                c.execute('''
                    INSERT OR IGNORE INTO forum_posts 
                    (id, username, title, content, category, related_document, created_at, updated_at, status, upvotes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    post['id'], post['username'], 
                    post['title'], post['content'], post.get('category', 'General'), 
                    post.get('related_document'), post.get('created_at', datetime.now().isoformat()), 
                    post.get('updated_at', datetime.now().isoformat()), post.get('status', 'open'), 
                    post.get('upvotes', 0)
                ))
                
                # Insert replies
                for reply in post.get('replies', []):
                    c.execute('''
                        INSERT INTO forum_replies
                        (post_id, username, content, created_at, is_answer)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        post['id'], reply['username'], 
                        reply['content'], reply.get('created_at', datetime.now().isoformat()), 
                        reply.get('is_answer', False)
                    ))
            
            conn.commit()
            
            # Rename the file
            os.rename("forum_db.json", "forum_db.json.bak")
            logger.info("Migration successful: forum_db.json.bak created.")
        except Exception as e:
            logger.error(f"Failed to migrate forum: {e}")
        finally:
            conn.close()

def migrate_to_bcnf():
    """Migrate existing schema to BCNF (removes obsolete columns safely)."""
    conn = sqlite3.connect(DB_PATH) # connect without foreign_keys=ON during migration
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        # Check if forum_posts has user_role
        c.execute("PRAGMA table_info(forum_posts)")
        columns = [row['name'] for row in c.fetchall()]
        if 'user_role' in columns:
            logger.info("Migrating to BCNF: Updating forum_posts...")
            c.execute('BEGIN TRANSACTION')
            c.execute('''
                CREATE TABLE forum_posts_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    title TEXT,
                    content TEXT,
                    category TEXT,
                    related_document TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    status TEXT,
                    upvotes INTEGER DEFAULT 0,
                    FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
                )
            ''')
            c.execute('''
                INSERT INTO forum_posts_new 
                (id, username, title, content, category, related_document, created_at, updated_at, status, upvotes)
                SELECT id, username, title, content, category, related_document, created_at, updated_at, status, upvotes 
                FROM forum_posts
            ''')
            c.execute('DROP TABLE forum_posts')
            c.execute('ALTER TABLE forum_posts_new RENAME TO forum_posts')
            c.execute('COMMIT')

        # Check if forum_replies has user_role
        c.execute("PRAGMA table_info(forum_replies)")
        columns = [row['name'] for row in c.fetchall()]
        if 'user_role' in columns:
            logger.info("Migrating to BCNF: Updating forum_replies...")
            c.execute('BEGIN TRANSACTION')
            c.execute('''
                CREATE TABLE forum_replies_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    username TEXT,
                    content TEXT,
                    created_at TIMESTAMP,
                    is_answer BOOLEAN,
                    FOREIGN KEY(post_id) REFERENCES forum_posts(id) ON DELETE CASCADE,
                    FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
                )
            ''')
            c.execute('''
                INSERT INTO forum_replies_new 
                (id, post_id, username, content, created_at, is_answer)
                SELECT id, post_id, username, content, created_at, is_answer
                FROM forum_replies
            ''')
            c.execute('DROP TABLE forum_replies')
            c.execute('ALTER TABLE forum_replies_new RENAME TO forum_replies')
            c.execute('COMMIT')

        # Check if videos has has_captions
        c.execute("PRAGMA table_info(videos)")
        columns = [row['name'] for row in c.fetchall()]
        if 'has_captions' in columns:
            logger.info("Migrating to BCNF: Updating videos...")
            c.execute('BEGIN TRANSACTION')
            c.execute('''
                CREATE TABLE videos_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    name TEXT,
                    video_path TEXT,
                    caption_path TEXT,
                    uploaded_by TEXT,
                    uploaded_at TIMESTAMP,
                    FOREIGN KEY(uploaded_by) REFERENCES users(username) ON DELETE CASCADE
                )
            ''')
            c.execute('''
                INSERT INTO videos_new 
                (id, filename, name, video_path, caption_path, uploaded_by, uploaded_at)
                SELECT id, filename, name, video_path, caption_path, uploaded_by, uploaded_at
                FROM videos
            ''')
            c.execute('DROP TABLE videos')
            c.execute('ALTER TABLE videos_new RENAME TO videos')
            c.execute('COMMIT')
            
        # Migrate documents to add FOREIGN KEY
        c.execute("PRAGMA foreign_key_list(documents)")
        if not c.fetchall():
            logger.info("Migrating to BCNF: Updating documents...")
            c.execute('BEGIN TRANSACTION')
            c.execute('''
                CREATE TABLE documents_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    upload_path TEXT,
                    uploaded_by TEXT,
                    uploaded_at TIMESTAMP,
                    FOREIGN KEY(uploaded_by) REFERENCES users(username) ON DELETE CASCADE
                )
            ''')
            c.execute('''
                INSERT INTO documents_new 
                (id, filename, upload_path, uploaded_by, uploaded_at)
                SELECT id, filename, upload_path, uploaded_by, uploaded_at
                FROM documents
            ''')
            c.execute('DROP TABLE documents')
            c.execute('ALTER TABLE documents_new RENAME TO documents')
            c.execute('COMMIT')

    except Exception as e:
        logger.error(f"Error migrating to BCNF: {e}")
        conn.rollback()
    finally:
        conn.close()

def ensure_migrated():
    """Run all schema setups and migrations securely."""
    init_db()
    migrate_to_bcnf()
    _migrate_users()
    _migrate_forum()
