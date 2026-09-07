import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import logging
from core.database import get_db_connection

logger = logging.getLogger(__name__)

def create_post(
    username: str, 
    user_role: str, 
    title: str, 
    content: str, 
    category: str = "General",
    related_document: Optional[str] = None
) -> Tuple[bool, str, Optional[int]]:
    """Create a new forum post."""
    if not title or not content:
        return False, "Title and content are required.", None
    
    if len(title) < 5:
        return False, "Title must be at least 5 characters long.", None
    
    if len(content) < 10:
        return False, "Content must be at least 10 characters long.", None
    
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO forum_posts (
                username, title, content, category, 
                related_document, created_at, updated_at, status, upvotes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            username, title, content, category, 
            related_document, datetime.now().isoformat(), datetime.now().isoformat(), 'open', 0
        ))
        conn.commit()
        post_id = c.lastrowid
        logger.info(f"Post created: ID={post_id}, Title='{title}', User={username}")
        success = True
        msg = "Post created successfully!"
    except Exception as e:
        logger.error(f"Error saving post: {str(e)}")
        success = False
        msg = "Error saving post."
        post_id = None
    finally:
        conn.close()
        
    return success, msg, post_id

def add_reply(
    post_id: int, 
    username: str, 
    user_role: str, 
    content: str,
    is_answer: bool = False
) -> Tuple[bool, str]:
    """Add a reply to a forum post."""
    if not content:
        return False, "Reply content is required."
    
    if len(content) < 5:
        return False, "Reply must be at least 5 characters long."
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT id FROM forum_posts WHERE id = ?', (post_id,))
    if not c.fetchone():
        conn.close()
        return False, "Post not found."
    
    try:
        is_ans = is_answer and user_role == "teacher"
        c.execute('''
            INSERT INTO forum_replies (post_id, username, content, created_at, is_answer)
            VALUES (?, ?, ?, ?, ?)
        ''', (post_id, username, content, datetime.now().isoformat(), is_ans))
        
        # Update post timestamp and status
        status_update = "status = 'answered'," if is_ans else ""
        c.execute(f'''
            UPDATE forum_posts 
            SET {status_update} updated_at = ?
            WHERE id = ?
        ''', (datetime.now().isoformat(), post_id))
        
        conn.commit()
        logger.info(f"Reply added to post {post_id} by {username}")
        success = True
        msg = "Reply added successfully!"
    except Exception as e:
        logger.error(f"Error saving reply: {str(e)}")
        success = False
        msg = "Error saving reply."
    finally:
        conn.close()
        
    return success, msg

def _build_post_dict(c, post_row) -> dict:
    """Helper to structure post dict with replies from DB rows."""
    post_id = post_row['id']
    # Fetch replies joined with user roles
    c.execute('''
        SELECT r.*, u.role as user_role 
        FROM forum_replies r
        LEFT JOIN users u ON r.username = u.username
        WHERE r.post_id = ?
        ORDER BY r.created_at ASC
    ''', (post_id,))
    reply_rows = c.fetchall()
    
    replies = []
    for r in reply_rows:
        replies.append({
            'id': r['id'],
            'username': r['username'],
            'user_role': r['user_role'] or 'student',
            'content': r['content'],
            'created_at': r['created_at'],
            'is_answer': bool(r['is_answer'])
        })
        
    return {
        'id': post_row['id'],
        'username': post_row['username'],
        'user_role': post_row['user_role'] or 'student',
        'title': post_row['title'],
        'content': post_row['content'],
        'category': post_row['category'],
        'related_document': post_row['related_document'],
        'created_at': post_row['created_at'],
        'updated_at': post_row['updated_at'],
        'status': post_row['status'],
        'upvotes': post_row['upvotes'],
        'replies': replies
    }

def get_all_posts(
    status: Optional[str] = None, 
    category: Optional[str] = None,
    sort_by: str = "recent"
) -> List[dict]:
    """Get all forum posts with optional filtering and sorting."""
    conn = get_db_connection()
    c = conn.cursor()
    
    query = '''
        SELECT p.*, u.role as user_role 
        FROM forum_posts p
        LEFT JOIN users u ON p.username = u.username
        WHERE 1=1
    '''
    params = []
    
    if status:
        query += " AND p.status = ?"
        params.append(status)
        
    if category and category != "All":
        query += " AND p.category = ?"
        params.append(category)
        
    if sort_by == "popular":
        query += " ORDER BY p.upvotes DESC, p.created_at DESC"
    else:
        query += " ORDER BY p.created_at DESC"
        
    c.execute(query, params)
    post_rows = c.fetchall()
    
    posts = [_build_post_dict(c, row) for row in post_rows]
    conn.close()
    return posts

def get_post_by_id(post_id: int) -> Optional[dict]:
    """Get a single post by ID with all replies."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT p.*, u.role as user_role 
        FROM forum_posts p
        LEFT JOIN users u ON p.username = u.username
        WHERE p.id = ?
    ''', (post_id,))
    post_row = c.fetchone()
    
    if not post_row:
        conn.close()
        return None
        
    post_dict = _build_post_dict(c, post_row)
    conn.close()
    return post_dict

def upvote_post(post_id: int) -> Tuple[bool, str]:
    """Upvote a forum post."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT upvotes FROM forum_posts WHERE id = ?', (post_id,))
    post = c.fetchone()
    if not post:
        conn.close()
        return False, "Post not found."
    
    try:
        c.execute('UPDATE forum_posts SET upvotes = upvotes + 1 WHERE id = ?', (post_id,))
        conn.commit()
        success = True
        msg = "Post upvoted!"
    except Exception as e:
        logger.error(f"Error upvoting: {str(e)}")
        success = False
        msg = "Error upvoting post."
    finally:
        conn.close()
        
    return success, msg

def update_post_status(post_id: int, status: str) -> Tuple[bool, str]:
    """Update post status (open, answered, closed)."""
    if status not in ['open', 'answered', 'closed']:
        return False, "Invalid status."
    
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT id FROM forum_posts WHERE id = ?', (post_id,))
    if not c.fetchone():
        conn.close()
        return False, "Post not found."
    
    try:
        c.execute('UPDATE forum_posts SET status = ?, updated_at = ? WHERE id = ?', 
                  (status, datetime.now().isoformat(), post_id))
        conn.commit()
        success = True
        msg = f"Status updated to '{status}'."
    except Exception as e:
        logger.error(f"Error updating status: {str(e)}")
        success = False
        msg = "Error updating status."
    finally:
        conn.close()
        
    return success, msg

def delete_post(post_id: int, username: str, user_role: str) -> Tuple[bool, str]:
    """Delete a post (only author or teacher can delete)."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('SELECT username FROM forum_posts WHERE id = ?', (post_id,))
    post = c.fetchone()
    
    if not post:
        conn.close()
        return False, "Post not found."
    
    # Check permissions
    if post['username'] != username and user_role != 'teacher' and user_role != 'admin':
        conn.close()
        return False, "Permission denied. Only author, teacher, or admin can delete."
    
    try:
        # Cascade delete is enabled via PRAGMA foreign_keys = ON, but we explicitly delete safely
        c.execute('DELETE FROM forum_posts WHERE id = ?', (post_id,))
        conn.commit()
        logger.info(f"Post {post_id} deleted by {username}")
        success = True
        msg = "Post deleted successfully."
    except Exception as e:
        logger.error(f"Error deleting post: {str(e)}")
        success = False
        msg = "Error deleting post."
    finally:
        conn.close()
        
    return success, msg

def get_pending_posts_count() -> int:
    """Get count of open questions that need answers."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM forum_posts WHERE status = 'open'")
    count = c.fetchone()[0]
    conn.close()
    return count

def get_forum_stats() -> Dict:
    """Get forum statistics."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM forum_posts")
    total_posts = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM forum_posts WHERE status = 'open'")
    open_posts = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM forum_posts WHERE status = 'answered'")
    answered_posts = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM forum_posts WHERE status = 'closed'")
    closed_posts = c.fetchone()[0]
    
    c.execute("SELECT COUNT(*) FROM forum_replies")
    total_replies = c.fetchone()[0]
    
    c.execute("SELECT SUM(upvotes) FROM forum_posts")
    upvotes_res = c.fetchone()[0]
    total_upvotes = upvotes_res if upvotes_res else 0
    
    conn.close()
    
    return {
        'total_posts': total_posts,
        'open_posts': open_posts,
        'answered_posts': answered_posts,
        'closed_posts': closed_posts,
        'total_replies': total_replies,
        'total_upvotes': total_upvotes,
        'pending_posts': open_posts
    }

def get_categories() -> List[str]:
    """Get list of available forum categories."""
    return [
        "All",
        "General",
        "Physics",
        "Chemistry",
        "Mathematics",
        "Biology",
        "Computer Science",
        "Social Studies",
        "Languages",
        "Exam Prep",
        "Homework Help"
    ]

def search_posts(query: str) -> List[dict]:
    """Search posts by title or content."""
    if not query:
        return []
        
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT p.*, u.role as user_role 
        FROM forum_posts p
        LEFT JOIN users u ON p.username = u.username
        WHERE p.title LIKE ? OR p.content LIKE ?
        ORDER BY p.created_at DESC
    ''', (f"%{query}%", f"%{query}%"))
    post_rows = c.fetchall()
    
    posts = [_build_post_dict(c, row) for row in post_rows]
    conn.close()
    return posts

def get_user_posts(username: str) -> List[dict]:
    """Get all posts by a specific user."""
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''
        SELECT p.*, u.role as user_role 
        FROM forum_posts p
        LEFT JOIN users u ON p.username = u.username
        WHERE p.username = ?
        ORDER BY p.created_at DESC
    ''', (username,))
    post_rows = c.fetchall()
    
    posts = [_build_post_dict(c, row) for row in post_rows]
    conn.close()
    return posts
