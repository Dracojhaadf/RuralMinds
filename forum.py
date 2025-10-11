import json
import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# Forum database file
FORUM_DB_PATH = "forum_db.json"

def load_forum_data() -> dict:
    """Load forum data from JSON file."""
    if not os.path.exists(FORUM_DB_PATH):
        default_data = {
            "posts": [],
            "next_id": 1
        }
        save_forum_data(default_data)
        return default_data
    
    try:
        with open(FORUM_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading forum data: {str(e)}")
        return {"posts": [], "next_id": 1}

def save_forum_data(data: dict) -> bool:
    """Save forum data to JSON file."""
    try:
        with open(FORUM_DB_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving forum data: {str(e)}")
        return False

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
    
    data = load_forum_data()
    post_id = data["next_id"]
    
    new_post = {
        "id": post_id,
        "username": username,
        "user_role": user_role,
        "title": title,
        "content": content,
        "category": category,
        "related_document": related_document,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": "open",
        "upvotes": 0,
        "replies": []
    }
    
    data["posts"].append(new_post)
    data["next_id"] += 1
    
    if save_forum_data(data):
        return True, "Post created successfully!", post_id
    else:
        return False, "Error saving post.", None

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
    
    data = load_forum_data()
    
    post = None
    for p in data["posts"]:
        if p["id"] == post_id:
            post = p
            break
    
    if not post:
        return False, "Post not found."
    
    reply = {
        "username": username,
        "user_role": user_role,
        "content": content,
        "created_at": datetime.now().isoformat(),
        "is_answer": is_answer and user_role == "teacher"
    }
    
    post["replies"].append(reply)
    post["updated_at"] = datetime.now().isoformat()
    
    if is_answer and user_role == "teacher":
        post["status"] = "answered"
    
    if save_forum_data(data):
        return True, "Reply added successfully!"
    else:
        return False, "Error saving reply."

def get_all_posts(
    filter_status: Optional[str] = None,
    filter_category: Optional[str] = None,
    sort_by: str = "recent"
) -> List[dict]:
    """Get all forum posts with filtering and sorting."""
    data = load_forum_data()
    posts = data["posts"]
    
    if filter_status:
        posts = [p for p in posts if p["status"] == filter_status]
    
    if filter_category and filter_category != "All":
        posts = [p for p in posts if p["category"] == filter_category]
    
    if sort_by == "recent":
        posts = sorted(posts, key=lambda x: x["updated_at"], reverse=True)
    elif sort_by == "popular":
        posts = sorted(posts, key=lambda x: x["upvotes"], reverse=True)
    elif sort_by == "unanswered":
        posts = [p for p in posts if p["status"] == "open"]
        posts = sorted(posts, key=lambda x: x["created_at"], reverse=True)
    
    return posts

def get_post_by_id(post_id: int) -> Optional[dict]:
    """Get a specific post by ID."""
    data = load_forum_data()
    for post in data["posts"]:
        if post["id"] == post_id:
            return post
    return None

def upvote_post(post_id: int) -> Tuple[bool, str]:
    """Upvote a post."""
    data = load_forum_data()
    
    for post in data["posts"]:
        if post["id"] == post_id:
            post["upvotes"] += 1
            if save_forum_data(data):
                return True, "Post upvoted!"
            else:
                return False, "Error saving upvote."
    
    return False, "Post not found."

def update_post_status(post_id: int, status: str) -> Tuple[bool, str]:
    """Update post status (teacher only)."""
    if status not in ["open", "answered", "closed"]:
        return False, "Invalid status."
    
    data = load_forum_data()
    
    for post in data["posts"]:
        if post["id"] == post_id:
            post["status"] = status
            post["updated_at"] = datetime.now().isoformat()
            if save_forum_data(data):
                return True, f"Post status updated to {status}."
            else:
                return False, "Error updating status."
    
    return False, "Post not found."

def delete_post(post_id: int, username: str, user_role: str) -> Tuple[bool, str]:
    """Delete a post (only by post owner or teacher)."""
    data = load_forum_data()
    
    for i, post in enumerate(data["posts"]):
        if post["id"] == post_id:
            if post["username"] == username or user_role == "teacher":
                del data["posts"][i]
                if save_forum_data(data):
                    return True, "Post deleted successfully."
                else:
                    return False, "Error deleting post."
            else:
                return False, "You don't have permission to delete this post."
    
    return False, "Post not found."

def get_pending_posts_count() -> int:
    """Get count of posts without teacher replies (for notifications)."""
    data = load_forum_data()
    pending = 0
    
    for post in data["posts"]:
        if post["status"] == "open":
            has_teacher_reply = any(r["user_role"] == "teacher" for r in post["replies"])
            if not has_teacher_reply:
                pending += 1
    
    return pending

def get_forum_stats() -> Dict:
    """Get forum statistics."""
    data = load_forum_data()
    posts = data["posts"]
    
    total_posts = len(posts)
    open_posts = len([p for p in posts if p["status"] == "open"])
    answered_posts = len([p for p in posts if p["status"] == "answered"])
    total_replies = sum(len(p["replies"]) for p in posts)
    pending_posts = get_pending_posts_count()
    
    return {
        "total_posts": total_posts,
        "open_posts": open_posts,
        "answered_posts": answered_posts,
        "closed_posts": total_posts - open_posts - answered_posts,
        "total_replies": total_replies,
        "pending_posts": pending_posts
    }

def get_categories() -> List[str]:
    """Get list of all categories."""
    default_categories = [
        "All", 
        "General", 
        "PDF Questions", 
        "Video Questions", 
        "Technical Help", 
        "Study Tips", 
        "Other"
    ]
    return default_categories

def search_posts(query: str) -> List[dict]:
    """Search posts by title or content."""
    data = load_forum_data()
    query_lower = query.lower()
    
    results = []
    for post in data["posts"]:
        if (query_lower in post["title"].lower() or 
            query_lower in post["content"].lower()):
            results.append(post)
    
    return results

def get_user_posts(username: str) -> List[dict]:
    """Get all posts by a specific user."""
    data = load_forum_data()
    return [p for p in data["posts"] if p["username"] == username]