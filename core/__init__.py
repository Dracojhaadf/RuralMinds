"""
Core data layer and domain models for RuralMinds.
"""
from core.database import get_db_connection, init_db, ensure_migrated, migrate_to_bcnf
from core.auth import (
    hash_password,
    create_user,
    authenticate_user,
    get_user_role,
    change_password,
    get_all_users,
    delete_user,
)
from core.forum import (
    create_post,
    add_reply,
    get_all_posts,
    get_post_by_id,
    upvote_post,
    update_post_status,
    delete_post,
    get_pending_posts_count,
    get_forum_stats,
    get_categories,
    search_posts,
    get_user_posts,
)

__all__ = [
    "get_db_connection",
    "init_db",
    "ensure_migrated",
    "migrate_to_bcnf",
    "hash_password",
    "create_user",
    "authenticate_user",
    "get_user_role",
    "change_password",
    "get_all_users",
    "delete_user",
    "create_post",
    "add_reply",
    "get_all_posts",
    "get_post_by_id",
    "upvote_post",
    "update_post_status",
    "delete_post",
    "get_pending_posts_count",
    "get_forum_stats",
    "get_categories",
    "search_posts",
    "get_user_posts",
]
