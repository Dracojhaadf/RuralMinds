"""
UI presentation layer for RuralMinds Streamlit application.
"""
from ui.styles import inject_custom_css
from ui.auth_view import show_auth_page, show_admin_login, validate_password, is_valid_email
from ui.admin_view import render_admin_panel
from ui.learning_hub_view import render_learning_hub
from ui.forum_view import render_forum

__all__ = [
    "inject_custom_css",
    "show_auth_page",
    "show_admin_login",
    "validate_password",
    "is_valid_email",
    "render_admin_panel",
    "render_learning_hub",
    "render_forum",
]
