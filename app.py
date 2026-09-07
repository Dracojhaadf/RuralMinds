"""
RuralMinds - Decentralized Offline-First Academic Infrastructure.
Main Streamlit application entrypoint.
"""
import streamlit as st
import logging

from core.database import ensure_migrated
from core.forum import get_pending_posts_count
from services.llm_service import preload_ollama_model
from ui.styles import inject_custom_css
from ui.auth_view import show_auth_page, show_admin_login
from ui.admin_view import render_admin_panel
from ui.learning_hub_view import render_learning_hub
from ui.forum_view import render_forum

logger = logging.getLogger(__name__)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Edubridge", 
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# PRELOAD MODELS
@st.cache_resource
def init_llm():
    """Preload Ollama model into RAM on startup."""
    preload_ollama_model()

init_llm()

# INJECT STYLES
inject_custom_css()

# INITIALIZE SESSION STATE
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'show_admin_login' not in st.session_state:
    st.session_state.show_admin_login = False

# ENSURE DATABASE SETUP & MIGRATIONS
if 'db_init' not in st.session_state:
    try:
        ensure_migrated()
        st.session_state.db_init = True
    except Exception as e:
        logger.error(f"Database init error: {e}")
        st.session_state.db_init = False

# AUTHENTICATION ROUTING
if not st.session_state.authenticated:
    if st.session_state.show_admin_login:
        show_admin_login()
    else:
        show_auth_page()
    st.stop()

# USER ROLES
user_role = st.session_state.user_data.get('role', 'student')
is_admin = (user_role == 'admin')
is_teacher = (user_role == 'teacher')
is_student = (user_role == 'student')

# ADMIN PANEL
if is_admin:
    render_admin_panel()
    st.stop()

# SIDEBAR FOR TEACHERS & STUDENTS
with st.sidebar:
    st.markdown("---")
    col_u1, col_u2 = st.columns([3, 3])
    with col_u1:
        st.write(f"**{st.session_state.user_data.get('name', 'User')}**")
        st.caption(f"{user_role.title()}")
    with col_u2:
        if st.button("Logout", help="Logout"):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.rerun()
    
    if is_teacher:
        pending = get_pending_posts_count()
        if pending > 0:
            st.info(f"🔔 {pending} Pending Posts")

# MAIN TABS (Learning Hub & Discussion Forum)
tab1, tab2 = st.tabs(["📚 Learning Hub", "💬 Discussion Forum"])

with tab1:
    render_learning_hub(is_teacher=is_teacher)

with tab2:
    render_forum(is_teacher=is_teacher)