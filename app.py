import streamlit as st
from backend import (
    process_and_save_pdf, 
    query_saved_document, 
    get_available_documents,
    save_video,
    save_caption_file,
    load_caption_file,
    process_video_captions,
    get_available_videos,
    delete_document,
    delete_video,
    get_document_stats
)
from auth import (
    authenticate_user,
    create_user,
    get_all_users,
    delete_user
)
from forum import (
    create_post,
    add_reply,
    get_all_posts,
    get_post_by_id,
    upvote_post,
    delete_post,
    get_forum_stats,
    get_categories,
    search_posts,
    get_pending_posts_count
)
import os
from pathlib import Path
from datetime import datetime
import re

# PAGE CONFIGURATION
st.set_page_config(
    page_title="RAG Teaching Assistant", 
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# CONFIGURATION
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", "source_folder")
DB_PATH = "chroma_db"
ADMIN_USERNAME = "administrator"
ADMIN_PASSWORD = "Admin@2024"  # Strong password

# SESSION STATE
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'show_admin_login' not in st.session_state:
    st.session_state.show_admin_login = False

# PASSWORD VALIDATION
def validate_password(password: str) -> tuple:
    """Validate password strength."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain at least one special character"
    return True, "Valid"

def is_valid_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def authenticate_admin(username: str, password: str) -> bool:
    """Authenticate administrator."""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def show_auth_page():
    """Display authentication page."""
    st.markdown("<h1 style='text-align: center;'>📚 RAG Teaching Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>AI-Powered Document Q&A System</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.show_signup:
            st.subheader("🔐 Login")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.form_submit_button("🚀 Login", type="primary", use_container_width=True):
                        if username and password:
                            success, user_data = authenticate_user(username, password)
                            if success:
                                st.session_state.authenticated = True
                                st.session_state.user_data = user_data
                                st.success(f"✅ Welcome, {user_data['name']}!")
                                st.rerun()
                            else:
                                st.error("❌ Invalid credentials")
                        else:
                            st.warning("⚠️ Enter username and password")
                with col_b:
                    if st.form_submit_button("📝 Student Signup", use_container_width=True):
                        st.session_state.show_signup = True
                        st.rerun()
            
            st.markdown("---")
            if st.button("🔑 Administrator Login", use_container_width=True):
                st.session_state.show_admin_login = True
                st.rerun()
            
            st.info("💡 **Students** can self-register. **Teachers** are created by admin.")
        else:
            st.subheader("📝 Student Registration")
            st.info("👨‍🎓 Self-registration is only for students. Teachers must be created by administrator.")
            
            with st.form("signup_form"):
                new_username = st.text_input("Username (min 3 chars)")
                new_password = st.text_input("Password (min 8 chars, mixed case, number, special char)", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.form_submit_button("✅ Sign Up", type="primary", use_container_width=True):
                        if not all([new_username, new_password, full_name, email]):
                            st.error("❌ All fields required")
                        elif new_password != confirm_password:
                            st.error("❌ Passwords don't match")
                        elif not is_valid_email(email):
                            st.error("❌ Invalid email format")
                        else:
                            valid, msg = validate_password(new_password)
                            if not valid:
                                st.error(f"❌ {msg}")
                            else:
                                # Force student role for public signup
                                success, message = create_user(new_username, new_password, "student", full_name, email)
                                if success:
                                    st.success(message)
                                    st.info("👉 You can now login")
                                    st.session_state.show_signup = False
                                else:
                                    st.error(message)
                with col_b:
                    if st.form_submit_button("⬅️ Back to Login", use_container_width=True):
                        st.session_state.show_signup = False
                        st.rerun()

def show_admin_login():
    """Display admin login page."""
    st.markdown("<h1 style='text-align: center;'>📚 Administrator Login</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("admin_login"):
            st.warning("🔒 Administrator access only")
            username = st.text_input("Admin Username")
            password = st.text_input("Admin Password", type="password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.form_submit_button("🔓 Login as Admin", type="primary", use_container_width=True):
                    if authenticate_admin(username, password):
                        st.session_state.authenticated = True
                        st.session_state.user_data = {
                            'username': 'administrator',
                            'role': 'admin',
                            'name': 'Administrator',
                            'email': 'admin@system.local'
                        }
                        st.success("✅ Admin access granted")
                        st.rerun()
                    else:
                        st.error("❌ Invalid admin credentials")
            with col_b:
                if st.form_submit_button("⬅️ Back", use_container_width=True):
                    st.session_state.show_admin_login = False
                    st.rerun()

# CHECK AUTHENTICATION
if not st.session_state.authenticated:
    if st.session_state.show_admin_login:
        show_admin_login()
    else:
        show_auth_page()
    st.stop()

# INITIALIZE DATABASE
def initialize_database():
    """Initialize database from source folder."""
    try:
        if os.path.exists(SOURCE_FOLDER):
            pdf_files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(".pdf")]
            if pdf_files:
                from backend import get_available_documents as get_docs
                existing = get_docs()
                # Processing happens in backend automatically
        return True
    except Exception as e:
        logger.error(f"Init error: {str(e)}")
        return True

if 'db_init' not in st.session_state:
    st.session_state.db_init = initialize_database()

# USER ROLES
is_admin = st.session_state.user_data['role'] == 'admin'
is_teacher = st.session_state.user_data['role'] == 'teacher'
is_student = st.session_state.user_data['role'] == 'student'

# =============================================================================
# ADMIN PANEL
# =============================================================================
if is_admin:
    st.title("📚 Administrator Panel")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**Logged in as:** {st.session_state.user_data['name']}")
    with col3:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.rerun()
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["👥 All Users", "👨‍🏫 Create Teacher", "📊 Statistics"])
    
    with tab1:
        users = get_all_users()
        st.metric("Total Users", len(users))
        
        # Separate by role
        teachers = [u for u in users if u['role'] == 'teacher']
        students = [u for u in users if u['role'] == 'student']
        
        st.subheader("👨‍🏫 Teachers")
        for user in teachers:
            col_a, col_b = st.columns([0.8, 0.2])
            with col_a:
                st.write(f"**{user['name']}** (@{user['username']}) - {user['email']}")
            with col_b:
                if user['username'] not in ['admin', 'administrator']:
                    if st.button("🗑️", key=f"del_t_{user['username']}"):
                        success, msg = delete_user(user['username'])
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        
        st.subheader("👨‍🎓 Students")
        for user in students:
            col_a, col_b = st.columns([0.8, 0.2])
            with col_a:
                st.write(f"**{user['name']}** (@{user['username']}) - {user['email']}")
            with col_b:
                if st.button("🗑️", key=f"del_s_{user['username']}"):
                    success, msg = delete_user(user['username'])
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    
    with tab2:
        st.subheader("➕ Create Teacher Account")
        st.info("🔒 Only administrators can create teacher accounts")
        
        with st.form("create_teacher_form"):
            t_username = st.text_input("Username*")
            t_email = st.text_input("Teacher Email*", placeholder="teacher@school.edu")
            t_name = st.text_input("Full Name*")
            t_password = st.text_input("Temporary Password*", type="password", 
                                      help="Min 8 chars with uppercase, lowercase, number, special char")
            
            if st.form_submit_button("✅ Create Teacher", type="primary"):
                if all([t_username, t_email, t_name, t_password]):
                    if not is_valid_email(t_email):
                        st.error("❌ Invalid email")
                    else:
                        valid, msg = validate_password(t_password)
                        if not valid:
                            st.error(f"❌ {msg}")
                        else:
                            # Create teacher account
                            success, message = create_user(t_username, t_password, "teacher", t_name, t_email)
                            if success:
                                st.success(f"✅ Teacher account created!")
                                st.info(f"📧 Notify {t_name} at {t_email} to change password on first login")
                            else:
                                st.error(message)
                else:
                    st.error("❌ Fill all fields")
    
    with tab3:
        forum_stats = get_forum_stats()
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Total Users", len(get_all_users()))
            st.metric("Teachers", len([u for u in get_all_users() if u['role'] == 'teacher']))
        with col_b:
            st.metric("Documents", len(get_available_documents()))
            st.metric("Videos", len(get_available_videos()))
        with col_c:
            st.metric("Forum Posts", forum_stats['total_posts'])
            st.metric("Pending Posts", forum_stats['pending_posts'])
    
    st.stop()

# =============================================================================
# MAIN APPLICATION (Teachers & Students)
# =============================================================================

# HEADER
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    role_emoji = "👨‍🏫" if is_teacher else "👨‍🎓"
    st.title(f"📚 RAG Assistant {role_emoji}")
with col2:
    st.write(f"**{st.session_state.user_data['name']}**")
    st.caption(f"{st.session_state.user_data['role'].title()}")
with col3:
    if is_teacher:
        pending = get_pending_posts_count()
        if pending > 0:
            st.metric("🔔 Pending", pending)
        else:
            st.metric("🔔", "0")
with col4:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.rerun()

st.markdown("---")

# MAIN TABS
tab1, tab2 = st.tabs(["📚 Learning Hub", "💬 Discussion Forum"])

# =============================================================================
# LEARNING HUB TAB
# =============================================================================
with tab1:
    # SIDEBAR
    with st.sidebar:
        st.header("📚 Content Management")
        
        if is_teacher:
            st1, st2, st3 = st.tabs(["📄 PDFs", "🎥 Videos", "⚙️ Manage"])
        else:
            st1, st2 = st.tabs(["📚 Browse", "🎥 Videos"])
            st3 = None
        
        # PDF TAB
        with st1:
            if is_teacher:
                st.subheader("Upload PDF")
                pdf = st.file_uploader("Choose PDF", type="pdf", key="pdf_up")
                if pdf:
                    with st.spinner("Processing..."):
                        success, msg = process_and_save_pdf(pdf)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.subheader("Available Documents")
                docs = get_available_documents()
                st.metric("Total", len(docs))
                for d in docs:
                    st.write(f"📄 {d}")
        
        # VIDEO TAB
        with st2:
            if is_teacher:
                st.subheader("Upload Video")
                vid = st.file_uploader("Choose Video", type=["mp4", "avi", "mov"], key="vid_up")
                if vid:
                    with st.spinner("Saving..."):
                        success, msg, path = save_video(vid)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                
                st.markdown("---")
            
            st.subheader("Video Library")
            videos = get_available_videos()
            if videos:
                sel_vid = st.selectbox("Select:", [v['name'] for v in videos], key="vid_sel")
                match = next((v for v in videos if v['name'] == sel_vid), None)
                if match and os.path.exists(match['path']):
                    with open(match['path'], 'rb') as f:
                        st.video(f.read())
                    
                    if match['has_captions']:
                        st.success("✅ Has captions")
            else:
                st.info("No videos yet")
        
        # MANAGE TAB (Teacher only)
        if st3:
            with st3:
                st.subheader("Delete Content")
                docs = get_available_documents()
                if docs:
                    sel = st.selectbox("Document:", docs, key="del_sel")
                    if st.button("🗑️ Delete Document"):
                        success, msg = delete_document(sel)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
    
    # MAIN CHAT AREA
    docs = get_available_documents()
    if docs:
        if 'selected_doc' not in st.session_state:
            st.session_state.selected_doc = docs[0]
        
        col_x, col_y, col_z = st.columns([2, 1, 1])
        with col_x:
            sel = st.selectbox("📖 Select Document:", docs, key="selected_doc")
        with col_y:
            stats = get_document_stats(sel)
            if stats:
                st.metric("Type", stats['type'])
        with col_z:
            if stats:
                st.metric("Chunks", stats['chunk_count'])
        
        st.markdown("---")
        
        # Initialize chat
        if 'messages' not in st.session_state or st.session_state.get('current_doc') != sel:
            st.session_state.messages = []
            st.session_state.current_doc = sel
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"👋 Hi! Ask me anything about **{sel}**. I'll answer based only on the document content."
            })
        
        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input
        if query := st.chat_input(f"Ask about {sel}..."):
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching document..."):
                    try:
                        answer, chunks = query_saved_document(sel, query)
                        st.write(answer)
                        
                        # Show sources
                        with st.expander("📄 View Retrieved Sources"):
                            for i, chunk in enumerate(chunks, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(chunk)
                                if i < len(chunks):
                                    st.markdown("---")
                    
                    except Exception as e:
                        answer = f"❌ Error: {str(e)}"
                        st.error(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("👈 No documents available. Teachers can upload PDFs in the sidebar.")

# =============================================================================
# DISCUSSION FORUM TAB
# =============================================================================
with tab2:
    def fmt_dt(iso):
        """Format ISO datetime to readable format."""
        try:
            return datetime.fromisoformat(iso).strftime("%b %d, %I:%M %p")
        except:
            return iso
    
    st.markdown("## 💬 Discussion Forum")
    
    # Forum stats
    stats = get_forum_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Posts", stats['total_posts'])
    c2.metric("❓ Open", stats['open_posts'])
    c3.metric("✅ Answered", stats['answered_posts'])
    if is_teacher:
        c4.metric("🔔 Pending", stats['pending_posts'])
    else:
        c4.metric("💬 Replies", stats['total_replies'])
    
    st.markdown("---")
    
    # Forum tabs
    ft1, ft2, ft3 = st.tabs(["📋 All Posts", "➕ Create Post", "🔍 Search"])
    
    # ALL POSTS TAB
    with ft1:
        col_filter, col_sort = st.columns(2)
        with col_filter:
            filt = st.selectbox("Filter:", ["All", "Open", "Answered"], key="forum_filter")
        with col_sort:
            sort = st.selectbox("Sort by:", ["Recent", "Popular"], key="forum_sort")
        
        status_map = {"All": None, "Open": "open", "Answered": "answered"}
        sort_map = {"Recent": "recent", "Popular": "popular"}
        
        posts = get_all_posts(status_map[filt], None, sort_map[sort])
        
        if not posts:
            st.info("📭 No posts yet. Be the first to ask a question!")
        
        for p in posts:
            # Post status emoji
            if p['status'] == 'open':
                status_emoji = "❓"
            elif p['status'] == 'answered':
                status_emoji = "✅"
            else:
                status_emoji = "🔒"
            
            # Post card
            with st.container():
                st.markdown(f"### {status_emoji} {p['title']}")
                
                # Post metadata
                col_meta1, col_meta2, col_meta3 = st.columns([2, 1, 1])
                with col_meta1:
                    role_badge = "👨‍🏫" if p['user_role'] == 'teacher' else "👨‍🎓"
                    st.caption(f"{role_badge} {p['username']} | 📅 {fmt_dt(p['created_at'])}")
                with col_meta2:
                    st.caption(f"📁 {p['category']}")
                with col_meta3:
                    st.caption(f"💬 {len(p['replies'])} replies")
                
                # Post content preview
                preview = p['content'][:200] + "..." if len(p['content']) > 200 else p['content']
                st.write(preview)
                
                # Action buttons
                col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 3])
                
                with col_a:
                    if st.button(f"👍 {p['upvotes']}", key=f"up_{p['id']}"):
                        upvote_post(p['id'])
                        st.rerun()
                
                with col_b:
                    if st.button("💬 View", key=f"view_{p['id']}", type="primary"):
                        st.session_state.viewing_post = p['id']
                        st.rerun()
                
                with col_c:
                    # Delete button (for post owner or teacher)
                    if is_teacher or p['username'] == st.session_state.user_data['username']:
                        if st.button("🗑️", key=f"del_{p['id']}"):
                            success, msg = delete_post(
                                p['id'], 
                                st.session_state.user_data['username'],
                                st.session_state.user_data['role']
                            )
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                st.markdown("---")
    
    # CREATE POST TAB
    with ft2:
        st.subheader("➕ Create New Post")
        
        with st.form("new_post_form", clear_on_submit=True):
            title = st.text_input("Title*", placeholder="Brief summary of your question")
            cat = st.selectbox("Category*", [c for c in get_categories() if c != "All"])
            content = st.text_area(
                "Question Details*", 
                height=200,
                placeholder="Describe your question in detail..."
            )
            
            # Optional: Link to document
            docs = get_available_documents()
            if docs:
                related_doc = st.selectbox(
                    "Related Document (optional):", 
                    ["None"] + docs
                )
            else:
                related_doc = "None"
            
            col_submit, col_info = st.columns([1, 2])
            
            with col_submit:
                submit = st.form_submit_button("📤 Post Question", type="primary", use_container_width=True)
            
            with col_info:
                st.caption("*Required fields")
            
            if submit:
                if not title or not content:
                    st.error("❌ Title and content are required")
                elif len(title) < 5:
                    st.error("❌ Title must be at least 5 characters")
                elif len(content) < 10:
                    st.error("❌ Content must be at least 10 characters")
                else:
                    u = st.session_state.user_data
                    rel_doc = None if related_doc == "None" else related_doc
                    
                    success, msg, post_id = create_post(
                        u['username'], 
                        u['role'], 
                        title, 
                        content, 
                        cat,
                        rel_doc
                    )
                    
                    if success:
                        st.success(f"✅ {msg}")
                        st.info(f"📌 Post ID: {post_id}")
                        # Clear the viewing post if any
                        if 'viewing_post' in st.session_state:
                            del st.session_state.viewing_post
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")
    
    # SEARCH TAB
    with ft3:
        st.subheader("🔍 Search Posts")
        
        search_query = st.text_input("Search for:", placeholder="Enter keywords...")
        
        if search_query:
            results = search_posts(search_query)
            st.write(f"**Found {len(results)} result(s)**")
            
            for r in results:
                with st.container():
                    status_emoji = "✅" if r['status'] == 'answered' else "❓"
                    st.markdown(f"#### {status_emoji} {r['title']}")
                    st.caption(f"By {r['username']} | {fmt_dt(r['created_at'])}")
                    st.write(r['content'][:150] + "...")
                    
                    if st.button("View Post", key=f"search_{r['id']}"):
                        st.session_state.viewing_post = r['id']
                        st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("👆 Enter a search term to find posts")
    
    # VIEW SINGLE POST (when a post is clicked)
    if 'viewing_post' in st.session_state:
        post = get_post_by_id(st.session_state.viewing_post)
        
        if post:
            st.markdown("---")
            st.markdown("---")
            
            # Post header
            status_emoji = "✅" if post['status'] == 'answered' else "❓"
            st.markdown(f"## {status_emoji} {post['title']}")
            
            # Post metadata
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                role_badge = "👨‍🏫" if post['user_role'] == 'teacher' else "👨‍🎓"
                st.write(f"**Posted by:** {role_badge} {post['username']}")
            with col_m2:
                st.write(f"**Category:** {post['category']}")
            with col_m3:
                st.write(f"**Status:** {post['status'].title()}")
            
            st.caption(f"📅 Created: {fmt_dt(post['created_at'])} | Updated: {fmt_dt(post['updated_at'])}")
            
            # Post content
            st.info(post['content'])
            
            # Related document
            if post.get('related_document'):
                st.caption(f"📄 Related Document: {post['related_document']}")
            
            st.markdown("---")
            
            # Replies section
            st.markdown(f"### 💬 {len(post['replies'])} Reply/Replies")
            
            if post['replies']:
                for idx, reply in enumerate(post['replies']):
                    # Highlight teacher answers
                    if reply.get('is_answer'):
                        bg_color = "rgba(76, 175, 80, 0.1)"  # Green tint
                        border = "2px solid #4CAF50"
                        prefix = "✅ **Teacher's Answer**"
                    else:
                        bg_color = "rgba(50, 50, 50, 0.05)"
                        border = "1px solid #ddd"
                        prefix = ""
                    
                    role_badge = "👨‍🏫" if reply['user_role'] == 'teacher' else "👨‍🎓"
                    
                    st.markdown(f"""
                    <div style="background:{bg_color}; padding:15px; border-radius:8px; 
                                margin:10px 0; border:{border}">
                        <div style="margin-bottom:8px">
                            <strong>{prefix}</strong><br>
                            <small>{role_badge} {reply['username']} | {fmt_dt(reply['created_at'])}</small>
                        </div>
                        <p style="margin:0">{reply['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("💬 No replies yet. Be the first to respond!")
            
            st.markdown("---")
            
            # Reply form
            st.subheader("📝 Add Your Reply")
            
            with st.form("reply_form"):
                reply_content = st.text_area(
                    "Your Reply*",
                    height=150,
                    placeholder="Write your reply here..."
                )
                
                # Mark as answer checkbox (teachers only)
                mark_as_answer = False
                if is_teacher:
                    mark_as_answer = st.checkbox(
                        "✅ Mark this as the answer (closes the question)",
                        help="This will mark the post as 'answered'"
                    )
                
                col_reply, col_back = st.columns([1, 3])
                
                with col_reply:
                    if st.form_submit_button("💬 Post Reply", type="primary"):
                        if not reply_content:
                            st.error("❌ Reply cannot be empty")
                        elif len(reply_content) < 5:
                            st.error("❌ Reply must be at least 5 characters")
                        else:
                            u = st.session_state.user_data
                            success, msg = add_reply(
                                post['id'],
                                u['username'],
                                u['role'],
                                reply_content,
                                mark_as_answer
                            )
                            
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                with col_back:
                    if st.form_submit_button("⬅️ Back to Posts"):
                        del st.session_state.viewing_post
                        st.rerun()
        else:
            st.error("❌ Post not found")
            if st.button("⬅️ Back to Forum"):
                del st.session_state.viewing_post
                st.rerun()