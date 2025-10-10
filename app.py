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
    get_document_stats,
    generate_captions_from_video,
    translate_text
)
from auth import (
    authenticate_user,
    create_user,
    get_user_role,
    change_password,
    get_all_users,
    delete_user
)
import os
import chromadb
from pathlib import Path

# --- PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="NFS", 
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION ---
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", "source_folder")
DB_PATH = "chroma_db"
ADMIN_USERNAME = "administrator"
ADMIN_PASSWORD = "admin"

# --- SESSION STATE INITIALIZATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'is_admin_login' not in st.session_state:
    st.session_state.is_admin_login = False
if 'show_admin_login' not in st.session_state:
    st.session_state.show_admin_login = False

# --- ADMIN AUTHENTICATION ---
def authenticate_admin(username: str, password: str) -> bool:
    """Verify admin credentials."""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

# --- USER AUTHENTICATION PAGE ---
def show_auth_page():
    """Display authentication page."""
    st.markdown("<h1 style='text-align: center;'>📚 NFS</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Document Management & Q&A System</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if not st.session_state.show_signup:
            # LOGIN FORM
            st.subheader("🔐 Login")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    login_button = st.form_submit_button("🚀 Login", type="primary", use_container_width=True)
                with col_b:
                    signup_button = st.form_submit_button("📝 Create Account", use_container_width=True)
                
                if login_button:
                    if username and password:
                        success, user_data = authenticate_user(username, password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user_data = user_data
                            st.session_state.is_admin_login = False
                            st.success(f"✅ Welcome, {user_data['name']}!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("⚠️ Please enter both username and password")
                
                if signup_button:
                    st.session_state.show_signup = True
                    st.rerun()
            
            # Admin login link at the bottom
            st.markdown("---")
            st.caption("Admin access?")
            if st.button("🔑 Admin Login", use_container_width=True):
                st.session_state.show_admin_login = True
                st.rerun()
        
        else:
            # SIGNUP FORM (Students only)
            st.subheader("📝 Create New Account")
            
            with st.form("signup_form"):
                new_username = st.text_input("Username (min 3 characters)", placeholder="Choose a username")
                new_password = st.text_input("Password (min 6 characters)", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
                full_name = st.text_input("Full Name", placeholder="Your full name")
                email = st.text_input("Email", placeholder="your.email@example.com")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    signup_submit = st.form_submit_button("✅ Sign Up", type="primary", use_container_width=True)
                with col_b:
                    back_button = st.form_submit_button("⬅️ Back to Login", use_container_width=True)
                
                if signup_submit:
                    if not all([new_username, new_password, confirm_password, full_name, email]):
                        st.error("❌ Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("❌ Passwords do not match")
                    else:
                        success, message = create_user(new_username, new_password, "student", full_name, email)
                        if success:
                            st.success(f"✅ {message}")
                            st.info("👉 You can now login with your credentials")
                            st.session_state.show_signup = False
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")
                
                if back_button:
                    st.session_state.show_signup = False
                    st.rerun()

# --- ADMIN LOGIN PAGE ---
def show_admin_login():
    """Display admin login page."""
    st.markdown("<h1 style='text-align: center;'>📚 NFS</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Admin Panel</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("🔑 Admin Login")
        
        with st.form("admin_login_form"):
            username = st.text_input("Admin Username", placeholder="Enter admin username")
            password = st.text_input("Admin Password", type="password", placeholder="Enter admin password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_button = st.form_submit_button("🔓 Login", type="primary", use_container_width=True)
            with col_b:
                back_button = st.form_submit_button("⬅️ Back", use_container_width=True)
            
            if login_button:
                if authenticate_admin(username, password):
                    st.session_state.authenticated = True
                    st.session_state.is_admin_login = True
                    st.session_state.user_data = {
                        'username': 'administrator',
                        'role': 'admin',
                        'name': 'Administrator',
                        'email': 'admin@nfs.local'
                    }
                    st.success("✅ Welcome, Admin!")
                    st.rerun()
                else:
                    st.error("❌ Invalid admin credentials")
            
            if back_button:
                st.session_state.show_admin_login = False
                st.rerun()

# --- CHECK AUTHENTICATION ---
if not st.session_state.authenticated:
    if st.session_state.show_admin_login:
        show_admin_login()
    else:
        show_auth_page()
    st.stop()

# --- ONE-TIME DATABASE INITIALIZATION ---
def initialize_database():
    """Initialize database from source folder."""
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        
        if os.path.exists(SOURCE_FOLDER):
            pdf_files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(".pdf")]
            
            if pdf_files:
                st.info(f"Found {len(pdf_files)} PDF(s) in source folder. Loading...")
                
                class LocalFile:
                    def __init__(self, path):
                        self.path = path
                        self.name = os.path.basename(path)
                    
                    def read(self):
                        with open(self.path, "rb") as f:
                            return f.read()
                
                existing_docs = get_available_documents()
                
                progress_bar = st.progress(0)
                for idx, pdf_file in enumerate(pdf_files):
                    doc_name = Path(pdf_file).stem
                    
                    if doc_name not in existing_docs:
                        file_path = os.path.join(SOURCE_FOLDER, pdf_file)
                        local_file_obj = LocalFile(file_path)
                        
                        with st.spinner(f"Processing '{pdf_file}'..."):
                            success, message = process_and_save_pdf(local_file_obj)
                            if not success:
                                st.error(f"Failed: {message}")
                    
                    progress_bar.progress((idx + 1) / len(pdf_files))
                
                st.success(f"Initialized with {len(pdf_files)} documents!")
        
        return True
    
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return False

if 'db_initialized' not in st.session_state:
    with st.spinner("Initializing database..."):
        st.session_state.db_initialized = initialize_database()

is_admin = st.session_state.user_data['role'] == 'admin'
is_teacher = st.session_state.user_data['role'] == 'teacher'

# --- ADMIN PANEL ---
if is_admin:
    st.markdown("<h1>📚 NFS - Admin Panel</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Logout button in header
    col1, col2 = st.columns([0.9, 0.1])
    with col2:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = None
            st.session_state.is_admin_login = False
            st.rerun()
    
    st.write(f"Welcome, **{st.session_state.user_data['name']}**")
    st.markdown("---")
    
    # Admin tabs
    admin_tabs = st.tabs(["👥 Users", "👨‍🏫 Teachers", "➕ Add Teacher"])
    
    # ===== USERS TAB =====
    with admin_tabs[0]:
        st.subheader("All Users")
        all_users = get_all_users()
        
        if all_users:
            st.metric("Total Users", len(all_users))
            st.markdown("---")
            
            for user in all_users:
                col1, col2 = st.columns([0.8, 0.2])
                
                with col1:
                    st.markdown(f"**{user['name']}** (@{user['username']})")
                    st.caption(f"Role: {user['role'].title()} | Email: {user['email']}")
                
                with col2:
                    if user['username'] != 'administrator':
                        if st.button("🗑️ Delete", key=f"del_user_{user['username']}", use_container_width=True):
                            success, message = delete_user(user['username'])
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
        else:
            st.info("No users found")
    
    # ===== TEACHERS TAB =====
    with admin_tabs[1]:
        st.subheader("Teachers List")
        all_users = get_all_users()
        teachers = [u for u in all_users if u['role'] == 'teacher']
        
        if teachers:
            st.metric("Total Teachers", len(teachers))
            st.markdown("---")
            
            for teacher in teachers:
                col1, col2 = st.columns([0.8, 0.2])
                
                with col1:
                    st.markdown(f"**{teacher['name']}** (@{teacher['username']})")
                    st.caption(f"Email: {teacher['email']}")
                
                with col2:
                    if st.button("🗑️ Delete", key=f"del_teacher_{teacher['username']}", use_container_width=True):
                        success, message = delete_user(teacher['username'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("No teachers created yet")
    
    # ===== ADD TEACHER TAB =====
    with admin_tabs[2]:
        st.subheader("Create New Teacher")
        
        with st.form("add_teacher_form"):
            teacher_username = st.text_input("Username (min 3)", placeholder="teacher_username")
            teacher_password = st.text_input("Password (min 6)", type="password", placeholder="temporary_password")
            teacher_name = st.text_input("Full Name", placeholder="Teacher Name")
            teacher_email = st.text_input("Email", placeholder="teacher@nfs.local")
            
            if st.form_submit_button("✅ Create Teacher Account", type="primary", use_container_width=True):
                if all([teacher_username, teacher_password, teacher_name, teacher_email]):
                    success, message = create_user(
                        teacher_username, 
                        teacher_password, 
                        "teacher", 
                        teacher_name, 
                        teacher_email
                    )
                    if success:
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ Please fill all fields")
    
    st.stop()

# --- HEADER WITH USER INFO (Teachers and Students) ---
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("📚 NFS")
with col2:
    st.write(f"**👤 {st.session_state.user_data['name']}**")
    st.caption(f"🎭 {st.session_state.user_data['role'].title()}")
with col3:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.session_state.is_admin_login = False
        st.rerun()

st.markdown("💬 Manage documents, videos, and ask questions using AI-powered retrieval.")
st.markdown("---")

# --- SIDEBAR (Teachers and Students) ---
with st.sidebar:
    st.header("📚 Content Management")
    
    if is_teacher:
        tabs = st.tabs(["📄 PDFs", "🎥 Videos", "⚙️ Manage"])
        upload_tab, video_tab, manage_tab = tabs
    else:
        tabs = st.tabs(["📚 Browse", "🎥 Videos"])
        upload_tab, video_tab = tabs
        manage_tab = None
    
    # ===== PDF UPLOAD TAB =====
    with upload_tab:
        if is_teacher:
            st.subheader("📤 Upload PDF Documents")
            
            uploaded_pdf = st.file_uploader(
                "Choose a PDF file", 
                type="pdf", 
                help="Upload a PDF document",
                key="pdf_uploader"
            )
            
            if uploaded_pdf:
                with st.spinner(f"Processing '{uploaded_pdf.name}'..."):
                    success, message = process_and_save_pdf(uploaded_pdf)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        else:
            st.subheader("📚 Available Documents")
            available_docs = get_available_documents()
            
            if available_docs:
                st.success(f"📊 {len(available_docs)} documents available")
                for doc in available_docs:
                    st.markdown(f"- 📄 {doc}")
            else:
                st.info("No documents available yet")
    
    # ===== VIDEO TAB =====
    with video_tab:
        if is_teacher:
            st.subheader("📤 Upload Videos")
            
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Upload a video file",
                key="video_uploader"
            )
            
            if uploaded_video:
                with st.spinner(f"Saving '{uploaded_video.name}'..."):
                    success, message, video_path = save_video(uploaded_video)
                    if success:
                        st.success(message)
                        st.session_state.last_uploaded_video = Path(uploaded_video.name).stem
                        st.rerun()
                    else:
                        st.error(message)
            
            st.markdown("---")
        
        st.subheader("🎬 Video Library")
        available_videos = get_available_videos()
        
        if available_videos:
            video_options = [v['name'] for v in available_videos]
            
            default_index = 0
            if 'last_uploaded_video' in st.session_state:
                try:
                    default_index = video_options.index(st.session_state.last_uploaded_video)
                except ValueError:
                    pass
            
            selected_preview_video = st.selectbox(
                "Select video:",
                options=video_options,
                index=default_index,
                key="preview_video_selector"
            )
            
            if selected_preview_video:
                matching_video = next((v for v in available_videos if v['name'] == selected_preview_video), None)
                
                if matching_video and os.path.exists(matching_video['path']):
                    with open(matching_video['path'], 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    if matching_video['has_captions'] and matching_video['caption_data']:
                        word_count = matching_video['caption_data'].get('word_count', 0)
                        st.success(f"✅ {word_count} words")
                    else:
                        st.info("ℹ️ No captions")
        else:
            st.info("📤 No videos uploaded yet")
        
        if is_teacher and available_videos:
            st.markdown("---")
            st.subheader("📝 Add Captions")
            
            video_options = [v['name'] for v in available_videos]
            selected_video = st.selectbox(
                "Video:",
                options=video_options,
                key="caption_video_selector"
            )
            
            caption_text = st.text_area(
                "Captions:",
                height=150,
                key="caption_input"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save", disabled=not caption_text):
                    success, message = save_caption_file(selected_video, caption_text)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            with col2:
                if st.button("🔍 Index", disabled=not caption_text):
                    save_caption_file(selected_video, caption_text)
                    success, message = process_video_captions(selected_video, caption_text)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
    
    # ===== MANAGE TAB (Teachers only) =====
    if manage_tab is not None:
        with manage_tab:
            st.subheader("📁 Manage Content")
            
            available_docs = get_available_documents()
            st.metric("Documents", len(available_docs))
            
            if available_docs:
                selected_doc = st.selectbox(
                    "Select:",
                    options=available_docs,
                    key="manage_doc_selector"
                )
                
                if selected_doc:
                    stats = get_document_stats(selected_doc)
                    if stats:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Chunks", stats['chunk_count'])
                        with col2:
                            st.metric("Type", stats['type'])
                    
                    st.markdown("---")
                    
                    if st.button("🗑️ Delete", type="secondary"):
                        success, message = delete_document(selected_doc)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

# --- MAIN CONTENT AREA (Teachers and Students) ---
available_docs = get_available_documents()

if available_docs:
    if "selected_doc" not in st.session_state or st.session_state.selected_doc not in available_docs:
        st.session_state.selected_doc = available_docs[0]
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_doc = st.selectbox(
            "📖 Select document to query:",
            options=available_docs,
            key="selected_doc"
        )
    
    with col2:
        stats = get_document_stats(selected_doc)
        if stats:
            doc_type = stats['type']
            type_icon = "📄" if doc_type == "pdf" else "🎥"
            st.metric("Type", f"{type_icon} {doc_type}")
    
    with col3:
        if stats:
            st.metric("Chunks", stats['chunk_count'])
    
    if stats and stats['type'] == 'video_caption':
        st.markdown("---")
        videos = get_available_videos()
        matching_video = next((v for v in videos if v['name'] == selected_doc), None)
        
        if matching_video and os.path.exists(matching_video['path']):
            with open(matching_video['path'], 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)

# --- CHAT INTERFACE (Teachers and Students) ---
if st.session_state.get('selected_doc'):
    st.markdown("---")
    st.subheader(f"💬 Chat: {st.session_state.selected_doc}")
    
    if "messages" not in st.session_state or st.session_state.get("current_doc") != st.session_state.selected_doc:
        st.session_state.messages = []
        st.session_state.current_doc = st.session_state.selected_doc
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Hi {st.session_state.user_data['name']}! Ask me anything about **{st.session_state.selected_doc}**."
        })
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if query := st.chat_input(f"Ask about {st.session_state.selected_doc}..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, retrieved_chunks = query_saved_document(
                        st.session_state.selected_doc, 
                        query
                    )
                    
                    st.markdown(answer)
                    
                    with st.expander("📄 View Source Context"):
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk)
                            if i < len(retrieved_chunks):
                                st.markdown("---")
                
                except Exception as e:
                    answer = f"Error: {str(e)}"
                    st.error(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Select or upload a document to start chatting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📄 PDFs
        1. Upload a PDF
        2. Select it
        3. Ask questions!
        """)
    
    with col2:
        st.markdown("""
        ### 🎥 Videos
        1. Upload video + captions
        2. Index for search
        3. Query the content!
        """)