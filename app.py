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
    page_title="RAG Teaching Assistant", 
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION ---
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", "source_folder")
DB_PATH = "chroma_db"

# --- SESSION STATE INITIALIZATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_data' not in st.session_state:
    st.session_state.user_data = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False

# --- AUTHENTICATION PAGE ---
def show_auth_page():
    """Display the authentication page."""
    st.markdown("<h1 style='text-align: center;'>🧠 RAG Teaching Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>AI-Powered Document Q&A System</h3>", unsafe_allow_html=True)
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
                            st.success(f"✅ Welcome back, {user_data['name']}!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("⚠️ Please enter both username and password")
                
                if signup_button:
                    st.session_state.show_signup = True
                    st.rerun()
            
            st.info("💡 **Default Login:** Username: `admin` | Password: `admin123`")
        
        else:
            # SIGNUP FORM
            st.subheader("📝 Create New Account")
            
            with st.form("signup_form"):
                new_username = st.text_input("Username (min 3 characters)", placeholder="Choose a username")
                new_password = st.text_input("Password (min 6 characters)", type="password", placeholder="Choose a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
                full_name = st.text_input("Full Name", placeholder="Your full name")
                email = st.text_input("Email", placeholder="your.email@example.com")
                role = st.selectbox("I am a:", ["student", "teacher"])
                
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
                        success, message = create_user(new_username, new_password, role, full_name, email)
                        if success:
                            st.success(f"✅ {message}")
                            st.info("👉 You can now login with your credentials")
                            st.session_state.show_signup = False
                        else:
                            st.error(f"❌ {message}")
                
                if back_button:
                    st.session_state.show_signup = False
                    st.rerun()

# --- CHECK AUTHENTICATION ---
if not st.session_state.authenticated:
    show_auth_page()
    st.stop()

# --- ONE-TIME DATABASE INITIALIZATION ---
def initialize_database():
    """Initialize database from source folder if not already done."""
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

# Initialize database once per session after authentication
if 'db_initialized' not in st.session_state:
    with st.spinner("Initializing database..."):
        st.session_state.db_initialized = initialize_database()

# --- HEADER WITH USER INFO ---
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("🧠 RAG Teaching Assistant")
with col2:
    st.write(f"**👤 {st.session_state.user_data['name']}**")
    st.caption(f"🎭 {st.session_state.user_data['role'].title()}")
with col3:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_data = None
        st.rerun()

st.markdown("💬 Upload documents and videos, then ask questions using AI-powered retrieval.")
st.markdown("---")

# Check if user is teacher for admin features
is_teacher = st.session_state.user_data['role'] == 'teacher'

# --- SIDEBAR ---
with st.sidebar:
    st.header("📚 Content Management")
    
    # Create tabs based on role
    if is_teacher:
        tabs = st.tabs(["📄 PDFs", "🎥 Videos", "⚙️ Manage", "👥 Users"])
        upload_tab, video_tab, manage_tab, users_tab = tabs
    else:
        tabs = st.tabs(["📚 Browse", "🎥 Videos"])
        upload_tab, video_tab = tabs
        manage_tab = None
        users_tab = None
    
    # ===== PDF UPLOAD TAB =====
    with upload_tab:
        if is_teacher:
            st.subheader("📤 Upload PDF Documents")
            
            uploaded_pdf = st.file_uploader(
                "Choose a PDF file", 
                type="pdf", 
                help="Upload a PDF document to add to the knowledge base",
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
        
        # VIDEO VIEWER
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
                        caption_text = matching_video['caption_data'].get('full_text', '')
                        word_count = matching_video['caption_data'].get('word_count', 0)
                        st.success(f"✅ {word_count} words")
                    else:
                        st.info("ℹ️ No captions")
        else:
            st.info("📤 No videos uploaded yet")
        
        if is_teacher and available_videos:
            st.markdown("---")
            st.subheader("📝 Add Captions")
            
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
    
    # ===== USERS TAB (Teachers only) =====
    if users_tab is not None:
        with users_tab:
            st.subheader("👥 User Management")
            
            all_users = get_all_users()
            st.metric("Total Users", len(all_users))
            
            for user in all_users:
                with st.expander(f"{user['name']} (@{user['username']})"):
                    st.text(f"Role: {user['role'].title()}")
                    st.text(f"Email: {user['email']}")
                    
                    if user['username'] != 'admin':
                        if st.button(f"Delete", key=f"del_{user['username']}"):
                            success, message = delete_user(user['username'])
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)

# --- MAIN CONTENT AREA ---
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
    
    # VIDEO PLAYER for video captions
    if stats and stats['type'] == 'video_caption':
        st.markdown("---")
        videos = get_available_videos()
        matching_video = next((v for v in videos if v['name'] == selected_doc), None)
        
        if matching_video and os.path.exists(matching_video['path']):
            with open(matching_video['path'], 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)

# --- CHAT INTERFACE ---
if st.session_state.get('selected_doc'):
    st.markdown("---")
    st.subheader(f"💬 Chat: {st.session_state.selected_doc}")
    
    # Initialize chat history
    if "messages" not in st.session_state or st.session_state.get("current_doc") != st.session_state.selected_doc:
        st.session_state.messages = []
        st.session_state.current_doc = st.session_state.selected_doc
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"👋 Hi {st.session_state.user_data['name']}! Ask me anything about **{st.session_state.selected_doc}**."
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input(f"Ask about {st.session_state.selected_doc}..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    answer, retrieved_chunks = query_saved_document(
                        st.session_state.selected_doc, 
                        query
                    )
                    
                    st.markdown(answer)
                    
                    # Show source context
                    with st.expander("📄 View Source Context"):
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk)
                            if i < len(retrieved_chunks):
                                st.markdown("---")
                
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    st.error(answer)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please select or upload a document to start chatting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📄 For PDFs
        1. Upload a PDF (teachers)
        2. Select it from dropdown
        3. Ask questions!
        """)
    
    with col2:
        st.markdown("""
        ### 🎥 For Videos
        1. Upload video + captions
        2. Index for search
        3. Query the content!
        """)