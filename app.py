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
import os
import chromadb
from pathlib import Path

# --- PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(
    page_title="RAG Teaching Assistant", 
    layout="wide",
    page_icon="🧠"
)

# --- CONFIGURATION ---
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", "source_folder")
DB_PATH = "chroma_db"

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

# Initialize database once per session
if 'db_initialized' not in st.session_state:
    with st.spinner("Initializing database..."):
        st.session_state.db_initialized = initialize_database()
    if st.session_state.db_initialized:
        st.rerun()

# --- HEADER ---
st.title("🧠 RAG Teaching Assistant")
st.markdown("Upload documents and videos, then ask questions using AI-powered retrieval.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("📚 Content Management")
    
    # Create tabs for different upload types
    upload_tab, video_tab, manage_tab = st.tabs(["📄 PDFs", "🎥 Videos", "⚙️ Manage"])
    
    # ===== PDF UPLOAD TAB =====
    with upload_tab:
        st.subheader("Upload PDF Documents")
        
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
    
    # ===== VIDEO UPLOAD TAB =====
    with video_tab:
        st.subheader("Upload Videos & Captions")
        
        # Video upload
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
        
        # VIDEO VIEWER - Preview any uploaded video
        st.subheader("📹 Video Viewer")
        available_videos = get_available_videos()
        
        if available_videos:
            video_options = [v['name'] for v in available_videos]
            
            # Auto-select last uploaded video
            default_index = 0
            if 'last_uploaded_video' in st.session_state:
                try:
                    default_index = video_options.index(st.session_state.last_uploaded_video)
                except ValueError:
                    pass
            
            selected_preview_video = st.selectbox(
                "Select video to preview:",
                options=video_options,
                index=default_index,
                key="preview_video_selector"
            )
            
            if selected_preview_video:
                # Find the video file
                matching_video = next((v for v in available_videos if v['name'] == selected_preview_video), None)
                
                if matching_video and os.path.exists(matching_video['path']):
                    # Display video player
                    with open(matching_video['path'], 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # Show status
                    if matching_video['has_captions']:
                        st.success(f"✅ This video has captions ({matching_video['caption_data'].get('word_count', 0)} words)")
                    else:
                        st.info("ℹ️ No captions yet. Add captions below to enable search.")
        else:
            st.info("📤 No videos uploaded yet.")
        
        st.markdown("---")
        
        # Caption upload section
        st.subheader("Add Captions")
        
        # Get available videos
        available_videos = get_available_videos()
        
        if available_videos:
            # Select video for caption
            video_options = [v['name'] for v in available_videos]
            
            # Auto-select last uploaded video if available
            default_index = 0
            if 'last_uploaded_video' in st.session_state:
                try:
                    default_index = video_options.index(st.session_state.last_uploaded_video)
                except ValueError:
                    pass
            
            selected_video = st.selectbox(
                "Select video for captions:",
                options=video_options,
                index=default_index,
                key="caption_video_selector"
            )
            
            # Check if captions already exist
            caption_data = load_caption_file(selected_video)
            
            if caption_data:
                st.info(f"✓ Captions exist ({caption_data.get('word_count', 0)} words)")
                if st.button("View Existing Captions"):
                    st.text_area(
                        "Current Captions:",
                        value=caption_data.get('full_text', ''),
                        height=200,
                        disabled=True
                    )
            
            # Caption input
            caption_text = st.text_area(
                "Enter or paste captions:",
                height=200,
                help="Enter the transcript/captions for the video",
                key="caption_input"
            )
            
            # Option for timestamped captions (future feature)
            with st.expander("🕐 Timestamped Captions (Optional)"):
                st.info("Coming soon: Add timestamps to your captions")
                st.text("Format: [00:00:15] First sentence here.\n[00:00:30] Second sentence here.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save Captions", type="primary", disabled=not caption_text):
                    with st.spinner("Saving captions..."):
                        # Save caption file
                        success, message = save_caption_file(selected_video, caption_text)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
            
            with col2:
                if st.button("🔍 Index for Search", disabled=not caption_text):
                    with st.spinner("Processing and indexing captions..."):
                        # Save and process captions
                        save_caption_file(selected_video, caption_text)
                        success, message = process_video_captions(selected_video, caption_text)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
        else:
            st.info("📤 No videos uploaded yet. Upload a video above to add captions.")
    
    # ===== MANAGE TAB =====
    with manage_tab:
        st.subheader("Document Library")
        
        available_docs = get_available_documents()
        st.metric("Total Documents", len(available_docs))
        
        if available_docs:
            # Document selector
            selected_doc = st.selectbox(
                "Select document:",
                options=available_docs,
                key="manage_doc_selector"
            )
            
            if selected_doc:
                # Show document stats
                stats = get_document_stats(selected_doc)
                if stats:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Chunks", stats['chunk_count'])
                    with col2:
                        doc_type_icon = "📄" if stats['type'] == "pdf" else "🎥"
                        st.metric("Type", f"{doc_type_icon} {stats['type']}")
                
                # Delete button with proper confirmation
                st.markdown("---")
                st.warning("⚠️ Deletion Warning")
                
                # Use a unique key for each document's delete confirmation
                confirm_key = f"confirm_delete_{selected_doc}"
                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False
                
                if st.button("🗑️ Delete Document", type="secondary", key=f"delete_btn_{selected_doc}"):
                    st.session_state[confirm_key] = True
                
                if st.session_state[confirm_key]:
                    st.error(f"Are you sure you want to delete '{selected_doc}'?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Yes, Delete", type="primary", key=f"confirm_yes_{selected_doc}"):
                            success, message = delete_document(selected_doc)
                            if success:
                                st.success(message)
                                st.session_state[confirm_key] = False
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with col2:
                        if st.button("❌ Cancel", key=f"confirm_no_{selected_doc}"):
                            st.session_state[confirm_key] = False
                            st.rerun()
        
        st.markdown("---")
        
        # Video library
        st.subheader("Video Library")
        videos = get_available_videos()
        st.metric("Total Videos", len(videos))
        
        if videos:
            for video in videos:
                with st.expander(f"{'✅' if video['has_captions'] else '❌'} {video['filename']}"):
                    st.text(f"Name: {video['name']}")
                    st.text(f"Captions: {'Yes' if video['has_captions'] else 'No'}")
                    if video['has_captions'] and video['caption_data']:
                        st.text(f"Words: {video['caption_data'].get('word_count', 0)}")
                    
                    # Delete video button
                    if st.button(f"🗑️ Delete Video", key=f"delete_video_{video['name']}"):
                        success, message = delete_video(video['name'])
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)

# --- MAIN CONTENT AREA ---
available_docs = get_available_documents()

# Document selector in main area
if available_docs:
    st.markdown("---")
    
    # Auto-select first document if none selected
    if "selected_doc" not in st.session_state or st.session_state.selected_doc not in available_docs:
        st.session_state.selected_doc = available_docs[0]
    
    # Document selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_doc = st.selectbox(
            "📖 Select document to query:",
            options=available_docs,
            key="selected_doc",
            label_visibility="visible"
        )
    
    with col2:
        # Show document type
        stats = get_document_stats(selected_doc)
        if stats:
            doc_type = stats['type']
            type_icon = "📄" if doc_type == "pdf" else "🎥" if doc_type == "video_caption" else "📝"
            st.metric("Type", f"{type_icon} {doc_type}")
    
    with col3:
        # Show chunk count
        if stats:
            st.metric("Chunks", stats['chunk_count'])
    
    # VIDEO PLAYER - Show video if document type is video_caption
    if stats and stats['type'] == 'video_caption':
        st.markdown("---")
        st.subheader("📹 Video Player")
        
        # Find the video file
        videos = get_available_videos()
        matching_video = next((v for v in videos if v['name'] == selected_doc), None)
        
        if matching_video and os.path.exists(matching_video['path']):
            # Display video
            with open(matching_video['path'], 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            # Show caption preview if available
            if matching_video['has_captions']:
                with st.expander("📝 View Full Transcript"):
                    caption_data = matching_video['caption_data']
                    st.text_area(
                        "Transcript",
                        value=caption_data.get('full_text', ''),
                        height=300,
                        disabled=True
                    )
        else:
            st.warning("⚠️ Video file not found on disk.")

# --- CHAT INTERFACE ---
if st.session_state.get('selected_doc'):
    st.header(f"💬 Chat with: `{st.session_state.selected_doc}`")
    
    # Initialize or reset chat history when document changes
    if "messages" not in st.session_state or st.session_state.get("current_doc") != st.session_state.selected_doc:
        st.session_state.messages = []
        st.session_state.current_doc = st.session_state.selected_doc
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"👋 Hi! I'm ready to answer questions about **{st.session_state.selected_doc}**. What would you like to know?"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if query := st.chat_input(f"Ask a question about {st.session_state.selected_doc}..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, retrieved_chunks = query_saved_document(
                        st.session_state.selected_doc, 
                        query
                    )
                    st.markdown(answer)
                    
                    # Show retrieved context
                    with st.expander("📄 View Retrieved Context"):
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(chunk)
                            if i < len(retrieved_chunks):
                                st.markdown("---")
                
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"
                    st.error(answer)
                    retrieved_chunks = []
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    # No document selected
    st.info("👈 Please upload a PDF or video from the sidebar, then select it to start chatting.")
    
    # Getting started guide
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📄 PDFs
        
        1. Go to **📄 PDFs** tab in sidebar
        2. Upload your PDF document
        3. Wait for processing
        4. Start asking questions!
        """)
    
    with col2:
        st.markdown("""
        ### 🎥 Videos
        
        1. Go to **🎥 Videos** tab in sidebar
        2. Upload your video file
        3. Add captions/transcript
        4. Click "Index for Search"
        5. Ask questions about the video!
        """)
    
    st.markdown("""
    ---
    ### ✨ Features
    - 🔍 Semantic search across documents
    - 💡 AI-powered answers with source citations
    - 📚 Support for PDFs and video captions
    - 🎬 Built-in video player for video content
    - ⚙️ Easy content management
    """)