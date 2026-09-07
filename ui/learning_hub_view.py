import os
import streamlit as st
from services.document_service import process_and_save_pdf
from services.vector_service import (
    get_available_documents,
    get_document_path,
    get_document_stats,
    delete_document
)
from services.video_service import (
    save_video,
    get_available_videos,
    delete_video
)
from services.audio_service import transcribe_audio
from services.llm_service import query_saved_document_stream

def render_learning_hub(is_teacher: bool):
    """Render the Learning Hub tab for both teachers and students."""
    # SIDEBAR CONTENT
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
                    if 'processed_pdfs' not in st.session_state:
                        st.session_state.processed_pdfs = set()
                    
                    pdf_key = f"{pdf.name}_{pdf.size}"
                    if pdf_key not in st.session_state.processed_pdfs:
                        with st.spinner("Processing PDF... please wait"):
                            success, msg = process_and_save_pdf(pdf)
                        if success:
                            st.session_state.processed_pdfs.add(pdf_key)
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.success(f"✅ '{pdf.name}' already processed.")
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
                    if 'processed_videos' not in st.session_state:
                        st.session_state.processed_videos = set()
                    
                    vid_key = f"{vid.name}_{vid.size}"
                    if vid_key not in st.session_state.processed_videos:
                        with st.spinner("Saving video..."):
                            success, msg, path = save_video(vid)
                        if success:
                            st.session_state.processed_videos.add(vid_key)
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.success(f"✅ '{vid.name}' already saved.")
                
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
                
                # Delete Documents
                docs = get_available_documents()
                if docs:
                    st.markdown("**Documents**")
                    sel_del = st.selectbox("Select PDF to remove:", docs, key="del_sel")
                    if st.button("🗑️ Delete Document"):
                        success, msg = delete_document(sel_del)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                            
                st.markdown("---")
                
                # Delete Videos
                vids = get_available_videos()
                if vids:
                    st.markdown("**Videos**")
                    vid_del_sel = st.selectbox("Select Video to remove:", [v['name'] for v in vids], key="vid_del_sel")
                    if st.button("🗑️ Delete Video"):
                        success, msg = delete_video(vid_del_sel)
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        
        st.markdown("---")
        st.header("💬 Documents")
        docs = get_available_documents()
        if docs:
            docs_options = ["-- Search All Documents --"] + docs
            sel = st.selectbox("Select Document:", docs_options, key="selected_doc_sidebar")
            
            stats = get_document_stats(sel)
            if stats:
                st.caption(f"Type: {stats.get('type','PDF')} | Pages: {stats.get('page_count','N/A')}")
                
            pdf_path = get_document_path(sel)
            if pdf_path and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()
                    
                st.download_button(
                    label="⬇️ Download Source PDF",
                    data=pdf_bytes,
                    file_name=f"{sel}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            sel = None
            st.info("No documents available.")

    # MAIN CHAT AREA
    if sel:
        chat_container = st.container()
        
        if 'messages' not in st.session_state or st.session_state.get('current_doc') != sel:
            st.session_state.messages = []
            st.session_state.current_doc = sel
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"👋 Hi! I'm ready to help you with **{sel}**."
            })
        
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        col_voice, col_text = st.columns([0.2, 0.8])
        
        with col_voice:
            voice_lang = st.selectbox(
                "Voice Language",
                ["EN", "HI", "ML"],
                label_visibility="collapsed",
                key="voice_lang_selector"
            )
            audio_val = st.audio_input("Record", label_visibility="collapsed")
            
        with col_text:
            text_query = st.chat_input(f"Ask about {sel}...")
        
        query = None
        if text_query:
            query = text_query
        elif audio_val:
            lang_map = {"EN": "en", "HI": "hi", "ML": "ml"}
            lang_code = lang_map.get(voice_lang, "en")
            
            with st.spinner("🎧 Transcribing..."):
                temp_filename = "temp_voice_query.wav"
                with open(temp_filename, "wb") as f:
                    f.write(audio_val.read())
                
                success, msg, data = transcribe_audio(temp_filename, lang_code)
                
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except Exception:
                        pass
                
                if not success:
                    st.error(f"❌ {msg}")
                    query = None
                else:
                    query = data.get("text", "").strip()
        
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.write(query)
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                sources = []
                
                with message_placeholder.container():
                    st.markdown("""
                        <div class="thinking-container">
                            <div class="loader-box">
                                <div class="loader-circle"></div>
                            </div>
                            <span class="thinking-text">Thinking...</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                try:
                    lang_map = {"EN": "en", "HI": "hi", "ML": "ml"}
                    forced_lang = lang_map.get(voice_lang, "en")

                    for chunk in query_saved_document_stream(
                        sel, 
                        query,
                        forced_language=forced_lang
                    ):
                        if isinstance(chunk, dict) and 'sources' in chunk:
                            sources = chunk['sources']
                            continue
                            
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    if sources:
                        with st.expander("retrieved context"):
                            for i, s_chunk in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.caption(s_chunk)
                                if i < len(sources):
                                    st.markdown("---")
                                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                except Exception as e:
                    message_placeholder.error(f"❌ Error: {str(e)}")

    else:
        st.empty()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👈 Please upload or select a document in the sidebar to start chatting.")
