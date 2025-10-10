import streamlit as st
from backend import process_and_save_pdf, query_saved_document, get_available_documents
import os
import chromadb

# --- 0. ONE-TIME DATABASE INITIALIZATION ---
# This block runs once per session to ensure the database is fresh and reflects the source_folder.
SOURCE_FOLDER = r"C:\oi\NFS---HackQuest-25\source_folder"

# Use a session state flag to ensure this logic runs only once per session
if 'db_initialized' not in st.session_state:
    with st.spinner(f"Initializing database from source folder..."):
        # Clear all existing collections from the database to ensure a fresh start
        client = chromadb.PersistentClient(path="chroma_db")
        for collection in client.list_collections():
            client.delete_collection(name=collection.name)
        
        # Ingest all PDF files from the specified source folder
        if os.path.exists(SOURCE_FOLDER):
            pdf_files = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith(".pdf")]
            
            # A simple helper class to mimic Streamlit's UploadedFile object structure
            # so we can reuse the existing 'process_and_save_pdf' function
            class LocalFile:
                def __init__(self, path):
                    self.path = path
                    self.name = os.path.basename(path)
                
                def read(self):
                    with open(self.path, "rb") as f:
                        return f.read()

            # Process each PDF found in the folder
            for pdf_file in pdf_files:
                file_path = os.path.join(SOURCE_FOLDER, pdf_file)
                local_file_obj = LocalFile(file_path)
                process_and_save_pdf(local_file_obj)
        else:
            # Display a warning if the source folder doesn't exist
            st.warning(f"Source folder not found at: {SOURCE_FOLDER}")

    # Set the flag to prevent this block from running again in this session
    st.session_state.db_initialized = True
    # Rerun the script to ensure the UI updates with the new document list
    st.rerun()

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="RAG Teaching Assistant", layout="wide")
st.title("🧠 RAG Teaching Assistant")
st.markdown("This app uses a pre-loaded database of documents. You can also upload new ones.")

# --- 2. SIDEBAR FOR DOCUMENT MANAGEMENT ---
with st.sidebar:
    st.header("📚 Document Management")
    
    # PDF uploader
    uploaded_file = st.file_uploader("Upload a new PDF to the database", type="pdf")
    if uploaded_file:
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            success, message = process_and_save_pdf(uploaded_file)
            if success:
                st.success(message)
                # Force a rerun to update the document list and select the new doc
                st.rerun() 
            else:
                st.error(message)

    st.markdown("---")
    
    # Get the list of documents already in the database
    available_docs = get_available_documents()

    # Logic to automatically select a document
    if available_docs:
        # If no document is selected in the session, default to the first one
        if "selected_doc" not in st.session_state or st.session_state.selected_doc not in available_docs:
            st.session_state.selected_doc = available_docs[0]

        # Display the dropdown to select the active document
        st.selectbox(
            "Active Document:",
            options=available_docs,
            key="selected_doc" # Link this widget to the session state
        )
    else:
        st.info("The database is empty. Upload a PDF to begin.")
        st.session_state.selected_doc = None

# --- 3. MAIN CHAT INTERFACE ---
if st.session_state.selected_doc:
    st.header(f"Chatting with: `{st.session_state.selected_doc}.pdf`")

    # Initialize or clear chat history when the document changes
    if "messages" not in st.session_state or st.session_state.get("current_doc") != st.session_state.selected_doc:
        st.session_state.messages = []
        st.session_state.current_doc = st.session_state.selected_doc
        st.session_state.messages.append({"role": "assistant", "content": f"Ask me anything about '{st.session_state.selected_doc}.pdf'!"})

    # Display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input(f"Ask a question about {st.session_state.selected_doc}..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, retrieved_chunks = query_saved_document(st.session_state.selected_doc, query)
                st.markdown(answer)
                with st.expander("View Retrieved Context"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.write(f"**Chunk {i+1}:** {chunk}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please run the 'ingest.py' script or upload a PDF to start the chat.")

