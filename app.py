import streamlit as st
from backend import process_and_save_pdf, query_saved_document, get_available_documents

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
    st.info("Please upload a PDF document using the sidebar to start the chat.")
