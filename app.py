import streamlit as st
from backend import get_answer

st.set_page_config(page_title="RAG Teaching Assistant", layout="wide")
st.title("🧠 RAG Teaching Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
query = st.text_input("Ask a question:")

if uploaded_file and query:
    answer, retrieved_chunks = get_answer(uploaded_file, query)
    
    st.subheader("Answer:")
    st.write(answer)
    
    st.subheader("Retrieved Context:")
    for chunk in retrieved_chunks:
        st.write("-", chunk)