# -*- coding: utf-8 -*-
"""
RAG Teaching Assistant - Streamlit App
"""

import streamlit as st
import os
import re
import pickle
import numpy as np
import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

nltk.download('punkt')
nltk.download('punkt_tab')
# =========================
# Page Config
# =========================
st.set_page_config(page_title="🧠 RAG Teaching Assistant", layout="wide")
st.title("🧠 RAG Teaching Assistant")
st.write("Upload a PDF and ask questions to get intelligent answers based on the document.")

# =========================
# PDF Upload
# =========================
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("PDF uploaded successfully!")

    # =========================
    # Extract & Clean Text
    # =========================
    def extract_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        return " ".join(pages)

    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text.strip()

    raw_text = extract_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)

    # =========================
    # Sentence-based Chunking
    # =========================
    def sentence_based_chunking(text, max_sentences=5, overlap=2):
        sentences = sent_tokenize(text)
        chunks = []
        for i in range(0, len(sentences), max_sentences - overlap):
            chunks.append(" ".join(sentences[i:i+max_sentences]))
        return chunks

    sentence_chunks = sentence_based_chunking(cleaned_text, max_sentences=5, overlap=2)
    st.write(f"Document split into {len(sentence_chunks)} chunks.")

    # =========================
    # Create Embeddings & FAISS Index
    # =========================
    st.write("Creating embeddings and building FAISS index...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embed_model.encode(sentence_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    index_flat = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(index_flat)
    ids = np.arange(len(embeddings)).astype('int64')
    index.add_with_ids(embeddings, ids)
    st.success(f"FAISS index created with {index.ntotal} vectors!")

    # =========================
    # Ask Questions
    # =========================
    query = st.text_input("Ask a question about the PDF:")
    if query:
        query_emb = embed_model.encode([query]).astype("float32")
        faiss.normalize_L2(query_emb)
        k = 5
        distances, indices = index.search(query_emb, k)

        retrieved_chunks = [sentence_chunks[i] for i in indices[0]]
        st.subheader("Retrieved Context:")
        for i, chunk in enumerate(retrieved_chunks):
            st.write(f"{i+1}. {chunk}")

        # =========================
        # Build Prompt for LLM
        # =========================
        def build_prompt(chunks, question):
            context = "\n".join(chunks)
            prompt = f"""
You are a helpful teaching assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the context."

Context:
{context}

Question: {question}

Answer:
"""
            return prompt

        prompt = build_prompt(retrieved_chunks, query)

        # =========================
        # Hugging Face LLM
        # =========================
        model_name = "google/flan-t5-base"  # lightweight model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"  # GPU if available
        )

        answer = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
        st.subheader("Answer:")
        st.write(answer)