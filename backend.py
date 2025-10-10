import os
import fitz
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import chromadb

# --- CONFIGURATION & GLOBAL MODELS ---
# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError: # Use the correct exception
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Load models once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qa_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Initialize ChromaDB client.
client = chromadb.PersistentClient(path="chroma_db")

# --- HELPER FUNCTION TO SANITIZE NAMES ---
def sanitize_collection_name(filename):
    """Sanitizes a filename to be a valid ChromaDB collection name."""
    # Remove the .pdf extension
    name_without_ext = filename.rsplit('.pdf', 1)[0]
    # Replace any character that is not a letter, number, underscore, dot, or hyphen with an underscore
    sanitized_name = re.sub(r'[^a-zA-Z0-9._-]', '_', name_without_ext)
    return sanitized_name

# --- TEXT PROCESSING FUNCTIONS ---
def extract_pdf(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return " ".join([page.get_text("text") for page in doc])

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def sentence_based_chunking(text, max_sentences=5, overlap=2):
    sentences = sent_tokenize(text)
    chunks = []
    if not sentences:
        return []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunks.append(" ".join(sentences[i:i+max_sentences]))
    return chunks

# --- CORE DATABASE LOGIC ---
def process_and_save_pdf(uploaded_file):
    """Processes a PDF and stores its chunks and embeddings in ChromaDB."""
    raw_text = extract_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)
    chunks = sentence_based_chunking(cleaned_text)
    
    if not chunks:
        return False, "Could not extract any text from the PDF."

    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    
    # Sanitize the filename here before creating the collection
    doc_name = sanitize_collection_name(uploaded_file.name)
    collection = client.get_or_create_collection(name=doc_name)
    
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
    return True, f"Successfully processed '{uploaded_file.name}'."

def query_saved_document(doc_name, query, k=5):
    """Queries a document stored in ChromaDB to get context for the LLM."""
    try:
        collection = client.get_collection(name=doc_name)
    except ValueError:
        return f"Error: The document '{doc_name}' was not found in the database.", []

    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=k)
    retrieved_chunks = results['documents'][0]
    
    context = "\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful teaching assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the context."

Context:
{context}

Question: {query}

Answer:
"""
    answer = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]
    return answer, retrieved_chunks

def get_available_documents():
    """Returns a list of processed document names from ChromaDB."""
    collections = client.list_collections()
    return sorted([col.name for col in collections])

