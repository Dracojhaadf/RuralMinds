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
except LookupError:
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

# Initialize ChromaDB client. It will store data in a 'chroma_db' folder.
client = chromadb.PersistentClient(path="chroma_db")

# --- TEXT PROCESSING FUNCTIONS ---
def extract_pdf(file_stream):
    """Extracts text from a file-like PDF object."""
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    return " ".join([page.get_text("text") for page in doc])

def clean_text(text):
    """Cleans the extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def sentence_based_chunking(text, max_sentences=5, overlap=2):
    """Splits text into sentence-based chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    if not sentences:
        return []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunks.append(" ".join(sentences[i:i+max_sentences]))
    return chunks

# --- DATABASE CORE LOGIC ---

def process_and_save_pdf(uploaded_file):
    """Processes a PDF and stores its chunks and embeddings in ChromaDB."""
    
    # 1. Extract, clean, and chunk text
    raw_text = extract_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)
    chunks = sentence_based_chunking(cleaned_text)
    
    if not chunks:
        return False, "Could not extract any text chunks from the PDF."

    # 2. Create embeddings
    embeddings = embed_model.encode(chunks, show_progress_bar=False)
    
    # 3. Store in ChromaDB
    doc_name = uploaded_file.name.replace('.pdf', '')
    collection = client.get_or_create_collection(name=doc_name)
    
    # Generate unique IDs for each chunk
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids
    )
        
    return True, f"Successfully processed and saved '{uploaded_file.name}' to the database."

def query_saved_document(doc_name, query, k=5):
    """Queries a document stored in ChromaDB to get context for the LLM."""
    
    # 1. Get the document's collection from ChromaDB
    try:
        collection = client.get_collection(name=doc_name)
    except ValueError:
        return f"Error: The document '{doc_name}' was not found in the database.", []

    # 2. Embed the query and search for similar chunks
    query_emb = embed_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k
    )
    retrieved_chunks = results['documents'][0]
    
    # 3. Build prompt and get answer from LLM
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

