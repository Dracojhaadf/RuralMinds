import fitz
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

nltk.download('punkt')

# Load embedding model and LLM globally (so they are loaded only once)
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

def extract_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = [page.get_text("text") for page in doc]
    return " ".join(pages)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def sentence_based_chunking(text, max_sentences=5, overlap=2):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap):
        chunks.append(" ".join(sentences[i:i+max_sentences]))
    return chunks

def get_answer(uploaded_file, query, k=5):
    # Extract & clean text
    raw_text = extract_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)
    
    # Chunk text
    sentence_chunks = sentence_based_chunking(cleaned_text)
    
    # Create embeddings & FAISS index
    embeddings = embed_model.encode(sentence_chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    index_flat = faiss.IndexFlatIP(d)
    index = faiss.IndexIDMap(index_flat)
    ids = np.arange(len(embeddings)).astype('int64')
    index.add_with_ids(embeddings, ids)
    
    # Embed query and search
    query_emb = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, k)
    
    retrieved_chunks = [sentence_chunks[i] for i in indices[0]]
    
    # Build prompt for LLM
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