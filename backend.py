import os
import fitz
import re
import json
import subprocess
from pathlib import Path
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import chromadb
from typing import Tuple, List, Optional
import logging
from datetime import datetime

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
MAX_SENTENCES_PER_CHUNK = 5
SENTENCE_OVERLAP = 2
DEFAULT_K_RESULTS = 5
MAX_ANSWER_TOKENS = 200

# Video-specific paths
VIDEO_STORAGE_PATH = "static/videos"
CAPTIONS_STORAGE_PATH = "captions"

# Ensure directories exist
os.makedirs(VIDEO_STORAGE_PATH, exist_ok=True)
os.makedirs(CAPTIONS_STORAGE_PATH, exist_ok=True)

# --- NLTK DATA INITIALIZATION ---
def ensure_nltk_data():
    """Ensure required NLTK data is available."""
    required_resources = ['tokenizers/punkt', 'tokenizers/punkt_tab']
    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            package_name = resource.split('/')[-1]
            nltk.download(package_name, quiet=True)

ensure_nltk_data()

# --- MODEL INITIALIZATION (LAZY LOADING) ---
_embed_model = None
_qa_pipeline = None
_client = None

def get_embedding_model():
    """Lazy load the embedding model."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model

def get_qa_pipeline():
    """Lazy load the QA pipeline."""
    global _qa_pipeline
    if _qa_pipeline is None:
        logger.info(f"Loading LLM model: {LLM_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
        _qa_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
    return _qa_pipeline

def get_client():
    """Get or create ChromaDB client."""
    global _client
    if _client is None:
        logger.info(f"Initializing ChromaDB at: {DB_PATH}")
        _client = chromadb.PersistentClient(path=DB_PATH)
    return _client

# --- HELPER FUNCTIONS ---
def sanitize_collection_name(filename: str) -> str:
    """Sanitizes a filename to be a valid ChromaDB collection name."""
    # Remove file extension
    name = Path(filename).stem
    
    # Replace invalid characters with underscore
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    
    # Ensure it starts and ends with alphanumeric
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    
    # Ensure minimum length
    if len(name) < 3:
        name = f"doc_{name}"
    
    # Ensure maximum length
    if len(name) > 63:
        name = name[:63]
    
    # Convert to lowercase for consistency
    name = name.lower()
    
    return name

# --- PDF TEXT EXTRACTION & PROCESSING ---
def extract_pdf(file_stream) -> str:
    """Extract text from a PDF file stream."""
    try:
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text_parts = []
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip():
                text_parts.append(text)
        
        doc.close()
        
        if not text_parts:
            logger.warning("No text extracted from PDF")
            return ""
        
        return " ".join(text_parts)
    
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-ASCII characters (optional)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove multiple spaces
    text = ' '.join(text.split())
    
    return text.strip()

def sentence_based_chunking(
    text: str, 
    max_sentences: int = MAX_SENTENCES_PER_CHUNK, 
    overlap: int = SENTENCE_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks based on sentences."""
    if not text or not text.strip():
        return []
    
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing sentences: {str(e)}")
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    overlap = min(overlap, max_sentences - 1)
    
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = " ".join(sentences[i:i+max_sentences])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

# --- PDF DATABASE LOGIC ---
def process_and_save_pdf(uploaded_file):
    """Processes a PDF and stores its chunks and embeddings in ChromaDB."""
    try:
        raw_text = extract_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = sentence_based_chunking(cleaned_text)
        
        if not chunks:
            return False, "Could not extract any text from the PDF."

        embed_model = get_embedding_model()
        embeddings = embed_model.encode(chunks, show_progress_bar=False)
        
        doc_name = sanitize_collection_name(uploaded_file.name)
        client = get_client()
        collection = client.get_or_create_collection(name=doc_name)
        
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]

        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{"type": "pdf", "chunk_id": i} for i in range(len(chunks))]
        )
        return True, f"Successfully processed '{uploaded_file.name}'."
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return False, f"Error: {str(e)}"

# --- VIDEO PROCESSING FUNCTIONS ---
def save_video(uploaded_file) -> Tuple[bool, str, Optional[str]]:
    """
    Save an uploaded video file to the static/videos directory.
    
    Returns:
        Tuple of (success, message, video_path)
    """
    try:
        # Sanitize filename
        original_name = uploaded_file.name
        safe_name = sanitize_collection_name(original_name)
        extension = Path(original_name).suffix.lower()
        
        # Add timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}{extension}"
        video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
        
        # Save video file
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        logger.info(f"Video saved: {video_path}")
        return True, f"Video '{original_name}' saved successfully.", video_path
    
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return False, f"Error saving video: {str(e)}", None

def save_caption_file(video_name: str, caption_text: str, timestamps: Optional[List[dict]] = None) -> Tuple[bool, str]:
    """
    Save caption text for a video as a JSON file.
    
    Args:
        video_name: Name of the video file (without extension)
        caption_text: Full caption text
        timestamps: Optional list of timestamp dictionaries with 'start', 'end', and 'text'
        
    Returns:
        Tuple of (success, message)
    """
    try:
        safe_name = sanitize_collection_name(video_name)
        caption_filename = f"{safe_name}_captions.json"
        caption_path = os.path.join(CAPTIONS_STORAGE_PATH, caption_filename)
        
        caption_data = {
            "video_name": video_name,
            "created_at": datetime.now().isoformat(),
            "full_text": caption_text,
            "timestamps": timestamps or [],
            "word_count": len(caption_text.split())
        }
        
        with open(caption_path, "w", encoding="utf-8") as f:
            json.dump(caption_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Caption saved: {caption_path}")
        return True, f"Caption saved for '{video_name}'."
    
    except Exception as e:
        logger.error(f"Error saving caption: {str(e)}")
        return False, f"Error saving caption: {str(e)}"

def load_caption_file(video_name: str) -> Optional[dict]:
    """Load caption data for a video."""
    try:
        safe_name = sanitize_collection_name(video_name)
        caption_filename = f"{safe_name}_captions.json"
        caption_path = os.path.join(CAPTIONS_STORAGE_PATH, caption_filename)
        
        if not os.path.exists(caption_path):
            return None
        
        with open(caption_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    except Exception as e:
        logger.error(f"Error loading caption: {str(e)}")
        return None

def process_video_captions(video_name: str, caption_text: str) -> Tuple[bool, str]:
    """
    Process video captions and store them in ChromaDB for querying.
    
    Args:
        video_name: Name of the video
        caption_text: Caption/transcript text
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Clean and chunk the caption text
        cleaned_text = clean_text(caption_text)
        chunks = sentence_based_chunking(cleaned_text)
        
        if not chunks:
            return False, "Could not create chunks from caption text."
        
        # Create embeddings
        embed_model = get_embedding_model()
        embeddings = embed_model.encode(chunks, show_progress_bar=False)
        
        # Store in ChromaDB
        doc_name = sanitize_collection_name(video_name)
        client = get_client()
        collection = client.get_or_create_collection(name=doc_name)
        
        ids = [f"{doc_name}_caption_chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{"type": "video_caption", "chunk_id": i} for i in range(len(chunks))]
        )
        
        return True, f"Captions for '{video_name}' indexed successfully."
    
    except Exception as e:
        logger.error(f"Error processing video captions: {str(e)}")
        return False, f"Error: {str(e)}"

def get_available_videos() -> List[dict]:
    """Get list of available videos with their caption status."""
    videos = []
    
    if not os.path.exists(VIDEO_STORAGE_PATH):
        return videos
    
    for filename in os.listdir(VIDEO_STORAGE_PATH):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
            video_name = Path(filename).stem
            
            # Check if captions exist
            caption_data = load_caption_file(video_name)
            has_captions = caption_data is not None
            
            videos.append({
                "filename": filename,
                "name": video_name,
                "path": video_path,
                "has_captions": has_captions,
                "caption_data": caption_data
            })
    
    return sorted(videos, key=lambda x: x['filename'])

def delete_video(video_name: str) -> Tuple[bool, str]:
    """
    Delete a video file and its associated caption file.
    Also deletes the ChromaDB collection if it exists.
    
    Args:
        video_name: Name of the video (without extension)
        
    Returns:
        Tuple of (success, message)
    """
    try:
        deleted_items = []
        
        # 1. Delete video file(s)
        if os.path.exists(VIDEO_STORAGE_PATH):
            for filename in os.listdir(VIDEO_STORAGE_PATH):
                # Check if this file matches the video name
                if Path(filename).stem == video_name:
                    video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
                    os.remove(video_path)
                    deleted_items.append(f"video file: {filename}")
                    logger.info(f"Deleted video file: {video_path}")
        
        # 2. Delete caption file
        safe_name = sanitize_collection_name(video_name)
        caption_filename = f"{safe_name}_captions.json"
        caption_path = os.path.join(CAPTIONS_STORAGE_PATH, caption_filename)
        
        if os.path.exists(caption_path):
            os.remove(caption_path)
            deleted_items.append("caption file")
            logger.info(f"Deleted caption file: {caption_path}")
        
        # 3. Delete ChromaDB collection if it exists
        try:
            client = get_client()
            collection_name = sanitize_collection_name(video_name)
            client.delete_collection(name=collection_name)
            deleted_items.append("database collection")
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
        except Exception as e:
            # Collection might not exist, which is fine
            logger.info(f"No ChromaDB collection found for {video_name}: {str(e)}")
        
        if deleted_items:
            items_str = ", ".join(deleted_items)
            return True, f"Successfully deleted {items_str} for '{video_name}'."
        else:
            return False, f"No files found for video '{video_name}'."
    
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        return False, f"Error deleting video: {str(e)}"

# --- QUERY FUNCTIONS ---
def query_saved_document(doc_name: str, query: str, k: int = DEFAULT_K_RESULTS):
    """Queries a document (PDF or video caption) stored in ChromaDB."""
    try:
        client = get_client()
        collection = client.get_collection(name=doc_name)
    except ValueError:
        return f"Error: The document '{doc_name}' was not found in the database.", []

    embed_model = get_embedding_model()
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
    qa_pipeline = get_qa_pipeline()
    answer = qa_pipeline(prompt, max_new_tokens=MAX_ANSWER_TOKENS)[0]["generated_text"]
    return answer, retrieved_chunks

def get_available_documents():
    """Returns a list of all processed documents (PDFs and videos) from ChromaDB."""
    client = get_client()
    collections = client.list_collections()
    return sorted([col.name for col in collections])

def delete_document(doc_name: str) -> Tuple[bool, str]:
    """
    Delete a document collection from ChromaDB.
    This only deletes the database collection, not the source files.
    """
    try:
        client = get_client()
        
        # Verify collection exists before attempting deletion
        try:
            collection = client.get_collection(name=doc_name)
        except Exception:
            return False, f"Document '{doc_name}' not found in database."
        
        # Delete the collection
        client.delete_collection(name=doc_name)
        logger.info(f"Deleted ChromaDB collection: {doc_name}")
        
        return True, f"Document '{doc_name}' deleted successfully from database."
    
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return False, f"Error deleting document: {str(e)}"

def get_document_stats(doc_name: str) -> Optional[dict]:
    """Get statistics about a document."""
    try:
        client = get_client()
        collection = client.get_collection(name=doc_name)
        count = collection.count()
        
        # Get a sample to determine type
        sample = collection.get(limit=1, include=['metadatas'])
        doc_type = "unknown"
        if sample and sample['metadatas']:
            doc_type = sample['metadatas'][0].get('type', 'unknown')
        
        return {
            "name": doc_name,
            "chunk_count": count,
            "type": doc_type
        }
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        return None