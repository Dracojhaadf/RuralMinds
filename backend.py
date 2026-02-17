import os
import fitz
import json
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging
# Import SOURCE_FOLDER from app config or define it here if needed. 
# It seems SOURCE_FOLDER is defined in app.py but used in backend.Let's define a default.
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", "source_folder")
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
import requests


try:
    from paperqa import Docs
except ImportError:
    Docs = None

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_SENTENCES_PER_CHUNK = 5
SENTENCE_OVERLAP = 2
DEFAULT_K_RESULTS = 3
MAX_CONTEXT_LENGTH = 3000

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

# Ensure NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# --- GLOBAL MODELS ---
from functools import lru_cache

_chroma_client = None
_chroma_client = None
_paperqa_docs = None
_whisper_model = None
import whisper

@lru_cache(maxsize=1)
def get_embedding_model():
    """Lazy load embedding model with caching."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB at: {DB_PATH}")
        _chroma_client = chromadb.PersistentClient(path=DB_PATH)
    return _chroma_client

def get_paperqa_docs():
    """Get or create PaperQA Docs instance."""
    global _paperqa_docs
    if _paperqa_docs is None:
        if Docs is None:
            raise ImportError("PaperQA not installed. Install with: pip install paperqa")
        logger.info("Initializing PaperQA")
        _paperqa_docs = Docs()
    return _paperqa_docs

def query_ollama_simple(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    """
    Simple Ollama query without context.
    Used for confidence checking.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            logger.warning(f"Ollama simple query failed: {response.status_code}")
            return None
    
    except requests.exceptions.ConnectionError:
        logger.warning("Could not connect to Ollama for confidence check")
        return None
    except Exception as e:
        logger.error(f"Error in simple Ollama query: {str(e)}")
        return None

def query_ollama_stream_simple(prompt: str, max_tokens: int = 1000):
    """
    Stream simple Ollama query without context.
    Yields chunks of text.
    """
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.3,
                "num_predict": max_tokens
            }
        }
        
        with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=60) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            # Parse JSON chunk
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']
                        except json.JSONDecodeError:
                            continue
            else:
                logger.warning(f"Ollama simple streaming failed: {response.status_code}")
                yield ""
    
    except Exception as e:
        logger.error(f"Error in simple Ollama streaming: {str(e)}")
        yield ""

def query_with_confidence(query: str, doc_name: str) -> Tuple[Optional[str], bool]:
    """
    Query LLM and check if it's confident about the answer.
    
    Returns:
        (answer, needs_rag): 
            - If confident: (answer_text, False)
            - If not confident: (None, True)
    """
    try:
        prompt = f"""You are an AI tutor. Answer the question if you're confident in your knowledge.

If you're NOT confident, or if the question is asking about a specific document/PDF/file, respond with exactly: [NEED_CONTEXT]

Question: {query}

Answer:"""
        
        logger.info("🤔 Checking LLM confidence...")
        response = query_ollama_simple(prompt, max_tokens=1000)
        
        if not response:
            logger.info("⚡ LLM unavailable - triggering RAG")
            return None, True
        
        # Check if LLM signals it needs context
        response_lower = response.lower()
        if "[need_context]" in response_lower or "need context" in response_lower or "[need context]" in response:
            logger.info("⚡ LLM not confident - triggering RAG")
            return None, True
        
        # LLM is confident
        logger.info("✓ LLM confident - returning direct answer")
        return response.strip(), False
    
    except Exception as e:
        logger.error(f"Error in confidence check: {str(e)}")
        return None, True  # On error, use RAG to be safe


# --- VOICE & TRANSLATION LAYER ---

def get_whisper_model():
    """Lazy load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio using Whisper.
    Automatically translates to English if source is not English.
    """
    try:
        model = get_whisper_model()
        
        # Transcribe (and translate to English)
        result = model.transcribe(audio_path, task="translate")
        text = result.get("text", "").strip()
        
        logger.info(f"🎤 Transcription: {text}")
        return text
    
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        return ""

def normalize_query(query: str) -> str:
    """
    Normalize query to English using LLM if needed.
    Handles Romanized Hindi/Malayalam or mixed text.
    """
    # Simple heuristic: If it looks like English/Code, return as is.
    # Otherwise, ask LLM to translate/normalize.
    
    try:
        # Prompt to normalize
        prompt = f"""Task: Translate/Normalize the following text to standard English.
If it is already English, output it exactly as is.
If it is Romanized Hindi/Malayalam, translate it to English.

Text: "{query}"

English Translation:"""
        
        normalized = query_ollama_simple(prompt, max_tokens=200)
        
        if normalized:
            clean = normalized.strip().strip('"')
            logger.info(f"🔄 Normalized: '{query}' -> '{clean}'")
            return clean
        
        return query
    except Exception as e:
        logger.warning(f"Normalization failed: {str(e)}")
        return query



# --- PDF PROCESSING ---
def sanitize_collection_name(filename: str) -> str:
    """Sanitize filename for ChromaDB collection name."""
    name = Path(filename).stem
    name = "".join(c if c.isalnum() or c in '._-' else '_' for c in name)
    name = name.lstrip("0123456789_-")
    if len(name) < 3:
        name = f"doc_{name}"
    return name[:63].lower()

def extract_pdf(file_stream) -> str:
    """Extract text from PDF file stream."""
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
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove non-ASCII (but keep common symbols)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
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
        logger.error(f"Error tokenizing: {str(e)}")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    overlap = min(overlap, max_sentences - 1)
    
    for i in range(0, len(sentences), max_sentences - overlap):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

# --- HYBRID RAG SYSTEM ---
def process_and_save_pdf(uploaded_file) -> Tuple[bool, str]:
    """Process PDF with both ChromaDB and PaperQA."""
    try:
        raw_text = extract_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = sentence_based_chunking(cleaned_text)
        
        if not chunks:
            return False, "Could not extract any text from the PDF."
        
        # Store in ChromaDB for retrieval
        embed_model = get_embedding_model()
        embeddings = embed_model.encode(chunks, show_progress_bar=False)
        
        doc_name = sanitize_collection_name(uploaded_file.name)
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=doc_name)
        
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{
                "type": "pdf",
                "chunk_id": i,
                "source": uploaded_file.name,
                "created_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
        )
        
        # Also add to PaperQA if available
        try:
            docs = get_paperqa_docs()
            # Save temp file for PaperQA to process
            temp_path = f"temp_{doc_name}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            docs.add(temp_path, docname=doc_name)
            os.remove(temp_path)
            logger.info(f"Added {doc_name} to PaperQA")
        except Exception as e:
            logger.warning(f"Could not add to PaperQA: {str(e)}")
        
        return True, f"Successfully processed '{uploaded_file.name}' ({len(chunks)} chunks)."
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return False, f"Error: {str(e)}"

# --- IMPROVED QUERY SYSTEM ---

def preload_ollama_model():
    """
    Forces Ollama to load the model into RAM with a warm-up query.
    This ensures the model is fully loaded and ready before the first user query.
    """
    try:
        logger.info(f"Warming up Ollama model: {OLLAMA_MODEL}...")
        
        # Send a warm-up query to force full model loading
        warmup_payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Hello",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_predict": 5  # Very short response for warm-up
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=warmup_payload, timeout=30)
        
        if response.status_code == 200:
            logger.info(f"✓ Ollama model {OLLAMA_MODEL} loaded and ready in RAM")
            return True
        else:
            logger.warning(f"Ollama warm-up returned status: {response.status_code}")
            return False
            
            return False
            
    except Exception as e:
        logger.warning(f"Could not preload Ollama model: {str(e)}")
        logger.warning("Model will load on first query instead")
        return False

def query_ollama_stream(context: str, query: str):
    """
    Stream query local Ollama instance with phi3:mini model.
    Yields chunks of text.
    """
    try:
        system_prompt = """You are a helpful AI tutor.
Answer clearly using only the provided context.
Use short structured explanations."""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
            "stream": True,  # Enable streaming
            "keep_alive": "10m",
            "options": {
                "temperature": 0.2,
                "num_predict": 150
            }
        }
        
        with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            # Parse JSON chunk
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']
                        except json.JSONDecodeError:
                            continue
            else:
                logger.warning(f"Ollama streaming returned status code: {response.status_code}")
                yield "Error: Could not connect to AI model."
            
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {str(e)}")
        yield f"Error: {str(e)}"


def query_ollama(context: str, query: str) -> Optional[str]:
    """
    Query local Ollama instance with phi3:mini model.
    Returns None if Ollama is not reachable.
    """
    try:
        
        system_prompt = """You are a helpful AI tutor.
Answer clearly using only the provided context.
Use short structured explanations."""

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.2,
                "num_predict": 150
            }
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            logger.warning(f"Ollama returned status code: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {str(e)}")
        return None

def generate_answer_from_context(
    retrieved_chunks: List[str],
    query: str,
    max_length: int = MAX_CONTEXT_LENGTH
) -> str:
    """
    Generate an answer based on retrieved chunks using Ollama (with fallback).
    """
    if not retrieved_chunks:
        return "No relevant information found in the document."
    
    # Combine chunks with length limit
    context = ""
    for chunk in retrieved_chunks:
        if len(context) + len(chunk) < max_length:
            context += chunk + "\n\n"
        else:
            break
    
    context = context.strip()
    
    if not context:
        return "Retrieved context too long to process."
    
    # Try Ollama first
    try:
        ollama_response = query_ollama(context, query)
        if ollama_response:
            return ollama_response
    except Exception as e:
        logger.warning(f"Ollama generation failed: {str(e)}")
    
    # Fallback to extractive QA if Ollama fails
    logger.info("Falling back to static template response")
    return f"""Based on the document, here is the relevant information:

QUESTION: {query}

RELEVANT EXCERPTS:
{context}

(Note: AI generation unavailable - showing direct extracts)"""

def generate_answer_from_context_stream(
    retrieved_chunks: List[str],
    query: str,
    max_length: int = MAX_CONTEXT_LENGTH
):
    """
    Generate an answer based on retrieved chunks using Ollama (streaming).
    Yields chunks.
    """
    if not retrieved_chunks:
        yield "No relevant information found in the document."
        return
    
    # Combine chunks with length limit
    context = ""
    for chunk in retrieved_chunks:
        if len(context) + len(chunk) < max_length:
            context += chunk + "\n\n"
        else:
            break
    
    context = context.strip()
    
    if not context:
        yield "Retrieved context too long to process."
        return
    
    # Try Ollama first
    try:
        # Check if Ollama is reachable first (optional optimization, but we'll just try stream)
        # We can reuse the stream function
        for chunk in query_ollama_stream(context, query):
            yield chunk
        return
            
    except Exception as e:
        logger.warning(f"Ollama generation failed: {str(e)}")
    
    # Fallback to extractive QA if Ollama fails
    logger.info("Falling back to static template response")
    fallback = f"""Based on the document, here is the relevant information:

QUESTION: {query}

RELEVANT EXCERPTS:
{context}

(Note: AI generation unavailable - showing direct extracts)"""
    
    yield fallback


def query_saved_document_hybrid(
    doc_name: str,
    query: str,
    k: int = DEFAULT_K_RESULTS
) -> Tuple[str, List[str]]:
    """
    Query documents using confidence-based hybrid approach:
    1. Try LLM with confidence check first
    2. If LLM is confident, return direct answer (fast)
    3. If LLM not confident, run full RAG pipeline
    """
    retrieved_chunks = []
    answer = ""
    
    # STEP 1: Try LLM with confidence check first
    try:
        llm_answer, needs_rag = query_with_confidence(query, doc_name)
        
        if not needs_rag and llm_answer:
            # LLM is confident - return fast answer
            return llm_answer, []
    except Exception as e:
        logger.warning(f"Confidence check failed: {str(e)}, proceeding to RAG")
    
    # STEP 2: LLM not confident - run full RAG pipeline
    logger.info("📚 Running full RAG pipeline...")
    
    # Skip PaperQA for speed (can re-enable if needed)
    # if Docs is not None:
    #     try:
    #         docs = get_paperqa_docs()
    #         result = docs.query(query, max_sources=k)
    #         
    #         if result and result.answer:
    #             answer = result.answer
    #             if hasattr(result, 'references') and result.references:
    #                 answer += "\n\n**Sources:**\n"
    #                 for ref in result.references[:3]:
    #                     answer += f"- {ref}\n"
    #             
    #             return answer, []
    #     except Exception as e:
    #         logger.warning(f"PaperQA query failed: {str(e)}")
    
    # Fall back to ChromaDB with improved approach
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=doc_name)
    except ValueError:
        return f"Error: Document '{doc_name}' not found.", []
    except Exception as e:
        return f"Error accessing database: {str(e)}", []
    
    try:
        embed_model = get_embedding_model()
        query_emb = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=k)
        retrieved_chunks = results['documents'][0] if results['documents'] else []
        
        if not retrieved_chunks:
            return "No relevant information found for this query.", []
        
        # Use context-based extraction instead of LLM
        answer = generate_answer_from_context(retrieved_chunks, query)
        
        return answer, retrieved_chunks
    
    except Exception as e:
        logger.error(f"Error querying: {str(e)}")
        return f"Error generating answer: {str(e)}", retrieved_chunks if retrieved_chunks else []

def query_saved_document_stream(
    doc_name: str,
    query: str,
    k: int = DEFAULT_K_RESULTS
):
    """
    Stream query documents using confidence-based hybrid approach.
    Yields chunks of text. 
    Final yield is a special dictionary with sources: {'sources': [...]}.
    """
    
    # STEP 1: Try LLM with confidence check (Streaming)
    is_confident = True
    try:
        prompt = f"""You are an AI tutor. Answer the question if you're confident in your knowledge.

If you're NOT confident, or if the question is asking about a specific document/PDF/file, respond with exactly: [NEED_CONTEXT]

Question: {query}

Answer:"""
        
        # Buffer to check for [NEED_CONTEXT]
        buffer = ""
        committed = False
        
        # We need a generator to handle the stream so we can peek at it
        stream_gen = query_ollama_stream_simple(prompt, max_tokens=1000)
        
        for chunk in stream_gen:
            if not committed:
                buffer += chunk
                
                # Check for refusal signal in buffer (case insensitive)
                buffer_lower = buffer.lower()
                if "[need" in buffer_lower and ("context]" in buffer_lower or " context]" in buffer_lower):
                    is_confident = False
                    break
                
                # If buffer is getting long and no refusal, it's probably a real answer
                # [NEED_CONTEXT] is ~14 chars. 
                if len(buffer) > 40:
                    yield buffer
                    buffer = ""
                    committed = True
            else:
                # Already committed to answering, just yield
                yield chunk
        
        # Handle remaining buffer if we finished stream without comitting or breaking
        if is_confident and not committed and buffer:
            if "[need" in buffer.lower() and "context]" in buffer.lower():
                is_confident = False
            else:
                yield buffer

        if is_confident:
            # Yield empty sources and return
            yield {'sources': []}
            return

    except Exception as e:
        logger.warning(f"Confidence check failed: {str(e)}, proceeding to RAG")
    
    # STEP 2: RAG Pipeline
    # If we are here, either !is_confident, or exception occurred
    logger.info("📚 Running full RAG pipeline (Streaming)...")
    
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=doc_name)
        embed_model = get_embedding_model()
        query_emb = embed_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_emb, n_results=k)
        retrieved_chunks = results['documents'][0] if results['documents'] else []
        
        if not retrieved_chunks:
            yield "No relevant information found for this query."
            yield {'sources': []}
            return
        
        # Stream the answer generation
        for chunk in generate_answer_from_context_stream(retrieved_chunks, query):
            yield chunk
            
        # Yield sources at the end
        yield {'sources': retrieved_chunks}
        
    except Exception as e:
        yield f"Error during streaming RAG: {str(e)}"
        yield {'sources': []}


# --- COMPATIBILITY FUNCTIONS ---
def query_saved_document(doc_name: str, query: str, k: int = DEFAULT_K_RESULTS) -> Tuple[str, List[str]]:
    """Backward compatible wrapper."""
    return query_saved_document_hybrid(doc_name, query, k)
    


def get_available_documents() -> List[str]:
    """Get list of all documents."""
    client = get_chroma_client()
    collections = client.list_collections()
    return sorted([col.name for col in collections])

def delete_document(doc_name: str) -> Tuple[bool, str]:
    """Delete a document."""
    try:
        client = get_chroma_client()
        
        try:
            client.get_collection(name=doc_name)
        except Exception:
            return False, f"Document '{doc_name}' not found."
        
        client.delete_collection(name=doc_name)
        logger.info(f"Deleted collection: {doc_name}")
        
        return True, f"Document '{doc_name}' deleted successfully."
    
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return False, f"Error deleting document: {str(e)}"

def rebuild_database() -> Tuple[bool, str]:
    """Rebuild the entire ChromaDB from source files."""
    try:
        logger.info("♻️ Starting database rebuild...")
        
        # 1. Process PDFs
        pdf_count = 0
        if os.path.exists(SOURCE_FOLDER):
            for filename in os.listdir(SOURCE_FOLDER):
                if filename.lower().endswith('.pdf'):
                    file_path = os.path.join(SOURCE_FOLDER, filename)
                    with open(file_path, 'rb') as f:
                        # Create a mock file object with name attribute
                        from io import BytesIO
                        class NamedBytesIO(BytesIO):
                            def __init__(self, content, name):
                                super().__init__(content)
                                self.name = name
                        
                        file_obj = NamedBytesIO(f.read(), filename)
                        success, msg = process_and_save_pdf(file_obj)
                        if success:
                            pdf_count += 1
                            logger.info(f"Re-indexed: {filename}")
                        else:
                            logger.error(f"Failed to re-index {filename}: {msg}")
        
        # 2. Process Videos (Captions)
        video_count = 0
        if os.path.exists(VIDEO_STORAGE_PATH):
            for filename in os.listdir(VIDEO_STORAGE_PATH):
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                    # Check if caption exists
                    video_name = Path(filename).stem
                    caption_data = load_caption_file(video_name)
                    
                    if caption_data:
                        # Re-index caption
                        success, msg = process_video_captions(video_name, caption_data['full_text'])
                        if success:
                            video_count += 1
                            logger.info(f"Re-indexed captions for: {video_name}")
                    else:
                        # Auto-transcribe if missing? Maybe too heavy.
                        # For now, just skip if no caption file.
                        pass

        logger.info(f"✅ Rebuild complete. PDFs: {pdf_count}, Videos: {video_count}")
        return True, f"Rebuild complete. Indexed {pdf_count} documents and {video_count} video captions."
        
    except Exception as e:
        logger.error(f"Database rebuild failed: {str(e)}")
        return False, f"Rebuild failed: {str(e)}"


def get_document_stats(doc_name: str) -> Optional[Dict]:
    """Get document statistics."""
    try:
        client = get_chroma_client()
        collection = client.get_collection(name=doc_name)
        count = collection.count()
        
        sample = collection.get(limit=1, include=['metadatas'])
        doc_type = "pdf"
        if sample and sample['metadatas']:
            doc_type = sample['metadatas'][0].get('type', 'pdf')
        
        return {
            "name": doc_name,
            "chunk_count": count,
            "type": doc_type
        }
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return None

# --- VIDEO FUNCTIONS (Preserved from original) ---
VIDEO_STORAGE_PATH = "static/videos"
CAPTIONS_STORAGE_PATH = "captions"

os.makedirs(VIDEO_STORAGE_PATH, exist_ok=True)
os.makedirs(CAPTIONS_STORAGE_PATH, exist_ok=True)

def save_video(uploaded_file) -> Tuple[bool, str, Optional[str]]:
    """Save an uploaded video file."""
    try:
        original_name = uploaded_file.name
        safe_name = sanitize_collection_name(original_name)
        extension = Path(original_name).suffix.lower()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}{extension}"
        video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
        
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        logger.info(f"Video saved: {video_path}")
        return True, f"Video '{original_name}' saved successfully.", video_path
    
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return False, f"Error saving video: {str(e)}", None

def save_caption_file(video_name: str, caption_text: str, timestamps: Optional[List[dict]] = None) -> Tuple[bool, str]:
    """Save caption text for a video as JSON."""
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
    """Process and index video captions in ChromaDB."""
    try:
        cleaned_text = clean_text(caption_text)
        chunks = sentence_based_chunking(cleaned_text)
        
        if not chunks:
            return False, "Could not create chunks from caption text."
        
        embed_model = get_embedding_model()
        embeddings = embed_model.encode(chunks, show_progress_bar=False)
        
        doc_name = sanitize_collection_name(video_name)
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=doc_name)
        
        ids = [f"{doc_name}_caption_chunk_{i}" for i in range(len(chunks))]
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{
                "type": "video_caption",
                "chunk_id": i,
                "source": video_name,
                "created_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
        )
        
        return True, f"Captions for '{video_name}' indexed successfully ({len(chunks)} chunks)."
    
    except Exception as e:
        logger.error(f"Error processing video captions: {str(e)}")
        return False, f"Error: {str(e)}"

def get_available_videos() -> List[dict]:
    """Get list of available videos."""
    videos = []
    
    if not os.path.exists(VIDEO_STORAGE_PATH):
        return videos
    
    for filename in os.listdir(VIDEO_STORAGE_PATH):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
            video_name = Path(filename).stem
            
            caption_data = load_caption_file(video_name)
            has_captions = caption_data is not None
            
            videos.append({
                "filename": filename,
                "name": video_name,
                "path": video_path,
                "has_captions": has_captions,
                "caption_data": caption_data,
                "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2)
            })
    
    return sorted(videos, key=lambda x: x['filename'])

def delete_video(video_name: str) -> Tuple[bool, str]:
    """Delete a video and its associated files."""
    try:
        deleted_items = []
        
        if os.path.exists(VIDEO_STORAGE_PATH):
            for filename in os.listdir(VIDEO_STORAGE_PATH):
                if Path(filename).stem == video_name:
                    video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
                    os.remove(video_path)
                    deleted_items.append(f"video file: {filename}")
                    logger.info(f"Deleted video file: {video_path}")
        
        safe_name = sanitize_collection_name(video_name)
        caption_filename = f"{safe_name}_captions.json"
        caption_path = os.path.join(CAPTIONS_STORAGE_PATH, caption_filename)
        
        if os.path.exists(caption_path):
            os.remove(caption_path)
            deleted_items.append("caption file")
            logger.info(f"Deleted caption file: {caption_path}")
        
        try:
            client = get_chroma_client()
            collection_name = sanitize_collection_name(video_name)
            client.delete_collection(name=collection_name)
            deleted_items.append("database collection")
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.info(f"No ChromaDB collection found for {video_name}: {str(e)}")
        
        if deleted_items:
            items_str = ", ".join(deleted_items)
            return True, f"Successfully deleted {items_str} for '{video_name}'."
        else:
            return False, f"No files found for video '{video_name}'."
    
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        return False, f"Error deleting video: {str(e)}"

# --- AUDIO & VIDEO CAPTION GENERATION ---
def extract_audio_from_video(video_path: str) -> Tuple[bool, str, Optional[str]]:
    """Extract audio from video using ffmpeg."""
    try:
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            audio_path
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            return False, f"FFmpeg error: {result.stderr}", None
        
        if not os.path.exists(audio_path):
            return False, "Audio file was not created", None
        
        return True, "Audio extracted successfully", audio_path
    
    except FileNotFoundError:
        return False, "FFmpeg not found. Install from: https://ffmpeg.org/download.html", None
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        return False, f"Error: {str(e)}", None

def get_whisper_model():
    """Lazy load Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            logger.info("Loading Whisper model (base)...")
            _whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded")
        except ImportError:
            logger.error("Whisper not installed")
            raise ImportError("Install with: pip install openai-whisper")
    return _whisper_model

_whisper_model = None

def transcribe_audio(audio_path: str, language: str = None) -> Tuple[bool, str, Optional[dict]]:
    """Transcribe audio using Whisper."""
    try:
        model = get_whisper_model()
        logger.info(f"Transcribing: {audio_path}")
        
        if language:
            result = model.transcribe(audio_path, language=language)
        else:
            result = model.transcribe(audio_path)
        
        transcription_data = {
            'text': result['text'],
            'segments': result.get('segments', []),
            'language': result.get('language', 'unknown')
        }
        
        try:
            os.remove(audio_path)
        except:
            pass
        
        return True, "Transcription completed", transcription_data
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return False, f"Error: {str(e)}", None

def generate_captions_from_video(video_path: str, video_name: str, language: str = None) -> Tuple[bool, str, Optional[str]]:
    """Generate captions from video using Whisper."""
    try:
        logger.info(f"Extracting audio from: {video_name}")
        success, message, audio_path = extract_audio_from_video(video_path)
        if not success:
            return False, message, None
        
        logger.info(f"Transcribing: {video_name}")
        success, message, transcription_data = transcribe_audio(audio_path, language)
        if not success:
            return False, message, None
        
        caption_text = transcription_data['text']
        
        timestamps = []
        for segment in transcription_data.get('segments', []):
            timestamps.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', '')
            })
        
        save_caption_file(video_name, caption_text, timestamps)
        
        detected_lang = transcription_data.get('language', 'unknown')
        word_count = len(caption_text.split())
        
        return True, f"Captions generated! ({word_count} words, language: {detected_lang})", caption_text
    
    except Exception as e:
        logger.error(f"Error generating captions: {str(e)}")
        return False, f"Error: {str(e)}", None

def translate_text(text: str, target_language: str = "es") -> Tuple[bool, str, Optional[str]]:
    """Translate text to target language."""
    try:
        if target_language == "en":
            return True, "No translation needed", text
        
        from transformers import pipeline
        
        lang_models = {
            "es": "Helsinki-NLP/opus-mt-en-es",
            "fr": "Helsinki-NLP/opus-mt-en-fr",
            "de": "Helsinki-NLP/opus-mt-en-de",
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "zh": "Helsinki-NLP/opus-mt-en-zh",
            "ar": "Helsinki-NLP/opus-mt-en-ar",
            "pt": "Helsinki-NLP/opus-mt-en-pt",
            "ru": "Helsinki-NLP/opus-mt-en-ru",
            "ja": "Helsinki-NLP/opus-mt-en-jap",
        }
        
        if target_language not in lang_models:
            return False, f"Language '{target_language}' not supported", None
        
        model_name = lang_models[target_language]
        logger.info(f"Loading translation model: {model_name}")
        translation_pipe = pipeline("translation", model=model_name)
        
        max_length = 500
        sentences = sent_tokenize(text)
        
        translated_chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_length and current_chunk:
                chunk_text = " ".join(current_chunk)
                result = translation_pipe(chunk_text, max_length=512)
                translated_chunks.append(result[0]['translation_text'])
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            result = translation_pipe(chunk_text, max_length=512)
            translated_chunks.append(result[0]['translation_text'])
        
        translated_text = " ".join(translated_chunks)
        return True, f"Translation to '{target_language}' completed", translated_text
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return False, f"Error: {str(e)}", None