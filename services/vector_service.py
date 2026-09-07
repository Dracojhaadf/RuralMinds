import os
import re
import logging
from functools import lru_cache
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from config.settings import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    UNIFIED_COLLECTION_NAME,
    SOURCE_FOLDER,
    UPLOADED_PDFS_FOLDER,
    DEFAULT_K_RESULTS
)
from core.database import get_db_connection

try:
    from paperqa import Docs
except ImportError:
    Docs = None

logger = logging.getLogger(__name__)

_chroma_client = None
_paperqa_docs = None

@lru_cache(maxsize=1)
def get_embedding_model():
    """Lazy load embedding model with caching."""
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)

def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        logger.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
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

def sanitize_collection_name(filename: str) -> str:
    """Sanitize filename to be a valid ChromaDB collection name."""
    clean = os.path.splitext(filename)[0]
    clean = re.sub(r'[^a-zA-Z0-9_-]', '_', clean)
    clean = clean[:63]
    if not clean or not clean[0].isalnum():
        clean = "doc_" + clean
    if len(clean) < 3:
        clean = clean + "_doc"
    return clean

def get_available_documents() -> List[str]:
    """Get list of all indexed documents in the unified Chroma collection."""
    try:
        client = get_chroma_client()
        try:
            coll = client.get_collection(name=UNIFIED_COLLECTION_NAME)
            results = coll.get(include=['metadatas'])
            docs = set()
            for meta in results['metadatas']:
                if meta and 'doc_name' in meta:
                    docs.add(meta['doc_name'])
            return sorted(list(docs))
        except Exception:
            return []
    except Exception as e:
        logger.error(f"Error getting available documents: {str(e)}")
        return []

def get_document_path(doc_name: str) -> Optional[str]:
    """Get the file path for a document from SQLite or local folders."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT upload_path FROM documents WHERE filename = ? OR filename = ?", 
              (doc_name, f"{doc_name}.pdf"))
    row = c.fetchone()
    conn.close()
    
    if row and os.path.exists(row['upload_path']):
        return row['upload_path']
        
    for base in [SOURCE_FOLDER, UPLOADED_PDFS_FOLDER]:
        for ext in ["", ".pdf", ".PDF"]:
            p = os.path.join(base, f"{doc_name}{ext}")
            if os.path.exists(p):
                return p
    return None

def delete_document(doc_name: str) -> Tuple[bool, str]:
    """Delete a document from ChromaDB, SQLite metadata, and filesystem."""
    try:
        client = get_chroma_client()
        try:
            coll = client.get_collection(name=UNIFIED_COLLECTION_NAME)
            coll.delete(where={"doc_name": doc_name})
            logger.info(f"Deleted chunks for '{doc_name}' from unified collection.")
        except Exception as e:
            logger.warning(f"Failed to delete '{doc_name}' from unified collection: {e}")

        # Delete from SQLite
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM documents WHERE filename = ? OR filename = ?", (doc_name, f"{doc_name}.pdf"))
        conn.commit()
        conn.close()

        # Delete physical file
        path = get_document_path(doc_name)
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"Deleted physical file: {path}")

        return True, f"Document '{doc_name}' deleted successfully."
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return False, f"Error deleting document: {str(e)}"

def get_document_stats(doc_name: str) -> Optional[Dict]:
    """Get statistics for a specific document."""
    try:
        client = get_chroma_client()
        coll = client.get_collection(name=UNIFIED_COLLECTION_NAME)
        results = coll.get(where={"doc_name": doc_name}, include=['metadatas'])
        count = len(results['ids']) if results and 'ids' in results else 0
        if count == 0:
            return None
        
        pages = set()
        for m in results['metadatas']:
            if m and 'page_number' in m:
                pages.add(m['page_number'])
                
        return {
            'chunk_count': count,
            'page_count': len(pages) if pages else 'N/A',
            'type': 'PDF'
        }
    except Exception as e:
        logger.error(f"Error getting document stats: {str(e)}")
        return None

def rebuild_database() -> Tuple[bool, str]:
    """Rebuild database from source folder."""
    try:
        from services.document_service import process_and_save_pdf
        client = get_chroma_client()
        try:
            client.delete_collection(name=UNIFIED_COLLECTION_NAME)
            logger.info(f"Deleted unified collection: {UNIFIED_COLLECTION_NAME}")
        except Exception as e:
            logger.warning(f"Could not delete collection during rebuild: {e}")

        rebuilt_count = 0
        for folder in [SOURCE_FOLDER, UPLOADED_PDFS_FOLDER]:
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    if fname.lower().endswith(".pdf"):
                        fpath = os.path.join(folder, fname)
                        with open(fpath, "rb") as f:
                            success, _ = process_and_save_pdf(f, original_filename=fname)
                            if success:
                                rebuilt_count += 1

        return True, f"Rebuilt database with {rebuilt_count} documents."
    except Exception as e:
        logger.error(f"Error rebuilding database: {str(e)}")
        return False, f"Failed to rebuild database: {str(e)}"
