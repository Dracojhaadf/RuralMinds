"""
Centralized Configuration and Settings for RuralMinds.
"""
import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Folders
SOURCE_FOLDER = os.getenv("SOURCE_FOLDER", str(BASE_DIR / "source_folder"))
UPLOADED_PDFS_FOLDER = os.getenv("UPLOADED_PDFS_FOLDER", str(BASE_DIR / "uploaded_pdfs"))
CAPTIONS_FOLDER = os.getenv("CAPTIONS_FOLDER", str(BASE_DIR / "captions"))
STATIC_FOLDER = os.getenv("STATIC_FOLDER", str(BASE_DIR / "static"))
TEMPLATES_FOLDER = os.getenv("TEMPLATES_FOLDER", str(BASE_DIR / "templates"))

# Database Paths
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "ruralminds.db"))

# Embedding & Vector Database
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
UNIFIED_COLLECTION_NAME = "ruralminds_docs"
MAX_SENTENCES_PER_CHUNK = 5
SENTENCE_OVERLAP = 2
DEFAULT_K_RESULTS = 5
MAX_CONTEXT_LENGTH = 4000
RERANK_RELEVANCE_THRESHOLD = -2.0  # CrossEncoder threshold; higher = more similar

# Ollama LLM Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", f"{OLLAMA_BASE_URL}/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")

# Speech Recognition & Multilingual Models
MALAYALAM_MODEL_PATH = os.getenv(
    "MALAYALAM_MODEL_PATH",
    str(BASE_DIR / "model" / "whisper-ml-model" / "content" / "whisper-ml-finetuned-final")
)

# Administrative Security
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "administrator")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "CHANGE_ME_IN_ENV")
