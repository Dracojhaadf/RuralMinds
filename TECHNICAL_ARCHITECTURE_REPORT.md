# Technical Architecture Report: Hybrid RAG System

**Date**: May 21, 2026  
**Project**: Winner - AI Tutor Platform  
**System Type**: Retrieval Augmented Generation (RAG) with Multi-Language Support

---

## Executive Summary

This document provides a comprehensive technical analysis of the backend architecture, focusing on:
- PDF ingestion and storage pipeline
- Vector embeddings generation and indexing
- Semantic search mechanism
- Data retrieval and answer generation
- Multi-language support infrastructure
- Video processing capabilities

The system implements a **hybrid confidence-based RAG architecture** that intelligently routes queries between direct LLM generation and document-based RAG pipelines.

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                        │
│                    (Web Frontend / API Endpoints)                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  QUERY ROUTER   │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       ┌────▼────┐    ┌─────▼─────┐    ┌────▼─────┐
       │Language │    │Confidence │    │Document  │
       │Detection│    │ Checker   │    │ Detector │
       └────┬────┘    └─────┬─────┘    └────┬─────┘
            │                │                │
       ┌────▼────────────────▼────────────────▼─────┐
       │  HYBRID ROUTING DECISION ENGINE             │
       │  (Confidence-Based vs Document-Based RAG)  │
       └────┬─────────────────────────────────────┬─┘
            │                                     │
      ┌─────▼─────────┐              ┌───────────▼──────┐
      │  Direct LLM   │              │  Document RAG    │
      │  Generation   │              │  Pipeline        │
      └─────┬─────────┘              └───────────┬──────┘
            │                                     │
            │         ┌───────────────────┐      │
            │         │  Vector Database  │◄─────┘
            │         │  (ChromaDB)       │
            │         └─────────┬─────────┘
            │                   │
            │         ┌─────────▼─────────┐
            │         │  Ollama LLM       │
            │         │  (Inference)      │
            └────────►│  Model Selection: │
                      │  - phi3:mini      │
                      └───────────────────┘
```

---

## 2. Component Architecture

### 2.1 Input Processing Pipeline

#### 2.1.1 Supported Input Types

| Input Type | Processing | Storage | Use Case |
|-----------|-----------|---------|----------|
| **PDF Documents** | Text extraction via PyMuPDF (fitz) | ChromaDB + Filesystem | Knowledge base indexing |
| **Video Files** | Audio extraction + ASR transcription | Filesystem + ChromaDB | Lecture/content capture |
| **Audio Files** | Direct transcription (multi-language ASR) | Captions JSON + ChromaDB | Voice notes |
| **Text Queries** | Language detection + optional translation | In-memory | Real-time Q&A |

#### 2.1.2 PDF Ingestion Flow

```
┌─────────────────┐
│ PDF Upload      │
│ (via API)       │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│ Extract Text via PyMuPDF │
│ - Page-by-page parsing   │
│ - Handle multi-page docs │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Text Cleaning            │
│ - Whitespace normalize   │
│ - ASCII encoding         │
│ - Remove artifacts       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Sentence-Based Chunking  │
│ - NLTK sent_tokenize()   │
│ - Overlap: 2 sentences   │
│ - Max: 5 sentences/chunk │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Generate Embeddings      │
│ SentenceTransformer:     │
│ all-MiniLM-L6-v2        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Store in ChromaDB        │
│ + Filesystem (backup)    │
│ + SQLite metadata        │
└──────────────────────────┘
```

**Code Reference**: [backend.py](backend.py#L639-L680)

---

## 3. Data Storage Architecture

### 3.1 Storage Layer Components

#### 3.1.1 ChromaDB (Primary Vector Store)

**Location**: `chroma_db/` directory  
**Type**: Persistent vector database with SQLite backend  
**Database File**: `chroma_db/chroma.sqlite3`

**Purpose**: 
- Stores vectorized document chunks
- Maintains embedding-to-document mappings
- Enables semantic similarity search

**Storage Structure**:
```
chroma_db/
├── chroma.sqlite3           # SQLite DB engine
└── [collection_uuid]/       # Per-document collection
    ├── data/
    ├── index/
    └── metadata.parquet
```

**Collection Naming Convention**:
```python
def sanitize_collection_name(filename: str) -> str:
    # Example: "Quantum Physics.pdf" → "quantum_physics"
    # Rules:
    # - Alphanumeric + ._- only
    # - Remove leading numbers
    # - Max 63 characters
    # - Lowercase
```

#### 3.1.2 Filesystem Storage

**PDF Storage**: `uploaded_pdfs/` directory
- Permanent backup of original PDFs
- Referenced by ChromaDB metadata

**Video Storage**: `static/videos/` directory
- Video files with timestamp prefixes
- Naming: `{video_name}_{YYYYMMDD_HHMMSS}{ext}`

**Caption Storage**: `captions/` directory
- JSON files with transcript + timestamps
- Format: `{video_name}_captions.json`

**Example Caption Structure**:
```json
{
  "video_name": "Quantum_Mechanics_101",
  "created_at": "2026-05-21T14:30:00",
  "full_text": "Quantum mechanics is the study of...",
  "timestamps": [
    {"start": 0.0, "end": 5.2, "text": "Quantum mechanics is..."},
    {"start": 5.2, "end": 12.1, "text": "The fundamental principle..."}
  ],
  "word_count": 1247
}
```

#### 3.1.3 SQLite Metadata Database

**Purpose**: Track documents and videos across system lifecycle

**Schema**:
```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE,           -- Sanitized collection name
    upload_path TEXT,               -- Path to original PDF
    uploaded_by TEXT,               -- User identifier
    uploaded_at TIMESTAMP
);

CREATE TABLE videos (
    id INTEGER PRIMARY KEY,
    filename TEXT,                  -- File with timestamp
    name TEXT,                      -- Video name (searchable)
    video_path TEXT,                -- Path to video file
    caption_path TEXT,              -- Path to captions JSON
    uploaded_by TEXT,
    uploaded_at TIMESTAMP
);
```

---

## 4. Vectorization Architecture

### 4.1 Embedding Model

**Model**: `all-MiniLM-L6-v2` (SentenceTransformer)

**Specifications**:
- **Source**: HuggingFace sentence-transformers library
- **Architecture**: MiniLM (lightweight BERT variant)
- **Embedding Dimension**: 384 dimensions per vector
- **Max Sequence Length**: 512 tokens
- **Training Data**: Semantic Similarity (STS) tasks
- **Performance**: Optimized for CPU inference

**Why This Model?**
- Small model size (~80MB) → Fast inference
- High quality embeddings for semantic search
- Supports 100+ languages (including Malayalam, Hindi)
- Efficient for real-time applications

### 4.2 Chunking Strategy

#### 4.2.1 Sentence-Based Chunking Algorithm

```python
def sentence_based_chunking(
    text: str,
    max_sentences: int = 5,        # MAX_SENTENCES_PER_CHUNK
    overlap: int = 2                # SENTENCE_OVERLAP
) -> List[str]
```

**Algorithm Flow**:
1. **Tokenize** text into sentences using NLTK
   - Primary: `sent_tokenize()` from punkt tokenizer
   - Fallback: Split on periods if tokenizer fails

2. **Chunking with Overlap**:
   - Window size: 5 sentences
   - Step size: 5 - 2 = 3 sentences (overlap)
   - Preserves context across chunks

**Example**:
```
Text: "Sentence 1. Sentence 2. Sentence 3. Sentence 4. 
       Sentence 5. Sentence 6. Sentence 7."

Chunks Generated:
├─ Chunk 1: [S1, S2, S3, S4, S5]
├─ Chunk 2: [S4, S5, S6, S7]      ← 2 sentence overlap
└─ Chunk 3: [S7]                  ← Remainder
```

**Configuration**:
```python
MAX_SENTENCES_PER_CHUNK = 5
SENTENCE_OVERLAP = 2
MAX_CONTEXT_LENGTH = 3000  # Max chars to pass to LLM
```

### 4.3 Embedding Generation & Storage

```python
# Phase 1: Encode chunks to vectors
embed_model = get_embedding_model()
embeddings = embed_model.encode(
    chunks,
    show_progress_bar=False
)  # Returns: List[List[float]] shape (n_chunks, 384)

# Phase 2: Store in ChromaDB
collection = client.get_or_create_collection(name=doc_name)
collection.add(
    embeddings=embeddings.tolist(),  # Convert numpy → list
    documents=chunks,                 # Raw text
    ids=ids,                          # Unique identifiers
    metadatas=[{                      # Document metadata
        "type": "pdf",
        "chunk_id": i,
        "source": uploaded_file.name,
        "created_at": datetime.now().isoformat()
    } for i in range(len(chunks))]
)
```

---

## 5. Vector Search Architecture

### 5.1 Semantic Search Pipeline

#### 5.1.1 Query Embedding & Retrieval

```
┌──────────────────┐
│ User Query       │
│ (Natural Text)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│ Encode Query to Vector   │
│ Using all-MiniLM-L6-v2   │
│ Output: [384 floats]     │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Compute Cosine Similarity│
│ Query Vec vs Chunk Vecs  │
│                          │
│ Formula:                 │
│ sim = (A·B)/(|A||B|)    │
│ Range: [-1, 1]          │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Rank by Similarity       │
│ Top-K (default K=3)      │
│ Retrieve Metadata        │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Return Top Chunks +      │
│ Similarity Scores        │
└──────────────────────────┘
```

#### 5.1.2 ChromaDB Query API

```python
def query_saved_document(doc_name: str, query: str, k: int = 3):
    # Step 1: Get collection
    client = get_chroma_client()
    collection = client.get_collection(name=doc_name)
    
    # Step 2: Embed query
    embed_model = get_embedding_model()
    query_emb = embed_model.encode([query]).tolist()
    
    # Step 3: Retrieve similar chunks
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k                    # Top-3 by default
    )
    
    # Step 4: Extract results
    retrieved_chunks = results['documents'][0]
    # Returns: List of k most similar text chunks
```

**Retrieval Output Format**:
```python
{
    'documents': [
        ['Chunk text 1...', 'Chunk text 2...', 'Chunk text 3...']
    ],
    'ids': [['chunk_id_1', 'chunk_id_2', 'chunk_id_3']],
    'distances': [[0.15, 0.22, 0.31]],      # Euclidean distance
    'metadatas': [[...metadata...]]
}
```

### 5.2 Similarity Metrics

**Distance Metric**: L2 Euclidean Distance (default in ChromaDB)

**Conversion to Similarity**:
- Lower distance = Higher similarity
- Score range: 0 to ∞ (distance)
- Typical good matches: distance < 0.5

**Formula**:
```
Similarity Score = 1 / (1 + distance)
```

---

## 6. Query Processing & Retrieval Pipeline

### 6.1 Hybrid Confidence-Based Routing

```
┌─────────────────────────────────────────────────────┐
│  INCOMING QUERY                                     │
│  "What is quantum entanglement?"                    │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │ Language Detection      │
        │ Output: 'en', 'hi', 'ml'│
        └────────────┬────────────┘
                     │
        ┌────────────▼──────────────────┐
        │ Is Document-Specific Query?   │
        │ Keywords: pdf, document, file │
        │ Output: True/False            │
        └────────────┬──────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    ┌───▼─────┐          ┌───────▼────┐
    │Document │          │General     │
    │Query?   │          │Query?      │
    └───┬─────┘          └───────┬────┘
        │                        │
    YES │                        │ NO
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌──────────────────────┐
│ QUERY WITH      │    │ CONFIDENCE CHECK     │
│ CONFIDENCE      │    │ Query Ollama:        │
└────────┬────────┘    │ "Am I confident?"    │
         │             └──────────┬───────────┘
         │                        │
         │          ┌─────────────┴──────────────┐
         │          │                           │
         │     YES  │ Confident          NO     │ Not Confident
         │    ┌─────▼─────┐            ┌──────▼──┐
         │    │Return LLM  │            │Trigger  │
         │    │Direct Ans. │            │RAG      │
         │    └─────┬─────┘            └──────┬──┘
         │          │                         │
         └──────────┼─────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │ RAG Pipeline        │
         │ - Embed query       │
         │ - Search ChromaDB   │
         │ - Retrieve chunks   │
         │ - Pass to Ollama    │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────────┐
         │ Ollama Generation       │
         │ (Streaming Response)    │
         └─────────────────────────┘
```

### 6.2 Confidence Checking Algorithm

```python
def query_with_confidence(query: str, doc_name: str):
    prompt = """You are an AI tutor. Answer if you're confident.
    If NOT confident or question asks about a specific document,
    respond with exactly: [NEED_CONTEXT]
    
    Question: {query}
    Answer:"""
    
    response = query_ollama_simple(prompt)
    
    # Check for confidence markers
    if "[need_context]" in response.lower():
        return None, True   # (answer, needs_rag=True)
    
    return response.strip(), False  # (answer, needs_rag=False)
```

**Confidence Decision Logic**:
1. Query LLM with prompt asking for confidence
2. If response contains `[NEED_CONTEXT]` → Trigger RAG
3. If LLM unavailable → Trigger RAG (fail-safe)
4. Otherwise → Return LLM response directly

---

## 7. Data Retrieval & Answer Generation

### 7.1 Context-Based Generation Pipeline

```
┌──────────────────────────────────────┐
│ Retrieved Chunks (Top-K)             │
│ ├─ Chunk 1: "Quantum..."             │
│ ├─ Chunk 2: "Entanglement..."        │
│ └─ Chunk 3: "Superposition..."       │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Context Assembly                     │
│ - Concatenate chunks with separators │
│ - Enforce MAX_CONTEXT_LENGTH (3000)  │
│ - Preserve chunk boundaries          │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Construct System Prompt              │
│ "You are an AI Tutor"                │
│ Include rules/guidelines             │
│ Add multi-language instructions      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Build Full Prompt                    │
│ ┌─────────────────────────────────┐  │
│ │ SYSTEM_PROMPT                   │  │
│ │                                 │  │
│ │ Context:                        │  │
│ │ [assembled chunks]              │  │
│ │                                 │  │
│ │ Question:                       │  │
│ │ [user query]                    │  │
│ │                                 │  │
│ │ Answer:                         │  │
│ └─────────────────────────────────┘  │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Ollama Inference (Streaming)         │
│ Model: phi3:mini                     │
│ Temperature: 0.2 (low randomness)    │
│ Max Tokens: 150                      │
└────────────┬─────────────────────────┘
             │
             ▼
┌──────────────────────────────────────┐
│ Stream Response to Client            │
│ Yield chunks as they arrive          │
│ Include metadata (sources)           │
└──────────────────────────────────────┘
```

### 7.2 Ollama Integration

**Model Configuration**:
```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"  # Lightweight LLM
```

**Inference Parameters**:
```python
payload = {
    "model": "phi3:mini",
    "prompt": full_prompt,
    "stream": True,                    # Enable streaming
    "keep_alive": "10m",               # Keep model in memory
    "options": {
        "temperature": 0.2,            # Low randomness
        "num_predict": 150             # Max output tokens
    }
}
```

**Streaming Response Format**:
```json
// Each line is a JSON object
{"response": "Quantum", "done": false}
{"response": " entanglement", "done": false}
{"response": " is a phenomenon", "done": false}
{"response": "...", "done": true}
```

### 7.3 Answer Generation Functions

#### Non-Streaming (Blocking)
```python
def query_ollama(context: str, query: str) -> Optional[str]
# Returns complete answer when ready
# Use case: Simple API responses
```

#### Streaming (Real-time)
```python
def query_ollama_stream(context: str, query: str)
# Yields text chunks as they're generated
# Use case: Web frontend with live updates
```

---

## 8. Multi-Language Support Architecture

### 8.1 Language Detection Pipeline

```python
def detect_language(text: str) -> str:
    """
    Returns: 'en' | 'hi' | 'ml'
    
    Detection Method:
    1. Check for Romanized keywords (hi, ml specific)
    2. Use langdetect library (probability-based)
    3. Default to 'en' if unclear
    """
```

**Keyword Heuristics**:
```python
indic_keywords = {
    'hi': ['bhai', 'kya', 'hai', 'kaise', 'kyun', 'aur'],
    'ml': ['entha', 'engane', 'evide', 'cheyyane', 'und', 'illa']
}
```

### 8.2 Translation Pipeline

**Models Used**: 
- Helsinki-NLP MarianMT models
- Covers: en↔hi, en↔ml

**Translation Process**:
```python
def translate_to_english(text: str, source_lang: str) -> str:
    # Load model (cached)
    tokenizer, model = get_translation_model(source_lang, 'en')
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    
    # Generate translation
    translated = model.generate(**inputs)
    
    # Decode
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    return result
```

### 8.3 Multilingual ASR (Speech-to-Text)

| Language | Model | Provider |
|----------|-------|----------|
| **English** | Whisper (small) | OpenAI |
| **Hindi** | IndicWav2Vec | AI4Bharat |
| **Malayalam** | Fine-tuned Whisper | Local Custom Model |

**Malayalam Fine-Tuned Model Path**:
```
model/whisper-ml-model/content/whisper-ml-finetuned-final/
├── config.json
├── generation_config.json
├── model.safetensors
├── processor_config.json
├── tokenizer.json
├── tokenizer_config.json
└── (training_args.bin - optional)
```

**ASR Processing Flow**:
```
Audio File → Load at 16kHz → Extract Features → 
Generate Decoder IDs → Model.generate() → Decode → Text
```

---

## 9. Video Processing Architecture

### 9.1 Video Ingestion Pipeline

```
┌─────────────┐
│ Video File  │
│ (.mp4, .avi)│
└─────┬───────┘
      │
      ▼
┌─────────────────────────┐
│ Save to Filesystem      │
│ Path: static/videos/    │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Extract Audio (ffmpeg)  │
│ → 16kHz mono WAV        │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Transcribe Audio (ASR)  │
│ Language-specific model │
│ Output: Text + timestamps
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Save Captions JSON      │
│ Path: captions/         │
└─────┬───────────────────┘
      │
      ▼
┌─────────────────────────┐
│ Index Captions in RAG   │
│ - Clean text            │
│ - Chunk sentences       │
│ - Generate embeddings   │
│ - Store in ChromaDB     │
└─────────────────────────┘
```

### 9.2 FFmpeg Audio Extraction

```bash
ffmpeg -i input.mp4 \
  -vn                      # No video
  -acodec pcm_s16le       # PCM 16-bit encoding
  -ar 16000               # Resample to 16kHz
  -ac 1                   # Mono (1 channel)
  -y                      # Overwrite
  output_audio.wav
```

---

## 10. Configuration Parameters

### 10.1 Chunking Configuration

```python
MAX_SENTENCES_PER_CHUNK = 5        # Sentences per chunk
SENTENCE_OVERLAP = 2                # Overlapping sentences
MAX_CONTEXT_LENGTH = 3000          # Max chars to LLM
DEFAULT_K_RESULTS = 3              # Default retrieval count
```

### 10.2 Embedding Configuration

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384                # Vector dimension
MODEL_MAX_LENGTH = 512             # Max input tokens
```

### 10.3 Ollama Configuration

```python
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"
TEMPERATURE = 0.2                  # Low randomness
NUM_PREDICT = 150                  # Max generation tokens
KEEP_ALIVE = "10m"                 # Model cache duration
```

---

## 11. Data Flow Diagram: Complete Query Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│  USER SUBMITS QUERY: "Explain quantum entanglement from page 42"│
└──────────────────────────┬──────────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │ Language Detection  │
                │ Output: 'en'        │
                └──────────┬──────────┘
                           │
                ┌──────────▼──────────────────────┐
                │ Is Document-Specific Query?     │
                │ Keywords found: "page", "from"  │
                │ Output: True                    │
                └──────────┬──────────────────────┘
                           │
                ┌──────────▼──────────────────────┐
                │ Query with Confidence Check     │
                │ Prompt: "Are you confident?"    │
                │ Response: "[NEED_CONTEXT]"      │
                │ Output: needs_rag = True        │
                └──────────┬──────────────────────┘
                           │
                ┌──────────▼──────────────────────┐
                │ TRIGGER RAG PIPELINE            │
                └──────────┬──────────────────────┘
                           │
        ┌──────────────────┴──────────────────────┐
        │                                         │
        ▼                                         ▼
┌──────────────────┐               ┌─────────────────────┐
│ Encode Query:    │               │ Get ChromaDB Client │
│ "quantum         │               │ Access collection   │
│ entanglement..." │               │ for document        │
│                  │               └─────────┬───────────┘
│ Output: [384     │                         │
│ floats]          │               ┌─────────▼───────────┐
└────────┬─────────┘               │ Query embeddings:   │
         │                         │ top_k=3 results     │
         │                         └─────────┬───────────┘
         │                                   │
         └───────────────┬───────────────────┘
                         │
                ┌────────▼────────┐
                │ Retrieved Chunks:
                │ • "Quantum ent. │
                │   is a phenomenon
                │   where particles│
                │   are correlated"│
                │ • "EPR Paradox  │
                │   suggested action
                │   at a distance" │
                │ • "Einstein called
                │   it 'spooky'..."│
                └────────┬────────┘
                         │
                ┌────────▼──────────────┐
                │ Assemble Full Prompt: │
                │ • System instructions │
                │ • Context (chunks)    │
                │ • Question            │
                └────────┬──────────────┘
                         │
                ┌────────▼──────────────┐
                │ Send to Ollama        │
                │ Stream=True           │
                │ Temperature=0.2       │
                └────────┬──────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   ┌─────────┐                    ┌──────────────┐
   │Ollama   │                    │ Stream to    │
   │processes│                    │ Frontend     │
   │&yields: │                    │              │
   │• "Quantum"                   │ "Quantum..." │
   │• " entanglement"             │ " entangle.."│
   │• " describes..."             │ " describes."│
   │• "..."                       │ "..."        │
   │• "[DONE]"                    │              │
   └─────────┘                    └──────────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                ┌────────▼────────────────┐
                │ Final Response with:    │
                │ • Generated answer      │
                │ • Source chunks used    │
                │ • Metadata              │
                └─────────────────────────┘
```

---

## 12. Performance Characteristics

### 12.1 Latency Profile

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| **PDF Upload (10MB)** | 2-5 sec | Extract + chunk + embed |
| **Single Query Embedding** | 50-100 ms | all-MiniLM-L6-v2 |
| **ChromaDB Retrieval (k=3)** | 10-30 ms | Cached index |
| **Ollama Generation (150 tokens)** | 3-8 sec | phi3:mini on CPU |
| **Full Pipeline (E2E)** | 4-10 sec | E2E RAG response |

### 12.2 Storage Requirements

| Component | Size | Notes |
|-----------|------|-------|
| **Embedding Model** | ~80 MB | all-MiniLM-L6-v2 |
| **Ollama Model** | ~3.8 GB | phi3:mini quantized |
| **Per 100 chunks** | ~50 KB | ChromaDB embeddings |
| **SQLite DB** | Varies | Metadata only |

### 12.3 Throughput

- **Max concurrent queries**: Depends on Ollama workers (default: 1)
- **Embedding throughput**: ~500 sentences/sec (CPU)
- **Index search**: O(n) linear scan (n = total chunks)

---

## 13. Fault Tolerance & Error Handling

### 13.1 Error Recovery Mechanisms

```python
try:
    # RAG pipeline
except FileNotFoundError:
    # Missing model/document → Surface clear error
    
except requests.exceptions.ConnectionError:
    # Ollama unavailable → Fall back to direct extraction
    
except json.JSONDecodeError:
    # Malformed response → Skip chunk, continue
    
except Exception as e:
    # Catch-all → Log error, return fallback message
```

### 13.2 Fallback Strategies

1. **Ollama Unavailable**: Return raw extracted text chunks
2. **Embedding Model Missing**: Lazy-load with HTTP download
3. **ChromaDB Corrupted**: Rebuild from source files
4. **Query Timeout**: Return partial results or timeout message

---

## 14. Security Considerations

### 14.1 Input Validation

```python
# Sanitize collection names (prevent injection)
def sanitize_collection_name(filename: str) -> str:
    # Only allow: alphanumeric, ._-
    # Prevent directory traversal
    
# Limit file sizes
MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
```

### 14.2 Data Privacy

- PDFs stored locally (no cloud upload)
- Video files encrypted by filesystem
- SQLite DB accessible only via Python API
- Embeddings not shared externally

---

## 15. Scaling Considerations

### 15.1 Horizontal Scaling

**Current Bottlenecks**:
- Single Ollama instance (CPU-bound)
- ChromaDB single-instance
- Embedding model on CPU

**Improvement Path**:
1. Deploy Ollama with multiple workers
2. Migrate ChromaDB to distributed backend (Qdrant, Milvus)
3. Use GPU for embeddings (CUDA acceleration)
4. Implement query caching layer (Redis)

### 15.2 Vertical Scaling

- Increase CPU cores for faster embeddings
- Allocate GPU memory for LLM inference
- Increase RAM for larger ChromaDB indexes

---

## 16. Summary: Component Interaction Map

```
INPUT                 PROCESSING                 STORAGE                 OUTPUT
─────────────────────────────────────────────────────────────────────────────

PDF Files ──┐
            ├─► Extract Text ────┐
            │   (PyMuPDF)         │
Video Files ┤                     ├─► Clean & Chunk ─┐
            │                     │   (NLTK)          │
Audio Files ─┘                    └────┬──────────────┤
                                       │               │
                                       ▼               │
                                  Embed (384-D) ┐     │
                              (all-MiniLM-L6v2)│     │
                                       │        │     │
                   ┌───────────────────┘        │     │
                   │                            │     │
                   ▼                            ▼     ▼
              ChromaDB ◄──────────────────────────────┘
         (Vector Store)
              │
              │ Query + Embed
              │
              ▼
         Similarity Search
         (Top-K Results)
              │
              ├─► Confidence Check
              │   (Ollama LLM)
              │
              ├─ If Confident: Direct Answer
              │
              └─ If Not Confident:
                 Retrieve Chunks ──┐
                                   ├─► Ollama
                                   │   (phi3:mini)
                 System Prompt ────┘
                                   │
                                   ▼
                              Stream Response
                              + Metadata
                                   │
                                   ▼
                            Client (JSON/SSE)
```

---

## 17. Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Vectorization** | SentenceTransformer (all-MiniLM-L6-v2) | Generate 384-D embeddings |
| **Vector Store** | ChromaDB + SQLite | Persistent index with metadata |
| **LLM** | Ollama + phi3:mini | Inference engine |
| **ASR** | Whisper + AI4Bharat + Fine-tuned | Speech recognition (3 languages) |
| **PDF Processing** | PyMuPDF (fitz) | Text extraction |
| **NLP** | NLTK + Transformers | Tokenization + translation |
| **Filesystem** | Local disk storage | Backup + persistence |
| **Database** | SQLite | Metadata tracking |

---

## Appendix: Key Code References

| Function | Location | Purpose |
|----------|----------|---------|
| `process_and_save_pdf()` | backend.py L639-680 | PDF ingestion pipeline |
| `sentence_based_chunking()` | backend.py L593-611 | Text chunking algorithm |
| `query_saved_document_stream()` | backend.py L994-1061 | Main query entry point |
| `get_malayalam_whisper()` | backend.py L281-340 | Malayalam ASR model loading |
| `query_with_confidence()` | backend.py L205-230 | Confidence-based routing |
| `extract_pdf()` | backend.py L558-575 | PDF text extraction |
| `query_ollama_stream()` | backend.py L884-914 | Streaming LLM inference |

---

**End of Report**
