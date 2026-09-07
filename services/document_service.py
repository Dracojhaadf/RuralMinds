import os
import re
import io
import unicodedata
import fitz
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
from nltk.tokenize import sent_tokenize

from config.settings import (
    UPLOADED_PDFS_FOLDER,
    UNIFIED_COLLECTION_NAME,
    MAX_SENTENCES_PER_CHUNK,
    SENTENCE_OVERLAP,
    SOURCE_FOLDER
)
from core.database import get_db_connection
from services.vector_service import (
    get_embedding_model,
    get_chroma_client,
    get_paperqa_docs,
    sanitize_collection_name
)

logger = logging.getLogger(__name__)

os.makedirs(UPLOADED_PDFS_FOLDER, exist_ok=True)


def _ocr_page(page) -> str:
    """OCR a single PDF page using pytesseract as fallback for scanned documents."""
    try:
        import pytesseract
        from PIL import Image

        # Render page to high-res image for OCR
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        # OCR with English + Hindi + Malayalam (if tesseract lang packs installed)
        try:
            text = pytesseract.image_to_string(img, lang='eng+hin+mal')
        except pytesseract.TesseractError:
            # Fall back to English-only if language packs not installed
            text = pytesseract.image_to_string(img, lang='eng')

        return text.strip()
    except ImportError:
        logger.warning("pytesseract not installed — OCR fallback unavailable")
        return ""
    except Exception as e:
        logger.warning(f"OCR failed for page: {str(e)}")
        return ""


def extract_pdf(file_stream) -> str:
    """Extract text from PDF file stream. Falls back to OCR for scanned pages."""
    try:
        if hasattr(file_stream, 'read'):
            stream_bytes = file_stream.read()
            if hasattr(file_stream, 'seek'):
                file_stream.seek(0)
        else:
            stream_bytes = file_stream

        doc = fitz.open(stream=stream_bytes, filetype="pdf")
        text_parts = []
        ocr_pages = 0
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text.strip() and len(text.strip()) > 20:
                text_parts.append(text)
            else:
                logger.info(f"Page {page_num}: insufficient text, attempting OCR…")
                ocr_text = _ocr_page(page)
                if ocr_text:
                    text_parts.append(ocr_text)
                    ocr_pages += 1
        doc.close()
        if ocr_pages > 0:
            logger.info(f"OCR applied to {ocr_pages} page(s)")
        if not text_parts:
            logger.warning("No text extracted from PDF (including OCR)")
            return ""
        return " ".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting PDF: {str(e)}")
        raise


def clean_text(text: str) -> str:
    """Clean and normalize text while preserving Unicode (Malayalam, Hindi, etc.)."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = " ".join(text.split())
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
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


def semantic_chunking(
    text: str,
    target_chunk_size: int = 384,
    overlap_size: int = 64
) -> List[str]:
    """
    Split text into semantically cohesive chunks using sentence embeddings.
    """
    if not text or not text.strip():
        return []
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Error tokenizing text: {str(e)}")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    if not sentences:
        return []

    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    try:
        embed_model = get_embedding_model()
        sentences_prefixed = [f"passage: {s}" for s in sentences]
        embeddings = embed_model.encode(sentences_prefixed, show_progress_bar=False)

        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                sim = 0.0
            similarities.append(sim)

        if similarities:
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            threshold = mean_sim - 0.8 * std_sim
            threshold = np.clip(threshold, 0.70, 0.85)
        else:
            threshold = 0.80

        chunks = []
        current_chunk = []
        current_words = 0

        min_words = 120
        max_words = 350
        overlap_words = 50

        for i, sentence in enumerate(sentences):
            words_in_sent = len(sentence.split())
            current_chunk.append(sentence)
            current_words += words_in_sent

            should_split = False
            if i < len(sentences) - 1:
                sim = similarities[i]
                if (sim < threshold and current_words >= min_words) or (current_words >= max_words):
                    should_split = True
            else:
                should_split = False

            if should_split:
                chunks.append(" ".join(current_chunk))
                overlap_sentences = []
                overlap_count = 0
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_count + sent_words > overlap_words and len(overlap_sentences) >= 1:
                        break
                    overlap_sentences.insert(0, sent)
                    overlap_count += sent_words
                current_chunk = list(overlap_sentences)
                current_words = overlap_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    except Exception as e:
        logger.error(f"Semantic chunking failed: {str(e)}. Falling back to sentence chunking.")
        return sentence_based_chunking(text)


def process_and_save_pdf(uploaded_file, original_filename: Optional[str] = None) -> Tuple[bool, str]:
    """Process PDF with ChromaDB and save globally to SQLite & File System."""
    try:
        raw_text = extract_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = semantic_chunking(cleaned_text)

        if not chunks:
            return False, "Could not extract any text from the PDF."

        embed_model = get_embedding_model()
        chunks_with_prefix = [f"passage: {chunk}" for chunk in chunks]
        embeddings = embed_model.encode(chunks_with_prefix, show_progress_bar=False)

        fname = getattr(uploaded_file, 'name', original_filename or 'document.pdf')
        doc_name = sanitize_collection_name(fname)
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=UNIFIED_COLLECTION_NAME)

        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{
                "type": "pdf",
                "doc_name": doc_name,
                "chunk_id": i,
                "source": doc_name,
                "created_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
        )

        permanent_path = os.path.join(UPLOADED_PDFS_FOLDER, f"{doc_name}.pdf")
        if hasattr(uploaded_file, 'getvalue'):
            with open(permanent_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        elif hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
            with open(permanent_path, "wb") as f:
                f.write(uploaded_file.read())

        conn = None
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                INSERT OR IGNORE INTO documents (filename, upload_path, uploaded_by, uploaded_at)
                VALUES (?, ?, ?, ?)
            ''', (doc_name, permanent_path, "teacher", datetime.now().isoformat()))
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not log to documents SQLite table: {str(e)}")
        finally:
            if conn:
                conn.close()

        try:
            docs = get_paperqa_docs()
            docs.add(permanent_path, docname=doc_name)
            logger.info(f"Added {doc_name} to PaperQA")
        except Exception as e:
            logger.warning(f"Could not add to PaperQA: {str(e)}")

        return True, f"Successfully processed '{fname}' ({len(chunks)} chunks)."

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        return False, f"Error: {str(e)}"
