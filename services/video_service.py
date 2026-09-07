import os
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List

from config.settings import (
    CAPTIONS_FOLDER,
    SOURCE_FOLDER,
    UNIFIED_COLLECTION_NAME
)
from core.database import get_db_connection
from services.vector_service import (
    get_chroma_client,
    get_embedding_model,
    sanitize_collection_name
)
from services.document_service import clean_text, semantic_chunking
from services.audio_service import transcribe_audio

logger = logging.getLogger(__name__)

VIDEO_STORAGE_PATH = SOURCE_FOLDER
CAPTIONS_STORAGE_PATH = CAPTIONS_FOLDER

os.makedirs(CAPTIONS_STORAGE_PATH, exist_ok=True)
os.makedirs(VIDEO_STORAGE_PATH, exist_ok=True)


def save_video(uploaded_file) -> Tuple[bool, str, Optional[str]]:
    """Save an uploaded video file and log to SQLite DBMS."""
    try:
        original_name = uploaded_file.name
        safe_name = sanitize_collection_name(original_name)
        extension = Path(original_name).suffix.lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}{extension}"
        video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
        
        uploaded_file.seek(0)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
            
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                INSERT INTO videos (filename, name, video_path, uploaded_by, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, Path(original_name).stem, video_path, "teacher", datetime.now().isoformat()))
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.warning(f"Failed to log video to SQLite: {db_err}")
            
        logger.info(f"Video saved: {video_path}")
        return True, f"Video '{original_name}' saved successfully.", video_path
    except Exception as e:
        logger.error(f"Error saving video: {str(e)}")
        return False, f"Error saving video: {str(e)}", None


def save_caption_file(video_name: str, caption_text: str, timestamps: Optional[List[dict]] = None) -> Tuple[bool, str]:
    """Save caption text for a video as JSON and update SQLite."""
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
            
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('''
                UPDATE videos SET caption_path = ? 
                WHERE name = ?
            ''', (caption_path, video_name))
            conn.commit()
            conn.close()
        except Exception as db_err:
            logger.warning(f"Failed to log caption to SQLite: {db_err}")
            
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
        chunks = semantic_chunking(cleaned_text)
        if not chunks:
            return False, "Could not create chunks from caption text."
        embed_model = get_embedding_model()
        chunks_with_prefix = [f"passage: {chunk}" for chunk in chunks]
        embeddings = embed_model.encode(chunks_with_prefix, show_progress_bar=False)
        doc_name = sanitize_collection_name(video_name)
        client = get_chroma_client()
        collection = client.get_or_create_collection(name=UNIFIED_COLLECTION_NAME)
        ids = [f"{doc_name}_caption_chunk_{i}" for i in range(len(chunks))]
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=[{
                "type": "video_caption",
                "doc_name": doc_name,
                "chunk_id": i,
                "source": doc_name,
                "created_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
        )
        return True, f"Captions for '{video_name}' indexed successfully ({len(chunks)} chunks)."
    except Exception as e:
        logger.error(f"Error processing video captions: {str(e)}")
        return False, f"Error: {str(e)}"


def get_available_videos() -> List[dict]:
    """Get list of available videos from SQLite."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT * FROM videos')
        rows = c.fetchall()
        
        videos = []
        for row in rows:
            if os.path.exists(row['video_path']):
                videos.append({
                    "filename": row['filename'],
                    "name": row['name'],
                    "path": row['video_path'],
                    "has_captions": bool(row['caption_path']),
                    "caption_data": load_caption_file(row['name']),
                    "size_mb": round(os.path.getsize(row['video_path']) / (1024 * 1024), 2)
                })
        conn.close()
        return sorted(videos, key=lambda x: x['filename'])
    except Exception as e:
        logger.error(f"Error fetching videos from DB: {str(e)}")
        return []


def delete_video(video_name: str) -> Tuple[bool, str]:
    """Delete a video and all its associated files from Disk and SQLite."""
    try:
        deleted_items = []
        
        try:
            conn = get_db_connection()
            c = conn.cursor()
            
            c.execute('SELECT video_path, caption_path FROM videos WHERE name = ?', (video_name,))
            row = c.fetchone()
            
            if row:
                if row['video_path'] and os.path.exists(row['video_path']):
                    os.remove(row['video_path'])
                    deleted_items.append("video file")
                if row['caption_path'] and os.path.exists(row['caption_path']):
                    os.remove(row['caption_path'])
                    deleted_items.append("caption file")
                    
            c.execute('DELETE FROM videos WHERE name = ?', (video_name,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed SQL Video Deletion cleanup: {e}")
            
        # Fallback disk scan for stragglers
        if not deleted_items:
            if os.path.exists(VIDEO_STORAGE_PATH):
                for filename in os.listdir(VIDEO_STORAGE_PATH):
                    if Path(filename).stem == video_name:
                        video_path = os.path.join(VIDEO_STORAGE_PATH, filename)
                        os.remove(video_path)
                        deleted_items.append(f"video file: {filename}")
            safe_name = sanitize_collection_name(video_name)
            caption_path = os.path.join(CAPTIONS_STORAGE_PATH, f"{safe_name}_captions.json")
            if os.path.exists(caption_path):
                os.remove(caption_path)
                deleted_items.append("caption file")
                
        try:
            client = get_chroma_client()
            collection = client.get_collection(name=UNIFIED_COLLECTION_NAME)
            doc_name = sanitize_collection_name(video_name)
            collection.delete(where={"source": doc_name})
            deleted_items.append("database collection")
        except Exception as e:
            logger.info(f"No ChromaDB collection chunks for {video_name}: {str(e)}")
            
        if deleted_items:
            return True, f"Deleted {', '.join(deleted_items)} for '{video_name}'."
        return False, f"No files found for video '{video_name}'."
    except Exception as e:
        logger.error(f"Error deleting video: {str(e)}")
        return False, f"Error deleting video: {str(e)}"


def extract_audio_from_video(video_path: str) -> Tuple[bool, str, Optional[str]]:
    """Extract mono 16 kHz WAV audio from video using ffmpeg."""
    try:
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        command = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1', '-y',
            audio_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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


def generate_captions_from_video(video_path: str, video_name: str, language: str = "en") -> Tuple[bool, str, Optional[str]]:
    """Generate captions from video using the appropriate ASR model."""
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
        word_count = len(caption_text.split())
        return True, f"Captions generated! ({word_count} words)", caption_text

    except Exception as e:
        logger.error(f"Error generating captions: {str(e)}")
        return False, f"Error: {str(e)}", None
