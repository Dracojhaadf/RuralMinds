"""
Services module for RuralMinds.
Provides AI, audio, translation, document processing, RAG retrieval, and video services.
"""
from services.vector_service import (
    get_embedding_model,
    get_chroma_client,
    get_paperqa_docs,
    sanitize_collection_name,
    get_available_documents,
    get_document_path,
    delete_document,
    get_document_stats,
    rebuild_database,
)
from services.audio_service import (
    get_whisper_model,
    get_hindi_asr_model,
    get_malayalam_whisper,
    transcribe_english,
    transcribe_hindi,
    transcribe_malayalam,
    transcribe_audio,
)
from services.translation_service import (
    detect_language,
    get_translation_model,
    translate_to_english,
    translate_from_english,
    translate_stream_from_english,
    normalize_query,
    translate_text,
)
from services.document_service import (
    extract_pdf,
    clean_text,
    sentence_based_chunking,
    semantic_chunking,
    process_and_save_pdf,
)
from services.retrieval_service import (
    is_document_query,
    reciprocal_rank_fusion,
    get_reranker_model,
    retrieve_context,
)
from services.llm_service import (
    query_ollama_simple,
    query_ollama_stream_simple,
    query_with_confidence,
    preload_ollama_model,
    query_ollama_stream,
    query_ollama,
    generate_answer_from_context,
    generate_answer_from_context_stream,
    query_saved_document_hybrid,
    query_saved_document_stream,
    query_saved_document,
)
from services.video_service import (
    save_video,
    save_caption_file,
    load_caption_file,
    process_video_captions,
    get_available_videos,
    delete_video,
    extract_audio_from_video,
    generate_captions_from_video,
)

__all__ = [
    # Vector & Document DB
    "get_embedding_model",
    "get_chroma_client",
    "get_paperqa_docs",
    "sanitize_collection_name",
    "get_available_documents",
    "get_document_path",
    "delete_document",
    "get_document_stats",
    "rebuild_database",
    # Audio
    "get_whisper_model",
    "get_hindi_asr_model",
    "get_malayalam_whisper",
    "transcribe_english",
    "transcribe_hindi",
    "transcribe_malayalam",
    "transcribe_audio",
    # Translation
    "detect_language",
    "get_translation_model",
    "translate_to_english",
    "translate_from_english",
    "translate_stream_from_english",
    "normalize_query",
    "translate_text",
    # Document Processing
    "extract_pdf",
    "clean_text",
    "sentence_based_chunking",
    "semantic_chunking",
    "process_and_save_pdf",
    # Retrieval
    "is_document_query",
    "reciprocal_rank_fusion",
    "get_reranker_model",
    "retrieve_context",
    # LLM
    "query_ollama_simple",
    "query_ollama_stream_simple",
    "query_with_confidence",
    "preload_ollama_model",
    "query_ollama_stream",
    "query_ollama",
    "generate_answer_from_context",
    "generate_answer_from_context_stream",
    "query_saved_document_hybrid",
    "query_saved_document_stream",
    "query_saved_document",
    # Video
    "save_video",
    "save_caption_file",
    "load_caption_file",
    "process_video_captions",
    "get_available_videos",
    "delete_video",
    "extract_audio_from_video",
    "generate_captions_from_video",
]
