import json
import logging
import requests
from typing import Optional, List, Tuple

from config.settings import (
    OLLAMA_MODEL,
    OLLAMA_API_URL,
    MAX_CONTEXT_LENGTH,
    DEFAULT_K_RESULTS
)
from services.translation_service import (
    detect_language,
    translate_to_english,
    translate_from_english,
    translate_stream_from_english
)
from services.retrieval_service import retrieve_context

logger = logging.getLogger(__name__)


def query_ollama_simple(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    """Simple Ollama query without context. Used for confidence checking."""
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
    """Stream simple Ollama query without context. Yields chunks of text."""
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
    Returns: (answer, needs_rag)
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
        response_lower = response.lower()
        if "[need_context]" in response_lower or "need context" in response_lower or "[need context]" in response:
            logger.info("⚡ LLM not confident - triggering RAG")
            return None, True
        logger.info("✓ LLM confident - returning direct answer")
        return response.strip(), False
    except Exception as e:
        logger.error(f"Error in confidence check: {str(e)}")
        return None, True


def preload_ollama_model():
    """Warm up Ollama so the model is in RAM before first user query."""
    try:
        logger.info(f"Warming up Ollama model: {OLLAMA_MODEL}…")
        warmup_payload = {
            "model": OLLAMA_MODEL,
            "prompt": "Hello",
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_predict": 5}
        }
        response = requests.post(OLLAMA_API_URL, json=warmup_payload, timeout=30)
        if response.status_code == 200:
            logger.info(f"✓ Ollama model {OLLAMA_MODEL} ready")
            return True
        logger.warning(f"Ollama warm-up status: {response.status_code}")
        return False
    except Exception as e:
        logger.warning(f"Could not preload Ollama: {str(e)}")
        return False


def query_ollama_stream(context: str, query: str):
    """Stream Ollama response with context. Yields text chunks."""
    try:
        system_prompt = """
You are an AI Tutor.

Rules:
0. You can get questions in English, Malayalam or Hindi.
1. Answer the question ONLY using the provided context. Do NOT use external knowledge.
2. If the context does not contain the answer, say "I could not find this in the document."
3. Do NOT guess the user's intention beyond the question.
4. Do NOT mention language detection.
5. Do NOT translate unless explicitly asked.
6. If the question is informal or partially in another language, interpret it as a simple academic question.
7. Never explain what language the user used.
8. Do not add extra commentary.
9. Keep answers clear, structured, and correct.
10. If you don't understand some words, don't ask about them in brackets.
11. "me farak" means difference between.
12. "kya hai" means what do you mean by that topic.
13. "enthanu" also means what do you mean by that topic.
14. Always check if the language is in Malayalam or Hindi (and same with words).
15. User will never ask you anything in Tamil; if you think it's Tamil, the language is Malayalam.
If the user asks in mixed language (e.g., Malayalam/Hindi written in English),
interpret the meaning and reply in that language.
"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
            "stream": True,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": 1024}
        }

        with requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'response' in chunk:
                                yield chunk['response']
                        except json.JSONDecodeError:
                            continue
            else:
                yield "Error: Could not connect to AI model."

    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {str(e)}")
        yield f"Error: {str(e)}"


def query_ollama(context: str, query: str) -> Optional[str]:
    """Blocking Ollama query with context."""
    try:
        system_prompt = """You are a helpful AI tutor.
Answer clearly using only the provided context.
If the context does not contain the answer, say so.
Use short structured explanations."""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.2, "num_predict": 1024}
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get('response', '')
        logger.warning(f"Ollama status: {response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {str(e)}")
        return None


def generate_answer_from_context(
    retrieved_chunks: List[str],
    query: str,
    max_length: int = MAX_CONTEXT_LENGTH
) -> str:
    """Generate answer from retrieved chunks (blocking)."""
    if not retrieved_chunks:
        return "No relevant information found in the document."

    context = ""
    for chunk in retrieved_chunks:
        if len(context) + len(chunk) < max_length:
            context += chunk + "\n\n"
        else:
            break
    context = context.strip()

    if not context:
        return "Retrieved context too long to process."

    try:
        response = query_ollama(context, query)
        if response:
            return response
    except Exception as e:
        logger.warning(f"Ollama generation failed: {str(e)}")

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
    """Generate answer from retrieved chunks (streaming)."""
    if not retrieved_chunks:
        yield "No relevant information found in the document."
        return

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

    try:
        for chunk in query_ollama_stream(context, query):
            yield chunk
        return
    except Exception as e:
        logger.warning(f"Ollama generation failed: {str(e)}")

    yield f"""Based on the document, here is the relevant information:

QUESTION: {query}

RELEVANT EXCERPTS:
{context}

(Note: AI generation unavailable - showing direct extracts)"""


def query_saved_document_hybrid(
    doc_name: str,
    query: str,
    k: int = DEFAULT_K_RESULTS
) -> Tuple[str, List[str]]:
    """Always-RAG query: retrieve from document, then generate answer."""
    logger.info("📚 Running RAG pipeline…")
    try:
        original_lang = detect_language(query)
        english_query = query
        if original_lang != 'en':
            english_query = translate_to_english(query, original_lang)

        retrieved_chunks, scores = retrieve_context(query, doc_name=doc_name, k=k)

        if not retrieved_chunks:
            return "No relevant information found for this query in the document.", []

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks (scores: {[f'{s:.3f}' for s in scores]})")

        answer_en = generate_answer_from_context(retrieved_chunks, english_query)
        answer = translate_from_english(answer_en, original_lang)
        return answer, retrieved_chunks

    except Exception as e:
        logger.error(f"Error during RAG: {str(e)}")
        return f"Error generating answer: {str(e)}", []


def query_saved_document_stream(
    doc_name: str,
    query: str,
    k: int = DEFAULT_K_RESULTS,
    forced_language: str = None
):
    """
    Streaming query pipeline — always uses RAG when a document is selected.
    """
    if forced_language:
        original_lang = forced_language
        logger.info(f"🔒 Forced language: {forced_language}")
    else:
        original_lang = detect_language(query)
        logger.info(f"🔍 Auto-detected language: {original_lang}")

    english_query = query
    if original_lang != 'en':
        english_query = translate_to_english(query, original_lang)
        logger.info(f"🔄 Translated query: {english_query}")

    logger.info(f"📚 Running RAG pipeline for '{doc_name}'…")
    try:
        retrieved_chunks, scores = retrieve_context(query, doc_name=doc_name, k=k)

        if not retrieved_chunks:
            yield "No relevant information found for this query in the document."
            yield {'sources': []}
            return

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks (scores: {[f'{s:.3f}' for s in scores]})")

        llm_stream = generate_answer_from_context_stream(retrieved_chunks, english_query)
        translated_stream = translate_stream_from_english(llm_stream, original_lang)
        
        for chunk in translated_stream:
            yield chunk
        yield {'sources': retrieved_chunks}

    except Exception as e:
        logger.error(f"Error during RAG: {str(e)}")
        yield f"Error during RAG: {str(e)}"
        yield {'sources': []}


def query_saved_document(doc_name: str, query: str, k: int = DEFAULT_K_RESULTS) -> Tuple[str, List[str]]:
    """Backward compatible wrapper."""
    return query_saved_document_hybrid(doc_name, query, k)
