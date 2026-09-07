import re
import logging
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

_translation_models = {}

def detect_language(text: str) -> str:
    """
    Detect language using langdetect + Romanized keyword heuristics.
    Returns: 'en' | 'hi' | 'ml'
    """
    try:
        from langdetect import detect

        indic_keywords = {
            'hi': ['bhai', 'kya', 'hai', 'kaise', 'kyun', 'aur', 'yeh', 'woh'],
            'ml': ['entha', 'engane', 'evide', 'cheyyane', 'und', 'illa', 'enth']
        }

        text_lower = text.lower()
        for lang, keywords in indic_keywords.items():
            if any(word in text_lower for word in keywords):
                logger.info(f"🔍 Detected Romanized {lang.upper()}")
                return lang

        detected = detect(text)
        logger.info(f"🔍 langdetect: {detected}")
        return detected if detected in ['hi', 'ml', 'en'] else 'en'

    except Exception as e:
        logger.warning(f"Language detection failed ({e}), defaulting to English")
        return 'en'


def get_translation_model(source_lang: str, target_lang: str):
    """Load and cache MarianMT translation model."""
    global _translation_models
    model_key = f"{source_lang}-{target_lang}"

    if model_key not in _translation_models:
        from transformers import MarianMTModel, MarianTokenizer

        model_map = {
            'hi-en': 'Helsinki-NLP/opus-mt-hi-en',
            'ml-en': 'Helsinki-NLP/opus-mt-ml-en',
            'en-hi': 'Helsinki-NLP/opus-mt-en-hi',
            'en-ml': 'Helsinki-NLP/opus-mt-en-ml'
        }

        if model_key not in model_map:
            logger.warning(f"No translation model for {model_key}")
            return None, None

        model_name = model_map[model_key]
        logger.info(f"📥 Loading translation model: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        _translation_models[model_key] = (tokenizer, model)
        logger.info(f"✓ Translation model loaded: {model_key}")

    return _translation_models[model_key]


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text from source_lang to English using MarianMT."""
    if source_lang == 'en':
        return text
    try:
        tokenizer, model = get_translation_model(source_lang, 'en')
        if not tokenizer or not model:
            return text
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        logger.info(f"🌍 Translated ({source_lang}→en): '{text}' → '{result}'")
        return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate text from English to target_lang using MarianMT."""
    if target_lang == 'en':
        return text
    try:
        tokenizer, model = get_translation_model('en', target_lang)
        if not tokenizer or not model:
            return text
        inputs = tokenizer([text], return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        logger.info(f"🌍 Translated (en→{target_lang}): '{text}' → '{result}'")
        return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text


def translate_stream_from_english(chunks_generator, target_lang: str):
    """
    Wraps an English text generator and yields translated chunks sentence-by-sentence.
    """
    if target_lang == 'en':
        for chunk in chunks_generator:
            yield chunk
        return

    buffer = ""
    for chunk in chunks_generator:
        if isinstance(chunk, dict):
            yield chunk
            continue
            
        buffer += chunk
        sentences = re.split(r'(?<=[.?!])\s+|\n', buffer)
        
        if len(sentences) > 1:
            for sentence in sentences[:-1]:
                if sentence.strip():
                    translated_sent = translate_from_english(sentence, target_lang)
                    yield translated_sent + " "
            buffer = sentences[-1]
            
    if buffer.strip():
        translated_sent = translate_from_english(buffer, target_lang)
        yield translated_sent


def normalize_query(query: str) -> str:
    """Detects lang and translates to EN."""
    lang = detect_language(query)
    if lang != 'en':
        return translate_to_english(query, lang)
    return query


def translate_text(text: str, target_language: str = "es") -> Tuple[bool, str, Optional[str]]:
    """Translate text to target language."""
    try:
        if target_language == "en":
            return True, "No translation needed", text
        from transformers import pipeline
        lang_models = {
            "hi": "Helsinki-NLP/opus-mt-en-hi",
            "ml": "Helsinki-NLP/opus-mt-en-ml",
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
