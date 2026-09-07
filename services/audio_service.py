import os
import logging
from typing import Optional, Tuple
from config.settings import MALAYALAM_MODEL_PATH

logger = logging.getLogger(__name__)

# ASR model cache globals
_whisper_model = None       # English Whisper (openai-whisper)
_hindi_asr_model = None     # Hindi (AI4Bharat)
_malayalam_processor = None # Fine-tuned Malayalam Whisper – WhisperProcessor
_malayalam_model = None     # Fine-tuned Malayalam Whisper – WhisperForConditionalGeneration


def get_whisper_model():
    """Lazy-load openai-whisper (small) for English ASR."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("🎤 Loading Whisper (small) for English…")
        import whisper
        _whisper_model = whisper.load_model("small")
        logger.info("✓ English Whisper loaded")
    return _whisper_model


def get_hindi_asr_model():
    """Lazy-load AI4Bharat Hindi ASR pipeline."""
    global _hindi_asr_model
    if _hindi_asr_model is None:
        logger.info("🎤 Loading AI4Bharat Hindi ASR…")
        from transformers import pipeline
        _hindi_asr_model = pipeline(
            "automatic-speech-recognition",
            model="ai4bharat/indicwav2vec-hindi",
            device="cpu"
        )
        logger.info("✓ Hindi ASR loaded")
    return _hindi_asr_model


def get_malayalam_whisper():
    """
    Lazy-load the locally fine-tuned Whisper Malayalam model.
    Reads from MALAYALAM_MODEL_PATH.
    """
    global _malayalam_processor, _malayalam_model

    if _malayalam_processor is None or _malayalam_model is None:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        # Validate model directory
        if not os.path.isdir(MALAYALAM_MODEL_PATH):
            raise FileNotFoundError(
                f"Malayalam model directory not found:\n  {MALAYALAM_MODEL_PATH}\n"
                "Please verify MALAYALAM_MODEL_PATH in config/settings.py."
            )

        required_files = [
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        missing = [f for f in required_files
                   if not os.path.isfile(os.path.join(MALAYALAM_MODEL_PATH, f))]
        if missing:
            raise FileNotFoundError(
                f"Malayalam model directory is missing files: {missing}\n"
                f"Directory: {MALAYALAM_MODEL_PATH}"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"🎤 Loading fine-tuned Malayalam Whisper…\n"
            f"   Path   : {MALAYALAM_MODEL_PATH}\n"
            f"   Device : {device}"
        )

        _malayalam_processor = WhisperProcessor.from_pretrained(
            MALAYALAM_MODEL_PATH,
            local_files_only=True,
        )

        _malayalam_model = WhisperForConditionalGeneration.from_pretrained(
            MALAYALAM_MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
        ).to(device)

        _malayalam_model.eval()

        # Patch generation_config so it always transcribes Malayalam
        _malayalam_model.generation_config.forced_decoder_ids = None
        _malayalam_model.generation_config.suppress_tokens = []

        logger.info("✓ Fine-tuned Malayalam Whisper loaded successfully")

    return _malayalam_processor, _malayalam_model


def transcribe_english(audio_path: str) -> str:
    """Transcribe English audio using openai-whisper."""
    model = get_whisper_model()
    result = model.transcribe(audio_path, language="en")
    return result.get("text", "").strip()


def transcribe_hindi(audio_path: str) -> str:
    """Transcribe Hindi audio using AI4Bharat model."""
    model = get_hindi_asr_model()
    result = model(audio_path)
    if isinstance(result, list):
        return result[0].get("text", "").strip()
    return result.get("text", "").strip()


def transcribe_malayalam(audio_path: str) -> str:
    """
    Transcribe Malayalam audio using the locally fine-tuned Whisper model.
    """
    import torch
    import librosa
    import numpy as np

    processor, model = get_malayalam_whisper()
    device = next(model.parameters()).device

    logger.info(f"  Loading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    audio = np.asarray(audio, dtype=np.float32)
    logger.info(f"  Audio shape={audio.shape}, sr={sr}")

    feat_out = processor.feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )
    if hasattr(feat_out, "input_features"):
        input_features = feat_out.input_features.to(device)
    elif isinstance(feat_out, dict):
        input_features = feat_out["input_features"].to(device)
    elif isinstance(feat_out, list):
        input_features = torch.tensor(np.array(feat_out[0])).unsqueeze(0).to(device)
    else:
        raise ValueError(f"Unexpected feature extractor output type: {type(feat_out)}")

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ml",
        task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=200,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    text = processor.tokenizer.decode(
        predicted_ids[0],
        skip_special_tokens=True
    )

    logger.info(f"🔥 Fine-Tuned Malayalam Output: {text}")
    return text.strip()


def transcribe_audio(audio_path: str, language_code: str = "en") -> tuple:
    """
    Transcribe audio using language-specific ASR models.
    """
    try:
        logger.info(f"🎤 Transcribing [{language_code.upper()}]: {audio_path}")

        if language_code == "ml":
            text = transcribe_malayalam(audio_path)
            if not text:
                return False, "Voice not clear or no speech detected. Please try again.", None
            return True, "Transcription successful", {"text": text}

        elif language_code == "hi":
            text = transcribe_hindi(audio_path)

        elif language_code == "en":
            text = transcribe_english(audio_path)

        else:
            logger.warning(f"Unknown language '{language_code}', falling back to English Whisper")
            text = transcribe_english(audio_path)

        if not text:
            return False, "Voice not clear. Please try again.", None

        logger.info(f"🎤 Transcription result: {text}")
        return True, "Transcription successful", {"text": text}

    except FileNotFoundError as e:
        logger.error(f"Model not found: {str(e)}")
        return False, f"Model not found: {str(e)}", None
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return False, f"Transcription error: {str(e)}", None
