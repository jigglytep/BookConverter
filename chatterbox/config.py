# =============================================================================
# TTS ENGINE, MODEL & CONVERSION SETTINGS (QWEN3-TTS DEFAULT / CHATTERBOX TOGGLE)
# =============================================================================

import os
from pathlib import Path


def normalize_tts_engine(engine: str | None) -> str:
    """Normalize TTS engine identifier to 'qwen3' (default) or 'chatterbox'."""
    val = (engine or os.environ.get("TTS_ENGINE", "qwen3")).strip().lower()
    if val in ("chatterbox", "chatterbox-tts", "chatterbox_tts", "cb"):
        return "chatterbox"
    if val in ("qwen3", "qwen", "qwen3-tts", "qwen3_tts", "qwen-tts", "qwen_tts", ""):
        return "qwen3"
    raise ValueError(f"Unsupported TTS_ENGINE={engine!r}. Choose 'qwen3' or 'chatterbox'.")


# Active TTS Engine: "qwen3" (default) or "chatterbox" (fallback toggle via TTS_ENGINE env var)
TTS_ENGINE = normalize_tts_engine(os.environ.get("TTS_ENGINE", "qwen3"))

# Dedicated RAID0 environment and HuggingFace cache paths (avoiding 15 GB root /)
QWEN_TTS_ENV = Path(os.environ.get("QWEN_TTS_ENV", "/mnt/raid0/qwen/src/tts-env"))
HF_HOME = os.environ.get("HF_HOME", "/mnt/raid0/qwen/huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)

# Model architecture:
# - For Qwen3-TTS: "standard"/"1.7b" (Qwen/Qwen3-TTS-12Hz-1.7B-Base) or "turbo"/"0.6b" (Qwen/Qwen3-TTS-12Hz-0.6B-Base)
# - For Chatterbox: "standard" (ChatterboxTTS 500M) or "turbo" (ChatterboxTurboTTS 350M)
MODEL_TYPE = os.environ.get("TTS_MODEL_TYPE", "standard")

QWEN_MODEL_MAP = {
    "standard": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "turbo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
QWEN_DEFAULT_MODEL = os.environ.get("QWEN_TTS_MODEL", QWEN_MODEL_MAP["standard"])

# Dual NVIDIA Tesla P100 (Pascal sm_60) execution settings:
# Enforce standard PyTorch FP16 and SDPA/CUDA paths (no bfloat16 or Tensor Core FlashAttention-2)
QWEN_DTYPE = "float16"
QWEN_ATTN_IMPLEMENTATION = "sdpa"
QWEN_LANGUAGE = os.environ.get("QWEN_TTS_LANGUAGE", "English")
CHATTERBOX_FP16 = True

# Audio sample rate (24 kHz for both Qwen3-TTS 12Hz and Chatterbox S3Gen)
SAMPLE_RATE = 24000

# Number of parallel TTS worker processes spawned per GPU.
# On 16 GB Tesla P100 GPUs, each Qwen3-TTS 1.7B FP16 worker uses ~4.5 GiB VRAM,
# so 3 workers per GPU utilizes ~13.5-14.5 GiB / 16 GiB (~85-90% VRAM) across both P100s.
WORKERS_PER_GPU = int(os.environ.get("TTS_WORKERS_PER_GPU", "3"))

# Base GPU devices and expanded multi-worker device allocation list (6 workers total across cuda:0 & cuda:1)
GPU_DEVICES = ["cuda:0", "cuda:1"]
DEVICES = GPU_DEVICES * max(1, WORKERS_PER_GPU)

# Default reference voice ID (folder inside references/ containing .wav audio and .lab transcript).
# Set to "practical_engineering" (Grady Hillhouse) for engaging, clear technical narration.
DEFAULT_REFERENCE_ID = "practical_engineering"

# Optional direct path to a reference audio file (and optional transcript) for zero-shot voice cloning.
DEFAULT_REFERENCE_AUDIO = None
DEFAULT_REFERENCE_TEXT = None

# Generation sampling parameters (tuned for ADHD engagement & technical clarity:
# slightly elevated expressiveness/CFG to prevent monotone drift while keeping
# temperature controlled for crisp articulation of engineering terminology)
TEMPERATURE = 0.72
TOP_P = 0.95
REPETITION_PENALTY = 1.25
CFG_WEIGHT = 0.55
EXAGGERATION = 0.58
MAX_NEW_TOKENS = 1000

# =============================================================================
# CHUNKING / PROCESSING SETTINGS (ADHD-OPTIMIZED)
# =============================================================================

# Target max words per speech/highlight chunk.
# 24 words (with single-sentence priority) keeps each EPUB3 SMIL read-along highlight
# span to ~1 sentence (3-7 seconds), anchoring visual tracking and preventing
# multi-sentence "wall of highlight" attention drift.
CHUNK_SIZE_WORDS = 72

# Silence gap (ms) inserted between chunks when assembling the audiobook.
# Combined with automatic leading/trailing dead-air trimming on each synthesized
# chunk, 180 ms keeps sentence transitions brisk and conversational.
CHUNK_GAP_MS = 180

# Maximum retry attempts per chunk if synthesis fails.
MAX_RETRIES = 3


# =============================================================================
# I/O SETTINGS
# =============================================================================

AUDIOBOOKS_FOLDER = "audiobooks"
CHUNKS_FOLDER = "chunks"
CACHE_FOLDER = "cache/audio_chunks"
LOGS_FOLDER = "logs"
REFERENCES_FOLDER = "references"

AUDIO_FORMAT = "m4b"      # m4b, mp3, wav, m4a, flac (anything ffmpeg supports)
                          # m4b embeds chapter markers for each source page
AUDIO_BITRATE = "192k"

# File extensions recognized when scanning directories of book pages.
PAGE_FILE_EXTENSIONS = (".txt", ".md")
