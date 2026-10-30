# =============================================================================
# CHATTERBOX TTS MODEL & CONVERSION SETTINGS
# =============================================================================

from pathlib import Path

# Model architecture: "standard" (ChatterboxTTS 500M) or "turbo" (ChatterboxTurboTTS 350M)
MODEL_TYPE = "standard"

# Audio sample rate (fixed at 24 kHz by Chatterbox S3Gen)
SAMPLE_RATE = 24000

# "cuda", "cuda:0", "cuda:1", "cpu", or list of devices for multi-GPU parallel synthesis e.g. ["cuda:0", "cuda:1"]
# None auto-detects available CUDA devices.
DEVICES = ["cuda:0", "cuda:1"]

# Default reference voice ID (folder inside references/ containing .wav audio).
# Set to "practical_engineering" (Grady Hillhouse) for engaging, clear technical narration.
DEFAULT_REFERENCE_ID = "practical_engineering"

# Optional direct path to a reference audio file for zero-shot voice cloning.
DEFAULT_REFERENCE_AUDIO = None

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
CHUNK_SIZE_WORDS = 24

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
