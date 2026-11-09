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

# Automatically calculate available GPU VRAM and spawn additional worker instances
# per GPU when sufficient free memory is available.
AUTO_SCALE_GPU_INSTANCES = True

# Estimated working VRAM per Chatterbox worker instance (MiB) used as a floor
# before/during warm-up (~6.0 GiB / ~37% of a 16 GB P100 for standard, ~4.2 GiB for turbo).
INSTANCE_VRAM_ESTIMATE_MIB = {
    "standard": 6000,
    "turbo": 4200,
}

# Minimum free VRAM headroom (MiB) to keep unallocated on each GPU when adding instances.
GPU_MEMORY_SAFETY_BUFFER_MIB = 800

# Maximum worker instances allowed per single GPU.
MAX_INSTANCES_PER_GPU = 4

# Interval (seconds) between dynamic GPU memory checks during synthesis.
GPU_SCALE_CHECK_INTERVAL_SEC = 45.0

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
# ARTIFACT & HALLUCINATION PREVENTION (Chatterbox-TTS-Extended / Issue #97)
# =============================================================================

# Merge short sentences (< MIN_SENTENCE_CHARS characters) with the sentence after
# or before them to avoid short-segment gibberish and hallucinations.
SMART_APPEND_SHORT_SENTENCES = True
MIN_SENTENCE_CHARS = 20

# Number of candidates generated per chunk (default: 3).
# Among candidates that pass Whisper Sync validation, the shortest-duration
# candidate is selected because hallucinations/artifacts lengthen the audio.
NUM_CANDIDATES = 3

# Maximum generation/validation retry attempts per candidate when Whisper validation fails (default: 3).
MAX_ATTEMPTS = 3

# Base random seed for deterministic per-chunk/per-candidate seed derivation (0 = deterministic chunk seed).
SEED = 0

# Whisper Sync validation settings
BYPASS_WHISPER_CHECKING = False
USE_FASTER_WHISPER = True
WHISPER_MODEL = "base"               # "tiny", "base", "small", "medium", "large"
WHISPER_DEVICE = "cpu"               # "cpu" (int8, zero GPU VRAM overhead) or "cuda"
WHISPER_VALIDATION_THRESHOLD = 0.85  # Fuzzy sequence match ratio required to pass validation
USE_LONGEST_TRANSCRIPT_ON_FAIL = True  # True: pick longest transcript on fail; False: pick highest fuzzy score

# Auto-Editor post-processing to strip low-sounding hallucinations / extended breathing
USE_AUTO_EDITOR = True
AUTO_EDITOR_THRESHOLD = 0.06         # Volume threshold under which audio is cut (default: 0.06)
AUTO_EDITOR_MARGIN = 0.2             # Buffer margin (seconds) kept before and after cuts (default: 0.2)
KEEP_ORIGINAL_WAV = False            # Keep pre-auto-editor backup wav Alongside cleaned wav


# =============================================================================
# I/O SETTINGS
# =============================================================================

AUDIOBOOKS_FOLDER = "audiobooks"
CHUNKS_FOLDER = "chunks"
CACHE_FOLDER = "caches/audio_chunks"
LOGS_FOLDER = "logs"
REFERENCES_FOLDER = "references"

AUDIO_FORMAT = "m4b"      # m4b, mp3, wav, m4a, flac (anything ffmpeg supports)
                          # m4b embeds chapter markers for each source page
AUDIO_BITRATE = "192k"

# File extensions recognized when scanning directories of book pages.
PAGE_FILE_EXTENSIONS = (".txt", ".md")
