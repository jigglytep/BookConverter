# =============================================================================
# KOKORO-82M MODEL SETTINGS
# =============================================================================

# Language code controls which G2P (text-to-phoneme) rules are used.
# Must match the prefix of VOICE below (see VOICE_LANG_PREFIXES).
#   a = American English   b = British English   e = Spanish
#   f = French             h = Hindi              i = Italian
#   j = Japanese           p = Brazilian Portuguese  z = Mandarin Chinese
# Leave as None to auto-detect from the voice name's prefix.
LANG_CODE = None

# Default voice. Override per-run with --voice.
VOICE = "af_heart"

# Speech rate multiplier (0.5 - 2.0 is a sane range).
SPEED = 1.0

# "cuda", "cpu", or None to auto-detect (uses CUDA if available).
DEVICE = None

# Sample rate Kokoro outputs at (fixed by the model, don't change).
SAMPLE_RATE = 24000

# Reference list of built-in voices (kokoro>=0.9). Not exhaustive if the
# package adds more later - use --list-voices to print this list.
VOICE_OPTIONS = {
    "a": {  # American English
        "female": ["af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
                   "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
        "male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                 "am_michael", "am_onyx", "am_puck", "am_santa"],
    },
    "b": {  # British English
        "female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
        "male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    },
    "j": {"female": ["jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"], "male": ["jm_kumo"]},
    "z": {"female": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi"], "male": ["zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"]},
    "e": {"female": ["ef_dora"], "male": ["em_alex", "em_santa"]},
    "f": {"female": ["ff_siwis"], "male": []},
    "h": {"female": ["hf_alpha", "hf_beta"], "male": ["hm_omega", "hm_psi"]},
    "i": {"female": ["if_sara"], "male": ["im_nicola"]},
    "p": {"female": ["pf_dora"], "male": ["pm_alex", "pm_santa"]},
}

VOICE_LANG_PREFIXES = {
    "a": ("af_", "am_"), "b": ("bf_", "bm_"), "e": ("ef_", "em_"),
    "f": ("ff_", "fm_"), "h": ("hf_", "hm_"), "i": ("if_", "im_"),
    "j": ("jf_", "jm_"), "p": ("pf_", "pm_"), "z": ("zf_", "zm_"),
}

# =============================================================================
# CHUNKING / PROCESSING SETTINGS
# =============================================================================

# Words per top-level chunk file. Kokoro's own pipeline further splits each
# chunk into model-sized sub-segments internally, so this mainly controls
# how granular caching/resuming/progress reporting is.
CHUNK_SIZE_WORDS = 500

# Silence gap (ms) inserted between chunks when the final file is assembled.
CHUNK_GAP_MS = 250

# Silence gap (ms) inserted between Kokoro's internal sub-segments within a
# single chunk (keeps sentence-heavy chunks from running together).
SUBSEGMENT_GAP_MS = 80

MAX_RETRIES = 3

# =============================================================================
# I/O SETTINGS
# =============================================================================

AUDIOBOOKS_FOLDER = "audiobooks"
CHUNKS_FOLDER = "chunks"
CACHE_FOLDER = "cache/audio_chunks"
LOGS_FOLDER = "logs"

AUDIO_FORMAT = "m4b"      # m4b, mp3, wav, m4a, flac (anything pydub/ffmpeg supports)
                          # m4b embeds a chapter marker for every source page (requires ffmpeg on PATH)
AUDIO_BITRATE = "192k"

# Text file extensions accepted when reading a directory of book pages.
PAGE_FILE_EXTENSIONS = (".txt", ".md")
