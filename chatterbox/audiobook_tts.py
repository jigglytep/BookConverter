#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Default HuggingFace cache to RAID0 storage to avoid filling the 15 GB root partition (/)
os.environ.setdefault("HF_HOME", "/mnt/raid0/qwen/huggingface")


def _detect_requested_engine(argv: list[str]) -> str:
    """Detect requested TTS engine ('qwen3' default or 'chatterbox') before venv re-exec."""
    engine = os.environ.get("TTS_ENGINE", "qwen3").strip().lower()
    for i, arg in enumerate(argv):
        if arg in ("--tts-engine", "--engine") and i + 1 < len(argv):
            engine = argv[i + 1].strip().lower()
        elif arg.startswith("--tts-engine=") or arg.startswith("--engine="):
            engine = arg.split("=", 1)[1].strip().lower()
        elif arg == "--chatterbox":
            engine = "chatterbox"
        elif arg in ("--qwen3", "--qwen"):
            engine = "qwen3"
    if engine in ("chatterbox", "chatterbox-tts", "chatterbox_tts", "cb"):
        return "chatterbox"
    return "qwen3"


# Automatically re-execute inside the appropriate dedicated environment on RAID0
_script_dir = Path(__file__).resolve().parent
_requested_engine = _detect_requested_engine(sys.argv[1:])
_qwen_env_dir = Path(os.environ.get("QWEN_TTS_ENV", "/mnt/raid0/qwen/src/tts-env"))
_qwen_python = _qwen_env_dir / "bin" / "python3"
_chatterbox_python = _script_dir / ".venv" / "bin" / "python3"

if _requested_engine == "qwen3" and _qwen_python.exists():
    _target_python = _qwen_python
    _target_site = _qwen_env_dir / "lib" / "python3.11" / "site-packages"
else:
    _target_python = _chatterbox_python if _chatterbox_python.exists() else _qwen_python
    _target_site = _script_dir / ".venv" / "lib" / "python3.11" / "site-packages"

if (
    __name__ == "__main__"
    and _target_python.exists()
    and Path(sys.executable) != _target_python
):
    os.execv(str(_target_python), [str(_target_python)] + sys.argv)

if _target_site.exists() and str(_target_site) not in sys.path:
    sys.path.insert(0, str(_target_site))
for _fallback_site in (
    _qwen_env_dir / "lib" / "python3.11" / "site-packages",
    _script_dir / ".venv" / "lib" / "python3.11" / "site-packages",
):
    if _fallback_site.exists() and str(_fallback_site) not in sys.path:
        sys.path.append(str(_fallback_site))

"""
Chatterbox TTS Audiobook & Document Converter
=============================================
Reads plain text or Markdown books (or directories of page files) and synthesizes
high-fidelity audiobooks (.m4b with chapter markers, .mp3, .wav) using Chatterbox TTS.

Features:
- Multi-GPU parallel synthesis: distributes chunks across available Tesla P100 / CUDA GPUs.
- Architecture choices: standard 500M ChatterboxTTS or 350M ChatterboxTurboTTS.
- Zero-shot voice cloning via reference audio, or built-in natural timbre.
- Sentence-aware word chunking tailored for stable prosody and token limits.
- Persistent MD5 chunk caching for instant resume.
- Embedded chapter markers for M4B output (one chapter per source page).
- Rich CLI progress bar with ETA, chunk rate, and elapsed time.
- Batch processing and per-page audio export modes.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.multiprocessing as mp
from pydub import AudioSegment
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Add chatterbox_repo/src to python path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_SRC = SCRIPT_DIR / "chatterbox_repo" / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import uuid
import config
from mineru_to_epub import (
    DEFAULT_CSS,
    PageDOMResult,
    PageOverlayTiming,
    SpeechChunk,
    _default_md_tts_opts,
    assemble_page_audio_and_smil,
    canonical_book_name,
    extract_page_number,
    extract_title_from_markdown,
    extract_toc_mapping,
    find_companion_markdown_dir,
    format_page_chapter_title,
    gather_page_documents,
    load_or_extract_book_metadata,
    package_epub3,
    prepare_page_dom_and_chunks,
    read_file_text,
)

MINERU_SCRIPTS_DIR = SCRIPT_DIR.parent / "mineru" / "scripts"
if MINERU_SCRIPTS_DIR.exists() and str(MINERU_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(MINERU_SCRIPTS_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))
try:
    from metadata_extractor import build_abs_metadata_json
except ImportError:
    build_abs_metadata_json = None
try:
    import status_display as _sdisp
except ImportError:
    _sdisp = None

console = Console()


def _progress_columns():
    return [
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TextColumn("[cyan]{task.fields[speed]}"),
        TextColumn("elapsed"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    ]


def _clean_and_normalize_chunk_wav(wav_np: np.ndarray, sr: int) -> np.ndarray:
    """
    ADHD-friendly audio cleanup per chunk:
    1. Trims variable leading/trailing dead air below -42 dB relative to active speech RMS
       (retaining a natural 25 ms pad and 12 ms cosine fade-in/out) so SMIL highlights
       and sentence transitions never lag or hang on trailing silence.
    2. Normalizes active speech RMS to -20 dBFS with soft peak limiting at -1.0 dBFS
       so volume remains steady across every sentence.
    """
    if wav_np.ndim > 1:
        wav_np = wav_np.squeeze()
    wav_np = np.nan_to_num(wav_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(wav_np) < int(sr * 0.08):
        return wav_np

    # Frame energy analysis (10 ms frames)
    win = max(1, int(sr * 0.01))
    n_win = len(wav_np) // win
    if n_win > 2:
        frames = wav_np[: n_win * win].reshape(n_win, win)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        ref_rms = float(np.percentile(frame_rms, 85))
        if ref_rms > 1e-5:
            thresh = ref_rms * (10.0 ** (-42.0 / 20.0))
            active_idx = np.where(frame_rms > thresh)[0]
            if len(active_idx) > 0:
                pad_samples = int(sr * 0.025)
                start_sample = max(0, int(active_idx[0] * win) - pad_samples)
                end_sample = min(len(wav_np), int((active_idx[-1] + 1) * win) + pad_samples)
                if end_sample - start_sample >= int(sr * 0.08):
                    wav_np = wav_np[start_sample:end_sample].copy()

    # Smooth 12 ms cosine fade-in / fade-out to prevent boundary clicks
    fade_len = min(int(sr * 0.012), len(wav_np) // 4)
    if fade_len > 1:
        fade_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len, dtype=np.float32) / fade_len))
        fade_out = fade_in[::-1]
        wav_np[:fade_len] *= fade_in
        wav_np[-fade_len:] *= fade_out

    # Normalize RMS to -20 dBFS (target_rms = 0.1) with peak ceiling at -1.0 dBFS (0.891)
    rms = float(np.sqrt(np.mean(wav_np ** 2) + 1e-12))
    if rms > 1e-4:
        target_rms = 0.10
        gain = min(target_rms / rms, 4.0)
        wav_np = wav_np * gain
    peak = float(np.max(np.abs(wav_np)))
    max_peak = 0.89125  # -1.0 dBFS
    if peak > max_peak:
        wav_np = wav_np * (max_peak / peak)

    return wav_np


def _worker_process_fn(
    device: str,
    model_type: str,
    ref_audio: Optional[str],
    gen_kwargs: dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """Worker process running on a dedicated GPU/device (supports Qwen3-TTS and Chatterbox in FP16)."""
    model = None
    voice_clone_prompt = None
    tts_engine = "qwen3"
    is_cuda = False
    ref_text = None
    try:
        os.environ.setdefault("HF_HOME", getattr(config, "HF_HOME", "/mnt/raid0/qwen/huggingface"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
        tts_engine = config.normalize_tts_engine(gen_kwargs.get("tts_engine", config.TTS_ENGINE))
        ref_text = gen_kwargs.get("ref_text")

        _sdir = Path(__file__).resolve().parent
        _qsite = Path(getattr(config, "QWEN_TTS_ENV", "/mnt/raid0/qwen/src/tts-env")) / "lib" / "python3.11" / "site-packages"
        _vsite = _sdir / ".venv" / "lib" / "python3.11" / "site-packages"
        if tts_engine == "qwen3":
            for p in (_qsite, _vsite):
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
        else:
            for p in (_vsite, _qsite):
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
        _rsrc = _sdir / "chatterbox_repo" / "src"
        if str(_rsrc) not in sys.path:
            sys.path.insert(0, str(_rsrc))

        is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        if is_cuda:
            # Optimize for NVIDIA Tesla P100 (Pascal sm_60: standard FP16 CUDA paths, no TF32/bfloat16)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = True
            try:
                dev_idx = int(str(device).split(":")[1]) if ":" in str(device) else 0
                torch.cuda.set_device(dev_idx)
                w_per_gpu = int(gen_kwargs.get("workers_per_gpu", getattr(config, "WORKERS_PER_GPU", 3)))
                if w_per_gpu >= 3:
                    torch.cuda.set_per_process_memory_fraction(0.31, dev_idx)
            except Exception:
                pass

        if tts_engine == "qwen3":
            import gc
            import transformers
            transformers.logging.set_verbosity_error()
            from qwen_tts import Qwen3TTSModel

            qwen_model_id = gen_kwargs.get("qwen_model_id") or config.QWEN_MODEL_MAP.get(
                str(model_type).lower(), config.QWEN_DEFAULT_MODEL
            )
            target_dtype = torch.float16 if is_cuda else torch.float32
            # Load on CPU first and cast speech_tokenizer from float32 -> float16 before moving to GPU
            # to avoid a temporary float32 VRAM spike when 3 workers share a 16 GB P100.
            model = Qwen3TTSModel.from_pretrained(
                qwen_model_id,
                device_map="cpu" if is_cuda else device,
                dtype=target_dtype,
                attn_implementation=getattr(config, "QWEN_ATTN_IMPLEMENTATION", "sdpa"),
            )
            st_model = None
            if is_cuda:
                if hasattr(model, "model") and model.model is not None:
                    if hasattr(model.model, "speech_tokenizer") and model.model.speech_tokenizer is not None:
                        st_model = getattr(model.model.speech_tokenizer, "model", None)
                        if st_model is not None:
                            st_model.to(dtype=torch.float16)
                            st_model.to(device=device, dtype=torch.float16)
                        model.model.speech_tokenizer.device = torch.device(device)
                    model.model.to(device=device, dtype=torch.float16)
                model.device = torch.device(device)
            if ref_audio:
                has_ref_text = bool(ref_text and str(ref_text).strip())
                with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=is_cuda):
                    voice_clone_prompt = model.create_voice_clone_prompt(
                        ref_audio=ref_audio,
                        ref_text=str(ref_text).strip() if has_ref_text else None,
                        x_vector_only_mode=not has_ref_text,
                    )
                # Once voice_clone_prompt is cached, speaker_encoder and speech_tokenizer.encoder
                # are never used again during chunk synthesis; free them to save ~400 MiB VRAM per worker.
                if is_cuda and voice_clone_prompt is not None:
                    if hasattr(model, "model") and model.model is not None:
                        model.model.speaker_encoder = None
                    if st_model is not None and hasattr(st_model, "encoder"):
                        st_model.encoder = None
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            if model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                model = ChatterboxTurboTTS.from_pretrained(device=device)
            else:
                from chatterbox.tts import ChatterboxTTS
                model = ChatterboxTTS.from_pretrained(device=device)
            if ref_audio:
                with torch.inference_mode(), torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=(is_cuda and getattr(config, "CHATTERBOX_FP16", True)),
                ):
                    model.prepare_conditionals(
                        ref_audio,
                        exaggeration=gen_kwargs.get("exaggeration", 0.5),
                    )
        if is_cuda:
            torch.cuda.empty_cache()
    except Exception as e:
        result_queue.put((-1, "init_error", f"Failed to load {tts_engine} model on {device}: {e}"))
        return

    result_queue.put((-1, "ready", device))

    try:
        while True:
            try:
                task = task_queue.get()
                if task is None:
                    break

                chunk_idx, text, cache_path_str, chunk_file_str = task
                cache_path = Path(cache_path_str)
                chunk_file = Path(chunk_file_str)

                # Check cache first
                if not gen_kwargs.get("no_cache", False) and cache_path.exists():
                    shutil.copy2(cache_path, chunk_file)
                    num_words = len(text.split())
                    result_queue.put((chunk_idx, "cached", num_words, 0.0, device))
                    continue

                # Synthesize (voice clone prompt / conditionals already prepared on worker init)
                t0 = time.time()
                if tts_engine == "qwen3":
                    has_ref_text = bool(ref_text and str(ref_text).strip())
                    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=is_cuda):
                        wavs, model_sr = model.generate_voice_clone(
                            text=text,
                            language=gen_kwargs.get("language", getattr(config, "QWEN_LANGUAGE", "English")),
                            voice_clone_prompt=voice_clone_prompt,
                            ref_audio=None if voice_clone_prompt is not None else ref_audio,
                            ref_text=None if voice_clone_prompt is not None else (str(ref_text).strip() if has_ref_text else None),
                            x_vector_only_mode=not has_ref_text,
                            temperature=gen_kwargs.get("temperature", 0.72),
                            top_p=gen_kwargs.get("top_p", 0.95),
                            repetition_penalty=gen_kwargs.get("repetition_penalty", 1.25),
                            max_new_tokens=gen_kwargs.get("max_new_tokens", getattr(config, "MAX_NEW_TOKENS", 1000)),
                        )
                    wav_np = np.asarray(wavs[0], dtype=np.float32).squeeze()
                else:
                    with torch.inference_mode(), torch.amp.autocast(
                        "cuda",
                        dtype=torch.float16,
                        enabled=(is_cuda and getattr(config, "CHATTERBOX_FP16", True)),
                    ):
                        if model_type == "turbo":
                            wav = model.generate(
                                text,
                                audio_prompt_path=None,
                                exaggeration=gen_kwargs.get("exaggeration", 0.5),
                            )
                        else:
                            wav = model.generate(
                                text,
                                audio_prompt_path=None,
                                exaggeration=gen_kwargs.get("exaggeration", 0.5),
                                cfg_weight=gen_kwargs.get("cfg_weight", 0.5),
                                temperature=gen_kwargs.get("temperature", 0.8),
                                top_p=gen_kwargs.get("top_p", 1.0),
                                repetition_penalty=gen_kwargs.get("repetition_penalty", 1.2),
                            )
                    model_sr = model.sr
                    wav_np = wav.squeeze().detach().cpu().numpy()

                t_gen = time.time() - t0
                wav_np = _clean_and_normalize_chunk_wav(wav_np, model_sr)

                # Ensure 16-bit PCM WAV (write cache atomically so concurrent workers never read partial files)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                chunk_file.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(chunk_file), wav_np, model_sr, subtype="PCM_16")
                tmp_cache = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
                shutil.copy2(chunk_file, tmp_cache)
                os.replace(tmp_cache, cache_path)

                if is_cuda and torch.cuda.memory_reserved(device) > 4700 * 1024 * 1024:
                    torch.cuda.empty_cache()

                num_words = len(text.split())
                dur = len(wav_np) / model_sr
                result_queue.put((chunk_idx, "synthesized", num_words, dur, device))

            except Exception as e:
                result_queue.put((chunk_idx, "error", str(e), 0.0, device))
    finally:
        # Explicitly unload model and release GPU memory
        try:
            del voice_clone_prompt
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


class ChatterboxAudiobookGenerator:
    """Text -> Audiobook synthesizer supporting Qwen3-TTS (default) and Chatterbox TTS (toggle)."""

    _SUPPORTED_AUDIO_FORMATS = ("m4b", "mp3", "wav", "m4a", "flac")

    def __init__(
        self,
        reference_id: Optional[str] = None,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        model_type: Optional[str] = None,
        devices: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
        no_cache: bool = False,
        tts_engine: Optional[str] = None,
        workers_per_gpu: Optional[int] = None,
    ):
        self.tts_engine = config.normalize_tts_engine(tts_engine)
        self.reference_id = reference_id or config.DEFAULT_REFERENCE_ID
        self.reference_audio = reference_audio or config.DEFAULT_REFERENCE_AUDIO
        self.reference_text = reference_text if reference_text is not None else getattr(config, "DEFAULT_REFERENCE_TEXT", None)
        self.model_type = model_type or config.MODEL_TYPE
        self.qwen_model_id = config.QWEN_MODEL_MAP.get(
            str(self.model_type).lower(),
            str(self.model_type) if "/" in str(self.model_type) else config.QWEN_DEFAULT_MODEL,
        )
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.top_p = top_p if top_p is not None else config.TOP_P
        self.repetition_penalty = repetition_penalty if repetition_penalty is not None else config.REPETITION_PENALTY
        self.cfg_weight = cfg_weight if cfg_weight is not None else config.CFG_WEIGHT
        self.exaggeration = exaggeration if exaggeration is not None else config.EXAGGERATION
        self.no_cache = no_cache
        self.workers_per_gpu = max(1, int(workers_per_gpu if workers_per_gpu is not None else getattr(config, "WORKERS_PER_GPU", 3)))

        # Setup devices (expand unique GPUs by workers_per_gpu when appropriate)
        if devices:
            base_devs = list(devices)
            if workers_per_gpu is not None and len(base_devs) == len(set(base_devs)):
                self.devices = base_devs * self.workers_per_gpu
            else:
                self.devices = base_devs
        elif getattr(config, "GPU_DEVICES", None) and workers_per_gpu is not None:
            self.devices = list(config.GPU_DEVICES) * self.workers_per_gpu
        elif config.DEVICES:
            self.devices = config.DEVICES if isinstance(config.DEVICES, list) else [config.DEVICES]
        else:
            if torch.cuda.is_available():
                self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] * self.workers_per_gpu
            else:
                self.devices = ["cpu"]

        # Resolve reference audio & companion transcript (.lab / .txt)
        ref_dir: Optional[Path] = None
        if not self.reference_audio and self.reference_id:
            ref_dir = Path(config.REFERENCES_FOLDER) / self.reference_id
            if not ref_dir.exists():
                ref_dir = SCRIPT_DIR / config.REFERENCES_FOLDER / self.reference_id
            wavs = sorted(ref_dir.glob("*.wav"))
            if wavs:
                self.reference_audio = str(wavs[0].resolve())

        if self.reference_text and Path(str(self.reference_text)).is_file():
            self.reference_text = self._read_text_file(Path(str(self.reference_text))).strip()
        elif not self.reference_text and self.reference_audio:
            wav_path = Path(self.reference_audio)
            transcript_candidates = [
                wav_path.with_suffix(".lab"),
                wav_path.with_suffix(".txt"),
            ]
            if ref_dir and ref_dir.is_dir():
                transcript_candidates.extend(sorted(ref_dir.glob("*.lab")))
                transcript_candidates.extend(sorted(ref_dir.glob("*.txt")))
            for cand in transcript_candidates:
                if cand.is_file():
                    txt_val = self._read_text_file(cand).strip()
                    if txt_val:
                        self.reference_text = txt_val
                        break

        self.logger = self._setup_logging()

    def unload_model(self):
        """Explicitly purge model references, force garbage collection, and release CUDA memory cache."""
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("ChatterboxAudiobook")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            log_dir = Path(config.LOGS_FOLDER)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(log_dir / f"chatterbox_{timestamp}.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)
        return logger

    # ------------------------------------------------------------------ #
    # Text Ingestion & Parsing
    # ------------------------------------------------------------------ #

    def _read_text_file(self, path: Path) -> str:
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                return path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return path.read_text(encoding="utf-8", errors="replace")

    def _page_label(self, path: Path) -> str:
        m = re.search(r"page_(\d+)", path.stem, re.IGNORECASE)
        if m:
            return f"Page {int(m.group(1))}"
        return path.stem.replace("_", " ").title()

    def load_input_pages(self, input_path: Path) -> List[Tuple[str, str, str]]:
        """Load source files as a list of (chapter_label, text, stem) tuples."""
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if input_path.is_file():
            text = self._read_text_file(input_path)
            # Check for Markdown headings
            heading_matches = list(re.finditer(r"^(#{1,2})\s+(.+)$", text, re.MULTILINE))
            if heading_matches:
                pages = []
                for i, match in enumerate(heading_matches):
                    title = match.group(2).strip()
                    start = match.start()
                    end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
                    section_text = text[start:end].strip()
                    pages.append((title, section_text, f"section_{i+1:03d}"))
                return [(l, t, s) for l, t, s in pages if t.strip()]
            return [(input_path.stem.title(), text, input_path.stem)]

        if input_path.is_dir():
            files = [
                f for f in sorted(input_path.iterdir())
                if f.is_file() and f.suffix.lower() in config.PAGE_FILE_EXTENSIONS
            ]
            if not files:
                # Check 1 level of subdirectories
                for sub in sorted(input_path.iterdir()):
                    if sub.is_dir():
                        sub_files = [
                            f for f in sorted(sub.iterdir())
                            if f.is_file() and f.suffix.lower() in config.PAGE_FILE_EXTENSIONS
                        ]
                        if sub_files:
                            files = sub_files
                            break

            if not files:
                raise ValueError(f"No {config.PAGE_FILE_EXTENSIONS} files found in {input_path}")

            pages = [(self._page_label(f), self._read_text_file(f), f.stem) for f in files]
            # Keep non-empty pages
            pages = [(label, text, stem) for label, text, stem in pages if len(text.strip()) > 1]
            if not pages:
                raise ValueError(f"No text extracted from files in {input_path}")
            return pages

        raise FileNotFoundError(input_path)

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into sentence-first chunks respecting word count limits (ADHD-optimized)."""
        if not text.strip():
            return []

        # Split on sentence terminals
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""
        current_words = 0
        limit = config.CHUNK_SIZE_WORDS
        min_standalone_words = 6

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_words = len(sentence.split())

            if sentence_words > limit:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_words = 0
                parts = [p.strip() for p in re.split(r"(?<=[,;:\u2014\u2013])\s+|\n+", sentence) if p.strip()]
                for part in parts:
                    part_words = len(part.split())
                    if part_words > limit:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                            current_words = 0
                        words = part.split()
                        for i in range(0, len(words), limit):
                            chunks.append(" ".join(words[i : i + limit]))
                    elif current_words + part_words <= limit:
                        current_chunk += part + " "
                        current_words += part_words
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                        current_words = part_words
                if current_chunk and current_words >= min_standalone_words:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_words = 0
            else:
                # Only merge with previous chunk if previous chunk is a tiny fragment (< 6 words)
                # or this sentence is a tiny fragment (< 6 words) and fits within limit.
                if current_chunk and (current_words < min_standalone_words or sentence_words < min_standalone_words) and (current_words + sentence_words <= limit):
                    current_chunk += sentence + " "
                    current_words += sentence_words
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_words = sentence_words

                if current_words >= min_standalone_words:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_words = 0

        if current_chunk.strip():
            if chunks and current_words < min_standalone_words and (len(chunks[-1].split()) + current_words <= limit + 4):
                chunks[-1] = f"{chunks[-1]} {current_chunk.strip()}"
            else:
                chunks.append(current_chunk.strip())

        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------ #
    # Caching
    # ------------------------------------------------------------------ #

    def get_cache_path(self, text: str) -> Path:
        if not hasattr(self, "_cached_ref_str"):
            ref_str = ""
            if self.reference_audio and Path(self.reference_audio).exists():
                ref_str = hashlib.md5(Path(self.reference_audio).read_bytes()).hexdigest()
            elif self.reference_id:
                ref_str = self.reference_id
            self._cached_ref_str = ref_str
        if not hasattr(self, "_cached_ref_txt_hash"):
            self._cached_ref_txt_hash = (
                hashlib.md5(str(self.reference_text).strip().encode("utf-8")).hexdigest()[:12]
                if self.reference_text
                else "xvec"
            )

        if self.tts_engine == "chatterbox":
            key = (
                f"chatterbox_{self.model_type}_{text}_{self._cached_ref_str}_{self.temperature}_"
                f"{self.top_p}_{self.repetition_penalty}_{self.cfg_weight}_{self.exaggeration}"
            )
        else:
            key = (
                f"qwen3_{self.qwen_model_id}_{text}_{self._cached_ref_str}_{self._cached_ref_txt_hash}_"
                f"{self.temperature}_{self.top_p}_{self.repetition_penalty}"
            )
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        cache_dir = Path(config.CACHE_FOLDER)
        if not cache_dir.is_absolute() and not cache_dir.exists():
            cache_dir = SCRIPT_DIR / config.CACHE_FOLDER
        return cache_dir / f"{digest}.wav"

    # ------------------------------------------------------------------ #
    # FFmpeg Audio Assembly & Chapters
    # ------------------------------------------------------------------ #

    @staticmethod
    def _escape_ffmeta(val: str) -> str:
        """Escape special characters (=, ;, #, \\, newline) for FFMETADATA1 format."""
        s = str(val).replace("\\", "\\\\")
        s = s.replace("=", "\\=").replace(";", "\\;").replace("#", "\\#")
        s = s.replace("\n", " ")
        return s

    def _write_ffmetadata(
        self,
        meta_path: Path,
        chapters: List[Tuple[str, int, int]],
        title: str,
        artist: str = "Chatterbox TTS",
    ):
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(";FFMETADATA1\n")
            f.write(f"title={self._escape_ffmeta(title)}\n")
            f.write(f"artist={self._escape_ffmeta(artist)}\n")
            f.write(f"album={self._escape_ffmeta(title)}\n\n")
            for chap_title, start_ms, end_ms in chapters:
                f.write("[CHAPTER]\n")
                f.write("TIMEBASE=1/1000\n")
                f.write(f"START={start_ms}\n")
                f.write(f"END={end_ms}\n")
                f.write(f"title={self._escape_ffmeta(chap_title)}\n\n")

    def _export_from_page_tracks(
        self,
        page_timings: List[PageOverlayTiming],
        output_path: Path,
        book_title: str,
        chunks_dir: Path,
        author: str = "Chatterbox TTS",
        cover_path: Optional[Path] = None,
    ) -> bool:
        """
        Reuse the generated per-page CBR MP3 audio tracks and exact cumulative durations
        from Step 3 to build the FFMETADATAFILE and mux the standalone .m4b (or .mp3/.m4a)
        without synthesizing or encoding the chunk audio twice.
        """
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to mux audiobook output but was not found on PATH")

        valid_timings = [t for t in page_timings if t.mp3_path and t.mp3_path.is_file()]
        if not valid_timings:
            raise RuntimeError("No per-page audio tracks found to mux audiobook")

        chapters: List[Tuple[str, int, int]] = [
            (t.chapter_title, t.start_ms, t.end_ms) for t in valid_timings
        ]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_meta = chunks_dir / "_chapters.txt"
        tmp_concat = chunks_dir / "_pages_concat.txt"

        fmt = output_path.suffix.lstrip(".").lower()
        if fmt not in self._SUPPORTED_AUDIO_FORMATS:
            fmt = config.AUDIO_FORMAT.lower()

        try:
            self._write_ffmetadata(tmp_meta, chapters, book_title, artist=author)
            with open(tmp_concat, "w", encoding="utf-8") as cf:
                for t in valid_timings:
                    escaped_path = str(t.mp3_path.resolve()).replace("'", "'\\''")
                    cf.write(f"file '{escaped_path}'\n")

            has_cover = bool(cover_path and Path(cover_path).is_file())
            if fmt == "m4b":
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(tmp_concat),
                    "-f", "ffmetadata",
                    "-i", str(tmp_meta),
                ]
                if has_cover:
                    cmd.extend(["-i", str(cover_path)])
                cmd.extend([
                    "-map", "0:a",
                    "-map_metadata", "1",
                ])
                if has_cover:
                    cmd.extend(["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"])
                cmd.extend([
                    "-c:a", "aac",
                    "-b:a", config.AUDIO_BITRATE,
                    "-movflags", "+faststart",
                    "-metadata", f"title={book_title}",
                    "-metadata", f"artist={author}",
                    str(output_path),
                ])
            elif fmt == "mp3":
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(tmp_concat),
                    "-f", "ffmetadata",
                    "-i", str(tmp_meta),
                    "-map", "0:a",
                    "-map_metadata", "1",
                    "-c:a", "copy",
                    "-metadata", f"title={book_title}",
                    "-metadata", f"artist={author}",
                    str(output_path),
                ]
            else:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(tmp_concat),
                    "-f", "ffmetadata",
                    "-i", str(tmp_meta),
                    "-map", "0:a",
                    "-map_metadata", "1",
                ]
                if fmt == "m4a":
                    cmd.extend(["-c:a", "aac", "-b:a", config.AUDIO_BITRATE])
                cmd.append(str(output_path))

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed to mux {fmt} from page tracks: {result.stderr[-2000:]}")
            self.logger.info(
                f"Audiobook muxed from {len(valid_timings)} page tracks: {output_path} ({len(chapters)} chapters)"
            )
            return True
        finally:
            tmp_meta.unlink(missing_ok=True)
            tmp_concat.unlink(missing_ok=True)

    def _export_m4b(
        self,
        tmp_raw: Path,
        output_path: Path,
        chapters: List[Tuple[str, int, int]],
        book_title: str,
        chunks_dir: Path,
        sample_width: int,
        frame_rate: int,
        channels: int,
    ):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to produce .m4b output but was not found on PATH")

        tmp_meta = chunks_dir / "_chapters.txt"
        try:
            self._write_ffmetadata(tmp_meta, chapters, book_title)
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "s16le",
                "-ar", str(frame_rate),
                "-ac", str(channels),
                "-i", str(tmp_raw),
                "-f", "ffmetadata",
                "-i", str(tmp_meta),
                "-map", "0:a",
                "-map_metadata", "1",
                "-c:a", "aac",
                "-b:a", config.AUDIO_BITRATE,
                "-movflags", "+faststart",
                "-metadata", f"title={book_title}",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed to build m4b: {result.stderr[-2000:]}")
        finally:
            tmp_meta.unlink(missing_ok=True)

    def _export_via_ffmpeg(
        self,
        tmp_raw: Path,
        output_path: Path,
        fmt: str,
        sample_width: int,
        frame_rate: int,
        channels: int,
        bitrate: Optional[str] = None,
    ):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for audio export")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "s16le",
            "-ar", str(frame_rate),
            "-ac", str(channels),
            "-i", str(tmp_raw),
        ]
        if bitrate:
            cmd.extend(["-b:a", bitrate])
        cmd.append(str(output_path))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to export {fmt}: {result.stderr[-2000:]}")

    def combine_chunks(
        self,
        total_chunks: int,
        chunk_page_index: List[int],
        page_labels: List[str],
        output_path: Path,
        book_title: str,
        chunks_dir: Path,
    ) -> bool:
        """Combine all individual chunk WAVs into a master file with chapters."""
        self.logger.info(f"Combining {total_chunks} chunks into {output_path}...")
        tmp_raw = output_path.with_suffix(".raw.tmp")
        try:
            total_frames = 0
            successful = 0
            missing = []
            chapters: List[Tuple[str, int, int]] = []
            current_page_idx = None
            page_start_ms = 0

            channels = 1
            sample_width = 2
            frame_rate = config.SAMPLE_RATE
            gap_frames = int(frame_rate * config.CHUNK_GAP_MS / 1000) if config.CHUNK_GAP_MS > 0 else 0
            gap_bytes = b"\x00" * (gap_frames * channels * sample_width) if gap_frames > 0 else b""

            chunks_dir.mkdir(parents=True, exist_ok=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_raw, "wb") as raw_out:
                for i in range(1, total_chunks + 1):
                    chunk_file = chunks_dir / f"chunk_{i:04d}.wav"
                    if not chunk_file.exists():
                        missing.append(i)
                        continue

                    try:
                        with wave.open(str(chunk_file), "rb") as wf:
                            channels = wf.getnchannels()
                            sample_width = wf.getsampwidth()
                            frame_rate = wf.getframerate()
                            nframes = wf.getnframes()
                            chunk_raw = wf.readframes(nframes)
                    except Exception as e:
                        self.logger.warning(f"Failed to read chunk {i}: {e}")
                        missing.append(i)
                        continue

                    current_ms = int(total_frames * 1000 / frame_rate)
                    page_idx = chunk_page_index[i - 1]
                    starting_new_page = page_idx != current_page_idx
                    if starting_new_page and current_page_idx is not None:
                        chapters.append((page_labels[current_page_idx], page_start_ms, current_ms))

                    if successful > 0 and gap_bytes:
                        raw_out.write(gap_bytes)
                        total_frames += gap_frames

                    if starting_new_page:
                        page_start_ms = int(total_frames * 1000 / frame_rate)
                        current_page_idx = page_idx

                    raw_out.write(chunk_raw)
                    total_frames += nframes
                    successful += 1

                if current_page_idx is not None:
                    final_ms = int(total_frames * 1000 / frame_rate)
                    chapters.append((page_labels[current_page_idx], page_start_ms, final_ms))

            if successful == 0:
                raise RuntimeError("No valid chunks were found to assemble")

            if missing:
                self.logger.warning(f"Missing {len(missing)} chunk(s): {missing[:10]}...")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            fmt = output_path.suffix.lstrip(".").lower()
            if fmt not in self._SUPPORTED_AUDIO_FORMATS:
                fmt = config.AUDIO_FORMAT.lower()

            if fmt == "m4b":
                self._export_m4b(
                    tmp_raw, output_path, chapters, book_title, chunks_dir,
                    sample_width=sample_width, frame_rate=frame_rate, channels=channels
                )
            else:
                bitrate = config.AUDIO_BITRATE if fmt in ("mp3", "m4a") else None
                self._export_via_ffmpeg(
                    tmp_raw, output_path, fmt,
                    sample_width=sample_width, frame_rate=frame_rate, channels=channels, bitrate=bitrate
                )

            self.logger.info(f"Audiobook saved: {output_path} ({successful}/{total_chunks} chunks, {len(chapters)} chapters)")
            return True
        finally:
            tmp_raw.unlink(missing_ok=True)

    def export_per_page(
        self,
        total_chunks: int,
        chunk_page_index: List[int],
        page_labels: List[str],
        page_stems: List[str],
        output_dir: Path,
        fmt: str,
        chunks_dir: Path,
    ) -> Tuple[int, List[str]]:
        """Export individual audio files per page."""
        output_dir.mkdir(parents=True, exist_ok=True)
        if fmt == "m4b":
            fmt = "mp3"

        sample_width = 2
        channels = 1
        frame_rate = config.SAMPLE_RATE
        gap_frames = int(frame_rate * config.CHUNK_GAP_MS / 1000) if config.CHUNK_GAP_MS > 0 else 0
        gap_bytes = b"\x00" * (gap_frames * channels * sample_width) if gap_frames > 0 else b""

        chunks_by_page: dict = {}
        for i in range(1, total_chunks + 1):
            chunks_by_page.setdefault(chunk_page_index[i - 1], []).append(i)

        export_kwargs = {"format": fmt}
        if fmt in ("mp3", "m4a"):
            export_kwargs["bitrate"] = config.AUDIO_BITRATE

        successful_pages = 0
        failed_pages = []

        for page_idx in sorted(chunks_by_page):
            raw_parts = []
            added = 0
            for i in chunks_by_page[page_idx]:
                chunk_file = chunks_dir / f"chunk_{i:04d}.wav"
                if not chunk_file.exists():
                    continue
                try:
                    with wave.open(str(chunk_file), "rb") as wf:
                        channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                        frame_rate = wf.getframerate()
                        chunk_raw = wf.readframes(wf.getnframes())
                except Exception:
                    continue

                if added > 0 and gap_bytes:
                    raw_parts.append(gap_bytes)
                raw_parts.append(chunk_raw)
                added += 1

            if added == 0:
                failed_pages.append(page_labels[page_idx])
                continue

            out_path = output_dir / f"{page_stems[page_idx]}.{fmt}"
            combined = AudioSegment(
                data=b"".join(raw_parts),
                sample_width=sample_width,
                frame_rate=frame_rate,
                channels=channels,
            )
            combined.export(str(out_path), **export_kwargs)
            successful_pages += 1

        return successful_pages, failed_pages

    # ------------------------------------------------------------------ #
    # High-Level Conversion Pipeline (EPUB3 + SMIL Media Overlays + M4B)
    # ------------------------------------------------------------------ #

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        per_page: bool = False,
        clean_chunks: bool = False,
        keep_chunks: Optional[bool] = None,
        epub_output_path: Optional[Path] = None,
        markdown_dir: Optional[Path] = None,
        toc_file: Optional[Path] = None,
        title: Optional[str] = None,
        author: Optional[str] = None,
        skip_epub: bool = False,
    ) -> bool:
        if keep_chunks is not None:
            clean_chunks = not keep_chunks
        start_time = time.time()
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        folder_label = input_path.stem if input_path.is_file() else input_path.name
        book_name = canonical_book_name(input_path)
        book_title = title or book_name
        engine_label = "Qwen3-TTS" if self.tts_engine == "qwen3" else "Chatterbox TTS"
        book_author = author or engine_label
        chunks_dir = Path(config.CHUNKS_FOLDER) / folder_label
        chunks_dir.mkdir(parents=True, exist_ok=True)

        if not skip_epub and epub_output_path is None:
            epub_output_path = output_path.with_suffix(".epub")
        elif epub_output_path is not None:
            epub_output_path = Path(epub_output_path).resolve()

        # Discover rich MinerU Markdown directory if available for visual HTML/MathML/images
        resolved_md_dir = Path(markdown_dir).resolve() if markdown_dir else find_companion_markdown_dir(input_path)
        dom_source_path = resolved_md_dir if (resolved_md_dir and resolved_md_dir.exists()) else input_path

        # Load Stage 1 book_metadata.json & cover.jpg (with automatic fallback extraction for backlog items)
        book_meta = load_or_extract_book_metadata(input_path, md_dir=resolved_md_dir)
        explicit_title = title if (title and title != book_name) else None
        explicit_author = author if (author and author not in ("Qwen3-TTS", "Chatterbox TTS", "MinerU")) else None
        if explicit_title:
            book_title = explicit_title
        elif book_meta.get("title"):
            book_title = book_meta["title"]

        if explicit_author:
            book_author = explicit_author
        elif book_meta.get("authors"):
            book_author = ", ".join(book_meta["authors"])

        book_publisher = book_meta.get("publisher")
        cover_path: Optional[Path] = (
            Path(book_meta["coverPath"]).resolve()
            if book_meta.get("coverPath") and Path(book_meta["coverPath"]).is_file()
            else None
        )

        active_model_display = (
            f"{self.qwen_model_id} (FP16)"
            if self.tts_engine == "qwen3"
            else f"Chatterbox {self.model_type.upper()} (FP16)"
        )
        console.print(
            Panel(
                f"[bold cyan]{engine_label} Audiobook & Synchronized EPUB3 Generator[/bold cyan]\n\n"
                f"• Input        : [yellow]{input_path}[/yellow]\n"
                f"• DOM Source   : [yellow]{dom_source_path}[/yellow]\n"
                f"• Audio Output : [green]{output_path}[/green]\n"
                f"• EPUB3 Output : [green]{epub_output_path if not skip_epub else 'Skipped (--skip-epub)'}[/green]\n"
                f"• TTS Engine   : [bold magenta]{self.tts_engine.upper()}[/bold magenta]\n"
                f"• Model        : [magenta]{active_model_display}[/magenta]\n"
                f"• Devices      : [cyan]{', '.join(self.devices)}[/cyan]\n"
                f"• Reference    : [blue]{self.reference_audio or f'Built-in {engine_label} Timbre'}[/blue]\n"
                f"• Chunk Size   : {config.CHUNK_SIZE_WORDS} words\n"
                f"• Sample Rate  : {config.SAMPLE_RATE} Hz",
                title="[bold blue]Conversion Job[/bold blue]",
                border_style="blue",
            )
        )

        # 1. Extract TOC mapping and prepare DOM-wrapped XHTML + SpeechChunk list per page
        with console.status("[bold green]Extracting TOC mapping and wrapping page DOM spans..."):
            toc_map = extract_toc_mapping(
                input_path,
                md_dir=resolved_md_dir,
                toc_file=Path(toc_file) if toc_file else None,
            )
            doc_files = gather_page_documents(dom_source_path)
            if not doc_files and dom_source_path != input_path:
                doc_files = gather_page_documents(input_path)
            if _sdisp is not None:
                _sdisp.start_stage("dom_prep", total_units=len(doc_files), unit_name="pages")

            if not explicit_title and not book_meta.get("title") and doc_files:
                book_title = extract_title_from_markdown(read_file_text(doc_files[0]), book_name)

            image_registry: dict = {}
            page_doms: List[PageDOMResult] = []
            page_stems: List[str] = []
            md_opts = _default_md_tts_opts()
            md_stats = {"equations": 0, "converted": 0, "errors": 0}

            for idx, doc_file in enumerate(doc_files, 1):
                pnum = extract_page_number(doc_file, idx)
                chap_title = format_page_chapter_title(pnum, toc_map)
                raw_text = read_file_text(doc_file)
                is_plain = doc_file.suffix.lower() in (".txt", ".text")
                pdom = prepare_page_dom_and_chunks(
                    page_num=pnum,
                    chapter_title=chap_title,
                    raw_markdown_or_text=raw_text,
                    base_dir=doc_file.parent,
                    image_registry=image_registry,
                    is_plain_text=is_plain,
                    language="en",
                    chunk_size_words=config.CHUNK_SIZE_WORDS,
                    md_opts=md_opts,
                    md_stats=md_stats,
                )
                page_doms.append(pdom)
                page_stems.append(doc_file.stem)
                if _sdisp is not None:
                    _sdisp.advance(1)
            if _sdisp is not None:
                _sdisp.end_stage("done")

        page_labels = [p.chapter_title for p in page_doms]
        chunks: List[str] = []
        chunk_page_index: List[int] = []
        for p_idx, pdom in enumerate(page_doms):
            for sc in pdom.chunks:
                chunks.append(sc.spoken_text)
                chunk_page_index.append(p_idx)

        total_words = sum(len(c.split()) for c in chunks)
        total_chunks = len(chunks)
        toc_matched_count = sum(1 for p in page_doms if p.page_num in toc_map)
        console.print(
            f"Prepared [bold]{len(page_doms)}[/bold] page(s) ([cyan]{toc_matched_count}[/cyan] TOC-matched chapters), "
            f"[bold]{total_words:,}[/bold] spoken words, wrapped into [bold cyan]{total_chunks}[/bold cyan] DOM span chunk(s)."
        )

        # 2. Check existing chunks / cache
        # Always verify against the voice/parameter content-addressed cache (`cache_path`)
        # rather than blindly trusting positional `chunk_{i:04d}.wav` files from prior runs
        # that may have used a different reference voice or chunking boundary.
        tasks_to_run = []
        cached_count = 0
        stale_removed = 0
        for i, chunk_text in enumerate(chunks, 1):
            chunk_file = chunks_dir / f"chunk_{i:04d}.wav"
            cache_path = self.get_cache_path(chunk_text)

            if not self.no_cache and cache_path.exists() and cache_path.stat().st_size > 44:
                if not chunk_file.exists() or chunk_file.stat().st_size != cache_path.stat().st_size:
                    shutil.copy2(cache_path, chunk_file)
                cached_count += 1
            else:
                if chunk_file.exists():
                    chunk_file.unlink(missing_ok=True)
                    stale_removed += 1
                tasks_to_run.append((i, chunk_text, str(cache_path), str(chunk_file)))

        # Remove any leftover trailing chunk files from prior runs with higher chunk counts
        for extra_wav in chunks_dir.glob("chunk_*.wav"):
            m_idx = re.match(r"^chunk_(\d+)\.wav$", extra_wav.name)
            if m_idx and int(m_idx.group(1)) > total_chunks:
                extra_wav.unlink(missing_ok=True)
                stale_removed += 1

        if stale_removed > 0:
            console.print(
                f"Removed [bold yellow]{stale_removed}[/bold yellow] stale positional chunk file(s) from prior voice/chunking config."
            )
        if cached_count > 0:
            console.print(f"Found [bold green]{cached_count}[/bold green] cache-verified chunks already synthesized (reused instantly).")
        if _sdisp is not None:
            _sdisp.start_stage("tts_synth", total_units=total_chunks, unit_name="chunks", initial_completed=cached_count)

        # 3. Multi-GPU / Multi-Worker Parallel Synthesis
        gen_kwargs = {
            "tts_engine": self.tts_engine,
            "qwen_model_id": self.qwen_model_id,
            "ref_text": self.reference_text,
            "no_cache": self.no_cache,
            "workers_per_gpu": self.workers_per_gpu,
            "language": getattr(config, "QWEN_LANGUAGE", "English"),
            "max_new_tokens": getattr(config, "MAX_NEW_TOKENS", 1000),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "cfg_weight": self.cfg_weight,
            "exaggeration": self.exaggeration,
        }

        if tasks_to_run:
            ctx = mp.get_context("spawn")
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()

            workers = []
            try:
                for d in self.devices:
                    p = ctx.Process(
                        target=_worker_process_fn,
                        args=(d, self.model_type, self.reference_audio, gen_kwargs, task_queue, result_queue),
                    )
                    p.daemon = True
                    p.start()
                    workers.append(p)

                # Wait for all workers to be ready
                ready_workers = 0
                while ready_workers < len(workers):
                    idx, status, msg = result_queue.get()
                    if status == "ready":
                        ready_workers += 1
                        console.print(f"Worker on [cyan]{msg}[/cyan] loaded and ready.")
                    elif status == "init_error":
                        console.print(f"[bold red]{msg}[/bold red]")
                        raise RuntimeError(msg)

                # Put all tasks into queue
                for task in tasks_to_run:
                    task_queue.put(task)

                # Put poison pills
                for _ in workers:
                    task_queue.put(None)

                # Process results with Rich progress bar
                with Progress(*_progress_columns(), console=console, disable=bool(os.environ.get("STATUS_DISPLAY_ACTIVE"))) as progress:
                    synth_task = progress.add_task(
                        "Synthesizing chunks...",
                        total=total_chunks,
                        completed=cached_count,
                        speed="0.0 chk/s",
                    )

                    completed = cached_count
                    total_synth_words = 0
                    synth_start_time = time.time()

                    while completed < total_chunks:
                        chunk_idx, status, words_or_err, dur, dev = result_queue.get()
                        if status in ("synthesized", "cached"):
                            completed += 1
                            total_synth_words += words_or_err
                            elapsed = time.time() - synth_start_time
                            chk_speed = (completed - cached_count) / max(elapsed, 0.001)
                            progress.update(
                                synth_task,
                                completed=completed,
                                speed=f"{chk_speed:.2f} chk/s",
                            )
                            if _sdisp is not None:
                                _sdisp.advance(1, audio_sec=float(dur or 0.0), words=int(words_or_err or 0), cached=(status == "cached"))
                        elif status == "error":
                            self.logger.error(f"Chunk {chunk_idx} failed on {dev}: {words_or_err}")
                            completed += 1
                            progress.update(synth_task, completed=completed)
                            if _sdisp is not None:
                                _sdisp.advance(1, retry=True)
            finally:
                # Terminate worker processes and purge GPU memory
                console.print("[cyan]Unloading model and releasing GPU memory...[/cyan]")
                for p in workers:
                    try:
                        p.join(timeout=3)
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=2)
                        if p.is_alive():
                            p.kill()
                            p.join()
                    except Exception:
                        pass
                workers.clear()
                self.unload_model()
                console.print(f"[bold green]✓ {engine_label} model unloaded from memory and GPU VRAM released.[/bold green]")

        # 4. Sample-Accurate Per-Page Audio Assembly (CBR MP3) & SMIL 3.0 Media Overlays
        if _sdisp is not None:
            _sdisp.end_stage("done")
            _sdisp.start_stage("smil_assembly", total_units=len(page_doms), unit_name="pages")
        console.print("[bold yellow]Computing sample-accurate SMIL timings and encoding per-page CBR MP3 tracks...[/bold yellow]")
        page_audio_dir = chunks_dir / "page_audio"
        page_audio_dir.mkdir(parents=True, exist_ok=True)

        page_timings_map: dict = {}
        page_timings_list: List[PageOverlayTiming] = []
        global_chunk_idx = 1
        book_clock = 0.0

        for pdom in page_doms:
            c_wavs = [
                chunks_dir / f"chunk_{global_chunk_idx + j:04d}.wav"
                for j in range(len(pdom.chunks))
            ]
            global_chunk_idx += len(pdom.chunks)
            mp3_path = page_audio_dir / f"page_{pdom.num_str}.mp3"
            timing = assemble_page_audio_and_smil(
                page_dom=pdom,
                chunk_wav_paths=c_wavs,
                out_mp3_path=mp3_path,
                sample_rate=config.SAMPLE_RATE,
                chunk_gap_ms=config.CHUNK_GAP_MS,
                book_start_sec=book_clock,
                mp3_bitrate=config.AUDIO_BITRATE,
            )
            page_timings_map[pdom.page_num] = timing
            page_timings_list.append(timing)
            book_clock += timing.duration_sec
            if _sdisp is not None:
                _sdisp.advance(1)
        if _sdisp is not None:
            _sdisp.end_stage("done")
            _sdisp.start_stage("epub_m4b_mux", total_units=3, unit_name="steps")

        # 5. Package Synchronized EPUB3 with SMIL Media Overlays
        if not skip_epub and epub_output_path is not None:
            console.print(f"[bold yellow]Packaging synchronized EPUB3 with Media Overlays: {epub_output_path}...[/bold yellow]")
            package_epub3(
                output_epub_path=epub_output_path,
                title=book_title,
                author=book_author,
                language="en",
                identifier=f"urn:uuid:{uuid.uuid4()}",
                publisher=book_publisher,
                css_content=DEFAULT_CSS,
                page_doms=page_doms,
                page_timings=page_timings_map,
                image_registry=image_registry,
                cover_image_path=str(cover_path) if cover_path else None,
            )
            console.print(f"[bold green]✓ Synchronized EPUB3 created: {epub_output_path}[/bold green]")

        # 6. Dual M4B Output Reuse from Per-Page Audio Tracks & Cumulative Timings
        console.print("[bold yellow]Muxing standalone audiobook from per-page tracks and chapter timings...[/bold yellow]")
        success = self._export_from_page_tracks(
            page_timings=page_timings_list,
            output_path=output_path,
            book_title=book_title,
            chunks_dir=chunks_dir,
            author=book_author,
            cover_path=cover_path,
        )

        # 6b. Generate AudioBookshelf metadata.json and companion cover.jpg
        abs_meta_path = output_path.with_suffix(".metadata.json")
        abs_cover_path = output_path.with_suffix(".cover.jpg")
        try:
            voice_label = self.reference_id or (Path(self.reference_audio).stem if self.reference_audio else "default")
            narrator_list = [f"{engine_label} ({voice_label})"]
            abs_chapters = [
                {
                    "id": idx,
                    "start": round(t.start_ms / 1000.0, 3),
                    "end": round(t.end_ms / 1000.0, 3),
                    "title": t.chapter_title,
                }
                for idx, t in enumerate(page_timings_list)
            ]
            if build_abs_metadata_json is not None:
                abs_meta = build_abs_metadata_json(
                    base_metadata=book_meta,
                    chapters=abs_chapters,
                    narrators=narrator_list,
                    title_override=explicit_title,
                    author_override=explicit_author,
                )
            else:
                abs_meta = {
                    "title": book_title,
                    "subtitle": book_meta.get("subtitle"),
                    "authors": book_meta.get("authors") or ([book_author] if book_author not in ("Qwen3-TTS", "Chatterbox TTS") else []),
                    "narrators": narrator_list,
                    "series": book_meta.get("series") or [],
                    "genres": book_meta.get("genres") or [],
                    "publishedYear": book_meta.get("publishedYear"),
                    "publishedDate": None,
                    "publisher": book_publisher,
                    "description": book_meta.get("description"),
                    "isbn": book_meta.get("isbn"),
                    "asin": None,
                    "language": "en",
                    "explicit": False,
                    "abridged": False,
                    "tags": ["TTS", "MinerU", "Qwen3-TTS" if self.tts_engine == "qwen3" else "Chatterbox"],
                    "chapters": abs_chapters,
                }
            abs_meta_path.write_text(json.dumps(abs_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            console.print(f"[bold green]✓ AudioBookshelf metadata saved: {abs_meta_path}[/bold green]")
            if cover_path and cover_path.is_file():
                shutil.copy2(cover_path, abs_cover_path)
                console.print(f"[bold green]✓ Companion cover image saved: {abs_cover_path}[/bold green]")
        except Exception as meta_err:
            self.logger.warning(f"Failed writing AudioBookshelf metadata.json: {meta_err}")

        # 7. Per-page audio export if requested
        if per_page:
            per_page_dir = output_path.parent / f"{folder_label}_pages"
            per_page_dir.mkdir(parents=True, exist_ok=True)
            console.print(f"[bold yellow]Exporting per-page audio files to {per_page_dir}...[/bold yellow]")
            succ_pages = 0
            for stem, timing in zip(page_stems, page_timings_list):
                if timing.mp3_path and timing.mp3_path.is_file():
                    shutil.copy2(timing.mp3_path, per_page_dir / f"{stem}.mp3")
                    succ_pages += 1
            console.print(f"[bold green]Exported {succ_pages}/{len(page_doms)} individual page audio files.[/bold green]")

        # 8. Clean up intermediate chunks if requested
        if clean_chunks and chunks_dir.exists():
            console.print(f"[yellow]Cleaning up intermediate chunks directory: {chunks_dir}[/yellow]")
            shutil.rmtree(chunks_dir, ignore_errors=True)
        if _sdisp is not None:
            _sdisp.advance(3)
            _sdisp.end_stage("done")

        elapsed = time.time() - start_time
        console.print(
            Panel(
                f"[bold green]Conversion Complete![/bold green]\n\n"
                f"• Master Audio : [bold cyan]{output_path}[/bold cyan]\n"
                f"• EPUB3 Overlay: [bold cyan]{epub_output_path if not skip_epub else 'Skipped'}[/bold cyan]\n"
                f"• Chapters     : {len(page_labels)} pages embedded\n"
                f"• Total Chunks : {total_chunks}\n"
                f"• Total Time   : {elapsed/60:.2f} minutes",
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )
        return success


# Aliases for clarity across Qwen3-TTS and Chatterbox callers
Qwen3AudiobookGenerator = ChatterboxAudiobookGenerator
TTSAudiobookGenerator = ChatterboxAudiobookGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert documents to audiobooks & synchronized EPUB3 via Qwen3-TTS (default) or Chatterbox TTS."
    )
    parser.add_argument("input_path", nargs="?", type=Path, default=None, help="Text/Markdown file or folder of page files.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output audiobook file (.m4b, .mp3, etc.).")
    parser.add_argument("-e", "--epub-output", type=Path, default=None, help="Output synchronized EPUB3 file (.epub).")
    parser.add_argument("--markdown-dir", type=Path, default=None, help="Optional rich MinerU Markdown directory for EPUB3 DOM.")
    parser.add_argument("--toc", type=Path, default=None, help="Optional toc.json mapping page numbers to TOC titles.")
    parser.add_argument("--title", type=str, default=None, help="Book title metadata.")
    parser.add_argument("--author", type=str, default=None, help="Book author metadata.")
    parser.add_argument("--skip-epub", "--no-epub", action="store_true", help="Skip generating synchronized EPUB3.")
    parser.add_argument("--format", type=str, default=None, choices=ChatterboxAudiobookGenerator._SUPPORTED_AUDIO_FORMATS)
    parser.add_argument(
        "--tts-engine",
        "--engine",
        dest="tts_engine",
        type=str,
        default=None,
        choices=["qwen3", "chatterbox", "qwen", "qwen3-tts", "cb"],
        help="TTS engine to use: 'qwen3' (default) or 'chatterbox' (or set TTS_ENGINE env var).",
    )
    parser.add_argument("--qwen3", "--qwen", dest="tts_engine", action="store_const", const="qwen3", help="Use Qwen3-TTS engine.")
    parser.add_argument("--chatterbox", dest="tts_engine", action="store_const", const="chatterbox", help="Use Chatterbox TTS engine.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model variant: 'standard'/'1.7b' (Qwen3 1.7B or Chatterbox 500M), 'turbo'/'0.6b' (Qwen3 0.6B or Chatterbox Turbo 350M), or HF repo/path.",
    )
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated device list (e.g. 'cuda:0,cuda:1').")
    parser.add_argument("--workers-per-gpu", type=int, default=None, help="Number of worker processes per GPU (default: 3).")
    parser.add_argument("--reference-id", type=str, default=None, help="Preset reference voice ID (default: practical_engineering).")
    parser.add_argument("--reference-audio", type=str, default=None, help="Path to reference audio WAV for voice cloning.")
    parser.add_argument("--reference-text", type=str, default=None, help="Reference transcript string or .lab/.txt file for voice cloning.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--cfg-weight", type=float, default=None)
    parser.add_argument("--exaggeration", type=float, default=None)
    parser.add_argument("--per-page", action="store_true", help="Also export individual audio files per page.")
    parser.add_argument("--clean-chunks", action="store_true", help="Delete intermediate chunk WAV files after successful conversion.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass audio chunk cache.")
    parser.add_argument("--list-references", action="store_true", help="List available reference voices.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_references:
        ref_dir = Path(config.REFERENCES_FOLDER)
        if not ref_dir.exists():
            ref_dir = SCRIPT_DIR / config.REFERENCES_FOLDER
        print(f"Available reference voices in {ref_dir}:")
        if ref_dir.exists():
            for d in sorted(ref_dir.iterdir()):
                if d.is_dir():
                    wavs = list(d.glob("*.wav"))
                    labs = list(d.glob("*.lab")) + list(d.glob("*.txt"))
                    print(f"  • {d.name} ({len(wavs)} audio file(s), {len(labs)} transcript file(s))")
        sys.exit(0)

    if not args.input_path:
        print("Error: input_path is required (unless using --list-references)")
        sys.exit(1)

    devices = [d.strip() for d in args.devices.split(",")] if args.devices else None

    # Resolve output path
    input_path = args.input_path.resolve()
    fmt = (args.format or config.AUDIO_FORMAT).lower()
    book_title = input_path.stem if input_path.is_file() else input_path.name

    if args.output:
        output_path = args.output.resolve()
    else:
        output_path = (Path(config.AUDIOBOOKS_FOLDER) / f"{book_title}.{fmt}").resolve()

    generator = ChatterboxAudiobookGenerator(
        reference_id=args.reference_id,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
        model_type=args.model,
        devices=devices,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration,
        no_cache=args.no_cache,
        tts_engine=args.tts_engine,
        workers_per_gpu=args.workers_per_gpu,
    )

    try:
        success = generator.convert(
            input_path=input_path,
            output_path=output_path,
            per_page=args.per_page,
            clean_chunks=args.clean_chunks,
            epub_output_path=args.epub_output,
            markdown_dir=args.markdown_dir,
            toc_file=args.toc,
            title=args.title,
            author=args.author,
            skip_epub=args.skip_epub,
        )
    finally:
        generator.unload_model()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

