#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Automatically re-execute inside .venv if invoked with system python
_script_dir = Path(__file__).resolve().parent
_venv_python = _script_dir / ".venv" / "bin" / "python3"
if (
    __name__ == "__main__"
    and _venv_python.exists()
    and Path(sys.executable).resolve() != _venv_python.resolve()
):
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

_venv_site = _script_dir / ".venv" / "lib" / "python3.11" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

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
    sanitize_spoken_for_tts,
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
    1. Trims variable leading/trailing dead air and detached onset/offset blips below -34 dB
       relative to active speech RMS (retaining a natural 25 ms pad and 12 ms cosine fade-in/out)
       so SMIL highlights and sentence transitions never lag or hang on trailing silence.
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
            thresh = ref_rms * (10.0 ** (-34.0 / 20.0))
            active_idx = np.where(frame_rms > thresh)[0]
            if len(active_idx) > 0:
                # Filter out tiny isolated leading/trailing vocoder clicks/blips (<= 40 ms)
                # separated from the main speech body by >= 100 ms of silence
                gaps = np.diff(active_idx)
                big_gaps = np.where(gaps >= 10)[0]
                if len(big_gaps) > 0:
                    clusters = np.split(active_idx, big_gaps + 1)
                    main_clusters = [c for c in clusters if len(c) > 4]
                    if main_clusters:
                        first_idx = int(main_clusters[0][0])
                        last_idx = int(main_clusters[-1][-1])
                    else:
                        first_idx = int(active_idx[0])
                        last_idx = int(active_idx[-1])
                else:
                    first_idx = int(active_idx[0])
                    last_idx = int(active_idx[-1])

                pad_samples = int(sr * 0.025)
                start_sample = max(0, first_idx * win - pad_samples)
                end_sample = min(len(wav_np), (last_idx + 1) * win + pad_samples)
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


# =============================================================================
# CHATTERBOX-TTS-EXTENDED HELPERS (Issue #97 / petermg workflow)
# =============================================================================

import difflib
import random
import string


def derive_seed(base_seed: int, chunk_idx: int, cand_idx: int, attempt_idx: int) -> int:
    """
    Deterministically derive a 32-bit seed for each (chunk, candidate, attempt)
    from the base seed so each candidate/retry produces distinct sampling without
    mutating global state unpredictably.
    """
    if base_seed == 0:
        base_seed = 424242
    mix = (
        np.uint64(base_seed) * np.uint64(1000003)
        + np.uint64(chunk_idx) * np.uint64(10007)
        + np.uint64(cand_idx) * np.uint64(10009)
        + np.uint64(attempt_idx) * np.uint64(101)
    )
    s = int(mix & np.uint64(0xFFFFFFFF))
    return s if s != 0 else 1


def smart_append_short_sentences(
    sentences: List[str],
    min_chars: int = 20,
    max_chars: int = 300,
) -> List[str]:
    """
    To avoid audio problems with short sentences (< min_chars, default 20),
    merge any short sentence with the sentence after it, or if not possible,
    with the sentence before it.
    """
    new_groups: List[str] = []
    i = 0
    while i < len(sentences):
        current = sentences[i].strip()
        if not current:
            i += 1
            continue
        if len(current) >= min_chars:
            new_groups.append(current)
            i += 1
        else:
            appended = False
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1].strip()
                if next_sentence and len(current + " " + next_sentence) <= max_chars:
                    new_groups.append(current + " " + next_sentence)
                    i += 2
                    appended = True
            if not appended and new_groups:
                if len(new_groups[-1] + " " + current) <= max_chars:
                    new_groups[-1] += " " + current
                    i += 1
                    appended = True
            if not appended:
                new_groups.append(current)
                i += 1
    return new_groups


def enforce_min_chunk_length(
    chunks: List[str],
    min_len: int = 20,
    max_len: int = 300,
) -> List[str]:
    """
    Post-pass ensuring chunks shorter than `min_len` chars are merged with the
    following or preceding chunk when within `max_len`.
    """
    out: List[str] = []
    i = 0
    while i < len(chunks):
        current = chunks[i].strip()
        if not current:
            i += 1
            continue
        if len(current) >= min_len:
            out.append(current)
            i += 1
        else:
            if i + 1 < len(chunks):
                merged = current + " " + chunks[i + 1].strip()
                if len(merged) <= max_len:
                    out.append(merged)
                    i += 2
                    continue
            if out and len(out[-1] + " " + current) <= max_len:
                out[-1] = out[-1] + " " + current
                i += 1
            else:
                out.append(current)
                i += 1
    return out


_NUMBER_WORD_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}


def normalize_for_compare_all_punct(text: str) -> str:
    """Strip dashes, punctuation, and extra whitespace for Whisper fuzzy comparison."""
    text = re.sub(r"\[\s*pause\s*:\s*[^\]]*\]", " ", str(text), flags=re.IGNORECASE)
    text = re.sub(r"[–—-]", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def _normalize_numbers_to_words(text: str) -> str:
    tokens = text.split()
    return " ".join(_NUMBER_WORD_MAP.get(tok, tok) for tok in tokens)


def fuzzy_match_score(transcribed: str, target_text: str) -> float:
    """
    Calculate SequenceMatcher similarity ratio between Whisper transcription and target text.
    Also checks number-word equivalence (e.g. 'Chapter 1' <-> 'chapter one').
    """
    t1 = normalize_for_compare_all_punct(transcribed)
    t2 = normalize_for_compare_all_punct(target_text)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    raw_score = difflib.SequenceMatcher(None, t1, t2).ratio()
    norm_score = difflib.SequenceMatcher(
        None,
        _normalize_numbers_to_words(t1),
        _normalize_numbers_to_words(t2),
    ).ratio()
    return max(raw_score, norm_score)


def fuzzy_match(text1: str, text2: str, threshold: float = 0.85) -> bool:
    return fuzzy_match_score(text1, text2) >= threshold


def load_whisper_backend(
    model_name: str = "base",
    use_faster_whisper: bool = True,
    device: str = "cpu",
):
    """
    Load Whisper backend (faster-whisper or openai-whisper) with VRAM/compute_type fallbacks:
      CUDA: try float16 -> int8_float16 -> int8 -> float32 (for Pascal sm_60) -> fallback CPU
      CPU : try int8 -> float32
    """
    dev_type = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    dev_idx = 0
    if dev_type == "cuda" and ":" in str(device):
        parts = str(device).split(":")
        if parts[1].isdigit():
            dev_idx = int(parts[1])

    if use_faster_whisper:
        from faster_whisper import WhisperModel as FasterWhisperModel

        if dev_type == "cuda":
            candidates = [
                ("cuda", dev_idx, "float16"),
                ("cuda", dev_idx, "int8_float16"),
                ("cuda", dev_idx, "int8"),
                ("cuda", dev_idx, "float32"),
                ("cpu", 0, "int8"),
                ("cpu", 0, "float32"),
            ]
        else:
            candidates = [
                ("cpu", 0, "int8"),
                ("cpu", 0, "float32"),
            ]

        last_err = None
        for d_name, d_idx, ct in candidates:
            try:
                return FasterWhisperModel(model_name, device=d_name, device_index=d_idx, compute_type=ct)
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to load faster-whisper '{model_name}': {last_err}")
    else:
        import whisper

        try:
            return whisper.load_model(model_name, device=device if dev_type == "cuda" else "cpu")
        except Exception:
            return whisper.load_model(model_name, device="cpu")


def whisper_check_mp(
    candidate_path: str,
    target_text: str,
    whisper_model,
    use_faster_whisper: bool = True,
) -> Tuple[str, float, str]:
    """
    Transcribe `candidate_path` with Whisper and return `(candidate_path, score, transcribed)`.
    """
    try:
        if use_faster_whisper and hasattr(whisper_model, "transcribe"):
            # Inspect whether this is faster_whisper.WhisperModel or openai-whisper
            cls_mod = type(whisper_model).__module__
            if "faster_whisper" in cls_mod:
                segments, _info = whisper_model.transcribe(str(candidate_path), beam_size=1, language="en")
                transcribed = "".join(seg.text for seg in segments).strip().lower()
            else:
                result = whisper_model.transcribe(str(candidate_path), language="en")
                transcribed = result.get("text", "").strip().lower()
        else:
            result = whisper_model.transcribe(str(candidate_path), language="en")
            transcribed = result.get("text", "").strip().lower()

        score = fuzzy_match_score(transcribed, target_text)

        # Check for detached leading/trailing onset/offset artifact clusters separated by a silence gap
        # (e.g. a 150-450ms pre-speech groan/breath followed by >= 80ms of silence before the actual words).
        # If trimming at the silence gap preserves or improves the Whisper match score (>= 0.85),
        # trim the detached onset/offset artifact in-place and update the candidate's duration.
        try:
            wav_np, sr = sf.read(str(candidate_path), dtype="float32")
            if wav_np.ndim > 1:
                wav_np = wav_np.squeeze()
            win = max(1, int(sr * 0.01))  # 10ms frames
            n_win = len(wav_np) // win
            if n_win > 25:
                frames = wav_np[: n_win * win].reshape(n_win, win)
                frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
                ref_rms = float(np.percentile(frame_rms, 85))
                if ref_rms > 1e-4:
                    thresh = ref_rms * (10.0 ** (-28.0 / 20.0))
                    active_idx = np.where(frame_rms > thresh)[0]
                    if len(active_idx) > 4:
                        gaps = np.diff(active_idx)
                        big_gaps = np.where(gaps >= 8)[0]  # >= 80 ms silence gap
                        if len(big_gaps) > 0:
                            clusters = np.split(active_idx, big_gaps + 1)
                            # Check if first cluster ends within first 650 ms and remaining speech >= 250 ms
                            first_cluster = clusters[0]
                            next_cluster_start = int(clusters[1][0])
                            if (
                                int(first_cluster[-1]) <= 65
                                and (int(clusters[-1][-1]) - next_cluster_start) >= 25
                            ):
                                cut_sample = max(0, (next_cluster_start - 2) * win)
                                trimmed_np = wav_np[cut_sample:]
                                tmp_trim = Path(str(candidate_path) + ".trim.wav")
                                sf.write(str(tmp_trim), trimmed_np, sr, subtype="PCM_16")
                                try:
                                    cls_mod = type(whisper_model).__module__
                                    if use_faster_whisper and "faster_whisper" in cls_mod:
                                        t_segs, _ = whisper_model.transcribe(str(tmp_trim), beam_size=1, language="en")
                                        t_txt = "".join(s.text for s in t_segs).strip().lower()
                                    else:
                                        t_res = whisper_model.transcribe(str(tmp_trim), language="en")
                                        t_txt = t_res.get("text", "").strip().lower()
                                    t_score = fuzzy_match_score(t_txt, target_text)
                                    if t_score >= max(score, 0.85):
                                        os.replace(tmp_trim, candidate_path)
                                        score = t_score
                                        transcribed = t_txt
                                finally:
                                    if tmp_trim.exists():
                                        tmp_trim.unlink(missing_ok=True)
        except Exception:
            pass

        return (str(candidate_path), float(score), transcribed)
    except Exception as e:
        return (str(candidate_path), 0.0, f"ERROR: {e}")


def select_best_candidate(
    passed_candidates: List[dict],
    failed_candidates: List[dict],
    use_longest_transcript_on_fail: bool = True,
    bypass_whisper_checking: bool = False,
) -> Optional[dict]:
    """
    Select the best candidate for a chunk following Chatterbox-TTS-Extended rules:
    1. If one or more candidates passed Whisper Sync validation (or Whisper was bypassed),
       pick the candidate with the SHORTEST duration (since hallucinations/artifacts add duration).
    2. If all candidates failed Whisper Sync validation:
       - If `use_longest_transcript_on_fail` is True, pick the candidate with the longest transcript.
       - Otherwise, pick the candidate with the highest Whisper fuzzy match score.
    """
    if passed_candidates:
        return min(passed_candidates, key=lambda c: c["duration"])
    if failed_candidates:
        if use_longest_transcript_on_fail:
            return max(
                failed_candidates,
                key=lambda c: (len(c.get("transcribed", "")), c.get("score", 0.0)),
            )
        return max(
            failed_candidates,
            key=lambda c: (c.get("score", 0.0), -c.get("duration", float("inf"))),
        )
    return None


def _resolve_auto_editor_cmd() -> List[str]:
    """Locate the auto-editor executable in .venv/bin or PATH, or fall back to `python -m auto_editor`."""
    venv_ae = SCRIPT_DIR / ".venv" / "bin" / "auto-editor"
    if venv_ae.exists():
        return [str(venv_ae)]
    which_ae = shutil.which("auto-editor")
    if which_ae:
        return [which_ae]
    return [sys.executable, "-m", "auto_editor"]


def run_auto_editor_on_wav(
    wav_path: Path,
    threshold: float = 0.06,
    margin: float = 0.2,
    keep_original: bool = False,
) -> bool:
    """
    Run auto-editor on `wav_path` in-place to remove low-sounding hallucinations
    (such as extended breathing sounds) below `threshold` with buffer `margin` seconds.
    """
    wav_path = Path(wav_path)
    if not wav_path.exists() or wav_path.stat().st_size <= 44:
        return False

    cleaned_output = wav_path.with_name(f"{wav_path.stem}_cleaned.wav")
    backup_path = wav_path.with_name(f"{wav_path.stem}_original.wav")

    try:
        if keep_original:
            shutil.copy2(wav_path, backup_path)
            auto_editor_input = backup_path
        else:
            auto_editor_input = wav_path

        base_cmd = _resolve_auto_editor_cmd()
        # Modern auto-editor (v26+/v29+) exports WAV directly from -o <file.wav>
        cmd_modern = base_cmd + [
            "--edit", f"audio:threshold={threshold}",
            "--margin", f"{margin}s",
            str(auto_editor_input),
            "-o", str(cleaned_output),
            "--no-open",
            "--quiet",
        ]
        res = subprocess.run(cmd_modern, capture_output=True, text=True, timeout=30)
        if res.returncode != 0 or not cleaned_output.exists():
            # Fallback for older auto-editor CLI versions that require `--export audio`
            cmd_legacy = base_cmd + [
                "--edit", f"audio:threshold={threshold}",
                "--margin", f"{margin}s",
                "--export", "audio",
                str(auto_editor_input),
                "-o", str(cleaned_output),
            ]
            subprocess.run(cmd_legacy, capture_output=True, text=True, timeout=30, check=True)

        if cleaned_output.exists() and cleaned_output.stat().st_size > 44:
            # Verify cleaned audio has non-zero samples before replacing
            data, _sr = sf.read(str(cleaned_output), dtype="float32")
            if len(data) > 0:
                os.replace(cleaned_output, wav_path)
                return True
        if cleaned_output.exists():
            cleaned_output.unlink(missing_ok=True)
        return False
    except Exception:
        if cleaned_output.exists():
            cleaned_output.unlink(missing_ok=True)
        return False


def _extract_pause_tags_and_clean_text(text: str) -> Tuple[float, str, float]:
    """
    Extract any leading/trailing `[pause:X.Xs]` tags so they become real digital silence
    instead of being spoken aloud or failing Whisper transcript validation.
    """
    lead_pause_sec = 0.0
    trail_pause_sec = 0.0
    s = str(text).strip()

    while True:
        m_lead = re.match(r"^\[\s*pause\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*\]\s*", s, flags=re.IGNORECASE)
        if not m_lead:
            break
        lead_pause_sec += float(m_lead.group(1))
        s = s[m_lead.end():].strip()

    while True:
        m_trail = re.search(r"\s*\[\s*pause\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*s?\s*\]$", s, flags=re.IGNORECASE)
        if not m_trail:
            break
        trail_pause_sec += float(m_trail.group(1))
        s = s[:m_trail.start()].strip()

    # Strip any remaining mid-sentence [pause:...] tags
    s = re.sub(r"\[\s*pause\s*:\s*[^\]]*\]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return lead_pause_sec, s, trail_pause_sec


def get_gpu_memory_info(device: str) -> Optional[Tuple[float, float, float]]:
    """
    Query (used_mib, free_mib, total_mib) for a CUDA device (e.g. 'cuda:0', 'cuda:1', 'cuda')
    without initializing a PyTorch CUDA context in the coordinator process when pynvml or
    nvidia-smi is available.
    """
    if not device.startswith("cuda"):
        return None
    parts = device.split(":")
    gpu_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    # 1. Prefer pynvml (zero-overhead hardware query)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mib = float(mem.total) / (1024 * 1024)
        used_mib = float(mem.used) / (1024 * 1024)
        free_mib = float(mem.free) / (1024 * 1024)
        return (used_mib, free_mib, total_mib)
    except Exception:
        pass

    # 2. Fallback to nvidia-smi
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_idx}",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        if out:
            line = out.splitlines()[0]
            vals = [float(x.strip()) for x in line.split(",")]
            if len(vals) >= 3:
                return (vals[0], vals[1], vals[2])
    except Exception:
        pass

    # 3. Fallback to torch.cuda.mem_get_info
    try:
        if torch.cuda.is_available() and gpu_idx < torch.cuda.device_count():
            free_b, total_b = torch.cuda.mem_get_info(gpu_idx)
            free_mib = float(free_b) / (1024 * 1024)
            total_mib = float(total_b) / (1024 * 1024)
            used_mib = total_mib - free_mib
            return (used_mib, free_mib, total_mib)
    except Exception:
        pass

    return None


def get_gpu_process_usages_mib(device: str) -> List[float]:
    """Return list of used VRAM (MiB) for active compute processes on the given CUDA device."""
    if not device.startswith("cuda"):
        return []
    parts = device.split(":")
    gpu_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        return [
            float(p.usedGpuMemory) / (1024 * 1024)
            for p in procs
            if getattr(p, "usedGpuMemory", None) is not None and p.usedGpuMemory > 0
        ]
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_idx}",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        usages = []
        for line in out.splitlines():
            s = line.strip()
            if s:
                try:
                    usages.append(float(s))
                except ValueError:
                    pass
        return usages
    except Exception:
        return []


def calculate_gpu_available_space(
    device: str,
    model_type: str = "standard",
    active_workers_on_gpu: int = 0,
    baseline_used_mib: Optional[float] = None,
    safety_buffer_mib: Optional[float] = None,
    max_instances_per_gpu: Optional[int] = None,
) -> dict:
    """
    Calculate available VRAM on `device` and determine whether additional Chatterbox TTS
    worker instances can fit on this GPU.

    Accounts for:
    - Expected steady-state working VRAM per model instance (~6.4 GiB / ~40% of 16 GB for standard,
      ~4.5 GiB for turbo) vs initial post-load weight footprint (~2.2 GiB) before first chunk synthesis.
    - Actual measured VRAM per running worker once warmed up.
    - Safety headroom buffer (`GPU_MEMORY_SAFETY_BUFFER_MIB`) to prevent CUDA OOM spikes.
    """
    est_dict = getattr(config, "INSTANCE_VRAM_ESTIMATE_MIB", {"standard": 6000, "turbo": 4200})
    est_instance_mib = float(est_dict.get(model_type, 6000))
    if safety_buffer_mib is None:
        safety_buffer_mib = float(getattr(config, "GPU_MEMORY_SAFETY_BUFFER_MIB", 800))
    if max_instances_per_gpu is None:
        max_instances_per_gpu = int(getattr(config, "MAX_INSTANCES_PER_GPU", 4))

    mem_info = get_gpu_memory_info(device)
    if mem_info is None:
        return {
            "device": device,
            "is_cuda": False,
            "used_mib": 0.0,
            "free_mib": 0.0,
            "total_mib": 0.0,
            "used_pct": 0.0,
            "free_pct": 0.0,
            "effective_free_mib": 0.0,
            "per_instance_mib": est_instance_mib,
            "required_mib": est_instance_mib + safety_buffer_mib,
            "can_add_instance": False,
            "additional_instances": 0,
        }

    used_mib, free_mib, total_mib = mem_info
    used_pct = (used_mib / total_mib * 100.0) if total_mib > 0 else 0.0
    free_pct = (free_mib / total_mib * 100.0) if total_mib > 0 else 0.0

    if baseline_used_mib is not None:
        # Coordinator tracking mode: we know baseline VRAM before our workers spawned
        our_used_mib = max(0.0, used_mib - baseline_used_mib)
        pending_warmup_mib = max(0.0, active_workers_on_gpu * est_instance_mib - our_used_mib)
        effective_free_mib = max(0.0, free_mib - pending_warmup_mib)
        per_instance_mib = est_instance_mib
        total_workers_on_gpu = active_workers_on_gpu
    else:
        # Standalone / external process inspection mode: inspect running compute apps on this GPU
        proc_usages = [u for u in get_gpu_process_usages_mib(device) if u >= 1000.0]
        pending_warmup_mib = sum(max(0.0, est_instance_mib - u) for u in proc_usages)
        effective_free_mib = max(0.0, free_mib - pending_warmup_mib)
        per_instance_mib = est_instance_mib
        total_workers_on_gpu = max(active_workers_on_gpu, len(proc_usages))

    required_mib = per_instance_mib + safety_buffer_mib
    remaining_slots = max(0, max_instances_per_gpu - total_workers_on_gpu)

    if effective_free_mib >= required_mib and remaining_slots > 0:
        raw_fit = int((effective_free_mib - safety_buffer_mib) // per_instance_mib)
        additional_instances = min(remaining_slots, max(1, raw_fit))
        can_add = True
    else:
        additional_instances = 0
        can_add = False

    return {
        "device": device,
        "is_cuda": True,
        "used_mib": used_mib,
        "free_mib": free_mib,
        "total_mib": total_mib,
        "used_pct": used_pct,
        "free_pct": free_pct,
        "effective_free_mib": effective_free_mib,
        "per_instance_mib": per_instance_mib,
        "required_mib": required_mib,
        "can_add_instance": can_add,
        "additional_instances": additional_instances,
        "active_workers": total_workers_on_gpu,
    }


def _worker_process_fn(
    device: str,
    model_type: str,
    ref_audio: Optional[str],
    gen_kwargs: dict,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    instance_idx: int = 0,
    cache_only: bool = False,
):
    """Worker process running on a dedicated GPU/device."""
    import sys
    from pathlib import Path
    _sdir = Path(__file__).resolve().parent
    _vsite = _sdir / ".venv" / "lib" / "python3.11" / "site-packages"
    if _vsite.exists() and str(_vsite) not in sys.path:
        sys.path.insert(0, str(_vsite))
    _rsrc = _sdir / "chatterbox_repo" / "src"
    if str(_rsrc) not in sys.path:
        sys.path.insert(0, str(_rsrc))

    worker_label = f"{device}#{instance_idx + 1}"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")

    try:
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(device)
        if model_type == "turbo":
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_pretrained(device=device)
        else:
            from chatterbox.tts import ChatterboxTTS
            model = ChatterboxTTS.from_pretrained(device=device)
        if ref_audio:
            model.prepare_conditionals(
                ref_audio,
                exaggeration=gen_kwargs.get("exaggeration", 0.5),
            )
    except Exception as e:
        err_str = str(e)
        if instance_idx > 0 and ("out of memory" in err_str.lower() or isinstance(e, torch.cuda.OutOfMemoryError)):
            result_queue.put((-1, "oom_init", f"CUDA OOM initializing extra worker {worker_label}: {e}", 0.0, worker_label))
        else:
            result_queue.put((-1, "init_error", f"Failed to load model on {worker_label}: {e}", 0.0, worker_label))
        return

    result_queue.put((-1, "ready", device, 0.0, worker_label))
    synth_counter = 0

    # Chatterbox-TTS-Extended (Issue #97) validation & post-processing settings
    num_candidates = max(1, int(gen_kwargs.get("num_candidates", getattr(config, "NUM_CANDIDATES", 3))))
    max_attempts = max(1, int(gen_kwargs.get("max_attempts", getattr(config, "MAX_ATTEMPTS", 3))))
    bypass_whisper = bool(gen_kwargs.get("bypass_whisper_checking", getattr(config, "BYPASS_WHISPER_CHECKING", False)))
    use_faster_whisper = bool(gen_kwargs.get("use_faster_whisper", getattr(config, "USE_FASTER_WHISPER", True)))
    whisper_model_name = str(gen_kwargs.get("whisper_model", getattr(config, "WHISPER_MODEL", "base")))
    whisper_device = str(gen_kwargs.get("whisper_device", getattr(config, "WHISPER_DEVICE", "cpu")))
    whisper_thresh = float(gen_kwargs.get("whisper_validation_threshold", getattr(config, "WHISPER_VALIDATION_THRESHOLD", 0.85)))
    use_longest_on_fail = bool(gen_kwargs.get("use_longest_transcript_on_fail", getattr(config, "USE_LONGEST_TRANSCRIPT_ON_FAIL", True)))
    use_auto_editor = bool(gen_kwargs.get("use_auto_editor", getattr(config, "USE_AUTO_EDITOR", True)))
    ae_threshold = float(gen_kwargs.get("ae_threshold", getattr(config, "AUTO_EDITOR_THRESHOLD", 0.06)))
    ae_margin = float(gen_kwargs.get("ae_margin", getattr(config, "AUTO_EDITOR_MARGIN", 0.2)))
    keep_original_wav = bool(gen_kwargs.get("keep_original_wav", getattr(config, "KEEP_ORIGINAL_WAV", False)))
    base_seed = int(gen_kwargs.get("seed", getattr(config, "SEED", 0)))

    whisper_model = None
    on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    rng_devices = [torch.cuda.current_device()] if on_cuda else []

    try:
        while True:
            task = None
            try:
                task = task_queue.get()
                if task is None:
                    break

                chunk_idx, text, cache_path_str, chunk_file_str = task
                cache_path = Path(cache_path_str)
                chunk_file = Path(chunk_file_str)

                # Check cache (or existing chunk_file in cache_only assist mode) first
                if cache_path.exists() and cache_path.stat().st_size > 44:
                    if not cache_only:
                        shutil.copy2(cache_path, chunk_file)
                    num_words = len(text.split())
                    result_queue.put((chunk_idx, "cached", num_words, 0.0, worker_label))
                    continue
                if cache_only and chunk_file.exists() and chunk_file.stat().st_size > 44:
                    num_words = len(text.split())
                    result_queue.put((chunk_idx, "cached", num_words, 0.0, worker_label))
                    continue

                # Lazy-load Whisper model on first uncached chunk synthesis if validation is enabled
                if not bypass_whisper and whisper_model is None:
                    try:
                        whisper_model = load_whisper_backend(
                            model_name=whisper_model_name,
                            use_faster_whisper=use_faster_whisper,
                            device=whisper_device,
                        )
                    except Exception:
                        whisper_model = None

                # Synthesize (conditionals already prepared on worker init)
                t0 = time.time()
                lead_pause_sec, stripped_text, trail_pause_sec = _extract_pause_tags_and_clean_text(text)
                clean_text = sanitize_spoken_for_tts(stripped_text) or stripped_text or " "
                if clean_text.strip() and clean_text.strip()[-1] not in ".!?:;\"'”’)":
                    clean_text = clean_text.strip() + "."
                num_words = len(clean_text.split())

                # Moderate sampling parameters on ultra-short chunks (<= 5 words) such as
                # headings and short list items to prevent T3 cross-attention onset garble
                base_exag = gen_kwargs.get("exaggeration", 0.5)
                base_cfg = gen_kwargs.get("cfg_weight", 0.5)
                base_temp = gen_kwargs.get("temperature", 0.8)
                if num_words <= 5:
                    chunk_exag = min(base_exag, 0.35)
                    chunk_cfg = min(base_cfg, 0.35)
                    chunk_temp = min(base_temp, 0.62)
                else:
                    chunk_exag = base_exag
                    chunk_cfg = base_cfg
                    chunk_temp = base_temp

                cache_path.parent.mkdir(parents=True, exist_ok=True)
                passed_candidates: List[dict] = []
                failed_candidates: List[dict] = []
                cand_temp_paths: List[Path] = []

                try:
                    for attempt_idx in range(max_attempts):
                        for cand_idx in range(num_candidates):
                            cand_seed = derive_seed(base_seed, chunk_idx, cand_idx, attempt_idx)
                            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                                torch.manual_seed(int(cand_seed))
                                if on_cuda:
                                    torch.cuda.manual_seed_all(int(cand_seed))
                                random.seed(int(cand_seed))
                                np.random.seed(int(cand_seed) & 0xFFFFFFFF)

                                if model_type == "turbo":
                                    wav = model.generate(
                                        clean_text,
                                        audio_prompt_path=None,
                                        exaggeration=chunk_exag,
                                    )
                                else:
                                    wav = model.generate(
                                        clean_text,
                                        audio_prompt_path=None,
                                        exaggeration=chunk_exag,
                                        cfg_weight=chunk_cfg,
                                        temperature=chunk_temp,
                                        top_p=gen_kwargs.get("top_p", 1.0),
                                        repetition_penalty=gen_kwargs.get("repetition_penalty", 1.2),
                                    )

                            cand_np = wav.squeeze().detach().cpu().numpy()
                            if cand_np.ndim > 1:
                                cand_np = cand_np.squeeze()
                            cand_np = np.nan_to_num(cand_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                            cand_dur = len(cand_np) / float(model.sr)

                            cand_path = cache_path.with_name(
                                f".{cache_path.stem}.{os.getpid()}.c{cand_idx}_a{attempt_idx}.wav"
                            )
                            sf.write(str(cand_path), cand_np, model.sr, subtype="PCM_16")
                            cand_temp_paths.append(cand_path)

                            if bypass_whisper or whisper_model is None:
                                passed_candidates.append(
                                    {
                                        "path": str(cand_path),
                                        "duration": cand_dur,
                                        "score": 1.0,
                                        "transcribed": clean_text,
                                        "cand_idx": cand_idx,
                                        "attempt": attempt_idx,
                                        "seed": cand_seed,
                                    }
                                )
                            else:
                                _p, score, transcribed = whisper_check_mp(
                                    str(cand_path),
                                    clean_text,
                                    whisper_model,
                                    use_faster_whisper=use_faster_whisper,
                                )
                                try:
                                    cand_dur = float(sf.info(str(cand_path)).duration)
                                except Exception:
                                    pass
                                req_thresh = 0.95 if attempt_idx > 0 else whisper_thresh
                                cand_entry = {
                                    "path": str(cand_path),
                                    "duration": cand_dur,
                                    "score": score,
                                    "transcribed": transcribed,
                                    "cand_idx": cand_idx,
                                    "attempt": attempt_idx,
                                    "seed": cand_seed,
                                }
                                if score >= req_thresh or score >= whisper_thresh:
                                    passed_candidates.append(cand_entry)
                                else:
                                    failed_candidates.append(cand_entry)

                        # Stop retrying once at least one candidate passes Whisper Sync validation
                        if passed_candidates:
                            break

                    best_cand = select_best_candidate(
                        passed_candidates=passed_candidates,
                        failed_candidates=failed_candidates,
                        use_longest_transcript_on_fail=use_longest_on_fail,
                        bypass_whisper_checking=bypass_whisper,
                    )
                    if best_cand is None:
                        raise RuntimeError(f"No valid audio candidates generated for chunk {chunk_idx}")

                    best_path = Path(best_cand["path"])

                    # Run auto-editor on the winning candidate to strip low-sounding hallucinations / breathing
                    if use_auto_editor:
                        if keep_original_wav and not cache_only:
                            orig_backup = chunk_file.with_name(f"{chunk_file.stem}_original.wav")
                            orig_backup.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(best_path, orig_backup)
                        run_auto_editor_on_wav(
                            best_path,
                            threshold=ae_threshold,
                            margin=ae_margin,
                            keep_original=False,
                        )

                    wav_np, _sr = sf.read(str(best_path), dtype="float32")
                    wav_np = _clean_and_normalize_chunk_wav(wav_np, model.sr)

                    # Prepend/append any explicit [pause:X.Xs] silence requested in the chunk
                    if lead_pause_sec > 0.0:
                        lead_zeros = np.zeros(int(model.sr * lead_pause_sec), dtype=np.float32)
                        wav_np = np.concatenate([lead_zeros, wav_np])
                    if trail_pause_sec > 0.0:
                        trail_zeros = np.zeros(int(model.sr * trail_pause_sec), dtype=np.float32)
                        wav_np = np.concatenate([wav_np, trail_zeros])
                finally:
                    for cp in cand_temp_paths:
                        try:
                            cp.unlink(missing_ok=True)
                        except Exception:
                            pass

                t_gen = time.time() - t0

                # Ensure atomic 16-bit PCM WAV writes so concurrent/assist workers never read partial files
                tmp_cache = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.wav")
                sf.write(str(tmp_cache), wav_np, model.sr, subtype="PCM_16")
                os.replace(tmp_cache, cache_path)
                if not cache_only:
                    chunk_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cache_path, chunk_file)

                num_words = len(clean_text.split())
                dur = len(wav_np) / model.sr
                synth_counter += 1
                if synth_counter % 20 == 0 and device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result_queue.put((chunk_idx, "synthesized", num_words, dur, worker_label))

            except Exception as e:
                err_str = str(e)
                if instance_idx > 0 and ("out of memory" in err_str.lower() or isinstance(e, torch.cuda.OutOfMemoryError)):
                    if task is not None:
                        task_queue.put(task)
                    result_queue.put((
                        -1,
                        "oom_retire",
                        f"CUDA OOM on {worker_label} during chunk {task[0] if task else '?'}; chunk re-queued and extra worker retired",
                        0.0,
                        worker_label,
                    ))
                    break
                result_queue.put((chunk_idx if task else -1, "error", err_str, 0.0, worker_label))
    finally:
        # Explicitly unload models and release GPU memory
        try:
            del whisper_model
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
    """Text -> Audiobook synthesizer using Chatterbox TTS."""

    _SUPPORTED_AUDIO_FORMATS = ("m4b", "mp3", "wav", "m4a", "flac")

    def __init__(
        self,
        reference_id: Optional[str] = None,
        reference_audio: Optional[str] = None,
        model_type: Optional[str] = None,
        devices: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        exaggeration: Optional[float] = None,
        no_cache: bool = False,
        auto_scale_gpu: Optional[bool] = None,
        max_instances_per_gpu: Optional[int] = None,
        num_candidates: Optional[int] = None,
        max_attempts: Optional[int] = None,
        bypass_whisper_checking: Optional[bool] = None,
        use_faster_whisper: Optional[bool] = None,
        whisper_model: Optional[str] = None,
        whisper_device: Optional[str] = None,
        whisper_validation_threshold: Optional[float] = None,
        use_longest_transcript_on_fail: Optional[bool] = None,
        use_auto_editor: Optional[bool] = None,
        ae_threshold: Optional[float] = None,
        ae_margin: Optional[float] = None,
        keep_original_wav: Optional[bool] = None,
        smart_append_short: Optional[bool] = None,
        min_sentence_chars: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.reference_id = reference_id or config.DEFAULT_REFERENCE_ID
        self.reference_audio = reference_audio or config.DEFAULT_REFERENCE_AUDIO
        self.model_type = model_type or config.MODEL_TYPE
        self.temperature = temperature if temperature is not None else config.TEMPERATURE
        self.top_p = top_p if top_p is not None else config.TOP_P
        self.repetition_penalty = repetition_penalty if repetition_penalty is not None else config.REPETITION_PENALTY
        self.cfg_weight = cfg_weight if cfg_weight is not None else config.CFG_WEIGHT
        self.exaggeration = exaggeration if exaggeration is not None else config.EXAGGERATION
        self.no_cache = no_cache
        self.auto_scale_gpu = (
            auto_scale_gpu
            if auto_scale_gpu is not None
            else getattr(config, "AUTO_SCALE_GPU_INSTANCES", True)
        )
        self.max_instances_per_gpu = (
            max_instances_per_gpu
            if max_instances_per_gpu is not None
            else int(getattr(config, "MAX_INSTANCES_PER_GPU", 4))
        )
        self.num_candidates = (
            num_candidates
            if num_candidates is not None
            else int(getattr(config, "NUM_CANDIDATES", 3))
        )
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else int(getattr(config, "MAX_ATTEMPTS", 3))
        )
        self.bypass_whisper_checking = (
            bypass_whisper_checking
            if bypass_whisper_checking is not None
            else bool(getattr(config, "BYPASS_WHISPER_CHECKING", False))
        )
        self.use_faster_whisper = (
            use_faster_whisper
            if use_faster_whisper is not None
            else bool(getattr(config, "USE_FASTER_WHISPER", True))
        )
        self.whisper_model = (
            whisper_model
            if whisper_model is not None
            else str(getattr(config, "WHISPER_MODEL", "base"))
        )
        self.whisper_device = (
            whisper_device
            if whisper_device is not None
            else str(getattr(config, "WHISPER_DEVICE", "cpu"))
        )
        self.whisper_validation_threshold = (
            whisper_validation_threshold
            if whisper_validation_threshold is not None
            else float(getattr(config, "WHISPER_VALIDATION_THRESHOLD", 0.85))
        )
        self.use_longest_transcript_on_fail = (
            use_longest_transcript_on_fail
            if use_longest_transcript_on_fail is not None
            else bool(getattr(config, "USE_LONGEST_TRANSCRIPT_ON_FAIL", True))
        )
        self.use_auto_editor = (
            use_auto_editor
            if use_auto_editor is not None
            else bool(getattr(config, "USE_AUTO_EDITOR", True))
        )
        self.ae_threshold = (
            ae_threshold
            if ae_threshold is not None
            else float(getattr(config, "AUTO_EDITOR_THRESHOLD", 0.06))
        )
        self.ae_margin = (
            ae_margin
            if ae_margin is not None
            else float(getattr(config, "AUTO_EDITOR_MARGIN", 0.2))
        )
        self.keep_original_wav = (
            keep_original_wav
            if keep_original_wav is not None
            else bool(getattr(config, "KEEP_ORIGINAL_WAV", False))
        )
        self.smart_append_short = (
            smart_append_short
            if smart_append_short is not None
            else bool(getattr(config, "SMART_APPEND_SHORT_SENTENCES", True))
        )
        self.min_sentence_chars = (
            min_sentence_chars
            if min_sentence_chars is not None
            else int(getattr(config, "MIN_SENTENCE_CHARS", 20))
        )
        self.seed = (
            seed
            if seed is not None
            else int(getattr(config, "SEED", 0))
        )

        # Setup devices
        if devices:
            self.devices = devices
        elif config.DEVICES:
            self.devices = config.DEVICES if isinstance(config.DEVICES, list) else [config.DEVICES]
        else:
            if torch.cuda.is_available():
                self.devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            else:
                self.devices = ["cpu"]

        # Resolve reference audio
        if not self.reference_audio and self.reference_id:
            ref_dir = Path(config.REFERENCES_FOLDER) / self.reference_id
            if not ref_dir.exists():
                ref_dir = SCRIPT_DIR / config.REFERENCES_FOLDER / self.reference_id
            wavs = sorted(ref_dir.glob("*.wav"))
            if wavs:
                self.reference_audio = str(wavs[0].resolve())

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
        soft_merge_limit = limit + min_standalone_words

        for sentence in sentences:
            sentence = sanitize_spoken_for_tts(sentence.strip())
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
                        if current_chunk and current_words >= min_standalone_words:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                            current_words = 0
                        combined_words = (current_chunk + " " + part).split() if current_chunk else part.split()
                        current_chunk = ""
                        current_words = 0
                        num_parts = max(2, (len(combined_words) + limit - 1) // limit)
                        step = max(min_standalone_words, (len(combined_words) + num_parts - 1) // num_parts)
                        for i in range(0, len(combined_words), step):
                            sub = combined_words[i : i + step]
                            if chunks and len(sub) < min_standalone_words:
                                chunks[-1] = f"{chunks[-1]} {' '.join(sub)}"
                            else:
                                chunks.append(" ".join(sub))
                    elif current_words < min_standalone_words or (current_words + part_words <= limit):
                        current_chunk += part + " "
                        current_words += part_words
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                        current_words = part_words
                if current_chunk:
                    if chunks and current_words < min_standalone_words and (len(chunks[-1].split()) + current_words <= soft_merge_limit):
                        chunks[-1] = f"{chunks[-1]} {current_chunk.strip()}"
                        current_chunk = ""
                        current_words = 0
                    elif current_words >= min_standalone_words:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                        current_words = 0
            else:
                # Only merge with previous chunk if previous chunk is a tiny fragment (< 6 words)
                # or this sentence is a tiny fragment (< 6 words) and fits within soft_merge_limit.
                if current_chunk and (current_words < min_standalone_words or sentence_words < min_standalone_words) and (current_words + sentence_words <= soft_merge_limit):
                    current_chunk += sentence + " "
                    current_words += sentence_words
                elif sentence_words < min_standalone_words and chunks and not current_chunk and (len(chunks[-1].split()) + sentence_words <= soft_merge_limit):
                    chunks[-1] = f"{chunks[-1]} {sentence}"
                    continue
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
            if chunks and current_words < min_standalone_words and (len(chunks[-1].split()) + current_words <= soft_merge_limit):
                chunks[-1] = f"{chunks[-1]} {current_chunk.strip()}"
            else:
                chunks.append(current_chunk.strip())

        chunks = [c for c in chunks if c.strip()]
        if self.smart_append_short and len(chunks) > 1:
            chunks = smart_append_short_sentences(chunks, min_chars=self.min_sentence_chars, max_chars=300)
            chunks = enforce_min_chunk_length(chunks, min_len=self.min_sentence_chars, max_len=300)
        return chunks

    def _build_gen_kwargs(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "cfg_weight": self.cfg_weight,
            "exaggeration": self.exaggeration,
            "num_candidates": self.num_candidates,
            "max_attempts": self.max_attempts,
            "bypass_whisper_checking": self.bypass_whisper_checking,
            "use_faster_whisper": self.use_faster_whisper,
            "whisper_model": self.whisper_model,
            "whisper_device": self.whisper_device,
            "whisper_validation_threshold": self.whisper_validation_threshold,
            "use_longest_transcript_on_fail": self.use_longest_transcript_on_fail,
            "use_auto_editor": self.use_auto_editor,
            "ae_threshold": self.ae_threshold,
            "ae_margin": self.ae_margin,
            "keep_original_wav": self.keep_original_wav,
            "seed": self.seed,
        }

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

        key = (
            f"chatterbox_v2_{self.model_type}_{text}_{self._cached_ref_str}_{self.temperature}_"
            f"{self.top_p}_{self.repetition_penalty}_{self.cfg_weight}_{self.exaggeration}_"
            f"c{self.num_candidates}_ae{int(self.use_auto_editor)}"
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
        book_author = author or "Chatterbox TTS"
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
        explicit_author = author if (author and author not in ("Chatterbox TTS", "MinerU")) else None
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

        console.print(
            Panel(
                f"[bold cyan]Chatterbox TTS Audiobook & Synchronized EPUB3 Generator[/bold cyan]\n\n"
                f"• Input        : [yellow]{input_path}[/yellow]\n"
                f"• DOM Source   : [yellow]{dom_source_path}[/yellow]\n"
                f"• Audio Output : [green]{output_path}[/green]\n"
                f"• EPUB3 Output : [green]{epub_output_path if not skip_epub else 'Skipped (--skip-epub)'}[/green]\n"
                f"• Model        : [magenta]{self.model_type.upper()}[/magenta]\n"
                f"• Devices      : [cyan]{', '.join(self.devices)}[/cyan]\n"
                f"• Reference    : [blue]{self.reference_audio or 'Built-in Chatterbox Timbre'}[/blue]\n"
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
        gen_kwargs = self._build_gen_kwargs()

        if tasks_to_run:
            ctx = mp.get_context("spawn")
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()

            # Put all tasks into queue first (poison pills are sent only on shutdown
            # so workers can scale up dynamically and re-queue tasks on OOM)
            for task in tasks_to_run:
                task_queue.put(task)

            workers = []
            instances_on_gpu: dict = {}
            next_instance_idx: dict = {}
            pending_init_on_gpu: dict = {}
            gpu_full: dict = {}
            baseline_used_mib: dict = {}
            synthesized_on_gpu: dict = {}

            unique_devices = list(dict.fromkeys(self.devices))
            for d in unique_devices:
                instances_on_gpu[d] = 0
                next_instance_idx[d] = 0
                pending_init_on_gpu[d] = 0
                gpu_full[d] = False
                synthesized_on_gpu[d] = 0
                mem_info = get_gpu_memory_info(d)
                if mem_info is not None:
                    used_mib, free_mib, total_mib = mem_info
                    baseline_used_mib[d] = used_mib
                    used_pct = (used_mib / total_mib * 100.0) if total_mib > 0 else 0.0
                    console.print(
                        f"GPU [cyan]{d}[/cyan] initial memory: [yellow]{used_mib:.0f}/{total_mib:.0f} MiB[/yellow] "
                        f"({used_pct:.1f}% used, [green]{free_mib:.0f} MiB free[/green])"
                    )

            def _spawn_worker_on_device(dev_str: str) -> str:
                inst_idx = next_instance_idx.get(dev_str, 0)
                next_instance_idx[dev_str] = inst_idx + 1
                pending_init_on_gpu[dev_str] = pending_init_on_gpu.get(dev_str, 0) + 1
                p = ctx.Process(
                    target=_worker_process_fn,
                    args=(
                        dev_str,
                        self.model_type,
                        self.reference_audio,
                        gen_kwargs,
                        task_queue,
                        result_queue,
                        inst_idx,
                        False,
                    ),
                )
                p.daemon = True
                p.start()
                workers.append(p)
                return f"{dev_str}#{inst_idx + 1}"

            try:
                # 3a. Spawn initial configured workers (capped by number of tasks)
                initial_devices = self.devices[: max(1, len(tasks_to_run))]
                for d in initial_devices:
                    _spawn_worker_on_device(d)

                # Wait for initial workers to finish loading
                pending_initial = len(initial_devices)
                ready_workers = 0
                deferred_results = []
                while pending_initial > 0:
                    msg_tuple = result_queue.get()
                    chunk_idx, status, payload, dur, w_label = msg_tuple
                    base_dev = w_label.split("#")[0]
                    if status == "ready":
                        pending_initial -= 1
                        pending_init_on_gpu[base_dev] = max(0, pending_init_on_gpu.get(base_dev, 1) - 1)
                        instances_on_gpu[base_dev] = instances_on_gpu.get(base_dev, 0) + 1
                        ready_workers += 1
                        console.print(f"Worker [cyan]{w_label}[/cyan] loaded and ready.")
                    elif status == "oom_init":
                        pending_initial -= 1
                        pending_init_on_gpu[base_dev] = max(0, pending_init_on_gpu.get(base_dev, 1) - 1)
                        gpu_full[base_dev] = True
                        console.print(f"[yellow]⚠ {payload} (marking {base_dev} at capacity)[/yellow]")
                    elif status == "init_error":
                        pending_initial -= 1
                        pending_init_on_gpu[base_dev] = max(0, pending_init_on_gpu.get(base_dev, 1) - 1)
                        console.print(f"[bold red]{payload}[/bold red]")
                        if ready_workers == 0 and pending_initial == 0:
                            raise RuntimeError(payload)
                    else:
                        deferred_results.append(msg_tuple)

                if ready_workers == 0:
                    raise RuntimeError("No Chatterbox TTS worker instances could be initialized.")

                # 3b. Calculate available GPU space and spawn additional instances where space permits
                if self.auto_scale_gpu and len(tasks_to_run) > ready_workers:
                    for d in unique_devices:
                        if not d.startswith("cuda"):
                            continue
                        while (
                            not gpu_full.get(d, False)
                            and instances_on_gpu.get(d, 0) < self.max_instances_per_gpu
                            and ready_workers < len(tasks_to_run)
                        ):
                            space_info = calculate_gpu_available_space(
                                device=d,
                                model_type=self.model_type,
                                active_workers_on_gpu=instances_on_gpu.get(d, 0),
                                baseline_used_mib=baseline_used_mib.get(d),
                                max_instances_per_gpu=self.max_instances_per_gpu,
                            )
                            console.print(
                                f"GPU [cyan]{d}[/cyan] space check: "
                                f"[yellow]{space_info['used_mib']:.0f}/{space_info['total_mib']:.0f} MiB[/yellow] used "
                                f"({space_info['used_pct']:.1f}%), "
                                f"[green]{space_info['effective_free_mib']:.0f} MiB[/green] effective free "
                                f"(need [magenta]{space_info['required_mib']:.0f} MiB[/magenta] for +1 instance)"
                            )
                            self.logger.info(
                                f"GPU {d} space check: used={space_info['used_mib']:.0f}MiB "
                                f"({space_info['used_pct']:.1f}%), effective_free={space_info['effective_free_mib']:.0f}MiB, "
                                f"required={space_info['required_mib']:.0f}MiB, can_add={space_info['can_add_instance']}"
                            )
                            if not space_info["can_add_instance"]:
                                break

                            new_label = _spawn_worker_on_device(d)
                            console.print(
                                f"[bold green]+ Adding Chatterbox instance [cyan]{new_label}[/cyan] on {d} "
                                f"({space_info['effective_free_mib']:.0f} MiB available >= {space_info['required_mib']:.0f} MiB required)...[/bold green]"
                            )
                            self.logger.info(f"Spawning additional worker instance {new_label} on {d}")

                            # Wait for the new instance to load before re-measuring GPU memory
                            while pending_init_on_gpu.get(d, 0) > 0:
                                msg_tuple = result_queue.get()
                                chunk_idx, status, payload, dur, w_label = msg_tuple
                                b_dev = w_label.split("#")[0]
                                if status == "ready":
                                    pending_init_on_gpu[b_dev] = max(0, pending_init_on_gpu.get(b_dev, 1) - 1)
                                    instances_on_gpu[b_dev] = instances_on_gpu.get(b_dev, 0) + 1
                                    ready_workers += 1
                                    console.print(f"Worker [cyan]{w_label}[/cyan] loaded and ready ({instances_on_gpu[b_dev]} active on {b_dev}).")
                                elif status in ("oom_init", "init_error"):
                                    pending_init_on_gpu[b_dev] = max(0, pending_init_on_gpu.get(b_dev, 1) - 1)
                                    gpu_full[b_dev] = True
                                    console.print(f"[yellow]⚠ Extra worker {w_label} could not start ({payload}); keeping {instances_on_gpu[b_dev]} instance(s) on {b_dev}.[/yellow]")
                                else:
                                    deferred_results.append(msg_tuple)

                # Put any chunk results that arrived while scaling back onto result_queue
                for item in deferred_results:
                    result_queue.put(item)

                scale_check_interval = float(getattr(config, "GPU_SCALE_CHECK_INTERVAL_SEC", 45.0))
                last_scale_check_ts = time.time()
                warmup_scale_checked: dict = {d: False for d in unique_devices}

                # Process results with Rich progress bar
                with Progress(*_progress_columns(), console=console, disable=bool(os.environ.get("STATUS_DISPLAY_ACTIVE"))) as progress:
                    synth_task = progress.add_task(
                        f"Synthesizing chunks ({ready_workers} GPU workers)...",
                        total=total_chunks,
                        completed=cached_count,
                        speed="0.0 chk/s",
                    )

                    completed = cached_count
                    total_synth_words = 0
                    synth_start_time = time.time()

                    while completed < total_chunks:
                        chunk_idx, status, words_or_err, dur, dev = result_queue.get()
                        base_dev = dev.split("#")[0]

                        if status == "ready":
                            pending_init_on_gpu[base_dev] = max(0, pending_init_on_gpu.get(base_dev, 1) - 1)
                            instances_on_gpu[base_dev] = instances_on_gpu.get(base_dev, 0) + 1
                            ready_workers += 1
                            console.print(f"Dynamic worker [cyan]{dev}[/cyan] loaded and ready ({ready_workers} total active workers).")
                            progress.update(synth_task, description=f"Synthesizing chunks ({ready_workers} GPU workers)...")
                            continue
                        elif status in ("oom_init", "init_error"):
                            pending_init_on_gpu[base_dev] = max(0, pending_init_on_gpu.get(base_dev, 1) - 1)
                            gpu_full[base_dev] = True
                            self.logger.warning(f"Extra worker init stopped on {dev}: {words_or_err}")
                            continue
                        elif status == "oom_retire":
                            instances_on_gpu[base_dev] = max(1, instances_on_gpu.get(base_dev, 1) - 1)
                            ready_workers = max(1, ready_workers - 1)
                            gpu_full[base_dev] = True
                            console.print(f"[yellow]⚠ {words_or_err}[/yellow]")
                            self.logger.warning(str(words_or_err))
                            progress.update(synth_task, description=f"Synthesizing chunks ({ready_workers} GPU workers)...")
                            continue

                        if status in ("synthesized", "cached"):
                            completed += 1
                            total_synth_words += words_or_err
                            if status == "synthesized":
                                synthesized_on_gpu[base_dev] = synthesized_on_gpu.get(base_dev, 0) + 1
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

                        # Dynamic runtime GPU space check:
                        # Trigger once a GPU's existing workers have warmed up (>= 2 synthesized chunks per instance)
                        # or periodically every `scale_check_interval` seconds in case external GPU memory freed up.
                        now_ts = time.time()
                        remaining_chunks = total_chunks - completed
                        if (
                            self.auto_scale_gpu
                            and remaining_chunks > ready_workers * 4
                        ):
                            should_check_dev = None
                            if (
                                base_dev.startswith("cuda")
                                and not warmup_scale_checked.get(base_dev, True)
                                and not gpu_full.get(base_dev, False)
                                and pending_init_on_gpu.get(base_dev, 0) == 0
                                and synthesized_on_gpu.get(base_dev, 0) >= max(2, instances_on_gpu.get(base_dev, 1) * 2)
                            ):
                                warmup_scale_checked[base_dev] = True
                                should_check_dev = base_dev
                            elif now_ts - last_scale_check_ts >= scale_check_interval:
                                last_scale_check_ts = now_ts
                                for cand_d in unique_devices:
                                    if (
                                        cand_d.startswith("cuda")
                                        and not gpu_full.get(cand_d, False)
                                        and pending_init_on_gpu.get(cand_d, 0) == 0
                                        and instances_on_gpu.get(cand_d, 0) < self.max_instances_per_gpu
                                    ):
                                        should_check_dev = cand_d
                                        break

                            if should_check_dev is not None:
                                space_info = calculate_gpu_available_space(
                                    device=should_check_dev,
                                    model_type=self.model_type,
                                    active_workers_on_gpu=instances_on_gpu.get(should_check_dev, 0),
                                    baseline_used_mib=baseline_used_mib.get(should_check_dev),
                                    max_instances_per_gpu=self.max_instances_per_gpu,
                                )
                                if space_info["can_add_instance"]:
                                    new_label = _spawn_worker_on_device(should_check_dev)
                                    console.print(
                                        f"[bold green]+ Dynamic GPU scale-up: spawning [cyan]{new_label}[/cyan] on {should_check_dev} "
                                        f"({space_info['used_pct']:.1f}% VRAM used, {space_info['effective_free_mib']:.0f} MiB available >= "
                                        f"{space_info['required_mib']:.0f} MiB required)[/bold green]"
                                    )
                                    self.logger.info(
                                        f"Dynamic GPU scale-up: spawned {new_label} on {should_check_dev} "
                                        f"(used={space_info['used_mib']:.0f}MiB/{space_info['total_mib']:.0f}MiB)"
                                    )
            finally:
                # Send poison pills and terminate worker processes, then purge GPU memory
                console.print("[cyan]Unloading model and releasing GPU memory...[/cyan]")
                for _ in workers:
                    try:
                        task_queue.put_nowait(None)
                    except Exception:
                        pass
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
                console.print("[bold green]✓ Chatterbox model unloaded from memory and GPU VRAM released.[/bold green]")

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
            narrator_list = [f"Chatterbox TTS ({voice_label})"]
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
                    "authors": book_meta.get("authors") or ([book_author] if book_author != "Chatterbox TTS" else []),
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
                    "tags": ["TTS", "MinerU", "Chatterbox"],
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

    def assist_active_conversion(
        self,
        input_path: Path,
        markdown_dir: Optional[Path] = None,
        toc_file: Optional[Path] = None,
    ) -> bool:
        """
        Calculate available GPU space while an existing Chatterbox conversion is running,
        spawn additional worker instances on GPUs that have free headroom, and synthesize
        uncached chunks in reverse order into the shared atomic chunk cache so the running
        conversion completes in ~half the time without interruption.
        """
        input_path = Path(input_path).resolve()
        folder_label = input_path.stem if input_path.is_file() else input_path.name
        chunks_dir = Path(config.CHUNKS_FOLDER) / folder_label

        unique_devices = list(dict.fromkeys(self.devices))
        spawn_plan: List[Tuple[str, int]] = []
        for d in unique_devices:
            if not d.startswith("cuda"):
                continue
            info = calculate_gpu_available_space(
                device=d,
                model_type=self.model_type,
                max_instances_per_gpu=self.max_instances_per_gpu,
            )
            console.print(
                f"GPU [cyan]{d}[/cyan]: [yellow]{info['used_mib']:.0f}/{info['total_mib']:.0f} MiB[/yellow] used "
                f"({info['used_pct']:.1f}%), [green]{info['effective_free_mib']:.0f} MiB[/green] effective free "
                f"(need [magenta]{info['required_mib']:.0f} MiB[/magenta] per additional instance) -> "
                f"can add [bold green]{info['additional_instances']}[/bold green] instance(s)"
            )
            for k in range(info["additional_instances"]):
                spawn_plan.append((d, info.get("active_workers", 1) + k))

        if not spawn_plan:
            console.print("[yellow]No additional GPU VRAM capacity available to add assist instances right now.[/yellow]")
            return False

        resolved_md_dir = Path(markdown_dir).resolve() if markdown_dir else find_companion_markdown_dir(input_path)
        dom_source_path = resolved_md_dir if (resolved_md_dir and resolved_md_dir.exists()) else input_path

        console.print(f"[cyan]Preparing chunk list from {dom_source_path} for reverse-order cache assist...[/cyan]")
        toc_map = extract_toc_mapping(
            input_path,
            md_dir=resolved_md_dir,
            toc_file=Path(toc_file) if toc_file else None,
        )
        doc_files = gather_page_documents(dom_source_path)
        if not doc_files and dom_source_path != input_path:
            doc_files = gather_page_documents(input_path)

        image_registry: dict = {}
        md_opts = _default_md_tts_opts()
        md_stats = {"equations": 0, "converted": 0, "errors": 0}
        chunks: List[str] = []

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
            for sc in pdom.chunks:
                chunks.append(sc.spoken_text)

        tasks_to_run = []
        for i, chunk_text in enumerate(chunks, 1):
            chunk_file = chunks_dir / f"chunk_{i:04d}.wav"
            cache_path = self.get_cache_path(chunk_text)
            if (cache_path.exists() and cache_path.stat().st_size > 44) or (
                chunk_file.exists() and chunk_file.stat().st_size > 44
            ):
                continue
            tasks_to_run.append((i, chunk_text, str(cache_path), str(chunk_file)))

        # Reverse order so assist workers synthesize from the end of the book backwards
        # and meet the forward-running primary job in the middle without overlap
        tasks_to_run.reverse()
        if not tasks_to_run:
            console.print("[bold green]All chunks are already cached/synthesized![/bold green]")
            return True

        console.print(
            f"[bold green]Spawning {len(spawn_plan)} additional Chatterbox instance(s) across GPUs to assist "
            f"with {len(tasks_to_run)} remaining uncached chunks (reverse order)...[/bold green]"
        )

        gen_kwargs = self._build_gen_kwargs()

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        for task in tasks_to_run:
            task_queue.put(task)

        workers = []
        try:
            for d, inst_idx in spawn_plan:
                p = ctx.Process(
                    target=_worker_process_fn,
                    args=(
                        d,
                        self.model_type,
                        self.reference_audio,
                        gen_kwargs,
                        task_queue,
                        result_queue,
                        inst_idx,
                        True,
                    ),
                )
                p.daemon = True
                p.start()
                workers.append(p)

            ready_workers = 0
            pending_init = len(workers)
            while pending_init > 0:
                chunk_idx, status, payload, dur, w_label = result_queue.get()
                if status == "ready":
                    pending_init -= 1
                    ready_workers += 1
                    console.print(f"Assist worker [cyan]{w_label}[/cyan] loaded and ready.")
                elif status in ("oom_init", "init_error"):
                    pending_init -= 1
                    console.print(f"[yellow]⚠ Assist worker {w_label} failed to initialize: {payload}[/yellow]")

            if ready_workers == 0:
                console.print("[red]No assist workers could be started.[/red]")
                return False

            done_count = 0
            synth_count = 0
            t0 = time.time()
            while done_count < len(tasks_to_run) and ready_workers > 0:
                chunk_idx, status, words_or_err, dur, dev = result_queue.get()
                if status == "oom_retire":
                    ready_workers -= 1
                    console.print(f"[yellow]⚠ {words_or_err}[/yellow]")
                    continue
                done_count += 1
                if status == "synthesized":
                    synth_count += 1
                    if synth_count % 25 == 0 or done_count == len(tasks_to_run):
                        elapsed = max(0.001, time.time() - t0)
                        console.print(
                            f"[Assist] Synthesized {synth_count} chunks ({done_count}/{len(tasks_to_run)} checked, "
                            f"{synth_count / elapsed:.2f} chk/s, latest: chunk_{chunk_idx:04d} on {dev})"
                        )
            console.print(f"[bold green]✓ Assist synthesis complete ({synth_count} chunks added to shared cache).[/bold green]")
            return True
        finally:
            for _ in workers:
                try:
                    task_queue.put_nowait(None)
                except Exception:
                    pass
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


def parse_args():
    parser = argparse.ArgumentParser(description="Convert documents to audiobooks & synchronized EPUB3 via Chatterbox TTS.")
    parser.add_argument("input_path", nargs="?", type=Path, default=None, help="Text/Markdown file or folder of page files.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output audiobook file (.m4b, .mp3, etc.).")
    parser.add_argument("-e", "--epub-output", type=Path, default=None, help="Output synchronized EPUB3 file (.epub).")
    parser.add_argument("--markdown-dir", type=Path, default=None, help="Optional rich MinerU Markdown directory for EPUB3 DOM.")
    parser.add_argument("--toc", type=Path, default=None, help="Optional toc.json mapping page numbers to TOC titles.")
    parser.add_argument("--title", type=str, default=None, help="Book title metadata.")
    parser.add_argument("--author", type=str, default=None, help="Book author metadata.")
    parser.add_argument("--skip-epub", "--no-epub", action="store_true", help="Skip generating synchronized EPUB3.")
    parser.add_argument("--format", type=str, default=None, choices=ChatterboxAudiobookGenerator._SUPPORTED_AUDIO_FORMATS)
    parser.add_argument("--model", type=str, default=None, choices=["standard", "turbo"], help="Chatterbox model variant.")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated device list (e.g. 'cuda:0,cuda:1').")
    parser.add_argument(
        "--auto-scale-gpu",
        dest="auto_scale_gpu",
        action="store_true",
        default=None,
        help="Automatically calculate available GPU VRAM and add extra worker instances per GPU (default: enabled).",
    )
    parser.add_argument(
        "--no-auto-scale-gpu",
        dest="auto_scale_gpu",
        action="store_false",
        help="Disable automatic multi-instance GPU scaling.",
    )
    parser.add_argument(
        "--max-instances-per-gpu",
        type=int,
        default=None,
        help="Maximum worker instances per single GPU when auto-scaling (default: config.MAX_INSTANCES_PER_GPU).",
    )
    parser.add_argument(
        "--assist",
        action="store_true",
        help="Calculate available GPU space and spawn additional reverse-order cache workers to assist an already-running conversion.",
    )
    parser.add_argument("--reference-id", type=str, default=None, help="Preset reference voice ID.")
    parser.add_argument("--reference-audio", type=str, default=None, help="Path to reference audio for voice cloning.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--cfg-weight", type=float, default=None)
    parser.add_argument("--exaggeration", type=float, default=None)
    parser.add_argument("--num-candidates", type=int, default=None, help="Number of candidates generated per chunk (default: 3).")
    parser.add_argument("--max-attempts", type=int, default=None, help="Max generation attempts per candidate on Whisper validation failure (default: 3).")
    parser.add_argument("--bypass-whisper", dest="bypass_whisper", action="store_true", default=None, help="Bypass Whisper Sync validation for faster generation.")
    parser.add_argument("--no-bypass-whisper", dest="bypass_whisper", action="store_false", help="Enable Whisper Sync validation (default).")
    parser.add_argument("--whisper-model", type=str, default=None, choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size for chunk validation (default: base).")
    parser.add_argument("--whisper-device", type=str, default=None, help="Device for Whisper validation (e.g. 'cpu', 'cuda', 'cuda:2').")
    parser.add_argument("--use-faster-whisper", dest="use_faster_whisper", action="store_true", default=None, help="Use faster-whisper backend (default).")
    parser.add_argument("--use-openai-whisper", dest="use_faster_whisper", action="store_false", help="Use openai-whisper backend.")
    parser.add_argument("--longest-on-fail", dest="use_longest_transcript_on_fail", action="store_true", default=None, help="Pick candidate with longest transcript if all candidates fail Whisper check.")
    parser.add_argument("--highest-score-on-fail", dest="use_longest_transcript_on_fail", action="store_false", help="Pick candidate with highest fuzzy score if all candidates fail Whisper check.")
    parser.add_argument("--auto-editor", dest="use_auto_editor", action="store_true", default=None, help="Post-process chunks with auto-editor (default: enabled).")
    parser.add_argument("--no-auto-editor", dest="use_auto_editor", action="store_false", help="Disable auto-editor post-processing.")
    parser.add_argument("--ae-threshold", type=float, default=None, help="auto-editor audio volume threshold (default: 0.06).")
    parser.add_argument("--ae-margin", type=float, default=None, help="auto-editor buffer margin in seconds (default: 0.2).")
    parser.add_argument("--keep-original-wav", action="store_true", default=None, help="Save pre-auto-editor original chunk WAV alongside cleaned WAV.")
    parser.add_argument("--smart-append-short", dest="smart_append_short", action="store_true", default=None, help="Merge sentences < 20 chars with neighboring sentences (default: enabled).")
    parser.add_argument("--no-smart-append-short", dest="smart_append_short", action="store_false", help="Disable merging sentences < 20 chars.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for deterministic candidate generation.")
    parser.add_argument("--per-page", action="store_true", help="Also export individual audio files per page.")
    parser.add_argument("--clean-chunks", action="store_true", help="Delete intermediate chunk WAV files after successful conversion.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass audio chunk cache.")
    parser.add_argument("--list-references", action="store_true", help="List available reference voices.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_references:
        ref_dir = Path(config.REFERENCES_FOLDER)
        print(f"Available reference voices in {ref_dir}:")
        if ref_dir.exists():
            for d in sorted(ref_dir.iterdir()):
                if d.is_dir():
                    wavs = list(d.glob("*.wav"))
                    print(f"  • {d.name} ({len(wavs)} audio file(s))")
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
        model_type=args.model,
        devices=devices,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration,
        no_cache=args.no_cache,
        auto_scale_gpu=args.auto_scale_gpu,
        max_instances_per_gpu=args.max_instances_per_gpu,
        num_candidates=args.num_candidates,
        max_attempts=args.max_attempts,
        bypass_whisper_checking=args.bypass_whisper,
        use_faster_whisper=args.use_faster_whisper,
        whisper_model=args.whisper_model,
        whisper_device=args.whisper_device,
        use_longest_transcript_on_fail=args.use_longest_transcript_on_fail,
        use_auto_editor=args.use_auto_editor,
        ae_threshold=args.ae_threshold,
        ae_margin=args.ae_margin,
        keep_original_wav=args.keep_original_wav,
        smart_append_short=args.smart_append_short,
        seed=args.seed,
    )

    try:
        if args.assist:
            success = generator.assist_active_conversion(
                input_path=input_path,
                markdown_dir=args.markdown_dir,
                toc_file=args.toc,
            )
        else:
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

