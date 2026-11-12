#!/usr/bin/env python3
"""
chatterbox_qa.py - artifact-reduction helpers for audiobook_tts.py

Drop next to audiobook_tts.py. Provides:

  normalize_for_tts()     strip markdown/OCR/URL/footnote noise before synthesis
  chunk_text()            sentence-first chunker with no tiny fragments and
                          abbreviation/dialogue-aware sentence boundaries
  validate_chunk()        objective QA gate on raw model output (rate, pauses,
                          clipping, abrupt end, optional ASR round-trip)
  synthesize_with_qa()    seeded generate -> validate -> retry with safer params
  atomic_write_wav() /
  read_cached()           crash-safe cache (no truncated WAVs on resume)
  postprocess_chunk()     DC/rumble removal, edge trim, active-speech gain
                          (apply AFTER the cache so it can be re-tuned freely)
"""
from __future__ import annotations

import difflib
import hashlib
import inspect
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    from scipy.signal import butter, sosfiltfilt
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

TARGET_WPS = 2.8  # rough centre of Chatterbox narration rate; used only to rank failed retries

# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
_BULLETS = re.compile(r"[\u2022\u25aa\u25e6\u25a0\u25a1\u25cf\u25cb\u25ba\u25b6\u2192\u2190\u25c6\u25c7\u2605\u2606]")
_FOOTNOTE = re.compile(r"\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")


def normalize_for_tts(text: str, drop_math: bool = True) -> str:
    """Remove things Chatterbox mispronounces or hallucinates on. Keeps paragraph breaks."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u00ad", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)                 # hyphenated line wraps
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)            # markdown images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)         # markdown links -> label
    t = re.sub(r"<[^>]+>", " ", t)                         # html tags
    if drop_math:
        t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
        t = re.sub(r"\$[^$\n]+\$", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " link ", t)
    t = re.sub(r"\S+@\S+\.\S+", " email address ", t)
    t = _FOOTNOTE.sub("", t)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)    # headings
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)         # list markers
    t = re.sub(r"[*_`~|]+", "", t)
    t = _BULLETS.sub(" ", t)
    t = re.sub(r"\.{2,}", ", ", t)
    t = re.sub(r"([!?,;:])\1+", r"\1", t)
    t = re.sub(r"[ \t]+([,.;:!?])", r"\1", t)
    t = re.sub(r"[ \t]*\n[ \t]*\n[\s]*", "\n\n", t)
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)                 # hard-wrapped lines -> spaces
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "fig", "eq",
    "no", "vol", "al", "inc", "ltd", "co", "e.g", "i.e", "u.s", "u.k", "approx",
    "dept", "est", "cf", "ch", "sec", "pp", "ed",
}
_BOUNDARY = re.compile(r'[.!?]["\u201d\u2019)\]]*\s+')


def _wc(s: str) -> int:
    return len(s.split())


def split_sentences(par: str) -> List[str]:
    out, start = [], 0
    for m in _BOUNDARY.finditer(par):
        end = m.end()
        nxt = par[end:end + 1]
        if nxt and nxt.islower():          # '"Stop," she said.' / 'etc. and' -> not a boundary
            continue
        if par[m.start()] == ".":
            toks = par[start:m.start()].split()
            tok = toks[-1].lower().strip("\"\u201c\u2018([") if toks else ""
            if tok in _ABBREV or (len(tok) == 1 and tok.isalpha()):
                continue
        out.append(par[start:end].strip())
        start = end
    tail = par[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


def _balanced(words: List[str], max_words: int) -> List[str]:
    """Split into equal-size pieces so we never emit a 1-2 word tail."""
    n = math.ceil(len(words) / max_words)
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def _split_long(s: str, max_words: int, min_words: int) -> List[str]:
    if _wc(s) <= max_words:
        return [s]
    clauses = [c for c in re.split(r"(?<=[,;:\u2014\u2013])\s+", s) if c.strip()]
    pieces, cur = [], ""
    for c in clauses:
        if _wc(c) > max_words:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_balanced(c.split(), max_words))
        elif cur and _wc(cur) + _wc(c) > max_words:
            pieces.append(cur)
            cur = c
        else:
            cur = f"{cur} {c}".strip()
    if cur:
        pieces.append(cur)
    out: List[str] = []
    for p in pieces:
        if out and (_wc(p) < min_words or _wc(out[-1]) < min_words):
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def _pack(units: List[str], max_words: int, min_words: int, pack: bool) -> List[str]:
    out: List[str] = []
    carry = ""
    for u in units:
        if carry:
            u, carry = f"{carry} {u}", ""
        if out and (_wc(u) < min_words or (pack and _wc(out[-1]) + _wc(u) <= max_words)):
            out[-1] = f"{out[-1]} {u}"
        elif not out and _wc(u) < min_words:
            carry = u
        else:
            out.append(u)
    if carry:
        out.append(carry)  # paragraph that is itself tiny (e.g. a heading); QA gate covers it
    return out


def _finish(c: str) -> str:
    c = c.strip()
    c = re.sub(r"[;:\u2014\u2013-]+$", ",", c)
    letters = [ch for ch in c if ch.isalpha()]
    if len(letters) >= 8 and sum(ch.isupper() for ch in letters) / len(letters) > 0.7:
        c = c.lower().capitalize()                    # ALL-CAPS headings get shouted/spelled out
    if c and c[-1] not in '.!?,"\u201d\u2019)':
        c += "."
    return c


def chunk_text(text: str, max_words: int = 40, min_words: int = 5, pack_sentences: bool = False) -> List[str]:
    """
    Sentence-first chunks, never crossing paragraph breaks.
    pack_sentences=False keeps one-sentence-per-chunk (needed if SMIL highlighting is
    sentence-aligned); True packs short neighbours up to max_words (fewer joins).
    """
    chunks: List[str] = []
    for par in re.split(r"\n{2,}", text):
        par = par.strip()
        if not par:
            continue
        units: List[str] = []
        for s in split_sentences(par):
            units.extend(_split_long(s, max_words, min_words))
        chunks.extend(_pack(units, max_words, min_words, pack_sentences))
    return [c for c in (_finish(x) for x in chunks) if c]


# --------------------------------------------------------------------------- #
# Audio analysis
# --------------------------------------------------------------------------- #
_FRAME_MS = 10


def _frame_rms(x: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    win = max(1, int(sr * _FRAME_MS / 1000))
    n = len(x) // win
    if n == 0:
        return np.zeros(0, dtype=np.float32), win
    fr = x[: n * win].reshape(n, win)
    return np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12), win


def active_span(x: np.ndarray, sr: int, floor_db: float = -42.0) -> Tuple[int, int, float]:
    rms, win = _frame_rms(x, sr)
    if len(rms) < 3:
        return 0, len(x), 0.0
    ref = float(np.percentile(rms, 85))
    if ref < 1e-5:
        return 0, len(x), ref
    idx = np.flatnonzero(rms > ref * 10 ** (floor_db / 20))
    if idx.size == 0:
        return 0, len(x), ref
    return int(idx[0] * win), min(len(x), int((idx[-1] + 1) * win)), ref


# --------------------------------------------------------------------------- #
# QA gate
# --------------------------------------------------------------------------- #
@dataclass
class QAResult:
    ok: bool
    hard: List[str] = field(default_factory=list)
    soft: List[str] = field(default_factory=list)
    wps: float = 0.0
    speech_s: float = 0.0
    similarity: Optional[float] = None


def make_asr_checker(model_size: str = "small.en", device: str = "cpu", compute_type: str = "int8"):
    """Optional round-trip check. Returns callable(x, sr, text)->similarity 0..1, or None if unavailable."""
    try:
        import torch
        import torchaudio
        from faster_whisper import WhisperModel
    except Exception:
        return None
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def _norm(s: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", s.lower())

    def check(x: np.ndarray, sr: int, text: str) -> float:
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        a = torchaudio.functional.resample(t, sr, 16000).numpy()
        segs, _ = model.transcribe(a, language="en", beam_size=1, condition_on_previous_text=False)
        heard = _norm(" ".join(s.text for s in segs))
        want = _norm(text)
        return 1.0 if not want else difflib.SequenceMatcher(None, want, heard).ratio()

    return check


def validate_chunk(
    wav,
    sr: int,
    text: str,
    *,
    min_wps: float = 1.5,
    max_wps: float = 4.6,
    max_internal_pause_s: float = 1.2,
    max_clip_frac: float = 1e-3,
    asr: Optional[Callable] = None,
    asr_min_similarity: float = 0.85,
) -> QAResult:
    """
    Run on RAW model output (before trimming). Bounds are deliberately loose
    'obvious failure' limits; log wps for a few hundred chunks and tighten.
    """
    x = np.asarray(wav, dtype=np.float32).squeeze()
    if x.size == 0 or not np.all(np.isfinite(x)):
        return QAResult(False, ["empty_or_nonfinite"])
    if float(np.max(np.abs(x))) < 1e-3:
        return QAResult(False, ["silent"])

    hard: List[str] = []
    soft: List[str] = []

    if float(np.mean(np.abs(x) >= 0.999)) > max_clip_frac:
        hard.append("clipped")

    s, e, ref = active_span(x, sr)
    speech_s = max((e - s) / sr, 1e-3)
    words = max(1, len(re.findall(r"\w+", text)))
    wps = words / speech_s
    if wps < min_wps:
        hard.append(f"too_slow({wps:.2f}wps)")      # babble, stuck/looping, runaway generation
    if wps > max_wps:
        hard.append(f"too_fast({wps:.2f}wps)")      # skipped words / truncation

    rms, win = _frame_rms(x, sr)
    if rms.size and ref > 0:
        active = rms[s // win: max(s // win + 1, e // win)] > ref * 10 ** (-42 / 20)
        run = best = 0
        for a in active:
            run = 0 if a else run + 1
            best = max(best, run)
        if best * _FRAME_MS / 1000 > max_internal_pause_s:
            hard.append(f"long_internal_pause({best * _FRAME_MS / 1000:.1f}s)")

    tail = x[-int(sr * 0.08):]
    speech_rms = float(np.sqrt(np.mean(x[s:e] ** 2) + 1e-12))
    if float(np.sqrt(np.mean(tail ** 2))) > 0.35 * speech_rms:
        soft.append("abrupt_end")                   # logged only; tighten if you want it to retry

    sim = None
    if asr is not None and not hard and not re.search(r"\d", text):
        sim = float(asr(x, sr, text))
        if sim < asr_min_similarity:
            hard.append(f"asr_mismatch({sim:.2f})")

    return QAResult(not hard, hard, soft, wps, speech_s, sim)


# --------------------------------------------------------------------------- #
# Seeded synthesis with retries
# --------------------------------------------------------------------------- #
def seed_for(text: str, attempt: int) -> int:
    return int(hashlib.md5(f"{text}|{attempt}".encode("utf-8")).hexdigest()[:8], 16)


def attempt_kwargs(base: dict, attempt: int, model_type: str) -> dict:
    kw = dict(base)
    if attempt == 0:
        return kw
    kw["temperature"] = max(0.4, float(kw.get("temperature", 0.8)) * (0.9 ** attempt))
    if model_type != "turbo":
        if attempt >= 2:
            kw["cfg_weight"] = min(0.7, float(kw.get("cfg_weight", 0.5)) + 0.05 * attempt)
        if attempt >= 3:
            kw["repetition_penalty"] = min(1.5, float(kw.get("repetition_penalty", 1.2)) + 0.1)
    return kw


def synthesize_with_qa(
    model,
    text: str,
    base_kwargs: dict,
    model_type: str,
    *,
    max_attempts: int = 4,
    asr: Optional[Callable] = None,
    qa_kwargs: Optional[dict] = None,
) -> Tuple[np.ndarray, QAResult, int]:
    """Returns (raw_wav_np, QAResult, attempts_used). If every attempt fails, returns the least-bad one (ok=False)."""
    import torch

    params = inspect.signature(model.generate).parameters  # drop kwargs this model build doesn't accept
    best = None
    for attempt in range(max_attempts):
        kw = {k: v for k, v in attempt_kwargs(base_kwargs, attempt, model_type).items() if k in params}
        torch.manual_seed(seed_for(text, attempt))
        wav = model.generate(text, audio_prompt_path=None, **kw)
        x = wav.squeeze().detach().cpu().numpy().astype(np.float32)
        qa = validate_chunk(x, model.sr, text, asr=asr, **(qa_kwargs or {}))
        if qa.ok:
            return x, qa, attempt + 1
        score = (len(qa.hard), abs(math.log(max(qa.wps, 1e-3) / TARGET_WPS)))
        if best is None or score < best[0]:
            best = (score, x, qa)
    return best[1], best[2], max_attempts


def log_flagged(path: Path, chunk_idx: int, text: str, qa: QAResult, attempts: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"chunk": chunk_idx, "text": text, "hard": qa.hard, "soft": qa.soft,
           "wps": round(qa.wps, 2), "attempts": attempts}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Crash-safe cache
# --------------------------------------------------------------------------- #
def atomic_write_wav(path: Path, wav: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        sf.write(str(tmp), wav, sr, subtype=subtype, format="WAV")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _riff_intact(path: Path) -> bool:
    """libsndfile silently accepts truncated WAVs, so compare the RIFF header size to the real file size."""
    with open(path, "rb") as f:
        hdr = f.read(12)
    if len(hdr) < 12 or hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
        return False
    return int.from_bytes(hdr[4:8], "little") + 8 <= path.stat().st_size


def read_cached(path: Path) -> Optional[Tuple[np.ndarray, int]]:
    """Return (wav, sr) or None. Deletes unreadable/truncated/non-finite entries."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        if not _riff_intact(path):
            raise ValueError("truncated or non-WAV cache entry")
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("bad cache entry")
        return x, sr
    except Exception:
        path.unlink(missing_ok=True)
        return None


# --------------------------------------------------------------------------- #
# Post-processing (apply after cache read, not before cache write)
# --------------------------------------------------------------------------- #
def postprocess_chunk(
    wav,
    sr: int,
    *,
    target_rms: float = 0.10,       # -20 dBFS
    min_gain: float = 0.5,
    max_gain: float = 2.0,          # was 4.0: +12 dB lifts the noise floor on quiet chunks
    peak_ceiling: float = 0.89125,  # -1 dBFS
) -> np.ndarray:
    x = np.nan_to_num(np.asarray(wav, dtype=np.float32).squeeze(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < int(sr * 0.08):
        return x

    x = x - float(np.mean(x))                                    # DC offset -> boundary clicks
    if _HAVE_SCIPY:
        x = sosfiltfilt(butter(2, 70, "hp", fs=sr, output="sos"), x).astype(np.float32)  # rumble

    s, e, _ = active_span(x, sr)
    s = max(0, s - int(sr * 0.020))
    e = min(len(x), e + int(sr * 0.060))                         # longer tail pad keeps natural decay
    if e - s >= int(sr * 0.08):
        x = x[s:e].copy()

    fi = min(int(sr * 0.008), len(x) // 4)
    fo = min(int(sr * 0.025), len(x) // 4)
    if fi > 1:
        x[:fi] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(fi, dtype=np.float32) / fi))
    if fo > 1:
        x[-fo:] *= (0.5 * (1.0 - np.cos(np.pi * np.arange(fo, dtype=np.float32) / fo)))[::-1]

    # Gain from ACTIVE speech frames only, so pause length doesn't change loudness
    frm, _ = _frame_rms(x, sr)
    if frm.size:
        ref = float(np.percentile(frm, 85))
        mask = frm > ref * 10 ** (-30 / 20)
        speech_rms = float(np.sqrt(np.mean(frm[mask] ** 2))) if mask.any() else float(np.sqrt(np.mean(x ** 2)))
    else:
        speech_rms = float(np.sqrt(np.mean(x ** 2)))
    if speech_rms > 1e-4:
        x = x * float(np.clip(target_rms / speech_rms, min_gain, max_gain))

    peak = float(np.max(np.abs(x)))
    if peak > peak_ceiling:
        x = x * (peak_ceiling / peak)
    return x.astype(np.float32)