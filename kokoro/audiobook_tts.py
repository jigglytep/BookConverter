#!/usr/bin/env python3
"""
Kokoro-82M Audiobook TTS Stage

Reads already-extracted/cleaned text (a single .txt/.md file, or a directory
of page files such as TTS_Script/out/<Book>/<Book>_page_0001.txt) and
synthesizes it into an audiobook using the local Kokoro-82M model.

This is intentionally just the TTS stage - PDF/EPUB extraction and math/number
cleanup are handled upstream by marker/mineru and TTS_Script/md_tts.py.
"""

import argparse
import hashlib
import logging
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
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

import config

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


class KokoroAudiobookGenerator:
    """Text -> audiobook using a local Kokoro-82M pipeline."""

    def __init__(self, voice: Optional[str] = None, lang_code: Optional[str] = None,
                 speed: Optional[float] = None, device: Optional[str] = None, workers: int = 1):
        self.voice = voice or config.VOICE
        self.speed = speed if speed is not None else config.SPEED
        self.lang_code = lang_code or config.LANG_CODE or self._infer_lang_code(self.voice)
        self.device = device or config.DEVICE
        self.workers = max(1, workers)

        self.setup_logging()
        self.setup_directories()
        self.pipelines: List = []
        self._pipeline_pool: "queue.Queue" = queue.Queue()
        self._cache_lock = threading.Lock()
        self.init_pipeline()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup_logging(self):
        Path(config.LOGS_FOLDER).mkdir(exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(f"{config.LOGS_FOLDER}/audiobook_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def setup_directories(self):
        for directory in (config.AUDIOBOOKS_FOLDER, config.CHUNKS_FOLDER, config.CACHE_FOLDER, config.LOGS_FOLDER):
            Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _infer_lang_code(voice: str) -> str:
        for lang, prefixes in config.VOICE_LANG_PREFIXES.items():
            if voice.startswith(prefixes):
                return lang
        return "a"

    _CUDA_DEVICE_RE = re.compile(r"^cuda(:(\d+))?$")
    _SUPPORTED_AUDIO_FORMATS = {"m4b", "mp3", "wav", "m4a", "flac"}

    @staticmethod
    def _cuda_free_memory() -> List[Tuple[int, int]]:
        """(gpu_index, free_bytes) for every visible CUDA device, most free first."""
        try:
            import torch
            if not torch.cuda.is_available():
                return []
        except ImportError:
            return []

        ranked = []
        for i in range(torch.cuda.device_count()):
            try:
                free, _total = torch.cuda.mem_get_info(i)
            except RuntimeError:
                continue
            ranked.append((i, free))
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    def _resolve_device_candidates(self, requested: Optional[str], min_free_gb: float = 1.5) -> List[str]:
        """Turn the requested --device value into an ordered list of devices to
        try. A specific index (cuda:1) is tried as-is with a CPU fallback. A bare
        "cuda"/None prefers GPU 0 and only overflows to another GPU when GPU 0
        doesn't have min_free_gb free (or when GPU 0 itself OOMs on load)."""
        if requested == "cpu":
            return ["cpu"]

        match = self._CUDA_DEVICE_RE.match(requested) if requested else None
        if match and match.group(2) is not None:
            idx = int(match.group(2))
            try:
                import torch
                available = torch.cuda.is_available() and idx < torch.cuda.device_count()
            except ImportError:
                available = False
            if available:
                return [requested, "cpu"]
            self.logger.warning(f"{requested} requested but not available, falling back to auto GPU selection")

        ranked = self._cuda_free_memory()
        if not ranked:
            if requested == "cuda":
                self.logger.warning("CUDA requested but not available, falling back to CPU")
            return ["cpu"]

        min_free_bytes = min_free_gb * 1e9
        gpu0_free = next((free for i, free in ranked if i == 0), None)
        others = [f"cuda:{i}" for i, _free in ranked if i != 0]

        if gpu0_free is not None and gpu0_free >= min_free_bytes:
            candidates = ["cuda:0"] + others
        else:
            if gpu0_free is not None:
                self.logger.info(
                    f"GPU 0 has {gpu0_free / 1e9:.1f}GB free (<{min_free_gb}GB), trying other GPUs first"
                )
            candidates = others + (["cuda:0"] if gpu0_free is not None else [])

        return candidates + ["cpu"]

    @staticmethod
    def _all_cuda_devices() -> List[str]:
        """Every visible CUDA device, in index order."""
        try:
            import torch
            if not torch.cuda.is_available():
                return []
        except ImportError:
            return []
        return [f"cuda:{i}" for i in range(torch.cuda.device_count())]

    def _resolve_worker_specs(self, requested: Optional[str], workers: int) -> List[List[str]]:
        """Return one device fallback-chain per worker to load. "all" spreads
        `workers` model instances across every visible GPU (round-robin); a
        comma-separated list (e.g. "cuda:0,cuda:1") does the same across just
        those devices; any other --device value loads `workers` instances
        using that value's normal single-device resolution (e.g. all stacked
        on the same GPU)."""
        if requested == "all":
            devices = self._all_cuda_devices()
            if not devices:
                self.logger.warning("--device all requested but no CUDA GPUs are visible, falling back to CPU")
                return [["cpu"] for _ in range(workers)]
            return [[dev, "cpu"] for _ in range(workers) for dev in devices]

        if requested and "," in requested:
            tokens = [t.strip() for t in requested.split(",") if t.strip()]
            return [
                self._resolve_device_candidates(token)
                for _ in range(workers)
                for token in tokens
            ]

        chain = self._resolve_device_candidates(requested)
        return [chain for _ in range(workers)]

    def _load_pipeline(self, KPipeline, candidates: List[str]):
        last_err = None
        for i, device in enumerate(candidates):
            is_last = i == len(candidates) - 1
            try:
                try:
                    return KPipeline(lang_code=self.lang_code, device=device), device
                except TypeError:
                    # Older kokoro releases don't accept a device kwarg.
                    return KPipeline(lang_code=self.lang_code), device
            except RuntimeError as e:
                last_err = e
                if not is_last and "out of memory" in str(e).lower():
                    self.logger.warning(f"{device} out of memory loading Kokoro-82M, trying next device...")
                    if device.startswith("cuda"):
                        import torch
                        torch.cuda.empty_cache()
                    continue
                raise
        raise last_err

    def init_pipeline(self):
        try:
            from kokoro import KPipeline
        except ImportError:
            console.print("[bold red]✗ The 'kokoro' package is not installed.[/]")
            console.print("Install it with: pip install -r requirements.txt")
            sys.exit(1)

        worker_chains = self._resolve_worker_specs(self.device, self.workers)
        self.logger.info(
            f"Loading Kokoro-82M ({len(worker_chains)} worker(s), lang={self.lang_code}, voice={self.voice}, "
            f"chains={worker_chains})..."
        )
        try:
            loaded_devices = []
            with console.status(
                f"[bold green]Loading Kokoro-82M[/] ({len(worker_chains)} worker(s), voice={self.voice}, "
                f"lang={self.lang_code})... [dim](first run downloads the model)[/]",
                spinner="dots",
            ):
                for chain in worker_chains:
                    pipeline, device = self._load_pipeline(KPipeline, chain)
                    self.pipelines.append(pipeline)
                    self._pipeline_pool.put(pipeline)
                    loaded_devices.append(device)

            counts: dict = {}
            for d in loaded_devices:
                counts[d] = counts.get(d, 0) + 1
            self.device = ", ".join(f"{d}×{n}" if n > 1 else d for d, n in counts.items())
            console.print(f"[bold green]✓[/] Kokoro-82M loaded  [dim](voice={self.voice}, lang={self.lang_code}, devices={self.device})[/]")
        except Exception as e:
            console.print("[bold red]✗ Failed to load Kokoro-82M[/]")
            console.print(f"[red]Error: {e}[/]")
            console.print("Make sure the 'kokoro' package and its dependencies (torch, espeak-ng for")
            console.print("non-English languages) are installed, and that the voice/lang combination is valid.")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Text loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _natural_sort_key(path: Path):
        nums = [int(n) for n in re.findall(r"\d+", path.stem)]
        return (nums, path.name)

    def _read_text_file(self, file_path: Path) -> str:
        raw = None
        for encoding in ("utf-8", "utf-16", "latin-1", "cp1252"):
            try:
                raw = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            self.logger.warning(f"Could not decode {file_path}, skipping")
            return ""

        if file_path.suffix.lower() == ".md":
            raw = self._strip_markdown(raw)

        return self._normalize_whitespace(raw)

    @staticmethod
    def _strip_markdown(text: str) -> str:
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _page_label(path: Path) -> str:
        nums = re.findall(r"\d+", path.stem)
        if nums:
            return f"Page {int(nums[-1])}"
        return path.stem

    def load_book_pages(self, input_path: Path) -> List[Tuple[str, str, str]]:
        """Return a list of (chapter_label, text, source_stem) triples, one per
        source page file (or a single entry for a plain text/markdown file)."""
        if input_path.is_file():
            text = self._read_text_file(input_path)
            if not text.strip():
                raise ValueError(f"No text extracted from {input_path}")
            return [(input_path.stem, text, input_path.stem)]

        if input_path.is_dir():
            files = sorted(
                (f for f in input_path.iterdir()
                 if f.is_file() and f.suffix.lower() in config.PAGE_FILE_EXTENSIONS),
                key=self._natural_sort_key,
            )
            if not files:
                raise ValueError(f"No {config.PAGE_FILE_EXTENSIONS} files found in {input_path}")
            self.logger.info(f"Found {len(files)} page file(s) in {input_path.name}")

            pages = [(self._page_label(f), self._read_text_file(f), f.stem) for f in files]
            pages = [(label, text, stem) for label, text, stem in pages if text.strip()]
            if not pages:
                raise ValueError(f"No text extracted from files in {input_path}")
            return pages

        raise FileNotFoundError(input_path)

    # ------------------------------------------------------------------ #
    # Chunking
    # ------------------------------------------------------------------ #

    def split_into_chunks(self, text: str) -> List[str]:
        if not text.strip():
            return []

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current_chunk = ""
        current_words = 0
        limit = config.CHUNK_SIZE_WORDS

        for sentence in sentences:
            sentence_words = len(sentence.split())

            if sentence_words > limit:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_words = 0
                parts = re.split(r"[,;:]", sentence)
                for part in parts:
                    part_words = len(part.split())
                    if current_words + part_words <= limit:
                        current_chunk += part + " "
                        current_words += part_words
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + " "
                        current_words = part_words
            else:
                if current_words + sentence_words <= limit:
                    current_chunk += sentence + " "
                    current_words += sentence_words
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                    current_words = sentence_words

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #

    def get_cache_path(self, text: str) -> Path:
        content = f"{text}_{self.voice}_{self.lang_code}_{self.speed}"
        digest = hashlib.md5(content.encode()).hexdigest()
        return Path(config.CACHE_FOLDER) / f"{digest}.wav"

    def _synthesize(self, text: str, pipeline) -> np.ndarray:
        """Run Kokoro on a chunk of text and return concatenated float32 audio."""
        generator = pipeline(text, voice=self.voice, speed=self.speed)
        segments = []
        for result in generator:
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            segments.append(np.asarray(audio, dtype=np.float32))

        if not segments:
            raise RuntimeError("Kokoro produced no audio for this chunk")

        if len(segments) == 1:
            return segments[0]

        gap = np.zeros(int(config.SAMPLE_RATE * config.SUBSEGMENT_GAP_MS / 1000), dtype=np.float32)
        parts = []
        for i, seg in enumerate(segments):
            if i > 0:
                parts.append(gap)
            parts.append(seg)
        return np.concatenate(parts)

    def generate_chunk(self, text: str, chunk_num: int, chunks_dir: Path) -> Optional[str]:
        try:
            cache_path = self.get_cache_path(text)
            output_path = chunks_dir / f"chunk_{chunk_num:04d}.wav"

            if cache_path.exists():
                shutil.copy2(cache_path, output_path)
                self.logger.debug(f"Using cached audio for chunk {chunk_num}")
                return str(output_path)

            pipeline = self._pipeline_pool.get()
            try:
                audio = self._synthesize(text, pipeline)
            finally:
                self._pipeline_pool.put(pipeline)
            sf.write(str(output_path), audio, config.SAMPLE_RATE)
            with self._cache_lock:
                shutil.copy2(output_path, cache_path)

            self.logger.debug(f"Chunk {chunk_num} generated successfully")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"Kokoro synthesis failed for chunk {chunk_num}: {e}")
            return None

    def process_chunk_with_retry(self, chunk_num: int, text: str, chunks_dir: Path) -> bool:
        for attempt in range(config.MAX_RETRIES):
            try:
                result = self.generate_chunk(text, chunk_num, chunks_dir)
                if result and Path(result).exists():
                    return True
                self.logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} failed")
            except Exception as e:
                self.logger.warning(f"Chunk {chunk_num} attempt {attempt + 1} error: {e}")

            if attempt < config.MAX_RETRIES - 1:
                time.sleep(1 + attempt)

        self.logger.error(f"Chunk {chunk_num} failed after {config.MAX_RETRIES} attempts")
        return False

    @staticmethod
    def _book_stem(input_path: Path) -> str:
        return input_path.stem if input_path.is_file() else input_path.name

    def _chunks_dir(self, input_path: Path) -> Path:
        return Path(config.CHUNKS_FOLDER) / self._book_stem(input_path)

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #

    def combine_chunks(self, total_chunks: int, output_path: Path, results: dict,
                        chunk_page_index: List[int], page_labels: List[str],
                        chunks_dir: Path, book_title: str = "Audiobook") -> bool:
        tmp_raw = chunks_dir / "_combined.raw"
        try:
            successful = 0
            missing_chunks = []
            chapters: List[Tuple[str, int, int]] = []
            current_page_idx = None
            page_start_ms = 0
            total_frames = 0
            sample_width = 2
            channels = 1
            frame_rate = config.SAMPLE_RATE

            gap_frames = int(frame_rate * config.CHUNK_GAP_MS / 1000) if config.CHUNK_GAP_MS > 0 else 0
            gap_bytes = b""

            chunks_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp_raw, "wb") as raw_out:
                for i in range(1, total_chunks + 1):
                    if not results.get(i, False):
                        missing_chunks.append(i)
                        continue

                    chunk_file = chunks_dir / f"chunk_{i:04d}.wav"
                    if not chunk_file.exists():
                        self.logger.warning(f"Chunk file not found: {chunk_file}")
                        missing_chunks.append(i)
                        continue

                    try:
                        with wave.open(str(chunk_file), "rb") as wf:
                            channels = wf.getnchannels()
                            sample_width = wf.getsampwidth()
                            frame_rate = wf.getframerate()
                            nframes = wf.getnframes()
                            chunk_raw = wf.readframes(nframes)
                    except Exception as e:
                        self.logger.warning(f"Failed to load chunk {i}: {e}")
                        missing_chunks.append(i)
                        continue

                    if gap_frames > 0 and not gap_bytes:
                        gap_bytes = b"\x00" * (gap_frames * channels * sample_width)

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
                    if successful % 50 == 0:
                        self.logger.info(f"Combined {successful}/{total_chunks} chunks")

                if current_page_idx is not None:
                    final_ms = int(total_frames * 1000 / frame_rate)
                    chapters.append((page_labels[current_page_idx], page_start_ms, final_ms))

            if successful == 0:
                raise RuntimeError("No valid chunks found")

            if missing_chunks:
                self.logger.warning(f"Missing chunks: {missing_chunks}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Honor an explicit -o extension over config.AUDIO_FORMAT: encoding to
            # config's format while ffmpeg muxes into a container implied by a
            # different -o extension produces an invalid file (e.g. an AAC stream
            # inside an .mp3, which ffmpeg's mp3 muxer rejects).
            requested_fmt = output_path.suffix.lstrip(".").lower()
            if requested_fmt in self._SUPPORTED_AUDIO_FORMATS:
                fmt = requested_fmt
            else:
                if requested_fmt:
                    self.logger.warning(
                        f"Unrecognized output extension '.{requested_fmt}', "
                        f"using configured format '.{config.AUDIO_FORMAT}' instead"
                    )
                fmt = config.AUDIO_FORMAT.lower()

            if fmt == "m4b":
                self._export_m4b(tmp_raw, output_path, chapters, book_title, chunks_dir,
                                 sample_width=sample_width, frame_rate=frame_rate, channels=channels)
            else:
                bitrate = config.AUDIO_BITRATE if fmt in ("mp3", "m4a") else None
                self._export_via_ffmpeg(tmp_raw, output_path, fmt,
                                        sample_width=sample_width, frame_rate=frame_rate,
                                        channels=channels, bitrate=bitrate)
                if chapters:
                    self.logger.info(
                        f"{len(chapters)} page(s) tracked, but chapters are only embedded in m4b output "
                        "(set AUDIO_FORMAT = \"m4b\" in config.py to enable them)"
                    )

            self.logger.info(
                f"Audiobook saved: {output_path} ({successful}/{total_chunks} chunks, {len(chapters)} chapters)"
            )
            if missing_chunks:
                console.print(f"[yellow]⚠ Missing chunks: {missing_chunks}[/]")
            return True
        except Exception as e:
            self.logger.error(f"Failed to combine chunks: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
        finally:
            tmp_raw.unlink(missing_ok=True)

    def export_per_page(self, total_chunks: int, results: dict, chunk_page_index: List[int],
                         page_labels: List[str], page_stems: List[str],
                         output_dir: Path, fmt: str, chunks_dir: Path) -> Tuple[int, List[str]]:
        """Export one audio file per page instead of a single combined book file."""
        if fmt == "m4b":
            self.logger.warning("m4b chapters only make sense for a combined file; using mp3 for per-page export")
            fmt = "mp3"

        output_dir.mkdir(parents=True, exist_ok=True)
        sample_width = 2
        channels = 1
        frame_rate = config.SAMPLE_RATE
        gap_frames = int(frame_rate * config.CHUNK_GAP_MS / 1000) if config.CHUNK_GAP_MS > 0 else 0
        gap_bytes = b""

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
                if not results.get(i, False):
                    continue
                chunk_file = chunks_dir / f"chunk_{i:04d}.wav"
                if not chunk_file.exists():
                    continue
                try:
                    with wave.open(str(chunk_file), "rb") as wf:
                        channels = wf.getnchannels()
                        sample_width = wf.getsampwidth()
                        frame_rate = wf.getframerate()
                        chunk_raw = wf.readframes(wf.getnframes())
                except Exception as e:
                    self.logger.warning(f"Failed to load chunk {i} for page {page_labels[page_idx]}: {e}")
                    continue
                if gap_frames > 0 and not gap_bytes:
                    gap_bytes = b"\x00" * (gap_frames * channels * sample_width)
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

        if failed_pages:
            self.logger.warning(f"Pages with no audio: {failed_pages}")

        self.logger.info(f"Exported {successful_pages}/{len(chunks_by_page)} page(s) to {output_dir}")
        return successful_pages, failed_pages

    @staticmethod
    def _ffmetadata_escape(text: str) -> str:
        return re.sub(r"([=;#\\\n])", r"\\\1", text)

    def _write_ffmetadata(self, path: Path, chapters: List[Tuple[str, int, int]], book_title: str):
        lines = [";FFMETADATA1", f"title={self._ffmetadata_escape(book_title)}", ""]
        for title, start_ms, end_ms in chapters:
            if end_ms <= start_ms:
                continue
            lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={self._ffmetadata_escape(title)}",
                "",
            ]
        path.write_text("\n".join(lines), encoding="utf-8")

    @staticmethod
    def _pcm_ffmpeg_format(sample_width: int) -> str:
        fmt = {1: "u8", 2: "s16le", 3: "s24le", 4: "s32le"}.get(sample_width)
        if fmt is None:
            raise RuntimeError(f"Unsupported sample width: {sample_width}")
        return fmt

    def _export_via_ffmpeg(self, tmp_raw: Path, output_path: Path,
                            fmt: str, sample_width: int, frame_rate: int,
                            channels: int, bitrate: Optional[str] = None):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(f"ffmpeg is required to produce .{fmt} output but was not found on PATH")

        cmd = [
            "ffmpeg", "-y",
            "-f", self._pcm_ffmpeg_format(sample_width),
            "-ar", str(frame_rate), "-ac", str(channels),
            "-i", str(tmp_raw),
        ]
        if fmt == "wav":
            cmd += ["-rf64", "auto"]
        if bitrate:
            cmd += ["-b:a", bitrate]
        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to encode {fmt}: {result.stderr[-2000:]}")

    def _export_m4b(self, tmp_raw: Path, output_path: Path,
                     chapters: List[Tuple[str, int, int]], book_title: str, chunks_dir: Path,
                     sample_width: int, frame_rate: int, channels: int):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required to produce .m4b output but was not found on PATH")

        tmp_meta = chunks_dir / "_chapters.txt"
        try:
            self._write_ffmetadata(tmp_meta, chapters, book_title)

            cmd = [
                "ffmpeg", "-y",
                "-f", self._pcm_ffmpeg_format(sample_width),
                "-ar", str(frame_rate), "-ac", str(channels),
                "-i", str(tmp_raw),
                "-f", "ffmetadata", "-i", str(tmp_meta),
                "-map", "0:a", "-map_metadata", "1",
                "-c:a", "aac", "-b:a", config.AUDIO_BITRATE,
                "-movflags", "+faststart",
                "-metadata", f"title={book_title}",
                str(output_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed to build m4b: {result.stderr[-2000:]}")
        finally:
            tmp_meta.unlink(missing_ok=True)

    def cleanup_chunks(self, chunks_dir: Path):
        try:
            shutil.rmtree(chunks_dir)
            self.logger.info(f"Cleaned up {chunks_dir}")
        except FileNotFoundError:
            pass
        except Exception as e:
            self.logger.warning(f"Failed to clean up {chunks_dir}: {e}")

    # ------------------------------------------------------------------ #
    # Top-level conversion
    # ------------------------------------------------------------------ #

    def synthesize_book(self, input_path: Path, progress: Progress, task_name: str) -> Optional[dict]:
        """GPU-bound stage: load pages, split into chunks, and synthesize each chunk to
        a per-book chunks/<book>/ directory. Returns everything the assembly stage needs,
        or None if no chunks could be produced at all."""
        self.logger.info(f"Converting: {input_path}")

        pages = self.load_book_pages(input_path)
        total_words = sum(len(text.split()) for _, text, _ in pages)
        self.logger.info(f"Loaded {len(pages)} page(s), {total_words} words")

        page_labels = [label for label, _, _ in pages]
        page_stems = [stem for _, _, stem in pages]
        chunks: List[str] = []
        chunk_page_index: List[int] = []
        for page_idx, (_, text, _) in enumerate(pages):
            for c in self.split_into_chunks(text):
                chunks.append(c)
                chunk_page_index.append(page_idx)

        total_chunks = len(chunks)
        if total_chunks == 0:
            self.logger.error(f"No chunks created for {input_path}")
            return None

        avg_words = sum(len(c.split()) for c in chunks) / total_chunks
        self.logger.info(
            f"Split into {total_chunks} chunks across {len(pages)} chapter(s) (avg {avg_words:.0f} words per chunk)"
        )

        console.print(Panel.fit(
            f"[bold]{input_path.name}[/]\n"
            f"{total_words:,} words -> {total_chunks} chunks, {len(pages)} chapter(s)  "
            f"[dim](voice={self.voice}, speed={self.speed}x, {len(self.pipelines)} worker(s) on {self.device})[/]",
            title="Synthesizing", border_style="blue",
        ))

        chunks_dir = self._chunks_dir(input_path)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        results = {}
        task_id = progress.add_task(task_name[:28], total=total_chunks, speed="-- wds/s")
        try:
            words_done = 0
            chunk_start = time.time()
            with ThreadPoolExecutor(max_workers=len(self.pipelines), thread_name_prefix="synth") as executor:
                futures = {
                    executor.submit(self.process_chunk_with_retry, i, chunk_text, chunks_dir): (i, chunk_text)
                    for i, chunk_text in enumerate(chunks, 1)
                }
                for future in as_completed(futures):
                    i, chunk_text = futures[future]
                    ok = future.result()
                    results[i] = ok
                    words_done += len(chunk_text.split())
                    elapsed = time.time() - chunk_start
                    wps = words_done / elapsed if elapsed > 0 else 0
                    progress.update(task_id, advance=1, speed=f"{wps:.0f} wds/s" if ok else "retry...")
                    if not ok:
                        self.logger.error(f"Chunk {i}/{total_chunks} failed")
        finally:
            progress.remove_task(task_id)

        successful = sum(1 for v in results.values() if v)
        if successful < total_chunks:
            console.print(f"[yellow]⚠ {total_chunks - successful} of {total_chunks} chunks failed[/] (see logs/)")

        return {
            "input_path": input_path,
            "chunks_dir": chunks_dir,
            "total_chunks": total_chunks,
            "results": results,
            "chunk_page_index": chunk_page_index,
            "page_labels": page_labels,
            "page_stems": page_stems,
            "successful": successful,
        }

    def assemble_book(self, job: dict, output_path: Path, keep_chunks: bool, per_page: bool) -> bool:
        """CPU-bound stage: combine/export the already-synthesized chunks into the final
        audiobook file(s). Only touches files under job['chunks_dir'], so it's safe to run
        in a background thread while the next book's chunks are being synthesized on the GPU."""
        input_path = job["input_path"]
        chunks_dir = job["chunks_dir"]
        start_time = time.time()
        success = False
        failed_pages: List[str] = []
        successful_pages = 0
        try:
            if job["successful"] == 0:
                self.logger.error(f"No chunks were successfully synthesized for {input_path.name}")
                console.print(f"[bold red]✗ {input_path.name}: no chunks synthesized[/]")
                return False

            if per_page:
                successful_pages, failed_pages = self.export_per_page(
                    job["total_chunks"], job["results"],
                    chunk_page_index=job["chunk_page_index"], page_labels=job["page_labels"],
                    page_stems=job["page_stems"], output_dir=output_path,
                    fmt=config.AUDIO_FORMAT.lower(), chunks_dir=chunks_dir,
                )
                success = successful_pages > 0
            else:
                success = self.combine_chunks(
                    job["total_chunks"], output_path, job["results"],
                    chunk_page_index=job["chunk_page_index"], page_labels=job["page_labels"],
                    book_title=input_path.stem, chunks_dir=chunks_dir,
                )

            duration = time.time() - start_time
            mins, secs = int(duration // 60), int(duration % 60)
            if success:
                self.logger.info(f"Assembly completed in {mins}m {secs}s: {output_path}")
                if per_page:
                    console.print(f"[bold green]✓ {input_path.name}[/] -> {output_path}/  [dim]({successful_pages} page(s), {mins}m {secs}s)[/]")
                    if failed_pages:
                        console.print(f"[yellow]⚠ Pages with no audio: {failed_pages}[/]")
                else:
                    console.print(f"[bold green]✓ {input_path.name}[/] -> {output_path}  [dim]({mins}m {secs}s)[/]")
            else:
                console.print(f"[bold red]✗ {input_path.name}: failed to assemble audiobook[/]")
            return success
        except Exception as e:
            self.logger.error(f"Assembly failed for {input_path.name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            console.print(f"[bold red]✗ {input_path.name}: {e}[/]")
            return False
        finally:
            if not keep_chunks:
                self.cleanup_chunks(chunks_dir)

    def convert(self, input_path: Path, output_path: Path, keep_chunks: bool = False,
                progress: Optional[Progress] = None, per_page: bool = False) -> bool:
        owns_progress = progress is None
        if owns_progress:
            progress = Progress(*_progress_columns(), console=console)
            progress.start()
        try:
            job = self.synthesize_book(input_path, progress, task_name=input_path.stem)
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            console.print(f"[bold red]✗ {input_path.name}: {e}[/]")
            job = None
        finally:
            if owns_progress:
                progress.stop()

        if job is None:
            return False

        if owns_progress:
            with console.status(f"[bold blue]Assembling {input_path.name}...[/]"):
                return self.assemble_book(job, output_path, keep_chunks, per_page)
        else:
            console.print(f"[dim]Assembling {input_path.name}...[/]")
            return self.assemble_book(job, output_path, keep_chunks, per_page)

    def convert_batch(self, root_dir: Path, keep_chunks: bool = False, per_page: bool = False) -> dict:
        book_dirs = sorted(d for d in root_dir.iterdir() if d.is_dir())
        if not book_dirs:
            console.print(f"[yellow]No subdirectories found in {root_dir}[/]")
            return {}

        console.print(f"[bold]Found {len(book_dirs)} book(s)[/] in {root_dir}\n")
        console.print("[dim]Assembly of each book runs in the background while the next book is synthesized.[/]\n")
        results: dict = {}
        durations: dict = {}

        with Progress(*_progress_columns(), console=console) as progress:
            books_task = progress.add_task(f"[magenta]Books ({len(book_dirs)} total)", total=len(book_dirs), speed="")

            def make_callback(name: str, book_start: float):
                def _on_done(future):
                    try:
                        ok = future.result()
                    except Exception as e:
                        self.logger.error(f"Assembly crashed for {name}: {e}")
                        ok = False
                    results[name] = ok
                    durations[name] = time.time() - book_start
                    progress.update(books_task, advance=1)
                return _on_done

            # A single background worker assembles one finished book at a time while
            # the main thread (and the GPU) moves straight on to synthesizing the next
            # book, instead of sitting idle during each book's CPU-bound audio export.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="assemble") as executor:
                try:
                    for book_dir in book_dirs:
                        book_start = time.time()
                        try:
                            job = self.synthesize_book(book_dir, progress, task_name=book_dir.name)
                        except Exception as e:
                            self.logger.error(f"Synthesis failed for {book_dir.name}: {e}")
                            import traceback
                            self.logger.error(traceback.format_exc())
                            job = None

                        if job is None:
                            results[book_dir.name] = False
                            durations[book_dir.name] = time.time() - book_start
                            progress.update(books_task, advance=1)
                            continue

                        if per_page:
                            output_path = Path(config.AUDIOBOOKS_FOLDER) / f"{book_dir.name}_pages"
                        else:
                            output_path = Path(config.AUDIOBOOKS_FOLDER) / f"{book_dir.name}.{config.AUDIO_FORMAT}"

                        future = executor.submit(self.assemble_book, job, output_path, keep_chunks, per_page)
                        future.add_done_callback(make_callback(book_dir.name, book_start))
                except KeyboardInterrupt:
                    console.print("\n[yellow]Batch interrupted; waiting for in-flight assembly to finish...[/]")
                # Exiting the `with` block waits for any queued/running assembly jobs.

        table = Table(title="Batch Summary")
        table.add_column("Book")
        table.add_column("Status")
        table.add_column("Duration", justify="right")
        for book_dir in book_dirs:
            name = book_dir.name
            if name not in results:
                continue
            status = "[bold green]OK[/]" if results[name] else "[bold red]FAILED[/]"
            d = durations.get(name, 0)
            table.add_row(name, status, f"{int(d // 60)}m {int(d % 60)}s")
        console.print(table)

        return results


def _device_arg(value: str) -> str:
    def _valid_token(token: str) -> bool:
        return token in ("cuda", "cpu") or bool(re.match(r"^cuda:\d+$", token))

    if value == "all" or _valid_token(value):
        return value
    if "," in value and all(_valid_token(t.strip()) for t in value.split(",") if t.strip()):
        return value
    raise argparse.ArgumentTypeError(
        "must be 'cuda', 'cpu', 'all', 'cuda:N' (e.g. cuda:1), or a comma list (e.g. cuda:0,cuda:1)"
    )


def print_voice_list():
    table = Table(title="Kokoro Voices")
    table.add_column("Lang")
    table.add_column("Female")
    table.add_column("Male")
    for lang, genders in config.VOICE_OPTIONS.items():
        table.add_row(lang, ", ".join(genders.get("female", [])) or "-", ", ".join(genders.get("male", [])) or "-")
    console.print(table)
    console.print("[dim]Set with --voice <name> (language is auto-detected from the voice prefix).[/]")


def main():
    parser = argparse.ArgumentParser(
        description="Convert text to an audiobook using the local Kokoro-82M model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single pre-cleaned text file
  python3 audiobook_tts.py mybook.txt -o audiobooks/mybook.mp3

  # A folder of page files, e.g. TTS_Script/out/PracticalElectronicsForInventors/
  python3 audiobook_tts.py ../TTS_Script/out/PracticalElectronicsForInventors

  # Batch mode: convert every book folder under TTS_Script/out/
  python3 audiobook_tts.py --batch ../TTS_Script/out

  # Different voice / speed
  python3 audiobook_tts.py mybook.txt --voice am_michael --speed 1.1

  # One audio file per page instead of one combined book file
  python3 audiobook_tts.py ../TTS_Script/out/PracticalElectronicsForInventors --per-page

  # Force a specific GPU (default is to prefer GPU 0, overflow if it's too full)
  python3 audiobook_tts.py mybook.txt --device cuda:1

  # Load 4 model instances on GPU 0 and synthesize chunks in parallel across them
  python3 audiobook_tts.py mybook.txt --device cuda:0 --workers 4

  # One model per visible GPU, 2 stacked on each (e.g. 6 workers on a 3-GPU box)
  python3 audiobook_tts.py mybook.txt --device all --workers 2

  # Just GPUs 0 and 1 (skip others), one model on each
  python3 audiobook_tts.py mybook.txt --device cuda:0,cuda:1

  python3 audiobook_tts.py --list-voices
""",
    )
    parser.add_argument("input", nargs="?", help="Text file, or directory of page files (or of book folders with --batch)")
    parser.add_argument("-o", "--output", help="Output audio file path (single-book mode); output directory when --per-page is set")
    parser.add_argument("--batch", action="store_true", help="Treat input as a directory containing multiple book subfolders")
    parser.add_argument("--per-page", action="store_true", help="Export one audio file per page instead of one combined book file")
    parser.add_argument("--voice", default=None, help=f"Kokoro voice (default: {config.VOICE})")
    parser.add_argument("--lang", default=None, help="Language code (default: inferred from voice)")
    parser.add_argument("--speed", type=float, default=None, help=f"Speech speed multiplier (default: {config.SPEED})")
    parser.add_argument("--device", default=None, type=_device_arg,
                         help="Inference device: cuda, cpu, cuda:N for a specific GPU, 'all' for every "
                              "visible GPU, or a comma list (e.g. cuda:0,cuda:1) for a specific subset "
                              "(default: prefer GPU 0, overflow if it's too full)")
    parser.add_argument("--workers", type=int, default=1,
                         help="Model instances to load per resolved device (default: 1). Chunks are "
                              "synthesized in parallel across all loaded instances. E.g. --device cuda:0 "
                              "--workers 4 stacks 4 models on GPU 0; --device all --workers 2 loads 2 per GPU.")
    parser.add_argument("--keep-chunks", action="store_true", help="Don't delete per-chunk wav files after assembly")
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")

    args = parser.parse_args()

    if args.list_voices:
        print_voice_list()
        return

    if not args.input:
        parser.error("input is required unless --list-voices is given")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[bold red]✗ Input not found: {input_path}[/]")
        sys.exit(1)

    try:
        generator = KokoroAudiobookGenerator(
            voice=args.voice, lang_code=args.lang, speed=args.speed, device=args.device, workers=args.workers
        )

        if args.batch:
            if not input_path.is_dir():
                console.print("[bold red]✗ --batch requires a directory input[/]")
                sys.exit(1)
            results = generator.convert_batch(input_path, keep_chunks=args.keep_chunks, per_page=args.per_page)
            sys.exit(0 if results and all(results.values()) else 1)
        else:
            stem = input_path.stem if input_path.is_file() else input_path.name
            if args.output:
                output_path = Path(args.output)
            elif args.per_page:
                output_path = Path(config.AUDIOBOOKS_FOLDER) / f"{stem}_pages"
            else:
                output_path = Path(config.AUDIOBOOKS_FOLDER) / f"{stem}.{config.AUDIO_FORMAT}"
            success = generator.convert(input_path, output_path, keep_chunks=args.keep_chunks, per_page=args.per_page)
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
