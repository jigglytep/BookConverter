#!/usr/bin/env python3
"""Live terminal status display and observability module for the BookConverters pipeline.

Provides:
- Real-time Rich Live dashboard (with plain ANSI fallback and non-TTY/unattended mode)
- Background GPU telemetry polling (pynvml preferred, nvidia-smi fallback) for Tesla P100s
- Host CPU, RAM, and output volume free disk monitoring
- Rolling-window throughput and ETA calculation (stage ETA + overall book ETA)
- Stage list tracking with active stage highlight, progress bar, TTS RTF, retries, and warnings
- Scrolling log pane capturing print(), Python logging, subprocess stdout/stderr, and library warnings
- Full timestamped log teeing to <LOG_DIR>/<run-timestamp>.log
- Decoupled programmatic API:
    start_run(queue), start_book(name), start_stage(name, total_units),
    advance(n=1), log(level, msg), end_stage(), end_book(summary), end_run()
"""

from __future__ import annotations

import argparse
import builtins
from collections import deque
from dataclasses import dataclass, field
import datetime
import io
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import warnings
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

# Prefer Rich if installed
try:
    from rich.box import ROUNDED, SIMPLE
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Prefer pynvml if installed
try:
    import pynvml  # type: ignore
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False


# =============================================================================
# Pipeline Stage Definitions
# =============================================================================

@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    default_unit: str
    weight_sec_per_page: float  # Prior heuristic for book ETA before stage runs


_ACTIVE_TTS_LABEL = (
    "Chatterbox"
    if os.environ.get("TTS_ENGINE", "qwen3").strip().lower().startswith("chatterbox")
    else "Qwen3-TTS"
)

PIPELINE_STAGES: Tuple[StageSpec, ...] = (
    StageSpec("mineru", "1. PDF Extract (MinerU)", "pages", 1.8),
    StageSpec("md_tts", "2. Text Clean & Math (md_tts)", "pages", 0.15),
    StageSpec("dom_prep", "3a. DOM & Chunk Prep", "pages", 0.08),
    StageSpec("tts_synth", f"3b. TTS Synthesis ({_ACTIVE_TTS_LABEL})", "chunks", 8.5),
    StageSpec("smil_assembly", "3c. Audio & SMIL Assembly", "pages", 0.25),
    StageSpec("epub_m4b_mux", "3d. EPUB3 & M4B Encode/Mux", "steps", 0.05),
    StageSpec("abs_upload", "3e. AudioBookshelf Upload", "steps", 0.03),
)

STAGE_ALIASES: Dict[str, str] = {
    "1": "mineru",
    "stage1": "mineru",
    "mineru": "mineru",
    "ocr": "mineru",
    "pdf extract": "mineru",
    "pdf extract (mineru)": "mineru",
    "1. pdf extract (mineru)": "mineru",
    "2": "md_tts",
    "stage2": "md_tts",
    "md_tts": "md_tts",
    "tts_script": "md_tts",
    "text clean": "md_tts",
    "text clean & math": "md_tts",
    "text clean & math (md_tts)": "md_tts",
    "2. text clean & math (md_tts)": "md_tts",
    "3a": "dom_prep",
    "dom": "dom_prep",
    "dom_prep": "dom_prep",
    "chunk": "dom_prep",
    "dom & chunk prep": "dom_prep",
    "3a. dom & chunk prep": "dom_prep",
    "3": "tts_synth",
    "3b": "tts_synth",
    "tts": "tts_synth",
    "tts_synth": "tts_synth",
    "qwen3": "tts_synth",
    "qwen3-tts": "tts_synth",
    "qwen_tts": "tts_synth",
    "chatterbox": "tts_synth",
    "tts synthesis": "tts_synth",
    "tts synthesis (qwen3-tts)": "tts_synth",
    "tts synthesis (chatterbox)": "tts_synth",
    "3b. tts synthesis (qwen3-tts)": "tts_synth",
    "3b. tts synthesis (chatterbox)": "tts_synth",
    "3c": "smil_assembly",
    "smil": "smil_assembly",
    "smil_assembly": "smil_assembly",
    "audio & smil assembly": "smil_assembly",
    "3c. audio & smil assembly": "smil_assembly",
    "3d": "epub_m4b_mux",
    "mux": "epub_m4b_mux",
    "encode": "epub_m4b_mux",
    "epub_m4b_mux": "epub_m4b_mux",
    "epub3 & m4b encode/mux": "epub_m4b_mux",
    "3d. epub3 & m4b encode/mux": "epub_m4b_mux",
    "3e": "abs_upload",
    "upload": "abs_upload",
    "abs": "abs_upload",
    "abs_upload": "abs_upload",
    "audiobookshelf upload": "abs_upload",
    "3e. audiobookshelf upload": "abs_upload",
}


def resolve_stage_spec(name: str) -> StageSpec:
    norm = name.strip().lower()
    key = STAGE_ALIASES.get(norm)
    if key:
        for spec in PIPELINE_STAGES:
            if spec.key == key:
                return spec
    for spec in PIPELINE_STAGES:
        if norm in spec.label.lower() or spec.key in norm:
            return spec
    return StageSpec(norm or "custom", name.strip() or "Processing", "units", 1.0)


# =============================================================================
# Pure ETA & Throughput Functions (Unit-Testable)
# =============================================================================

def compute_rolling_rate(
    samples: Sequence[Tuple[float, float]] | Deque[Tuple[float, float]],
    min_samples: int = 3,
    window_sec: float = 60.0,
    min_elapsed: float = 0.25,
) -> Optional[float]:
    """Compute rolling throughput (units/sec) over a deque/sequence of (timestamp, units_done) samples.

    Args:
        samples: Sequence or deque of (timestamp_sec, cumulative_units_done) ordered chronologically.
        min_samples: Minimum number of sample points required before reporting a rate.
        window_sec: Rolling window duration in seconds looking back from the latest sample.
        min_elapsed: Minimum elapsed wall time (seconds) across the chosen sample window.

    Returns:
        Rolling throughput in units/second, or None if insufficient samples or non-positive progress.
    """
    if len(samples) < max(2, min_samples):
        return None

    t_newest, u_newest = samples[-1]
    cutoff = t_newest - window_sec

    # Find the earliest sample within the rolling window; if fewer than min_samples fall
    # inside the window, use the last `min_samples` points so slow stages still get a rate.
    window_pts = [pt for pt in samples if pt[0] >= cutoff]
    if len(window_pts) < min_samples:
        window_pts = list(samples)[-min_samples:]

    t_oldest, u_oldest = window_pts[0]
    dt = t_newest - t_oldest
    du = u_newest - u_oldest

    if dt < min_elapsed or du <= 0:
        return None

    return du / dt


def compute_eta(
    samples: Sequence[Tuple[float, float]] | Deque[Tuple[float, float]],
    remaining_units: float,
    min_samples: int = 3,
    window_sec: float = 60.0,
    min_elapsed: float = 0.25,
) -> Optional[float]:
    """Pure function computing remaining ETA in seconds from rolling (timestamp, units_done) samples.

    Returns:
        0.0 if remaining_units <= 0 and at least 1 sample exists;
        estimated seconds remaining (float) when enough samples exist;
        None ("calculating...") when there are not enough samples yet.
    """
    if remaining_units <= 0:
        return 0.0 if len(samples) >= 1 else None

    rate = compute_rolling_rate(
        samples=samples,
        min_samples=min_samples,
        window_sec=window_sec,
        min_elapsed=min_elapsed,
    )
    if rate is None or rate <= 0:
        return None

    return float(remaining_units) / rate


def format_duration(seconds: Optional[float]) -> str:
    """Format seconds as human-readable duration (e.g. '1h 12m 04s', '4m 18s', '12s', or 'calculating...')."""
    if seconds is None or math.isnan(seconds) or math.isinf(seconds) or seconds < 0:
        return "calculating..."
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def format_mmss(seconds: Optional[float]) -> str:
    """Format audio duration in HH:MM:SS or MM:SS."""
    if seconds is None or seconds < 0:
        return "n/a"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# =============================================================================
# Telemetry Data Structures & Background Poller
# =============================================================================

@dataclass
class GPUStats:
    index: int = 0
    name: str = "Tesla P100-PCIE-16GB"
    mem_used_mib: Optional[float] = None
    mem_total_mib: Optional[float] = None
    temp_c: Optional[float] = None
    util_pct: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    sm_clock_mhz: Optional[int] = None
    throttle_reasons: str = "OK"
    available: bool = False

    @property
    def mem_pct(self) -> Optional[float]:
        if self.mem_used_mib is not None and self.mem_total_mib and self.mem_total_mib > 0:
            return (self.mem_used_mib / self.mem_total_mib) * 100.0
        return None


@dataclass
class HostStats:
    cpu_pct: Optional[float] = None
    ram_used_gib: Optional[float] = None
    ram_total_gib: Optional[float] = None
    ram_pct: Optional[float] = None
    disk_free_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None


def decode_throttle_bitmask(raw_val: str | int) -> str:
    """Decode NVML / nvidia-smi clocks_throttle_reasons.active bitmask into compact flags."""
    try:
        if isinstance(raw_val, str):
            s = raw_val.strip()
            if not s or s.lower() in ("n/a", "[not supported]", "none"):
                return "OK"
            mask = int(s, 16) if s.startswith(("0x", "0X")) else int(s)
        else:
            mask = int(raw_val)
    except Exception:
        return "OK"

    if mask == 0:
        return "OK"
    # Ignore pure GpuIdle (0x1) or AppClockSetting (0x2) unless accompanied by real throttling
    flags: List[str] = []
    if mask & 0x0000000000000040:
        flags.append("HW_THERM")
    if mask & 0x0000000000000020:
        flags.append("SW_THERM")
    if mask & 0x0000000000000080:
        flags.append("PWR_BRAKE")
    if mask & 0x0000000000000008:
        flags.append("HW_SLOW")
    if mask & 0x0000000000000004:
        flags.append("PWR_CAP")
    if not flags:
        if mask & 0x0000000000000001:
            return "IDLE"
        return "OK"
    return ",".join(flags)


class SystemPoller:
    """Background daemon poller for GPU and Host metrics."""

    def __init__(
        self,
        output_path: Path,
        poll_interval: float = 1.5,
        gpu_indices: Optional[List[int]] = None,
        demo_mode: bool = False,
    ):
        self.output_path = output_path
        self.poll_interval = max(0.2, poll_interval)
        self.gpu_indices = gpu_indices
        self.demo_mode = demo_mode

        self._lock = threading.Lock()
        self._gpus: List[GPUStats] = []
        self._host = HostStats()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_cpu_times: Optional[Tuple[int, int]] = None
        self._nvml_initialized = False
        self._demo_phase = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._poll_once()
        self._thread = threading.Thread(target=self._run_loop, name="StatusDisplayPoller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._nvml_initialized and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False

    def snapshot(self) -> Tuple[List[GPUStats], HostStats]:
        with self._lock:
            return list(self._gpus), HostStats(
                cpu_pct=self._host.cpu_pct,
                ram_used_gib=self._host.ram_used_gib,
                ram_total_gib=self._host.ram_total_gib,
                ram_pct=self._host.ram_pct,
                disk_free_gb=self._host.disk_free_gb,
                disk_total_gb=self._host.disk_total_gb,
            )

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            try:
                self._poll_once()
            except Exception:
                pass

    def _poll_once(self) -> None:
        if self.demo_mode:
            gpus = self._poll_demo_gpus()
            host = self._poll_host(demo=True)
        else:
            gpus = self._poll_gpus()
            host = self._poll_host(demo=False)

        with self._lock:
            self._gpus = gpus
            self._host = host

    def _poll_demo_gpus(self) -> List[GPUStats]:
        self._demo_phase += 0.35
        g0_util = 88.0 + 10.0 * math.sin(self._demo_phase)
        g1_util = 82.0 + 14.0 * math.cos(self._demo_phase * 0.9)
        g0_temp = 68.0 + 9.0 * max(0.0, math.sin(self._demo_phase * 0.4))
        g1_temp = 64.0 + 6.0 * max(0.0, math.cos(self._demo_phase * 0.4))
        return [
            GPUStats(
                index=0,
                name="Tesla P100-PCIE-16GB",
                mem_used_mib=11420.0 + 180.0 * math.sin(self._demo_phase),
                mem_total_mib=16384.0,
                temp_c=round(g0_temp, 0),
                util_pct=round(max(0.0, min(100.0, g0_util)), 0),
                power_draw_w=round(168.5 + 22.0 * math.sin(self._demo_phase), 1),
                power_limit_w=250.0,
                sm_clock_mhz=1328,
                throttle_reasons="OK",
                available=True,
            ),
            GPUStats(
                index=1,
                name="Tesla P100-PCIE-16GB",
                mem_used_mib=11180.0 + 150.0 * math.cos(self._demo_phase),
                mem_total_mib=16384.0,
                temp_c=round(g1_temp, 0),
                util_pct=round(max(0.0, min(100.0, g1_util)), 0),
                power_draw_w=round(154.0 + 19.0 * math.cos(self._demo_phase), 1),
                power_limit_w=250.0,
                sm_clock_mhz=1328,
                throttle_reasons="OK",
                available=True,
            ),
        ]

    def _poll_gpus(self) -> List[GPUStats]:
        # 1. Try pynvml if available
        if PYNVML_AVAILABLE and pynvml is not None:
            try:
                if not self._nvml_initialized:
                    pynvml.nvmlInit()
                    self._nvml_initialized = True
                count = pynvml.nvmlDeviceGetCount()
                indices = self.gpu_indices if self.gpu_indices is not None else list(range(count))
                stats: List[GPUStats] = []
                for idx in indices:
                    if idx < 0 or idx >= count:
                        continue
                    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    name_raw = pynvml.nvmlDeviceGetName(h)
                    name = name_raw.decode("utf-8", errors="replace") if isinstance(name_raw, bytes) else str(name_raw)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                    try:
                        util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                    except Exception:
                        util = None
                    try:
                        pwr_draw = float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
                    except Exception:
                        pwr_draw = None
                    try:
                        pwr_lim = float(pynvml.nvmlDeviceGetEnforcedPowerLimit(h)) / 1000.0
                    except Exception:
                        pwr_lim = None
                    try:
                        sm_clk = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                    except Exception:
                        sm_clk = None
                    try:
                        thr_mask = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
                        thr_str = decode_throttle_bitmask(thr_mask)
                    except Exception:
                        thr_str = "OK"

                    stats.append(
                        GPUStats(
                            index=idx,
                            name=name,
                            mem_used_mib=float(mem.used) / (1024.0 * 1024.0),
                            mem_total_mib=float(mem.total) / (1024.0 * 1024.0),
                            temp_c=temp,
                            util_pct=util,
                            power_draw_w=pwr_draw,
                            power_limit_w=pwr_lim,
                            sm_clock_mhz=sm_clk,
                            throttle_reasons=thr_str,
                            available=True,
                        )
                    )
                if stats:
                    return stats
            except Exception:
                pass

        # 2. Fallback to nvidia-smi
        if shutil.which("nvidia-smi") is None:
            return [GPUStats(available=False)]

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw,power.limit,clocks.current.sm,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0, check=False)
            if res.returncode != 0 or not res.stdout.strip():
                return [GPUStats(available=False)]

            stats = []
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                idx = _safe_int(parts[0], 0)
                if self.gpu_indices is not None and idx not in self.gpu_indices:
                    continue
                name = parts[1]
                mem_used = _safe_float(parts[2])
                mem_total = _safe_float(parts[3])
                temp = _safe_float(parts[4])
                util = _safe_float(parts[5]) if len(parts) > 5 else None
                pwr_draw = _safe_float(parts[6]) if len(parts) > 6 else None
                pwr_lim = _safe_float(parts[7]) if len(parts) > 7 else None
                sm_clk = _safe_int(parts[8]) if len(parts) > 8 else None
                thr_raw = parts[9] if len(parts) > 9 else "0x0"

                stats.append(
                    GPUStats(
                        index=idx,
                        name=name,
                        mem_used_mib=mem_used,
                        mem_total_mib=mem_total,
                        temp_c=temp,
                        util_pct=util,
                        power_draw_w=pwr_draw,
                        power_limit_w=pwr_lim,
                        sm_clock_mhz=sm_clk,
                        throttle_reasons=decode_throttle_bitmask(thr_raw),
                        available=True,
                    )
                )
            return stats if stats else [GPUStats(available=False)]
        except Exception:
            return [GPUStats(available=False)]

    def _poll_host(self, demo: bool = False) -> HostStats:
        hs = HostStats()
        # CPU % from /proc/stat
        try:
            stat_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
            parts = [int(x) for x in stat_line.split()[1:]]
            idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
            total = sum(parts)
            if self._prev_cpu_times is not None:
                prev_idle, prev_total = self._prev_cpu_times
                d_total = total - prev_total
                d_idle = idle - prev_idle
                if d_total > 0:
                    hs.cpu_pct = max(0.0, min(100.0, (1.0 - d_idle / d_total) * 100.0))
            self._prev_cpu_times = (idle, total)
        except Exception:
            pass

        if demo and hs.cpu_pct is None:
            hs.cpu_pct = 24.5

        # RAM from /proc/meminfo
        try:
            meminfo: Dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    val_parts = v.strip().split()
                    if val_parts and val_parts[0].isdigit():
                        meminfo[k.strip()] = int(val_parts[0])  # kB
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            if total_kb > 0:
                used_kb = max(0, total_kb - avail_kb)
                hs.ram_total_gib = total_kb / (1024.0 * 1024.0)
                hs.ram_used_gib = used_kb / (1024.0 * 1024.0)
                hs.ram_pct = (used_kb / total_kb) * 100.0
        except Exception:
            pass

        # Disk usage on output volume
        try:
            target = self.output_path if self.output_path.exists() else Path.cwd()
            du = shutil.disk_usage(str(target))
            hs.disk_free_gb = du.free / (1024.0 ** 3)
            hs.disk_total_gb = du.total / (1024.0 ** 3)
        except Exception:
            pass

        return hs


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# =============================================================================
# Pure Plain-Mode Formatters (Unit-Testable)
# =============================================================================

def format_plain_event(
    timestamp: str,
    level: str,
    message: str,
    book: Optional[str] = None,
    stage: Optional[str] = None,
) -> str:
    """Format a single timestamped event line for plain / non-TTY mode."""
    lvl = (level or "INFO").upper().strip()
    ctx_parts: List[str] = []
    if book:
        ctx_parts.append(book)
    if stage:
        ctx_parts.append(stage)
    ctx_str = f" [{' | '.join(ctx_parts)}]" if ctx_parts else ""
    clean_msg = " ".join(message.strip().splitlines())
    return f"[{timestamp}] [{lvl:<5}]{ctx_str} {clean_msg}"


def format_plain_heartbeat(
    timestamp: str,
    book_name: str,
    book_pos: str,
    stage_name: str,
    completed: int,
    total: int,
    eta_sec: Optional[float],
    book_eta_sec: Optional[float],
    elapsed_sec: float,
    gpus: Sequence[GPUStats],
    rtf: Optional[float] = None,
) -> str:
    """Format a periodic progress heartbeat line for plain / non-TTY mode."""
    pct_str = f"{(completed / total * 100.0):.1f}%" if total > 0 else "0.0%"
    counts_str = f"{completed}/{total}" if total > 0 else f"{completed}"
    eta_str = format_duration(eta_sec)
    book_eta_str = format_duration(book_eta_sec)
    elapsed_str = format_duration(elapsed_sec)

    gpu_strs: List[str] = []
    for g in gpus:
        if g.available and g.mem_used_mib is not None and g.mem_total_mib:
            mem_pct = g.mem_pct or 0.0
            t_str = f"{int(round(g.temp_c))}°C" if g.temp_c is not None else "n/a"
            gpu_strs.append(
                f"GPU{g.index} {int(round(g.mem_used_mib))}/{int(round(g.mem_total_mib))}MiB ({mem_pct:.0f}%) {t_str}"
            )
    gpu_summary = " | ".join(gpu_strs) if gpu_strs else "GPU n/a"
    rtf_part = f" | RTF {rtf:.2f}x" if rtf is not None and rtf > 0 else ""

    return (
        f"[{timestamp}] [HEARTBEAT] Book ({book_pos}): {book_name} | "
        f"Stage: {stage_name} {counts_str} ({pct_str}) | "
        f"Elapsed: {elapsed_str} | Stage ETA: {eta_str} | Book ETA: {book_eta_str}"
        f"{rtf_part} | {gpu_summary}"
    )


# =============================================================================
# Stream & Logging Interceptors
# =============================================================================

class _StreamRedirector(io.TextIOBase):
    """File-like stream that routes writes line-by-line into StatusDisplay.log()."""

    def __init__(self, display: "StatusDisplay", level: str = "INFO", orig_stream: Any = None):
        super().__init__()
        self._display = display
        self._level = level
        self._orig_stream = orig_stream
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                clean = line.rstrip("\r")
                if clean.strip():
                    self._display._handle_captured_line(clean, default_level=self._level)
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if self._buf.strip():
                self._display._handle_captured_line(self._buf.rstrip("\r\n"), default_level=self._level)
            self._buf = ""

    def isatty(self) -> bool:
        return False

    @property
    def encoding(self) -> str:
        return getattr(self._orig_stream, "encoding", "utf-8") or "utf-8"


class _DisplayLogHandler(logging.Handler):
    """Python logging.Handler that routes log records to StatusDisplay.log()."""

    def __init__(self, display: "StatusDisplay"):
        super().__init__()
        self._display = display

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            lvl = record.levelname
            if lvl == "WARNING":
                lvl = "WARN"
            elif lvl == "CRITICAL":
                lvl = "ERROR"
            self._display.log(lvl, msg)
        except Exception:
            pass


# =============================================================================
# Cross-Process Event Bridge (Used when child scripts run under supervisor)
# =============================================================================

def emit_ipc_event(event_type: str, **payload: Any) -> bool:
    """Append a structured JSON event to STATUS_DISPLAY_EVENT_FILE if active."""
    event_file = os.environ.get("STATUS_DISPLAY_EVENT_FILE")
    if not event_file:
        return False
    try:
        record = {"ts": time.time(), "event": event_type, **payload}
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(event_file, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        return True
    except Exception:
        return False


# =============================================================================
# Main StatusDisplay Class
# =============================================================================

@dataclass
class BookRecord:
    name: str
    status: str = "done"  # done, failed, skipped
    duration_sec: float = 0.0
    audio_duration_sec: float = 0.0
    rtf: Optional[float] = None
    retries: int = 0
    warnings: int = 0
    output_path: str = ""
    failed_stage: str = ""


_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_RICH_MARKUP_RE = re.compile(r"\[/?(?:bold|dim|italic|underline|cyan|green|yellow|red|magenta|blue|white|[a-z ]+)\]")


def strip_formatting(text: str) -> str:
    s = _ANSI_ESCAPE_RE.sub("", text)
    s = _RICH_MARKUP_RE.sub("", s)
    return s


class StatusDisplay:
    """Live status display for the PDF-to-audiobook pipeline."""

    def __init__(
        self,
        log_dir: Path | str = "logs",
        output_dir: Path | str = "chatterbox/audiobooks",
        plain: bool = False,
        quiet: bool = False,
        verbose: bool = False,
        heartbeat_sec: float = 30.0,
        poll_interval_sec: float = 1.5,
        temp_warn_c: float = 75.0,
        temp_crit_c: float = 85.0,
        disk_warn_gb: float = 10.0,
        demo_mode: bool = False,
        stream: Any = None,
    ):
        self.base_dir = Path(__file__).resolve().parent
        self.log_dir = Path(log_dir) if Path(log_dir).is_absolute() else (self.base_dir / log_dir)
        self.output_dir = Path(output_dir) if Path(output_dir).is_absolute() else (self.base_dir / output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.log_dir / f"{run_ts}.log"

        self._orig_stdout = stream or sys.__stdout__ or sys.stdout
        self._orig_stderr = sys.__stderr__ or sys.stderr
        self._orig_print = builtins.print
        self._orig_showwarning = warnings.showwarning

        is_tty = bool(
            hasattr(self._orig_stdout, "isatty")
            and self._orig_stdout.isatty()
            and os.environ.get("TERM", "") not in ("", "dumb")
        )
        self.plain = bool(plain or not is_tty)
        self.quiet = quiet
        self.verbose = verbose
        self.heartbeat_sec = max(1.0, float(heartbeat_sec))
        self.temp_warn_c = temp_warn_c
        self.temp_crit_c = temp_crit_c
        self.disk_warn_gb = disk_warn_gb
        self.demo_mode = demo_mode

        self._lock = threading.RLock()
        self._poller = SystemPoller(
            output_path=self.output_dir,
            poll_interval=poll_interval_sec,
            demo_mode=demo_mode,
        )

        # Run state
        self.run_active = False
        self.run_start_time = 0.0
        self.queue: List[str] = []
        self.current_book_idx = 0  # 1-based
        self.current_book_name = ""
        self.current_book_start_time = 0.0
        self.current_book_pages = 0
        self.current_book_chunks = 0
        self.book_records: List[BookRecord] = []
        self.books_done = 0
        self.books_failed = 0
        self.books_skipped = 0

        # Stage state
        self.stage_statuses: Dict[str, str] = {s.key: "pending" for s in PIPELINE_STAGES}
        self.stage_durations: Dict[str, float] = {}
        self.historical_stage_sec_per_page: Dict[str, float] = {
            s.key: s.weight_sec_per_page for s in PIPELINE_STAGES
        }
        self.active_stage_key: str = ""
        self.active_stage_label: str = "Idle"
        self.active_stage_unit: str = "units"
        self.stage_start_time: float = 0.0
        self.stage_total_units: int = 0
        self.stage_completed_units: int = 0
        self.stage_initial_units: int = 0
        self.stage_samples: Deque[Tuple[float, float]] = deque(maxlen=400)

        # TTS / Audio metrics for current book
        self.tts_synth_Start_time: float = 0.0
        self.tts_audio_produced_sec: float = 0.0
        self.tts_cached_chunks: int = 0
        self.tts_synthesized_chunks: int = 0
        self.tts_words_done: int = 0
        self.tts_total_words: int = 0
        self.book_retries: int = 0
        self.book_warnings: int = 0
        self.total_retries: int = 0
        self.total_warnings: int = 0
        self.last_output_path: str = ""

        # Scrolling log pane
        self.log_lines: Deque[Tuple[str, str, str]] = deque(maxlen=12)  # (HH:MM:SS, LEVEL, msg)
        self.book_summaries: Deque[str] = deque(maxlen=4)

        # Live UI rendering state
        self._live: Optional[Any] = None
        self._console: Optional[Any] = None
        self._refresh_stop = threading.Event()
        self._refresh_thread: Optional[threading.Thread] = None
        self._last_heartbeat_time = 0.0
        self._streams_hooked = False
        self._log_handler: Optional[_DisplayLogHandler] = None
        self._ansi_prev_lines = 0

    # -------------------------------------------------------------------------
    # Lifecycle & Public API
    # -------------------------------------------------------------------------

    def start_run(self, queue: Sequence[str], hook_streams: bool = True) -> None:
        if emit_ipc_event("start_run", queue=list(queue)):
            return

        with self._lock:
            self.run_active = True
            self.run_start_time = time.time()
            self._last_heartbeat_time = self.run_start_time
            self.queue = [self._clean_book_label(q) for q in queue]
            self.current_book_idx = 0
            self.current_book_name = ""
            self.books_done = 0
            self.books_failed = 0
            self.books_skipped = 0
            self.book_records.clear()

        self._write_log_file("INFO", f"=== Pipeline Run Started (Queue: {len(self.queue)} book(s)) ===")
        self._write_log_file("INFO", f"Log file: {self.log_file_path}")

        self._poller.start()

        if not self.plain:
            try:
                if RICH_AVAILABLE:
                    self._console = Console(file=self._orig_stdout, force_terminal=True, highlight=False)
                    self._live = Live(
                        self._build_rich_renderable(),
                        console=self._console,
                        refresh_per_second=4,
                        transient=False,
                    )
                    self._live.start()
            except Exception as e:
                self._write_log_file("WARN", f"Rich Live initialization failed ({e}); falling back to plain mode.")
                self.plain = True
                self._live = None

        if hook_streams and not self.plain:
            self._install_stream_hooks()

        if self.plain:
            self._emit_plain_line("INFO", f"Starting pipeline run with {len(self.queue)} queued book(s). Log: {self.log_file_path}")

        self._refresh_stop.clear()
        self._refresh_thread = threading.Thread(target=self._refresh_loop, name="StatusDisplayRefresh", daemon=True)
        self._refresh_thread.start()

    def start_book(
        self,
        name: str,
        index: Optional[int] = None,
        total: Optional[int] = None,
        next_book: Optional[str] = None,
    ) -> None:
        if emit_ipc_event("start_book", name=name, index=index, total=total, next_book=next_book):
            return

        clean_name = self._clean_book_label(name)
        with self._lock:
            if total is not None and total > len(self.queue):
                # Adjust queue size if discovered dynamically
                while len(self.queue) < total:
                    self.queue.append("Pending Book")
            if index is not None:
                self.current_book_idx = index
            else:
                if clean_name in self.queue:
                    self.current_book_idx = self.queue.index(clean_name) + 1
                else:
                    self.current_book_idx += 1
                    if self.current_book_idx > len(self.queue):
                        self.queue.append(clean_name)
            if 1 <= self.current_book_idx <= len(self.queue):
                self.queue[self.current_book_idx - 1] = clean_name
            if next_book and self.current_book_idx < len(self.queue):
                self.queue[self.current_book_idx] = self._clean_book_label(next_book)

            self.current_book_name = clean_name
            self.current_book_start_time = time.time()
            self.current_book_pages = 0
            self.current_book_chunks = 0
            self.stage_statuses = {s.key: "pending" for s in PIPELINE_STAGES}
            self.stage_durations.clear()
            self.active_stage_key = ""
            self.active_stage_label = "Initializing"
            self.stage_total_units = 0
            self.stage_completed_units = 0
            self.stage_samples.clear()

            # Reset per-book counters
            self.tts_synth_Start_time = 0.0
            self.tts_audio_produced_sec = 0.0
            self.tts_cached_chunks = 0
            self.tts_synthesized_chunks = 0
            self.tts_words_done = 0
            self.tts_total_words = 0
            self.book_retries = 0
            self.book_warnings = 0
            self.last_output_path = ""

            pos_str = f"{self.current_book_idx} of {max(1, len(self.queue))}"

        self.log("INFO", f"Starting book ({pos_str}): {clean_name}")
        self._trigger_redraw()

    def start_stage(
        self,
        name: str,
        total_units: int = 0,
        unit_name: Optional[str] = None,
        initial_completed: int = 0,
    ) -> None:
        if emit_ipc_event(
            "start_stage",
            name=name,
            total_units=total_units,
            unit_name=unit_name,
            initial_completed=initial_completed,
        ):
            return

        spec = resolve_stage_spec(name)
        now = time.time()
        with self._lock:
            # Mark any previously active stage as done
            if self.active_stage_key and self.stage_statuses.get(self.active_stage_key) == "active":
                self.stage_statuses[self.active_stage_key] = "done"
                if self.stage_start_time > 0:
                    self.stage_durations[self.active_stage_key] = now - self.stage_start_time

            # Also mark earlier stages in pipeline order as done/skipped if jumping ahead (e.g. backlog)
            for s in PIPELINE_STAGES:
                if s.key == spec.key:
                    break
                if self.stage_statuses.get(s.key) == "pending":
                    self.stage_statuses[s.key] = "done"

            self.active_stage_key = spec.key
            self.active_stage_label = spec.label
            self.active_stage_unit = unit_name or spec.default_unit
            self.stage_statuses[spec.key] = "active"
            self.stage_start_time = now
            self.stage_total_units = max(0, int(total_units))
            self.stage_completed_units = max(0, int(initial_completed))
            self.stage_initial_units = self.stage_completed_units
            self.stage_samples.clear()
            self.stage_samples.append((now, float(self.stage_completed_units)))

            if spec.key in ("mineru", "md_tts", "dom_prep", "smil_assembly") and self.stage_total_units > 0:
                self.current_book_pages = max(self.current_book_pages, self.stage_total_units)
            elif spec.key == "tts_synth":
                self.tts_synth_Start_time = now
                if self.stage_total_units > 0:
                    self.current_book_chunks = self.stage_total_units
                if initial_completed > 0:
                    self.tts_cached_chunks = initial_completed

        unit_desc = f" ({self.stage_completed_units}/{self.stage_total_units} {self.active_stage_unit})" if self.stage_total_units > 0 else ""
        self.log("INFO", f"Stage started: {spec.label}{unit_desc}")
        self._trigger_redraw()

    def set_stage_total(self, total_units: int, completed_units: Optional[int] = None) -> None:
        if emit_ipc_event("set_stage_total", total_units=total_units, completed_units=completed_units):
            return
        now = time.time()
        with self._lock:
            self.stage_total_units = max(0, int(total_units))
            if completed_units is not None:
                self.stage_completed_units = max(0, int(completed_units))
                self.stage_initial_units = self.stage_completed_units
                self.stage_samples.clear()
                self.stage_samples.append((now, float(self.stage_completed_units)))
            if self.active_stage_key in ("mineru", "md_tts", "dom_prep", "smil_assembly") and self.stage_total_units > 0:
                self.current_book_pages = max(self.current_book_pages, self.stage_total_units)
            elif self.active_stage_key == "tts_synth" and self.stage_total_units > 0:
                self.current_book_chunks = self.stage_total_units
        self._trigger_redraw()

    def advance(
        self,
        n: int = 1,
        audio_sec: float = 0.0,
        words: int = 0,
        cached: bool = False,
        retry: bool = False,
    ) -> None:
        if emit_ipc_event(
            "advance",
            n=n,
            audio_sec=audio_sec,
            words=words,
            cached=cached,
            retry=retry,
        ):
            return

        now = time.time()
        with self._lock:
            if retry:
                self.book_retries += 1
                self.total_retries += 1
            if n > 0:
                self.stage_completed_units += int(n)
                if self.stage_total_units > 0 and self.stage_completed_units > self.stage_total_units:
                    self.stage_total_units = self.stage_completed_units
                if not cached:
                    self.stage_samples.append((now, float(self.stage_completed_units)))
                elif len(self.stage_samples) == 1:
                    # Reset baseline if initial cached items were loaded rapidly
                    self.stage_samples[0] = (now, float(self.stage_completed_units))

            if audio_sec > 0:
                self.tts_audio_produced_sec += float(audio_sec)
            if words > 0:
                self.tts_words_done += int(words)
            if self.active_stage_key == "tts_synth" and n > 0:
                if cached:
                    self.tts_cached_chunks += int(n)
                else:
                    self.tts_synthesized_chunks += int(n)

        self._maybe_emit_plain_heartbeat(now)
        self._trigger_redraw()

    def log(self, level: str, msg: str) -> None:
        if emit_ipc_event("log", level=level, msg=msg):
            return

        lvl = (level or "INFO").upper().strip()
        if lvl in ("WARNING", "WARN"):
            lvl = "WARN"
        elif lvl in ("ERR", "FATAL", "CRITICAL", "ERROR"):
            lvl = "ERROR"
        elif lvl in ("SUCCESS", "STEP", "OK"):
            lvl = "INFO"
        elif lvl not in ("DEBUG", "INFO", "WARN", "ERROR"):
            lvl = "INFO"

        clean_msg = strip_formatting(str(msg)).strip()
        if not clean_msg:
            return

        now_dt = datetime.datetime.now()
        ts_short = now_dt.strftime("%H:%M:%S")

        with self._lock:
            if lvl == "WARN":
                self.book_warnings += 1
                self.total_warnings += 1

            self._write_log_file(lvl, clean_msg, now_dt=now_dt)

            # Filter log pane / plain output according to --quiet / --verbose
            if lvl == "DEBUG" and not self.verbose:
                return
            if self.quiet and lvl in ("DEBUG", "INFO"):
                return

            self.log_lines.append((ts_short, lvl, clean_msg))

        if self.plain:
            self._emit_plain_line(lvl, clean_msg, now_dt=now_dt)
        else:
            self._trigger_redraw()

    def end_stage(self, status: str = "done") -> None:
        if emit_ipc_event("end_stage", status=status):
            return

        now = time.time()
        with self._lock:
            if not self.active_stage_key:
                return
            dur = max(0.0, now - self.stage_start_time) if self.stage_start_time > 0 else 0.0
            self.stage_durations[self.active_stage_key] = dur
            self.stage_statuses[self.active_stage_key] = status
            if status == "done" and self.stage_total_units > 0:
                self.stage_completed_units = self.stage_total_units
            if status == "done" and self.current_book_pages > 0 and dur > 0.5:
                self.historical_stage_sec_per_page[self.active_stage_key] = dur / float(self.current_book_pages)
            stage_label = self.active_stage_label

        self.log("INFO", f"Completed {stage_label} in {format_duration(dur)}")
        self._trigger_redraw()

    def end_book(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if emit_ipc_event("end_book", summary=summary):
            return

        now = time.time()
        summary = summary or {}
        with self._lock:
            bname = summary.get("name") or self.current_book_name or "Unknown Book"
            status = summary.get("status", "done").lower()
            total_dur = float(summary.get("duration_sec") or (now - self.current_book_start_time if self.current_book_start_time > 0 else 0.0))
            audio_dur = float(summary.get("audio_duration_sec") or self.tts_audio_produced_sec or 0.0)
            out_path = str(summary.get("output_path") or self.last_output_path or f"chatterbox/audiobooks/{bname}.m4b")
            retries = int(summary.get("retries", self.book_retries))
            warns = int(summary.get("warnings", self.book_warnings))
            failed_stage = str(summary.get("failed_stage") or (self.active_stage_label if status == "failed" else ""))

            tts_wall = self.stage_durations.get("tts_synth", 0.0)
            if tts_wall <= 0 and self.tts_synth_Start_time > 0:
                tts_wall = max(0.001, now - self.tts_synth_Start_time)
            rtf = summary.get("rtf")
            if rtf is None and audio_dur > 0 and tts_wall > 0:
                rtf = audio_dur / tts_wall

            if status == "done":
                self.books_done += 1
                for k, v in list(self.stage_statuses.items()):
                    if v == "active":
                        self.stage_statuses[k] = "done"
            elif status == "skipped":
                self.books_skipped += 1
            else:
                self.books_failed += 1
                if self.active_stage_key:
                    self.stage_statuses[self.active_stage_key] = "failed"

            rec = BookRecord(
                name=bname,
                status=status,
                duration_sec=total_dur,
                audio_duration_sec=audio_dur,
                rtf=rtf,
                retries=retries,
                warnings=warns,
                output_path=out_path,
                failed_stage=failed_stage,
            )
            self.book_records.append(rec)

            rtf_str = f"{rtf:.2f}x" if rtf is not None and rtf > 0 else "n/a"
            audio_str = format_mmss(audio_dur) if audio_dur > 0 else "n/a"
            if status == "done":
                line = (
                    f"✔ {bname} | time: {format_duration(total_dur)} | audio: {audio_str} | "
                    f"RTF: {rtf_str} | retries: {retries} | warnings: {warns} | out: {out_path}"
                )
            elif status == "skipped":
                line = f"↷ {bname} | skipped (already converted) | out: {out_path}"
            else:
                line = (
                    f"✖ {bname} | FAILED at {failed_stage or 'pipeline'} after {format_duration(total_dur)} | "
                    f"retries: {retries} | warnings: {warns}"
                )
            self.book_summaries.append(line)

        self._write_log_file("INFO", f"Book Summary: {line}")
        if self.plain:
            self._emit_plain_raw(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [BOOK-SUMMARY] {line}")
        else:
            self.log("INFO" if status != "failed" else "ERROR", line)
            self._trigger_redraw()

    def end_run(self, aborted: bool = False, abort_reason: Optional[str] = None) -> None:
        if emit_ipc_event("end_run", aborted=aborted, abort_reason=abort_reason):
            return

        with self._lock:
            if not self.run_active:
                return
            self.run_active = False
            total_run_sec = max(0.0, time.time() - self.run_start_time) if self.run_start_time > 0 else 0.0
            aborted_book = self.current_book_name
            aborted_stage = self.active_stage_label

        self._refresh_stop.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=1.5)

        self._uninstall_stream_hooks()

        if self._live is not None:
            try:
                self._trigger_redraw()
                self._live.stop()
            except Exception:
                pass
            self._live = None

        self._poller.stop()

        self._print_final_summary(
            total_run_sec=total_run_sec,
            aborted=aborted,
            abort_reason=abort_reason,
            aborted_book=aborted_book,
            aborted_stage=aborted_stage,
        )

    # -------------------------------------------------------------------------
    # Internal Computations (ETA, RTF, Next Book)
    # -------------------------------------------------------------------------

    def _compute_current_etas(self, now: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Return (stage_eta_sec, book_eta_sec, rolling_rate_units_per_sec) under lock."""
        rem_units = max(0, self.stage_total_units - self.stage_completed_units)
        rate = compute_rolling_rate(self.stage_samples, min_samples=3, window_sec=60.0, min_elapsed=0.25)
        stage_eta = compute_eta(self.stage_samples, rem_units, min_samples=3, window_sec=60.0, min_elapsed=0.25)

        if stage_eta is None:
            return None, None, rate

        # Estimate remaining stages for the current book
        est_pages = max(1, self.current_book_pages or 50)
        rem_stages_sec = 0.0
        seen_active = False
        for s in PIPELINE_STAGES:
            if s.key == self.active_stage_key:
                seen_active = True
                continue
            if not seen_active:
                continue
            if self.stage_statuses.get(s.key) == "pending":
                sec_per_pg = self.historical_stage_sec_per_page.get(s.key, s.weight_sec_per_page)
                rem_stages_sec += sec_per_pg * est_pages

        return stage_eta, stage_eta + rem_stages_sec, rate

    def _compute_tts_metrics(self, now: float) -> Tuple[Optional[float], Optional[float]]:
        """Return (rtf, estimated_total_audio_sec) under lock."""
        if self.tts_audio_produced_sec <= 0:
            return None, None
        tts_wall = self.stage_durations.get("tts_synth", 0.0)
        if self.active_stage_key == "tts_synth" and self.tts_synth_Start_time > 0:
            tts_wall = max(0.001, now - self.tts_synth_Start_time)
        rtf = (self.tts_audio_produced_sec / tts_wall) if tts_wall > 0.1 else None

        est_total_audio: Optional[float] = None
        if self.stage_total_units > 0 and self.stage_completed_units > 0:
            avg_chunk_audio = self.tts_audio_produced_sec / max(1, self.tts_synthesized_chunks or self.stage_completed_units)
            est_total_audio = avg_chunk_audio * self.stage_total_units
        return rtf, est_total_audio

    def _next_book_info(self) -> Tuple[str, int]:
        """Return (next_book_name, remaining_books_count) under lock."""
        total = len(self.queue)
        if total == 0 or self.current_book_idx <= 0:
            return ("None", 0)
        rem = max(0, total - self.current_book_idx)
        if rem > 0 and self.current_book_idx < total:
            return (self.queue[self.current_book_idx], rem)
        return ("None (last book in queue)", 0)

    # -------------------------------------------------------------------------
    # Rendering (Rich Live + ANSI Fallback + Plain Heartbeat)
    # -------------------------------------------------------------------------

    def _refresh_loop(self) -> None:
        while not self._refresh_stop.wait(0.5):
            now = time.time()
            if self.plain:
                self._maybe_emit_plain_heartbeat(now)
            else:
                self._trigger_redraw()

    def _maybe_emit_plain_heartbeat(self, now: float) -> None:
        if not self.plain or not self.run_active or not self.current_book_name:
            return
        if (now - self._last_heartbeat_time) < self.heartbeat_sec:
            return
        self._last_heartbeat_time = now
        gpus, _ = self._poller.snapshot()
        with self._lock:
            stage_eta, book_eta, _ = self._compute_current_etas(now)
            rtf, _ = self._compute_tts_metrics(now)
            elapsed = max(0.0, now - self.stage_start_time) if self.stage_start_time > 0 else 0.0
            pos_str = f"{self.current_book_idx} of {max(1, len(self.queue))}"
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            hb_line = format_plain_heartbeat(
                timestamp=ts,
                book_name=self.current_book_name,
                book_pos=pos_str,
                stage_name=self.active_stage_label,
                completed=self.stage_completed_units,
                total=self.stage_total_units,
                eta_sec=stage_eta,
                book_eta_sec=book_eta,
                elapsed_sec=elapsed,
                gpus=gpus,
                rtf=rtf,
            )
        self._write_log_file("HEARTBEAT", hb_line)
        self._emit_plain_raw(hb_line)

    def _trigger_redraw(self) -> None:
        if self.plain or not self.run_active:
            return
        try:
            if self._live is not None and RICH_AVAILABLE:
                self._live.update(self._build_rich_renderable())
            else:
                self._render_ansi_fallback()
        except Exception as e:
            # Never crash the pipeline on UI error; fall back to plain logging
            self._write_log_file("WARN", f"Display redraw error ({e}); switching to plain mode.")
            self.plain = True
            if self._live is not None:
                try:
                    self._live.stop()
                except Exception:
                    pass
                self._live = None

    def _build_rich_renderable(self) -> Any:
        now = time.time()
        gpus, host = self._poller.snapshot()

        with self._lock:
            stage_eta, book_eta, rate = self._compute_current_etas(now)
            rtf, est_total_audio = self._compute_tts_metrics(now)
            next_book, rem_books = self._next_book_info()
            stage_elapsed = max(0.0, now - self.stage_start_time) if self.stage_start_time > 0 else 0.0
            book_elapsed = max(0.0, now - self.current_book_start_time) if self.current_book_start_time > 0 else 0.0
            run_elapsed = max(0.0, now - self.run_start_time) if self.run_start_time > 0 else 0.0
            pos_str = f"{self.current_book_idx} of {max(1, len(self.queue))}" if self.current_book_idx > 0 else f"0 of {len(self.queue)}"
            cur_book = self.current_book_name or "Waiting for queue..."

            # 1. Hardware & Host Header Table
            hw_table = Table.grid(expand=True, padding=(0, 1))
            hw_table.add_column(ratio=1)
            for g in gpus:
                if not g.available or g.mem_used_mib is None or not g.mem_total_mib:
                    hw_table.add_row(Text(f"GPU {g.index}: n/a (driver/telemetry unavailable)", style="yellow"))
                    continue
                mem_pct = g.mem_pct or 0.0
                temp_val = g.temp_c
                if temp_val is None:
                    temp_markup = "[dim]n/a[/dim]"
                elif temp_val >= self.temp_crit_c:
                    temp_markup = f"[bold red]{temp_val:.0f}°C (!)[/bold red]"
                elif temp_val >= self.temp_warn_c:
                    temp_markup = f"[bold yellow]{temp_val:.0f}°C[/bold yellow]"
                else:
                    temp_markup = f"[bold green]{temp_val:.0f}°C[/bold green]"

                util_str = f"{g.util_pct:.0f}%" if g.util_pct is not None else "n/a"
                pwr_str = (
                    f"{g.power_draw_w:.0f}/{g.power_limit_w:.0f}W"
                    if g.power_draw_w is not None and g.power_limit_w is not None
                    else "n/a"
                )
                clk_str = f"{g.sm_clock_mhz}MHz" if g.sm_clock_mhz is not None else "n/a"
                thr_style = "green" if g.throttle_reasons in ("OK", "IDLE") else "bold red"
                gpu_line = (
                    f"[bold cyan]GPU {g.index}[/bold cyan] ({g.name}): "
                    f"VRAM [bold]{g.mem_used_mib:.0f} / {g.mem_total_mib:.0f} MiB[/bold] ({mem_pct:.1f}%) │ "
                    f"Temp {temp_markup} │ Util [bold]{util_str}[/bold] │ "
                    f"Pwr {pwr_str} │ SM {clk_str} │ [{thr_style}]{g.throttle_reasons}[/{thr_style}]"
                )
                hw_table.add_row(Text.from_markup(gpu_line))

            cpu_str = f"{host.cpu_pct:.1f}%" if host.cpu_pct is not None else "n/a"
            ram_str = (
                f"{host.ram_used_gib:.1f}/{host.ram_total_gib:.1f} GiB ({host.ram_pct:.0f}%)"
                if host.ram_used_gib is not None and host.ram_total_gib is not None and host.ram_pct is not None
                else "n/a"
            )
            if host.disk_free_gb is not None:
                disk_style = "bold red" if host.disk_free_gb < self.disk_warn_gb else "green"
                disk_str = f"[{disk_style}]{host.disk_free_gb:.1f} GB free[/{disk_style}]"
            else:
                disk_str = "n/a"

            host_line = (
                f"[bold magenta]Host[/bold magenta]: CPU [bold]{cpu_str}[/bold] │ "
                f"RAM [bold]{ram_str}[/bold] │ Disk {disk_str} │ "
                f"Run Elapsed: [bold]{format_duration(run_elapsed)}[/bold]"
            )
            hw_table.add_row(Text.from_markup(host_line))

            # 2. Queue & Book Info Table
            q_table = Table.grid(expand=True, padding=(0, 1))
            q_table.add_column(width=16, style="cyan")
            q_table.add_column(ratio=1)
            q_table.add_row("Current Book:", Text.from_markup(f"[bold white]{cur_book}[/bold white]  [bold yellow]({pos_str})[/bold yellow]"))
            q_table.add_row(
                "Next in Queue:",
                Text.from_markup(
                    f"[white]{next_book}[/white]  [dim]({rem_books} remaining │ "
                    f"done: {self.books_done}, failed: {self.books_failed}, skipped: {self.books_skipped})[/dim]"
                ),
            )

            # 3. Stages & Progress Panel
            stage_table = Table(box=SIMPLE, expand=True, show_header=True, padding=(0, 1))
            stage_table.add_column("Stage", ratio=3)
            stage_table.add_column("Status", width=12)
            stage_table.add_column("Progress", ratio=3)
            stage_table.add_column("Elapsed", width=11, justify="right")

            for spec in PIPELINE_STAGES:
                st = self.stage_statuses.get(spec.key, "pending")
                if st == "active":
                    s_label = f"[bold cyan]▶ {spec.label}[/bold cyan]"
                    s_status = "[bold cyan]ACTIVE[/bold cyan]"
                    pct = (self.stage_completed_units / self.stage_total_units * 100.0) if self.stage_total_units > 0 else 0.0
                    bar_w = 18
                    filled = int(round((pct / 100.0) * bar_w)) if self.stage_total_units > 0 else 0
                    bar_str = "█" * max(0, min(bar_w, filled)) + "░" * max(0, bar_w - filled)
                    if self.stage_total_units > 0:
                        prog_str = f"[cyan]{bar_str}[/cyan] [bold]{self.stage_completed_units}/{self.stage_total_units}[/bold] {self.active_stage_unit} ({pct:.1f}%)"
                    else:
                        prog_str = f"[cyan]{self.stage_completed_units}[/cyan] {self.active_stage_unit}"
                    el_str = f"[bold]{format_duration(stage_elapsed)}[/bold]"
                elif st == "done":
                    s_label = f"[green]✔ {spec.label}[/green]"
                    s_status = "[green]DONE[/green]"
                    prog_str = "[green]100%[/green]"
                    el_str = format_duration(self.stage_durations.get(spec.key, 0.0))
                elif st == "failed":
                    s_label = f"[bold red]✖ {spec.label}[/bold red]"
                    s_status = "[bold red]FAILED[/bold red]"
                    prog_str = f"[red]{self.stage_completed_units}/{self.stage_total_units}[/red]"
                    el_str = format_duration(self.stage_durations.get(spec.key, stage_elapsed))
                else:
                    s_label = f"[dim]  {spec.label}[/dim]"
                    s_status = "[dim]PENDING[/dim]"
                    prog_str = "[dim]—[/dim]"
                    el_str = "[dim]—[/dim]"

                stage_table.add_row(
                    Text.from_markup(s_label),
                    Text.from_markup(s_status),
                    Text.from_markup(prog_str),
                    Text.from_markup(el_str),
                )

            # Metrics footer inside stage section
            rate_str = f"{rate:.2f} {self.active_stage_unit}/s" if rate is not None else "calculating..."
            rtf_str = f"{rtf:.2f}x" if rtf is not None else "n/a"
            if self.tts_audio_produced_sec > 0:
                audio_len_str = f"{format_mmss(self.tts_audio_produced_sec)} / ~{format_mmss(est_total_audio)}"
            else:
                audio_len_str = "n/a"

            warn_color = "bold yellow" if self.book_warnings > 0 else "dim"
            retry_color = "bold yellow" if self.book_retries > 0 else "dim"
            metrics_line = (
                f"[bold]Stage ETA:[/bold] [cyan]{format_duration(stage_eta)}[/cyan] │ "
                f"[bold]Book ETA:[/bold] [cyan]{format_duration(book_eta)}[/cyan] │ "
                f"[bold]Book Elapsed:[/bold] {format_duration(book_elapsed)} │ "
                f"[bold]Throughput:[/bold] {rate_str}\n"
                f"[bold]TTS RTF:[/bold] [magenta]{rtf_str}[/magenta] │ "
                f"[bold]Audio Produced:[/bold] {audio_len_str} │ "
                f"[bold]Retries:[/bold] [{retry_color}]{self.book_retries}[/{retry_color}] │ "
                f"[bold]Warnings:[/bold] [{warn_color}]{self.book_warnings}[/{warn_color}]"
            )

            # 4. Recent Log Pane (last ~10 lines)
            log_table = Table.grid(expand=True, padding=(0, 1))
            log_table.add_column(width=10, style="dim")
            log_table.add_column(width=7)
            log_table.add_column(ratio=1)
            recent_logs = list(self.log_lines)[-10:]
            if not recent_logs:
                log_table.add_row("", "", Text("Waiting for pipeline events...", style="dim"))
            else:
                for ts_s, lvl, lmsg in recent_logs:
                    lvl_style = {
                        "DEBUG": "dim",
                        "INFO": "cyan",
                        "WARN": "bold yellow",
                        "ERROR": "bold red",
                    }.get(lvl, "white")
                    log_table.add_row(ts_s, Text(f"[{lvl}]", style=lvl_style), Text(lmsg, overflow="ellipsis", no_wrap=True))

            # Optional Completed Book Summaries Panel
            parts: List[Any] = [
                Panel(hw_table, title="[bold cyan]GPU & Host Telemetry[/bold cyan]", border_style="cyan", box=ROUNDED, padding=(0, 1)),
                Panel(q_table, title="[bold blue]Book Queue[/bold blue]", border_style="blue", box=ROUNDED, padding=(0, 1)),
                Panel(
                    Group(stage_table, Text.from_markup(metrics_line)),
                    title="[bold green]Pipeline Stages & Live ETA[/bold green]",
                    border_style="green",
                    box=ROUNDED,
                    padding=(0, 1),
                ),
            ]
            if self.book_summaries:
                sum_text = "\n".join(self.book_summaries)
                parts.append(
                    Panel(Text(sum_text), title="[bold magenta]Completed Books[/bold magenta]", border_style="magenta", box=ROUNDED, padding=(0, 1))
                )
            parts.append(
                Panel(
                    log_table,
                    title=f"[bold white]Recent Logs (last 10) — Full log: {self.log_file_path.name}[/bold white]",
                    border_style="white",
                    box=ROUNDED,
                    padding=(0, 1),
                )
            )
            return Group(*parts)

    def _render_ansi_fallback(self) -> None:
        """Minimal ANSI redraw fallback when `rich` is not installed."""
        now = time.time()
        gpus, host = self._poller.snapshot()
        with self._lock:
            stage_eta, book_eta, rate = self._compute_current_etas(now)
            rtf, _ = self._compute_tts_metrics(now)
            next_book, rem_books = self._next_book_info()
            stage_elapsed = max(0.0, now - self.stage_start_time) if self.stage_start_time > 0 else 0.0
            pos_str = f"{self.current_book_idx} of {max(1, len(self.queue))}"

            lines: List[str] = []
            lines.append("=" * 88)
            for g in gpus:
                if g.available and g.mem_used_mib is not None and g.mem_total_mib:
                    lines.append(
                        f"GPU {g.index} ({g.name}): {g.mem_used_mib:.0f}/{g.mem_total_mib:.0f} MiB "
                        f"({(g.mem_pct or 0):.1f}%) | Temp: {g.temp_c or 0:.0f}°C | Util: {g.util_pct or 0:.0f}%"
                    )
                else:
                    lines.append(f"GPU {g.index}: n/a")
            lines.append(
                f"Host: CPU {host.cpu_pct or 0:.1f}% | RAM {host.ram_used_gib or 0:.1f}/{host.ram_total_gib or 0:.1f} GiB | "
                f"Disk Free: {host.disk_free_gb or 0:.1f} GB"
            )
            lines.append("-" * 88)
            lines.append(f"Current Book ({pos_str}): {self.current_book_name}  |  Next: {next_book} ({rem_books} left)")
            lines.append(
                f"Active Stage: {self.active_stage_label} [{self.stage_completed_units}/{self.stage_total_units} {self.active_stage_unit}]"
            )
            lines.append(
                f"Elapsed: {format_duration(stage_elapsed)} | Stage ETA: {format_duration(stage_eta)} | "
                f"Book ETA: {format_duration(book_eta)} | Rate: {f'{rate:.2f}/s' if rate else 'calculating...'} | "
                f"RTF: {f'{rtf:.2f}x' if rtf else 'n/a'} | Retries: {self.book_retries} | Warns: {self.book_warnings}"
            )
            lines.append("-" * 88)
            for ts_s, lvl, lmsg in list(self.log_lines)[-8:]:
                lines.append(f"  {ts_s} [{lvl:<5}] {lmsg[:72]}")
            lines.append("=" * 88)

            out = ""
            if self._ansi_prev_lines > 0:
                out += f"\033[{self._ansi_prev_lines}A\r"
            out += "\n".join(f"\033[2K{l}" for l in lines) + "\n"
            self._ansi_prev_lines = len(lines)
            self._orig_stdout.write(out)
            self._orig_stdout.flush()

    # -------------------------------------------------------------------------
    # Log File, Stream Capture & Line Parsing
    # -------------------------------------------------------------------------

    def _write_log_file(self, level: str, msg: str, now_dt: Optional[datetime.datetime] = None) -> None:
        dt = now_dt or datetime.datetime.now()
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self.log_file_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] [{level:<5}] {msg}\n")
        except Exception:
            pass

    def _emit_plain_line(self, level: str, msg: str, now_dt: Optional[datetime.datetime] = None) -> None:
        dt = now_dt or datetime.datetime.now()
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            bname = self.current_book_name or None
            sname = self.active_stage_label if self.active_stage_key else None
        line = format_plain_event(ts, level, msg, book=bname, stage=sname)
        self._emit_plain_raw(line)

    def _emit_plain_raw(self, line: str) -> None:
        try:
            self._orig_stdout.write(line + "\n")
            self._orig_stdout.flush()
        except Exception:
            pass

    def _install_stream_hooks(self) -> None:
        if self._streams_hooked:
            return
        self._streams_hooked = True
        sys.stdout = _StreamRedirector(self, level="INFO", orig_stream=self._orig_stdout)  # type: ignore
        sys.stderr = _StreamRedirector(self, level="WARN", orig_stream=self._orig_stderr)  # type: ignore

        def _custom_print(*args: Any, **kwargs: Any) -> None:
            file_arg = kwargs.get("file")
            if file_arg is not None and file_arg not in (sys.stdout, sys.stderr, self._orig_stdout, self._orig_stderr):
                self._orig_print(*args, **kwargs)
                return
            sep = kwargs.get("sep", " ")
            text = sep.join(str(a) for a in args)
            lvl = "WARN" if file_arg in (sys.stderr, self._orig_stderr) else "INFO"
            for ln in text.splitlines():
                if ln.strip():
                    self._handle_captured_line(ln, default_level=lvl)

        builtins.print = _custom_print

        def _custom_showwarning(message: Any, category: Any, filename: Any, lineno: Any, file: Any = None, line: Any = None) -> None:
            cat_name = getattr(category, "__name__", "Warning")
            self.log("WARN", f"{cat_name} ({Path(str(filename)).name}:{lineno}): {message}")

        warnings.showwarning = _custom_showwarning

        root_logger = logging.getLogger()
        self._log_handler = _DisplayLogHandler(self)
        self._log_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root_logger.addHandler(self._log_handler)

    def _uninstall_stream_hooks(self) -> None:
        if not self._streams_hooked:
            return
        self._streams_hooked = False
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        builtins.print = self._orig_print
        warnings.showwarning = self._orig_showwarning
        if self._log_handler is not None:
            try:
                logging.getLogger().removeHandler(self._log_handler)
            except Exception:
                pass
            self._log_handler = None

    def _handle_captured_line(self, raw_line: str, default_level: str = "INFO") -> None:
        """Parse output lines from auto_pipeline.sh / mineru / md_tts / chatterbox and log them."""
        clean = strip_formatting(raw_line).strip()
        if not clean:
            return
        # Ignore pure ASCII separator lines
        if re.match(r"^[=\-_─━]{10,}$", clean):
            return

        # Detect log level from line content
        lvl = default_level
        m_log = re.match(r"^\[[\d\- :]+\]\s*\[(INFO|SUCCESS|WARN|WARNING|ERROR|STEP|DEBUG)\]\s*(.*)$", clean)
        if m_log:
            tag, body = m_log.groups()
            clean = body.strip()
            if tag in ("WARN", "WARNING"):
                lvl = "WARN"
            elif tag == "ERROR":
                lvl = "ERROR"
            elif tag == "DEBUG":
                lvl = "DEBUG"
            else:
                lvl = "INFO"
        elif any(w in clean for w in ("UserWarning:", "FutureWarning:", "DeprecationWarning:", ">>> WARNING:", "[warn]")):
            lvl = "WARN"
        elif any(w in clean for w in ("Traceback (most recent call last)", "FATAL ERROR", ">>> ERROR:", "[error]")):
            lvl = "ERROR"

        # Auto-detect pipeline events from child script output if structured IPC wasn't sent
        self._auto_parse_pipeline_line(clean)
        self.log(lvl, clean)

    def _auto_parse_pipeline_line(self, clean: str) -> None:
        """Heuristically extract stage/unit progress from existing pipeline stdout/stderr lines."""
        # 1. Queue count from auto_pipeline.sh
        m_q = re.search(r"Found (\d+) PDF file\(s\) in queue to process", clean)
        if m_q:
            cnt = int(m_q.group(1))
            with self._lock:
                if len(self.queue) == 0:
                    self.queue = [f"Book {i+1}" for i in range(cnt)]

        # 2. Book start from auto_pipeline.sh
        m_book = re.search(r">>> Processing Book:\s*(.+)$", clean)
        if m_book:
            bname = self._clean_book_label(m_book.group(1))
            if bname != self.current_book_name:
                self.start_book(bname)
            return

        # 3. MinerU stage & page count
        if "[Stage 1/3] Moving to mineru/in/" in clean:
            if self.active_stage_key != "mineru":
                self.start_stage("mineru", total_units=0, unit_name="pages")
            return

        m_tot_pg = re.search(r"Total Pages\s*:\s*(\d+)", clean)
        if m_tot_pg and self.active_stage_key == "mineru":
            self.set_stage_total(int(m_tot_pg.group(1)))
            return

        m_already_pg = re.search(r"Already Processed\s*:\s*(\d+)\s*pages", clean)
        if m_already_pg and self.active_stage_key == "mineru":
            self.set_stage_total(self.stage_total_units, completed_units=int(m_already_pg.group(1)))
            return

        # MinerU worker page completion: "Worker 0 │ Page   12/320 │ SUCCESS"
        m_w_pg = re.search(r"Worker\s+\d+\s*│\s*Page\s+(\d+)/(\d+)\s*│\s*(SUCCESS|FAILED)", clean)
        if m_w_pg:
            _, p_tot, status = m_w_pg.groups()
            if self.active_stage_key != "mineru":
                self.start_stage("mineru", total_units=int(p_tot), unit_name="pages")
            elif self.stage_total_units == 0:
                self.set_stage_total(int(p_tot))
            if status == "SUCCESS":
                self.advance(1)
            else:
                self.advance(0, retry=True)
            return

        # 4. Stage 2: TTS_Script md_tts.py
        if "[Stage 2/3] Stripping Markdown" in clean or "--- Running Stage 2 (TTS_Script)" in clean:
            if self.active_stage_key != "md_tts":
                self.start_stage("md_tts", total_units=self.current_book_pages, unit_name="pages")
            return

        if self.active_stage_key == "md_tts" and "  ->  " in clean and clean.endswith(".txt"):
            self.advance(1)
            return

        # 5. Stage 3: Chatterbox sub-stages
        if "Extracting TOC mapping and wrapping page DOM spans" in clean:
            if self.active_stage_key != "dom_prep":
                self.start_stage("dom_prep", total_units=self.current_book_pages, unit_name="pages")
            return

        m_prep = re.search(r"Prepared\s+(\d+)\s+page\(s\).*?([\d,]+)\s+spoken words,\s+wrapped into\s+(\d+)\s+DOM span chunk", clean)
        if m_prep:
            pgs = int(m_prep.group(1))
            words = int(m_prep.group(2).replace(",", ""))
            chks = int(m_prep.group(3))
            with self._lock:
                self.current_book_pages = pgs
                self.current_book_chunks = chks
                self.tts_total_words = words
            if self.active_stage_key == "dom_prep":
                self.set_stage_total(pgs, completed_units=pgs)
                self.end_stage("done")
            self.start_stage("tts_synth", total_units=chks, unit_name="chunks")
            return

        m_cached = re.search(r"Found\s+(\d+)\s+cache-verified chunks already synthesized", clean)
        if m_cached and self.active_stage_key == "tts_synth":
            c_cnt = int(m_cached.group(1))
            self.set_stage_total(self.stage_total_units, completed_units=c_cnt)
            with self._lock:
                self.tts_cached_chunks = c_cnt
            return

        if "Computing sample-accurate SMIL timings" in clean:
            if self.active_stage_key != "smil_assembly":
                self.start_stage("smil_assembly", total_units=self.current_book_pages, unit_name="pages")
            return

        if "Packaging synchronized EPUB3 with Media Overlays" in clean or "Muxing standalone audiobook from per-page tracks" in clean:
            if self.active_stage_key != "epub_m4b_mux":
                self.start_stage("epub_m4b_mux", total_units=3, unit_name="steps")
            else:
                self.advance(1)
            return

        m_saved = re.search(r"Audiobook synthesized successfully:\s*(.+)$", clean)
        if m_saved:
            with self._lock:
                self.last_output_path = m_saved.group(1).strip()
            return

        if "--- [3/3] Uploading to AudioBookshelf Server ---" in clean:
            if self.active_stage_key != "abs_upload":
                self.start_stage("abs_upload", total_units=1, unit_name="steps")
            return

        if "Uploaded successfully to AudioBookshelf" in clean:
            if self.active_stage_key == "abs_upload":
                self.advance(1)
                self.end_stage("done")
            return

    def _print_final_summary(
        self,
        total_run_sec: float,
        aborted: bool,
        abort_reason: Optional[str],
        aborted_book: str,
        aborted_stage: str,
    ) -> None:
        status_hdr = "PIPELINE RUN ABORTED" if aborted else "PIPELINE RUN SUMMARY"
        self._write_log_file("ERROR" if aborted else "INFO", f"=== {status_hdr} ===")
        if aborted:
            abort_msg = (
                f"Aborted during Book: '{aborted_book or 'n/a'}' | "
                f"Stage: '{aborted_stage or 'n/a'}'"
                + (f" | Reason: {abort_reason}" if abort_reason else "")
            )
            self._write_log_file("ERROR", abort_msg)

        summary_lines = [
            f"Total Run Time : {format_duration(total_run_sec)}",
            f"Books Queued   : {len(self.queue)}",
            f"Books Done     : {self.books_done}",
            f"Books Failed   : {self.books_failed}",
            f"Books Skipped  : {self.books_skipped}",
            f"Total Retries  : {self.total_retries}",
            f"Total Warnings : {self.total_warnings}",
            f"Full Run Log   : {self.log_file_path}",
        ]
        for sl in summary_lines:
            self._write_log_file("INFO", sl)

        if RICH_AVAILABLE and not self.plain and self._console is not None:
            border = "red" if (aborted or self.books_failed > 0) else "green"
            body_lines: List[str] = []
            if aborted:
                body_lines.append(
                    f"[bold red]✖ Run Aborted![/bold red]\n"
                    f"  • Failed/Active Book  : [bold yellow]{aborted_book or 'n/a'}[/bold yellow]\n"
                    f"  • Failed/Active Stage : [bold yellow]{aborted_stage or 'n/a'}[/bold yellow]"
                    + (f"\n  • Reason              : {abort_reason}" if abort_reason else "")
                    + "\n"
                )
            elif self.books_failed > 0:
                failed_list = [r for r in self.book_records if r.status == "failed"]
                for fr in failed_list:
                    body_lines.append(
                        f"[bold red]✖ Failed Book:[/bold red] {fr.name} (Stage: {fr.failed_stage or 'unknown'})"
                    )
                body_lines.append("")

            for sl in summary_lines:
                k, v = sl.split(":", 1)
                body_lines.append(f"[cyan]{k}:[/cyan][bold white]{v}[/bold white]")

            if self.book_summaries:
                body_lines.append("\n[bold magenta]Per-Book Results:[/bold magenta]")
                for bs in self.book_summaries:
                    body_lines.append(f"  {bs}")

            self._console.print(
                Panel(
                    "\n".join(body_lines),
                    title=f"[bold {border}]{status_hdr}[/bold {border}]",
                    border_style=border,
                    box=ROUNDED,
                    padding=(1, 2),
                )
            )
        else:
            sep = "=" * 80
            out_lines = [sep, f" {status_hdr}", sep]
            if aborted:
                out_lines.append(f"  ABORTED AT BOOK : {aborted_book or 'n/a'}")
                out_lines.append(f"  ABORTED AT STAGE: {aborted_stage or 'n/a'}")
                if abort_reason:
                    out_lines.append(f"  ABORT REASON    : {abort_reason}")
                out_lines.append("-" * 80)
            for sl in summary_lines:
                out_lines.append(f"  {sl}")
            if self.book_summaries:
                out_lines.append("-" * 80)
                for bs in self.book_summaries:
                    out_lines.append(f"  {bs}")
            out_lines.append(sep)
            self._emit_plain_raw("\n".join(out_lines))

    @staticmethod
    def _clean_book_label(path_or_name: str) -> str:
        name = Path(path_or_name).name
        for suf in (".pdf", ".PDF", "_txt", "_md", "_pages", ".m4b", ".epub"):
            if name.endswith(suf):
                name = name[: -len(suf)]
        return name.strip() or str(path_or_name)


# =============================================================================
# Module-Level Singleton & Clean Functional Interface
# =============================================================================

_GLOBAL_DISPLAY: Optional[StatusDisplay] = None


def get_display() -> Optional[StatusDisplay]:
    return _GLOBAL_DISPLAY


def set_display(display: Optional[StatusDisplay]) -> None:
    global _GLOBAL_DISPLAY
    _GLOBAL_DISPLAY = display


def start_run(queue: Sequence[str], **kwargs: Any) -> StatusDisplay:
    global _GLOBAL_DISPLAY
    if _GLOBAL_DISPLAY is None:
        _GLOBAL_DISPLAY = StatusDisplay(**kwargs)
    _GLOBAL_DISPLAY.start_run(queue)
    return _GLOBAL_DISPLAY


def start_book(
    name: str,
    index: Optional[int] = None,
    total: Optional[int] = None,
    next_book: Optional[str] = None,
) -> None:
    if _GLOBAL_DISPLAY is not None:
        _GLOBAL_DISPLAY.start_book(name, index=index, total=total, next_book=next_book)
    else:
        emit_ipc_event("start_book", name=name, index=index, total=total, next_book=next_book)


def start_stage(
    name: str,
    total_units: int = 0,
    unit_name: Optional[str] = None,
    initial_completed: int = 0,
) -> None:
    if _GLOBAL_DISPLAY is not None:
        _GLOBAL_DISPLAY.start_stage(
            name,
            total_units=total_units,
            unit_name=unit_name,
            initial_completed=initial_completed,
        )
    else:
        emit_ipc_event(
            "start_stage",
            name=name,
            total_units=total_units,
            unit_name=unit_name,
            initial_completed=initial_completed,
        )


def advance(
    n: int = 1,
    audio_sec: float = 0.0,
    words: int = 0,
    cached: bool = False,
    retry: bool = False,
) -> None:
    if _GLOBAL_DISPLAY is not None:
        _GLOBAL_DISPLAY.advance(n=n, audio_sec=audio_sec, words=words, cached=cached, retry=retry)
    else:
        emit_ipc_event("advance", n=n, audio_sec=audio_sec, words=words, cached=cached, retry=retry)


def log(level: str, msg: str) -> None:
    if _GLOBAL_DISPLAY is not None:
        _GLOBAL_DISPLAY.log(level, msg)
    else:
        emit_ipc_event("log", level=level, msg=msg)


def end_stage(status: str = "done") -> None:
    if _GLOBAL_DISPLAY is not None:
        _GLOBAL_DISPLAY.end_stage(status=status)
    else:
        emit_ipc_event("end_stage", status=status)


def end_book(summary: Optional[Dict[str, Any]] = None) -> None:
    if _GLOBAL_DISPLAY is not None:
        _GLOBAL_DISPLAY.end_book(summary=summary)
    else:
        emit_ipc_event("end_book", summary=summary)


def end_run(aborted: bool = False, abort_reason: Optional[str] = None) -> None:
    global _GLOBAL_DISPLAY
    if _GLOBAL_DISPLAY is not None:
        _GLOBAL_DISPLAY.end_run(aborted=aborted, abort_reason=abort_reason)
        _GLOBAL_DISPLAY = None
    else:
        emit_ipc_event("end_run", aborted=aborted, abort_reason=abort_reason)


# =============================================================================
# Supervisor Runner for auto_convert.sh / auto_pipeline.sh
# =============================================================================

def run_pipeline_supervisor(
    pipeline_cmd: List[str],
    plain: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    heartbeat_sec: float = 30.0,
) -> int:
    """Execute auto_pipeline.sh under StatusDisplay supervision, consuming both IPC events and stdout/stderr."""
    base_dir = Path(__file__).resolve().parent
    disp = StatusDisplay(
        log_dir=base_dir / "logs",
        output_dir=base_dir / "chatterbox" / "audiobooks",
        plain=plain,
        quiet=quiet,
        verbose=verbose,
        heartbeat_sec=heartbeat_sec,
    )
    set_display(disp)

    # Create temporary IPC event file for child bash/python scripts
    event_file = base_dir / "logs" / f".status_events_{os.getpid()}.jsonl"
    event_file.parent.mkdir(parents=True, exist_ok=True)
    event_file.write_text("", encoding="utf-8")

    # Pre-scan queue in `in/` so initial queue is visible immediately
    in_dir = base_dir / "in"
    initial_queue: List[str] = []
    if in_dir.is_dir():
        for f in sorted(in_dir.iterdir()):
            if f.is_file() and f.suffix.lower() == ".pdf" and not f.name.startswith("."):
                initial_queue.append(f.stem)

    disp.start_run(initial_queue, hook_streams=False)

    env = os.environ.copy()
    env["STATUS_DISPLAY_ACTIVE"] = "1"
    env["STATUS_DISPLAY_EVENT_FILE"] = str(event_file)
    env["PYTHONUNBUFFERED"] = "1"

    proc: Optional[subprocess.Popen] = None
    aborted = False
    abort_reason: Optional[str] = None

    def _sig_handler(signum: int, _frame: Any) -> None:
        nonlocal aborted, abort_reason
        sig_name = "SIGINT (Ctrl-C)" if signum == signal.SIGINT else f"Signal {signum}"
        aborted = True
        abort_reason = f"Interrupted by user ({sig_name})"
        disp.log("WARN", f"Received {sig_name} — stopping child processes and restoring terminal...")
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    orig_sigint = signal.getsignal(signal.SIGINT)
    orig_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    stop_tail = threading.Event()

    def _tail_events() -> None:
        try:
            with event_file.open("r", encoding="utf-8", errors="replace") as ef:
                while not stop_tail.is_set():
                    line = ef.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    _dispatch_ipc_line(disp, line)
                # Drain remaining
                for line in ef:
                    _dispatch_ipc_line(disp, line)
        except Exception:
            pass

    tail_thread = threading.Thread(target=_tail_events, name="StatusEventTailer", daemon=True)
    tail_thread.start()

    exit_code = 0
    try:
        proc = subprocess.Popen(
            pipeline_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid,
        )
        assert proc.stdout is not None
        for raw_line in iter(proc.stdout.readline, ""):
            if not raw_line:
                break
            disp._handle_captured_line(raw_line.rstrip("\r\n"), default_level="INFO")

        exit_code = proc.wait()
        if exit_code != 0 and not aborted:
            aborted = True
            abort_reason = f"Pipeline exited with code {exit_code}"
    except KeyboardInterrupt:
        aborted = True
        abort_reason = "KeyboardInterrupt (Ctrl-C)"
        exit_code = 130
    finally:
        stop_tail.set()
        tail_thread.join(timeout=1.0)
        try:
            event_file.unlink(missing_ok=True)
        except Exception:
            pass
        signal.signal(signal.SIGINT, orig_sigint)
        signal.signal(signal.SIGTERM, orig_sigterm)
        disp.end_run(aborted=aborted, abort_reason=abort_reason)
        set_display(None)

    return 130 if aborted and exit_code == 0 else exit_code


def _dispatch_ipc_line(disp: StatusDisplay, line: str) -> None:
    try:
        data = json.loads(line.strip())
    except Exception:
        return
    ev = data.get("event")
    if ev == "start_run":
        with disp._lock:
            q = [disp._clean_book_label(x) for x in data.get("queue", [])]
            if q:
                disp.queue = q
        disp._trigger_redraw()
    elif ev == "start_book":
        disp.start_book(
            name=data.get("name", "Book"),
            index=data.get("index"),
            total=data.get("total"),
            next_book=data.get("next_book"),
        )
    elif ev == "start_stage":
        disp.start_stage(
            name=data.get("name", ""),
            total_units=int(data.get("total_units", 0)),
            unit_name=data.get("unit_name"),
            initial_completed=int(data.get("initial_completed", 0)),
        )
    elif ev == "set_stage_total":
        disp.set_stage_total(
            total_units=int(data.get("total_units", 0)),
            completed_units=data.get("completed_units"),
        )
    elif ev == "advance":
        disp.advance(
            n=int(data.get("n", 1)),
            audio_sec=float(data.get("audio_sec", 0.0)),
            words=int(data.get("words", 0)),
            cached=bool(data.get("cached", False)),
            retry=bool(data.get("retry", False)),
        )
    elif ev == "log":
        disp.log(data.get("level", "INFO"), data.get("msg", ""))
    elif ev == "end_stage":
        disp.end_stage(status=data.get("status", "done"))
    elif ev == "end_book":
        disp.end_book(summary=data.get("summary"))


# =============================================================================
# Demo Mode (--demo)
# =============================================================================

def run_demo(
    duration_sec: float = 10.0,
    plain: bool = False,
    quiet: bool = False,
    verbose: bool = False,
    heartbeat_sec: float = 3.0,
) -> int:
    """Simulate a realistic multi-book conversion run with fake Tesla P100 GPU readings."""
    demo_books = [
        "Practical Electronics for Inventors (4th Ed)",
        "Foundations of Bayesian Statistics for Data Scientists",
        "AI and Machine Learning for Coders in PyTorch",
    ]
    disp = StatusDisplay(
        plain=plain,
        quiet=quiet,
        verbose=verbose,
        heartbeat_sec=heartbeat_sec,
        poll_interval_sec=0.5,
        demo_mode=True,
    )
    set_display(disp)

    # Scale step sleep to requested demo duration
    scale = max(0.05, duration_sec / 10.0)
    aborted = False
    abort_reason = None

    try:
        disp.start_run(demo_books, hook_streams=False)
        disp.log("INFO", "Initialized dual Tesla P100-PCIE-16GB conversion pipeline (demo mode)")

        for b_idx, book in enumerate(demo_books[:2], 1):
            next_b = demo_books[b_idx] if b_idx < len(demo_books) else None
            pages = 48 if b_idx == 1 else 36
            chunks = pages * 5

            disp.start_book(book, index=b_idx, total=len(demo_books), next_book=next_b)

            # Stage 1: MinerU PDF Extract
            disp.start_stage("mineru", total_units=pages, unit_name="pages")
            disp.log("DEBUG", f"pdfinfo: {pages} pages detected (Born-Digital layout)")
            for p in range(1, pages + 1):
                time.sleep(0.025 * scale)
                disp.advance(1)
                if p == pages // 2:
                    disp.log("INFO", f"[Worker 0] Page {p}/{pages} extracted (1.2s/page)")
            disp.end_stage("done")

            # Stage 2: Text Clean & Math
            disp.start_stage("md_tts", total_units=pages, unit_name="pages")
            disp.log("INFO", "mathspeak-api healthy on http://localhost:8099 (clearspeak)")
            for _ in range(pages):
                time.sleep(0.01 * scale)
                disp.advance(1)
            disp.log("INFO", f"Verbalized {pages * 3} LaTeX math expressions with 0 errors")
            disp.end_stage("done")

            # Stage 3a: DOM & Chunk Prep
            disp.start_stage("dom_prep", total_units=pages, unit_name="pages")
            for _ in range(pages):
                time.sleep(0.008 * scale)
                disp.advance(1)
            disp.end_stage("done")

            # Stage 3b: Chatterbox TTS Synthesis
            cached = 15 if b_idx == 1 else 0
            disp.start_stage("tts_synth", total_units=chunks, unit_name="chunks", initial_completed=cached)
            if cached > 0:
                disp.log("INFO", f"Reused {cached} cache-verified WAV chunks from audio_chunks/")
            for c in range(cached + 1, chunks + 1):
                time.sleep(0.012 * scale)
                if b_idx == 1 and c == 65:
                    disp.log("WARN", "Chunk 65 had trailing silence artifact; regenerating chunk (retry 1/3)")
                    disp.advance(0, retry=True)
                disp.advance(1, audio_sec=4.2, words=22)
                if c % 60 == 0:
                    disp.log("DEBUG", f"[cuda:{c % 2}] Synthesized chunk {c}/{chunks} (-20 dBFS normalized)")
            disp.end_stage("done")

            # Stage 3c: Audio & SMIL Assembly
            disp.start_stage("smil_assembly", total_units=pages, unit_name="pages")
            for _ in range(pages):
                time.sleep(0.008 * scale)
                disp.advance(1)
            disp.end_stage("done")

            # Stage 3d: EPUB3 & M4B Mux
            disp.start_stage("epub_m4b_mux", total_units=3, unit_name="steps")
            for step_msg in (
                "Packaged synchronized EPUB3 with SMIL 3.0 Media Overlays",
                "Muxed standalone .m4b audiobook with chapter markers and cover art",
                "Generated AudioBookshelf companion .metadata.json and .cover.jpg",
            ):
                time.sleep(0.08 * scale)
                disp.advance(1)
                disp.log("INFO", step_msg)
            disp.end_stage("done")

            # Stage 3e: AudioBookshelf Upload
            disp.start_stage("abs_upload", total_units=1, unit_name="steps")
            time.sleep(0.1 * scale)
            disp.advance(1)
            disp.log("INFO", f"Uploaded '{book}' (.m4b + .epub + .pdf) to AudioBookshelf [Library: TTS]")
            disp.end_stage("done")

            slug = re.sub(r"[^A-Za-z0-9]+", "_", book).strip("_")
            disp.end_book({
                "name": book,
                "status": "done",
                "output_path": f"chatterbox/audiobooks/{slug}.m4b",
            })

        # Mark 3rd demo book as skipped to showcase all summary states
        disp.start_book(demo_books[2], index=3, total=len(demo_books), next_book=None)
        time.sleep(0.15 * scale)
        disp.end_book({
            "name": demo_books[2],
            "status": "skipped",
            "output_path": "chatterbox/audiobooks/AI_and_Machine_Learning_for_Coders_in_PyTorch.m4b",
        })

    except KeyboardInterrupt:
        aborted = True
        abort_reason = "Interrupted by user (Ctrl-C)"
    finally:
        disp.end_run(aborted=aborted, abort_reason=abort_reason)
        set_display(None)

    return 130 if aborted else 0


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live terminal status display for BookConverters PDF-to-Audiobook pipeline."
    )
    parser.add_argument("--demo", action="store_true", help="Run interactive simulation with fake queue and GPU stats.")
    parser.add_argument("--demo-duration", type=float, default=10.0, help="Duration in seconds for --demo mode (default: 10).")
    parser.add_argument("--plain", action="store_true", help="Disable live TUI redraw; emit timestamped lines + periodic heartbeats.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress INFO/DEBUG log lines (show warnings, errors, heartbeats, summaries).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Include DEBUG-level log messages in the log pane.")
    parser.add_argument("--heartbeat", type=float, default=30.0, help="Heartbeat interval in seconds for --plain / non-TTY mode (default: 30).")
    parser.add_argument(
        "--exec-pipeline",
        action="store_true",
        help="Internal supervisor flag: execute the command following '--' under live display supervision.",
    )
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to supervise when using --exec-pipeline.")

    args = parser.parse_args(argv)

    if args.demo:
        return run_demo(
            duration_sec=args.demo_duration,
            plain=args.plain,
            quiet=args.quiet,
            verbose=args.verbose,
            heartbeat_sec=min(args.heartbeat, 3.0),
        )

    if args.exec_pipeline:
        cmd = list(args.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        if not cmd:
            cmd = ["bash", str(Path(__file__).resolve().parent / "auto_pipeline.sh")]
        return run_pipeline_supervisor(
            pipeline_cmd=cmd,
            plain=args.plain,
            quiet=args.quiet,
            verbose=args.verbose,
            heartbeat_sec=args.heartbeat,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
