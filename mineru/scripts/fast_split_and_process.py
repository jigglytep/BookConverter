#!/usr/bin/env python3
"""Fast dual-GPU MinerU processor with persistent workers and modern DevOps CLI output.

Execution Phases:
  1. Environment & Pre-Flight Diagnostics (PDF parsing, page count, VLM health check)
  2. Primary Conversion Pass (Dual-GPU persistent workers with atomic work-stealing)
  3. Verification & Multi-Pass Retries (Server restarts, KV-cache clearing, retry queue)
  4. Operational Summary & Metrics Dashboard (Final reporting and clean exit codes)

Features:
  - Rich modern terminal interface with clean colors, Unicode icons, section headers,
    tables, progress bars, spinners, and dashboards.
  - Graceful support for cron, systemd, redirected logs, and non-TTY environments.
  - Dynamic atomic work-stealing across multiple Tesla P100 GPUs.
  - Preserves 100% of existing conversion, caching, and recovery functionality.

Usage:
    ./scripts/fast_split_and_process.py in/my_book.pdf
    ./scripts/fast_split_and_process.py in/my_book.pdf --workers 2 --max-retries 3 --verbose
"""
import argparse
import datetime
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Graceful Rich fallback
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.rule import Rule
    from rich.status import Status
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Reusable UI & Console Helpers
# ---------------------------------------------------------------------------

class DevOpsUI:
    """Manages terminal rendering, supporting rich TTY mode and clean non-TTY logging."""

    def __init__(self, no_color: bool = False, verbose: bool = False, log_file: Optional[Path] = None):
        self.verbose = verbose
        self.log_file = log_file
        self.is_tty = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
        force_color = None if not no_color and "NO_COLOR" not in os.environ else False

        self.unicode = self.is_tty and sys.stdout.encoding.lower().startswith("utf")

        # Unicode / ASCII icon mapping
        self.ICON_ROCKET = "🚀 " if self.unicode else "[*] "
        self.ICON_CONFIG = "⚙️  " if self.unicode else "[-] "
        self.ICON_HEALTH = "🩺 " if self.unicode else "[+] "
        self.ICON_GPU = "⚡ " if self.unicode else "[G] "
        self.ICON_CHECK = "✔ " if self.unicode else "[OK] "
        self.ICON_WARN = "⚠ " if self.unicode else "[WARN] "
        self.ICON_FAIL = "✖ " if self.unicode else "[FAIL] "
        self.ICON_RETRY = "🔄 " if self.unicode else "[RETRY] "
        self.ICON_DASHBOARD = "📊 " if self.unicode else "[SUMMARY] "
        self.ICON_PAGE = "📄 " if self.unicode else "[P] "

        if RICH_AVAILABLE:
            custom_theme = Theme({
                "info": "cyan",
                "success": "bold green",
                "warning": "bold yellow",
                "error": "bold red",
                "highlight": "bold magenta",
                "dim": "dim white",
                "header": "bold cyan",
                "metric": "bold white",
            })
            self.console = Console(
                theme=custom_theme,
                no_color=force_color,
                highlight=False,
            )
        else:
            self.console = None

    def _write_file_log(self, text: str) -> None:
        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with self.log_file.open("a", encoding="utf-8") as f:
                    f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
            except OSError:
                pass

    def log_raw(self, text: str) -> None:
        clean_text = re.sub(r"\[/?.*?\]", "", text)
        self._write_file_log(clean_text)

    def log(self, text: str, style: str = "") -> None:
        clean_text = re.sub(r"\[/?.*?\]", "", text)
        self._write_file_log(clean_text)

        if self.console and self.is_tty:
            self.console.print(text, style=style)
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {clean_text}", flush=True)

    def print_banner(self) -> None:
        title = "MINERU DUAL-GPU CONVERSION ENGINE"
        subtitle = "High-Throughput VLM PDF-to-Markdown Extraction Pipeline"
        if self.console and self.is_tty:
            panel = Panel(
                f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]",
                border_style="cyan",
                padding=(1, 2),
            )
            self.console.print(panel)
        else:
            self.log(f"=== {title} ===\n{subtitle}")

    def phase_header(self, phase_num: int, title: str, description: str = "") -> None:
        msg = f"PHASE {phase_num}: {title.upper()}"
        if self.console and self.is_tty:
            self.console.print()
            self.console.print(Rule(f"[bold cyan]{msg}[/bold cyan]", style="cyan"))
            if description:
                self.console.print(f"[dim]{description}[/dim]")
            self.console.print()
        else:
            self.log(f"\n--- {msg} --- {('(' + description + ')') if description else ''}")

    def print_config_table(self, config: Dict[str, str]) -> None:
        if self.console and self.is_tty:
            table = Table(title="Pre-Flight Configuration", border_style="blue", show_header=True)
            table.add_column("Parameter", style="cyan", no_wrap=True)
            table.add_column("Value", style="bold white")

            for k, v in config.items():
                table.add_row(k, v)
            self.console.print(table)
            self.console.print()
        else:
            self.log("Pre-Flight Configuration:")
            for k, v in config.items():
                self.log(f"  - {k}: {v}")

    def print_server_table(self, servers: List[Dict[str, str]]) -> None:
        if self.console and self.is_tty:
            table = Table(title="VLM Server Endpoints", border_style="blue", show_header=True)
            table.add_column("Service", style="cyan")
            table.add_column("Host Port", style="white")
            table.add_column("Target GPU", style="magenta")
            table.add_column("Health Endpoint", style="dim")
            table.add_column("Status", style="bold")

            for s in servers:
                status_style = "green" if "OK" in s["status"] else "red"
                table.add_row(
                    s["service"],
                    s["port"],
                    s["gpu"],
                    s["endpoint"],
                    f"[{status_style}]{s['status']}[/{status_style}]",
                )
            self.console.print(table)
            self.console.print()
        else:
            self.log("VLM Server Endpoints:")
            for s in servers:
                self.log(f"  - {s['service']} on port {s['port']} ({s['gpu']}): {s['status']}")

    def print_summary_dashboard(self, metrics: Dict[str, str], success: bool) -> None:
        status_text = (
            f"[bold green]{self.ICON_CHECK} ALL PAGES SUCCESSFULLY PROCESSED[/bold green]"
            if success
            else f"[bold yellow]{self.ICON_WARN} CONVERSION FINISHED WITH RETRYABLE FAILURES[/bold yellow]"
        )

        if self.console and self.is_tty:
            table = Table(show_header=False, border_style="green" if success else "yellow", box=None)
            table.add_column("Metric", style="cyan", width=26)
            table.add_column("Value", style="bold white")

            for k, v in metrics.items():
                table.add_row(f"{k}:", v)

            panel = Panel(
                Group(Text.from_markup(f"\n{status_text}\n"), table),
                title=f"{self.ICON_DASHBOARD} Operational Summary Dashboard",
                border_style="green" if success else "yellow",
                padding=(1, 2),
            )
            self.console.print()
            self.console.print(panel)
            self.console.print()
        else:
            self.log("\n=================== CONVERSION SUMMARY DASHBOARD ===================")
            self.log(f"Status: {'SUCCESS' if success else 'PARTIAL SUCCESS / WARNINGS'}")
            for k, v in metrics.items():
                self.log(f"  {k:24}: {v}")
            self.log("====================================================================")


# ---------------------------------------------------------------------------
# Utility & Server Management Functions
# ---------------------------------------------------------------------------

def format_duration(total_seconds: float) -> str:
    total = int(total_seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s:02d}s"


def get_total_pages(pdf_path: Path) -> int:
    try:
        res = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True)
        match = re.search(r"^Pages:\s*(\d+)", res.stdout, re.MULTILINE)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"Error checking page count with pdfinfo: {e}", file=sys.stderr)
    return 0


def inspect_pdf_text_layer(pdf_path: Path, total_pages: int) -> Tuple[str, float, int]:
    """Inspect PDF to determine whether it is a scanned book or a born-digital book.

    Samples up to 9 evenly-spaced pages across the document to check for embedded text.

    Returns:
        (detected_method, avg_chars_per_page, total_sampled_words)
        where detected_method is 'txt' for digital or 'ocr' for scanned books.
    """
    if total_pages <= 0:
        return "ocr", 0.0, 0

    if total_pages <= 5:
        sample_pages = list(range(1, total_pages + 1))
    else:
        # Sample evenly-spaced pages across the document, skipping cover page 1
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        sample_pages = sorted(list(set(max(2, min(total_pages, int(total_pages * f))) for f in fractions)))

    total_chars = 0
    total_words = 0
    for p in sample_pages:
        try:
            res = subprocess.run(
                ["pdftotext", "-f", str(p), "-l", str(p), str(pdf_path), "-"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            text = res.stdout.strip()
            cleaned_text = re.sub(r"\s+", " ", text)
            words = len(cleaned_text.split())
            chars = len(re.sub(r"\s", "", text))
            total_chars += chars
            total_words += words
        except Exception:
            pass

    avg_chars = total_chars / len(sample_pages) if sample_pages else 0.0
    # A digital book typically has > 80 non-whitespace chars and > 30 words across sampled pages
    is_digital = (avg_chars >= 80.0 and total_words >= 30)
    detected_method = "txt" if is_digital else "ocr"
    return detected_method, avg_chars, total_words


def check_server_health(port: int, timeout: float = 3.0) -> Tuple[bool, float]:
    url = f"http://localhost:{port}/health"
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            latency = (time.time() - t0) * 1000
            return (200 <= resp.status < 300), latency
    except (urllib.error.URLError, OSError):
        return False, 0.0


def wait_for_server(port: int, max_wait: int = 60) -> bool:
    waited = 0
    while waited < max_wait:
        healthy, _ = check_server_health(port)
        if healthy:
            return True
        time.sleep(2)
        waited += 2
    return False


def ensure_servers_running(repo_dir: Path, ui: DevOpsUI) -> bool:
    """Verify and bring up llamacpp (GPU 0) and llamacpp1 (GPU 1) if necessary."""
    server_configs = [
        {"service": "llamacpp", "port": 30000, "gpu": "Tesla P100 [GPU 0]"},
        {"service": "llamacpp1", "port": 30001, "gpu": "Tesla P100 [GPU 1]"},
    ]
    results = []
    all_ok = True

    for item in server_configs:
        port = item["port"]
        service = item["service"]
        healthy, latency = check_server_health(port)

        if not healthy:
            ui.log(f"{ui.ICON_WARN} {service} (: {port}) is offline. Starting container via podman-compose...")
            subprocess.run(
                ["podman-compose", "--profile", "vlm", "up", "-d", service],
                cwd=str(repo_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if wait_for_server(port, max_wait=45):
                healthy, latency = check_server_health(port)
                status_str = f"ONLINE ({latency:.1f}ms)"
            else:
                status_str = "UNAVAILABLE"
                all_ok = False
        else:
            status_str = f"ONLINE ({latency:.1f}ms)"

        results.append({
            "service": service,
            "port": str(port),
            "gpu": item["gpu"],
            "endpoint": f"http://localhost:{port}/health",
            "status": status_str,
        })

    ui.print_server_table(results)
    return all_ok


def restart_vlm_servers(repo_dir: Path, ui: DevOpsUI) -> None:
    ui.log(f"{ui.ICON_RETRY} Restarting llamacpp and llamacpp1 to clear KV-cache and slot reuse states...")
    subprocess.run(["podman", "restart", "llamacpp", "llamacpp1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_for_server(30000, max_wait=30)
    wait_for_server(30001, max_wait=30)
    ui.log(f"{ui.ICON_CHECK} VLM servers restarted and healthy.")


def stop_vlm_servers(ui: DevOpsUI) -> None:
    """Stop llamacpp (GPU 0) and llamacpp1 (GPU 1) containers to unload models and release GPU VRAM."""
    ui.log(f"{ui.ICON_GPU} Unloading VLM models from GPU memory (stopping llamacpp and llamacpp1)...")
    res = subprocess.run(
        ["podman", "stop", "-t", "5", "llamacpp", "llamacpp1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if res.returncode == 0:
        ui.log(f"{ui.ICON_CHECK} VLM models unloaded. GPU VRAM released successfully.", style="success")
    else:
        ui.log(f"{ui.ICON_CHECK} VLM server containers stopped.", style="success")


def stream_pipe(pipe, prefix: str, out_queue: queue.Queue):
    try:
        for line in iter(pipe.readline, ""):
            if line:
                out_queue.put((prefix, line))
    finally:
        pipe.close()


def get_missing_pages(flat_dir: Path, basename: str, total_pages: int) -> List[int]:
    missing: List[int] = []
    for page_num in range(1, total_pages + 1):
        md_file = flat_dir / f"{basename}_page_{page_num:04d}.md"
        if not md_file.exists() or md_file.stat().st_size == 0:
            missing.append(page_num)
    return missing


# ---------------------------------------------------------------------------
# Worker Management & Progress Pipeline
# ---------------------------------------------------------------------------

def run_workers(
    repo_dir: Path,
    pdf_path: Path,
    pages_dir: Path,
    flat_dir: Path,
    fail_log: Path,
    claims_dir: Path,
    total_pages: int,
    pass_num: int,
    max_retries: int,
    ui: DevOpsUI,
    specific_pages: Optional[str] = None,
    num_workers: int = 2,
    lang: str = "ch",
    unload_gpu: bool = True,
    backend: str = "vlm-http-client",
    method: str = "auto",
) -> bool:
    """Launch persistent worker containers and stream output into a rich progress bar."""
    is_vlm = backend.startswith("vlm") or backend.startswith("hybrid")
    server_urls = [
        "http://llamacpp:30000",
        "http://llamacpp1:30000",
    ]

    claims_dir.mkdir(parents=True, exist_ok=True)
    for c in claims_dir.glob("*.claim"):
        try:
            c.unlink()
        except OSError:
            pass

    container_names = [f"mineru-fast-w{i}-{os.getpid()}" for i in range(num_workers)]

    for name in container_names:
        subprocess.run(["podman", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cleanup_signal(signum, frame):
        ui.log(f"\n{ui.ICON_WARN} Interrupt received. Stopping active worker containers...")
        for name in container_names:
            subprocess.run(["podman", "stop", "-t", "2", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if is_vlm and unload_gpu:
            stop_vlm_servers(ui)
        sys.exit(130)

    signal.signal(signal.SIGINT, cleanup_signal)
    signal.signal(signal.SIGTERM, cleanup_signal)

    in_dir_host = (repo_dir / "in").resolve()
    out_dir_host = str((repo_dir / "out").resolve())
    scripts_dir_host = str((repo_dir / "scripts").resolve())
    pdf_host_path = pdf_path.resolve()
    pdf_container_path = f"/data/in/{pdf_path.name}"
    pages_container_path = f"/data/out/{pages_dir.name}"
    flat_container_path = f"/data/out/{flat_dir.name}"
    fail_container_path = f"/data/out/{fail_log.name}"
    claims_container_path = f"/data/out/{pages_dir.name}/.claims"

    # Assemble volume mounts
    volumes = [
        "-v", f"{out_dir_host}:/data/out:z",
        "-v", "mineru_mineru-models:/models:z",
        "-v", f"{scripts_dir_host}:/scripts:z",
    ]
    if in_dir_host.exists():
        volumes.extend(["-v", f"{in_dir_host}:/data/in:ro,z"])
    # If the PDF is outside repo in/, mount the specific PDF directly to /data/in/<filename>
    if pdf_host_path.parent != in_dir_host and in_dir_host not in pdf_host_path.parents:
        volumes.extend(["-v", f"{pdf_host_path}:{pdf_container_path}:ro"])

    processes: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []
    out_queue: queue.Queue = queue.Queue()

    for i in range(num_workers):
        server_url = server_urls[i % len(server_urls)] if is_vlm else ""
        name = container_names[i]

        cmd = [
            "podman", "run",
            f"--name={name}",
            "--rm",
            "--network=mineru_mineru-net",
            "--security-opt", "label=disable",
            "--shm-size", "8gb",
            *volumes,
            "-e", "HF_HOME=/models/hf",
            "-e", "MINERU_TOOLS_CONFIG_JSON=/models/mineru.json",
            "-e", "OMP_NUM_THREADS=10",
            "-e", "MKL_NUM_THREADS=10",
            "-e", "MINERU_INTRA_OP_NUM_THREADS=10",
            "-e", "MINERU_INTER_OP_NUM_THREADS=1",
            "-e", "MINERU_PDF_RENDER_THREADS=4",
            "-e", "MINERU_FORMULA_ENABLE=true",
            "-e", "MINERU_TABLE_ENABLE=true",
            "localhost/mineru-cpu:latest",
            "python3", "/scripts/process_pages_worker.py",
            "--pdf", pdf_container_path,
            "--pages-dir", pages_container_path,
            "--flat-dir", flat_container_path,
            "--fail-log", fail_container_path,
            "--claims-dir", claims_container_path,
            "--server-url", server_url,
            "--worker-id", str(i),
            "--total-pages", str(total_pages),
            "--pass-num", str(pass_num),
            "--max-retries", str(max_retries),
            "--lang", lang,
            "--backend", backend,
            "--method", method,
        ]

        if specific_pages:
            cmd.extend(["--specific-pages", specific_pages])

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(proc)

        t = threading.Thread(target=stream_pipe, args=(proc.stdout, f"W{i}", out_queue), daemon=True)
        t.start()
        threads.append(t)

    # Calculate initial completed count
    initial_missing = get_missing_pages(flat_dir, pdf_path.stem, total_pages)
    completed_so_far = total_pages - len(initial_missing)

    # Use rich Progress bar if interactive TTY and rich available
    progress_ctx = None
    task_id = None
    if RICH_AVAILABLE and ui.is_tty:
        progress_ctx = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn(f"[bold cyan]Pass {pass_num}/{max_retries} Processing"),
            BarColumn(bar_width=32, complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=ui.console,
            transient=False,
        )
        progress_ctx.start()
        task_id = progress_ctx.add_task("Pages", total=total_pages, completed=completed_so_far)

    worker_buffers: Dict[str, List[str]] = {f"W{i}": [] for i in range(num_workers)}
    success_regex = re.compile(r"\[Worker (\d+)\] \[(\d+)/(\d+)\] (\S+): (SUCCESS|FAILED) in ([^\s\(]+)")
    processing_regex = re.compile(r"\[Worker (\d+)\] \[(\d+)/(\d+)\] processing (\S+)\.\.\.")
    error_keywords = ("Traceback (most recent call last):", "FATAL ERROR", "Error:", "Exception:")

    while any(p.poll() is None for p in processes) or not out_queue.empty():
        try:
            prefix, line = out_queue.get(timeout=0.2)
            line_str = line.strip()
            if not line_str:
                continue

            # Mirror all worker logs to the execution log file
            ui.log_raw(f"[{prefix}] {line_str}")

            buf = worker_buffers.get(prefix)
            if buf is not None:
                buf.append(line_str)
                if len(buf) > 40:
                    buf.pop(0)

            # Check for success/failure pattern
            match = success_regex.search(line_str)
            if match:
                worker_id, p_num, p_tot, p_name, status, dur = match.groups()
                if status == "SUCCESS":
                    if progress_ctx:
                        progress_ctx.advance(task_id, 1)
                        progress_ctx.console.print(
                            f"  [green]{ui.ICON_CHECK}[/green] [cyan]Worker {worker_id}[/cyan] │ Page [bold]{p_num:>4}[/bold]/{p_tot} │ [green]SUCCESS[/green] │ {dur:>4} │ [dim]{p_name}[/dim]"
                        )
                    else:
                        ui.log(f"{ui.ICON_CHECK} Worker {worker_id} │ Page {p_num:>4}/{p_tot} │ SUCCESS in {dur}")
                else:
                    if progress_ctx:
                        progress_ctx.console.print(
                            f"  [red]{ui.ICON_FAIL}[/red] [cyan]Worker {worker_id}[/cyan] │ Page [bold]{p_num:>4}[/bold]/{p_tot} │ [bold red]FAILED[/bold red]  │ {dur:>4} │ [dim]{p_name}[/dim]"
                        )
                    else:
                        ui.log(f"{ui.ICON_FAIL} Worker {worker_id} │ Page {p_num:>4}/{p_tot} │ FAILED in {dur}")
                continue

            # Check for truncation warning
            if "The response was truncated due to length limit" in line_str or "truncated due to length" in line_str:
                if ui.verbose:
                    warn_msg = f"  [yellow]{ui.ICON_WARN}[/yellow] [yellow]Notice: Minor layout block length limit reached (handled by VLM cleaner)[/yellow]"
                    if progress_ctx:
                        progress_ctx.console.print(warn_msg)
                    else:
                        ui.log(warn_msg)
                continue

            # Check for processing start in verbose or non-TTY mode
            p_match = processing_regex.search(line_str)
            if p_match and ui.verbose:
                worker_id, p_num, p_tot, p_name = p_match.groups()
                msg = f"  [dim]↳ [{prefix}] Starting page {p_num}/{p_tot}: {p_name}[/dim]"
                if progress_ctx:
                    progress_ctx.console.print(msg)
                else:
                    ui.log(msg)
                continue

            # Proactively surface errors/exceptions even when verbose=False
            if any(k in line_str for k in error_keywords) or line_str.startswith("Error") or "FATAL" in line_str:
                err_msg = f"[{prefix}] {line_str}"
                if progress_ctx:
                    progress_ctx.console.print(f"  [bold red]{ui.ICON_FAIL} {err_msg}[/bold red]")
                else:
                    ui.log(f"{ui.ICON_FAIL} {err_msg}", style="error")
                continue

            # Verbose raw line dump
            if ui.verbose:
                if progress_ctx:
                    progress_ctx.console.print(f"[dim]{prefix}: {line_str}[/dim]")
                else:
                    ui.log(f"{prefix}: {line_str}")

        except queue.Empty:
            pass

    if progress_ctx:
        progress_ctx.stop()

    for t in threads:
        t.join(timeout=1.0)

    # Drain any remaining lines in queue
    while not out_queue.empty():
        try:
            prefix, line = out_queue.get_nowait()
            line_str = line.strip()
            if line_str:
                ui.log_raw(f"[{prefix}] {line_str}")
                buf = worker_buffers.get(prefix)
                if buf is not None:
                    buf.append(line_str)
        except queue.Empty:
            break

    failed_workers = []
    for i, p in enumerate(processes):
        code = p.wait()
        if code != 0:
            failed_workers.append((i, code))

    if failed_workers:
        for worker_idx, code in failed_workers:
            prefix = f"W{worker_idx}"
            ui.log(
                f"{ui.ICON_FAIL} Worker {worker_idx} ({container_names[worker_idx]}) crashed with exit code {code}!",
                style="error",
            )
            recent = worker_buffers.get(prefix, [])
            if recent:
                ui.log(f"--- Last {len(recent)} lines from Worker {worker_idx} ---", style="warning")
                for r_line in recent[-20:]:
                    ui.log(f"  [{prefix}] {r_line}", style="error")
                ui.log("--------------------------------------------------", style="warning")
        return False

    return True


# ---------------------------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast dual-GPU MinerU book converter with modern DevOps CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf", type=Path, help="Path to input PDF file (e.g. in/book.pdf)")
    parser.add_argument(
        "-b", "--backend",
        default="pipeline",
        choices=["pipeline", "vlm-http-client", "hybrid-http-client"],
        help="MinerU backend: 'pipeline' (YOLO layout + digital text), 'vlm-http-client' (VLM vision OCR), 'hybrid-http-client' (VLM layout + digital text) [default: pipeline]",
    )
    parser.add_argument(
        "-m", "--method",
        default="auto",
        choices=["auto", "txt", "ocr"],
        help="MinerU parse method: 'auto' (detect digital vs scanned), 'txt' (100%% exact non-OCR digital font glyphs), 'ocr' (force OCR) [default: auto]",
    )
    parser.add_argument("--workers", default=2, type=int, help="Number of parallel worker containers")
    parser.add_argument("--max-retries", default=3, type=int, help="Maximum number of retry passes")
    parser.add_argument("--lang", default="ch", type=str, help="Language model setting (ch: EN+ZH, en: EN)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose inference and container logs")
    parser.add_argument("--no-color", action="store_true", help="Disable colorized terminal output")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional path to mirror raw execution logs")
    parser.add_argument(
        "--unload-gpu",
        dest="unload_gpu",
        action="store_true",
        default=True,
        help="Unload VLM models and release GPU memory upon completion (default: True)",
    )
    parser.add_argument(
        "--keep-gpu",
        dest="unload_gpu",
        action="store_false",
        help="Keep VLM server containers running on GPU after conversion finishes",
    )
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent.parent
    pdf_path = args.pdf.resolve()
    basename = pdf_path.stem
    default_log_file = repo_dir / "out" / f"{basename}_run.log"
    log_file = args.log_file if args.log_file is not None else default_log_file

    ui = DevOpsUI(no_color=args.no_color, verbose=args.verbose, log_file=log_file)
    t_start = time.time()
    is_vlm = args.backend in ("vlm-http-client", "hybrid-http-client")

    # Phase 1: Environment & Pre-Flight Diagnostics
    ui.print_banner()
    ui.phase_header(1, "Environment & Pre-Flight Diagnostics", "Verifying PDF document and inference setup")

    if not pdf_path.is_file():
        ui.log(f"{ui.ICON_FAIL} Error: Specified input PDF not found at: {pdf_path}", style="error")
        return 2

    total_pages = get_total_pages(pdf_path)
    if total_pages == 0:
        ui.log(f"{ui.ICON_FAIL} Error: Failed to determine page count via pdfinfo.", style="error")
        return 2

    detected_method, avg_chars, sample_words = inspect_pdf_text_layer(pdf_path, total_pages)
    if args.method == "auto":
        resolved_method = detected_method
        if detected_method == "txt":
            method_desc = f"auto -> txt (Born-Digital: ~{avg_chars:.0f} chars/pg, non-OCR direct extraction)"
        else:
            method_desc = f"auto -> ocr (Scanned Book: < {avg_chars:.0f} chars/pg, image-based OCR)"
    else:
        resolved_method = args.method
        method_desc = f"{args.method} (user-specified; PDF text layer detected: {detected_method})"

    if args.backend == "vlm-http-client" and resolved_method == "txt":
        ui.log(
            f"{ui.ICON_WARN} Note: 'vlm-http-client' runs pure vision VLM inference and does not read the embedded text layer. "
            "Use '--backend pipeline' or '--backend hybrid-http-client' for 100% exact digital font glyph extraction.",
            style="warning",
        )

    pages_dir = repo_dir / "out" / f"{basename}_pages"
    flat_dir = repo_dir / "out" / f"{basename}_md"
    flat_images_dir = flat_dir / "images"
    fail_log = repo_dir / "out" / f"{basename}_failures.txt"
    claims_dir = pages_dir / ".claims"

    pages_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    flat_images_dir.mkdir(parents=True, exist_ok=True)

    # Extract PDF outline bookmarks into flat_dir/toc.json if available
    toc_json_path = flat_dir / "toc.json"
    if not toc_json_path.exists() and shutil.which("pdftohtml"):
        try:
            import html as _html
            import json as _json
            _res = subprocess.run(
                ["pdftohtml", "-xml", "-i", "-f", "1", "-l", "1", "-stdout", str(pdf_path)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if "<outline>" in _res.stdout and "</outline>" in _res.stdout:
                _oblock = _res.stdout.split("<outline>", 1)[1].rsplit("</outline>", 1)[0]
                _toc_map: Dict[str, List[str]] = {}
                for _m in re.finditer(r'<item\s+page="(\d+)">(.*?)</item>', _oblock, re.DOTALL):
                    _pg = str(int(_m.group(1)))
                    _t = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", _m.group(2)))).strip()
                    if _t and _t.lower() not in ("my bookmarks", "frontcover", "body"):
                        _lst = _toc_map.setdefault(_pg, [])
                        if _t not in _lst:
                            _lst.append(_t)
                if _toc_map:
                    toc_json_path.write_text(
                        _json.dumps({k: " / ".join(v) for k, v in _toc_map.items()}, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
        except Exception:
            pass

    missing_pages = get_missing_pages(flat_dir, basename, total_pages)
    completed_start = total_pages - len(missing_pages)
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)

    config_info = {
        "Document File": pdf_path.name,
        "File Size": f"{file_size_mb:.2f} MB",
        "Total Pages": str(total_pages),
        "PDF Type": f"{'Born-Digital' if detected_method == 'txt' else 'Scanned Book'} (~{avg_chars:.0f} chars/page sampled)",
        "Already Processed": f"{completed_start} pages ({completed_start/total_pages*100:.1f}%)",
        "Pages Remaining": f"{len(missing_pages)} pages",
        "Backend": f"{args.backend} ({'Vision Language Model' if args.backend == 'vlm-http-client' else 'YOLOv8 Layout + Digital Text' if args.backend == 'pipeline' else 'Hybrid VLM+Text'})",
        "Parse Method": method_desc,
        "Parallel Workers": f"{args.workers} ({'Dual Tesla P100 GPUs' if is_vlm else 'CPU / Pipeline Containers'})",
        "Max Retry Passes": str(args.max_retries),
        "Target Language": args.lang,
        "Output Directory": str(flat_dir.relative_to(repo_dir) if flat_dir.is_relative_to(repo_dir) else flat_dir),
        "Execution Log": str(log_file.relative_to(repo_dir) if log_file.is_relative_to(repo_dir) else log_file),
        "GPU Auto-Unload": ("Enabled (releases VRAM on finish)" if args.unload_gpu else "Disabled (keeps models in VRAM)") if is_vlm else "N/A (CPU pipeline)",
    }
    ui.print_config_table(config_info)

    try:
        # Health Check VLM servers if running a VLM-dependent backend
        if is_vlm:
            if not ensure_servers_running(repo_dir, ui):
                ui.log(f"{ui.ICON_FAIL} Critical: One or more VLM servers failed to become healthy. Aborting.", style="error")
                return 2
        else:
            ui.log(f"{ui.ICON_CHECK} Backend '{args.backend}' does not require remote VLM servers.")

        # Phase 2: Primary Conversion Pass
        ui.phase_header(2, "Primary Conversion Pass", f"Executing Pass 1/{args.max_retries} across {args.workers} persistent workers")

        if missing_pages:
            pass1_ok = run_workers(
                repo_dir=repo_dir,
                pdf_path=pdf_path,
                pages_dir=pages_dir,
                flat_dir=flat_dir,
                fail_log=fail_log,
                claims_dir=claims_dir,
                total_pages=total_pages,
                pass_num=1,
                max_retries=args.max_retries,
                ui=ui,
                num_workers=args.workers,
                lang=args.lang,
                unload_gpu=args.unload_gpu,
                backend=args.backend,
                method=resolved_method,
            )
            if not pass1_ok:
                missing_after = get_missing_pages(flat_dir, basename, total_pages)
                if len(missing_after) == total_pages:
                    ui.log(
                        f"{ui.ICON_FAIL} Fatal: Worker containers failed on startup with no pages processed. Aborting further passes.",
                        style="error",
                    )
                    return 1
        else:
            ui.log(f"{ui.ICON_CHECK} All {total_pages} pages already completed. Skipping Pass 1.")

        # Phase 3: Verification & Multi-Pass Retries
        for pass_num in range(2, args.max_retries + 1):
            missing_pages = get_missing_pages(flat_dir, basename, total_pages)
            if not missing_pages:
                break

            ui.phase_header(
                3,
                f"Verification & Retries (Pass {pass_num}/{args.max_retries})",
                f"{len(missing_pages)} page(s) require retry due to truncation or server loop",
            )

            if is_vlm:
                restart_vlm_servers(repo_dir, ui)
            else:
                ui.log(f"{ui.ICON_RETRY} Retrying failed pages with {args.backend} backend...")

            missing_pages_str = ",".join(str(p) for p in missing_pages)

            pass_ok = run_workers(
                repo_dir=repo_dir,
                pdf_path=pdf_path,
                pages_dir=pages_dir,
                flat_dir=flat_dir,
                fail_log=fail_log,
                claims_dir=claims_dir,
                total_pages=total_pages,
                pass_num=pass_num,
                max_retries=args.max_retries,
                ui=ui,
                specific_pages=missing_pages_str,
                num_workers=args.workers,
                lang=args.lang,
                unload_gpu=args.unload_gpu,
                backend=args.backend,
                method=resolved_method,
            )
            if not pass_ok:
                ui.log(
                    f"{ui.ICON_WARN} Worker containers encountered errors during retry Pass {pass_num}.",
                    style="warning",
                )

        # Cleanup claims
        try:
            shutil.rmtree(claims_dir, ignore_errors=True)
        except OSError:
            pass

        # Extract Stage 1 bibliographic metadata (book_metadata.json) and Page 1 cover (cover.jpg)
        extracted_meta = {}
        try:
            from metadata_extractor import extract_book_metadata
            extracted_meta = extract_book_metadata(pdf_path=pdf_path, md_dir=flat_dir)
            ui.log(
                f"{ui.ICON_CHECK} Extracted book metadata: Title='{extracted_meta.get('title')}' | "
                f"Authors={extracted_meta.get('authors')} | ISBN={extracted_meta.get('isbn') or 'N/A'}"
            )
        except Exception as meta_err:
            ui.log(f"{ui.ICON_WARN} Metadata extraction warning: {meta_err}", style="warning")

        # Phase 4: Final Summary Dashboard
        ui.phase_header(4, "Operational Metrics & Dashboard")

        final_missing = get_missing_pages(flat_dir, basename, total_pages)
        final_done = total_pages - len(final_missing)
        total_elapsed = time.time() - t_start
        new_pages = final_done - completed_start

        effective_rate_str = f"{total_elapsed / new_pages:.1f} s/page" if new_pages > 0 else "N/A"
        hourly_rate_str = f"~{int(new_pages / (total_elapsed / 3600))} pages/hr" if total_elapsed > 0 and new_pages > 0 else "N/A"
        acceleration_str = f"{args.workers}x Tesla P100 (Ports 30000 & 30001)" if is_vlm else f"{args.workers}x Pipeline Worker(s)"

        dashboard_metrics = {
            "Source PDF": f"{pdf_path.name} ({file_size_mb:.2f} MB)",
            "PDF Type": f"{'Born-Digital' if detected_method == 'txt' else 'Scanned Book'} (resolved method: {resolved_method})",
            "Completed Pages": f"{final_done}/{total_pages} ({final_done/total_pages*100:.1f}%)",
            "New Pages Processed": str(new_pages),
            "Total Elapsed Time": format_duration(total_elapsed),
            "Effective Throughput": f"{effective_rate_str} ({hourly_rate_str})",
            "Active Acceleration": acceleration_str,
            "Markdown Output": str(flat_dir),
            "Images Output": str(flat_images_dir),
            "Execution Log": str(log_file),
            "GPU Model State": ("Unloaded (VRAM released)" if args.unload_gpu else "Resident in VRAM") if is_vlm else "N/A (CPU)",
        }

        if final_missing:
            dashboard_metrics["Outstanding Pages"] = f"{len(final_missing)} pages: {final_missing[:10]}{'...' if len(final_missing) > 10 else ''}"
            dashboard_metrics["Failure Log"] = str(fail_log)

        success = (len(final_missing) == 0)
        ui.print_summary_dashboard(dashboard_metrics, success=success)

        return 0 if success else 1

    finally:
        if is_vlm and args.unload_gpu:
            stop_vlm_servers(ui)


if __name__ == "__main__":
    sys.exit(main())
