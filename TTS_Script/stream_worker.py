#!/usr/bin/env python3
"""
stream_worker.py — Background pipeline worker that runs concurrently alongside MinerU.

Watches mineru/out/<name>_md for new markdown page files, immediately:
  1. Converts each new markdown page into clean TTS text via md_tts.py (and mathspeak-api).
  2. (Optionally) Pre-synthesizes audio chunks into Kokoro's cache using KokoroAudiobookGenerator
     in a decoupled background thread.

This ensures that while MinerU takes 15-30s per page on the VLM, the CPU/GPU
finishes text cleanup and audio chunk synthesis in parallel, drastically reducing
overall end-to-end turnaround time.
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

# Add Kokoro and TTS_Script directories to path
TTS_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = TTS_SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "kokoro"))
sys.path.insert(0, str(TTS_SCRIPT_DIR))

import md_tts

running = True


def sig_handler(signum, frame):
    global running
    running = False


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def is_file_ready(path: Path, known_sizes: dict) -> bool:
    """Check if a file has finished being written by MinerU."""
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = stat.st_mtime
    except OSError:
        return False
    if size == 0:
        return False

    now = time.time()
    prev_size, prev_time = known_sizes.get(path, (None, None))
    # If the file hasn't been modified in >= 1.0 second, it's safe to process
    if now - mtime >= 1.0:
        known_sizes[path] = (size, now)
        return True

    # Otherwise, require size to remain stable across consecutive checks
    if prev_size is None or prev_size != size:
        known_sizes[path] = (size, now)
        return False

    return True


def process_md_page(md_path: Path, txt_out_dir: Path, opts, stats: dict) -> Path:
    """Convert a single markdown page to text."""
    txt_out_dir.mkdir(parents=True, exist_ok=True)
    dest = txt_out_dir / (md_path.stem + ".txt")
    if dest.exists() and dest.stat().st_mtime >= md_path.stat().st_mtime and dest.stat().st_size > 0:
        return dest
    raw = md_path.read_text(encoding="utf-8", errors="replace")
    clean_text = md_tts.process(raw, opts, stats)
    dest.write_text(clean_text, encoding="utf-8")
    return dest


def synthesis_worker(generator, synthesis_queue: queue.Queue, synthesized_txt: set, sf, config):
    """Background worker that consumes clean text pages and caches audio chunks in Kokoro."""
    while running or not synthesis_queue.empty():
        try:
            txt_file = synthesis_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if txt_file in synthesized_txt or not txt_file.exists() or txt_file.stat().st_size == 0:
            synthesis_queue.task_done()
            continue

        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            chunks = generator.split_into_chunks(text)
            for chunk in chunks:
                if not running:
                    break
                cache_path = generator.get_cache_path(chunk)
                if not cache_path.exists():
                    pipeline = generator._pipeline_pool.get()
                    try:
                        audio = generator._synthesize(chunk, pipeline)
                        sf.write(str(cache_path), audio, config.SAMPLE_RATE)
                    finally:
                        generator._pipeline_pool.put(pipeline)
            synthesized_txt.add(txt_file)
            print(f"[stream_worker] [Synthesized & Cached] {txt_file.name} ({len(chunks)} chunk(s))", flush=True)
        except Exception as e:
            print(f"[stream_worker] Error synthesizing {txt_file.name}: {e}", flush=True)
        finally:
            synthesis_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description="Concurrent streaming worker for MinerU -> TTS -> Kokoro")
    parser.add_argument("--md-dir", required=True, help="Directory where MinerU writes markdown pages")
    parser.add_argument("--txt-dir", required=True, help="Directory where clean text pages are written")
    parser.add_argument("--recursive", action="store_true", default=True, help="Scan recursively for nested page markdown files (default: True)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursive markdown scanning")
    parser.add_argument("--synthesize", action="store_true", help="Pre-synthesize audio chunks into Kokoro cache in background")
    parser.add_argument("--voice", default=None, help="Kokoro voice")
    parser.add_argument("--speed", type=float, default=None, help="Kokoro speed")
    parser.add_argument("--device", default=None, help="Kokoro device")
    parser.add_argument("--workers", type=int, default=1, help="Kokoro worker instances")
    parser.add_argument("--numbers", choices=["keep", "words"], default="words", help="Number handling (default: words)")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between scan loops")
    parser.add_argument("--watch-pid", type=int, default=None, help="MinerU process PID to monitor; exits when PID terminates")
    args = parser.parse_args()

    md_dir = Path(args.md_dir)
    txt_dir = Path(args.txt_dir)
    md_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Configure md_tts options
    class MdOpts:
        math_api = "http://localhost:8099"
        math_style = "clearspeak"
        on_math_error = "keep"
        math_error_placeholder = ""
        timeout = 30.0
        code = "drop"
        images = "caption"
        tables = "note"
        headings = "keep"
        keep_footnotes = False
        tts_punct = True
        numbers = args.numbers

    md_opts = MdOpts()
    stats = {"equations": 0, "converted": 0, "errors": 0}

    # Initialize Kokoro generator and background thread if synthesis is requested
    generator = None
    synthesis_queue = queue.Queue()
    synthesized_txt = set()
    synthesis_thread = None

    if args.synthesize:
        try:
            import soundfile as sf
            import config
            from audiobook_tts import KokoroAudiobookGenerator
            generator = KokoroAudiobookGenerator(
                voice=args.voice, speed=args.speed, device=args.device, workers=args.workers
            )
            print(f"[stream_worker] Kokoro initialized for background audio synthesis ({args.workers} worker(s))", flush=True)
            synthesis_thread = threading.Thread(
                target=synthesis_worker,
                args=(generator, synthesis_queue, synthesized_txt, sf, config),
                daemon=True,
            )
            synthesis_thread.start()
        except Exception as e:
            print(f"[stream_worker] Warning: Kokoro initialization failed ({e}). Proceeding with text-only streaming.", flush=True)
            generator = None

    processed_md = set()
    known_sizes = {}

    print(f"[stream_worker] Monitoring {md_dir} -> {txt_dir} (recursive: {args.recursive}, PID watch: {args.watch_pid})", flush=True)

    while running:
        # Check if parent / MinerU process is still alive
        if args.watch_pid is not None:
            try:
                os.kill(args.watch_pid, 0)
            except OSError:
                print(f"[stream_worker] Monitored PID {args.watch_pid} finished. Completing final pass...", flush=True)
                break

        # Find all available .md pages (recursive if enabled)
        md_pattern = "**/*.md" if args.recursive else "*.md"
        md_files = sorted(md_dir.glob(md_pattern))
        for md_file in md_files:
            if not running:
                break

            # Skip if currently being written or empty
            if not is_file_ready(md_file, known_sizes):
                continue

            if md_file not in processed_md:
                try:
                    txt_path = process_md_page(md_file, txt_dir, md_opts, stats)
                    processed_md.add(md_file)
                    print(f"[stream_worker] [Cleaned] {md_file.name} -> {txt_path.name}", flush=True)
                    if generator is not None:
                        synthesis_queue.put(txt_path)
                except Exception as e:
                    print(f"[stream_worker] Error converting {md_file.name}: {e}", flush=True)

        # Enqueue any pre-existing .txt files if not yet synthesized
        if generator is not None:
            for txt_file in sorted(txt_dir.glob("*.txt")):
                if txt_file not in synthesized_txt and txt_file.stat().st_size > 0:
                    synthesis_queue.put(txt_file)

        time.sleep(args.poll_interval)

    # Final pass over any remaining files
    print("[stream_worker] Running final cleanup pass...", flush=True)
    md_pattern = "**/*.md" if args.recursive else "*.md"
    for md_file in sorted(md_dir.glob(md_pattern)):
        try:
            if md_file.stat().st_size > 0:
                txt_path = process_md_page(md_file, txt_dir, md_opts, stats)
                processed_md.add(md_file)
                if generator is not None and txt_path not in synthesized_txt:
                    synthesis_queue.put(txt_path)
        except Exception as e:
            print(f"[stream_worker] Final pass error on {md_file.name}: {e}", flush=True)

    # Wait for background audio synthesis to complete if active
    if generator is not None and synthesis_thread is not None:
        print("[stream_worker] Waiting for background audio chunk synthesis to complete...", flush=True)
        synthesis_queue.join()

    print(f"[stream_worker] Done. Processed {len(processed_md)} markdown pages, cached audio for {len(synthesized_txt)} pages.", flush=True)


if __name__ == "__main__":
    main()
