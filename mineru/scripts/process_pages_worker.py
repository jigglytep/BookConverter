#!/usr/bin/env python3
"""MinerU persistent page worker.

Processes PDF pages in a loop within a single Python process, avoiding the
~11-second container startup and library import penalty per page.
Supports atomic work-stealing across multiple workers via a claims directory.
"""
import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from loguru import logger
from mineru.cli.common import do_parse, read_fn


def format_duration(total_seconds: float) -> str:
    total = int(total_seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:02d}m{s:02d}s"
    return f"{s:02d}s"


def clean_repetitive_vlm_output(text: str) -> str:
    """Collapse runaway autoregressive VLM loops while preserving legitimate code and text."""
    if len(text) < 5000:
        return text

    # 1. 000... loop (like 10**100 googol where model outputs 16,000 zeros)
    text = re.sub(r"0{150,}", "0" * 100, text)
    # 2. Repeated single delimiter like '===========' or 'xxxxxxxxxx' (> 50 chars)
    text = re.sub(r"([=\-_*~xX])(?:\s*\1){49,}", r"\1\1\1\1\1\1\1\1\1\1\n", text)
    # 3. Repeated word/phrase/token loop (repeated 25+ times consecutively)
    pattern = re.compile(r"((?:[^\n]{2,50}?))(?:\s*\1){24,}")

    def repl(m):
        chunk = m.group(1).strip()
        return f"{chunk} {chunk} {chunk} ..."

    text = pattern.sub(repl, text)
    # 4. Repeated array tokens like ', true, true...' repeated 25+ times
    pattern_arr = re.compile(r"([,\s]+(?:true|false|nan|null|none|[0-9\.\-]+))(?:\1){24,}", re.IGNORECASE)
    text = pattern_arr.sub(r"\1\1\1 ...", text)
    return text


def try_claim_page(claims_dir: Path, page_num: int, worker_id: int) -> bool:
    """Atomically claim a page number using O_CREAT | O_EXCL."""
    claim_path = claims_dir / f"{page_num}.claim"
    try:
        fd = os.open(str(claim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(f"worker_{worker_id}_{os.getpid()}\n")
        return True
    except FileExistsError:
        return False
    except OSError:
        return False


def release_claim(claims_dir: Path, page_num: int) -> None:
    claim_path = claims_dir / f"{page_num}.claim"
    try:
        if claim_path.exists():
            claim_path.unlink()
    except OSError:
        pass


def inspect_pdf_text_layer(pdf_path: Path, total_pages: int) -> str:
    """Inspect PDF to determine whether it is a scanned book or a born-digital book."""
    import subprocess
    if total_pages <= 0:
        return "ocr"
    if total_pages <= 5:
        sample_pages = list(range(1, total_pages + 1))
    else:
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
    return "txt" if (avg_chars >= 80.0 and total_words >= 30) else "ocr"


def log_failure(fail_log: Path, page_name: str, reason: str, pass_num: int, max_retries: int) -> None:
    try:
        with fail_log.open("a", encoding="utf-8") as f:
            f.write(f"{page_name}: {reason} (pass {pass_num}/{max_retries})\n")
    except OSError as e:
        print(f"Error writing to failure log: {e}", file=sys.stderr)


def process_single_page(
    pdf_bytes: bytes,
    basename: str,
    page_num: int,
    pages_dir: Path,
    flat_dir: Path,
    flat_images_dir: Path,
    fail_log: Path,
    server_url: str,
    lang: str,
    pass_num: int,
    max_retries: int,
    backend: str = "vlm-http-client",
    method: str = "auto",
) -> bool:
    page_name = f"{basename}_page_{page_num:04d}"
    flat_md = flat_dir / f"{page_name}.md"
    page_id = page_num - 1  # 0-indexed

    # Determine expected subfolder structure based on backend and method
    if backend.startswith("vlm"):
        expected_subfolder = "vlm"
    elif backend.startswith("hybrid"):
        expected_subfolder = f"hybrid_{method}"
    else:  # pipeline
        expected_subfolder = method

    src_md = pages_dir / page_name / expected_subfolder / f"{page_name}.md"
    if not src_md.is_file():
        # Fallback check if any markdown matching page_name was generated
        candidates = list((pages_dir / page_name).glob(f"**/{page_name}.md"))
        if candidates:
            src_md = candidates[0]

    # Only invoke parsing if markdown hasn't been parsed yet in pages_dir
    if not src_md.is_file():
        page_output_dir = pages_dir / page_name
        if page_output_dir.exists():
            shutil.rmtree(page_output_dir, ignore_errors=True)

        captured_logs: List[str] = []

        def log_capture(message):
            captured_logs.append(str(message))

        handler_id = logger.add(log_capture, level="DEBUG")

        parse_kwargs = {
            "output_dir": str(pages_dir),
            "pdf_file_names": [page_name],
            "pdf_bytes_list": [pdf_bytes],
            "p_lang_list": [lang],
            "backend": backend,
            "parse_method": method,
            "start_page_id": page_id,
            "end_page_id": page_id,
            "max_concurrency": 1,
        }
        if backend.endswith("-client") or backend.startswith("hybrid"):
            parse_kwargs["server_url"] = server_url

        try:
            do_parse(**parse_kwargs)
        except Exception as e:
            logger.remove(handler_id)
            log_failure(fail_log, page_name, f"exception: {e}", pass_num, max_retries)
            return False
        finally:
            try:
                logger.remove(handler_id)
            except ValueError:
                pass

        if not src_md.is_file():
            candidates = list((pages_dir / page_name).glob(f"**/{page_name}.md"))
            if candidates:
                src_md = candidates[0]
            else:
                all_logs = "".join(captured_logs)
                if "truncated due to length" in all_logs or "finish_reason=length" in all_logs:
                    log_failure(fail_log, page_name, "truncated due to length, no md generated", pass_num, max_retries)
                else:
                    log_failure(fail_log, page_name, f"no md generated at {src_md}", pass_num, max_retries)
                return False

    # Read and process generated markdown
    try:
        content = src_md.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            # Valid empty/blank page in PDF
            content = "<!-- blank page -->\n"
            src_md.write_text(content, encoding="utf-8")
        else:
            cleaned = clean_repetitive_vlm_output(content)
            if len(cleaned) < len(content):
                content = cleaned
                src_md.write_text(content, encoding="utf-8")

        flat_md.write_text(content, encoding="utf-8")

        # Copy images if any
        src_images_dir = src_md.parent / "images"
        if src_images_dir.is_dir():
            for img in src_images_dir.iterdir():
                if img.is_file():
                    dst_img = flat_images_dir / img.name
                    if not dst_img.exists():
                        shutil.copy2(img, dst_img)
        return True
    except OSError as e:
        log_failure(fail_log, page_name, f"file write/copy error: {e}", pass_num, max_retries)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="MinerU persistent page worker")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to input PDF inside container")
    parser.add_argument("--pages-dir", required=True, type=Path, help="Per-page output directory")
    parser.add_argument("--flat-dir", required=True, type=Path, help="Flattened markdown directory")
    parser.add_argument("--fail-log", required=True, type=Path, help="Failure log path")
    parser.add_argument("--claims-dir", required=True, type=Path, help="Claims directory for work stealing")
    parser.add_argument("--server-url", default="", type=str, help="VLM server URL (llama.cpp)")
    parser.add_argument("--worker-id", required=True, type=int, help="Worker ID (e.g. 0, 1)")
    parser.add_argument("--total-pages", required=True, type=int, help="Total pages in PDF")
    parser.add_argument("--lang", default="ch", type=str, help="Language code (default ch)")
    parser.add_argument("--pass-num", default=1, type=int, help="Current pass number")
    parser.add_argument("--max-retries", default=3, type=int, help="Max retry passes")
    parser.add_argument("--specific-pages", default=None, type=str, help="Comma-separated page numbers to process")
    parser.add_argument("--backend", default="vlm-http-client", choices=["vlm-http-client", "pipeline", "hybrid-http-client"], help="MinerU backend (default vlm-http-client)")
    parser.add_argument("--method", default="auto", choices=["auto", "txt", "ocr"], help="MinerU parse method (default auto)")

    args = parser.parse_args()

    pdf_path = args.pdf
    basename = pdf_path.stem
    pages_dir = args.pages_dir
    flat_dir = args.flat_dir
    flat_images_dir = flat_dir / "images"
    fail_log = args.fail_log
    claims_dir = args.claims_dir
    server_url = args.server_url
    worker_id = args.worker_id
    total_pages = args.total_pages
    lang = args.lang
    pass_num = args.pass_num
    max_retries = args.max_retries
    backend = args.backend
    method = args.method
    if method == "auto":
        method = inspect_pdf_text_layer(pdf_path, total_pages)

    pages_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    flat_images_dir.mkdir(parents=True, exist_ok=True)
    claims_dir.mkdir(parents=True, exist_ok=True)

    if args.specific_pages:
        candidate_pages = [int(p.strip()) for p in args.specific_pages.split(",") if p.strip()]
    else:
        candidate_pages = list(range(1, total_pages + 1))

    server_info = f"on {server_url}" if server_url else "local pipeline"
    print(
        f"[Worker {worker_id}] Starting {server_info} (backend: {backend}, method: {method}) | {len(candidate_pages)} candidate pages | pass {pass_num}/{max_retries}",
        flush=True,
    )

    if not pdf_path.is_file():
        print(
            f"[Worker {worker_id}] FATAL ERROR: Input PDF not found inside container at {pdf_path}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    try:
        t0_read = time.time()
        pdf_bytes = read_fn(pdf_path)
        print(
            f"[Worker {worker_id}] Loaded PDF into memory ({len(pdf_bytes)/(1024*1024):.1f} MB in {time.time()-t0_read:.2f}s)",
            flush=True,
        )
    except Exception as e:
        print(
            f"[Worker {worker_id}] FATAL ERROR reading PDF {pdf_path}: {e}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    worker_processed = 0
    worker_elapsed = 0.0

    for page_num in candidate_pages:
        page_name = f"{basename}_page_{page_num:04d}"
        flat_md = flat_dir / f"{page_name}.md"

        # Check if already completed
        if flat_md.exists() and flat_md.stat().st_size > 0:
            # Backfill images if missing
            page_output_dir = pages_dir / page_name
            if page_output_dir.is_dir():
                for img_dir in page_output_dir.glob("**/images"):
                    if img_dir.is_dir():
                        for img in img_dir.iterdir():
                            if img.is_file():
                                dst_img = flat_images_dir / img.name
                                if not dst_img.exists():
                                    try:
                                        shutil.copy2(img, dst_img)
                                    except OSError:
                                        pass
            continue

        # Attempt atomic claim
        if not try_claim_page(claims_dir, page_num, worker_id):
            continue

        # Page claimed by this worker
        t_page_start = time.time()
        print(f"[Worker {worker_id}] [{page_num}/{total_pages}] processing {page_name}...", flush=True)

        success = process_single_page(
            pdf_bytes=pdf_bytes,
            basename=basename,
            page_num=page_num,
            pages_dir=pages_dir,
            flat_dir=flat_dir,
            flat_images_dir=flat_images_dir,
            fail_log=fail_log,
            server_url=server_url,
            lang=lang,
            pass_num=pass_num,
            max_retries=max_retries,
            backend=backend,
            method=method,
        )

        page_duration = time.time() - t_page_start
        worker_elapsed += page_duration
        worker_processed += 1
        avg_duration = worker_elapsed / worker_processed

        # Release claim after processing
        release_claim(claims_dir, page_num)

        if success:
            print(
                f"[Worker {worker_id}] [{page_num}/{total_pages}] {page_name}: SUCCESS in {format_duration(page_duration)} (worker avg {avg_duration:.1f}s/page, {worker_processed} done)",
                flush=True,
            )
        else:
            print(
                f"[Worker {worker_id}] [{page_num}/{total_pages}] {page_name}: FAILED after {format_duration(page_duration)} (see {fail_log.name})",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"[Worker {worker_id}] Complete: {worker_processed} pages processed in {format_duration(worker_elapsed)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
