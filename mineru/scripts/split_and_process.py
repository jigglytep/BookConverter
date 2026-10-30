#!/usr/bin/env python3
"""Run mineru on a PDF one page at a time, without physically pre-splitting it.

Why per-page: a single malformed VLM response can hard-crash a whole
multi-hour mineru task (see runbook 1.6 / the 2026-07-28 incident where a
500 "peg-native format" error killed a 4.5h/640-page run with nothing
written to disk). One task per page means one bad page costs one page,
and the script is resumable -- rerun it and it skips pages already done.

Why no physical split file: this originally used `pdfseparate`, but on this
book poppler hits a "recursive dicts" bug that bloats every single-page
output to ~30MB (1056 pages would need ~31GB -- more than was free on this
disk). scripts/process_one_page.py instead passes start_page_id/end_page_id
straight through to mineru, which extracts pages via pdfium in-memory --
the same page comes out around 100KB, no disk write needed at all.

Each page is run through scripts/process_one_page.py rather than the
`mineru` CLI directly, since that's the only way to force max_concurrency=1
(the CLI has no flag for it -- the http-client backend otherwise fires up
to 100 concurrent block-level requests for a single page).

A page whose response was truncated (finish_reason=length) still exits 0
and still writes a .md -- just with incomplete content. That's silently
accepted output, not a real success, so it's discarded and retried up to
MAX_RETRIES times. Note: sampling is fully greedy (temp=0, top_k=1), so a
retry only has a chance of curing a *repetition-loop* truncation (server
side KV-cache/slot-reuse state can shift float rounding enough to change
which token wins an argmax tie); it will NOT help a page that is
genuinely too dense to fit in ctx-size -- that will truncate every time,
which is why there's a retry cap instead of looping forever.

Retries happen in later passes over the whole book, not immediately after
a page fails: pass 1 runs every page once; any that failed go in a queue
that's retried in pass 2, then pass 3, etc, up to MAX_RETRIES passes. This
way one page's transient failure doesn't stall the rest of the book behind
repeated immediate retries -- everything else keeps moving first, and the
retry queue is naturally smaller (and cheaper) by the time it's addressed.

Before every retry pass (not the first pass) llamacpp itself is restarted.
Greedy decoding means the model will otherwise reliably reproduce the exact
same truncating loop on a bare re-request; a restart clears its KV-cache /
slot-reuse state so the retry isn't just replaying the same computation.

Usage: ./scripts/split_and_process.py in/PracticalElectronicsForInventors.pdf
"""
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

VLM_PORT = os.environ.get("VLM_PORT", "30000")
LLAMACPP_HEALTH_URL = f"http://localhost:{VLM_PORT}/health"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))


def format_duration(total: int) -> str:
    """Formats a second count as e.g. "1h05m02s" / "5m02s" / "42s"."""
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Timer:
    """Tracks the running avg over all pages processed this session and
    prints a timing line for one page. `remaining` is how many pages are
    left in the *current pass* (first pass or a retry pass), not the whole
    book, since that's what the printed ETA is estimating time for.
    """

    def __init__(self) -> None:
        self.total_elapsed = 0
        self.processed_this_run = 0

    def report(self, label: str, elapsed: int, remaining: int) -> None:
        self.total_elapsed += elapsed
        self.processed_this_run += 1
        avg = self.total_elapsed // self.processed_this_run
        eta = avg * remaining
        print(
            f"{label}: {format_duration(elapsed)} "
            f"(avg {format_duration(avg)}/page, "
            f"ETA {format_duration(eta)} for {remaining} page(s) left)"
        )


def llamacpp_healthy() -> bool:
    try:
        with urllib.request.urlopen(LLAMACPP_HEALTH_URL, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError):
        return False


def restart_llamacpp() -> None:
    """Restarts llamacpp and blocks until it reports healthy again. Rootless
    Podman's network teardown on restart is occasionally flaky ("rootless
    netns: kill network process: permission denied"), which can leave the
    shared network briefly broken for other containers -- so both the
    restart command itself and the health check are retried a few times
    rather than barreling ahead against a network that isn't back yet.
    """
    print("Restarting llamacpp before retry pass...")
    for restart_tries in range(1, 4):
        subprocess.run(
            ["podman", "restart", "llamacpp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        waited = 0
        while not llamacpp_healthy() and waited < 30:
            time.sleep(1)
            waited += 1
        if llamacpp_healthy():
            return
        print(
            f"llamacpp restart attempt {restart_tries}/3 not healthy yet, "
            "retrying restart...",
            file=sys.stderr,
        )
        time.sleep(5)
    print(
        "llamacpp still not healthy after 3 restart attempts -- proceeding anyway",
        file=sys.stderr,
    )


def attempt_page(
    basename: str,
    pages_dir: Path,
    fail_log: Path,
    total_pages: int,
    page_num: int,
    page_name: str,
    flat_md: Path,
    flat_images_dir: Path,
    attempt: int,
) -> bool:
    """Runs one page for a single attempt/pass. Returns True and leaves
    flat_md in place on success; returns False (with a fail_log entry) on
    truncation or error, leaving it up to the caller to queue the page for
    a later retry pass.
    """
    print(f"[{page_num}/{total_pages}] processing {page_name} (pass {attempt}/{MAX_RETRIES})")

    page_dir = pages_dir / page_name
    if page_dir.exists():
        subprocess.run(["rm", "-rf", str(page_dir)])

    cmd = [
        "podman-compose", "--profile", "client", "run", "--rm",
        "-v", f"{Path(__file__).resolve().parent}:/scripts:z",
        "mineru-vlm",
        "python3", "/scripts/process_one_page.py",
        f"/data/in/{basename}.pdf",
        str(page_num),
        f"/data/out/{basename}_pages",
        "http://llamacpp:30000",
    ]
    # print("*"*90)

    print(f"Running: {' '.join(cmd)}")
    print(f"Output will be written to {flat_md} and {flat_images_dir}")
    # print("*" * 80)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    output_lines = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)
    proc.wait()
    output = "".join(output_lines)

    with fail_log.open("a") as f:
        if "truncated due to length" in output:
            f.write(f"{page_name}: truncated on pass {attempt}/{MAX_RETRIES}, discarding\n")
            return False

        if proc.returncode == 0:
            src_md = pages_dir / page_name / "vlm" / f"{page_name}.md"
            if src_md.is_file():
                flat_md.write_bytes(src_md.read_bytes())
                # Images referenced by the md (e.g. "images/<hash>.jpg") live
                # in a sibling "images" dir next to the source md -- copy
                # them into the same aggregate images dir so relative image
                # links in the flattened md still resolve. Filenames are
                # content-hashed, so pages sharing an image can't collide.
                src_images_dir = pages_dir / page_name / "vlm" / "images"
                if src_images_dir.is_dir():
                    for src_image in src_images_dir.iterdir():
                        if src_image.is_file():
                            shutil.copy2(src_image, flat_images_dir / src_image.name)
                return True
            f.write(f"{page_name}: mineru exited 0 but no md at {src_md} (pass {attempt})\n")
        else:
            f.write(f"{page_name}: mineru exited non-zero (pass {attempt})\n")

    return False


def get_total_pages(pdf: Path) -> int:
    result = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    match = re.search(r"^Pages:\s*(\d+)", result.stdout, re.MULTILINE)
    if not match:
        print("Could not determine page count via pdfinfo", file=sys.stderr)
        sys.exit(1)
    return int(match.group(1))


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <pdf>", file=sys.stderr)
        sys.exit(1)

    pdf = Path(sys.argv[1]).resolve()
    basename = pdf.stem
    os.environ["IN_DIR"] = str(pdf.parent.resolve())

    # Note: intentionally NOT named OUT_DIR -- compose.yaml also interpolates
    # an OUT_DIR var (from mineru/.env, exported into this script's
    # environment by start.sh's `set -a; source .env`). Reusing that name
    # here previously clobbered the exported value (bash keeps a variable
    # exported across plain reassignment), which made podman-compose
    # bind-mount /data/out one directory too deep and doubled the _pages
    # suffix in every page's output path -- every page silently succeeded
    # one level down from where this script looked, so it was endlessly
    # (and wrongly) retried as "no md" forever.
    pages_dir = Path(f"./out/{basename}_pages")
    flat_dir = Path(f"./out/{basename}_md")
    flat_images_dir = flat_dir / "images"
    fail_log = Path(f"./out/{basename}_failures.txt")

    pages_dir.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    flat_images_dir.mkdir(parents=True, exist_ok=True)

    total_pages = get_total_pages(pdf)
    print(f"Processing {total_pages} pages from {pdf}")

    fail_log.write_text("")
    done_count = 0
    timer = Timer()

    def run_page(page_num: int, attempt: int) -> bool:
        page_name = f"{basename}_page_{page_num:04d}"
        flat_md = flat_dir / f"{page_name}.md"
        return attempt_page(
            basename, pages_dir, fail_log, total_pages,
            page_num, page_name, flat_md, flat_images_dir, attempt,
        )

    retry_queue = []

    # Pass 1: every page not already done, once each. Failures are queued
    # for later rather than retried on the spot -- see module docstring.
    for page_num in range(1, total_pages + 1):
        page_name = f"{basename}_page_{page_num:04d}"
        flat_md = flat_dir / f"{page_name}.md"

        if flat_md.is_file():
            done_count += 1
            # Backfills images for pages that were completed by an older
            # version of this script (before it copied images), so a rerun
            # over an already-processed book doesn't need to redo any pages
            # just to pick up their images.
            src_images_dir = pages_dir / page_name / "vlm" / "images"
            if src_images_dir.is_dir():
                for src_image in src_images_dir.iterdir():
                    if src_image.is_file():
                        dest_image = flat_images_dir / src_image.name
                        if not dest_image.exists():
                            shutil.copy2(src_image, dest_image)
            continue

        start = time.monotonic()
        if run_page(page_num, 1):
            done_count += 1
        else:
            retry_queue.append(page_num)

        # Average/ETA are over pages actually run this session -- pages
        # skipped because they were already done (above) cost ~0s and would
        # otherwise drag the average down and make the ETA meaningless.
        elapsed = int(time.monotonic() - start)
        remaining = total_pages - page_num
        timer.report(f"[{page_num}/{total_pages}] {page_name}", elapsed, remaining)

    # Passes 2..MAX_RETRIES: retry whatever's left in the queue, restarting
    # llamacpp once per pass (not once per page -- see module docstring on
    # why a restart matters for a bare retry).
    attempt = 2
    while attempt <= MAX_RETRIES and retry_queue:
        print(f"Retry pass {attempt}/{MAX_RETRIES} for {len(retry_queue)} failed page(s)")
        restart_llamacpp()
        next_queue = []
        retry_total = len(retry_queue)
        for idx, page_num in enumerate(retry_queue, start=1):
            page_name = f"{basename}_page_{page_num:04d}"

            start = time.monotonic()
            if run_page(page_num, attempt):
                done_count += 1
            else:
                next_queue.append(page_num)

            elapsed = int(time.monotonic() - start)
            remaining = retry_total - idx
            timer.report(
                f"[retry {attempt}/{MAX_RETRIES} {idx}/{retry_total}] {page_name}",
                elapsed, remaining,
            )
        retry_queue = next_queue
        attempt += 1

    failed_count = len(retry_queue)
    with fail_log.open("a") as f:
        for page_num in retry_queue:
            page_name = f"{basename}_page_{page_num:04d}"
            f.write(
                f"{page_name}: FAILED after {MAX_RETRIES} passes "
                "(truncation and/or errors) -- see entries above\n"
            )

    print(f"Done. {done_count}/{total_pages} pages have markdown in {flat_dir}")
    if failed_count > 0:
        print(f"{failed_count} page(s) failed -- see {fail_log}. Rerun this script to retry just those.")

    print("Unloading VLM model from GPU...")
    subprocess.run(["podman", "stop", "-t", "5", "llamacpp"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("GPU VRAM released.")


if __name__ == "__main__":
    main()
