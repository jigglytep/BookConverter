#!/usr/bin/env bash
# Run mineru on a PDF one page at a time, without physically pre-splitting it.
#
# Why per-page: a single malformed VLM response can hard-crash a whole
# multi-hour mineru task (see runbook 1.6 / the 2026-07-28 incident where a
# 500 "peg-native format" error killed a 4.5h/640-page run with nothing
# written to disk). One task per page means one bad page costs one page,
# and the script is resumable -- rerun it and it skips pages already done.
#
# Why no physical split file: this originally used `pdfseparate`, but on this
# book poppler hits a "recursive dicts" bug that bloats every single-page
# output to ~30MB (1056 pages would need ~31GB -- more than was free on this
# disk). scripts/process_one_page.py instead passes start_page_id/end_page_id
# straight through to mineru, which extracts pages via pdfium in-memory --
# the same page comes out around 100KB, no disk write needed at all.
#
# Each page is run through scripts/process_one_page.py rather than the
# `mineru` CLI directly, since that's the only way to force max_concurrency=1
# (the CLI has no flag for it -- the http-client backend otherwise fires up
# to 100 concurrent block-level requests for a single page).
#
# A page whose response was truncated (finish_reason=length) still exits 0
# and still writes a .md -- just with incomplete content. That's silently
# accepted output, not a real success, so it's discarded and retried up to
# MAX_RETRIES times. Note: sampling is fully greedy (temp=0, top_k=1), so a
# retry only has a chance of curing a *repetition-loop* truncation (server
# side KV-cache/slot-reuse state can shift float rounding enough to change
# which token wins an argmax tie); it will NOT help a page that is
# genuinely too dense to fit in ctx-size -- that will truncate every time,
# which is why there's a retry cap instead of looping forever.
#
# Retries happen in later passes over the whole book, not immediately after
# a page fails: pass 1 runs every page once; any that failed go in a queue
# that's retried in pass 2, then pass 3, etc, up to MAX_RETRIES passes. This
# way one page's transient failure doesn't stall the rest of the book behind
# repeated immediate retries -- everything else keeps moving first, and the
# retry queue is naturally smaller (and cheaper) by the time it's addressed.
#
# Before every retry pass (not the first pass) llamacpp itself is restarted.
# Greedy decoding means the model will otherwise reliably reproduce the exact
# same truncating loop on a bare re-request; a restart clears its KV-cache /
# slot-reuse state so the retry isn't just replaying the same computation.
#
# Usage: ./scripts/split_and_process.sh in/PracticalElectronicsForInventors.pdf
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMACPP_HEALTH_URL="http://localhost:${VLM_PORT:-30000}/health"

PDF="$1"
PDF_DIR="$(cd "$(dirname "$PDF")" && pwd)"
export IN_DIR="$PDF_DIR"
BASENAME="$(basename "$PDF" .pdf)"
MAX_RETRIES="${MAX_RETRIES:-3}"

# Note: intentionally NOT named OUT_DIR -- compose.yaml also interpolates an
# OUT_DIR var (from mineru/.env, exported into this script's environment by
# start.sh's `set -a; source .env`). Reusing that name here previously
# clobbered the exported value (bash keeps a variable exported across plain
# reassignment), which made podman-compose bind-mount /data/out one directory
# too deep and doubled the _pages suffix in every page's output path -- every
# page silently succeeded one level down from where this script looked,
# so it was endlessly (and wrongly) retried as "no md" forever.
PAGES_DIR="./out/${BASENAME}_pages"
FLAT_DIR="./out/${BASENAME}_md"
FLAT_IMAGES_DIR="$FLAT_DIR/images"
FAIL_LOG="./out/${BASENAME}_failures.txt"

mkdir -p "$PAGES_DIR" "$FLAT_DIR" "$FLAT_IMAGES_DIR"

TOTAL_PAGES="$(pdfinfo "$PDF" | awk '/^Pages:/ {print $2}')"
if [ -z "$TOTAL_PAGES" ]; then
    echo "Could not determine page count via pdfinfo" >&2
    exit 1
fi
echo "Processing $TOTAL_PAGES pages from $PDF"

: > "$FAIL_LOG"
done_count=0
failed_count=0
total_elapsed=0
processed_this_run=0

# Formats a second count as e.g. "1h05m02s" / "5m02s" / "42s".
format_duration() {
    local total=$1 h m s
    h=$((total / 3600))
    m=$(((total % 3600) / 60))
    s=$((total % 60))
    if [ "$h" -gt 0 ]; then
        printf '%dh%02dm%02ds' "$h" "$m" "$s"
    elif [ "$m" -gt 0 ]; then
        printf '%dm%02ds' "$m" "$s"
    else
        printf '%ds' "$s"
    fi
}

# Updates the running avg over all pages processed this session and prints
# a timing line for one page. $remaining is how many pages are left in the
# *current pass* (first pass or a retry pass), not the whole book, since
# that's what the printed ETA is estimating time for.
report_timing() {
    local label="$1" elapsed="$2" remaining="$3"
    total_elapsed=$((total_elapsed + elapsed))
    processed_this_run=$((processed_this_run + 1))
    local avg=$((total_elapsed / processed_this_run))
    local eta=$((avg * remaining))
    echo "$label: $(format_duration "$elapsed") (avg $(format_duration "$avg")/page, ETA $(format_duration "$eta") for $remaining page(s) left)"
}

# Restarts llamacpp and blocks until it reports healthy again. Rootless
# Podman's network teardown on restart is occasionally flaky ("rootless
# netns: kill network process: permission denied"), which can leave the
# shared network briefly broken for other containers -- so both the restart
# command itself and the health check are retried a few times rather than
# barreling ahead against a network that isn't back yet.
restart_llamacpp() {
    echo "Restarting llamacpp before retry pass..."
    local restart_tries=0
    while [ "$restart_tries" -lt 3 ]; do
        podman restart llamacpp >/dev/null 2>&1
        local waited=0
        until curl -fsS "$LLAMACPP_HEALTH_URL" >/dev/null 2>&1; do
            sleep 1
            waited=$((waited + 1))
            if [ "$waited" -ge 30 ]; then
                break
            fi
        done
        if curl -fsS "$LLAMACPP_HEALTH_URL" >/dev/null 2>&1; then
            return 0
        fi
        restart_tries=$((restart_tries + 1))
        echo "llamacpp restart attempt $restart_tries/3 not healthy yet, retrying restart..." >&2
        sleep 5
    done
    echo "llamacpp still not healthy after 3 restart attempts -- proceeding anyway" >&2
}

# Runs one page for a single attempt/pass. Returns 0 and leaves $flat_md in
# place on success; returns 1 (with a $FAIL_LOG entry) on truncation or error,
# leaving it up to the caller to queue the page for a later retry pass.
attempt_page() {
    local page_num="$1"
    local page_name="$2"
    local flat_md="$3"
    local attempt="$4"
    local page_log
    page_log="$(mktemp)"

    echo "[$page_num/$TOTAL_PAGES] processing $page_name (pass $attempt/$MAX_RETRIES)"
    rm -rf "${PAGES_DIR:?}/${page_name:?}"
    if podman-compose --profile client run --rm -v "$SCRIPT_DIR:/scripts:z" mineru-vlm \
         python3 /scripts/process_one_page.py \
                "/data/in/${BASENAME}.pdf" \
                "$page_num" \
                "/data/out/${BASENAME}_pages" \
                http://llamacpp:30000 2>&1 | tee "$page_log"; then
        cmd_status=0
    else
        cmd_status=1
    fi

    if grep -q "truncated due to length" "$page_log"; then
        echo "$page_name: truncated on pass $attempt/$MAX_RETRIES, discarding" >> "$FAIL_LOG"
        rm -f "$page_log"
        return 1
    fi

    if [ "$cmd_status" -eq 0 ]; then
        src_md="$PAGES_DIR/$page_name/vlm/${page_name}.md"
        if [ -f "$src_md" ]; then
            cp "$src_md" "$flat_md"
            # Images referenced by the md (e.g. "images/<hash>.jpg") live in
            # a sibling "images" dir next to the source md -- copy them into
            # the aggregate images dir so relative image links in the
            # flattened md still resolve. Filenames are content-hashed, so
            # pages sharing an image can't collide.
            src_images_dir="$PAGES_DIR/$page_name/vlm/images"
            if [ -d "$src_images_dir" ]; then
                cp -n "$src_images_dir"/* "$FLAT_IMAGES_DIR"/ 2>/dev/null
            fi
            rm -f "$page_log"
            return 0
        fi
        echo "$page_name: mineru exited 0 but no md at $src_md (pass $attempt)" >> "$FAIL_LOG"
    else
        echo "$page_name: mineru exited non-zero (pass $attempt)" >> "$FAIL_LOG"
    fi
    rm -f "$page_log"
    return 1
}

retry_queue=()

# Pass 1: every page not already done, once each. Failures are queued for
# later rather than retried on the spot -- see header comment.
for ((page_num=1; page_num<=TOTAL_PAGES; page_num++)); do
    page_name="$(printf '%s_page_%04d' "$BASENAME" "$page_num")"
    flat_md="$FLAT_DIR/${page_name}.md"

    if [ -f "$flat_md" ]; then
        done_count=$((done_count + 1))
        # Backfills images for pages that were completed by an older
        # version of this script (before it copied images), so a rerun
        # over an already-processed book doesn't need to redo any pages
        # just to pick up their images.
        src_images_dir="$PAGES_DIR/$page_name/vlm/images"
        if [ -d "$src_images_dir" ]; then
            cp -n "$src_images_dir"/* "$FLAT_IMAGES_DIR"/ 2>/dev/null
        fi
        continue
    fi

    page_start=$SECONDS
    if attempt_page "$page_num" "$page_name" "$flat_md" 1; then
        done_count=$((done_count + 1))
    else
        retry_queue+=("$page_num")
    fi

    # Average/ETA are over pages actually run this session -- pages skipped
    # because they were already done (above) cost ~0s and would otherwise
    # drag the average down and make the ETA meaningless.
    page_elapsed=$((SECONDS - page_start))
    remaining=$((TOTAL_PAGES - page_num))
    report_timing "[$page_num/$TOTAL_PAGES] $page_name" "$page_elapsed" "$remaining"
done

# Passes 2..MAX_RETRIES: retry whatever's left in the queue, restarting
# llamacpp once per pass (not once per page -- see header comment on why
# a restart matters for a bare retry).
attempt=2
while [ "$attempt" -le "$MAX_RETRIES" ] && [ "${#retry_queue[@]}" -gt 0 ]; do
    echo "Retry pass $attempt/$MAX_RETRIES for ${#retry_queue[@]} failed page(s)"
    restart_llamacpp
    next_queue=()
    retry_total=${#retry_queue[@]}
    idx=0
    for page_num in "${retry_queue[@]}"; do
        idx=$((idx + 1))
        page_name="$(printf '%s_page_%04d' "$BASENAME" "$page_num")"
        flat_md="$FLAT_DIR/${page_name}.md"

        page_start=$SECONDS
        if attempt_page "$page_num" "$page_name" "$flat_md" "$attempt"; then
            done_count=$((done_count + 1))
        else
            next_queue+=("$page_num")
        fi

        page_elapsed=$((SECONDS - page_start))
        remaining=$((retry_total - idx))
        report_timing "[retry $attempt/$MAX_RETRIES $idx/$retry_total] $page_name" "$page_elapsed" "$remaining"
    done
    retry_queue=("${next_queue[@]}")
    attempt=$((attempt + 1))
done

failed_count=${#retry_queue[@]}
for page_num in "${retry_queue[@]}"; do
    page_name="$(printf '%s_page_%04d' "$BASENAME" "$page_num")"
    echo "$page_name: FAILED after $MAX_RETRIES passes (truncation and/or errors) -- see entries above" >> "$FAIL_LOG"
done

echo "Done. $done_count/$TOTAL_PAGES pages have markdown in $FLAT_DIR"
if [ "$failed_count" -gt 0 ]; then
    echo "$failed_count page(s) failed -- see $FAIL_LOG. Rerun this script to retry just those."
fi

echo "Unloading VLM model from GPU..."
podman stop -t 5 llamacpp >/dev/null 2>&1 || true
echo "GPU VRAM released."
