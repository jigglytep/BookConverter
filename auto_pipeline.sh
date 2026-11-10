#!/usr/bin/env bash
# ==============================================================================
# auto_pipeline.sh - Fully Automated PDF to Audiobook Conversion Pipeline
#
# Orchestrates 3 conversion stages sequentially, processing one PDF at a time:
#   1. MinerU OCR (mineru/batch_fast.sh)
#   2. TTS Markdown Stripper (TTS_Script/batch_fast.sh)
#   3. Qwen3-TTS / Chatterbox TTS & AudioBookshelf Upload (chatterbox/batch_fast.sh)
#
# Designed for standalone execution and recurring CRON automation.
# Uses a strict PID lockfile to prevent overlapping runs and GPU contention.
# ==============================================================================
set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

_CALLER_TTS_ENGINE="${TTS_ENGINE:-}"
_CALLER_QWEN_TTS_ENV="${QWEN_TTS_ENV:-}"
_CALLER_HF_HOME="${HF_HOME:-}"

set -a
[[ -f "$BASE_DIR/.env" ]] && source "$BASE_DIR/.env"
set +a

export TTS_ENGINE="${_CALLER_TTS_ENGINE:-${TTS_ENGINE:-qwen3}}"
export QWEN_TTS_ENV="${_CALLER_QWEN_TTS_ENV:-${QWEN_TTS_ENV:-/mnt/raid0/qwen/src/tts-env}}"
export HF_HOME="${_CALLER_HF_HOME:-${HF_HOME:-/mnt/raid0/qwen/huggingface}}"

IN_DIR="${BASE_DIR}/in"
COMPLETED_DIR="${BASE_DIR}/completed"
FAILED_DIR="${BASE_DIR}/failed"
LOGS_DIR="${BASE_DIR}/logs"

MINERU_DIR="${BASE_DIR}/mineru"
TTS_DIR="${BASE_DIR}/TTS_Script"
CHATTERBOX_DIR="${BASE_DIR}/chatterbox"

LOCK_FILE="${LOGS_DIR}/auto_pipeline.lock"
LOG_FILE="${LOGS_DIR}/auto_pipeline_$(date +%Y%m%d).log"
FAILURES_FILE="${LOGS_DIR}/pipeline_failures.txt"

# Default flags
DRY_RUN=false
DRAIN_BACKLOG=true
STOP_ON_ERROR=false
FORCE=false
CLEAN_IN_PROGRESS=false
CUSTOM_PDF=""
chatterbox_forward_args=()
mineru_forward_args=()
display_args=()
RUN_DEMO=false
ORIG_ARGS=("$@")

emit_status_event() {
  if [[ -n "${STATUS_DISPLAY_EVENT_FILE:-}" ]]; then
    echo "$1" >> "$STATUS_DISPLAY_EVENT_FILE" 2>/dev/null || true
  fi
}

# Terminal styling
if [[ -t 1 ]]; then
  C_RESET="\033[0m"
  C_BOLD="\033[1m"
  C_RED="\033[31m"
  C_GREEN="\033[32m"
  C_YELLOW="\033[33m"
  C_BLUE="\033[34m"
  C_CYAN="\033[36m"
else
  C_RESET=""
  C_BOLD=""
  C_RED=""
  C_GREEN=""
  C_YELLOW=""
  C_BLUE=""
  C_CYAN=""
fi

log() {
  local timestamp
  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  local level="${1:-INFO}"
  shift
  local color="$C_RESET"
  case "$level" in
    INFO)    color="$C_CYAN" ;;
    SUCCESS) color="$C_GREEN" ;;
    WARN)    color="$C_YELLOW" ;;
    ERROR)   color="$C_RED" ;;
    STEP)    color="$C_BOLD$C_BLUE" ;;
  esac

  echo -e "${color}[${timestamp}] [${level}] $*${C_RESET}"
  echo "[${timestamp}] [${level}] $*" >> "$LOG_FILE" 2>/dev/null || true
}

show_help() {
  cat << EOF
Usage: $0 [options] [path/to/specific.pdf]

Automates the complete PDF-to-audiobook conversion pipeline:
  1. Ingests PDF from in/ -> mineru/in/ -> mineru/batch_fast.sh
  2. Runs TTS_Script/batch_fast.sh (Markdown -> Clean Text)
  3. Runs chatterbox/batch_fast.sh (Audiobook .m4b + EPUB + AudioBookshelf upload)

Options:
  -n, --dry-run          Preview actions and pending queue without moving files or executing stages
  -c, --clean, --cleanup Clear all existing in-progress items (mineru/in, mineru/out, TTS_Script/out,
                         unfinished stage artifacts) and Chatterbox chunks & audio cache
  --no-drain             Skip backlog draining (do not process leftover mineru/out or TTS_Script/out folders)
  --force                Force re-conversion even if final audiobook (.m4b) already exists
  --stop-on-error        Halt immediately if any book fails (default: quarantine to failed/ and continue)
  --skip-upload          Skip AudioBookshelf upload step
  --skip-epub            Skip EPUB generation step
  --skip-pdf             Skip uploading companion PDF to AudioBookshelf
  --library <name>       Target AudioBookshelf library name (default: TTS)
  --tts-engine <engine>  TTS engine: qwen3 (default) or chatterbox (or env TTS_ENGINE)
  --qwen3                Use Qwen3-TTS engine (default)
  --chatterbox           Use Chatterbox TTS fallback engine
  --model <model>        TTS model: standard, turbo (default: standard)
  --reference-id <id>    Preset reference voice ID (default: practical_engineering)
  --reference-audio <f>  Path to reference audio WAV for voice cloning
  --reference-text <txt> Transcript text or .lab/.txt path for reference audio
  --workers <N>          Worker count forwarded to mineru/batch_fast.sh
  --plain                Non-TTY / unattended mode: line-per-event + periodic progress heartbeat
  --quiet, -q            Suppress routine INFO/DEBUG lines in status display
  --verbose, -v          Include DEBUG-level log lines in status display
  --heartbeat <sec>      Heartbeat interval in seconds for --plain mode (default: 30)
  --demo                 Simulate a fake book queue and GPU telemetry without using the GPU
  -h, --help             Show this help message

Examples:
  $0                             # Process all PDFs in in/
  $0 --dry-run                   # Check queue and status without executing
  $0 --clean                     # Clear all in-progress items, chunks, and TTS audio cache
  $0 in/MyBook.pdf               # Process single specific PDF
  $0 --model turbo --skip-upload # Process queue with turbo TTS and no ABS upload
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    -n|--dry-run)
      DRY_RUN=true
      shift
      ;;
    -c|--clean|--cleanup|--clear-cache)
      CLEAN_IN_PROGRESS=true
      shift
      ;;
    --no-drain)
      DRAIN_BACKLOG=false
      shift
      ;;
    --force)
      FORCE=true
      chatterbox_forward_args+=("--force")
      shift
      ;;
    --stop-on-error)
      STOP_ON_ERROR=true
      shift
      ;;
    --skip-upload|--no-upload)
      chatterbox_forward_args+=("--skip-upload")
      shift
      ;;
    --skip-epub|--no-epub)
      chatterbox_forward_args+=("--skip-epub")
      shift
      ;;
    --skip-pdf|--no-pdf)
      chatterbox_forward_args+=("--skip-pdf")
      shift
      ;;
    --library)
      chatterbox_forward_args+=("--library" "$2")
      shift 2
      ;;
    --library=*)
      chatterbox_forward_args+=("$1")
      shift
      ;;
    --tts-engine|--engine)
      export TTS_ENGINE="$2"
      chatterbox_forward_args+=("--tts-engine" "$2")
      shift 2
      ;;
    --tts-engine=*|--engine=*)
      export TTS_ENGINE="${1#*=}"
      chatterbox_forward_args+=("--tts-engine=${TTS_ENGINE}")
      shift
      ;;
    --qwen3)
      export TTS_ENGINE="qwen3"
      chatterbox_forward_args+=("--qwen3")
      shift
      ;;
    --chatterbox)
      export TTS_ENGINE="chatterbox"
      chatterbox_forward_args+=("--chatterbox")
      shift
      ;;
    --model)
      chatterbox_forward_args+=("--model" "$2")
      shift 2
      ;;
    --model=*)
      chatterbox_forward_args+=("$1")
      shift
      ;;
    --reference-id)
      chatterbox_forward_args+=("--reference-id" "$2")
      shift 2
      ;;
    --reference-id=*)
      chatterbox_forward_args+=("$1")
      shift
      ;;
    --reference-audio)
      chatterbox_forward_args+=("--reference-audio" "$2")
      shift 2
      ;;
    --reference-audio=*)
      chatterbox_forward_args+=("$1")
      shift
      ;;
    --reference-text)
      chatterbox_forward_args+=("--reference-text" "$2")
      shift 2
      ;;
    --reference-text=*)
      chatterbox_forward_args+=("$1")
      shift
      ;;
    --workers-per-gpu)
      chatterbox_forward_args+=("--workers-per-gpu" "$2")
      shift 2
      ;;
    --workers-per-gpu=*)
      chatterbox_forward_args+=("$1")
      shift
      ;;
    --workers)
      mineru_forward_args+=("--workers" "$2")
      shift 2
      ;;
    --workers=*)
      mineru_forward_args+=("$1")
      shift
      ;;
    --plain|--quiet|-q|--verbose|-v)
      display_args+=("$1")
      shift
      ;;
    --heartbeat)
      display_args+=("--heartbeat" "$2")
      shift 2
      ;;
    --heartbeat=*)
      display_args+=("$1")
      shift
      ;;
    --demo)
      RUN_DEMO=true
      display_args+=("--demo")
      shift
      ;;
    --demo-duration)
      display_args+=("--demo-duration" "$2")
      shift 2
      ;;
    --demo-duration=*)
      display_args+=("$1")
      shift
      ;;
    *)
      if [[ -f "$1" && ("$1" == *.pdf || "$1" == *.PDF) ]]; then
        CUSTOM_PDF="$1"
      else
        log WARN "Unrecognized argument: $1"
      fi
      shift
      ;;
  esac
done

# Ensure directories exist
mkdir -p "$IN_DIR" "$COMPLETED_DIR" "$FAILED_DIR" "$LOGS_DIR"

# Launch live status display supervisor unless already active or running dry-run/clean
if [[ -z "${STATUS_DISPLAY_ACTIVE:-}" ]]; then
  PYTHON_DISPLAY_BIN="${BASE_DIR}/.venv/bin/python3"
  [[ ! -x "$PYTHON_DISPLAY_BIN" ]] && PYTHON_DISPLAY_BIN="$(command -v python3 || true)"
  if [[ -n "$PYTHON_DISPLAY_BIN" && -f "${BASE_DIR}/status_display.py" ]]; then
    if [[ "$RUN_DEMO" == "true" ]]; then
      exec "$PYTHON_DISPLAY_BIN" "${BASE_DIR}/status_display.py" "${display_args[@]}"
    elif [[ "$DRY_RUN" == "false" && "$CLEAN_IN_PROGRESS" == "false" ]]; then
      exec "$PYTHON_DISPLAY_BIN" "${BASE_DIR}/status_display.py" "${display_args[@]}" --exec-pipeline -- "$0" "${ORIG_ARGS[@]}"
    fi
  fi
fi

# Concurrency protection using PID lockfile
acquire_lock() {
  if [[ -f "$LOCK_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$LOCK_FILE" 2>/dev/null || echo "")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      log INFO "Another pipeline instance is actively running (PID: ${existing_pid}). Exiting cleanly to avoid GPU collision."
      exit 0
    else
      log WARN "Removing stale lockfile (previous PID: ${existing_pid} not running)."
      rm -f "$LOCK_FILE"
    fi
  fi

  if [[ "$DRY_RUN" == "false" ]]; then
    echo "$$" > "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"' EXIT INT TERM
  fi
}

# Clean any 0-byte corrupt files in mineru/in
clean_mineru_in() {
  shopt -s nullglob
  local corrupted=("$MINERU_DIR/in"/*)
  shopt -u nullglob

  for item in "${corrupted[@]}"; do
    if [[ -f "$item" && ! -s "$item" ]]; then
      log WARN "Removing empty (0-byte) file in mineru/in/: $(basename "$item")"
      if [[ "$DRY_RUN" == "false" ]]; then
        rm -f "$item"
      fi
    fi
  done
}

# Format seconds to human-readable duration (e.g. 1h 23m 45s)
format_duration() {
  local total_secs=$1
  local hours=$((total_secs / 3600))
  local mins=$(( (total_secs % 3600) / 60 ))
  local secs=$((total_secs % 60))
  if (( hours > 0 )); then
    printf "%dh %dm %ds" "$hours" "$mins" "$secs"
  elif (( mins > 0 )); then
    printf "%dm %ds" "$mins" "$secs"
  else
    printf "%ds" "$secs"
  fi
}

# Check whether a book has already been fully converted into a final audiobook
is_book_completed() {
  local bname="$1"
  [[ "$FORCE" == "true" ]] && return 1
  if [[ -s "${CHATTERBOX_DIR}/audiobooks/${bname}.m4b" ]]; then
    return 0
  fi
  return 1
}

# Archive any residual intermediate folders and link the source PDF into root completed/
mark_book_completed() {
  local bname="$1"
  local pdf_file="${bname}.pdf"

  # Archive leftover TTS_Script/out/<book>_txt -> chatterbox/completed/<book>_txt
  local pending_txt="${TTS_DIR}/out/${bname}_txt"
  if [[ -d "$pending_txt" ]]; then
    mkdir -p "${CHATTERBOX_DIR}/completed"
    rm -rf "${CHATTERBOX_DIR}/completed/${bname}_txt"
    mv "$pending_txt" "${CHATTERBOX_DIR}/completed/"
  fi

  # Archive leftover mineru/out/<book>_md and _pages -> TTS_Script/completed/
  local pending_md="${MINERU_DIR}/out/${bname}_md"
  if [[ -d "$pending_md" ]]; then
    mkdir -p "${TTS_DIR}/completed"
    rm -rf "${TTS_DIR}/completed/${bname}_md"
    mv "$pending_md" "${TTS_DIR}/completed/"
  fi
  local pending_pages="${MINERU_DIR}/out/${bname}_pages"
  if [[ -d "$pending_pages" ]]; then
    mkdir -p "${TTS_DIR}/completed"
    rm -rf "${TTS_DIR}/completed/${bname}_pages"
    mv "$pending_pages" "${TTS_DIR}/completed/"
  fi

  # Ensure root completed/<book>.pdf is linked once the full pipeline is done
  local mineru_completed_pdf="${MINERU_DIR}/completed/${pdf_file}"
  if [[ -f "$mineru_completed_pdf" ]]; then
    ln -f "$mineru_completed_pdf" "${COMPLETED_DIR}/${pdf_file}" 2>/dev/null || \
      cp -p "$mineru_completed_pdf" "${COMPLETED_DIR}/${pdf_file}" 2>/dev/null || true
  fi
}

# Clear all existing in-progress items, unfinished stage artifacts, and TTS caches
clean_in_progress_and_cache() {
  log STEP "Cleaning all in-progress items, intermediate chunks, and TTS audio cache..."

  shopt -s nullglob

  # 1. Clean unfinished Stage 2/3 books in TTS_Script/out/ (and their Stage 1/2 artifacts if not completed)
  local tts_items=("$TTS_DIR/out"/*)
  for item in "${tts_items[@]}"; do
    local name
    name="$(basename "$item")"
    if [[ -d "$item" ]]; then
      local bname="${name%_txt}"
      bname="${bname%_md}"
      log INFO "Removing in-progress TTS text folder: ${item}"
      if [[ "$DRY_RUN" == "false" ]]; then
        rm -rf "$item"
      fi
      # If this book has not produced a final .m4b, also remove its unfinished Stage 1/2 artifacts
      if [[ ! -s "${CHATTERBOX_DIR}/audiobooks/${bname}.m4b" ]]; then
        for artifact in \
          "${TTS_DIR}/completed/${bname}_md" \
          "${TTS_DIR}/completed/${bname}_pages" \
          "${MINERU_DIR}/completed/${bname}.pdf" \
          "${COMPLETED_DIR}/${bname}.pdf"; do
          if [[ -e "$artifact" ]]; then
            log INFO "Removing unfinished book artifact: ${artifact}"
            if [[ "$DRY_RUN" == "false" ]]; then
              rm -rf "$artifact"
            fi
          fi
        done
      fi
    elif [[ -f "$item" ]]; then
      log INFO "Removing file in TTS_Script/out/: ${item}"
      if [[ "$DRY_RUN" == "false" ]]; then
        rm -f "$item"
      fi
    fi
  done

  # 2. Clean in-progress Stage 1 MinerU inputs and outputs (mineru/in/* and mineru/out/*)
  local mineru_in_items=("$MINERU_DIR/in"/*)
  for item in "${mineru_in_items[@]}"; do
    log INFO "Removing staged MinerU input: ${item}"
    if [[ "$DRY_RUN" == "false" ]]; then
      rm -rf "$item"
    fi
  done

  local mineru_out_items=("$MINERU_DIR/out"/*)
  for item in "${mineru_out_items[@]}"; do
    log INFO "Removing MinerU output/log item: ${item}"
    if [[ "$DRY_RUN" == "false" ]]; then
      rm -rf "$item"
    fi
  done

  # 3. Clean Chatterbox intermediate chunk directories (chatterbox/chunks/*)
  local chunk_dirs=("$CHATTERBOX_DIR/chunks"/*)
  if (( ${#chunk_dirs[@]} > 0 )); then
    log INFO "Removing ${#chunk_dirs[@]} intermediate chunk folder(s) in ${CHATTERBOX_DIR}/chunks/..."
    if [[ "$DRY_RUN" == "false" ]]; then
      rm -rf "${CHATTERBOX_DIR}/chunks"/*
    fi
  fi

  # 4. Clean Chatterbox TTS audio chunk cache (chatterbox/cache/audio_chunks/*)
  if [[ -d "${CHATTERBOX_DIR}/cache/audio_chunks" ]]; then
    local cache_count
    cache_count=$(find "${CHATTERBOX_DIR}/cache/audio_chunks" -maxdepth 1 -type f 2>/dev/null | wc -l)
    if (( cache_count > 0 )); then
      log INFO "Clearing ${cache_count} cached WAV chunk(s) in ${CHATTERBOX_DIR}/cache/audio_chunks/..."
      if [[ "$DRY_RUN" == "false" ]]; then
        find "${CHATTERBOX_DIR}/cache/audio_chunks" -maxdepth 1 -type f -delete
      fi
    fi
  fi

  # 5. Clean any partial .raw.tmp files in chatterbox/audiobooks/
  local tmp_raws=("$CHATTERBOX_DIR/audiobooks"/*.raw.tmp)
  for item in "${tmp_raws[@]}"; do
    log INFO "Removing partial raw audio temp file: ${item}"
    if [[ "$DRY_RUN" == "false" ]]; then
      rm -f "$item"
    fi
  done

  shopt -u nullglob
  log SUCCESS "Cleanup of in-progress items and caches complete."
}

# Drain existing backlog in TTS_Script/out and mineru/out
drain_backlog() {
  log STEP "Checking for backlog in TTS_Script/out/ and mineru/out/..."

  # 1. Check TTS_Script/out for text directories awaiting Chatterbox
  shopt -s nullglob
  local tts_pending=("$TTS_DIR/out"/*)
  shopt -u nullglob

  for txt_dir in "${tts_pending[@]}"; do
    [[ -d "$txt_dir" ]] || continue
    local folder_name
    folder_name="$(basename "$txt_dir")"
    local book_name="${folder_name%_txt}"
    book_name="${book_name%_md}"

    if is_book_completed "$book_name"; then
      log INFO "Backlog item '${book_name}' is already converted (audiobooks/${book_name}.m4b exists). Marking as processed and skipping."
      if [[ "$DRY_RUN" == "false" ]]; then
        mark_book_completed "$book_name"
      fi
      continue
    fi

    log INFO "Found pending TTS text folder: ${folder_name}"

    if [[ "$DRY_RUN" == "true" ]]; then
      log INFO "[DRY-RUN] Would run: bash ${CHATTERBOX_DIR}/batch_fast.sh ${txt_dir} ${chatterbox_forward_args[*]}"
      continue
    fi

    log STEP "--- Running Stage 3 (Chatterbox TTS) on backlog: ${folder_name} ---"
    emit_status_event "{\"event\":\"start_book\",\"name\":\"${book_name}\"}"
    if bash "${CHATTERBOX_DIR}/batch_fast.sh" "$txt_dir" "${chatterbox_forward_args[@]}"; then
      mark_book_completed "$book_name"
      emit_status_event "{\"event\":\"end_book\",\"summary\":{\"name\":\"${book_name}\",\"status\":\"done\",\"output_path\":\"chatterbox/audiobooks/${book_name}.m4b\"}}"
      log SUCCESS "Backlog item ${folder_name} completed Stage 3 successfully."
    else
      emit_status_event "{\"event\":\"end_book\",\"summary\":{\"name\":\"${book_name}\",\"status\":\"failed\"}}"
      log ERROR "Backlog item ${folder_name} failed Stage 3."
      echo "Backlog Stage 3 failure: ${folder_name} at $(date)" >> "$FAILURES_FILE"
      if [[ "$STOP_ON_ERROR" == "true" ]]; then
        exit 1
      fi
    fi
  done

  # 2. Check mineru/out for markdown directories awaiting TTS_Script
  shopt -s nullglob
  local mineru_pending=("$MINERU_DIR/out"/*_md)
  shopt -u nullglob

  for md_dir in "${mineru_pending[@]}"; do
    [[ -d "$md_dir" ]] || continue
    local folder_name
    folder_name="$(basename "$md_dir")"
    local book_name="${folder_name%_md}"
    local txt_out_dir="${TTS_DIR}/out/${book_name}_txt"

    if is_book_completed "$book_name"; then
      log INFO "Backlog item '${book_name}' is already converted (audiobooks/${book_name}.m4b exists). Marking as processed and skipping."
      if [[ "$DRY_RUN" == "false" ]]; then
        mark_book_completed "$book_name"
      fi
      continue
    fi

    log INFO "Found pending MinerU markdown folder: ${folder_name}"

    if [[ "$DRY_RUN" == "true" ]]; then
      log INFO "[DRY-RUN] Would run: bash ${TTS_DIR}/batch_fast.sh ${md_dir}"
      log INFO "[DRY-RUN] Would then run: bash ${CHATTERBOX_DIR}/batch_fast.sh ${txt_out_dir} ${chatterbox_forward_args[*]}"
      continue
    fi

    log STEP "--- Running Stage 2 (TTS_Script) on backlog: ${folder_name} ---"
    emit_status_event "{\"event\":\"start_book\",\"name\":\"${book_name}\"}"
    if bash "${TTS_DIR}/batch_fast.sh" "$md_dir"; then
      log SUCCESS "Backlog item ${folder_name} stripped successfully."

      log STEP "--- Running Stage 3 (Chatterbox TTS) on backlog: ${book_name} ---"
      if bash "${CHATTERBOX_DIR}/batch_fast.sh" "$txt_out_dir" "${chatterbox_forward_args[@]}"; then
        mark_book_completed "$book_name"
        emit_status_event "{\"event\":\"end_book\",\"summary\":{\"name\":\"${book_name}\",\"status\":\"done\",\"output_path\":\"chatterbox/audiobooks/${book_name}.m4b\"}}"
        log SUCCESS "Backlog item ${book_name} completed Stage 3 successfully."
      else
        emit_status_event "{\"event\":\"end_book\",\"summary\":{\"name\":\"${book_name}\",\"status\":\"failed\"}}"
        log ERROR "Backlog item ${book_name} failed Stage 3."
        echo "Backlog Stage 3 failure: ${book_name} at $(date)" >> "$FAILURES_FILE"
        if [[ "$STOP_ON_ERROR" == "true" ]]; then
          exit 1
        fi
      fi
    else
      log ERROR "Backlog item ${folder_name} failed Stage 2."
      echo "Backlog Stage 2 failure: ${folder_name} at $(date)" >> "$FAILURES_FILE"
      if [[ "$STOP_ON_ERROR" == "true" ]]; then
        exit 1
      fi
    fi
  done

  log SUCCESS "Backlog check complete."
}

# Main pipeline process for a single PDF
process_single_pdf() {
  local pdf="$1"
  local pdf_filename
  pdf_filename="$(basename "$pdf")"
  local book_name="${pdf_filename%.*}"

  echo "================================================================================"
  log STEP ">>> Processing Book: ${pdf_filename}"
  echo "================================================================================"

  # 1. Validation: AppleDouble check
  if [[ "$pdf_filename" == ._* || "$pdf_filename" == .* ]]; then
    log WARN "Ignoring metadata / hidden file: ${pdf_filename}"
    return 0
  fi

  # 2. Validation: Empty file check
  if [[ ! -s "$pdf" ]]; then
    log ERROR "File is empty (0 bytes): ${pdf_filename}. Quarantining to ${FAILED_DIR}/."
    if [[ "$DRY_RUN" == "false" ]]; then
      mv "$pdf" "${FAILED_DIR}/"
      echo "${pdf_filename}: Skipped (0 bytes) at $(date)" >> "$FAILURES_FILE"
    fi
    return 1
  fi

  # 3. Validation: Stable file check (ensure write/copy is finished)
  if [[ "$DRY_RUN" == "false" ]]; then
    local size1 size2
    size1=$(stat -c %s "$pdf" 2>/dev/null || echo 0)
    sleep 2
    size2=$(stat -c %s "$pdf" 2>/dev/null || echo 0)
    if [[ "$size1" -ne "$size2" ]]; then
      log WARN "File size changing (${size1} -> ${size2} bytes). Write in progress. Skipping for next run."
      return 0
    fi
  fi

  # 4. Validation: %PDF magic bytes check
  local magic
  magic=$(head -c 4 "$pdf" 2>/dev/null || true)
  if [[ "$magic" != "%PDF" ]]; then
    log ERROR "${pdf_filename} is not a valid PDF file (header: '${magic}'). Quarantining to ${FAILED_DIR}/."
    if [[ "$DRY_RUN" == "false" ]]; then
      mv "$pdf" "${FAILED_DIR}/"
      echo "${pdf_filename}: Invalid PDF header at $(date)" >> "$FAILURES_FILE"
    fi
    return 1
  fi

  # 5. Check if book is already fully converted
  if is_book_completed "$book_name"; then
    log INFO "Book '${book_name}' is already converted (audiobooks/${book_name}.m4b exists). Marking as processed and skipping."
    if [[ "$DRY_RUN" == "false" ]]; then
      mv -f "$pdf" "${MINERU_DIR}/completed/${pdf_filename}"
      mark_book_completed "$book_name"
      emit_status_event "{\"event\":\"end_book\",\"summary\":{\"name\":\"${book_name}\",\"status\":\"skipped\",\"output_path\":\"chatterbox/audiobooks/${book_name}.m4b\"}}"
    fi
    return 0
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    log INFO "[DRY-RUN] Would move ${pdf} -> ${MINERU_DIR}/in/${pdf_filename}"
    log INFO "[DRY-RUN] Would run: bash ${MINERU_DIR}/batch_fast.sh ${mineru_forward_args[*]}"
    log INFO "[DRY-RUN] Would run: bash ${TTS_DIR}/batch_fast.sh ${MINERU_DIR}/out/${book_name}_md"
    log INFO "[DRY-RUN] Would run: bash ${CHATTERBOX_DIR}/batch_fast.sh ${TTS_DIR}/out/${book_name}_txt ${chatterbox_forward_args[*]}"
    log INFO "[DRY-RUN] Would link ${MINERU_DIR}/completed/${pdf_filename} -> ${COMPLETED_DIR}/${pdf_filename}"
    return 0
  fi

  local book_start_time
  book_start_time=$(date +%s)

  # ----------------------------------------------------------------------------
  # Stage 1: Move to mineru/in/ and run MinerU OCR
  # ----------------------------------------------------------------------------
  log STEP "[Stage 1/3] Moving to mineru/in/ and executing MinerU OCR..."
  local mineru_target="${MINERU_DIR}/in/${pdf_filename}"
  mv "$pdf" "$mineru_target"

  local s1_start s1_end s1_dur
  s1_start=$(date +%s)

  if ! bash "${MINERU_DIR}/batch_fast.sh" "${mineru_forward_args[@]}"; then
    s1_end=$(date +%s)
    s1_dur=$((s1_end - s1_start))
    log ERROR "MinerU OCR failed for ${pdf_filename} after $(format_duration "$s1_dur")."
    if [[ -f "$mineru_target" ]]; then
      log WARN "Quarantining ${pdf_filename} -> ${FAILED_DIR}/"
      mv "$mineru_target" "${FAILED_DIR}/"
    fi
    echo "${pdf_filename}: Stage 1 MinerU failed at $(date)" >> "$FAILURES_FILE"
    if [[ "$STOP_ON_ERROR" == "true" ]]; then
      exit 1
    fi
    return 1
  fi

  s1_end=$(date +%s)
  s1_dur=$((s1_end - s1_start))
  log SUCCESS "Stage 1 (MinerU OCR) finished successfully in $(format_duration "$s1_dur")."

  # ----------------------------------------------------------------------------
  # Stage 2: Run TTS_Script Markdown Stripper
  # ----------------------------------------------------------------------------
  log STEP "[Stage 2/3] Stripping Markdown to plain text via TTS_Script..."
  local md_dir="${MINERU_DIR}/out/${book_name}_md"
  local txt_dir="${TTS_DIR}/out/${book_name}_txt"

  local s2_start s2_end s2_dur
  s2_start=$(date +%s)

  if ! bash "${TTS_DIR}/batch_fast.sh" "$md_dir"; then
    s2_end=$(date +%s)
    s2_dur=$((s2_end - s2_start))
    log ERROR "TTS_Script markdown stripping failed for ${book_name} after $(format_duration "$s2_dur")."
    echo "${pdf_filename}: Stage 2 TTS_Script failed at $(date)" >> "$FAILURES_FILE"
    if [[ "$STOP_ON_ERROR" == "true" ]]; then
      exit 1
    fi
    return 1
  fi

  s2_end=$(date +%s)
  s2_dur=$((s2_end - s2_start))
  log SUCCESS "Stage 2 (TTS_Script) finished successfully in $(format_duration "$s2_dur")."

  # ----------------------------------------------------------------------------
  # Stage 3: Run Chatterbox TTS, EPUB generation, and AudioBookshelf Upload
  # ----------------------------------------------------------------------------
  log STEP "[Stage 3/3] Synthesizing audiobook, generating EPUB, and uploading to AudioBookshelf..."

  local s3_start s3_end s3_dur
  s3_start=$(date +%s)

  if ! bash "${CHATTERBOX_DIR}/batch_fast.sh" "$txt_dir" "${chatterbox_forward_args[@]}"; then
    s3_end=$(date +%s)
    s3_dur=$((s3_end - s3_start))
    log ERROR "Chatterbox TTS / AudioBookshelf pipeline failed for ${book_name} after $(format_duration "$s3_dur")."
    echo "${pdf_filename}: Stage 3 Chatterbox failed at $(date)" >> "$FAILURES_FILE"
    if [[ "$STOP_ON_ERROR" == "true" ]]; then
      exit 1
    fi
    return 1
  fi

  s3_end=$(date +%s)
  s3_dur=$((s3_end - s3_start))
  log SUCCESS "Stage 3 (Chatterbox & ABS) finished successfully in $(format_duration "$s3_dur")."

  mark_book_completed "$book_name"

  local book_end_time total_book_dur
  book_end_time=$(date +%s)
  total_book_dur=$((book_end_time - book_start_time))
  emit_status_event "{\"event\":\"end_book\",\"summary\":{\"name\":\"${book_name}\",\"status\":\"done\",\"duration_sec\":${total_book_dur},\"output_path\":\"chatterbox/audiobooks/${book_name}.m4b\"}}"

  echo "--------------------------------------------------------------------------------"
  log SUCCESS ">>> FULL PIPELINE COMPLETED for: ${pdf_filename}"
  log SUCCESS "    Total time: $(format_duration "$total_book_dur") (OCR: $(format_duration "$s1_dur"), Text: $(format_duration "$s2_dur"), Audio & ABS: $(format_duration "$s3_dur"))"
  echo "--------------------------------------------------------------------------------"
  echo
  return 0
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
echo "================================================================================"
echo -e "${C_BOLD}Automated PDF to Audiobook Conversion Pipeline${C_RESET}"
echo "================================================================================"
log INFO "TTS Engine:          ${TTS_ENGINE}"
log INFO "Input Directory:     ${IN_DIR}"
log INFO "Completed Directory: ${COMPLETED_DIR}"
log INFO "Failed Directory:    ${FAILED_DIR}"
log INFO "Logs Directory:      ${LOGS_DIR}"
log INFO "Dry Run Mode:        ${DRY_RUN}"
log INFO "Drain Backlog:       ${DRAIN_BACKLOG}"
echo "================================================================================"

acquire_lock
clean_mineru_in

if [[ "$CLEAN_IN_PROGRESS" == "true" ]]; then
  clean_in_progress_and_cache
  if [[ -z "$CUSTOM_PDF" ]]; then
    exit 0
  fi
fi

# Handle backlog first if requested
if [[ "$DRAIN_BACKLOG" == "true" ]]; then
  drain_backlog
fi

# Determine candidate PDFs
queue=()
if [[ -n "$CUSTOM_PDF" ]]; then
  if [[ -f "$CUSTOM_PDF" ]]; then
    queue+=("$CUSTOM_PDF")
  else
    log ERROR "Specified PDF file not found: $CUSTOM_PDF"
    exit 1
  fi
else
  shopt -s nullglob
  raw_files=("$IN_DIR"/*.pdf "$IN_DIR"/*.PDF)
  shopt -u nullglob

  for f in "${raw_files[@]}"; do
    local_name="$(basename "$f")"
    # Skip dotfiles and AppleDouble files
    if [[ "$local_name" == ._* || "$local_name" == .* ]]; then
      continue
    fi
    queue+=("$f")
  done
fi

if [[ ${#queue[@]} -eq 0 ]]; then
  log INFO "No new PDF files found in ${IN_DIR}/. Pipeline queue is empty."
  log SUCCESS "All done. Exiting cleanly."
  exit 0
fi

log INFO "Found ${#queue[@]} PDF file(s) in queue to process."
echo

if [[ -n "${STATUS_DISPLAY_EVENT_FILE:-}" ]]; then
  "${BASE_DIR}/.venv/bin/python3" -c 'import json, os, sys; open(os.environ["STATUS_DISPLAY_EVENT_FILE"], "a").write(json.dumps({"event": "start_run", "queue": sys.argv[1:]}) + "\n")' "${queue[@]}" 2>/dev/null || true
fi

total_queued=${#queue[@]}
success_count=0
fail_count=0
pipeline_start_time=$(date +%s)

for pdf in "${queue[@]}"; do
  if process_single_pdf "$pdf"; then
    ((success_count++))
  else
    ((fail_count++))
    bname_fail="$(basename "${pdf%.*}")"
    emit_status_event "{\"event\":\"end_book\",\"summary\":{\"name\":\"${bname_fail}\",\"status\":\"failed\"}}"
  fi
done

pipeline_end_time=$(date +%s)
total_pipeline_dur=$((pipeline_end_time - pipeline_start_time))

echo "================================================================================"
log STEP "Pipeline Batch Summary:"
log INFO "  Total Queued:    ${total_queued}"
log SUCCESS "  Successful:      ${success_count}"
if (( fail_count > 0 )); then
  log ERROR "  Failed:          ${fail_count}"
else
  log INFO "  Failed:          0"
fi
log INFO "  Total Duration:  $(format_duration "$total_pipeline_dur")"
echo "================================================================================"

if (( fail_count > 0 )); then
  exit 1
fi
exit 0
