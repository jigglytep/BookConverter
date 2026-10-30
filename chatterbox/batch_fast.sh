#!/usr/bin/env bash
# ==============================================================================
# Batch Audiobook & EPUB Pipeline using Chatterbox TTS, mineru_to_epub, and ABS
#
# Processes text page folders in ../TTS_Script/out/
# 1. Synthesizes audiobooks (.m4b) via audiobook_tts.py
# 2. Moves completed source folder to completed/
# 3. Generates companion EPUB3 ebook (.epub) via mineru_to_epub.py
# 4. Uploads .m4b and .epub together to AudioBookshelf via upload_to_abs.py
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate Python virtual environment
VENV_ACTIVATE="${VENV_PATH:-$SCRIPT_DIR/../.venv/bin/activate}"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
elif [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/../.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/../.venv/bin/activate"
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/python3" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
elif [[ -x "$SCRIPT_DIR/../.venv/bin/python3" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

set -a
[[ -f "$SCRIPT_DIR/../.env" ]] && source "$SCRIPT_DIR/../.env"
[[ -f "$SCRIPT_DIR/.env" ]] && source "$SCRIPT_DIR/.env"
set +a

show_help() {
  echo "Usage: $0 [options] [target_dir ...]"
  echo
  echo "Batch converts text page folders (from TTS_Script) into audiobooks via audiobook_tts.py,"
  echo "moves completed source folders to completed/, generates companion EPUB ebooks via mineru_to_epub.py,"
  echo "and uploads both .m4b and companion .epub files to AudioBookshelf via upload_to_abs.py."
  echo
  echo "Batch Pipeline Options:"
  echo "  --in-dir <dir>          Input directory containing text page folders"
  echo "                          (default: ../TTS_Script/out)"
  echo "  --out-dir <dir>         Output directory for audiobooks and epubs (default: audiobooks)"
  echo "  --completed-dir <dir>   Destination directory for completed source folders (default: completed)"
  echo "  --format <fmt>          Audio format: m4b, mp3, wav, m4a, flac (default: m4b)"
  echo "  --prefer-markdown       Prefer rich MinerU markdown with images/formulas for EPUB (default: true)"
  echo "  --text-epub             Force generating EPUB strictly from stripped text in completed/"
  echo "  --skip-epub             Skip EPUB generation step"
  echo "  --skip-pdf              Skip uploading companion PDF to AudioBookshelf"
  echo "  --skip-upload           Skip AudioBookshelf upload step"
  echo "  --scan                  Trigger library scan in AudioBookshelf after upload (default: true)"
  echo "  --no-scan               Do not trigger library scan in AudioBookshelf after upload"
  echo
  echo "AudioBookshelf Options (defaults configured in .env):"
  echo "  --server <url>          AudioBookshelf server URL (default: http://localhost:3378)"
  echo "  --token <token>         AudioBookshelf API token"
  echo "  --username <user>       AudioBookshelf username"
  echo "  --password <pass>       AudioBookshelf password"
  echo "  --library <name>        Target AudioBookshelf library name (default: TTS)"
  echo "  --folder <id>           Target AudioBookshelf folder ID"
  echo "  --title <title>         Override book title"
  echo "  --author <name>         Book author"
  echo "  --series <name>         Book series"
  echo
  echo "Chatterbox TTS Options (forwarded directly to audiobook_tts.py):"
  echo "  --model {standard,turbo}"
  echo "  --devices <devices>     Comma-separated device list (e.g. 'cuda:0,cuda:1')"
  echo "  --reference-id <id>     Preset reference voice ID"
  echo "  --reference-audio <path>Path to reference audio for voice cloning"
  echo "  --temperature <val>     Sampling temperature"
  echo "  --clean-chunks          Delete intermediate chunk WAV files after assembly"
  echo "  --no-cache              Bypass chunk audio cache"
  echo "  --per-page              Export individual audio files per page"
  echo
  "$PYTHON_BIN" "$SCRIPT_DIR/audiobook_tts.py" --help
}

# Defaults
IN_DIR="${IN_DIR:-$SCRIPT_DIR/../TTS_Script/out}"
IN_DIR="${IN_DIR/#\~/$HOME}"
OUT_DIR="${OUT_DIR:-audiobooks}"
OUT_DIR="${OUT_DIR/#\~/$HOME}"
COMPLETED_DIR="${COMPLETED_DIR:-completed}"
COMPLETED_DIR="${COMPLETED_DIR/#\~/$HOME}"

AUDIO_FORMAT="${AUDIO_FORMAT:-m4b}"
PREFER_MARKDOWN="${PREFER_MARKDOWN:-true}"
SKIP_EPUB=false
SKIP_PDF=false
SKIP_UPLOAD=false
FORCE=false

ABS_SERVER="${ABS_SERVER:-http://localhost:3378}"
ABS_TOKEN="${ABS_TOKEN:-}"
ABS_USERNAME="${ABS_USERNAME:-}"
ABS_PASSWORD="${ABS_PASSWORD:-}"
ABS_LIBRARY="${ABS_LIBRARY:-TTS}"
ABS_FOLDER="${ABS_FOLDER:-}"
ABS_SCAN="${ABS_SCAN:-true}"

BOOK_TITLE=""
BOOK_AUTHOR="${BOOK_AUTHOR:-}"
BOOK_SERIES="${BOOK_SERIES:-}"

tts_args=()
custom_dirs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --in-dir)
      IN_DIR="$2"
      IN_DIR="${IN_DIR/#\~/$HOME}"
      shift 2
      ;;
    --in-dir=*)
      IN_DIR="${1#*=}"
      IN_DIR="${IN_DIR/#\~/$HOME}"
      shift
      ;;
    --out-dir)
      OUT_DIR="$2"
      OUT_DIR="${OUT_DIR/#\~/$HOME}"
      shift 2
      ;;
    --out-dir=*)
      OUT_DIR="${1#*=}"
      OUT_DIR="${OUT_DIR/#\~/$HOME}"
      shift
      ;;
    --completed-dir)
      COMPLETED_DIR="$2"
      COMPLETED_DIR="${COMPLETED_DIR/#\~/$HOME}"
      shift 2
      ;;
    --completed-dir=*)
      COMPLETED_DIR="${1#*=}"
      COMPLETED_DIR="${COMPLETED_DIR/#\~/$HOME}"
      shift
      ;;
    --format)
      AUDIO_FORMAT="$2"
      shift 2
      ;;
    --format=*)
      AUDIO_FORMAT="${1#*=}"
      shift
      ;;
    --prefer-markdown)
      PREFER_MARKDOWN=true
      shift
      ;;
    --text-epub)
      PREFER_MARKDOWN=false
      shift
      ;;
    --skip-epub|--no-epub)
      SKIP_EPUB=true
      shift
      ;;
    --skip-pdf|--no-pdf)
      SKIP_PDF=true
      shift
      ;;
    --skip-upload|--no-upload)
      SKIP_UPLOAD=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --scan)
      ABS_SCAN=true
      shift
      ;;
    --no-scan)
      ABS_SCAN=false
      shift
      ;;
    --server)
      ABS_SERVER="$2"
      shift 2
      ;;
    --server=*)
      ABS_SERVER="${1#*=}"
      shift
      ;;
    --token)
      ABS_TOKEN="$2"
      shift 2
      ;;
    --token=*)
      ABS_TOKEN="${1#*=}"
      shift
      ;;
    --username)
      ABS_USERNAME="$2"
      shift 2
      ;;
    --username=*)
      ABS_USERNAME="${1#*=}"
      shift
      ;;
    --password)
      ABS_PASSWORD="$2"
      shift 2
      ;;
    --password=*)
      ABS_PASSWORD="${1#*=}"
      shift
      ;;
    --library)
      ABS_LIBRARY="$2"
      shift 2
      ;;
    --library=*)
      ABS_LIBRARY="${1#*=}"
      shift
      ;;
    --folder)
      ABS_FOLDER="$2"
      shift 2
      ;;
    --folder=*)
      ABS_FOLDER="${1#*=}"
      shift
      ;;
    --title)
      BOOK_TITLE="$2"
      shift 2
      ;;
    --title=*)
      BOOK_TITLE="${1#*=}"
      shift
      ;;
    --author)
      BOOK_AUTHOR="$2"
      shift 2
      ;;
    --author=*)
      BOOK_AUTHOR="${1#*=}"
      shift
      ;;
    --series)
      BOOK_SERIES="$2"
      shift 2
      ;;
    --series=*)
      BOOK_SERIES="${1#*=}"
      shift
      ;;
    --model)
      tts_args+=("--model" "$2")
      shift 2
      ;;
    --model=*)
      tts_args+=("$1")
      shift
      ;;
    --devices)
      tts_args+=("--devices" "$2")
      shift 2
      ;;
    --devices=*)
      tts_args+=("$1")
      shift
      ;;
    --reference-id)
      tts_args+=("--reference-id" "$2")
      shift 2
      ;;
    --reference-id=*)
      tts_args+=("$1")
      shift
      ;;
    --reference-audio)
      tts_args+=("--reference-audio" "$2")
      shift 2
      ;;
    --reference-audio=*)
      tts_args+=("$1")
      shift
      ;;
    --clean-chunks)
      tts_args+=("--clean-chunks")
      shift
      ;;
    --no-cache)
      tts_args+=("--no-cache")
      shift
      ;;
    --per-page)
      tts_args+=("--per-page")
      shift
      ;;
    --temperature|--top-p|--repetition-penalty|--cfg-weight|--exaggeration)
      tts_args+=("$1" "$2")
      shift 2
      ;;
    --temperature=*|--top-p=*|--repetition-penalty=*|--cfg-weight=*|--exaggeration=*)
      tts_args+=("$1")
      shift
      ;;
    *)
      if [[ -d "$1" ]]; then
        custom_dirs+=("$1")
      else
        tts_args+=("$1")
      fi
      shift
      ;;
  esac
done

IN_DIR="${IN_DIR%/}"
mkdir -p "$OUT_DIR" "$COMPLETED_DIR" "$SCRIPT_DIR/logs"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
COMPLETED_DIR="$(cd "$COMPLETED_DIR" && pwd)"

if [[ ${#custom_dirs[@]} -gt 0 ]]; then
  target_dirs=("${custom_dirs[@]}")
else
  mkdir -p "$IN_DIR"
  shopt -s nullglob
  target_dirs=()
  for d in "$IN_DIR"/*; do
    if [[ -d "$d" ]]; then
      target_dirs+=("$d")
    fi
  done
  shopt -u nullglob
fi

if [[ ${#target_dirs[@]} -eq 0 ]]; then
  echo "No book folders found in ${IN_DIR}/"
  exit 0
fi

echo "================================================================================"
echo " Chatterbox TTS & EPUB Batch Pipeline"
echo "================================================================================"
echo "Found ${#target_dirs[@]} folder(s) to process."
echo "Input directory:     ${IN_DIR}"
echo "Output directory:    ${OUT_DIR}"
echo "Completed directory: ${COMPLETED_DIR}"
echo "Audio format:        ${AUDIO_FORMAT}"
echo "Prefer markdown:     ${PREFER_MARKDOWN}"
echo "Skip EPUB:           ${SKIP_EPUB}"
echo "Skip Upload:         ${SKIP_UPLOAD}"
echo "================================================================================"
echo

# Helper function to find best source directory for EPUB generation
find_epub_source() {
  local bname="$1"
  local moved_dir="$2"

  if [[ "$PREFER_MARKDOWN" == "true" ]]; then
    local candidates=(
      "${SCRIPT_DIR}/../TTS_Script/completed/${bname}_md"
      "${SCRIPT_DIR}/../mineru/out/${bname}_md"
      "${SCRIPT_DIR}/../mineru/completed/${bname}_md"
    )
    for cand in "${candidates[@]}"; do
      if [[ -d "$cand" ]]; then
        echo "$cand"
        return 0
      fi
    done
  fi

  # Fall back to moved source folder in completed/
  echo "$moved_dir"
}

total_count=${#target_dirs[@]}
current_idx=0
success_count=0
fail_count=0

for target_dir in "${target_dirs[@]}"; do
  [ -d "$target_dir" ] || continue
  ((current_idx++))
  folder_name="$(basename "$target_dir")"

  # Determine canonical book name
  if [[ "$folder_name" == *_txt ]]; then
    book_name="${folder_name%_txt}"
  elif [[ "$folder_name" == *_md ]]; then
    book_name="${folder_name%_md}"
  else
    book_name="$folder_name"
  fi

  item_title="${BOOK_TITLE:-$book_name}"
  audio_out="${OUT_DIR}/${book_name}.${AUDIO_FORMAT}"
  epub_out="${OUT_DIR}/${book_name}.epub"

  echo "================================================================================"
  echo ">>> [${current_idx}/${total_count}] Processing: ${folder_name}"
  echo "================================================================================"
  echo "Book Name:      ${book_name}"
  echo "Audiobook Out:  ${audio_out}"
  echo "EPUB Out:       ${epub_out}"
  echo

  # Skip if already converted unless --force is passed
  if [[ "$FORCE" != "true" && -s "$audio_out" && ( "$SKIP_EPUB" == "true" || -s "$epub_out" ) ]]; then
    echo ">>> Book '${book_name}' is already converted (${audio_out} exists)."
    dest_src="${COMPLETED_DIR}/${folder_name}"
    if [[ "$target_dir" != "$dest_src" ]]; then
      rm -rf "$dest_src"
      echo ">>> Archiving source: ${target_dir} -> ${COMPLETED_DIR}/"
      mv "$target_dir" "${COMPLETED_DIR}/"
    fi
    ((success_count++))
    echo
    continue
  fi

  # Sanity check: Ensure directory contains text or markdown files
  file_count=$(find "$target_dir" -maxdepth 2 -type f \( -name "*.txt" -o -name "*.md" \) 2>/dev/null | wc -l)
  if [ "$file_count" -eq 0 ]; then
    echo "--------------------------------------------------------------------------------"
    echo ">>> WARNING: ${folder_name} contains no .txt or .md files. Skipping." >&2
    echo "${folder_name}: Skipped (no text files found) at $(date)" >> "${OUT_DIR}/batch_failures.txt"
    echo "--------------------------------------------------------------------------------"
    ((fail_count++))
    continue
  fi

  # ----------------------------------------------------------------------------
  # Step 1: Synthesize Audiobook & Synchronized EPUB3 via audiobook_tts.py
  # ----------------------------------------------------------------------------
  echo "--- [1/3] Synthesizing Audiobook & Synchronized EPUB3 with Chatterbox TTS ---"
  epub_pre_source=$(find_epub_source "$book_name" "$target_dir")
  step1_args=("$target_dir" -o "$audio_out" --format "$AUDIO_FORMAT")
  if [[ -n "$BOOK_TITLE" ]]; then
    step1_args+=(--title "$BOOK_TITLE")
  fi
  if [[ -n "$BOOK_AUTHOR" ]]; then
    step1_args+=(--author "$BOOK_AUTHOR")
  fi
  if [[ "$SKIP_EPUB" == "true" ]]; then
    step1_args+=(--skip-epub)
  else
    step1_args+=(--epub-output "$epub_out" --markdown-dir "$epub_pre_source")
  fi
  step1_args+=("${tts_args[@]}")

  "$PYTHON_BIN" "$SCRIPT_DIR/audiobook_tts.py" "${step1_args[@]}"
  tts_exit=$?

  if [ $tts_exit -ne 0 ] || [ ! -s "$audio_out" ]; then
    echo "--------------------------------------------------------------------------------"
    echo ">>> ERROR: TTS synthesis failed for ${folder_name} (exit code: ${tts_exit})." >&2
    echo ">>> Retaining source in ${target_dir} for inspection/retry." >&2
    echo "${folder_name}: TTS synthesis failed at $(date)" >> "${OUT_DIR}/batch_failures.txt"
    echo "${folder_name}: TTS synthesis failed at $(date)" >> "${SCRIPT_DIR}/logs/batch_failures.txt"
    echo "--------------------------------------------------------------------------------"
    ((fail_count++))
    echo
    continue
  fi

  echo ">>> Audiobook synthesized successfully: ${audio_out}"

  # ----------------------------------------------------------------------------
  # Step 2: Move finished source folder to completed/
  # ----------------------------------------------------------------------------
  dest_src="${COMPLETED_DIR}/${folder_name}"
  if [ -e "$dest_src" ]; then
    echo ">>> Note: ${dest_src} already exists in ${COMPLETED_DIR}/. Replacing it."
    rm -rf "$dest_src"
  fi
  echo ">>> Moving source: ${target_dir} -> ${COMPLETED_DIR}/"
  mv "$target_dir" "${COMPLETED_DIR}/"

  # Clean any residual failure log in parent directory if it succeeded now
  parent_dir="$(dirname "$target_dir")"
  fail_file="${parent_dir}/${book_name}_failures.txt"
  if [ -f "$fail_file" ]; then
    rm -f "$fail_file"
  fi

  # ----------------------------------------------------------------------------
  # Step 3: Verify / Fallback Companion EPUB3 Ebook via mineru_to_epub.py
  # ----------------------------------------------------------------------------
  if [[ "$SKIP_EPUB" != "true" ]]; then
    echo
    echo "--- [2/3] Verifying Companion EPUB3 with SMIL Media Overlays ---"
    if [ -s "$epub_out" ]; then
      echo ">>> SUCCESS: Synchronized EPUB3 with Media Overlays ready: ${epub_out}"
    else
      epub_source=$(find_epub_source "$book_name" "$dest_src")
      if [[ "$epub_source" != "$dest_src" ]]; then
        echo ">>> Found rich MinerU Markdown folder with images: ${epub_source}"
      else
        echo ">>> Using converted text source folder: ${epub_source}"
      fi

      epub_cmd=("$PYTHON_BIN" "$SCRIPT_DIR/mineru_to_epub.py" "$epub_source" -o "$epub_out" --chunks-dir "$SCRIPT_DIR/chunks/$folder_name")
      if [[ -n "$BOOK_TITLE" ]]; then
        epub_cmd+=("--title" "$BOOK_TITLE")
      fi
      if [[ -n "$BOOK_AUTHOR" ]]; then
        epub_cmd+=("--author" "$BOOK_AUTHOR")
      fi

      "${epub_cmd[@]}"
      epub_exit=$?

      if [ $epub_exit -ne 0 ] || [ ! -s "$epub_out" ]; then
        echo ">>> WARNING: EPUB generation encountered errors (exit code: ${epub_exit})." >&2
        echo "${folder_name}: EPUB generation warning at $(date)" >> "${OUT_DIR}/batch_failures.txt"
      else
        echo ">>> SUCCESS: EPUB generated successfully: ${epub_out}"
      fi
    fi
  else
    echo
    echo "--- [2/3] Skipping EPUB Generation (--skip-epub specified) ---"
  fi

  # ----------------------------------------------------------------------------
  # Step 4: Upload Audiobook, EPUB, PDF, metadata.json & cover.jpg via upload_to_abs.py
  # ----------------------------------------------------------------------------
  if [[ "$SKIP_UPLOAD" != "true" ]]; then
    echo
    echo "--- [3/3] Uploading to AudioBookshelf Server ---"
    abs_cmd=("$PYTHON_BIN" "$SCRIPT_DIR/upload_to_abs.py" "$audio_out")
    if [[ -n "$BOOK_TITLE" ]]; then
      abs_cmd+=("--title" "$BOOK_TITLE")
    fi

    if [ -s "$epub_out" ]; then
      abs_cmd+=("--epub" "$epub_out")
    else
      abs_cmd+=("--no-epub")
    fi

    if [[ "$SKIP_PDF" == "true" ]]; then
      abs_cmd+=("--no-pdf")
    fi

    if [[ -n "$ABS_SERVER" ]]; then abs_cmd+=("--server" "$ABS_SERVER"); fi
    if [[ -n "$ABS_TOKEN" ]]; then abs_cmd+=("--token" "$ABS_TOKEN"); fi
    if [[ -n "$ABS_USERNAME" ]]; then abs_cmd+=("--username" "$ABS_USERNAME"); fi
    if [[ -n "$ABS_PASSWORD" ]]; then abs_cmd+=("--password" "$ABS_PASSWORD"); fi
    if [[ -n "$ABS_LIBRARY" ]]; then abs_cmd+=("--library" "$ABS_LIBRARY"); fi
    if [[ -n "$ABS_FOLDER" ]]; then abs_cmd+=("--folder" "$ABS_FOLDER"); fi
    if [[ -n "$BOOK_AUTHOR" ]]; then abs_cmd+=("--author" "$BOOK_AUTHOR"); fi
    if [[ -n "$BOOK_SERIES" ]]; then abs_cmd+=("--series" "$BOOK_SERIES"); fi
    if [[ "$ABS_SCAN" == "true" ]]; then abs_cmd+=("--scan"); fi

    "${abs_cmd[@]}"
    upload_exit=$?

    if [ $upload_exit -ne 0 ]; then
      echo ">>> WARNING: AudioBookshelf upload failed (exit code: ${upload_exit})." >&2
      echo "${folder_name}: ABS upload failed at $(date)" >> "${OUT_DIR}/batch_failures.txt"
    else
      echo ">>> SUCCESS: Uploaded successfully to AudioBookshelf."
    fi
  else
    echo
    echo "--- [3/3] Skipping ABS Upload (--skip-upload specified) ---"
  fi

  echo "--------------------------------------------------------------------------------"
  echo ">>> COMPLETED: ${folder_name} successfully processed."
  echo "--------------------------------------------------------------------------------"
  ((success_count++))
  echo
done

echo "================================================================================"
echo "Batch processing finished: ${success_count} succeeded, ${fail_count} failed."
echo "================================================================================"

if [ $fail_count -gt 0 ]; then
  exit 1
fi
exit 0
