#!/usr/bin/env bash
# ==============================================================================
# Batch Markdown Stripper using md_tts.py
# Processes all *_md folders in ../mineru/out/, strips markdown into TTS-ready text,
# and upon successful conversion moves each source folder and its _pages folder to ./completed
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate Python virtual environment
VENV_ACTIVATE="${VENV_PATH:-$SCRIPT_DIR/../.venv/bin/activate}"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
elif [[ -f "$SCRIPT_DIR/../.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/../.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.venv/bin/activate"
fi

if [[ -x "$SCRIPT_DIR/../.venv/bin/python3" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/../.venv/bin/python3"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python3" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

set -a
[[ -f .env ]] && source .env
set +a

show_help() {
  echo "Usage: $0 [options] [target_dir_md ...]"
  echo
  echo "Batch converts and strips Markdown folders (e.g. from MinerU OCR) into TTS-ready plain text."
  echo "Upon verified successful conversion, each finished source folder and its corresponding"
  echo "_pages folder are moved to completed/."
  echo
  echo "Batch Options:"
  echo "  --in-dir <dir>         Input directory containing *_md folders (default: ../mineru/out)"
  echo "  --out-dir <dir>        Output directory for stripped text files (default: out)"
  echo "  --completed-dir <dir>  Destination directory for completed source folders (default: completed)"
  echo
  echo "Options forwarded directly to md_tts.py:"
  "$PYTHON_BIN" "$SCRIPT_DIR/md_tts.py" --help
}

IN_DIR="${IN_DIR:-../mineru/out}"
OUT_DIR="${OUT_DIR:-out}"
COMPLETED_DIR="${COMPLETED_DIR:-completed}"
MATH_API_URL="${MATH_API_URL:-http://localhost:8099}"

ensure_mathspeak_container() {
  local url="${MATH_API_URL:-http://localhost:8099}"
  if curl -s --max-time 2 "${url%/}/healthz" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "$url" =~ localhost|127\.0\.0\.1|0\.0\.0\.0 ]]; then
    if command -v podman >/dev/null 2>&1; then
      if podman ps -a --format '{{.Names}}' 2>/dev/null | grep -qw "mathspeak-api"; then
        echo ">>> mathspeak-api is not running. Starting podman container mathspeak-api..."
        podman start mathspeak-api >/dev/null 2>&1 || true
        for i in {1..20}; do
          if curl -s --max-time 1 "${url%/}/healthz" >/dev/null 2>&1; then
            echo ">>> mathspeak-api container is now running and ready."
            return 0
          fi
          sleep 0.5
        done
        echo ">>> WARNING: Started mathspeak-api container but health check did not respond in time." >&2
      else
        echo ">>> WARNING: mathspeak-api container not found in podman ps -a." >&2
      fi
    fi
  fi
  return 1
}

forward_args=()
custom_dirs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --in-dir)
      IN_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --completed-dir)
      COMPLETED_DIR="$2"
      shift 2
      ;;
    --math-api)
      MATH_API_URL="$2"
      forward_args+=("$1" "$2")
      shift 2
      ;;
    *)
      if [[ -d "$1" ]]; then
        custom_dirs+=("$1")
      else
        forward_args+=("$1")
      fi
      shift
      ;;
  esac
done

IN_DIR="${IN_DIR%/}"
OUT_DIR="${OUT_DIR%/}"
COMPLETED_DIR="${COMPLETED_DIR%/}"

mkdir -p "$OUT_DIR" "$COMPLETED_DIR"

if [[ ${#custom_dirs[@]} -gt 0 ]]; then
  target_dirs=("${custom_dirs[@]}")
else
  mkdir -p "$IN_DIR"
  shopt -s nullglob
  target_dirs=("$IN_DIR"/*_md)
  shopt -u nullglob
fi

if [[ ${#target_dirs[@]} -eq 0 ]]; then
  echo "No *_md folders found in ${IN_DIR}/"
  exit 0
fi

echo "Found ${#target_dirs[@]} folder(s) to process."
echo "Input directory:     ${IN_DIR}"
echo "Output directory:    ${OUT_DIR}"
echo "Completed directory: ${COMPLETED_DIR}"
echo "MathSpeak API:       ${MATH_API_URL}"
echo

ensure_mathspeak_container

total_count=${#target_dirs[@]}
current_idx=0
success_count=0
fail_count=0

for md_dir in "${target_dirs[@]}"; do
  [ -d "$md_dir" ] || continue
  ((current_idx++))
  folder_name="$(basename "$md_dir")"

  if [[ "$folder_name" == *_md ]]; then
    name="${folder_name%_md}"
    out_txt_dir="${OUT_DIR}/${name}_txt"
  else
    name="$folder_name"
    out_txt_dir="${OUT_DIR}/${folder_name}"
  fi

  parent_dir="$(dirname "$md_dir")"
  pages_dir="${parent_dir}/${name}_pages"
  fail_file="${parent_dir}/${name}_failures.txt"

  echo "================================================================================"
  echo ">>> [${current_idx}/${total_count}] Stripping: ${folder_name}"
  echo "================================================================================"

  # Execute md_tts.py with passed flags and recursive mode
  local_args=()
  if [[ ! " ${forward_args[*]:-} " =~ " --math-api " ]]; then
    local_args+=(--math-api "$MATH_API_URL")
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/md_tts.py" "$md_dir" -o "$OUT_DIR" -r "${local_args[@]}" "${forward_args[@]}"
  exit_code=$?

  # Sanity check: Ensure text files were produced
  txt_count=0
  if [ -d "$out_txt_dir" ]; then
    txt_count=$(find "$out_txt_dir" -type f -name "*.txt" 2>/dev/null | wc -l)
  fi

  if [ $exit_code -eq 0 ] && [ "$txt_count" -gt 0 ]; then
    # If book_metadata.json was not generated yet (e.g. older backlog item), extract it now
    if [ ! -f "$md_dir/book_metadata.json" ] && [ -f "$SCRIPT_DIR/../mineru/scripts/metadata_extractor.py" ]; then
      "$PYTHON_BIN" "$SCRIPT_DIR/../mineru/scripts/metadata_extractor.py" "$md_dir" >/dev/null 2>&1 || true
    fi
    if [ -d "$out_txt_dir" ]; then
      for sidecar in toc.json book_metadata.json cover.jpg; do
        if [ -f "$md_dir/$sidecar" ]; then
          cp -f "$md_dir/$sidecar" "$out_txt_dir/$sidecar"
        fi
      done
    fi
    echo "--------------------------------------------------------------------------------"
    echo ">>> SUCCESS: ${folder_name} stripped successfully (${txt_count} text files produced)."

    # Move source folder to completed/
    dest_md="${COMPLETED_DIR}/${folder_name}"
    if [ -e "$dest_md" ]; then
      echo ">>> Note: ${dest_md} already exists in ${COMPLETED_DIR}/. Replacing it."
      rm -rf "$dest_md"
    fi
    echo ">>> Moving source: ${md_dir} -> ${COMPLETED_DIR}/"
    mv "$md_dir" "${COMPLETED_DIR}/"

    # Move corresponding _pages folder if present
    if [ -d "$pages_dir" ]; then
      dest_pages="${COMPLETED_DIR}/$(basename "$pages_dir")"
      if [ -e "$dest_pages" ]; then
        echo ">>> Note: ${dest_pages} already exists in ${COMPLETED_DIR}/. Replacing it."
        rm -rf "$dest_pages"
      fi
      echo ">>> Moving pages:  ${pages_dir} -> ${COMPLETED_DIR}/"
      mv "$pages_dir" "${COMPLETED_DIR}/"
    fi

    # Move failure log if present
    if [ -f "$fail_file" ]; then
      echo ">>> Moving log:    ${fail_file} -> ${COMPLETED_DIR}/"
      mv -f "$fail_file" "${COMPLETED_DIR}/"
    fi

    echo "--------------------------------------------------------------------------------"
    ((success_count++))
  else
    echo "--------------------------------------------------------------------------------"
    if [ $exit_code -ne 0 ]; then
      echo ">>> ERROR: ${folder_name} encountered error during md_tts.py execution (exit code: ${exit_code})." >&2
    else
      echo ">>> WARNING: ${folder_name} exited 0 but no .txt files found in ${out_txt_dir}." >&2
    fi
    echo ">>> Retaining ${md_dir} in ${parent_dir}/ for inspection/retry." >&2
    echo "${folder_name}: Stripping incomplete/failed at $(date)" >> "${OUT_DIR}/batch_failures.txt"
    echo "--------------------------------------------------------------------------------"
    ((fail_count++))
  fi
  echo
done

echo "================================================================================"
echo "Batch processing finished: ${success_count} succeeded, ${fail_count} failed."
echo "================================================================================"

if [ $fail_count -gt 0 ]; then
  exit 1
fi
exit 0
