#!/usr/bin/env bash
# ==============================================================================
# Batch PDF processor using fast_split_and_process.py
# Processes all PDFs in in/, and upon successful completion moves each PDF to completed/
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate Python virtual environment
VENV_PATH="${VENV_PATH:-${SCRIPT_DIR}/../.venv/bin/activate}"
if [[ -f "$VENV_PATH" ]]; then
  set +u
  source "$VENV_PATH"
  set -u
elif [[ -f "${SCRIPT_DIR}/../.venv/bin/activate" ]]; then
  set +u
  source "${SCRIPT_DIR}/../.venv/bin/activate"
  set -u
elif [[ -f "${SCRIPT_DIR}/.venv/bin/activate" ]]; then
  set +u
  source "${SCRIPT_DIR}/.venv/bin/activate"
  set -u
fi

# Intercept help flag so we display usage and forward help without moving any files
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "Usage: $0 [options]"
    echo
    echo "Batch converts all PDF files in in/ using fast_split_and_process.py."
    echo "Upon verified successful conversion, each finished PDF is moved to completed/."
    echo
    echo "Options are forwarded directly to fast_split_and_process.py:"
    echo "  --backend {pipeline,vlm-http-client,hybrid-http-client}"
    echo "  --method  {auto,txt,ocr}"
    echo "  --workers <count>"
    echo "  --max-retries <count>"
    echo
    python3 scripts/fast_split_and_process.py --help
    exit 0
  fi
done

set -a
[[ -f .env ]] && source .env
set +a

IN_DIR="${IN_DIR:-in}"
OUT_DIR="${OUT_DIR:-out}"
COMPLETED_DIR="completed"

mkdir -p "$IN_DIR" "$OUT_DIR" "$COMPLETED_DIR"

shopt -s nullglob
pdf_files=("$IN_DIR"/*.pdf "$IN_DIR"/*.PDF)
shopt -u nullglob

if [ ${#pdf_files[@]} -eq 0 ]; then
  echo "No PDF files found in ${IN_DIR}/"
  exit 0
fi

echo "Found ${#pdf_files[@]} PDF(s) in ${IN_DIR}/ to process."
echo

total_count=${#pdf_files[@]}
current_idx=0
success_count=0
fail_count=0

for pdf in "${pdf_files[@]}"; do
  [ -f "$pdf" ] || continue
  ((current_idx++))
  pdf_filename="$(basename "$pdf")"
  name="${pdf_filename%.*}"
  out_md_dir="${OUT_DIR}/${name}_md"
  
  echo "================================================================================"
  echo ">>> [${current_idx}/${total_count}] Processing: ${pdf_filename}"
  echo "================================================================================"

  # Handle 0-byte or empty corrupt files cleanly
  if [ ! -s "$pdf" ]; then
    echo "--------------------------------------------------------------------------------"
    echo ">>> WARNING: ${pdf_filename} is an empty file (0 bytes). Skipping." >&2
    echo "${pdf_filename}: Skipped (0 bytes) at $(date)" >> "${OUT_DIR}/batch_failures.txt"
    echo "--------------------------------------------------------------------------------"
    ((fail_count++))
    continue
  fi

  # Execute fast_split_and_process.py with passed flags (e.g. --workers, --backend, --method)
  if python3 scripts/fast_split_and_process.py "$pdf" "$@"; then
    # Sanity check: Ensure markdown files were actually produced before moving the PDF
    if [ -d "$out_md_dir" ] && find "$out_md_dir" -maxdepth 1 -name "*.md" -print -quit | grep -q .; then
      echo "--------------------------------------------------------------------------------"
      echo ">>> SUCCESS: ${pdf_filename} completed successfully."
      echo ">>> Moving ${pdf} -> ${COMPLETED_DIR}/"
      mv "$pdf" "${COMPLETED_DIR}/"
      echo "--------------------------------------------------------------------------------"
      ((success_count++))
    else
      echo "--------------------------------------------------------------------------------"
      echo ">>> WARNING: ${pdf_filename} exited 0 but no markdown files found in ${out_md_dir}." >&2
      echo ">>> Retaining ${pdf} in ${IN_DIR}/." >&2
      echo "--------------------------------------------------------------------------------"
      ((fail_count++))
    fi
  else
    echo "--------------------------------------------------------------------------------"
    echo ">>> ERROR: ${pdf_filename} encountered errors or incomplete pages." >&2
    echo ">>> Leaving ${pdf} in ${IN_DIR}/ for inspection/retry." >&2
    echo "${pdf_filename}: Conversion incomplete/failed at $(date)" >> "${OUT_DIR}/batch_failures.txt"
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
