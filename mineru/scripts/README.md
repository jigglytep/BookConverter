# MinerU Fast Dual-GPU Book Processing

High-performance, dual-GPU batch PDF conversion pipeline with persistent container workers, atomic work-stealing, and crash resilience.

---

## 1. Overview

The original single-GPU pipeline (`split_and_process.sh`) created a new Podman container for every page, incurring a **~11.3s cold-start penalty** (Python startup + PyTorch/Transformers/MinerU module imports) per page, leaving the second Tesla P100 completely idle.

`fast_split_and_process.py` solves this by:
1. **Utilizing Both Tesla P100 GPUs:** Distributes pages across `llamacpp` (GPU 0, port `30000`) and `llamacpp1` (GPU 1, port `30001`).
2. **Persistent Python Workers:** Maintains long-running containers (`process_pages_worker.py`), keeping MinerU and PyTorch loaded in memory.
3. **Atomic Work-Stealing:** Uses an atomic claim mechanism (`out/<book>_pages/.claims/`) so workers dynamically grab the next available page. Dense pages don't block subsequent pages on the other GPU.
4. **Per-Page Crash Isolation & Truncation Resilience:** Uses `try...except` around each page's inference. False-positive block truncations are gracefully kept and cleaned, and errors log to `out/<book>_failures.txt` without crashing the run or losing completed progress.
5. **Autoregressive Loop Collapser:** Built-in pattern recognition (`clean_repetitive_vlm_output`) detects and collapses infinite token repetition traps (e.g. 16,000 zeros on $10^{100}$ or repeated boolean tables) while preserving legitimate code, math, and tables.
6. **Automatic Blank-Page Handling:** Spacer and divider pages with zero text and zero images are recognized and preserved with `<!-- blank page -->` markers instead of failing.
7. **Multi-Pass KV-Cache Reset:** Automatically restarts both `llamacpp` servers between retry passes to clear slot-reuse/KV-cache states for truncated pages.

### Performance Benchmark (714-Page Book)

| Metric | Legacy `split_and_process.sh` | Optimized `fast_split_and_process.py` |
|---|---|---|
| **Active GPUs** | 1 (GPU 0 only) | **2** (GPU 0 & GPU 1 in parallel) |
| **Per-Page Import Overhead** | ~11.3s per page | **0s** (loaded once into memory) |
| **Worker Page Time** | ~26s / page | **~8–10s / page** |
| **Effective Throughput** | ~135 pages / hour | **~450–600 pages / hour** |
| **700-Page Total Runtime** | **~5.1 hours** | **~1 hour 15 minutes** (~4x speedup) |

---

## 2. Conversion Examples & Command Usage

Always run commands from the project root (`/path/to/BookConverters/mineru`).

### 2.1 Fast Batch Conversion (Production)

`fast_split_and_process.py` defaults to `--method auto` and `--backend pipeline`. It inspects the PDF at startup:
- **Born-Digital Books (with text layer)**: Automatically selects `txt` mode, extracting 100% exact digital font glyphs with zero OCR errors and AI-determined layout.
- **Scanned Books (images only)**: Automatically selects `ocr` mode, using high-accuracy OCR.

```bash
# Convert any book (auto-detects digital vs scanned):
./scripts/fast_split_and_process.py in/carl_rpg_core_rulebook_digital_hi-res.pdf

# Or using the bash wrapper:
./scripts/fast_split_and_process.sh in/carl_rpg_core_rulebook_digital_hi-res.pdf
```

#### Useful Flags & Options
```bash
# Explicitly force Digital Text Mode (YOLOv8 layout + non-OCR digital font glyphs)
./scripts/fast_split_and_process.py in/book.pdf --backend pipeline --method txt

# Explicitly force OCR Mode (e.g. if a scanned PDF has corrupt/invisible text layers)
./scripts/fast_split_and_process.py in/book.pdf --backend pipeline --method ocr

# Native Hybrid Mode (VLM layout + digital text extraction)
./scripts/fast_split_and_process.py in/book.pdf --backend hybrid-http-client --method txt

# Pure Dual-GPU VLM Mode (full multimodal vision OCR via llama.cpp on P100 GPUs)
./scripts/fast_split_and_process.py in/book.pdf --backend vlm-http-client

# Set custom number of workers (default is 2)
./scripts/fast_split_and_process.py in/book.pdf --workers 2

# Single-worker run
./scripts/fast_split_and_process.py in/book.pdf --workers 1

# Increase retry passes (default is 3)
./scripts/fast_split_and_process.py in/book.pdf --max-retries 5

# Set language (default is "ch", which supports both English and Chinese)
./scripts/fast_split_and_process.py in/book.pdf --lang en

# Enable verbose logging and mirror raw logs to a file
./scripts/fast_split_and_process.py in/book.pdf --verbose --log-file out/conversion.log
```

#### Resuming an Interrupted Run
The script is **100% resumable**. If stopped with `Ctrl+C` or interrupted by a reboot, re-run the exact same command:
```bash
./scripts/fast_split_and_process.py in/Python_for_Finance_Mastering.pdf
```
It scans `out/<book>_md/`, counts existing markdown files, and processes only the missing pages without redoing completed work.

---

### 2.2 Pure Digital CV Pipeline (100% Exact Code & Character Fidelity)

For born-digital books with dense Python code, mathematical formulas, and numerical tables where you want **zero OCR error and zero hallucination**:
* **Visual Layout:** Uses YOLOv8 computer vision to detect headers, footers, code fences, and figures.
* **Character Extraction:** Extracts digital glyphs directly from the PDF text layer using `pdfium`.

```bash
podman run --rm \
  -v /path/to/BookConverters/mineru/in:/data/in:ro,z \
  -v /path/to/BookConverters/mineru/out:/data/out:z \
  -v mineru_mineru-models:/models:z \
  -e MINERU_TOOLS_CONFIG_JSON=/models/mineru.json \
  localhost/mineru-cpu:latest \
  mineru -p /data/in/Python_for_Finance_Mastering.pdf -o /data/out -b pipeline
```

---

### 2.3 Native Hybrid Mode (Visual Layout + Digital PDF Text Insertion)

Combines the VLM server's visual layout understanding with lossless digital text extraction from the PDF:

```bash
podman run --rm \
  --network=mineru_mineru-net \
  -v /path/to/BookConverters/mineru/in:/data/in:ro,z \
  -v /path/to/BookConverters/mineru/out:/data/out:z \
  -v mineru_mineru-models:/models:z \
  -e MINERU_TOOLS_CONFIG_JSON=/models/mineru.json \
  localhost/mineru-cpu:latest \
  mineru -p /data/in/Python_for_Finance_Mastering.pdf -o /data/out -b hybrid-http-client --server-url http://llamacpp:30000
```

---

### 2.4 Converting a Specific Page Range Only

To test layout parsing on a specific page range (e.g. pages 10 to 25):

```bash
podman run --rm \
  -v /path/to/BookConverters/mineru/in:/data/in:ro,z \
  -v /path/to/BookConverters/mineru/out:/data/out:z \
  -v mineru_mineru-models:/models:z \
  -e MINERU_TOOLS_CONFIG_JSON=/models/mineru.json \
  localhost/mineru-cpu:latest \
  mineru -p /data/in/Python_for_Finance_Mastering.pdf -o /data/out -b pipeline -s 9 -e 24
```
*(Note: `-s` and `-e` use 0-indexed page numbers).*

---

### 2.5 Batch Converting All PDFs in `in/` Directory

To sequentially convert all PDF files in your `in/` folder using the Dual-GPU pipeline:

```bash
for pdf in in/*.pdf; do
  echo ">>> Processing $pdf..."
  ./scripts/fast_split_and_process.py "$pdf"
done
```

---

## 3. Architecture & File Layout

```
mineru/
├── compose.yaml                      # Defines llamacpp (GPU 0) and llamacpp1 (GPU 1)
├── in/                               # Input PDFs
├── out/
│   ├── <book>_md/                    # Completed individual markdown pages
│   │   ├── <book>_page_0001.md
│   │   ├── ...
│   │   └── images/                   # Extracted figures and diagrams
│   ├── <book>_pages/                 # Detailed intermediate JSON & layout data
│   │   └── .claims/                  # Dynamic lock files for atomic work-stealing
│   └── <book>_failures.txt           # Failure log for pages requiring retries
└── scripts/
    ├── fast_split_and_process.py     # Host orchestrator: server health, worker lifecycle & retries
    ├── fast_split_and_process.sh     # Bash convenience wrapper
    ├── process_pages_worker.py       # Container-side worker: persistent loop & claim logic
    ├── split_and_process.sh          # Legacy single-container per-page runner (backup)
    └── process_one_page.py           # Legacy one-shot single page script
```

---

## 4. Monitoring & Diagnostics

### Check GPU Utilization
Run in another terminal while a conversion is running:

```bash
watch -n 1 nvidia-smi
```

Both Tesla P100 cards should show ~80–99% GPU compute utilization and ~2.1 GB VRAM usage.

### Check VLM Server Health

```bash
# GPU 0 server (port 30000)
curl -fsS http://localhost:30000/health

# GPU 1 server (port 30001)
curl -fsS http://localhost:30001/health
```

### Manually Restart VLM Servers

If you need to reset the llama.cpp server states manually:

```bash
podman restart llamacpp llamacpp1
```

### Count Completed Pages

```bash
ls -1 out/<book>_md/*.md | wc -l
```
