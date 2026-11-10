# Qwen3-TTS & Chatterbox TTS Audiobook & Document Toolkit

A high-fidelity audiobook creation and document processing pipeline powered by **Qwen3-TTS** (default, running in `/mnt/raid0/qwen/src/tts-env`) with a built-in toggle to switch back to **Chatterbox TTS** (Resemble AI). Converts documents and OCR outputs (such as **MinerU / magic-pdf**) into **EPUB3 ebooks with MathML**, synthesizes natural audiobooks with **zero-shot voice cloning (`practical_engineering` reference voice) and dual-GPU FP16 parallel acceleration on NVIDIA Tesla P100s (`sm_60`)**, and uploads both to **Audiobookshelf**.

```
books/*.pdf  ──mineru/magic-pdf──>  Markdown + LaTeX + Images
                                       ├──>  mineru_to_epub.py  ──>  Book.epub (MathML + Images) ────────┐
                                       └──>  audiobook_tts.py   ──>  Book.m4b (Qwen3-TTS / Chatterbox) ──┴──>  upload_to_abs.py  ──>  Audiobookshelf
```

---

## Table of Contents

1. [Chatterbox Audiobook TTS (`audiobook_tts.py`)](#chatterbox-audiobook-tts)
   - [Overview & Capabilities](#overview--capabilities)
   - [CLI Quickstart](#cli-quickstart)
   - [Voice Cloning & References](#voice-cloning--references)
   - [Multi-GPU Acceleration](#multi-gpu-acceleration)
   - [Configuration Defaults (`config.py`)](#configuration-defaults-configpy)
2. [Automated Batch Pipeline (`batch_fast.sh`)](#automated-batch-pipeline-batch_fastsh)
3. [MinerU to EPUB3 Converter (`mineru_to_epub.py`)](#mineru-to-epub3-converter)
4. [Audiobookshelf Uploader (`upload_to_abs.py`)](#audiobookshelf-uploader)
5. [GPU Inspection (`check_gpus.py`)](#gpu-inspection)
6. [Installation & Setup](#installation--setup)

---

## Chatterbox Audiobook TTS

[`audiobook_tts.py`](audiobook_tts.py) synthesizes texts into professional-grade audiobooks (`.m4b`, `.mp3`, `.wav`) using **Chatterbox TTS**, featuring an autoregressive T3 Transformer and S3Gen neural vocoder operating at 24 kHz.

### Overview & Capabilities

- **Zero-Shot Voice Cloning**: Clone any speaker from a reference audio file (`.wav`).
- **Dual-GPU Parallel Synthesis**: Spawns concurrent worker processes across available GPUs (e.g. `cuda:0` and `cuda:1`) to double synthesis throughput.
- **Embedded Chapter Markers**: Embeds native chapter markers and titles directly into `.m4b` containers via FFmpeg (one chapter per source page).
- **Sentence-Aware Dynamic Chunking**: Splits paragraphs cleanly at sentence boundaries (`.!?`), respecting configurable word limits (default: 50 words) to ensure optimal prosody and avoid token truncation.
- **Persistent MD5 Caching**: Audio chunks are hashed and cached in `cache/audio_chunks/`. Interrupting or resuming conversions reuses existing audio instantaneously.
- **Per-Page Audio Export**: Supports `--per-page` export to save individual `.mp3` or `.wav` files per page.

### CLI Quickstart

```bash
# 1. Convert an entire folder of MinerU/OCR page files to an M4B audiobook with chapters:
python3 audiobook_tts.py /path/to/txt_pages/ -o audiobooks/mybook.m4b

# 2. Export individual audio files per page in addition to the master M4B:
python3 audiobook_tts.py /path/to/txt_pages/ -o audiobooks/mybook.m4b --per-page

# 3. Specify GPUs (multi-GPU parallel synthesis):
python3 audiobook_tts.py /path/to/txt_pages/ --devices "cuda:0,cuda:1"

# 4. Clean up intermediate chunk files after successful assembly:
python3 audiobook_tts.py /path/to/txt_pages/ -o audiobooks/mybook.m4b --clean-chunks

# 5. Use a custom cloned reference voice:
python3 audiobook_tts.py mybook.txt --reference-audio sample.wav

# 6. List available reference voices:
python3 audiobook_tts.py --list-references

# 7. Toggle between Qwen3-TTS (default) and Chatterbox TTS:
python3 audiobook_tts.py /path/to/txt_pages/ --tts-engine qwen3       # Default (Qwen3-TTS 1.7B Base FP16)
python3 audiobook_tts.py /path/to/txt_pages/ --tts-engine chatterbox  # Fallback to Chatterbox TTS FP16
TTS_ENGINE=chatterbox ./batch_fast.sh /path/to/txt_pages/             # Toggle via environment variable
```

> **Note on Automatic Cleanup**: As soon as chunk synthesis completes (and before FFmpeg container encoding begins), `audiobook_tts.py` automatically shuts down all GPU worker processes, deletes the model from memory, and purges PyTorch CUDA cache (`empty_cache` + `ipc_collect`). If `--clean-chunks` is passed, it also removes intermediate WAV files once the audiobook is assembled.

---

## Configuration Defaults (`config.py`)

Central configuration is stored in [`config.py`](config.py):

| Parameter | Default | Description |
|:---|:---|:---|
| `TTS_ENGINE` | `"qwen3"` | Active TTS engine: `"qwen3"` (default) or `"chatterbox"` |
| `QWEN_TTS_ENV` | `"/mnt/raid0/qwen/src/tts-env"` | Dedicated Qwen3-TTS virtualenv on RAID0 |
| `DEFAULT_REFERENCE_ID` | `"practical_engineering"` | Default zero-shot voice clone reference (`practical_engineering.wav` + `.lab`) |
| `MODEL_TYPE` | `"standard"` | `"standard"` (`Qwen3-TTS-12Hz-1.7B-Base` / Chatterbox 500M) or `"turbo"` (`0.6B` / 350M) |
| `SAMPLE_RATE` | `24000` | Audio output sample rate (24 kHz) |
| `DEVICES` | `["cuda:0", "cuda:1"]` | Multi-GPU worker allocation (dual Tesla P100s in FP16) |
| `CHUNK_SIZE_WORDS` | `24` | Target words per chunk |
| `CHUNK_GAP_MS` | `180` | Silence gap (ms) between chunks |
| `AUDIO_FORMAT` | `"m4b"` | Default export container |
| `AUDIO_BITRATE` | `"192k"` | AAC/MP3 export bitrate |

---

## Automated Batch Pipeline (`batch_fast.sh`)

[`batch_fast.sh`](batch_fast.sh) (aliased as `batch.sh`) orchestrates the end-to-end processing pipeline:
1. **Chatterbox TTS Audiobook Synthesis**: Reads text page directories in `../TTS_Script/out/` (or specified directories) and synthesizes multi-GPU accelerated `.m4b` audiobooks with embedded chapter markers.
2. **Safe Source Archiving**: Once verified and completely synthesized, moves the processed folder to `completed/`.
3. **EPUB Generation**: Automatically runs [`mineru_to_epub.py`](mineru_to_epub.py) to produce a companion `.epub` ebook (detects and utilizes rich MinerU Markdown with MathML and images if present, or uses the completed text pages).
4. **Audiobookshelf Upload**: Runs [`upload_to_abs.py`](upload_to_abs.py) to upload both `.m4b` and `.epub` into AudioBookshelf as a unified audio + ebook item, then triggers a library scan.

### Usage Examples

```bash
# 1. Run batch processing over all folders in TTS_Script/out:
./batch_fast.sh

# 2. Process specific book directory:
./batch_fast.sh /path/to/my_book_txt/

# 3. Specify multi-GPU devices and clean up intermediate chunk WAV files:
./batch_fast.sh --devices "cuda:0,cuda:1" --clean-chunks

# 4. Skip upload or skip EPUB generation:
./batch_fast.sh --skip-upload
./batch_fast.sh --skip-epub

# 5. Override directories or AudioBookshelf connection:
./batch_fast.sh --in-dir /custom/in --out-dir /custom/audiobooks --server http://localhost:3378
```

---

## Companion Tools

- **MinerU to EPUB3 Converter** (`mineru_to_epub.py`):
  Converts text/markdown to compliant EPUB3 ebooks with MathML equations and embedded images.
  ```bash
  python3 mineru_to_epub.py path/to/input.md -o output.epub
  ```

- **Audiobookshelf Uploader** (`upload_to_abs.py`):
  Uploads `.m4b` audiobooks and companion `.epub` ebooks directly to an Audiobookshelf server.
  ```bash
  python3 upload_to_abs.py audiobooks/mybook.m4b --epub mybook.epub
  ```

- **GPU Inspection** (`check_gpus.py`):
  Checks real-time GPU VRAM, compute usage, and running process mappings.
  ```bash
  python3 check_gpus.py
  ```
