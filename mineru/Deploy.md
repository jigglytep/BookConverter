# Deploy.md — Recreating and Running MinerU on robie

This documents the **actual, currently-running** MinerU deployment on `robie`,
as a disaster-recovery / rebuild guide. `RunBook.md` in this repo is the
exploration log that got here (three machines, three backend tracks, lots of
"unverified"); this file is the shorter "what's real and how do I get back to
it" version, scoped to robie only.

Status as of writing: **Track C (llama.cpp + GGUF → `vlm-http-client`) is
confirmed working and is what robie actually runs in production**, not just a
theory from the runbook. `RunBook.md` §1.4/1.6 still has useful background on
*why* it's built this way (sampling params, repeat-penalty trap, etc.) —
read it if something breaks.

---

## 0. What "recreate" means here

**This directory is not a git repository.** There is no remote to `git clone`
from — `compose.yaml`, `Dockerfile.cpu`, `Containerfile.llamacpp`, `.env`,
`batch.sh`, and `scripts/` are the only copies that exist, all live on
robie's disk at `~/mineru`. If robie's disk is lost, this whole deployment
has to be reconstructed from this Deploy.md plus whatever backup you made of
the directory. **Back this directory up** (e.g. `git init` it and push to a
remote, or periodic `rsync` elsewhere) before you rely on this doc as your
only copy — it currently isn't.

Also note: `env.example` referenced by `README.md`/`RunBook.md` does not
exist on disk. Only the real `.env` does. Treat §3 below as the template.

---

## 1. Hardware and OS (robie, as installed)

| Item | Value |
|---|---|
| CPU | 2× Intel Xeon E5-2640 v4 @ 2.40GHz (2 sockets, 10 physical cores/socket, 40 threads total) |
| GPU | 2× Tesla P100-PCIE-16GB (SM 6.0/Pascal) + 1× Quadro P1000 (display, not used by MinerU) |
| Driver | NVIDIA 580.173.02 |
| Host CUDA toolkit | 12.4 (`nvcc` — used only to confirm the driver stack; containers bring their own CUDA runtime) |
| OS | Ubuntu 26.04 LTS (Resolute Raccoon) |
| Container storage | `/mnt/raid0` (2.7T, plenty of room) — **not** `/` (root fs is ~94% full, only ~5.8G free; don't put large volumes there) |
| Podman | 5.7.0, with `podman-compose` 1.5.0 as the compose provider |

**P100 = SM 6.0.** This is below the SM 7.0 floor vLLM/LMDeploy require, which
is why robie doesn't run MinerU's default `hybrid-engine`/`vlm-engine`
backends directly and instead routes through llama.cpp. See `RunBook.md`
Part 1 for the full reasoning; the short version is CUDA 13 dropped Pascal
entirely, so everything GPU-side here is deliberately pinned to CUDA 12.4.x.

---

## 2. Rebuilding from bare metal

### 2.1 — OS-level prerequisites

```bash
# NVIDIA driver + container toolkit (already on robie: driver 580.173.02)
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit   # provides nvidia-ctk

# Podman + compose provider
sudo apt-get install -y podman podman-compose
podman --version          # need 4.7+
podman compose version    # should report it's delegating to podman-compose
```

### 2.2 — GPU passthrough (CDI)

Podman does not read Docker's `deploy.resources.reservations.devices` block —
it needs a CDI spec. Run once, and again after **every driver update**:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list        # expect nvidia.com/gpu=all plus one entry per GPU
```

Verify passthrough before building anything:

```bash
podman run --rm --device nvidia.com/gpu=all --security-opt label=disable \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see all three GPUs (2× P100 + the P1000). If you get "no devices
found," the CDI file is stale (regenerate it) or SELinux is blocking
(`--security-opt label=disable` covers that).

### 2.3 — Point container storage at the big disk

Root (`/`) has ~6G free on robie; container images and the GGUF model volume
do not fit there. Podman on robie is already configured to store under
`/mnt/raid0/containers/storage` (check `podman info --format
'{{.Store.GraphRoot}}'`). If setting this up fresh, configure that in
`/etc/containers/storage.conf` (`graphroot = "/mnt/raid0/containers/storage"`)
**before** pulling or building any images.

### 2.4 — Get the deployment files

Since there's no git remote, restore `~/mineru` from whatever backup you took
(§0). At minimum you need:

```
compose.yaml
Dockerfile.cpu
Containerfile.llamacpp
.env                    # or reconstruct from §3 below
batch.sh
scripts/process_one_page.py
scripts/split_and_process.sh
```

```bash
mkdir -p ~/mineru && cd ~/mineru
mkdir -p in out models/gguf
```

### 2.5 — Get the GGUF model (llama.cpp path)

This is the large, slow part — 13G+ of quantizations. robie currently has the
**entire** quant ladder pulled from `Mungert/MinerU2.5-2509-1.2B-GGUF`
(bf16 down to iq3_xxs) in `models/gguf/`, but only two files are actually
used at runtime — the ones named in `.env` (`GGUF_FILE` / `MMPROJ_FILE`,
currently `q8_0`). You don't need the rest; pulling just those two is enough
to run:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Mungert/MinerU2.5-2509-1.2B-GGUF \
  MinerU2.5-2509-1.2B-q8_0.gguf MinerU2.5-2509-1.2B-q8_0.mmproj \
  --local-dir ./models/gguf
```

Q8_0 (not Q4) is the deliberate choice — at 1.2B params the disk savings from
heavier quantization aren't worth it, and this model emits structured
coordinate/layout tokens that degrade badly under aggressive quantization.
You need **both** the model file and the `mmproj` (vision projector); without
the projector the model is text-only and produces garbage on page images.

### 2.6 — Build the images

```bash
cd ~/mineru
podman compose --profile batch build        # Dockerfile.cpu -> localhost/mineru-cpu:latest (~2G)
podman compose --profile vlm build llamacpp  # Containerfile.llamacpp -> localhost/llamacpp-pascal:latest (~2.4G, compiles CUDA, 10-20 min)
```

`Containerfile.llamacpp` builds llama.cpp from source pinned to
`CUDA_ARCH=60` (from `.env`) because prebuilt llama.cpp CUDA images don't
reliably ship `sm_60` kernels for Pascal.

### 2.7 — Start the VLM server and confirm it's real

```bash
podman compose --profile vlm up -d llamacpp
podman logs -f llamacpp     # wait for "server is listening"
```

```bash
curl -s http://localhost:30000/health
curl -s http://localhost:30000/v1/models | python3 -m json.tool
nvidia-smi                   # expect ~2-3G used on GPU 0 (CUDA_VISIBLE_DEVICES in .env)
```

If VRAM usage stays near zero, `--n-gpu-layers 99` didn't take — check
`podman logs llamacpp | grep -i cuda` to confirm the build actually has CUDA
support compiled in.

### 2.8 — Smoke test end to end

```bash
podman compose --profile client run --rm mineru-vlm \
  mineru -p /data/in/<some.pdf> -o /data/out -b vlm-http-client \
  -u http://llamacpp:30000 -s 0 -e 2
```

Confirms the whole chain — container networking (`mineru-net`), the client
image, and the llama.cpp server — before trusting it with a full book.

---

## 3. `.env` reference (current robie config)

This is what's actually deployed. Copy it as the starting template if
rebuilding (there is no `env.example` on disk to copy from):

```ini
IN_DIR=./in
OUT_DIR=./out
GGUF_DIR=./models/gguf

MINERU_THREADS=10        # one socket's worth of physical cores — NOT 20.
                          # Two-socket NUMA hurts these small models; ONNX/torch's
                          # OpenMP pool doesn't NUMA-partition itself.
MINERU_METHOD=auto        # auto = decide OCR vs text-layer per page; use "txt"
                          # for a known-born-digital book (faster + better)

MINERU_MODEL_SOURCE=huggingface   # switch to "local" after first successful
                                   # run so a model-hub change can't silently
                                   # alter output mid-project
MINERU_VERSION=           # empty = latest at build time. Pin for reproducibility.

API_PORT=8000
API_CONCURRENCY=2

# VLM profile (robie-specific — Pascal / llama.cpp path)
CUDA_ARCH=60
GPU_IDS=0
VLM_PORT=30000
GGUF_FILE=MinerU2.5-2509-1.2B-q8_0.gguf
MMPROJ_FILE=MinerU2.5-2509-1.2B-q8_0.mmproj
VLM_MODEL_NAME=MinerU2.5-2509-1.2B

# client profile (used when a client container points at llamacpp)
VLM_URL=http://llamacpp:30000

MINERU_TASK_RESULT_TIMEOUT_SECONDS=43200   # 12h — large scanned books can run 8h+
MINERU_API_MAX_CONCURRENT_REQUESTS=1        # one task owns the GPU

CUDA_VISIBLE_DEVICES=1     # note: this line as written is a stray/no-op in
                            # top-level .env — the effective value is set per-
                            # service inside compose.yaml (mineru-batch/-api
                            # force "", llamacpp uses GPU_IDS)
MINERU_VIRTUAL_VRAM_SIZE=12   # assume 12G, leaves headroom on the 16G card
```

`MINERU_METHOD=auto` here differs from the `txt` that `README.md`/
`RunBook.md` recommend for born-digital PDFs — that's robie's live setting,
not necessarily what you want for a new book. Set `MINERU_METHOD=txt`
per-run if you know the source is a clean digital PDF, not a scan.

---

## 4. Running MinerU on robie

There are three distinct ways this gets run. **Use §4.3 for anything that
matters** — it's the one actually validated against robie's hardware. §4.1
and §4.2 exist and work, but at lower accuracy or higher risk of losing a
long run.

### 4.1 — CPU-only batch (`pipeline` backend, no GPU, lowest accuracy)

Baseline / fallback. Works everywhere, doesn't touch the P100s at all.

```bash
cd ~/mineru
# drop PDFs into ./in first
podman compose --profile batch run --rm mineru-batch
```

Or `batch.sh` for a directory of PDFs, one container invocation per file
(so one bad PDF doesn't kill the whole run):

```bash
./batch.sh
```

Single file / page range for testing:

```bash
podman compose --profile batch run --rm mineru-batch \
  mineru -p /data/in/book.pdf -o /data/out -b pipeline -m txt -s 0 -e 20
```

### 4.2 — Whole-PDF VLM run via compose (`vlm-http-client`)

```bash
podman compose --profile vlm up -d llamacpp        # start the server if not already up
podman compose --profile client run --rm mineru-vlm
```

This runs `mineru -b vlm-http-client -u http://llamacpp:30000` against
**everything** in `in/` in one task. **Known risk:** a single malformed VLM
response (repetition-loop truncation, format error) can hard-crash a whole
multi-hour task with nothing written to disk — this actually happened on
robie (2026-07-28, a 4.5h/640-page run lost outright). That incident is why
§4.3 exists and is the recommended path for anything long-running.

### 4.3 — Fast Dual-GPU Per-page VLM Run (`scripts/fast_split_and_process.py`)

This is the optimized production path on robie. It:
- Utilizes **both Tesla P100 GPUs** simultaneously (`llamacpp` on GPU 0 :30000, `llamacpp1` on GPU 1 :30001).
- Keeps persistent worker containers (`mineru-fast-w0` and `mineru-fast-w1`), eliminating the ~11.3s per-page container startup and Python/PyTorch import overhead.
- Uses dynamic atomic work-stealing so uneven page complexities don't leave either GPU idle.
- Preserves full per-page crash safety and automatic retry passes.
- Effective throughput: **~4.5–5 seconds per page** (~450–500 pages/hour), cutting a 700-page book conversion from ~5 hours to ~1.2 hours.

```bash
cd ~/mineru
./scripts/fast_split_and_process.py in/PracticalElectronicsForInventors.pdf
```

Legacy single-GPU per-container script (`scripts/split_and_process.sh`) is still available if needed.

What it does:
- Checks `http://localhost:${VLM_PORT}/health` and starts `llamacpp` if it's
  not already up.
- Reads the page count via `pdfinfo`.
- For each page not already present as
  `out/<book>_md/<book>_page_NNNN.md`, runs
  `scripts/process_one_page.py` inside the `mineru-vlm` container — which
  calls `mineru`'s `do_parse()` directly with `start_page_id`/`end_page_id`
  pinned to that one page and `max_concurrency=1` forced (the CLI has no flag
  for this; the http-client backend otherwise fires up to 100 concurrent
  block-level requests for a single page, which just queues uselessly since
  `llama-server` runs with `--parallel 1`).
- Pages are extracted via mineru's own pdfium path in-memory, **not**
  `pdfseparate`. On at least one book, `pdfseparate` hit a poppler
  "recursive dicts" bug that bloated every single-page PDF to ~30M (a
  1000+ page book would need 30G+ of scratch disk); pdfium extracts the same
  page at ~100K, no disk write needed.
- If a page's response comes back truncated (`finish_reason=length`), the
  output is discarded and the page is queued for retry — but not retried on
  the spot. Pass 1 runs every page in the book once; only after that full
  pass finishes does pass 2 retry whatever failed, then pass 3, etc, up to
  `MAX_RETRIES` passes (default 3), with `llamacpp` **restarted** once at the
  start of each retry pass. This matters because sampling is fully greedy
  (`temp=0`, `top_k=1`): without a KV-cache/slot-state reset, a bare retry
  reliably reproduces the exact same truncating loop. A page that's
  genuinely too dense for `--ctx-size` will truncate every time regardless,
  so infinite retry isn't useful.
- Failures still outstanding after all passes land in
  `out/<book>_failures.txt`. Rerun the script to retry only what failed.

Output layout:
```
out/<book>_pages/<book>_page_NNNN/vlm/<book>_page_NNNN.md   # per-page mineru output
out/<book>_md/<book>_page_NNNN.md                            # flattened copy, one per page
out/<book>_failures.txt                                      # failure log
```

Override retry count: `MAX_RETRIES=5 ./scripts/split_and_process.sh in/book.pdf`

### 4.4 — Persistent API service

```bash
podman compose --profile api up -d mineru-api
curl http://localhost:8000/docs
```

CPU-only (`pipeline` backend), same as §4.1 but as a long-running HTTP
service instead of one-shot runs. `MINERU_API_MAX_CONCURRENT_REQUESTS`
controls how many requests run in parallel — each holds its own model copy
in memory, so this is the OOM knob.

### 4.5 — Second GPU

robie has two P100s; only GPU 0 is in use by default (`GPU_IDS=0` in `.env`).
To split a book across both:

```bash
GPU_IDS=1 VLM_PORT=30001 podman compose --profile vlm up -d llamacpp
```

Give the container a distinct name (podman refuses a duplicate) and point a
second `split_and_process.sh` run's `VLM_URL`/`VLM_PORT` at the new port.
This is a data-parallel split (two independent servers, half the pages
each), not tensor parallelism — the right shape for batch work.

---

## 5. Output shape (all backends)

Each input gets a folder under the output dir:

- `<name>.md` — Markdown. Formulas as LaTeX, tables as HTML or pipe tables.
- `<name>_content_list.json` — flat ordered list of blocks with type labels
  (`text`, `title`, `equation`, `table`, `image`). **Use this, not the
  Markdown**, for anything downstream (e.g. routing equations/tables to a
  math-to-speech pass and prose straight to TTS).
- `<name>_middle.json` — full layout intermediate with bounding boxes.
- `images/` — extracted figures.

Verify these filenames on any MinerU version bump — they've changed between
minor releases before.

---

## 6. Verifying the deployment is healthy

```bash
# GPU passthrough alive
nvidia-smi

# llama.cpp server up and actually using GPU 0
curl -s http://localhost:30000/health
podman logs llamacpp | grep -i cuda

# containers/images present
podman images | grep -E 'mineru-cpu|llamacpp-pascal'
podman ps --filter name=llamacpp

# GGUF files present and match .env
ls -la ~/mineru/models/gguf | grep -E "$(grep -oP '(?<=GGUF_FILE=).*' ~/mineru/.env)"
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no CUDA-capable device` in container | Stale CDI spec after driver update | `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` |
| Container can't read mounted PDFs | SELinux | `:z` on the bind mount (already in `compose.yaml`) |
| Output files owned by root/165536 | Rootful podman / userns mismatch | Rootless: root-in-container = your UID (robie's default). Rootful: add `userns_mode: "keep-id"` |
| `Layout output does not match expected format` | Sampling params mangling structured output | Confirm `--repeat-penalty 1.0 --temp 0.0 --top-k 1` in `compose.yaml`'s `llamacpp` command — never re-add a nonzero repeat-penalty or DRY sampling here (see `compose.yaml` comment: DRY sampling was tried 2026-07-28, corrupted structured tags, killed a 4.5h run) |
| llama.cpp builds but runs on CPU | `sm_60` kernels missing | Rebuild with `CUDA_ARCH=60` (already default); check `podman logs llamacpp \| grep -i cuda` |
| `torch not compiled with CUDA capability sm_60` | CUDA 13 / torch cu128+ wheels | Pin CUDA 12.4.x; `Containerfile.llamacpp` already does this |
| Bus errors / dataloader crashes | 128M default `/dev/shm` | `shm_size: "8gb"` (already set in `compose.yaml`) |
| OOM on `mineru-api` | Each concurrent request holds its own models | Lower `MINERU_API_MAX_CONCURRENT_REQUESTS` |
| 600-page book exhausts RAM (pipeline backend) | Processing window too large | Lower `MINERU_PROCESSING_WINDOW_SIZE` |
| Whole run lost after hours, nothing on disk | A single bad page hard-crashed the task (§4.2) | Use `scripts/split_and_process.sh` (§4.3) instead — one page per task |
| Page truncated, same result every retry | Greedy decoding replays the same truncating loop without a state reset | `split_and_process.sh` already restarts `llamacpp` once at the start of each retry pass (retries run after the whole book, not immediately) — if still failing after `MAX_RETRIES`, the page is likely too dense for `--ctx-size`; raise `--ctx-size` in `compose.yaml` |
| `podman restart llamacpp` intermittently fails ("kill network process: permission denied") | Rootless podman network teardown flakiness | `split_and_process.sh` already retries the restart + health check up to 3 times |
| Root filesystem full during build/pull | Podman storage pointed at `/` instead of `/mnt/raid0` | Check `podman info --format '{{.Store.GraphRoot}}'`; should be under `/mnt/raid0` |

---

## 8. What's out of scope here

- **BlackBetty / M1** setups — see `RunBook.md` Parts 2 and 3. Not deployed
  anywhere; documented as untested alternatives.
- **`hybrid-engine`/`vlm-engine`/vLLM/LMDeploy** — require SM 7.0+, which the
  P100s don't have. Not applicable to robie.
- Math-to-speech / TTS pipeline downstream of MinerU's output — separate
  system (`qwen3-tts` container, already running on robie for other work),
  not covered by this file.
