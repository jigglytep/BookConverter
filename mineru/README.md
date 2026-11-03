# MinerU on Podman

Extract Markdown + structured JSON from PDFs. Built for a nightly batch job on
CPU, with an optional GPU profile for machines that can actually use it.

## Files

| File | Purpose |
|---|---|
| `compose.yaml` | Three profiles: `batch`, `api`, `gpu` |
| `Dockerfile.cpu` | Slim CPU-only image (~2.5 GB vs ~20 GB official) |
| `env.example` | Copy to `.env` and edit |

## Setup

```bash
cp env.example .env && $EDITOR .env
mkdir -p in out
podman compose --profile batch build
```

`podman compose` needs podman 4.7+ (it delegates to `docker-compose` or
`podman-compose`). If profile handling misbehaves under `podman-compose`,
name the service explicitly instead: `podman-compose run --rm mineru-batch`.

## Run a batch

Drop PDFs in `./in`, then:

```bash
podman compose --profile batch run --rm mineru-batch
```

First run downloads ~2 GB of models into the `mineru-models` named volume.
After that succeeds, set `MINERU_MODEL_SOURCE=local` in `.env` so a model-hub
change can't silently alter your output.

Single file, or a page range for testing:

```bash
podman compose --profile batch run --rm mineru-batch \
  mineru -p /data/in/book.pdf -o /data/out -b pipeline -m txt -s 0 -e 20
```

## Fast Dual-GPU VLM Book Processing (Production)

For high-throughput, dual-GPU batch conversion using persistent container workers, atomic work-stealing, and automatic repetition/blank-page handling:

```bash
# Recommended: Convert any book using both Tesla P100 GPUs
./scripts/fast_split_and_process.py in/Python_for_Finance_Mastering.pdf

# Or with options (e.g. custom retry passes or verbose logging)
./scripts/fast_split_and_process.py in/book.pdf --max-retries 5 --verbose
```

For digital technical books where 100% exact code/character accuracy is required, you can also run the **Pipeline** or **Hybrid** modes:
```bash
# Pure Digital CV Pipeline (YOLOv8 layout + 100% exact digital font glyphs)
podman run --rm -v ./in:/data/in:ro,z -v ./out:/data/out:z -v mineru_mineru-models:/models:z \
  -e MINERU_TOOLS_CONFIG_JSON=/models/mineru.json localhost/mineru-cpu:latest \
  mineru -p /data/in/book.pdf -o /data/out -b pipeline

# Native Hybrid Mode (VLM layout + digital text insertion)
podman run --rm --network=mineru_mineru-net -v ./in:/data/in:ro,z -v ./out:/data/out:z -v mineru_mineru-models:/models:z \
  -e MINERU_TOOLS_CONFIG_JSON=/models/mineru.json localhost/mineru-cpu:latest \
  mineru -p /data/in/book.pdf -o /data/out -b hybrid-http-client --server-url http://llamacpp:30000
```

See [scripts/README.md](mineru/scripts/README.md) and [Deploy.md](mineru/Deploy.md#L279) for complete command options, batch scripts, and architecture details.

## Output shape

Each input gets its own folder under `out/`:

- `<name>.md` — the Markdown. Formulas as LaTeX, tables as HTML or pipe tables.
- `<name>_content_list.json` — flat ordered list of blocks with type labels
  (`text`, `title`, `equation`, `table`, `image`). **This is the file you want
  downstream**, not the Markdown — it lets you route equations and tables to a
  math-to-speech pass and send only prose blocks straight to TTS.
- `<name>_middle.json` — full layout intermediate, with bounding boxes.
- `images/` — extracted figures.

Verify the exact filenames on your first run; MinerU has renamed these between
minor versions.

## Backends

| Backend | Needs | Notes |
|---|---|---|
| `pipeline` | CPU or any CUDA GPU | Small specialist models: layout, OCR, formula, table. What this stack defaults to. |
| `hybrid-engine` | **SM 7.0+** GPU, 8 GB+ VRAM | Current default upstream. Small models + a VLM. Best accuracy. |
| `vlm-engine` | **SM 7.0+** GPU | Pure VLM. |
| `hybrid-http-client` / `vlm-http-client` | CPU + network | Thin client against a remote GPU server. |

The SM 7.0 floor comes from vLLM/lmdeploy. Pascal (SM 6.0, i.e. P100) cannot
run the hybrid or VLM engines at all — not slowly, not at all. It *can* run the
`pipeline` backend on GPU, but only if you pin a torch build that still ships
`sm_60` kernels; recent CUDA 12.8+ wheels dropped them.

## Useful knobs

Set in `.env` or the `environment:` block:

- `MINERU_THREADS` — physical cores. Past that, ONNX gets slower, not faster.
- `MINERU_FORMULA_ENABLE=false` / `MINERU_TABLE_ENABLE=false` — big speedup if
  you don't need them. You do need them for technical books.
- `MINERU_PROCESSING_WINDOW_SIZE` — lower it if a 600-page book blows RAM.
- `MINERU_API_MAX_CONCURRENT_REQUESTS` — each concurrent request holds its own
  model copy. This is the OOM knob on the API service.

## Podman gotchas this file already handles

- **`:z` on every bind mount.** Without it SELinux blocks container reads on
  RHEL/Fedora. Use `:Z` instead if the volume is used by exactly one container.
- **CDI for GPUs.** Podman ignores docker's
  `deploy.resources.reservations.devices` block. The `gpu` profile uses
  `devices: nvidia.com/gpu=all`, which requires running once on the host:
  ```bash
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  ```
  Regenerate this after every driver update or the container stops seeing GPUs.
- **File ownership.** The image runs as root, which under rootless podman maps
  to *your* UID on the host, so `out/` is writable and owned by you. Under
  rootful podman add `userns_mode: "keep-id"` instead.
- **`shm_size`.** Torch dataloaders will fail with cryptic bus errors on the
  128 MB default.

## Batch driver

For a nightly job, loop on the host rather than pointing MinerU at a 200-file
directory — one bad PDF shouldn't take the whole run down:

```bash
#!/usr/bin/env bash
set -uo pipefail
for pdf in in/*.pdf; do
  name=$(basename "$pdf" .pdf)
  [[ -d "out/$name" ]] && { echo "skip $name"; continue; }
  echo "=== $name"
  timeout 4h podman compose --profile batch run --rm mineru-batch \
    mineru -p "/data/in/$(basename "$pdf")" -o /data/out -b pipeline -m txt \
    || echo "FAILED: $name" >> out/failures.txt
done
```
