#!/usr/bin/env python3
"""
Smoke test and benchmark for the Chatterbox TTS GPU implementation.

Checks:
  1. PyTorch sees CUDA and reports device details
  2. Chatterbox imports and loads onto the GPU
  3. All model parameters actually reside on the GPU
  4. Generation works (warmup + timed runs), output audio is valid
  5. Reports latency, real-time factor (RTF), and peak VRAM
  6. Optional CPU baseline comparison

Install:
  pip install chatterbox-tts

Usage:
  python test_chatterbox_gpu.py
  python test_chatterbox_gpu.py --device cuda:1 --runs 5
  python test_chatterbox_gpu.py --voice ref.wav --outdir out
  python test_chatterbox_gpu.py --cpu-baseline
  python test_chatterbox_gpu.py --json results.json
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

DEFAULT_TEXTS = [
    "Chapter 1",
]

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"


def log(tag, msg):
    print(f"{tag} {msg}", flush=True)


def check_cuda(device_str):
    import torch

    log("[INFO]", f"torch {torch.__version__}, CUDA build: {torch.version.cuda}")
    if not torch.cuda.is_available():
        log(FAIL, "torch.cuda.is_available() is False. Check driver, CUDA build "
                  "of torch, and (in containers) --gpus / CDI device passthrough.")
        return False

    n = torch.cuda.device_count()
    log(PASS, f"CUDA available, {n} device(s)")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        log("[INFO]", f"  cuda:{i} {p.name} | {p.total_memory / 2**30:.1f} GiB | "
                      f"compute {p.major}.{p.minor}")

    try:
        idx = torch.device(device_str).index or 0
        if idx >= n:
            log(FAIL, f"Requested {device_str} but only {n} device(s) present")
            return False
        # Sanity: real kernel on the device
        x = torch.randn(1024, 1024, device=device_str)
        y = (x @ x).sum().item()
        torch.cuda.synchronize(device_str)
        log(PASS, f"Matmul kernel executed on {device_str} (checksum {y:.2f})")
    except Exception as e:
        log(FAIL, f"CUDA kernel test failed: {e!r}")
        return False
    return True


def sync(device):
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def load_model(device):
    from chatterbox.tts import ChatterboxTTS

    t0 = time.perf_counter()
    model = ChatterboxTTS.from_pretrained(device=device)
    sync(device)
    return model, time.perf_counter() - t0


def verify_placement(model, device):
    """Confirm submodules' parameters live on the requested device."""
    import torch

    target = torch.device(device)
    bad = {}
    total = 0
    for name in ("t3", "s3gen", "ve"):
        mod = getattr(model, name, None)
        if mod is None or not hasattr(mod, "parameters"):
            continue
        for p in mod.parameters():
            total += 1
            if p.device.type != target.type:
                bad.setdefault(name, 0)
                bad[name] += 1
    if bad:
        log(FAIL, f"Parameters off-device: {bad}")
        return False
    log(PASS, f"All inspected parameters ({total}) are on {device}")
    return True


def generate(model, text, voice, device):
    import torch

    kwargs = {"text": text}
    if voice:
        kwargs["audio_prompt_path"] = voice
    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        wav = model.generate(**kwargs)
    sync(device)
    dt = time.perf_counter() - t0
    return wav, dt


def validate_audio(wav, sr):
    import torch

    if not isinstance(wav, torch.Tensor):
        return False, "output is not a tensor"
    if not torch.isfinite(wav).all():
        return False, "output contains NaN/Inf"
    dur = wav.shape[-1] / sr
    if dur < 0.2:
        return False, f"audio too short ({dur:.2f}s)"
    peak = wav.abs().max().item()
    if peak < 1e-4:
        return False, f"audio is silent (peak {peak:.2e})"
    return True, f"{dur:.2f}s, peak {peak:.3f}"


def run_benchmark(model, device, texts, runs, voice, outdir, label):
    import torch
    import torchaudio as ta

    results = []
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    log("[INFO]", f"[{label}] warmup run...")
    _, warm = generate(model, "Warm up the model.", voice, device)
    log("[INFO]", f"[{label}] warmup took {warm:.2f}s")

    ok_all = True
    for ti, text in enumerate(texts):
        times, rtfs = [], []
        for r in range(runs):
            wav, dt = generate(model, text, voice, device)
            ok, info = validate_audio(wav, model.sr)
            if not ok:
                log(FAIL, f"[{label}] text {ti} run {r}: {info}")
                ok_all = False
                continue
            dur = wav.shape[-1] / model.sr
            times.append(dt)
            rtfs.append(dt / dur)
            if r == 0 and outdir:
                path = Path(outdir) / f"{label}_text{ti}.wav"
                ta.save(str(path), wav.cpu(), model.sr)
        if times:
            res = {
                "label": label,
                "text_index": ti,
                "chars": len(text),
                "mean_s": statistics.mean(times),
                "min_s": min(times),
                "rtf_mean": statistics.mean(rtfs),
            }
            results.append(res)
            log(PASS, f"[{label}] text {ti} ({len(text)} chars): "
                      f"mean {res['mean_s']:.2f}s, min {res['min_s']:.2f}s, "
                      f"RTF {res['rtf_mean']:.2f}x ({info})")

    peak = None
    if is_cuda:
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        log("[INFO]", f"[{label}] peak VRAM allocated: {peak:.2f} GiB")
    return ok_all, results, peak


def main():
    ap = argparse.ArgumentParser(description="Chatterbox TTS GPU test")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:0, cuda:1, ...")
    ap.add_argument("--runs", type=int, default=3, help="timed runs per text")
    ap.add_argument("--voice", default=None, help="reference voice wav (optional)")
    ap.add_argument("--outdir", default="chatterbox_test_out")
    ap.add_argument("--cpu-baseline", action="store_true",
                    help="also run the first text on CPU for speedup comparison")
    ap.add_argument("--json", default=None, help="write summary to this JSON file")
    args = ap.parse_args()

    if args.voice and not Path(args.voice).is_file():
        log(FAIL, f"Voice file not found: {args.voice}")
        return 2

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    summary = {"device": args.device, "passed": False}

    if not check_cuda(args.device):
        return 1

    try:
        import chatterbox  # noqa: F401
    except ImportError:
        log(FAIL, "chatterbox not installed. Run: pip install chatterbox-tts")
        return 1

    import torch

    try:
        model, load_s = load_model(args.device)
    except Exception as e:
        log(FAIL, f"Model load failed: {e!r}")
        return 1
    log(PASS, f"Model loaded on {args.device} in {load_s:.1f}s (sr={model.sr})")
    summary["load_s"] = load_s

    if not verify_placement(model, args.device):
        return 1

    if str(args.device).startswith("cuda"):
        alloc = torch.cuda.memory_allocated(args.device) / 2**30
        log("[INFO]", f"VRAM allocated after load: {alloc:.2f} GiB")
        summary["vram_after_load_gib"] = alloc

    try:
        ok, gpu_results, peak = run_benchmark(
            model, args.device, DEFAULT_TEXTS, args.runs, args.voice,
            args.outdir, "gpu")
    except torch.cuda.OutOfMemoryError:
        log(FAIL, "CUDA out of memory during generation")
        return 1
    except Exception as e:
        log(FAIL, f"Generation failed: {e!r}")
        return 1

    summary.update(gpu=gpu_results, peak_vram_gib=peak)

    if args.cpu_baseline:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log("[INFO]", "Loading CPU baseline...")
        try:
            cpu_model, _ = load_model("cpu")
            _, cpu_results, _ = run_benchmark(
                cpu_model, "cpu", DEFAULT_TEXTS[:1], 1, args.voice,
                args.outdir, "cpu")
            if cpu_results and gpu_results:
                speedup = cpu_results[0]["mean_s"] / gpu_results[0]["mean_s"]
                log(PASS if speedup > 1 else WARN,
                    f"GPU speedup over CPU (text 0): {speedup:.1f}x")
                summary["speedup_text0"] = speedup
                summary["cpu"] = cpu_results
        except Exception as e:
            log(WARN, f"CPU baseline failed: {e!r}")

    summary["passed"] = bool(ok)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        log("[INFO]", f"Summary written to {args.json}")

    print()
    log(PASS if ok else FAIL,
        "Chatterbox GPU test " + ("PASSED" if ok else "FAILED"))
    log("[INFO]", f"WAV samples in: {args.outdir}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())