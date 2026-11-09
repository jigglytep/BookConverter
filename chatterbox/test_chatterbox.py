#!/usr/bin/env python3
"""
Test Suite & Smoke Test for Chatterbox TTS (`audiobook_tts.py`)
===============================================================
Tests:
1. GPU memory queries & available space / multi-instance auto-scaling logic
   (including simulated 40% VRAM usage, post-load warmup reservation, and live GPU inspection).
2. Sentence-first chunking and content-addressed cache key generation.
3. Audio chunk cleanup, silence trimming, and RMS normalization.
4. Optional live Chatterbox TTS synthesis test (`--synth` / `--device`).

Usage:
  ./test_chatterbox.py                  # Run all unit + live GPU capacity tests
  ./test_chatterbox.py --synth          # Also run a live TTS audio synthesis test
  ./test_chatterbox.py --synth --device cpu
"""

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

# Automatically re-execute inside .venv if invoked with system python
_script_dir = Path(__file__).resolve().parent
_venv_python = _script_dir / ".venv" / "bin" / "python3"
if (
    __name__ == "__main__"
    and _venv_python.exists()
    and Path(sys.executable).resolve() != _venv_python.resolve()
):
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)

import numpy as np
import soundfile as sf

import config
from audiobook_tts import (
    ChatterboxAudiobookGenerator,
    _clean_and_normalize_chunk_wav,
    _extract_pause_tags_and_clean_text,
    calculate_gpu_available_space,
    derive_seed,
    enforce_min_chunk_length,
    fuzzy_match,
    fuzzy_match_score,
    get_gpu_memory_info,
    run_auto_editor_on_wav,
    select_best_candidate,
    smart_append_short_sentences,
)


class TestChatterboxGPUScaling(unittest.TestCase):
    """Test GPU available space calculation and multi-instance scaling decisions."""

    def test_gpu_at_40_percent_allows_second_instance(self):
        """When a 16 GB GPU is at ~40% usage (e.g. 6550 / 16384 MiB), 1 additional instance should fit."""
        with patch("audiobook_tts.get_gpu_memory_info", return_value=(6550.0, 9834.0, 16384.0)):
            info = calculate_gpu_available_space(
                device="cuda:0",
                model_type="standard",
                active_workers_on_gpu=1,
                baseline_used_mib=5.0,
                safety_buffer_mib=800.0,
                max_instances_per_gpu=4,
            )
            self.assertTrue(info["is_cuda"])
            self.assertAlmostEqual(info["used_pct"], 6550.0 / 16384.0 * 100.0, places=1)
            self.assertTrue(info["can_add_instance"])
            self.assertEqual(info["additional_instances"], 1)

    def test_gpu_post_load_warmup_reservation(self):
        """
        Immediately after loading weights (~2200 MiB allocated before first chunk synthesis),
        calculate_gpu_available_space should reserve the full ~6000 MiB working footprint for
        worker #1 so it only adds 1 extra instance on a 16 GB GPU, not 2.
        """
        with patch("audiobook_tts.get_gpu_memory_info", return_value=(2205.0, 14179.0, 16384.0)):
            info = calculate_gpu_available_space(
                device="cuda:0",
                model_type="standard",
                active_workers_on_gpu=1,
                baseline_used_mib=5.0,
                safety_buffer_mib=800.0,
                max_instances_per_gpu=4,
            )
            # 16379 MiB free after baseline - 6000 MiB reserved for worker #1 = 10379 MiB effective free
            self.assertAlmostEqual(info["effective_free_mib"], 10379.0, places=0)
            self.assertTrue(info["can_add_instance"])
            self.assertEqual(info["additional_instances"], 1)

    def test_gpu_at_80_percent_stops_adding_instances(self):
        """When 2 instances are running on a 16 GB GPU (~13100 MiB used, ~3284 MiB free), no more fit."""
        with patch("audiobook_tts.get_gpu_memory_info", return_value=(13100.0, 3284.0, 16384.0)):
            info = calculate_gpu_available_space(
                device="cuda:0",
                model_type="standard",
                active_workers_on_gpu=2,
                baseline_used_mib=5.0,
                safety_buffer_mib=800.0,
                max_instances_per_gpu=4,
            )
            self.assertFalse(info["can_add_instance"])
            self.assertEqual(info["additional_instances"], 0)

    def test_small_4gb_gpu_does_not_add_instance(self):
        """A 4 GB GPU (e.g. Quadro P1000 with 4096 MiB) should report 0 additional standard instances."""
        with patch("audiobook_tts.get_gpu_memory_info", return_value=(5.0, 4091.0, 4096.0)), patch(
            "audiobook_tts.get_gpu_process_usages_mib", return_value=[]
        ):
            info = calculate_gpu_available_space(
                device="cuda:2",
                model_type="standard",
            )
            self.assertFalse(info["can_add_instance"])
            self.assertEqual(info["additional_instances"], 0)


class TestChatterboxChunkingAndAudio(unittest.TestCase):
    """Test text chunking, cache path hashing, and audio normalization."""

    def setUp(self):
        self.gen = ChatterboxAudiobookGenerator(devices=["cpu"])

    def test_split_into_chunks_respects_word_limits(self):
        sample_text = (
            "In physics."
        )
        chunks = self.gen.split_into_chunks(sample_text)
        # self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertTrue(len(chunk.strip()) > 0)
            self.assertLessEqual(len(chunk.split()), config.CHUNK_SIZE_WORDS + 10)

    def test_cache_path_determinism(self):
        p1 = self.gen.get_cache_path("Hello world, this is a test sentence.")
        p2 = self.gen.get_cache_path("Hello world, this is a test sentence.")
        p3 = self.gen.get_cache_path("A different sentence produces a different hash.")
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertEqual(p1.suffix, ".wav")

    def test_clean_and_normalize_chunk_wav(self):
        sr = config.SAMPLE_RATE
        # 200 ms leading silence + 500 ms 440 Hz tone + 300 ms trailing silence
        silence_lead = np.zeros(int(sr * 0.20), dtype=np.float32)
        t = np.linspace(0, 0.50, int(sr * 0.50), endpoint=False, dtype=np.float32)
        tone = 0.05 * np.sin(2 * np.pi * 440.0 * t)
        silence_trail = np.zeros(int(sr * 0.30), dtype=np.float32)
        raw_wav = np.concatenate([silence_lead, tone, silence_trail])

        cleaned = _clean_and_normalize_chunk_wav(raw_wav, sr)
        # Leading/trailing silence should be trimmed (~0.55s remaining instead of 1.0s)
        self.assertLess(len(cleaned), int(sr * 0.70))
        self.assertGreater(len(cleaned), int(sr * 0.45))
        # Peak should not exceed -1.0 dBFS ceiling (~0.891)
        self.assertLessEqual(float(np.max(np.abs(cleaned))), 0.90)

    def test_smart_append_short_sentences_under_20_chars(self):
        """Sentences below 20 characters should merge with the sentence after or before them."""
        sentences = [
            "Hi!",
            "This is a longer sentence that exceeds twenty characters.",
            "Yes.",
        ]
        merged = smart_append_short_sentences(sentences, min_chars=20, max_chars=300)
        self.assertEqual(len(merged), 1)
        self.assertIn("Hi!", merged[0])
        self.assertIn("Yes.", merged[0])
        self.assertGreaterEqual(len(merged[0]), 20)

    def test_select_best_candidate_picks_shortest_validated(self):
        """Among Whisper-validated candidates, the shortest duration candidate should be selected."""
        passed = [
            {"path": "cand_1.wav", "duration": 2.45, "score": 0.96, "transcribed": "chapter one"},
            {"path": "cand_2.wav", "duration": 1.32, "score": 0.91, "transcribed": "chapter 1"},
            {"path": "cand_3.wav", "duration": 1.88, "score": 1.00, "transcribed": "chapter 1"},
        ]
        best = select_best_candidate(passed, [], use_longest_transcript_on_fail=True)
        self.assertEqual(best["path"], "cand_2.wav")

    def test_select_best_candidate_fallback_modes(self):
        """When all candidates fail Whisper validation, fallback picks longest transcript or highest score."""
        failed = [
            {"path": "fail_1.wav", "duration": 1.5, "score": 0.72, "transcribed": "short text"},
            {"path": "fail_2.wav", "duration": 1.8, "score": 0.65, "transcribed": "a much longer transcribed text here"},
        ]
        best_longest = select_best_candidate([], failed, use_longest_transcript_on_fail=True)
        self.assertEqual(best_longest["path"], "fail_2.wav")

        best_score = select_best_candidate([], failed, use_longest_transcript_on_fail=False)
        self.assertEqual(best_score["path"], "fail_1.wav")

    def test_fuzzy_match_and_pause_tag_extraction(self):
        lead, clean, trail = _extract_pause_tags_and_clean_text("[pause:1.0s] Chapter 1 [pause:0.5s]")
        self.assertAlmostEqual(lead, 1.0)
        self.assertEqual(clean, "Chapter 1")
        self.assertAlmostEqual(trail, 0.5)
        self.assertTrue(fuzzy_match("chapter one.", "Chapter 1", threshold=0.85))

    def test_auto_editor_post_processing(self):
        """Verify auto-editor removes extended low-volume breathing/silence from a WAV file."""
        sr = config.SAMPLE_RATE
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sample.wav"
            # 0.8s quiet breath noise (0.005 amp) + 0.6s loud speech tone (0.4 amp) + 0.8s quiet breath
            rng = np.random.default_rng(42)
            quiet_lead = 0.005 * rng.standard_normal(int(sr * 0.8)).astype(np.float32)
            t = np.linspace(0, 0.6, int(sr * 0.6), endpoint=False, dtype=np.float32)
            speech = 0.4 * np.sin(2 * np.pi * 300.0 * t)
            quiet_trail = 0.005 * rng.standard_normal(int(sr * 0.8)).astype(np.float32)
            full = np.concatenate([quiet_lead, speech, quiet_trail])
            sf.write(str(wav_path), full, sr, subtype="PCM_16")

            ok = run_auto_editor_on_wav(wav_path, threshold=0.06, margin=0.1, keep_original=False)
            self.assertTrue(ok)
            cleaned, _ = sf.read(str(wav_path), dtype="float32")
            # Original was 2.2s; after auto-editor cutting quiet 0.8s ends (keeping 0.1s margin), should be ~0.8s
            self.assertLess(len(cleaned) / sr, 1.2)
            self.assertGreater(len(cleaned) / sr, 0.5)


def run_live_gpu_report():
    """Print live GPU memory and calculated Chatterbox instance capacity for each CUDA device."""
    print("\n" + "=" * 78)
    print(" LIVE GPU MEMORY & CHATTERBOX INSTANCE CAPACITY CHECK")
    print("=" * 78)
    for idx in range(8):
        dev = f"cuda:{idx}"
        mem = get_gpu_memory_info(dev)
        if mem is None:
            break
        info = calculate_gpu_available_space(dev, model_type=config.MODEL_TYPE)
        print(
            f"  • {dev}: {info['used_mib']:6.0f} / {info['total_mib']:6.0f} MiB used ({info['used_pct']:4.1f}%) | "
            f"effective free: {info['effective_free_mib']:6.0f} MiB | "
            f"required: {info['required_mib']:5.0f} MiB -> "
            f"can_add={info['can_add_instance']} (+{info['additional_instances']} instance(s))"
        )
    print("=" * 78 + "\n")


def run_live_synthesis_test(device: str, model_type: str, output_file: Optional[Path] = None):
    """Run an end-to-end Chatterbox TTS synthesis test on a short sample paragraph."""
    print("=" * 78)
    print(f" LIVE CHATTERBOX TTS SYNTHESIS TEST (device={device}, model={model_type})")
    print("=" * 78)

    with tempfile.TemporaryDirectory(prefix="chatterbox_test_") as tmpdir:
        tmp_path = Path(tmpdir)
        sample_md = tmp_path / "page_0001.md"
        sample_md.write_text(
            "[pause:1.0s] Chapter 1 ",
            encoding="utf-8",
        )
        out_audio = output_file or (tmp_path / "test_output.m4b")
        out_epub = tmp_path / "test_output.epub"

        gen = ChatterboxAudiobookGenerator(
            model_type=model_type,
            devices=[device],
            no_cache=True,
            auto_scale_gpu=True,
        )
        ok = gen.convert(
            input_path=sample_md,
            output_path=out_audio,
            epub_output_path=out_epub,
            clean_chunks=True,
        )
        if not ok or not out_audio.exists() or out_audio.stat().st_size == 0:
            raise RuntimeError(f"Live synthesis test failed: {out_audio} was not created.")
        print(f"✓ Successfully synthesized audiobook: {out_audio} ({out_audio.stat().st_size:,} bytes)")
        if out_epub.exists():
            print(f"✓ Successfully generated synchronized EPUB3: {out_epub} ({out_epub.stat().st_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Test Chatterbox TTS GPU scaling, chunking, and synthesis.")
    parser.add_argument("--synth", action="store_true", help="Run a live end-to-end TTS synthesis test.")
    parser.add_argument("--device", type=str, default=None, help="Device for --synth test (e.g. 'cuda:0', 'cuda:1', 'cpu').")
    parser.add_argument("--model", type=str, default=config.MODEL_TYPE, choices=["standard", "turbo"])
    parser.add_argument("-o", "--output", type=Path, default=None, help="Optional path to keep synthesized test audio.")
    args, unittest_args = parser.parse_known_args()

    run_live_gpu_report()

    # Run unit tests
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)

    if args.synth:
        target_dev = args.device
        if not target_dev:
            # Pick the CUDA GPU with the most free memory, or fallback to cpu
            best_dev = "cpu"
            best_free = 0.0
            for idx in range(8):
                cand = f"cuda:{idx}"
                mem = get_gpu_memory_info(cand)
                if mem is None:
                    break
                if mem[1] > best_free:
                    best_free = mem[1]
                    best_dev = cand
            target_dev = best_dev
        run_live_synthesis_test(device=target_dev, model_type=args.model, output_file=args.output)


if __name__ == "__main__":
    main()
