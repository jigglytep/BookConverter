#!/usr/bin/env python3
"""Unit tests for status_display.py (ETA calculation, plain-mode formatters, and display lifecycle)."""

from collections import deque
import io
import logging
from pathlib import Path
import tempfile
import unittest
import warnings

import status_display as sd


class TestETACalculation(unittest.TestCase):
    """Tests for pure rolling-window ETA and throughput functions."""

    def test_insufficient_samples_returns_none(self) -> None:
        samples: deque = deque()
        self.assertIsNone(sd.compute_eta(samples, remaining_units=100))

        samples.append((10.0, 0.0))
        self.assertIsNone(sd.compute_eta(samples, remaining_units=100))

        samples.append((11.0, 5.0))
        # Default min_samples=3, so 2 samples should still return None ("calculating...")
        self.assertIsNone(sd.compute_eta(samples, remaining_units=100, min_samples=3))
        self.assertEqual(sd.format_duration(sd.compute_eta(samples, remaining_units=100)), "calculating...")

    def test_linear_throughput_eta(self) -> None:
        # 2 units/sec steady rate: t=0->0, t=5->10, t=10->20
        samples = deque([(0.0, 0.0), (5.0, 10.0), (10.0, 20.0)])
        rate = sd.compute_rolling_rate(samples, min_samples=3, window_sec=60.0)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate or 0.0, 2.0, places=5)

        eta = sd.compute_eta(samples, remaining_units=80, min_samples=3, window_sec=60.0)
        self.assertIsNotNone(eta)
        self.assertAlmostEqual(eta or 0.0, 40.0, places=5)

    def test_rolling_window_ignores_stale_samples(self) -> None:
        # Early slow rate (0.5 units/s from t=0..20), then fast rate (5 units/s from t=100..120)
        samples = deque([
            (0.0, 0.0),
            (10.0, 5.0),
            (20.0, 10.0),
            (100.0, 50.0),
            (110.0, 100.0),
            (120.0, 150.0),
        ])
        # With window_sec=30.0, only samples at t=100, 110, 120 are inside [90, 120] -> rate = 100 / 20 = 5.0/s
        rate = sd.compute_rolling_rate(samples, min_samples=3, window_sec=30.0)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate or 0.0, 5.0, places=5)

        eta = sd.compute_eta(samples, remaining_units=50, min_samples=3, window_sec=30.0)
        self.assertIsNotNone(eta)
        self.assertAlmostEqual(eta or 0.0, 10.0, places=5)

    def test_zero_remaining_units(self) -> None:
        samples = deque([(0.0, 10.0)])
        self.assertEqual(sd.compute_eta(samples, remaining_units=0), 0.0)
        self.assertIsNone(sd.compute_eta(deque(), remaining_units=0))

    def test_zero_progress_or_tiny_dt_returns_none(self) -> None:
        # Stalled units
        stalled = deque([(0.0, 5.0), (5.0, 5.0), (10.0, 5.0)])
        self.assertIsNone(sd.compute_eta(stalled, remaining_units=20))

        # Tiny dt below min_elapsed
        burst = deque([(1.00, 1.0), (1.01, 2.0), (1.02, 3.0)])
        self.assertIsNone(sd.compute_eta(burst, remaining_units=20, min_elapsed=0.25))


class TestPlainModeFormatter(unittest.TestCase):
    """Tests for plain-mode event and heartbeat formatters."""

    def test_format_plain_event(self) -> None:
        line = sd.format_plain_event(
            timestamp="2026-10-29 17:30:00",
            level="info",
            message="Synthesizing chunk 42\nwith extra newline",
            book="Practical Electronics",
            stage="3b. TTS Synthesis (Chatterbox)",
        )
        self.assertEqual(
            line,
            "[2026-10-29 17:30:00] [INFO ] [Practical Electronics | 3b. TTS Synthesis (Chatterbox)] Synthesizing chunk 42 with extra newline",
        )

    def test_format_plain_heartbeat_with_gpus_and_calculating_eta(self) -> None:
        gpus = [
            sd.GPUStats(
                index=0,
                name="Tesla P100-PCIE-16GB",
                mem_used_mib=11200.0,
                mem_total_mib=16384.0,
                temp_c=67.0,
                util_pct=94.0,
                available=True,
            ),
            sd.GPUStats(
                index=1,
                name="Tesla P100-PCIE-16GB",
                mem_used_mib=10800.0,
                mem_total_mib=16384.0,
                temp_c=64.0,
                util_pct=88.0,
                available=True,
            ),
        ]
        hb = sd.format_plain_heartbeat(
            timestamp="2026-10-29 17:30:30",
            book_name="Bayesian_Stats",
            book_pos="2 of 5",
            stage_name="3b. TTS Synthesis (Chatterbox)",
            completed=150,
            total=600,
            eta_sec=None,
            book_eta_sec=None,
            elapsed_sec=65.0,
            gpus=gpus,
            rtf=2.45,
        )
        self.assertIn("[2026-10-29 17:30:30] [HEARTBEAT] Book (2 of 5): Bayesian_Stats", hb)
        self.assertIn("Stage: 3b. TTS Synthesis (Chatterbox) 150/600 (25.0%)", hb)
        self.assertIn("Elapsed: 1m 05s", hb)
        self.assertIn("Stage ETA: calculating...", hb)
        self.assertIn("Book ETA: calculating...", hb)
        self.assertIn("RTF 2.45x", hb)
        self.assertIn("GPU0 11200/16384MiB (68%) 67°C", hb)
        self.assertIn("GPU1 10800/16384MiB (66%) 64°C", hb)

    def test_format_plain_heartbeat_unavailable_gpu(self) -> None:
        hb = sd.format_plain_heartbeat(
            timestamp="2026-10-29 17:31:00",
            book_name="MyBook",
            book_pos="1 of 1",
            stage_name="1. PDF Extract (MinerU)",
            completed=10,
            total=100,
            eta_sec=90.0,
            book_eta_sec=420.0,
            elapsed_sec=10.0,
            gpus=[sd.GPUStats(available=False)],
        )
        self.assertIn("Stage ETA: 1m 30s", hb)
        self.assertIn("Book ETA: 7m 00s", hb)
        self.assertIn("GPU n/a", hb)

    def test_decode_throttle_bitmask(self) -> None:
        self.assertEqual(sd.decode_throttle_bitmask("0x0000000000000000"), "OK")
        self.assertEqual(sd.decode_throttle_bitmask("0x0000000000000001"), "IDLE")
        self.assertEqual(sd.decode_throttle_bitmask("0x0000000000000004"), "PWR_CAP")
        self.assertEqual(sd.decode_throttle_bitmask("0x0000000000000060"), "HW_THERM,SW_THERM")


class TestStatusDisplayLifecycle(unittest.TestCase):
    """Tests for StatusDisplay lifecycle, log file teeing, and stream capture."""

    def test_plain_lifecycle_and_log_file_tee(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = io.StringIO()
            disp = sd.StatusDisplay(
                log_dir=tmpdir,
                output_dir=tmpdir,
                plain=True,
                demo_mode=True,
                stream=buf,
            )
            disp.start_run(["Book_A.pdf", "Book_B.pdf"], hook_streams=False)
            disp.start_book("Book_A.pdf")
            disp.start_stage("tts_synth", total_units=10, unit_name="chunks")
            for _ in range(5):
                disp.advance(1, audio_sec=3.5, words=20)
            disp.log("WARN", "Sample warning message")
            disp.end_stage("done")
            disp.end_book({"name": "Book_A", "status": "done", "output_path": "out/Book_A.m4b"})
            disp.end_run()

            output_text = buf.getvalue()
            self.assertIn("Starting book (1 of 2): Book_A", output_text)
            self.assertIn("[WARN ]", output_text)
            self.assertIn("Sample warning message", output_text)
            self.assertIn("[BOOK-SUMMARY]", output_text)
            self.assertIn("PIPELINE RUN SUMMARY", output_text)
            self.assertTrue(disp.log_file_path.is_file())
            file_content = disp.log_file_path.read_text(encoding="utf-8")
            self.assertIn("Sample warning message", file_content)
            self.assertIn("Books Done     : 1", file_content)


if __name__ == "__main__":
    unittest.main()
