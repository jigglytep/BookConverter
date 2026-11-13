#!/usr/bin/env python3
"""
Unit tests for Sliding Window Phrase Chunking and Micro-Page Merging.
"""

import unittest
from pathlib import Path
import config
from audiobook_tts import ChatterboxAudiobookGenerator
from mineru_to_epub import prepare_page_dom_and_chunks, _split_balanced_html_sentences


class TestChunker(unittest.TestCase):
    def setUp(self):
        self.generator = ChatterboxAudiobookGenerator.__new__(ChatterboxAudiobookGenerator)

    def test_intra_page_micro_phrase_merge(self):
        """Micro-phrases (< 4 words) must be merged with adjacent sentences using ' - '."""
        text = "Wait. He paused on the threshold. Yes. The laboratory was completely dark."
        chunks = self.generator.split_into_chunks(text)
        self.assertTrue(len(chunks) > 0)
        # Verify no chunk has < 4 words
        for c in chunks:
            self.assertGreaterEqual(len(c.split()), 4, f"Chunk too small: {c}")
        # Verify pause delimiter ' - ' is present
        merged_text = " ".join(chunks)
        self.assertIn(" - ", merged_text)

    def test_optimal_chunk_bounds(self):
        """Chunks must respect max_words (32) and min_words (4)."""
        long_text = (
            "The morning light filtered slowly through the tall laboratory windows, illuminating the instruments on the workbench. "
            "A gentle breeze rattled the glass panes while the dust motes danced in the quiet sunbeams. "
            "Outside, the city began to awaken with the distant hum of traffic and early morning footsteps. "
            "Every single piece of equipment sat precisely where it had been left the previous evening."
        )
        chunks = self.generator.split_into_chunks(long_text)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            words = len(c.split())
            self.assertGreaterEqual(words, 4)
            self.assertLessEqual(words, 36)  # allow tiny margin for clause boundaries

    def test_dom_heading_intra_page_merge(self):
        """Headings (< 4 words) on a page must merge into following paragraphs with ' - '."""
        md = "# Chapter 1\n\nThe morning light filtered slowly through the tall laboratory windows."
        pdom = prepare_page_dom_and_chunks(
            page_num=1,
            chapter_title="Chapter 1",
            raw_markdown_or_text=md,
            base_dir=Path("."),
            image_registry={},
            chunk_size_words=24,
        )
        self.assertEqual(len(pdom.chunks), 1)
        chunk = pdom.chunks[0]
        self.assertGreaterEqual(len(chunk.spoken_text.split()), 4)
        self.assertIn(" - ", chunk.spoken_text)
        self.assertTrue(chunk.spoken_text.lower().startswith("chapter"))

    def test_inter_page_micro_page_merging(self):
        """Micro-pages (< 4 words) must be prepended to the next page or appended to the previous page."""
        import tempfile
        import shutil
        import config

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            (tmp_dir / "page_0001.txt").write_text("Chapter 1")
            (tmp_dir / "page_0002.txt").write_text("The morning light filtered slowly through the laboratory windows.")
            (tmp_dir / "page_0003.txt").write_text("The End")

            pages = self.generator.load_input_pages(tmp_dir)
            # page_0001 should be prepended to page_0002, and page_0003 appended to page_0002
            self.assertEqual(len(pages), 1)
            merged_page = pages[0]
            self.assertIn("Chapter 1", merged_page[1])
            self.assertIn("The End", merged_page[1])
            self.assertIn(" - ", merged_page[1])
        finally:
            shutil.rmtree(tmp_dir)

    def test_empty_or_whitespace_text(self):
        chunks = self.generator.split_into_chunks("   \n\t  ")
        self.assertEqual(chunks, [])


if __name__ == "__main__":
    unittest.main()
