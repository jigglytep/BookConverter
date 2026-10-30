#!/usr/bin/env python3
"""
Verification & Dry-Run Test Suite for Synchronized EPUB3 (SMIL 3.0 Media Overlays) + M4B
========================================================================================
Tests a 3-page sample from 'The Lazy Dungeon Master' (pages 3, 4, and 12):
- Page 3 matches TOC entry -> 'Page 3: Contents'
- Page 4 matches TOC entry -> 'Page 4: I. About This Book'
- Page 12 has no TOC entry -> 'Page 12' (plus inline LaTeX math to verify visual MathML vs spoken TTS)

Verifies:
1. EPUB3 zip packaging ('mimetype' first, ZIP_STORED, META-INF/container.xml, OEBPS/).
2. Consistent chapter naming across <nav epub:type="toc">, <nav epub:type="page-list">, and M4B FFMETADATA chapters.
3. 1-to-1 alignment between XHTML <span id="p{page}_s{seq}"> IDs and SMIL <par><text src="..."/></par> targets.
4. Sample-accurate clipBegin/clipEnd timestamps from raw PCM samples + silence gap padding.
5. content.opf media:duration metadata, manifest media-overlay attributes, and ebook.css active class styling.
6. Dual M4B muxing reuse via ffprobe chapter inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config
from audiobook_tts import ChatterboxAudiobookGenerator
from mineru_to_epub import (
    DEFAULT_CSS,
    _default_md_tts_opts,
    assemble_page_audio_and_smil,
    extract_page_number,
    extract_toc_mapping,
    format_page_chapter_title,
    format_smil_timestamp,
    gather_page_documents,
    package_epub3,
    prepare_page_dom_and_chunks,
    read_file_text,
)


def _synthesize_deterministic_test_wav(path: Path, spoken_text: str, sr: int = 24000) -> int:
    """Create a deterministic 16-bit mono PCM WAV whose sample count is proportional to word count."""
    words = max(1, len(spoken_text.split()))
    # Use deterministic non-integer duration per word (e.g. 317ms/word) to test millisecond rounding & zero drift
    n_samples = int(words * 0.317 * sr)
    t = np.arange(n_samples, dtype=np.float32) / sr
    tone = (0.15 * np.sin(2.0 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(tone.tobytes())
    return n_samples


def run_verification() -> int:
    book_base = "The Lazy Dungeon Master (Michael E Shea) (z-library.sk, 1lib.sk, z-lib.sk)"
    src_md_dir = SCRIPT_DIR.parent / "TTS_Script" / "completed" / f"{book_base}_md"
    src_pdf = SCRIPT_DIR.parent / "mineru" / "completed" / f"{book_base}.pdf"

    assert src_md_dir.is_dir(), f"Source markdown directory not found: {src_md_dir}"
    assert src_pdf.is_file(), f"Companion PDF not found: {src_pdf}"

    with tempfile.TemporaryDirectory(prefix="epub3_smil_verify_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        sample_md_dir = tmp_root / f"{book_base}_md"
        sample_md_dir.mkdir(parents=True, exist_ok=True)
        if (src_md_dir / "images").is_dir():
            shutil.copytree(src_md_dir / "images", sample_md_dir / "images")

        # Copy 3 sample pages: page 3 (TOC match), page 4 (TOC match), page 12 (no TOC match)
        selected_pages = [3, 4, 12]
        for pnum in selected_pages:
            src_page = src_md_dir / f"{book_base}_page_{pnum:04d}.md"
            content = src_page.read_text(encoding="utf-8")
            if pnum == 12:
                # Append a sentence with inline LaTeX math to test visual MathML vs spoken TTS string
                content += "\n\nThe arcane energy scales according to $E = m c^2$ across 3 dungeon levels.\n"
            (sample_md_dir / src_page.name).write_text(content, encoding="utf-8")

        # 1. Extract TOC mapping from companion PDF
        toc_map = extract_toc_mapping(sample_md_dir, md_dir=sample_md_dir, pdf_path=src_pdf)
        expected_titles = {
            3: "Page 3: Contents / About the Author",
            4: "Page 4: I. About This Book",
            12: "Page 12",
        }
        for pnum, exp_title in expected_titles.items():
            actual = format_page_chapter_title(pnum, toc_map)
            assert actual == exp_title, f"Page {pnum} title mismatch: expected {exp_title!r}, got {actual!r}"

        print("✓ [1/6] TOC extraction & chapter title formatting verified:")
        for pnum in selected_pages:
            print(f"        Page {pnum:2d} -> {format_page_chapter_title(pnum, toc_map)}")

        # 2. DOM wrapping & speech chunk preparation
        doc_files = gather_page_documents(sample_md_dir)
        image_registry = {}
        page_doms = []
        md_opts = _default_md_tts_opts()
        md_stats = {"equations": 0, "converted": 0, "errors": 0}

        for idx, doc_file in enumerate(doc_files, 1):
            pnum = extract_page_number(doc_file, idx)
            chap_title = format_page_chapter_title(pnum, toc_map)
            pdom = prepare_page_dom_and_chunks(
                page_num=pnum,
                chapter_title=chap_title,
                raw_markdown_or_text=read_file_text(doc_file),
                base_dir=doc_file.parent,
                image_registry=image_registry,
                is_plain_text=False,
                language="en",
                chunk_size_words=config.CHUNK_SIZE_WORDS,
                md_opts=md_opts,
                md_stats=md_stats,
            )
            page_doms.append(pdom)

        # Verify Page 12 has MathML inside its DOM span while spoken_text is verbalized
        p12_dom = [p for p in page_doms if p.page_num == 12][0]
        assert p12_dom.has_mathml, "Expected Page 12 to have MathML from inline LaTeX"
        assert "<math" in p12_dom.xhtml_content, "Expected <math> element inside Page 12 XHTML"
        last_chunk = p12_dom.chunks[-1]
        assert "$" not in last_chunk.spoken_text, f"Expected LaTeX delimiters stripped in spoken_text: {last_chunk.spoken_text}"
        print(f"✓ [2/6] DOM wrapping & math verbalization verified ({sum(len(p.chunks) for p in page_doms)} total spans)")
        print(f"        Page 12 final chunk ID: {last_chunk.chunk_id}")
        print(f"        Spoken text passed to TTS: {last_chunk.spoken_text!r}")

        # 3. Generate deterministic raw PCM WAV chunks & assemble per-page CBR MP3 + SMIL 3.0
        chunks_dir = tmp_root / "chunks"
        page_audio_dir = chunks_dir / "page_audio"
        sr = config.SAMPLE_RATE
        gap_ms = config.CHUNK_GAP_MS
        gap_sec = int(round(sr * (gap_ms / 1000.0))) / float(sr)

        page_timings_map = {}
        page_timings_list = []
        global_idx = 1
        book_clock = 0.0

        for pdom in page_doms:
            c_wavs = []
            expected_par = []
            expected_page_clock = 0.0
            for sc in pdom.chunks:
                wpath = chunks_dir / f"chunk_{global_idx:04d}.wav"
                global_idx += 1
                nsamp = _synthesize_deterministic_test_wav(wpath, sc.spoken_text, sr=sr)
                dur = nsamp / float(sr)
                expected_par.append(
                    (
                        sc.chunk_id,
                        format_smil_timestamp(expected_page_clock),
                        format_smil_timestamp(expected_page_clock + dur),
                    )
                )
                expected_page_clock += dur + gap_sec
                c_wavs.append(wpath)

            mp3_path = page_audio_dir / f"page_{pdom.num_str}.mp3"
            timing = assemble_page_audio_and_smil(
                page_dom=pdom,
                chunk_wav_paths=c_wavs,
                out_mp3_path=mp3_path,
                sample_rate=sr,
                chunk_gap_ms=gap_ms,
                book_start_sec=book_clock,
                mp3_bitrate="128k",
            )
            assert timing.par_entries == expected_par, f"SMIL <par> timing mismatch on page {pdom.page_num}"
            assert abs(timing.duration_sec - expected_page_clock) < 1e-9
            assert mp3_path.is_file() and mp3_path.stat().st_size > 0
            page_timings_map[pdom.page_num] = timing
            page_timings_list.append(timing)
            book_clock += timing.duration_sec

        print("✓ [3/6] Sample-accurate PCM timing & single-pass per-page CBR MP3 encoding verified")

        # 4. Package EPUB3 & verify ZIP layout, XHTML span IDs, SMIL XML, nav.xhtml, and content.opf
        epub_path = tmp_root / "sample_verified.epub"
        package_epub3(
            output_epub_path=epub_path,
            title="The Lazy Dungeon Master",
            author="Michael E. Shea",
            language="en",
            identifier=f"urn:uuid:{uuid.uuid4()}",
            publisher=None,
            css_content=DEFAULT_CSS,
            page_doms=page_doms,
            page_timings=page_timings_map,
            image_registry=image_registry,
        )

        with zipfile.ZipFile(epub_path, "r") as zf:
            names = zf.namelist()
            assert names[0] == "mimetype", f"First file in EPUB must be 'mimetype', got {names[0]!r}"
            m_info = zf.getinfo("mimetype")
            assert m_info.compress_type == zipfile.ZIP_STORED, "mimetype must be uncompressed (ZIP_STORED)"
            assert zf.read("mimetype") == b"application/epub+zip"
            assert "META-INF/container.xml" in names
            assert "OEBPS/content.opf" in names
            assert "OEBPS/nav.xhtml" in names
            assert "OEBPS/styles/ebook.css" in names

            css_txt = zf.read("OEBPS/styles/ebook.css").decode("utf-8")
            assert ".-epub-media-overlay-active" in css_txt
            assert "@media (prefers-color-scheme: dark)" in css_txt

            # Verify nav.xhtml toc and page-list
            nav_soup = BeautifulSoup(zf.read("OEBPS/nav.xhtml").decode("utf-8"), "html.parser")
            toc_nav = nav_soup.find("nav", attrs={"epub:type": "toc"})
            pagelist_nav = nav_soup.find("nav", attrs={"epub:type": "page-list"})
            assert toc_nav is not None and pagelist_nav is not None
            toc_titles = [a.get_text() for a in toc_nav.find_all("a")]
            pagelist_titles = [a.get_text() for a in pagelist_nav.find_all("a")]
            assert toc_titles == list(expected_titles.values()), f"TOC nav mismatch: {toc_titles}"
            assert pagelist_titles == list(expected_titles.values()), f"page-list nav mismatch: {pagelist_titles}"

            # Verify content.opf media overlay metadata & manifest
            opf_txt = zf.read("OEBPS/content.opf").decode("utf-8")
            assert 'prefix="media: http://www.idpf.org/epub/vocab/overlays/#"' in opf_txt
            assert '<meta property="media:active-class">-epub-media-overlay-active</meta>' in opf_txt
            total_dur_str = format_smil_timestamp(sum(t.duration_sec for t in page_timings_list))
            assert f'<meta property="media:duration">{total_dur_str}</meta>' in opf_txt

            # Verify every page's XHTML span IDs match SMIL <par> targets 1-to-1
            for pdom in page_doms:
                t = page_timings_map[pdom.page_num]
                page_dur_str = format_smil_timestamp(t.duration_sec)
                assert (
                    f'<meta property="media:duration" refines="#smil_page_{pdom.num_str}">{page_dur_str}</meta>'
                    in opf_txt
                )
                assert f'media-overlay="smil_page_{pdom.num_str}"' in opf_txt

                xhtml_bytes = zf.read(f"OEBPS/{pdom.xhtml_filename}")
                smil_bytes = zf.read(f"OEBPS/{pdom.smil_filename}")
                assert f"OEBPS/{pdom.audio_filename}" in names

                page_soup = BeautifulSoup(xhtml_bytes.decode("utf-8"), "html.parser")
                xhtml_span_ids = [
                    sp.get("id")
                    for sp in page_soup.find_all("span")
                    if sp.get("id", "").startswith(f"p{pdom.page_num:03d}_s")
                ]

                smil_root = ET.fromstring(smil_bytes.decode("utf-8"))
                ns = {"smil": "http://www.w3.org/ns/SMIL"}
                par_elems = smil_root.findall(".//smil:par", ns)
                smil_text_refs = []
                for par in par_elems:
                    txt_el = par.find("smil:text", ns)
                    aud_el = par.find("smil:audio", ns)
                    assert txt_el is not None and aud_el is not None
                    src_attr = txt_el.attrib["src"]
                    expected_prefix = f"../text/page_{pdom.num_str}.xhtml#"
                    assert src_attr.startswith(expected_prefix), f"Unexpected SMIL text src: {src_attr}"
                    smil_text_refs.append(src_attr.split("#", 1)[1])
                    assert aud_el.attrib["src"] == f"../audio/page_{pdom.num_str}.mp3"

                assert xhtml_span_ids == smil_text_refs, (
                    f"Page {pdom.page_num} span IDs {xhtml_span_ids} != SMIL refs {smil_text_refs}"
                )

        print(f"✓ [4/6] EPUB3 archive structure, content.opf durations ({total_dur_str}), and XHTML<->SMIL IDs verified")

        # 5. Mux standalone .m4b from the per-page MP3 tracks & verify chapter titles/timings with ffprobe
        m4b_path = tmp_root / "sample_verified.m4b"
        gen = ChatterboxAudiobookGenerator(devices=["cpu"])
        gen._export_from_page_tracks(
            page_timings=page_timings_list,
            output_path=m4b_path,
            book_title="The Lazy Dungeon Master",
            chunks_dir=chunks_dir,
            author="Michael E. Shea",
        )
        assert m4b_path.is_file() and m4b_path.stat().st_size > 0

        ffprobe_res = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_chapters", str(m4b_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        m4b_chapters = json.loads(ffprobe_res.stdout).get("chapters", [])
        assert len(m4b_chapters) == len(page_doms), f"Expected {len(page_doms)} M4B chapters, got {len(m4b_chapters)}"

        m4b_titles = [ch.get("tags", {}).get("title") for ch in m4b_chapters]
        assert m4b_titles == list(expected_titles.values()), (
            f"M4B chapter titles {m4b_titles} != expected {list(expected_titles.values())}"
        )

        for ch, timing in zip(m4b_chapters, page_timings_list):
            assert int(ch["start"]) == timing.start_ms, f"M4B chapter start {ch['start']} != {timing.start_ms}"
            assert int(ch["end"]) == timing.end_ms, f"M4B chapter end {ch['end']} != {timing.end_ms}"

        print("✓ [5/6] M4B chapter markers & cumulative timestamps verified via ffprobe:")
        for ch in m4b_chapters:
            print(
                f"        [{int(ch['start']):6d}ms -> {int(ch['end']):6d}ms] title={ch['tags']['title']!r}"
            )

        print("✓ [6/6] All EPUB3 + SMIL 3.0 Media Overlay + M4B checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(run_verification())
