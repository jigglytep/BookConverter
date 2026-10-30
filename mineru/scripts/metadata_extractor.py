#!/usr/bin/env python3
"""
metadata_extractor.py
=====================
Stage 1 bibliographic metadata and cover image extractor for BookConverters.

Collects metadata deterministically using:
  1. Filename heuristics (stripping z-library/watermarked suffixes, parsing Author - Title - Year,
     Title (Author), Series, and CamelCase/snake_case stems).
  2. Embedded PDF metadata via `pdfinfo` (Title, Author, Subject, Keywords, CreationDate).
  3. Front-matter Markdown/Text inspection across the first 12 pages (ISBN, Copyright Year,
     Publisher, Authors from copyright/CIP/title page, Subtitle, Series, Genres, and Description).
  4. Cover image rendering from Page 1 of the PDF via `pdftoppm` (`cover.jpg`).

Writes `<md_dir>/book_metadata.json` and `<md_dir>/cover.jpg`, which are forwarded through
Stage 2 (`TTS_Script`) and finalized in Stage 3 (`chatterbox`) with exact chapter timings
and narrator information into AudioBookshelf's `metadata.json` schema.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


KNOWN_PUBLISHERS = [
    ("O'Reilly Media", [r"\bO['’]Reilly\s+Media\b", r"\bO['’]REILLY\b", r"@oreillymedia\b"]),
    ("CRC Press", [r"\bCRC\s+Press\b", r"\bCHAPMAN\s*&\s*HALL/CRC\b"]),
    ("McGraw-Hill Education", [r"\bMcGraw-Hill\s+Education\b"]),
    ("McGraw-Hill", [r"\bMcGraw-Hill\b"]),
    ("Springer", [r"\bSpringer(?:\s+Nature|\s+Verlag)?\b"]),
    ("Pearson", [r"\bPearson(?:\s+Education)?\b"]),
    ("Addison-Wesley", [r"\bAddison-Wesley\b"]),
    ("John Wiley & Sons", [r"\bJohn\s+Wiley\s*&\s*Sons\b", r"\bWiley\b"]),
    ("No Starch Press", [r"\bNo\s+Starch\s+Press\b"]),
    ("Manning Publications", [r"\bManning\s+Publications\b"]),
    ("Packt Publishing", [r"\bPackt\s+Publishing\b"]),
    ("Apress", [r"\bApress\b"]),
    ("MIT Press", [r"\bMIT\s+Press\b"]),
    ("Cambridge University Press", [r"\bCambridge\s+University\s+Press\b"]),
    ("Oxford University Press", [r"\bOxford\s+University\s+Press\b"]),
    ("Princeton University Press", [r"\bPrinceton\s+University\s+Press\b"]),
    ("Wizards of the Coast", [r"\bWizards\s+of\s+the\s+Coast\b", r"\bWizards\s+RPG\s+Team\b"]),
    ("Ghostfire Gaming", [r"\bGhostfire\s+Gaming\b"]),
    ("Sly Flourish", [r"\bSly\s+Flourish\b"]),
    ("Elsevier", [r"\bElsevier\b"]),
    ("Academic Press", [r"\bAcademic\s+Press\b"]),
    ("Routledge", [r"\bRoutledge\b"]),
    ("Taylor & Francis", [r"\bTaylor\s*&\s*Francis\b"]),
    ("Bibliotex", [r"\bBIBLIOTEX\b", r"\bBibliotex\b"]),
]

CORPORATE_HOLDER_WORDS = (
    "press", "publishing", "publishers", "publisher", "education", "media",
    "company", "companies", "inc", "llc", "ltd", "gmbh", "group", "corporation",
    "wizards of the coast", "mcgraw-hill", "springer", "pearson", "wiley",
    "o'reilly", "taylor & francis", "elsevier", "bibliotex", "routledge",
    "all rights reserved", "printed in", "author", "authors",
)

MINOR_TITLE_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into",
    "nor", "of", "on", "onto", "or", "over", "per", "so", "than", "that",
    "the", "to", "up", "via", "with", "yet",
}

COMMON_TYPO_FIXES = {
    "Fundametals": "Fundamentals",
    "Inventrs": "Inventors",
    "INVENTRS": "INVENTORS",
}


def _natural_sort_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _smart_title_case(s: str) -> str:
    """Convert ALL-CAPS or messy title strings into clean title case while preserving acronyms."""
    words = s.split()
    if not words:
        return s
    acronyms = {"AI", "ML", "NLP", "R", "C", "C++", "3D", "5E", "DND", "D&D", "RPG", "DC", "AC", "IC", "LED", "SMD"}
    result = []
    for i, w in enumerate(words):
        clean_w = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", w)
        if clean_w.upper() in acronyms:
            result.append(w.upper())
        elif w.isupper() and len(clean_w) > 1:
            low = w.lower()
            if i > 0 and i < len(words) - 1 and low in MINOR_TITLE_WORDS:
                result.append(low)
            else:
                result.append(low.capitalize())
        else:
            result.append(w)
    return " ".join(result)


def _fix_common_typos(s: str) -> str:
    for bad, good in COMMON_TYPO_FIXES.items():
        s = re.sub(rf"\b{re.escape(bad)}\b", good, s)
    return s


def _humanize_identifier(stem: str) -> str:
    """Convert CamelCase or snake_case identifier into spaced words."""
    s = re.sub(r"^\d+[\._\-]+", "", stem)
    # Replace underscores with spaces if mostly snake_case
    if "_" in s and " " not in s:
        s = s.replace("_", " ")
    # Split CamelCase (e.g. PracticalElectronicsForInventors -> Practical Electronics For Inventors)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _fix_common_typos(s)
    # Lowercase minor words in the middle
    words = s.split()
    if len(words) > 2:
        fixed = [words[0]]
        for w in words[1:-1]:
            if w.lower() in MINOR_TITLE_WORDS:
                fixed.append(w.lower())
            else:
                fixed.append(w)
        fixed.append(words[-1])
        s = " ".join(fixed)
    return s


def _clean_author_name(raw: str) -> List[str]:
    """Normalize author strings (including 'Last, First' or 'Last  First') into clean names."""
    s = raw.strip().strip(";,. ")
    if not s:
        return []
    # Remove academic/affiliation suffixes or degrees
    s = re.sub(r"\b(?:Ph\.?D\.?|M\.?D\.?|Jr\.?|Sr\.?)\b", "", s).strip()
    # Split multiple authors separated by ';', '|', ' and ', '&'
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
    elif "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
    elif "," in s and " and " in s:
        s_norm = s.replace(" and ", ", ")
        parts = [p.strip() for p in s_norm.split(",") if p.strip()]
    elif "," in s:
        subparts = [p.strip() for p in s.split(",") if p.strip()]
        # Check if 'Last, First' (2 parts, each 1-2 words, no 'and')
        if len(subparts) == 2 and all(len(sp.split()) <= 2 for sp in subparts):
            parts = [f"{subparts[1]} {subparts[0]}"]
        else:
            parts = subparts
    elif " and " in s:
        parts = [p.strip() for p in re.split(r"\s+and\s+", s) if p.strip()]
    else:
        # Check if filename style 'Last  First' (double space between Last and First)
        if "  " in s:
            dbl = [p.strip() for p in re.split(r"\s{2,}", s) if p.strip()]
            if len(dbl) == 2 and len(dbl[0].split()) == 1 and len(dbl[1].split()) == 1:
                parts = [f"{dbl[1]} {dbl[0]}"]
            else:
                parts = [" ".join(s.split())]
        else:
            parts = [" ".join(s.split())]

    cleaned: List[str] = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip(" ,.;:-")
        if not p or len(p) < 3:
            continue
        # Title-case if all lowercase or all uppercase
        if p.isupper() or p.islower():
            p = " ".join(w.capitalize() for w in p.split())
        if p not in cleaned:
            cleaned.append(p)
    return cleaned


def canonical_book_stem(path_or_name: Path | str) -> str:
    """Return canonical book stem without _md, _txt, _pages, or file extensions."""
    p = Path(path_or_name)
    name = p.stem if p.is_file() or p.suffix.lower() in (".pdf", ".epub", ".m4b", ".mp3", ".json") else p.name
    for suffix in (".metadata", "_txt", "_md", "_pages"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    m = re.match(r"^(.+)_page_\d+$", name, re.IGNORECASE)
    if m:
        name = m.group(1)
    return name


def parse_filename_metadata(stem: str) -> Dict[str, Any]:
    """Extract candidate title, authors, series, and publishedYear from a PDF filename stem."""
    s = canonical_book_stem(stem)

    # Strip z-library / anna's archive / watermarked noise
    s = re.sub(r"\s*\((?:z-library\.[^)]*|1lib\.[^)]*|z-lib\.[^)]*)\)\s*", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"_?digital_hi-res(?:_watermarked)?", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\d+[\._]\s*", "", s)
    s = s.strip()

    authors: List[str] = []
    series: List[str] = []
    published_year: Optional[str] = None
    title = s

    # Pattern 1: Author - Title (Series) - Year  OR  Author - Title - Year
    m_aty = re.match(r"^(?P<author>[^-]+?)\s+-\s+(?P<title>.+?)\s+-\s+(?P<year>(?:19|20)\d{2})\s*$", s)
    if m_aty:
        raw_author = m_aty.group("author").strip()
        raw_title = m_aty.group("title").strip()
        published_year = m_aty.group("year").strip()
        # Check if raw_title has trailing (Series)
        m_ser = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", raw_title)
        if m_ser:
            raw_title = m_ser.group(1).strip()
            series.append(m_ser.group(2).strip())
        # Check if raw_author looks like a person's name (2 words 'Last First' or 'Last  First')
        if "  " in raw_author:
            authors = _clean_author_name(raw_author)
        else:
            awords = raw_author.split()
            if len(awords) == 2 and "," not in raw_author:
                # In 'Klein Bernd - Numeric Python - 2026', 'Klein Bernd' is Last First
                authors = [f"{awords[1]} {awords[0]}"]
            else:
                authors = _clean_author_name(raw_author)
        title = raw_title
    else:
        # Pattern 2: Title (Author) at the end of the string
        m_ta = re.match(r"^(?P<title>.+?)\s*\((?P<author>[^()]+)\)\s*$", s)
        if m_ta:
            cand_title = m_ta.group("title").strip()
            cand_author = m_ta.group("author").strip()
            # Ensure cand_author is not an edition or format tag
            if not re.search(r"\b(?:edition|ed\.|vol\.|volume|pdf|ebook)\b", cand_author, re.IGNORECASE):
                title = cand_title
                authors = _clean_author_name(cand_author)
                # Check if title has "Series - BookTitle" (e.g. "Dungeons and Dragons DND 5E - Vecna Dossier")
                if " - " in title:
                    lhs, rhs = title.split(" - ", 1)
                    if lhs.strip() and rhs.strip():
                        series.append(lhs.strip())
                        title = rhs.strip()

    # If title still has no spaces (CamelCase or snake_case), humanize it
    if "_" in title or (" " not in title and any(c.isupper() for c in title[1:])):
        title = _humanize_identifier(title)
    else:
        title = _fix_common_typos(re.sub(r"\s+", " ", title).strip())

    return {
        "title": title,
        "authors": authors,
        "series": series,
        "publishedYear": published_year,
    }


def parse_pdfinfo(pdf_path: Optional[Path]) -> Dict[str, Any]:
    """Run `pdfinfo` on the source PDF to extract embedded metadata tags."""
    info: Dict[str, Any] = {
        "title": None,
        "authors": [],
        "subject": None,
        "keywords": [],
        "publishedYear": None,
        "pages": 0,
    }
    if not pdf_path or not pdf_path.is_file() or shutil.which("pdfinfo") is None:
        return info

    try:
        res = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if res.returncode != 0:
            return info

        fields: Dict[str, str] = {}
        for line in res.stdout.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip().lower()] = v.strip()

        raw_title = fields.get("title", "")
        if raw_title and len(raw_title) >= 3:
            # Filter out junk filenames or word processor titles
            junk_patterns = (
                r"^untitled",
                r"^microsoft word",
                r"\.(?:pdf|doc|docx|indd|ps|dvi|tex)$",
                r"^[0-9a-f\-_]{16,}$",
            )
            if not any(re.search(p, raw_title, re.IGNORECASE) for p in junk_patterns):
                info["title"] = _fix_common_typos(re.sub(r"\s+", " ", raw_title).strip())

        raw_author = fields.get("author", "")
        if raw_author and len(raw_author) >= 2:
            if not re.search(r"^(?:admin|user|unknown|default|owner)$", raw_author, re.IGNORECASE):
                info["authors"] = _clean_author_name(raw_author)

        raw_subj = fields.get("subject", "")
        if raw_subj and len(raw_subj) >= 3:
            info["subject"] = raw_subj.strip()

        raw_kw = fields.get("keywords", "")
        if raw_kw:
            kws = [k.strip() for k in re.split(r"[,;]", raw_kw) if k.strip()]
            info["keywords"] = kws

        if "pages" in fields and fields["pages"].isdigit():
            info["pages"] = int(fields["pages"])

        # Do not treat CreationDate as publishedYear unless no copyright year is found,
        # so store it separately as pdfCreationYear
        cdate = fields.get("creationdate", "")
        m_yr = re.search(r"\b(19\d{2}|20\d{2})\b", cdate)
        if m_yr:
            info["pdfCreationYear"] = m_yr.group(1)

    except Exception:
        pass

    return info


def _strip_md_formatting(text: str) -> str:
    s = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    return s


def scan_markdown_frontmatter(md_dir: Optional[Path], max_pages: int = 12) -> Dict[str, Any]:
    """
    Scan the first `max_pages` Markdown/Text files in `md_dir` to extract:
    - title, subtitle
    - authors (from copyright notice, CIP cataloging block, or title page)
    - isbn
    - publisher
    - publishedYear
    - series
    - genres
    - description
    """
    result: Dict[str, Any] = {
        "title": None,
        "subtitle": None,
        "authors": [],
        "isbn": None,
        "publisher": None,
        "publishedYear": None,
        "series": [],
        "genres": [],
        "description": None,
    }
    if not md_dir or not md_dir.is_dir():
        return result

    page_files = [
        f for f in md_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".md", ".markdown", ".txt") and f.name.lower() != "toc.json"
    ]
    page_files.sort(key=_natural_sort_key)
    front_files = page_files[:max_pages]
    if not front_files:
        return result

    page_texts: List[Tuple[int, str, str]] = []
    for idx, f in enumerate(front_files, 1):
        m_pg = re.search(r"page[_-]?(\d+)", f.stem, re.IGNORECASE)
        pnum = int(m_pg.group(1)) if m_pg else idx
        raw = _read_text(f)
        clean = _strip_md_formatting(raw)
        page_texts.append((pnum, raw, clean))

    combined_clean = "\n\n".join(c for _, _, c in page_texts)

    # 1. ISBN extraction (prefer 13-digit 978/979 ISBNs)
    isbn_candidates: List[str] = []
    for m_isbn in re.finditer(
        r"ISBN(?:-1[03])?\s*:?\s*((?:97[89][-\s]?)?[0-9]{1,5}[-\s]?[0-9]+[-\s]?[0-9]+[-\s]?[0-9Xx])\b",
        combined_clean,
        re.IGNORECASE,
    ):
        raw_code = m_isbn.group(1).strip()
        digits = re.sub(r"[^0-9Xx]", "", raw_code)
        if len(digits) in (10, 13):
            isbn_candidates.append(re.sub(r"\s+", "-", raw_code))
    if isbn_candidates:
        isbn_13 = [c for c in isbn_candidates if len(re.sub(r"[^0-9Xx]", "", c)) == 13]
        result["isbn"] = isbn_13[0] if isbn_13 else isbn_candidates[0]

    # 2. Published Year extraction from copyright / publication lines
    year_patterns = [
        r"First\s+edition\s+published\s+(20\d{2}|19\d{2})\b",
        r"Copyright\s*(?:©|\(c\))?\s*(20\d{2}|19\d{2})\b",
        r"©\s*(20\d{2}|19\d{2})\b",
        r"e-book\s+Edition\s+(20\d{2}|19\d{2})\b",
        r"Published\s+in\s+(20\d{2}|19\d{2})\b",
    ]
    for pat in year_patterns:
        m_yr = re.search(pat, combined_clean, re.IGNORECASE)
        if m_yr:
            result["publishedYear"] = m_yr.group(1)
            break

    # 3. Publisher extraction
    for pub_name, regex_list in KNOWN_PUBLISHERS:
        if any(re.search(rx, combined_clean, re.IGNORECASE) for rx in regex_list):
            result["publisher"] = pub_name
            break
    if not result["publisher"]:
        m_pub = re.search(
            r"(?:Published\s+by|First\s+edition\s+published\s+\d{4}\s+by)\s+([A-Z][A-Za-z0-9 &.,'’-]{2,45}?)(?:,|\.|;|\n|\s+\d)",
            combined_clean,
        )
        if m_pub:
            result["publisher"] = m_pub.group(1).strip()

    # 4. Series extraction
    m_series = re.search(
        r"\b([A-Z][A-Za-z0-9 &/\-–]{3,55}\s+(?:Book\s+Series|Texts\s+in\s+[A-Za-z0-9 &,\-]+\s+Series|Statistical\s+Science\s+Series|Computer\s+Science\s+Series|Mathematics\s+Series|Engineering\s+Series))\b",
        combined_clean,
    )
    if m_series:
        ser = re.sub(r"\s+", " ", m_series.group(1)).strip()
        result["series"] = [ser]

    # 5. Authors extraction from CIP / Copyright / Title pages
    # 5a. Library of Congress Cataloging-in-Publication Data: "Title / Author1, Author2. —"
    m_cip = re.search(
        r"Cataloging-in-Publication\s+Data[\s\S]{1,350}?/\s*(?:by\s+)?([A-Z][^\n/—\-]+?)\.\s*(?:—|--|-|\n|p\.\s*cm|ISBN)",
        combined_clean,
        re.IGNORECASE,
    )
    if m_cip:
        cip_authors = _clean_author_name(m_cip.group(1))
        if cip_authors:
            result["authors"] = cip_authors

    # 5b. Copyright line: "© 2026 Alan Agresti, Maria Kateri, Ranjini Grove, and Antonietta Mira"
    if not result["authors"]:
        for m_cop in re.finditer(
            r"(?:Copyright\s*)?©\s*(?:(?:19|20)\d{2}\s*,\s*)*(?:19|20)\d{2}\s*(?:by\s+)?([^\n.]{4,120})",
            combined_clean,
            re.IGNORECASE,
        ):
            holder = m_cop.group(1).strip()
            holder_low = holder.lower()
            if not any(cw in holder_low for cw in CORPORATE_HOLDER_WORDS):
                cand_authors = _clean_author_name(holder)
                if cand_authors and all(2 <= len(a.split()) <= 4 for a in cand_authors):
                    result["authors"] = cand_authors
                    break

    # 5c. Page 1 / Title page author line (e.g. "Paul Scherz and Simon Monk" or "Charles K. Alexander | Matthew N. O. Sadiku")
    if not result["authors"]:
        for pnum, raw, _ in page_texts[:4]:
            if pnum > 4:
                break
            for line in raw.splitlines():
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("!") or s.startswith("<"):
                    continue
                s = re.sub(r"\s+Foreword\s+by\s+.*$", "", s, flags=re.IGNORECASE).strip()
                if "|" in s or " and " in s:
                    cand = _clean_author_name(s)
                    if (
                        1 <= len(cand) <= 5
                        and all(2 <= len(a.split()) <= 4 for a in cand)
                        and not any(w in s.lower() for w in ("edition", "chapter", "prior", "likelihood", "series", "press", "university", "copyright", "table"))
                    ):
                        result["authors"] = cand
                        break
            if result["authors"]:
                break

    # 6. Title & Subtitle from H1 headings in front-matter
    h1_candidates: List[Tuple[int, str, Optional[str]]] = []
    for pnum, raw, _ in page_texts[:9]:
        lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.strip().startswith("<!--")]
        for i, line in enumerate(lines):
            if line.startswith("# "):
                h1_text = _strip_md_formatting(line[2:]).strip()
                h1_text = _fix_common_typos(re.sub(r"\s+", " ", h1_text))
                if not h1_text or len(h1_text) < 3:
                    continue
                if re.match(r"^(?:praise\s+for|contents|table\s+of\s+contents|preface|foreword|dedication|acknowledgments)\b", h1_text, re.IGNORECASE):
                    continue
                # Strip trailing edition from H1 if present (e.g. "Practical Electronics for Inventors Fourth Edition")
                h1_text = re.sub(
                    r"\s*,?\s*(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|\d+(?:st|nd|rd|th))\s+Edition\s*$",
                    "",
                    h1_text,
                    flags=re.IGNORECASE,
                ).strip()
                # Check if next line on Page 1 is a subtitle
                sub_cand = None
                if pnum <= 2 and i + 1 < len(lines):
                    nxt = lines[i + 1]
                    if not nxt.startswith(("#", "!", "<", "•", "-")) and 3 <= len(nxt.split()) <= 12:
                        if not re.search(r"\b(?:edition|by|copyright|isbn)\b", nxt, re.IGNORECASE):
                            # Check it's not the author line
                            if not (result["authors"] and any(a.split()[-1].lower() in nxt.lower() for a in result["authors"])):
                                sub_cand = _strip_md_formatting(nxt).strip()
                h1_candidates.append((pnum, h1_text, sub_cand))

    if h1_candidates:
        # Prefer mixed-case H1 over ALL-CAPS H1 if both exist
        mixed_case = [c for c in h1_candidates if not c[1].isupper()]
        chosen = mixed_case[0] if mixed_case else h1_candidates[0]
        raw_t = chosen[1]
        # Fix OCR "Al " -> "AI "
        raw_t = re.sub(r"\bAl\s+and\s+Machine\s+Learning\b", "AI and Machine Learning", raw_t)
        result["title"] = _smart_title_case(raw_t) if raw_t.isupper() else raw_t
        # Look for subtitle from Page 1 if available
        for _, _, sub_c in h1_candidates:
            if sub_c:
                # Avoid setting subtitle if it is already part of title
                if sub_c.lower() not in result["title"].lower():
                    result["subtitle"] = sub_c
                break

    # 7. Description extraction from front-matter overview / back-cover blurb (pages 1-4)
    desc_paragraphs: List[str] = []
    for pnum, raw, _ in page_texts[:4]:
        # Skip pages that are praise quotes or tables or copyright pages
        if re.search(r"(?:Praise\s+for|Cataloging-in-Publication|All\s+rights\s+reserved|<table\b)", raw, re.IGNORECASE):
            continue
        blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
        for b in blocks:
            if b.startswith(("#", "!", "<", "<table>", "•", "*", "—", "-")):
                continue
            clean_b = re.sub(r"\s+", " ", _strip_md_formatting(b)).strip()
            words = clean_b.split()
            if len(words) < 22 or len(words) > 160:
                continue
            if not re.search(
                r"\b(?:book|reader|learn|guide|introduction|overview|chapter|edition|students|course|programmers|engineers|designed|practical|textbook|methods)\b",
                clean_b,
                re.IGNORECASE,
            ):
                continue
            if any(
                phrase in clean_b.lower()
                for phrase in (
                    "all rights reserved",
                    "without written permission",
                    "copyright clearance center",
                    "trademark notice",
                    "for more information about this series",
                    "is grateful to",
                )
            ):
                continue
            desc_paragraphs.append(clean_b)
            if len(desc_paragraphs) >= 2:
                break
        if desc_paragraphs:
            break

    if desc_paragraphs:
        result["description"] = "\n\n".join(desc_paragraphs)

    # 8. Genres / Subjects
    # Check CIP subjects: "1. Electric circuits."
    m_loc_subj = re.search(
        r"ISBN\s+[0-9\-Xx]+[^\n]*\n+\s*1\.\s*([A-Za-z][A-Za-z0-9\s\-—]+?)\.",
        combined_clean,
    )
    if m_loc_subj:
        subj = m_loc_subj.group(1).split("—")[0].strip()
        if subj:
            result["genres"].append(subj)

    return result


def infer_genres_from_title(title: str, series: List[str], existing_genres: List[str]) -> List[str]:
    """Infer sensible broad genres from title/series keywords if not already populated."""
    genres = list(existing_genres)
    combined = f"{title} {' '.join(series)}".lower()
    keyword_map = [
        ("Artificial Intelligence & Machine Learning", ("machine learning", "artificial intelligence", "deep learning", "pytorch", "tensorflow", "ai ")),
        ("Mathematics & Statistics", ("mathematics", "math ", "statistics", "bayesian", "geometry", "calculus", "probability")),
        ("Programming & Computer Science", ("python", "coders", "programmer", "computer science", "algorithms", "coding")),
        ("Electrical & Electronics Engineering", ("electrical", "electronics", "circuits", "inventors", "engineering")),
        ("Finance & Economics", ("finance", "financial", "economics", "trading")),
        ("Tabletop RPG & Fantasy", ("dungeons", "dragons", "dnd", "5e", "dungeon master", "gamemaster", "rpg", "rulebook", "vecna", "planar", "campaign", "monsters")),
    ]
    for genre_label, kws in keyword_map:
        if any(k in combined for k in kws) and genre_label not in genres:
            genres.append(genre_label)
    return genres


def extract_pdf_cover(pdf_path: Optional[Path], output_cover_path: Path, dpi: int = 150) -> Optional[Path]:
    """Render Page 1 of `pdf_path` to `output_cover_path` (JPEG) using `pdftoppm`."""
    if output_cover_path.is_file() and output_cover_path.stat().st_size > 1024:
        return output_cover_path.resolve()
    if not pdf_path or not pdf_path.is_file() or shutil.which("pdftoppm") is None:
        return None

    try:
        output_cover_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = output_cover_path.with_suffix("")
        res = subprocess.run(
            [
                "pdftoppm",
                "-jpeg",
                "-r", str(dpi),
                "-f", "1",
                "-l", "1",
                "-singlefile",
                str(pdf_path),
                str(prefix),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        candidate = Path(f"{prefix}.jpg")
        if res.returncode == 0 and candidate.is_file() and candidate.stat().st_size > 1024:
            if candidate.resolve() != output_cover_path.resolve():
                shutil.move(str(candidate), str(output_cover_path))
            return output_cover_path.resolve()
    except Exception:
        pass
    return None


def find_source_pdf_for_book(book_stem_or_path: Path | str, root_dir: Optional[Path] = None) -> Optional[Path]:
    """Locate the original source PDF for a book across workspace folders."""
    bname = canonical_book_stem(book_stem_or_path)
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent

    search_dirs = [
        root_dir / "mineru" / "in",
        root_dir / "mineru" / "completed",
        root_dir / "completed",
        root_dir / "in",
        root_dir / "PDF",
    ]
    if isinstance(book_stem_or_path, Path):
        p = book_stem_or_path.resolve()
        search_dirs.insert(0, p if p.is_dir() else p.parent)

    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        for ext in (".pdf", ".PDF"):
            cand = sdir / f"{bname}{ext}"
            if cand.is_file() and cand.stat().st_size > 0:
                return cand.resolve()
    return None


def extract_book_metadata(
    pdf_path: Optional[Path] = None,
    md_dir: Optional[Path] = None,
    book_stem: Optional[str] = None,
    output_dir: Optional[Path] = None,
    extract_cover: bool = True,
) -> Dict[str, Any]:
    """
    Collect and merge Stage 1 bibliographic metadata from:
      1. Filename heuristics
      2. `pdfinfo` tags
      3. Front-matter Markdown pages (first 12 pages)
      4. Page 1 cover extraction (`cover.jpg`)

    Saves `book_metadata.json` and `cover.jpg` in `output_dir` (defaults to `md_dir`).
    """
    stem_source = book_stem or (pdf_path.stem if pdf_path else (md_dir.name if md_dir else "Unknown Book"))
    c_stem = canonical_book_stem(stem_source)

    if pdf_path is None:
        pdf_path = find_source_pdf_for_book(md_dir or c_stem)

    fn_meta = parse_filename_metadata(c_stem)
    pdf_meta = parse_pdfinfo(pdf_path)
    md_meta = scan_markdown_frontmatter(md_dir)

    # Resolve Title:
    # Prefer front-matter H1 title or pdfinfo title if clean; otherwise filename title
    title = md_meta.get("title") or pdf_meta.get("title") or fn_meta["title"]
    # If filename title is longer and contains subtitle while md_meta title is shorter, keep md_meta title
    # and extract subtitle from filename if subtitle not already set
    subtitle = md_meta.get("subtitle")
    if not subtitle and fn_meta["title"] and title:
        fn_t = fn_meta["title"]
        if fn_t.lower().startswith(title.lower()) and len(fn_t) > len(title) + 4:
            rem = fn_t[len(title):].lstrip(" :—-")
            if rem:
                subtitle = rem

    # Resolve Authors:
    # Prefer front-matter copyright/CIP authors (which list all co-authors cleanly),
    # then filename authors, then pdfinfo authors
    authors = md_meta.get("authors") or fn_meta.get("authors") or pdf_meta.get("authors") or []

    # Resolve Published Year:
    # Prefer front-matter copyright year or filename year over PDF CreationDate
    published_year = (
        md_meta.get("publishedYear")
        or fn_meta.get("publishedYear")
        or pdf_meta.get("publishedYear")
        or pdf_meta.get("pdfCreationYear")
    )

    # Resolve Series:
    series = fn_meta.get("series") or md_meta.get("series") or []

    # Resolve Publisher, ISBN, Description, Genres:
    publisher = md_meta.get("publisher")
    if not authors and publisher:
        authors = [publisher]
    isbn = md_meta.get("isbn")
    description = md_meta.get("description") or pdf_meta.get("subject")
    genres = infer_genres_from_title(
        title=title,
        series=series,
        existing_genres=md_meta.get("genres", []),
    )

    target_out_dir = output_dir or md_dir
    cover_path: Optional[Path] = None
    if extract_cover and target_out_dir and target_out_dir.is_dir() and pdf_path:
        cover_path = extract_pdf_cover(pdf_path, target_out_dir / "cover.jpg")

    metadata: Dict[str, Any] = {
        "canonicalBookName": c_stem,
        "title": title,
        "subtitle": subtitle,
        "authors": authors,
        "narrators": [],
        "series": series,
        "genres": genres,
        "publishedYear": str(published_year) if published_year else None,
        "publishedDate": None,
        "publisher": publisher,
        "description": description,
        "isbn": isbn,
        "asin": None,
        "language": "en",
        "explicit": False,
        "abridged": False,
        "tags": ["TTS", "MinerU"],
        "sourcePdf": str(pdf_path) if pdf_path else None,
        "coverPath": str(cover_path) if cover_path else None,
    }

    if target_out_dir and target_out_dir.is_dir():
        meta_file = target_out_dir / "book_metadata.json"
        try:
            meta_file.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    return metadata


def build_abs_metadata_json(
    base_metadata: Dict[str, Any],
    chapters: List[Dict[str, Any]],
    narrators: Optional[List[str]] = None,
    title_override: Optional[str] = None,
    author_override: Optional[str] = None,
    series_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the final AudioBookshelf-compliant `metadata.json` dictionary by combining
    Stage 1 bibliographic metadata with Stage 3 narrator and chapter timing data.
    """
    title = title_override or base_metadata.get("title") or base_metadata.get("canonicalBookName") or "Untitled Book"

    authors: List[str] = list(base_metadata.get("authors") or [])
    if author_override and author_override not in ("Chatterbox TTS", "MinerU"):
        authors = _clean_author_name(author_override)

    series: List[str] = list(base_metadata.get("series") or [])
    if series_override:
        series = [series_override.strip()]

    final_narrators = narrators if narrators is not None else list(base_metadata.get("narrators") or ["Chatterbox TTS"])

    tags = list(base_metadata.get("tags") or ["TTS", "MinerU"])
    if "Chatterbox" not in tags:
        tags.append("Chatterbox")

    return {
        "title": title,
        "subtitle": base_metadata.get("subtitle"),
        "authors": authors,
        "narrators": final_narrators,
        "series": series,
        "genres": list(base_metadata.get("genres") or []),
        "publishedYear": base_metadata.get("publishedYear"),
        "publishedDate": base_metadata.get("publishedDate"),
        "publisher": base_metadata.get("publisher"),
        "description": base_metadata.get("description"),
        "isbn": base_metadata.get("isbn"),
        "asin": base_metadata.get("asin"),
        "language": base_metadata.get("language") or "en",
        "explicit": bool(base_metadata.get("explicit", False)),
        "abridged": bool(base_metadata.get("abridged", False)),
        "tags": tags,
        "chapters": chapters,
    }


def extract_chapters_from_m4b(m4b_path: Path) -> List[Dict[str, Any]]:
    """Read embedded chapter markers from an existing .m4b/.m4a/.mp3 file using `ffprobe`."""
    chapters: List[Dict[str, Any]] = []
    if not m4b_path or not m4b_path.is_file() or shutil.which("ffprobe") is None:
        return chapters
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_chapters", str(m4b_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout)
            for idx, ch in enumerate(data.get("chapters", [])):
                start_sec = round(float(ch.get("start_time", 0.0)), 3)
                end_sec = round(float(ch.get("end_time", start_sec)), 3)
                title = (ch.get("tags") or {}).get("title") or f"Page {idx + 1}"
                chapters.append({
                    "id": idx,
                    "start": start_sec,
                    "end": end_sec,
                    "title": title,
                })
    except Exception:
        pass
    return chapters


def backfill_abs_metadata_for_m4b(
    m4b_path: Path,
    narrator_label: str = "Chatterbox TTS (practical_engineering)",
) -> Tuple[Path, Optional[Path], Dict[str, Any]]:
    """
    Generate `<stem>.metadata.json` and `<stem>.cover.jpg` alongside an existing `.m4b` audiobook
    by combining Stage 1 metadata (`book_metadata.json` or live extraction) with embedded `.m4b` chapters.
    """
    m4b_p = m4b_path.resolve()
    bname = canonical_book_stem(m4b_p)
    root_dir = Path(__file__).resolve().parent.parent.parent

    # Locate companion _md or _txt directory
    md_dir: Optional[Path] = None
    for cand in (
        root_dir / "TTS_Script" / "completed" / f"{bname}_md",
        root_dir / "mineru" / "out" / f"{bname}_md",
        root_dir / "mineru" / "completed" / f"{bname}_md",
        root_dir / "chatterbox" / "completed" / f"{bname}_txt",
        root_dir / "TTS_Script" / "out" / f"{bname}_txt",
    ):
        if cand.is_dir():
            md_dir = cand
            break

    pdf_path = find_source_pdf_for_book(bname, root_dir=root_dir)
    base_meta = extract_book_metadata(
        pdf_path=pdf_path,
        md_dir=md_dir,
        book_stem=bname,
        output_dir=md_dir,
        extract_cover=True,
    )

    chapters = extract_chapters_from_m4b(m4b_p)
    abs_meta = build_abs_metadata_json(
        base_metadata=base_meta,
        chapters=chapters,
        narrators=[narrator_label],
    )

    out_meta_path = m4b_p.with_suffix(".metadata.json")
    out_meta_path.write_text(json.dumps(abs_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    out_cover_path: Optional[Path] = m4b_p.with_suffix(".cover.jpg")
    if base_meta.get("coverPath") and Path(base_meta["coverPath"]).is_file():
        shutil.copy2(base_meta["coverPath"], out_cover_path)
    elif pdf_path:
        out_cover_path = extract_pdf_cover(pdf_path, out_cover_path)
    else:
        out_cover_path = None

    return out_meta_path, out_cover_path, abs_meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Stage 1 bibliographic metadata and cover image from a PDF, *_md folder, or backfill an .m4b."
    )
    parser.add_argument("input", type=Path, help="Path to source PDF file, MinerU *_md directory, or .m4b audiobook.")
    parser.add_argument("--md-dir", type=Path, default=None, help="Path to MinerU *_md directory.")
    parser.add_argument("--pdf", type=Path, default=None, help="Path to source PDF file.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None, help="Directory to write book_metadata.json and cover.jpg.")
    parser.add_argument("--no-cover", action="store_true", help="Skip extracting Page 1 cover.jpg.")
    args = parser.parse_args()

    inp = args.input.resolve()
    if inp.is_file() and inp.suffix.lower() in (".m4b", ".mp3", ".m4a"):
        meta_path, cov_path, abs_meta = backfill_abs_metadata_for_m4b(inp)
        print(f"Generated AudioBookshelf metadata: {meta_path}")
        if cov_path:
            print(f"Generated cover image:            {cov_path}")
        print(json.dumps({k: v for k, v in abs_meta.items() if k != "chapters"}, indent=2, ensure_ascii=False))
        print(f"Total chapters embedded: {len(abs_meta.get('chapters', []))}")
        return 0

    pdf_path = args.pdf.resolve() if args.pdf else (inp if inp.is_file() else None)
    md_dir = args.md_dir.resolve() if args.md_dir else (inp if inp.is_dir() else None)

    meta = extract_book_metadata(
        pdf_path=pdf_path,
        md_dir=md_dir,
        output_dir=args.output_dir.resolve() if args.output_dir else md_dir,
        extract_cover=not args.no_cover,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
