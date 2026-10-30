#!/usr/bin/env python3
"""
upload_to_abs.py -- upload an audiobook and companion files (EPUB, PDF, metadata.json, cover.jpg)
to a self-hosted AudioBookshelf server.

Authenticates (API token, or username/password), resolves the target library and
folder, and POSTs the file(s) to /api/upload so AudioBookshelf ingests them into
the exact same book directory as a unified audio + ebook item with full metadata.

Requires: pip install requests

Usage:
    # Upload an m4b with auto-detected adjacent .epub, .pdf, .metadata.json, and .cover.jpg
    python upload_to_abs.py book.m4b

    # Preview discovered companion files and resolved metadata without uploading
    python upload_to_abs.py book.m4b --dry-run

    # Explicitly specify companion files or skip specific types
    python upload_to_abs.py book.m4b --no-pdf --library "TTS"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    sys.exit("requests is required. Install it with:  pip install requests")


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
MINERU_SCRIPTS_DIR = ROOT_DIR / "mineru" / "scripts"
if MINERU_SCRIPTS_DIR.exists() and str(MINERU_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(MINERU_SCRIPTS_DIR))
try:
    from metadata_extractor import backfill_abs_metadata_for_m4b
except ImportError:
    backfill_abs_metadata_for_m4b = None


def _load_dotenv() -> None:
    """Load key=value pairs from project root .env and local .env if present."""
    for env_file in (ROOT_DIR / ".env", SCRIPT_DIR / ".env"):
        if env_file.is_file():
            for raw_line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


_load_dotenv()


def login(server: str, username: str, password: str) -> str:
    r = requests.post(
        f"{server}/login",
        json={"username": username, "password": password},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["user"]["token"]


def get_libraries(server: str, headers: dict) -> list:
    r = requests.get(f"{server}/api/libraries", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["libraries"]


def pick_library(libs: list, name_or_id: Optional[str]):
    if name_or_id:
        for lib in libs:
            if lib["id"] == name_or_id or lib["name"].lower() == name_or_id.lower():
                return lib
        sys.exit(
            f"No library matching '{name_or_id}'. Available: "
            + ", ".join(l["name"] for l in libs)
        )
    book_libs = [l for l in libs if l.get("mediaType") == "book"]
    if len(book_libs) == 1:
        return book_libs[0]
    sys.exit(
        "Multiple libraries found; choose one with --library. Available: "
        + ", ".join(f"{l['name']} ({l.get('mediaType')})" for l in libs)
    )


def find_companion_epub(file_path: Path) -> Optional[Path]:
    """Finds an EPUB file matching the audio file's name in the same or parent directories."""
    candidates = [
        file_path.with_suffix(".epub"),
        file_path.parent / f"{file_path.stem}.epub",
        file_path.parent.parent / f"{file_path.stem}.epub",
        file_path.parent.parent / "out" / file_path.stem / f"{file_path.stem}.epub",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    return None


def find_companion_pdf(file_path: Path) -> Optional[Path]:
    """Finds the original source PDF matching the audio file's stem across workspace directories."""
    stem = file_path.stem
    search_dirs = [
        file_path.parent,
        ROOT_DIR / "completed",
        ROOT_DIR / "mineru" / "completed",
        ROOT_DIR / "in",
        ROOT_DIR / "mineru" / "in",
        ROOT_DIR / "PDF",
    ]
    for sdir in search_dirs:
        if not sdir.is_dir():
            continue
        for ext in (".pdf", ".PDF"):
            candidate = sdir / f"{stem}{ext}"
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate.resolve()
    return None


def find_companion_metadata(file_path: Path) -> Optional[Path]:
    """Finds the generated AudioBookshelf metadata JSON file matching the audio file's stem."""
    stem = file_path.stem
    candidates = [
        file_path.with_suffix(".metadata.json"),
        file_path.parent / f"{stem}.metadata.json",
        file_path.parent / "metadata.json",
        ROOT_DIR / "chatterbox" / "completed" / f"{stem}_txt" / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    return None


def find_companion_cover(file_path: Path) -> Optional[Path]:
    """Finds the extracted cover image matching the audio file's stem."""
    stem = file_path.stem
    candidates = [
        file_path.with_suffix(".cover.jpg"),
        file_path.parent / f"{stem}.cover.jpg",
        file_path.parent / "cover.jpg",
        ROOT_DIR / "TTS_Script" / "completed" / f"{stem}_md" / "cover.jpg",
        ROOT_DIR / "chatterbox" / "completed" / f"{stem}_txt" / "cover.jpg",
        ROOT_DIR / "mineru" / "out" / f"{stem}_md" / "cover.jpg",
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 1024:
            return candidate.resolve()
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Upload an audiobook file (and optional companion EPUB, PDF, metadata.json, cover.jpg) into AudioBookshelf."
    )
    ap.add_argument("file", type=Path, help="Audiobook file to upload (e.g. a .m4b or .mp3).")
    ap.add_argument(
        "--epub",
        type=Path,
        default=None,
        help="Companion EPUB file to upload into the same book directory (auto-detected if adjacent <name>.epub exists).",
    )
    ap.add_argument(
        "--no-epub",
        "--skip-epub",
        action="store_true",
        help="Disable automatic companion EPUB discovery.",
    )
    ap.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Companion PDF file to upload into the same book directory (auto-detected from completed/ or in/).",
    )
    ap.add_argument(
        "--no-pdf",
        "--skip-pdf",
        action="store_true",
        help="Disable automatic companion PDF discovery.",
    )
    ap.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Companion AudioBookshelf metadata.json file (auto-detected if <name>.metadata.json exists).",
    )
    ap.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable automatic companion metadata.json upload.",
    )
    ap.add_argument(
        "--cover",
        type=Path,
        default=None,
        help="Companion cover.jpg file (auto-detected if <name>.cover.jpg exists).",
    )
    ap.add_argument(
        "--no-cover",
        action="store_true",
        help="Disable automatic companion cover.jpg upload.",
    )
    ap.add_argument(
        "--server",
        default=os.environ.get("ABS_SERVER", "http://localhost:3378"),
        help="ABS base URL (or set ABS_SERVER in .env; default: http://localhost:3378).",
    )
    ap.add_argument(
        "--token",
        default=os.environ.get("ABS_TOKEN") or None,
        help="API token (or set ABS_TOKEN in .env).",
    )
    ap.add_argument(
        "--username",
        default=os.environ.get("ABS_USERNAME", ""),
        help="Username if not using --token (or set ABS_USERNAME in .env).",
    )
    ap.add_argument(
        "--password",
        default=os.environ.get("ABS_PASSWORD", ""),
        help="Password if not using --token (or set ABS_PASSWORD in .env).",
    )
    ap.add_argument(
        "--library",
        default=os.environ.get("ABS_LIBRARY", "TTS"),
        help="Target library name or ID (default: TTS).",
    )
    ap.add_argument(
        "--folder",
        default=os.environ.get("ABS_FOLDER") or None,
        help="Target folder ID (default: the library's first folder).",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Book title (default: title from metadata.json or file name without extension).",
    )
    ap.add_argument("--author", default="", help="Author (default: authors from metadata.json).")
    ap.add_argument("--series", default="", help="Series (default: series from metadata.json).")
    ap.add_argument(
        "--scan",
        action="store_true",
        help="Trigger a library scan after upload.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered files and resolved metadata without uploading to AudioBookshelf.",
    )
    args = ap.parse_args()

    if not args.file.is_file():
        sys.exit(f"Not a file: {args.file}")
    server = args.server.rstrip("/")

    # List of (local_path, upload_filename)
    upload_entries: List[Tuple[Path, str]] = [(args.file.resolve(), args.file.name)]

    # 1. Resolve companion EPUB
    epub_path: Optional[Path] = None
    if args.epub:
        if not args.epub.is_file():
            sys.exit(f"Specified EPUB not found: {args.epub}")
        epub_path = args.epub.resolve()
    elif not args.no_epub and args.file.suffix.lower() != ".epub":
        epub_path = find_companion_epub(args.file)
        if epub_path:
            print(f"Auto-detected companion EPUB:     '{epub_path}'")
    if epub_path and all(p != epub_path for p, _ in upload_entries):
        upload_entries.append((epub_path, epub_path.name))

    # 2. Resolve companion PDF
    pdf_path: Optional[Path] = None
    if args.pdf:
        if not args.pdf.is_file():
            sys.exit(f"Specified PDF not found: {args.pdf}")
        pdf_path = args.pdf.resolve()
    elif not args.no_pdf and args.file.suffix.lower() != ".pdf":
        pdf_path = find_companion_pdf(args.file)
        if pdf_path:
            print(f"Auto-detected companion PDF:      '{pdf_path}'")
    if pdf_path and all(p != pdf_path for p, _ in upload_entries):
        upload_entries.append((pdf_path, pdf_path.name))

    # 3. Resolve companion metadata.json
    meta_path: Optional[Path] = None
    meta_dict: Dict = {}
    if args.metadata:
        if not args.metadata.is_file():
            sys.exit(f"Specified metadata JSON not found: {args.metadata}")
        meta_path = args.metadata.resolve()
    elif not args.no_metadata:
        meta_path = find_companion_metadata(args.file)
        if not meta_path and backfill_abs_metadata_for_m4b is not None:
            try:
                meta_path, _, _ = backfill_abs_metadata_for_m4b(args.file)
                print(f"Auto-generated companion Metadata: '{meta_path}'")
            except Exception as e:
                print(f"Warning: fallback metadata generation failed: {e}", file=sys.stderr)
        elif meta_path:
            print(f"Auto-detected companion Metadata: '{meta_path}'")
    if meta_path:
        try:
            meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: could not parse {meta_path}: {e}", file=sys.stderr)
        if all(p != meta_path for p, _ in upload_entries):
            upload_entries.append((meta_path, "metadata.json"))

    # 4. Resolve companion cover.jpg
    cover_path: Optional[Path] = None
    if args.cover:
        if not args.cover.is_file():
            sys.exit(f"Specified cover image not found: {args.cover}")
        cover_path = args.cover.resolve()
    elif not args.no_cover:
        cover_path = find_companion_cover(args.file)
        if cover_path:
            print(f"Auto-detected companion Cover:    '{cover_path}'")
    if cover_path and all(p != cover_path for p, _ in upload_entries):
        upload_entries.append((cover_path, "cover.jpg"))

    # Resolve title, author, series using CLI overrides first, then metadata.json, then filename stem
    stem = args.file.stem
    if args.title and args.title != stem:
        title = args.title
    else:
        title = meta_dict.get("title") or args.title or stem

    if args.author:
        author = args.author
    elif isinstance(meta_dict.get("authors"), list) and meta_dict["authors"]:
        author = ", ".join(str(a) for a in meta_dict["authors"] if a)
    else:
        author = ""

    if args.series:
        series = args.series
    elif isinstance(meta_dict.get("series"), list) and meta_dict["series"]:
        series = str(meta_dict["series"][0])
    else:
        series = ""

    files_desc = " + ".join(
        f"'{local_p.name}' (as '{remote_name}')" if local_p.name != remote_name else f"'{local_p.name}'"
        for local_p, remote_name in upload_entries
    )

    if args.dry_run:
        print("\n[DRY-RUN] Upload Payload Summary:")
        print(f"  Server  : {server}")
        print(f"  Library : {args.library}")
        print(f"  Title   : {title}")
        print(f"  Author  : {author or '(none)'}")
        print(f"  Series  : {series or '(none)'}")
        print(f"  Files   : {files_desc}")
        return

    token = args.token
    if not token:
        if not (args.username and args.password):
            sys.exit("Provide --token, or both --username and --password.")
        try:
            token = login(server, args.username, args.password)
        except Exception as exc:
            sys.exit(f"Login failed: {exc}")
    headers = {"Authorization": f"Bearer {token}"}

    try:
        libs = get_libraries(server, headers)
    except Exception as exc:
        sys.exit(f"Could not list libraries (check --server and credentials): {exc}")
    library = pick_library(libs, args.library)

    folders = library.get("folders", [])
    if args.folder:
        folder_id = args.folder
    elif folders:
        folder_id = folders[0]["id"]
    else:
        sys.exit(f"Library '{library['name']}' has no folders configured.")

    data = {
        "title": title,
        "author": author,
        "series": series,
        "library": library["id"],
        "folder": folder_id,
    }

    print(
        f"Uploading {files_desc} -> library '{library['name']}' "
        f"as '{title}' (author: '{author or 'N/A'}') ..."
    )

    try:
        with contextlib.ExitStack() as stack:
            files_payload = {}
            for idx, (fpath, remote_name) in enumerate(upload_entries):
                fh = stack.enter_context(fpath.open("rb"))
                files_payload[str(idx)] = (remote_name, fh)

            r = requests.post(
                f"{server}/api/upload",
                headers=headers,
                data=data,
                files=files_payload,
                timeout=None,
            )
    except Exception as exc:
        sys.exit(f"Upload request failed: {exc}")

    if r.status_code == 200:
        print("✓ Upload succeeded into book directory.")
    elif r.status_code == 403:
        sys.exit(
            "403 Forbidden: this account lacks upload permission "
            "(enable it in Settings -> Users)."
        )
    elif r.status_code == 404:
        sys.exit(f"404: library not found (id {library['id']}). Check --library.")
    else:
        sys.exit(f"Upload failed: HTTP {r.status_code}: {r.text[:300]}")

    if args.scan:
        try:
            requests.post(
                f"{server}/api/libraries/{library['id']}/scan",
                headers=headers,
                timeout=30,
            )
            print("Library scan triggered.")
        except Exception as exc:
            print(f"(Scan request failed: {exc})", file=sys.stderr)

    # Also push full metadata (authors, narrators, publisher, ISBN, genres, description, etc.)
    # directly to the AudioBookshelf item via PATCH /api/items/{id}/media if metadata_dict is available
    if meta_dict:
        try:
            items_resp = requests.get(
                f"{server}/api/libraries/{library['id']}/items",
                headers=headers,
                timeout=30,
            )
            if items_resp.status_code == 200:
                norm_prefix = " ".join(args.file.stem.split())[:40]
                for it in items_resp.json().get("results", []):
                    detail = requests.get(f"{server}/api/items/{it['id']}", headers=headers, timeout=15).json()
                    lib_files = [f["metadata"]["filename"] for f in detail.get("libraryFiles", [])]
                    if any(" ".join(lf.split()).startswith(norm_prefix) for lf in lib_files):
                        patch_payload = {
                            "metadata": {
                                "title": title,
                                "subtitle": meta_dict.get("subtitle"),
                                "authors": [{"name": a} for a in (meta_dict.get("authors") or [])],
                                "narrators": meta_dict.get("narrators") or [],
                                "series": [{"name": sr} for sr in (meta_dict.get("series") or [])],
                                "genres": meta_dict.get("genres") or [],
                                "publishedYear": meta_dict.get("publishedYear"),
                                "publisher": meta_dict.get("publisher"),
                                "description": meta_dict.get("description"),
                                "isbn": meta_dict.get("isbn"),
                                "language": meta_dict.get("language") or "en",
                                "tags": meta_dict.get("tags") or [],
                            }
                        }
                        requests.patch(
                            f"{server}/api/items/{it['id']}/media",
                            headers=headers,
                            json=patch_payload,
                            timeout=15,
                        )
                        print(f"✓ Synced rich metadata to AudioBookshelf item ({it['id']}).")
                        break
        except Exception as exc:
            print(f"(Metadata sync notice: {exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
