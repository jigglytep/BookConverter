#!/usr/bin/env python3
"""Run mineru's vlm-http-client backend on a single page of a PDF, with
max_concurrency forced down to 1.

The `mineru` CLI has no flag for max_concurrency -- the http-client backend
defaults to firing up to 100 concurrent block-level requests per page (see
mineru/backend/vlm/vlm_analyze.py, ModelSingleton.get_model). llama-server
only has one execution slot (--parallel 1) so those requests just queue up
regardless, but this forces true one-request-at-a-time submission at the
client level too.

Pages are extracted with mineru's own pdfium-based start_page_id/end_page_id
(not a physically pre-split file). This particular book triggers a poppler
bug ("recursive dicts") that makes `pdfseparate` bloat every single-page
output to ~30MB -- pdfium extracts the same page cleanly at ~100KB, and it's
done in-memory against the original PDF rather than writing bloated files to
disk.

Usage: process_one_page.py <pdf_path> <page_number_1_indexed> <output_dir> <server_url> [lang]
"""
import sys
from pathlib import Path

from mineru.cli.common import do_parse, read_fn


def main() -> None:
    if len(sys.argv) < 5:
        print(
            f"Usage: {sys.argv[0]} <pdf_path> <page_number_1_indexed> <output_dir> <server_url> [lang]",
            file=sys.stderr,
        )
        sys.exit(2)

    pdf_path = Path(sys.argv[1])
    page_number = int(sys.argv[2])  # 1-indexed
    output_dir = sys.argv[3]
    server_url = sys.argv[4]
    lang = sys.argv[5] if len(sys.argv) > 5 else "ch"
    print("" + "=" * 80)
    print(f"Processing {pdf_path} page {page_number} to {output_dir} with server {server_url} and lang {lang}")
    print("" + "=" * 80)

    pdf_bytes = read_fn(pdf_path)
    page_id = page_number - 1  # do_parse's start/end_page_id are 0-indexed
    doc_name = f"{pdf_path.stem}_page_{page_number:04d}"

    do_parse(
        output_dir=output_dir,
        pdf_file_names=[doc_name],
        pdf_bytes_list=[pdf_bytes],
        p_lang_list=[lang],
        backend="vlm-http-client",
        server_url=server_url,
        start_page_id=page_id,
        end_page_id=page_id,
        max_concurrency=1,
    )


if __name__ == "__main__":
    main()
