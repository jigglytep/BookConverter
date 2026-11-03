# 📝 TTS_Script & Markdown Preprocessor Documentation (`NOTES.md`)

`md_tts.py` converts standard Markdown, MyST Markdown (Jupyter Book), and MinerU OCR page outputs into clean, TTS-ready plain text tailored for **Kokoro-82M TTS** audio synthesis and `.m4b` audiobook creation.

---

## 🚀 Quick Usage Examples

### 1. Installation
```bash
pip install -r requirements.txt
# Optional for stream_worker.py audio pre-synthesis:
pip install soundfile
```

### 2. Single File Conversion
```bash
python3 md_tts.py chapter.md -o chapter.txt
```

### 3. Convert an Entire Book Directory (Recursive)
```bash
python3 md_tts.py /path/to/book_md/ -o out/book_txt/ --recursive
```

### 4. Stream Directly to TTS via Stdout
```bash
python3 md_tts.py chapter.md --stdout | your-tts-cli
```

### 5. Spell Out Numbers, Currencies, Units & Percentages for Natural TTS
```bash
# e.g., "100" -> "one hundred", "$5.50" -> "five dollars and fifty cents", "1995" -> "nineteen ninety-five"
# e.g., "50kg" -> "fifty kilograms", "16GB" -> "sixteen gigabytes", "10-20" -> "ten to twenty"
python3 md_tts.py chapter.md -o chapter.txt --numbers words
```

### 6. Math Error Resilience & Offline Fallback
```bash
# Automatically verbalizes equations locally even if mathspeak-api is unavailable
python3 md_tts.py book/ -o out/ -r --on-math-error keep
```

### 7. Automated Batch Stripping (`batch_fast.sh`)
Scans `../mineru/out/*_md`, strips all markdown pages to clean text in `out/<Book>_txt/`, verifies the output, and safely moves completed source folders and their `_pages` directories to `completed/`:
```bash
./batch_fast.sh
# Or forward custom options:
./batch_fast.sh --tables drop --numbers words
```

---

## 🛠️ Key Features & Processing Rules

### 📊 1. Table Handling (`--tables`, `--table-template`)
Reading raw tables row-by-row in an audiobook sounds awkward and unreadable. `md_tts.py` intelligently identifies all table formats and replaces them with a clean note or removes them.

- **Supported Formats**:
  - **MyST `list-table`**: ````{list-table} Title ... ````
  - **MyST `table`**: ````{table} Title ... ````
  - **Markdown Pipe Tables**: `| Header 1 | Header 2 |` ... `| --- | --- |`
  - **Grid Tables**: `+---+---+` ... `| Cell 1 | Cell 2 |`
  - **HTML Tables**: `<table>...</table>` (including MinerU stat blocks and OCR tables)
- **Modes & Custom Templates**:
  - `note` *(default)*: Replaces tables with `See table <title/name>.` if a title, caption, or name is detected, or `See table.` if no title is present.
  - **Custom Template**: Pass any template using `--table-template "See table {name}."` or `--tables "See table {name}."`. For bracketed notes, use `--table-template "[Table: {name}]"`.
  - `drop`: Completely strips all tables without any placeholder (`--tables drop`).
  - `linearize`: Reads rows sequentially as comma-separated lines (`--tables linearize`).

### 📐 2. Spoken LaTeX Math (`mathspeak-api` & Local Fallback)
- **Fast-Path & In-Memory Cache**: Single-letter variables (`$x$`, `$y$`), Greek letters (`$\alpha$`, `$\beta$`), and simple subscripts (`$w_0$`, `$x_1$`) resolve instantly without network roundtrips. Converted equations are cached across batches and pages.
- **Speech Rule Engine (SRE / MathJax)**: Complex math (`$$...$$`, ````{math}````, ````{amsmath}````) is sent to the local Speech Rule Engine API on port 8099.
- **Offline / Local Fallback Verbalizer**: If `mathspeak-api` is offline or fails, equations are verbalized into natural spoken words (e.g. `\frac{a}{b}` $\rightarrow$ `a over b`, `\sqrt{w}` $\rightarrow$ `the square root of w`, `\pm` $\rightarrow$ `plus or minus`) instead of emitting raw LaTeX code with backslashes.
- **Strict CommonMark Math Delimiters**: Inline math requires non-whitespace adjacent delimiters (`(?<![\$\\])\$(?!\s|\$)...(?<!\s|\$)\$(?!\d)`), preventing dollar prices (e.g. `$5 and $10`) from colliding with equation extraction.

### 📖 3. MyST & Jupyter Book Syntax Cleaning
- **YAML Frontmatter**: Automatically strips leading `--- \n title: ... \n ---` metadata blocks.
- **Blockquotes & Horizontal Rules**: Strips blockquote `>` symbols and `---` dividers so they aren't vocalized as "greater than" or "dash dash dash".
- **Figures / Images**: ````{figure} ... ```` and `![alt](url)` are converted to `Figure: <caption/alt>` (or dropped with `--images drop`).
- **Admonitions**: `::: {admonition} Title` and ````{admonition}```` are formatted as `Note: <title>.\n\n<content>`.
- **Code Blocks**: ````{code-block}````, generic ` ``` `, and tilde `~~~` blocks are dropped by default (`--code drop`) or replaced with `[Code block: <desc>]` (`--code placeholder`).
- **Citations**:
  - Parenthetical `{cite:p}` citations are removed.
  - Textual `{cite:t}` citations are converted into spoken author references (e.g., `Gelman et al.`).
- **Cross-References**: Resolves `{numref}`, `{ref}`, and `{eq}` labels into natural prose references (`Figure X`, `Table Y`, `the equation`).
- **Footnotes**: Drops inline `[^1]` footnote marks and trailing definitions (`--keep-footnotes` to preserve).

### 🔤 4. Text & Punctuation Normalization for Kokoro TTS
- **Ellipsis Preservation**: Preserves conversational ellipses (`...`) to allow Kokoro to generate natural pauses, while collapsing unintended repeated punctuation (`??`, `!!`, `,,`).
- **Abbreviation Expansion**: Expands abbreviations (`e.g.` $\rightarrow$ `for example,`, `i.e.` $\rightarrow$ `that is,`, `et al.` $\rightarrow$ `and colleagues,`, `Fig.` $\rightarrow$ `Figure`, `vs.` $\rightarrow$ `versus`) to prevent Kokoro from splitting sentences prematurely on abbreviation periods.
- **OCR Line-Break De-hyphenation**: Automatically joins hyphenated words split across line breaks (`funda-\nmental` $\rightarrow$ `fundamental`).
- **List Pacing & Punctuation**: Ensures bullet points and numbered list items end with punctuation so they do not concatenate into a breathless run-on sentence.
- **Identifier Readability**: Converts snake_case identifiers (`process_md_page` $\rightarrow$ `process md page`) into spoken words.
- **Unit & Measurement Expansion**: Expands units attached to numbers (`50kg` $\rightarrow$ `fifty kilograms`, `16GB` $\rightarrow$ `sixteen gigabytes`, `3.2GHz` $\rightarrow$ `three point two gigahertz`).
- **Number Ranges & Negatives**: Reads numeric ranges naturally (`10-20` $\rightarrow$ `ten to twenty`, `-5.5` $\rightarrow$ `minus five point five`).
- **HTML Unescaping & Spacing**: Unescapes entities (`&quot;`, `&#x27;`) and replaces tags with appropriate spacing so adjacent words are not smashed together (`Hello<br>world` $\rightarrow$ `Hello world`).

---

## ⚡ Streaming Pipeline Integration (`stream_worker.py`)

When running inside the automated pipeline:
- `stream_worker.py` monitors the `mineru/out/<Book>_md/` directory (recursively scanning `**/*.md`) while MinerU runs OCR.
- **Partial Write Detection**: Ensures files are fully written before conversion to avoid reading truncated pages.
- **Decoupled Producer-Consumer Architecture**: Text conversion runs immediately as pages appear, enqueueing clean text files to a background Kokoro GPU synthesis thread for non-blocking concurrent caching.

---

## 🎛️ CLI Options Reference

| Option | Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `inputs` | Files / Directories | *(required)* | Path(s) to Markdown/MyST files or directories. |
| `-o`, `--output` | Path | `None` | Output file path (single file) or output directory. |
| `-r`, `--recursive` | *(flag)* | `False` | Search recursively for `**/*.md` in input directories. |
| `--stdout` | *(flag)* | `False` | Output cleaned text directly to stdout. |
| `--tables` | `note`, `drop`, `linearize`, or template | `note` | Table handling: replace with note, drop, linearize, or custom template. |
| `--table-template` | Template string | `See table {name}.` | Format for table notes (e.g. `'See table {name}.'`, `'[Table: {name}]'`). |
| `--numbers` | `keep`, `words` | `words` | Spell numbers/currencies/units out as words via `num2words`, or keep digits. |
| `--code` | `drop`, `placeholder`, `keep` | `drop` | Code block handling. |
| `--images` | `caption`, `drop` | `caption` | Image/figure handling. |
| `--headings` | `keep`, `drop` | `keep` | Heading handling. |
| `--math-api` | URL | `http://localhost:8099` | Base URL of the MathSpeak Speech Rule Engine API. |
| `--math-style` | `clearspeak`, `mathspeak` | `clearspeak` | SRE speech style rule set. |
| `--on-math-error` | `keep`, `fallback`, `fail`, `skip` | `keep` | Fallback behavior when MathSpeak API fails (keep/fallback uses local verbalizer). |
| `--no-tts-punct` | *(flag)* | `False` | Disable automatic sentence-ending periods on headings. |
| `--keep-footnotes`| *(flag)* | `False` | Render footnote definitions at the end of the document. |
| `--timeout` | Seconds | `30.0` | MathSpeak API HTTP request timeout. |