#!/usr/bin/env python3
"""
md_tts.py — Convert Markdown and MyST Markdown into TTS-ready plain text for Kokoro TTS.

Supports:
  - Standard CommonMark and MyST Markdown (Jupyter Book syntax)
  - Math extraction (```{math}```, ```{amsmath}```, $$...$$, $...$) verbalized via mathspeak-api (SRE / MathJax)
    with fast-path local resolution, in-memory caching, and offline fallback verbalization
  - Currency protection and unit-aware number normalization via num2words
  - MyST directives (```{figure}```, ```{list-table}```, ```{table}```, ```{code-block}```, ::: {admonition} :::, ```{epigraph}```)
  - Table removal and replacement with spoken notes ([Table: title] or [Table omitted])
  - Markdown pipe tables, grid tables, and HTML tables
  - MyST roles ({numref}, {ref}, {eq}, {cite:p}, {cite:t})
  - MyST anchors and labels ((target)=)
  - Paragraph flow & punctuation normalization tailored for Kokoro TTS audio synthesis:
    - Ellipsis preservation (...) for natural speech pauses
    - Abbreviation expansion (e.g., i.e., et al., Fig., vs.) to prevent premature sentence splitting
    - OCR soft-hyphen de-hyphenation across line wraps
    - Bullet and numbered list pacing and terminal punctuation
    - Snake_case and technical identifier readability

Requires:  pip install num2words
"""

import argparse
import glob
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

try:
    from num2words import num2words
except ImportError:
    num2words = None

MATH_SENTINEL_RE = re.compile(r"\x00MATHSENTINEL(\d+)\x00")

# --------------------------------------------------------------------------- #
# Units & Measurement Normalization
# --------------------------------------------------------------------------- #
UNIT_MAP = {
    "kg": ("kilogram", "kilograms"),
    "g": ("gram", "grams"),
    "mg": ("milligram", "milligrams"),
    "km": ("kilometer", "kilometers"),
    "m": ("meter", "meters"),
    "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"),
    "ms": ("millisecond", "milliseconds"),
    "s": ("second", "seconds"),
    "min": ("minute", "minutes"),
    "hr": ("hour", "hours"),
    "hrs": ("hour", "hours"),
    "GB": ("gigabyte", "gigabytes"),
    "MB": ("megabyte", "megabytes"),
    "KB": ("kilobyte", "kilobytes"),
    "TB": ("terabyte", "terabytes"),
    "GHz": ("gigahertz", "gigahertz"),
    "MHz": ("megahertz", "megahertz"),
    "kHz": ("kilohertz", "kilohertz"),
    "Hz": ("hertz", "hertz"),
    "W": ("watt", "watts"),
    "kW": ("kilowatt", "kilowatts"),
    "MW": ("megawatt", "megawatts"),
    "V": ("volt", "volts"),
    "mV": ("millivolt", "millivolts"),
    "A": ("amp", "amps"),
    "mA": ("milliamp", "milliamps"),
    "mph": ("mile per hour", "miles per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "dB": ("decibel", "decibels"),
}

UNIT_RE = re.compile(
    r"\b(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>"
    + "|".join(re.escape(k) for k in sorted(UNIT_MAP.keys(), key=len, reverse=True))
    + r")\b"
)


def expand_units(text: str) -> str:
    """Expand units adjacent to numbers: '50kg' -> '50 kilograms'."""
    def _repl(m):
        val_str = m.group("val")
        unit = m.group("unit")
        val = float(val_str)
        sing, plur = UNIT_MAP[unit]
        unit_word = sing if val == 1.0 else plur
        return val_str + " " + unit_word

    return UNIT_RE.sub(_repl, text)


# --------------------------------------------------------------------------- #
# Number & Currency Normalization
# --------------------------------------------------------------------------- #
NUMBER_RE = re.compile(
    r"(?P<curr_sym>(?P<sym>[\$€£¥])\s?(?P<sym_num>\d[\d,]*(?:\.\d+)?))"
    r"|(?P<curr_code>(?P<code_num>\d[\d,]*(?:\.\d+)?)\s*(?P<code>USD|EUR|GBP|JPY))\b"
    r"|(?P<percent>(?P<pct_num>\d[\d,]*(?:\.\d+)?)\s?%)"
    r"|(?P<range>(?P<r_start>\d+)\s*[-–—]\s*(?P<r_end>\d+))\b"
    r"|(?P<ordinal>(?P<ord_num>\d+)(?:st|nd|rd|th)\b)"
    r"|(?P<negative>(?:(?<=\s)|(?<=^)|(?<=\())-(?P<neg_num>\d[\d,]*(?:\.\d+)?))"
    r"|(?P<year>\b(?P<year_num>(?:1[7-9]\d{2}|20\d{2}))\b)"
    r"|(?P<decimal>\b\d+\.\d+\b)"
    r"|(?P<plain>\b\d{1,3}(?:,\d{3})+\b|\b\d+\b)"
)


def _spell_decimal(num_str: str) -> str:
    """'3.14' -> 'three point one four'."""
    int_part, _, frac_part = num_str.replace(",", "").partition(".")
    words = num2words(int(int_part or 0))
    if frac_part:
        words += " point " + " ".join(num2words(int(d)) for d in frac_part)
    return words


def _spell_currency(symbol_or_code: str, num_str: str) -> str:
    clean_num = num_str.replace(",", "")
    dollars_str, _, cents_str = clean_num.partition(".")
    dollars = int(dollars_str or 0)
    cents = round(float("0." + cents_str) * 100) if cents_str else 0

    if symbol_or_code in ("$", "USD"):
        maj_sing, maj_plur = "dollar", "dollars"
        min_sing, min_plur = "cent", "cents"
    elif symbol_or_code in ("€", "EUR"):
        maj_sing, maj_plur = "euro", "euros"
        min_sing, min_plur = "cent", "cents"
    elif symbol_or_code in ("£", "GBP"):
        maj_sing, maj_plur = "pound", "pounds"
        min_sing, min_plur = "penny", "pence"
    elif symbol_or_code in ("¥", "JPY"):
        maj_sing, maj_plur = "yen", "yen"
        min_sing, min_plur = "", ""
    else:
        maj_sing, maj_plur = "dollar", "dollars"
        min_sing, min_plur = "cent", "cents"

    parts = []
    if dollars or not cents:
        parts.append(f"{num2words(dollars)} {maj_sing if dollars == 1 else maj_plur}")
    if cents and min_sing:
        parts.append(f"{num2words(cents)} {min_sing if cents == 1 else min_plur}")
    return " and ".join(parts)


def _number_repl(m):
    if m.group("curr_sym"):
        words = _spell_currency(m.group("sym"), m.group("sym_num"))
    elif m.group("curr_code"):
        words = _spell_currency(m.group("code"), m.group("code_num"))
    elif m.group("percent"):
        words = _spell_decimal(m.group("pct_num")) + " percent"
    elif m.group("range"):
        start_w = num2words(int(m.group("r_start")))
        end_w = num2words(int(m.group("r_end")))
        words = f"{start_w} to {end_w}"
    elif m.group("ordinal"):
        words = num2words(int(m.group("ord_num")), to="ordinal")
    elif m.group("negative"):
        neg_val = m.group("neg_num")
        if "." in neg_val:
            words = "minus " + _spell_decimal(neg_val)
        else:
            words = "minus " + num2words(int(neg_val.replace(",", "")))
    elif m.group("year"):
        words = num2words(int(m.group("year_num")), to="year")
    elif m.group("decimal"):
        words = _spell_decimal(m.group("decimal"))
    else:
        words = num2words(int(m.group("plain").replace(",", "")))

    if m.start() > 0 and m.string[m.start() - 1].isalpha():
        words = " " + words
    return words


def convert_numbers(text: str) -> str:
    text = expand_units(text)
    if num2words is None:
        return text
    return NUMBER_RE.sub(_number_repl, text)


# --------------------------------------------------------------------------- #
# Math API Integration, Fast-Path, Caching & Fallback Verbalizer
# --------------------------------------------------------------------------- #
GREEK_LETTERS = {
    "\\alpha": "alpha", "\\beta": "beta", "\\gamma": "gamma", "\\delta": "delta",
    "\\epsilon": "epsilon", "\\varepsilon": "epsilon", "\\zeta": "zeta", "\\eta": "eta",
    "\\theta": "theta", "\\vartheta": "theta", "\\iota": "iota", "\\kappa": "kappa",
    "\\lambda": "lambda", "\\mu": "mu", "\\nu": "nu", "\\xi": "xi", "\\pi": "pi",
    "\\rho": "rho", "\\sigma": "sigma", "\\tau": "tau", "\\upsilon": "upsilon",
    "\\phi": "phi", "\\varphi": "phi", "\\chi": "chi", "\\psi": "psi", "\\omega": "omega",
    "\\Gamma": "Gamma", "\\Delta": "Delta", "\\Theta": "Theta", "\\Lambda": "Lambda",
    "\\Sigma": "Sigma", "\\Phi": "Phi", "\\Psi": "Psi", "\\Omega": "Omega"
}

_MATH_CACHE = {}


def fast_math(tex: str):
    """Instantly resolve common single-letter and symbol math without network overhead."""
    tex = tex.strip()
    if len(tex) == 1 and tex.isalpha():
        return tex
    if tex.isdigit():
        return tex
    if tex in GREEK_LETTERS:
        return GREEK_LETTERS[tex]
    if len(tex) == 3 and tex[1] == "_" and tex[0].isalpha() and tex[2].isalnum():
        return f"{tex[0]} sub {tex[2]}"
    m_greek_sub = re.match(r"^(\\[a-zA-Z]+)_([a-zA-Z0-9])$", tex)
    if m_greek_sub and m_greek_sub.group(1) in GREEK_LETTERS:
        return f"{GREEK_LETTERS[m_greek_sub.group(1)]} sub {m_greek_sub.group(2)}"
    if tex.endswith("^2") and len(tex) == 3 and tex[0].isalpha():
        return f"{tex[0]} squared"
    if tex.endswith("^3") and len(tex) == 3 and tex[0].isalpha():
        return f"{tex[0]} cubed"
    return None


def fallback_verbalize_latex(tex: str) -> str:
    """Offline / local fallback verbalizer for math when API is unavailable or errors out."""
    for k, v in GREEK_LETTERS.items():
        tex = re.sub(re.escape(k) + r"\b", v, tex)
    # Fractions: \frac{a}{b} -> a over b
    tex = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1 over \2", tex)
    # Square root: \sqrt{x} -> the square root of x
    tex = re.sub(r"\\sqrt\{([^{}]+)\}", r"the square root of \1", tex)
    # Text commands: \text{...} -> ...
    tex = re.sub(r"\\[a-zA-Z]+\{([^{}]+)\}", r"\1", tex)
    # Subscripts: x_{foo} -> x sub foo, x_i -> x sub i
    tex = re.sub(r"([a-zA-Z0-9])_\{([^{}]+)\}", r"\1 sub \2", tex)
    tex = re.sub(r"([a-zA-Z0-9])_([a-zA-Z0-9])", r"\1 sub \2", tex)
    # Superscripts: x^2 -> x squared, x^{y} -> x to the power of y
    tex = re.sub(r"([a-zA-Z0-9])\^2\b", r"\1 squared", tex)
    tex = re.sub(r"([a-zA-Z0-9])\^3\b", r"\1 cubed", tex)
    tex = re.sub(r"([a-zA-Z0-9])\^\{([^{}]+)\}", r"\1 to the power of \2", tex)
    tex = re.sub(r"([a-zA-Z0-9])\^([a-zA-Z0-9])", r"\1 to the power of \2", tex)
    # Symbols & relations
    tex = re.sub(r"\\(?:le|leq)\b", "less than or equal to", tex)
    tex = re.sub(r"\\(?:ge|geq)\b", "greater than or equal to", tex)
    tex = re.sub(r"\\(?:ne|neq)\b", "not equal to", tex)
    tex = re.sub(r"\\approx\b", "approximately equal to", tex)
    tex = re.sub(r"\\(?:times|cdot)\b", "times", tex)
    tex = re.sub(r"\\pm\b", "plus or minus", tex)
    tex = re.sub(r"\\in\b", "in", tex)
    tex = re.sub(r"\\notin\b", "not in", tex)
    tex = re.sub(r"\\infty\b", "infinity", tex)
    tex = re.sub(r"\\sum\b", "the sum of", tex)
    tex = re.sub(r"\\prod\b", "the product of", tex)
    tex = re.sub(r"\\int\b", "the integral of", tex)
    tex = re.sub(r"\\partial\b", "partial", tex)
    # Remove remaining TeX commands and braces
    tex = re.sub(r"\\[a-zA-Z]+", "", tex)
    tex = re.sub(r"[{}]", "", tex)
    # Spoken operators
    tex = re.sub(r"\s*=\s*", " equals ", tex)
    tex = re.sub(r"\s*\+\s*", " plus ", tex)
    tex = re.sub(r"\s*-\s*", " minus ", tex)
    tex = re.sub(r"\s*/\s*", " over ", tex)
    tex = re.sub(r"\s*\|\s*", " given ", tex)
    tex = re.sub(r"\s+", " ", tex).strip()
    return tex


LATEX_KNOWN_WORDS = {
    "sin", "cos", "tan", "log", "exp", "lim", "max", "min", "inf", "det", "dim",
    "gcd", "hom", "ker", "deg", "arg", "sec", "csc", "cot", "sinh", "cosh", "tanh",
    "coth", "arcsin", "arccos", "arctan", "begin", "end", "left", "right", "over",
    "frac", "sqrt", "cdot", "cdots", "dots", "sum", "prod", "int", "approx", "sim",
    "partial", "mid", "propto", "subset", "subseteq", "in", "forall", "exists",
    "quad", "qquad", "text", "texttt", "textbf", "textit", "mathrm", "mathbf",
    "mathit", "mathbb", "mathcal", "mathscr", "mathfrak", "hline", "split",
    "aligned", "array", "matrix", "pmatrix", "bmatrix", "cases",
    "var", "cov", "std", "bias", "diag", "rank", "tr", "mod", "argmax", "argmin",
    "const", "proj", "span", "ln", "sign", "sgn", "supp", "trace", "erf", "vec", "vol"
}


UNWRAP_FONT_CMDS = {
    "mathrm", "mathsf", "mathtt", "mathbf", "mathit", "boldsymbol", "bm",
    "mathcal", "mathscr", "mathfrak", "textrm", "textsf", "texttt", "textbf",
    "textit", "mbox", "hbox"
}

STRIP_STYLE_CMDS_RE = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle"
    r"|rm|sf|bf|it|tt|cal|Bbb"
    r"|tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\b\s*"
)


def _unwrap_font_commands(tex: str) -> str:
    """Unwrap LaTeX font commands (\mathrm{...}, \mathsf{...}, etc.) with nested brace support."""
    all_cmds = UNWRAP_FONT_CMDS | {"text"}
    pattern = re.compile(r"\\(" + "|".join(sorted(all_cmds, key=len, reverse=True)) + r")\s*\{")
    pos = 0
    while True:
        m = pattern.search(tex, pos)
        if not m:
            break
        cmd = m.group(1)
        start = m.start()
        brace_open = m.end() - 1
        depth = 0
        end = -1
        for i in range(brace_open, len(tex)):
            if tex[i] == "{" and (i == 0 or tex[i - 1] != "\\"):
                depth += 1
            elif tex[i] == "}" and (i == 0 or tex[i - 1] != "\\"):
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            # Unbalanced brace: strip the command token and continue
            tex = tex[:start] + tex[brace_open:]
            pos = start + 1
            continue

        inner = tex[brace_open + 1:end].strip()
        # Collapse MinerU OCR spaced-out words inside font/text commands: e.g. "s t a r t" -> "start", "U S D" -> "USD"
        if re.match(r"^[a-zA-Z](?:\s+[a-zA-Z]){2,}$", inner):
            word = re.sub(r"\s+", "", inner)
            if word.lower() in ("sin", "cos", "tan", "log", "exp", "lim", "max", "min", "sup", "inf", "det", "dim"):
                repl = f"\\{word.lower()}"
            else:
                repl = f"\\text{{{word}}}"
            tex = tex[:start] + repl + tex[end + 1:]
            pos = start + len(repl)
            continue

        if cmd == "text":
            # Keep \text{...} intact and advance past the command header
            pos = brace_open + 1
            continue

        # Check if preceded by _ or ^ so subscript/superscript grouping is preserved
        prefix_s = tex[:start].rstrip()
        needs_braces = prefix_s.endswith(("_", "^"))
        if re.match(r"^[a-zA-Z]{3,}$", inner) and inner.lower() not in LATEX_KNOWN_WORDS:
            repl = f"\\text{{{inner}}}"
            if needs_braces:
                repl = f"{{{repl}}}"
        else:
            repl = f"{{{inner}}}" if needs_braces else inner

        tex = tex[:start] + repl + tex[end + 1:]
        pos = start
    return tex


def sanitize_latex(tex: str) -> str:
    """Fix common LaTeX and MinerU OCR quirks before passing to MathJax / mathspeak-api."""
    # Remove % comments
    lines = [re.sub(r"(?<!\\)%.*$", "", l) for l in tex.splitlines()]
    tex = "\n".join(lines).strip()

    # Strip standalone font/size switch commands (\scriptstyle, \rm, \bf, \Large, etc.)
    tex = STRIP_STYLE_CMDS_RE.sub("", tex)

    # Unwrap \mathrm{...}, \mathsf{...}, \mathbf{...}, etc. (prevents SRE reading "normal R" or "amperes" for A)
    tex = _unwrap_font_commands(tex)

    # Replace LaTeX spacing commands that MathJax/SRE verbalizes as "empty"
    tex = re.sub(r"\\(?:quad|qquad|enspace|thinspace|medspace|thickspace)\b", " ", tex)
    tex = re.sub(r"\\[ ,;:\!]", " ", tex)
    tex = re.sub(r"(?<!\\)~", " ", tex)

    # Normalize variant Greek commands
    tex = re.sub(r"\\varrho\b", r"\\rho", tex)
    tex = re.sub(r"\\varDelta\b", r"\\Delta", tex)
    tex = re.sub(r"\\vartheta\b", r"\\theta", tex)
    tex = re.sub(r"\\varepsilon\b", r"\\epsilon", tex)
    tex = re.sub(r"\\varphi\b", r"\\phi", tex)

    # Fix MinerU OCR scientific notation where \times was dropped: "3 . 8 1 0 ^ { - 3 }" -> "3.8 \times 10^{-3}"
    tex = re.sub(r"(?<=\d)\s+1\s*0\s*\^", r" \\times 10^", tex)
    # Collapse OCR spaces around decimal points: "3 . 8" -> "3.8"
    tex = re.sub(r"(?<=\d)\s*\.\s*(?=\d)", ".", tex)
    # Collapse OCR spaces between digits: "1 0 ^ { 1 1 }" -> "10 ^ { 11 }"
    tex = re.sub(r"(?<=\d)\s+(?=\d)", "", tex)

    # Unwrap redundant outer braces around the entire expression: "{ L }" -> "L"
    while tex.startswith("{") and tex.endswith("}"):
        depth = 0
        balanced_outer = True
        for i, ch in enumerate(tex):
            if ch == "{" and (i == 0 or tex[i - 1] != "\\"):
                depth += 1
            elif ch == "}" and (i == 0 or tex[i - 1] != "\\"):
                depth -= 1
                if depth == 0 and i < len(tex) - 1:
                    balanced_outer = False
                    break
        if balanced_outer and depth == 0:
            tex = tex[1:-1].strip()
        else:
            break

    # Normalize simple spaced subscripts/superscripts: "R _ { 0 }" -> "R_0", "x ^ { 2 }" -> "x^2"
    tex = re.sub(r"\s*([_\^])\s*\{\s*([a-zA-Z0-9])\s*\}", r"\1\2", tex)
    tex = re.sub(r"\s*([_\^])\s*([a-zA-Z0-9])\b", r"\1\2", tex)

    # Fix \beta_\text{foo} -> \beta_{\text{foo}}
    tex = re.sub(r"_\\text\{([^}]+)\}", r"_{\\text{\1}}", tex)
    tex = re.sub(r"_\\texttt\{([^}]+)\}", r"_{\\texttt{\1}}", tex)
    # Escape unescaped underscores in \text{...} or \texttt{...}
    def _esc_text(m):
        cmd = m.group(1)
        body = m.group(2)
        body_esc = re.sub(r"(?<!\\)_", r"\\_", body)
        return f"{cmd}{{{body_esc}}}"
    tex = re.sub(r"(\\text(?:tt|sf|bf|rm)?)\{([^\}]+)\}", _esc_text, tex)

    # Protect existing \command[...] and \command{...}{...}
    placeholders = {}
    def _hold(m):
        key = f"\x00CMD{len(placeholders)}\x00"
        placeholders[key] = m.group(0)
        return key

    cmd_pattern = re.compile(r"\\[a-zA-Z]+(?:\s*\[[^\]]*\])?(?:\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*")
    tex_prot = cmd_pattern.sub(_hold, tex)

    # Wrap bare words of 3+ letters in \text{} so they aren't spelled out letter-by-letter
    def _wrap_word(m):
        w = m.group(0)
        if w.lower() in LATEX_KNOWN_WORDS:
            return w
        return f"\\text{{{w}}}"

    tex_prot = re.sub(r"\b([a-zA-Z]{3,})\b", _wrap_word, tex_prot)

    for k, v in placeholders.items():
        tex_prot = tex_prot.replace(k, v)

    return re.sub(r"\s+", " ", tex_prot).strip()


def clean_math_speech(speech: str) -> str:
    """Clean font/artifact announcements emitted by Speech Rule Engine (SRE)."""
    if not speech:
        return speech
    speech = re.sub(r"\bnormal triangle\b", "Delta", speech)
    speech = re.sub(r"\b(?:normal|sans serif|monospace|bold|italic)\s+", "", speech)
    speech = re.sub(r"\bempty\b", "", speech)
    speech = re.sub(r"\bdivides\b", "given", speech)
    return re.sub(r"\s+", " ", speech).strip()


def api_batch(base, texs, style, is_display, timeout):
    sanitized = [sanitize_latex(t) for t in texs]
    payload = json.dumps({"items": sanitized, "style": style, "display": is_display}).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/speak/batch",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return [it.get("speech") for it in json.loads(r.read().decode("utf-8")).get("results", [])]


_MATH_API_CHECKED = False


def ensure_math_api_running(api_url: str):
    """If math_api points to localhost, ensure it is responsive or start mathspeak-api container."""
    global _MATH_API_CHECKED
    if _MATH_API_CHECKED:
        return
    _MATH_API_CHECKED = True

    try:
        import urllib.parse
        parsed = urllib.parse.urlparse(api_url)
        if parsed.hostname not in ("localhost", "127.0.0.1", "0.0.0.0"):
            return
        base = api_url.rstrip("/")
        health_url = f"{base}/healthz"

        # Check if already healthy
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass

        # Not responding: attempt to start podman container
        import shutil
        import subprocess
        import time

        podman_bin = shutil.which("podman")
        if not podman_bin:
            return

        res = subprocess.run([podman_bin, "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True)
        if "mathspeak-api" in res.stdout.split():
            print("[info] mathspeak-api is not running. Starting container via podman...", file=sys.stderr)
            subprocess.run([podman_bin, "start", "mathspeak-api"], capture_output=True)
            for _ in range(20):
                time.sleep(0.5)
                try:
                    req = urllib.request.Request(health_url)
                    with urllib.request.urlopen(req, timeout=1.0) as resp:
                        if resp.status == 200:
                            print("[info] mathspeak-api container is running and healthy.", file=sys.stderr)
                            return
                except Exception:
                    pass
            print("[warn] Started mathspeak-api container, but health check timed out.", file=sys.stderr)
    except Exception as e:
        print(f"[warn] Failed to verify/start mathspeak-api: {e}", file=sys.stderr)


class Ctx:
    def __init__(self, opts):
        self.opts = opts
        self.math = []  # list of (latex, is_block)

    def add_math(self, latex, is_block):
        cleaned = sanitize_latex(latex)
        trailing_punct = ""
        if not is_block and cleaned:
            m_punct = re.search(r"(?<![\.\\])([,.;:])$", cleaned)
            if m_punct:
                trailing_punct = m_punct.group(1)
                cleaned = cleaned[:-1].strip()
        if not cleaned:
            return trailing_punct
        idx = len(self.math)
        self.math.append((cleaned, is_block))
        return f"\x00MATHSENTINEL{idx}\x00{trailing_punct}"


def convert_math(ctx, stats):
    opts = ctx.opts
    speech = {}
    if not ctx.math:
        return speech

    stats.setdefault("equations", 0)
    stats.setdefault("converted", 0)
    stats.setdefault("errors", 0)

    # Pass 1: Resolve fast-path math and in-memory cached results
    pending_idxs = []
    for idx, (tex, is_block) in enumerate(ctx.math):
        cache_key = (tex, opts.math_style, is_block)
        quick = fast_math(tex)
        if quick is not None:
            speech[idx] = quick
            stats["converted"] += 1
        elif cache_key in _MATH_CACHE:
            speech[idx] = _MATH_CACHE[cache_key]
            stats["converted"] += 1
        else:
            pending_idxs.append(idx)

    if not pending_idxs:
        return speech

    # Ensure mathspeak-api is running before making HTTP batch calls
    ensure_math_api_running(opts.math_api)

    # Pass 2: Batch HTTP requests for remaining expressions
    groups = {True: [], False: []}
    for idx in pending_idxs:
        is_block = ctx.math[idx][1]
        groups[is_block].append(idx)

    for is_block, idxs in groups.items():
        if not idxs:
            continue
        texs = [ctx.math[i][0] for i in idxs]
        batch_size = 100
        for b_start in range(0, len(idxs), batch_size):
            b_idxs = idxs[b_start:b_start + batch_size]
            b_texs = texs[b_start:b_start + batch_size]
            try:
                results = api_batch(opts.math_api, b_texs, opts.math_style, is_block, opts.timeout)
            except Exception as e:
                if opts.on_math_error == "fail":
                    raise SystemExit(
                        f"[error] mathspeak API failed for {len(b_texs)} expression(s): {e}\n"
                        f"        (--math-api {opts.math_api}; use --on-math-error keep|skip to continue)"
                    )
                results = [None] * len(b_texs)

            for j, orig_idx in enumerate(b_idxs):
                tex = ctx.math[orig_idx][0]
                cache_key = (tex, opts.math_style, is_block)
                r = results[j] if j < len(results) else None
                if r:
                    r = clean_math_speech(r)
                    speech[orig_idx] = r
                    _MATH_CACHE[cache_key] = r
                    stats["converted"] += 1
                elif opts.on_math_error in ("keep", "fallback"):
                    # Use fallback verbalizer so Kokoro doesn't read TeX backslashes
                    fallback_text = fallback_verbalize_latex(tex)
                    speech[orig_idx] = fallback_text if fallback_text else tex
                    stats["errors"] += 1
                else:
                    speech[orig_idx] = opts.math_error_placeholder
                    stats["errors"] += 1

    return speech


# --------------------------------------------------------------------------- #
# Abbreviations Normalization
# --------------------------------------------------------------------------- #
ABBREVIATIONS = [
    (re.compile(r"\be\.g\.,?\s*", re.IGNORECASE), "for example, "),
    (re.compile(r"\bi\.e\.,?\s*", re.IGNORECASE), "that is, "),
    (re.compile(r"\bet al\.,?\s*", re.IGNORECASE), "and colleagues, "),
    (re.compile(r"\betc\.\b", re.IGNORECASE), "etcetera"),
    (re.compile(r"\bvs\.\s*", re.IGNORECASE), "versus "),
    (re.compile(r"\bapprox\.\s*", re.IGNORECASE), "approximately "),
    (re.compile(r"\bdept\.\s*", re.IGNORECASE), "department "),
    (re.compile(r"\bFig\.\s*(\d+)", re.IGNORECASE), r"Figure \1"),
    (re.compile(r"\bFigs\.\s*(\d+)", re.IGNORECASE), r"Figures \1"),
    (re.compile(r"\bTab\.\s*(\d+)", re.IGNORECASE), r"Table \1"),
    (re.compile(r"\bTabs\.\s*(\d+)", re.IGNORECASE), r"Tables \1"),
    (re.compile(r"\bCh\.\s*(\d+)", re.IGNORECASE), r"Chapter \1"),
    (re.compile(r"\bSec\.\s*(\d+)", re.IGNORECASE), r"Section \1"),
    (re.compile(r"\bpp\.\s*(\d+)", re.IGNORECASE), r"pages \1"),
    (re.compile(r"\bp\.\s*(\d+)", re.IGNORECASE), r"page \1"),
    (re.compile(r"\bDr\.\s+", re.IGNORECASE), "Doctor "),
    (re.compile(r"\bProf\.\s+", re.IGNORECASE), "Professor "),
    (re.compile(r"\bMr\.\s+", re.IGNORECASE), "Mr "),
    (re.compile(r"\bMrs\.\s+", re.IGNORECASE), "Mrs "),
    (re.compile(r"\bMs\.\s+", re.IGNORECASE), "Ms "),
]


# --------------------------------------------------------------------------- #
# Directives & Tables
# --------------------------------------------------------------------------- #
def _clean_directive_options(content: str) -> str:
    """Strip :label:, :name:, :width:, etc. from directives."""
    lines = []
    for line in content.splitlines():
        line_s = line.strip()
        if line_s.startswith(":") and not line_s.startswith(":\\"):
            continue
        if line_s.startswith("---"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_directive_title(first_line: str, body: str) -> str:
    """Extract caption/title or name from directive header line or body options."""
    title = first_line.strip()
    if title:
        title = re.sub(r"^:[a-zA-Z0-9_\-]+:.*$", "", title).strip()
        if title:
            return title
    label = ""
    for line in body.splitlines()[:15]:
        line_s = line.strip()
        if line_s.startswith(":caption:"):
            return line_s.split(":", 2)[-1].strip()
        if line_s.startswith(":title:"):
            return line_s.split(":", 2)[-1].strip()
        if line_s.startswith(":name:"):
            label = line_s.split(":", 2)[-1].strip()
    if label:
        return re.sub(r"^(?:tab|table):", "", label, flags=re.IGNORECASE).strip()
    return ""


def _clean_table_title(title: str) -> str:
    """Clean markdown formatting and HTML entities from a table title."""
    title = re.sub(r"[\*`~]", "", title)
    title = re.sub(r"_", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    title = html.unescape(title).strip()
    return title


def _format_table_note(title: str, opts) -> str:
    """Format table replacement when opts.tables is note, custom template, or drop."""
    tables_mode = getattr(opts, "tables", "note")
    if tables_mode == "drop":
        return ""

    template = getattr(opts, "table_template", None)
    if not template:
        if tables_mode not in ("note", "placeholder", "linearize"):
            template = tables_mode
        else:
            template = "See table {name}."

    if title:
        clean = _clean_table_title(title).rstrip(".")
        if clean:
            if "{name}" in template or "{title}" in template:
                note = template.replace("{name}", clean).replace("{title}", clean)
            else:
                note = f"{template} {clean}"
            if not note.endswith((".", "!", "?", "]")):
                note += "."
            return f"\n\n{note}\n\n"

    # Fallback when no title/name exists
    if "{name}" in template or "{title}" in template:
        note = template.replace("{name}", "").replace("{title}", "").strip()
        note = re.sub(r"\s+([,.;:!?])", r"\1", note)
        if not note or note in ("See table", "See table."):
            note = "See table."
        elif note in ("[Table:]", "[Table: .]", "[Table]"):
            note = "[Table omitted]"
    else:
        note = template.strip()
    if not note.endswith((".", "!", "?", "]")):
        note += "."
    return f"\n\n{note}\n\n"


def _format_list_table(title: str, content: str, opts) -> str:
    if opts.tables == "drop":
        return ""
    effective_title = _extract_directive_title(title, content)
    if opts.tables != "linearize":
        return _format_table_note(effective_title, opts)

    # Linearize mode
    rows = []
    current_row = []
    current_cell = []

    for line in content.splitlines():
        line_s = line.strip()
        if line_s.startswith(":") and not line_s.startswith(":\\"):
            continue
        if line_s.startswith("* -"):
            if current_cell:
                current_row.append(" ".join(current_cell).strip())
                current_cell = []
            if current_row:
                rows.append(current_row)
                current_row = []
            cell_text = line_s[3:].strip()
            if cell_text:
                current_cell.append(cell_text)
        elif line_s.startswith("-"):
            if current_cell:
                current_row.append(" ".join(current_cell).strip())
                current_cell = []
            cell_text = line_s[1:].strip()
            if cell_text:
                current_cell.append(cell_text)
        else:
            if line_s:
                current_cell.append(line_s)

    if current_cell:
        current_row.append(" ".join(current_cell).strip())
    if current_row:
        rows.append(current_row)

    out = []
    if effective_title:
        clean = _clean_table_title(effective_title)
        if clean:
            out.append(f"Table: {clean}.")
    for r in rows:
        clean = [c for c in r if c]
        if clean:
            out.append(", ".join(clean) + ".")
    return ("\n".join(out) + "\n\n") if out else ""


def _process_html_tables(text: str, opts) -> str:
    def _sub_table(m):
        if opts.tables == "drop":
            return ""
        table_html = m.group(0)
        cap_m = re.search(r"<caption[^>]*>(.*?)</caption>", table_html, flags=re.DOTALL | re.IGNORECASE)
        caption = ""
        if cap_m:
            caption = re.sub(r"<[^>]+>", "", cap_m.group(1)).strip()
            caption = html.unescape(caption).strip()

        if opts.tables != "linearize":
            return _format_table_note(caption, opts)

        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.DOTALL | re.IGNORECASE):
            cells = [re.sub(r"<[^>]+>", "", td).strip() for td in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.DOTALL | re.IGNORECASE)]
            cells = [html.unescape(c).strip() for c in cells if c.strip()]
            if cells:
                rows.append(", ".join(cells) + ".")
        out = []
        if caption:
            out.append(f"Table: {caption}.")
        out.extend(rows)
        return ("\n\n" + "\n".join(out) + "\n\n") if out else ""

    return re.sub(r"<table\b[^>]*>.*?</table>", _sub_table, text, flags=re.DOTALL | re.IGNORECASE)


DELIM_ROW_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{1,}:?[ \t]*(?:\|[ \t]*:?-{1,}:?[ \t]*)+\|?[ \t]*$"
    r"|^[ \t]*\|[ \t]*:?-{1,}:?[ \t]*\|[ \t]*$"
)
TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$|^[ \t]*[^\n\|]+\|[^\n\|]+[ \t]*$")
GRID_BORDER_RE = re.compile(r"^[ \t]*\+[-+=]+\+[ \t]*$")


def _process_pipe_and_grid_tables(text: str, opts) -> str:
    lines = text.split("\n")
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Grid table
        if GRID_BORDER_RE.match(line) and i + 1 < len(lines) and TABLE_ROW_RE.match(lines[i + 1]):
            start = i
            i += 1
            while i < len(lines) and (TABLE_ROW_RE.match(lines[i]) or GRID_BORDER_RE.match(lines[i])):
                is_last_border = GRID_BORDER_RE.match(lines[i])
                i += 1
                if is_last_border and (i >= len(lines) or not TABLE_ROW_RE.match(lines[i])):
                    break
            if opts.tables == "drop":
                continue
            elif opts.tables == "linearize":
                rows = []
                for row_line in lines[start:i]:
                    if GRID_BORDER_RE.match(row_line):
                        continue
                    cells = [c.strip() for c in row_line.strip().strip("|").split("|") if c.strip()]
                    if cells:
                        rows.append(", ".join(cells) + ".")
                if rows:
                    out_lines.append("")
                    out_lines.extend(rows)
                    out_lines.append("")
            else:
                out_lines.append("")
                out_lines.append(_format_table_note("", opts).strip())
                out_lines.append("")
            continue

        # Pipe table
        if (
            i + 1 < len(lines)
            and DELIM_ROW_RE.match(lines[i + 1])
            and TABLE_ROW_RE.match(line)
        ):
            start = i
            i += 2
            while i < len(lines) and TABLE_ROW_RE.match(lines[i]) and not DELIM_ROW_RE.match(lines[i]):
                i += 1
            table_lines = lines[start:i]
            if opts.tables == "drop":
                continue
            elif opts.tables == "linearize":
                rows = []
                header_cells = [c.strip() for c in table_lines[0].strip().strip("|").split("|") if c.strip()]
                if header_cells:
                    rows.append(", ".join(header_cells) + ".")
                for row_line in table_lines[2:]:
                    cells = [c.strip() for c in row_line.strip().strip("|").split("|") if c.strip()]
                    if cells:
                        rows.append(", ".join(cells) + ".")
                if rows:
                    out_lines.append("")
                    out_lines.extend(rows)
                    out_lines.append("")
            else:
                out_lines.append("")
                out_lines.append(_format_table_note("", opts).strip())
                out_lines.append("")
            continue

        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


# --------------------------------------------------------------------------- #
# MyST & Markdown Preprocessor & Cleaner
# --------------------------------------------------------------------------- #
def process_myst_and_markdown(text: str, ctx: Ctx) -> str:
    opts = ctx.opts

    # 1. YAML frontmatter
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

    # 2. HTML comments & % comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*%.*$", "", text, flags=re.MULTILINE)

    # 3. Soft-hyphen de-hyphenation from OCR line wraps (funda-\nmental -> fundamental)
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)

    # 4. Blockquotes (> quote)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)

    # 5. Horizontal rules (---, ***, ___)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 6. MyST Anchors (e.g. (chap1)=)
    text = re.sub(r"^\s*\([a-zA-Z0-9_\-:]+\)=\s*$", "", text, flags=re.MULTILINE)

    # 7. Pandoc cross-reference spans e.g. [fig:foo]{#fig:foo label="fig:foo"}
    text = re.sub(r"\[[^\]]*\]\{#[^\}]+\}", "", text)

    # 8. Block Math Directives: ```{math} ... ```, ````{amsmath} ... ````
    def _math_block_sub(m):
        raw = _clean_directive_options(m.group(1))
        if not raw:
            return ""
        sentinel = ctx.add_math(raw, is_block=True)
        return f"\n\n{sentinel}\n\n"

    text = re.sub(r"^\s*```+\{math\}(.*?)\n\s*```+\s*$", _math_block_sub, text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^\s*```+\{amsmath\}(.*?)\n\s*```+\s*$", _math_block_sub, text, flags=re.DOTALL | re.MULTILINE)

    # 9. Display Math: $$ ... $$ and \[ ... \]
    def _display_math_sub(m):
        raw = m.group(1).strip()
        if not raw:
            return ""
        sentinel = ctx.add_math(raw, is_block=True)
        return f"\n\n{sentinel}\n\n"

    text = re.sub(r"\$\$(.*?)\$\$", _display_math_sub, text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", _display_math_sub, text, flags=re.DOTALL)

    # 10. Figures: ```{figure} ... ```
    def _figure_sub(m):
        if opts.images == "drop":
            return ""
        caption = _clean_directive_options(m.group(2))
        if caption:
            return f"\n\nFigure: {caption}\n\n"
        return ""

    text = re.sub(r"^\s*```+\{figure\}[ \t]*([^\n]*)\n(.*?)\n\s*```+\s*$", _figure_sub, text, flags=re.DOTALL | re.MULTILINE)

    # 11. Tables: MyST {list-table} and {table}
    def _list_table_sub(m):
        title = m.group(1).strip()
        content = m.group(2)
        return _format_list_table(title, content, opts)

    def _table_directive_sub(m):
        title = m.group(1).strip()
        content = m.group(2)
        effective_title = _extract_directive_title(title, content)
        if opts.tables == "linearize":
            clean_content = _clean_directive_options(content)
            return _process_pipe_and_grid_tables(clean_content, opts)
        return _format_table_note(effective_title, opts)

    text = re.sub(r"^\s*```+\{list-table\}[ \t]*([^\n]*)\n(.*?)\n\s*```+\s*$", _list_table_sub, text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^\s*```+\{table\}[ \t]*([^\n]*)\n(.*?)\n\s*```+\s*$", _table_directive_sub, text, flags=re.DOTALL | re.MULTILINE)

    # 11b. HTML Tables: <table>...</table>
    text = _process_html_tables(text, opts)

    # 11c. Markdown Pipe Tables & Grid Tables
    text = _process_pipe_and_grid_tables(text, opts)

    # 12. Admonitions: ::: {admonition} Title ... ::: or ```{admonition} ... ```
    def _admonition_sub(m):
        title = m.group(1).strip()
        content = m.group(2).strip()
        if title:
            return f"\n\nNote: {title}.\n\n{content}\n\n"
        return f"\n\n{content}\n\n"

    text = re.sub(r"^\s*:::+\s*\{admonition\}[ \t]*([^\n]*)\n(.*?)\n\s*:::+\s*$", _admonition_sub, text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^\s*```+\{admonition\}[ \t]*([^\n]*)\n(.*?)\n\s*```+\s*$", _admonition_sub, text, flags=re.DOTALL | re.MULTILINE)

    # 13. Epigraphs
    def _epigraph_sub(m):
        return f"\n\n{m.group(1).strip()}\n\n"

    text = re.sub(r"^\s*```+\{epigraph\}(.*?)\n\s*```+\s*$", _epigraph_sub, text, flags=re.DOTALL | re.MULTILINE)

    # 14. Bibliography
    text = re.sub(r"^\s*```+\{bibliography\}.*?\n\s*```+\s*$", "", text, flags=re.DOTALL | re.MULTILINE)

    # 15. Code Blocks: ```{code-block} ... ```, generic ```lang ... ```, and tilde ~~~lang ... ~~~
    def _code_block_sub(m):
        if opts.code == "keep":
            code = m.group(2).strip()
            return f"\n\n{code}\n\n"
        elif opts.code == "placeholder":
            block_info = m.group(1) or ""
            caption = ""
            for line in m.group(2).splitlines()[:5]:
                if line.strip().startswith(":caption:"):
                    caption = line.split(":", 2)[-1].strip()
                    break
            desc = caption or block_info.strip() or "Code snippet"
            return f"\n\n[Code block: {desc}]\n\n"
        return ""

    text = re.sub(r"^\s*```+\{code-block\}[ \t]*([^\n]*)\n(.*?)\n\s*```+\s*$", _code_block_sub, text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^\s*```+([a-zA-Z0-9_\-+]*)\n(.*?)\n\s*```+\s*$", _code_block_sub, text, flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r"^\s*~~~+([a-zA-Z0-9_\-+]*)\n(.*?)\n\s*~~~+\s*$", _code_block_sub, text, flags=re.DOTALL | re.MULTILINE)

    # 16. Inline Math: \( ... \) and $ ... $
    def _inline_math_sub(m):
        raw = m.group(1).strip()
        if not raw:
            return ""
        return ctx.add_math(raw, is_block=False)

    text = re.sub(r"\\\((.*?)\\\)", _inline_math_sub, text, flags=re.DOTALL)
    text = re.sub(r"(?<![\$\\])\$(?!\$)([^\$\n]+?)(?<!\\)\$(?!\d)", _inline_math_sub, text)

    # 17. MyST Roles
    # 17a. {cite:p}`key` -> drop parenthetical citations for TTS narration
    text = re.sub(r"\{cite:p\}`[^`]+`", "", text)

    # 17b. {cite:t}`key` -> textual citations (e.g. Gelman2013 -> Gelman et al.)
    def _cite_t_sub(m):
        key = m.group(1).strip()
        author_m = re.match(r"^([A-Z][a-z]+)", key)
        if author_m:
            return f"{author_m.group(1)} et al."
        return key

    text = re.sub(r"\{cite:t\}`([^`]+)`", _cite_t_sub, text)

    # 17c. {numref}`label` -> Figure/Table label
    def _numref_sub(m):
        ref = m.group(1).strip()
        if ref.startswith("fig:"):
            name = ref[4:].replace("_", " ")
            return f"Figure {name}"
        elif ref.startswith("tab:"):
            name = ref[4:].replace("_", " ")
            return f"Table {name}"
        return ref.replace("_", " ")

    text = re.sub(r"\{numref\}`([^`]+)`", _numref_sub, text)

    # 17d. {ref}`label`
    def _ref_sub(m):
        ref = m.group(1).strip()
        return ref.replace("_", " ").replace(":", " ")

    text = re.sub(r"\{ref\}`([^`]+)`", _ref_sub, text)

    # 17e. {eq}`label`
    def _eq_sub(m):
        ref = m.group(1).strip()
        if ref.startswith("eq:"):
            name = ref[3:].replace("_", " ")
            return f"{name}"
        return "the equation"

    text = re.sub(r"\{eq\}`([^`]+)`", _eq_sub, text)

    # 18. Footnotes
    text = re.sub(r"\[\^[a-zA-Z0-9_\-]+\]", "", text)
    if not opts.keep_footnotes:
        text = re.sub(r"^\s*\[\^[a-zA-Z0-9_\-]+\]:[^\n]*(?:\n(?!\s*\[\^|\s*\n)[^\n]*)*", "", text, flags=re.MULTILINE)

    # 19. Images: ![alt](url) -> caption (or dropped)
    def _image_sub(m):
        alt = m.group(1).strip()
        if opts.images == "drop" or not alt:
            return ""
        return f"\n\nFigure: {alt}\n\n"

    text = re.sub(r"!\[([^\]]*)\]\(([^\)]+)\)", _image_sub, text)

    # 20. Links: [text](url) -> text
    def _link_sub(m):
        label = m.group(1).strip()
        url = m.group(2).strip()
        if not label or label.startswith("http") or label == url:
            return ""
        return label

    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", _link_sub, text)

    # 21. Bullet / numbered lists with terminal punctuation
    def _list_sub(m):
        content = m.group(1).strip()
        if content and content[-1] not in ".!?:;":
            content += "."
        return content

    text = re.sub(r"^\s*[-*+]\s+(.+)$", _list_sub, text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+(.+)$", _list_sub, text, flags=re.MULTILINE)

    # 22. Bold, italics, inline code, strikethrough, snake_case
    text = re.sub(r"\*\*([^\*\n]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^\*\n]+)\*", r"\1", text)
    text = re.sub(r"(?<!\w)__([^_\n]+)__(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", text)
    text = re.sub(r"~~([^~\n]+)~~", r"\1", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Replace remaining intra-word underscores in identifiers with space so TTS pronounces words
    text = re.sub(r"(?<=[a-zA-Z0-9])_(?=[a-zA-Z0-9])", " ", text)

    # 23. Headings: # Header -> Header.\n
    def _header_sub(m):
        if opts.headings == "drop":
            return ""
        h_text = m.group(1).strip()
        if not h_text:
            return ""
        if opts.tts_punct and h_text[-1] not in ".!?:":
            h_text += "."
        return f"\n\n{h_text}\n\n"

    text = re.sub(r"^\s*#{1,6}\s+(.+)$", _header_sub, text, flags=re.MULTILINE)

    # 24. Abbreviation expansion
    for pattern, repl in ABBREVIATIONS:
        text = pattern.sub(repl, text)

    # 25. HTML tags: replace blocks with double newlines, breaks with newline, inline formatting with empty string, others with space
    text = re.sub(r"</?(?:p|div|section|article|h[1-6]|li|blockquote)[^>]*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:sup|sub|em|strong|b|i|u|s|span|small|mark|del|ins)\b[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    return text


def normalize_paragraphs(text: str, opts) -> str:
    """Split into paragraphs, unwrap internal line breaks, normalize whitespace."""
    raw_paras = text.split("\n\n")
    cleaned_paras = []
    for p in raw_paras:
        p_clean = " ".join(line.strip() for line in p.splitlines() if line.strip())
        p_clean = re.sub(r"[ \t]+", " ", p_clean).strip()
        if not p_clean:
            continue
        if opts.numbers == "words":
            p_clean = convert_numbers(p_clean)
        # Fix redundant punctuation spacing
        p_clean = re.sub(r"\s+([,.;:!?])", r"\1", p_clean)
        # Preserve ellipses (...) while cleaning redundant punctuation
        p_clean = p_clean.replace("…", "...")
        p_clean = re.sub(r"\.{4,}", "...", p_clean)
        p_clean = re.sub(r"(?<!\.)\.\.(?!\.)", ".", p_clean)
        p_clean = re.sub(r"!{2,}", "!", p_clean)
        p_clean = re.sub(r"\?{2,}", "?", p_clean)
        p_clean = re.sub(r",+", ",", p_clean)
        p_clean = re.sub(r";+", ";", p_clean)
        p_clean = re.sub(r":+", ":", p_clean)
        cleaned_paras.append(p_clean)

    return "\n\n".join(cleaned_paras) + "\n"


def process(text: str, opts, stats: dict) -> str:
    stats.setdefault("equations", 0)
    stats.setdefault("converted", 0)
    stats.setdefault("errors", 0)

    ctx = Ctx(opts)
    preprocessed = process_myst_and_markdown(text, ctx)
    stats["equations"] += len(ctx.math)

    # Verbalize math expressions
    speech_map = convert_math(ctx, stats)

    def _sub_sentinel(m):
        idx = int(m.group(1))
        spoken = speech_map.get(idx, "")
        is_block = ctx.math[idx][1] if idx < len(ctx.math) else False
        if is_block:
            if spoken and spoken[-1] not in ".!?:":
                spoken += "."
            return f"\n\n{spoken}\n\n"
        return f" {spoken} "

    final_text = MATH_SENTINEL_RE.sub(_sub_sentinel, preprocessed)
    return normalize_paragraphs(final_text, opts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def gather(paths, recursive):
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out_base = re.sub(r"_(?:md|txt)$", "_txt", path.name)
            md_files = list(path.glob("**/*.md" if recursive else "*.md"))
            if md_files:
                matched = md_files
            else:
                matched = list(path.glob("**/*.txt" if recursive else "*.txt"))
            for f in sorted(matched):
                subdir = Path(out_base) / f.parent.relative_to(path)
                files.append((f, subdir))
        elif path.is_file():
            files.append((path, None))
        else:
            print(f"[warn] not found: {p}", file=sys.stderr)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("inputs", nargs="+", help="Markdown / MyST file(s) or directory(ies)")
    ap.add_argument("-o", "--output", help="Output file (one input) or directory")
    ap.add_argument("-r", "--recursive", action="store_true", help="Recurse for **/*.md")
    ap.add_argument("--stdout", action="store_true", help="Write to stdout")
    ap.add_argument("--math-api", default="http://localhost:8099", help="mathspeak API base URL (default: %(default)s)")
    ap.add_argument("--math-style", choices=["clearspeak", "mathspeak"], default="clearspeak", help="SRE speech style (default: %(default)s)")
    ap.add_argument("--on-math-error", choices=["fail", "keep", "fallback", "skip"], default="keep",
                    help="If math API fails: keep/fallback (default: local verbalizer), fail, or skip")
    ap.add_argument("--math-error-placeholder", default="")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--code", choices=["drop", "placeholder", "keep"], default="drop", help="Code block handling (default: %(default)s)")
    ap.add_argument("--images", choices=["caption", "drop"], default="caption", help="Figure handling (default: %(default)s)")
    ap.add_argument("--tables", default="note",
                    help="Table handling: note (default, 'See table <name>.'), drop, linearize, or a custom template like 'See table {name}.'")
    ap.add_argument("--table-template", default=None,
                    help="Custom template for table replacement notes, e.g. 'See table {name}.' or '[Table: {name}]' (default: 'See table {name}.')")
    ap.add_argument("--headings", choices=["keep", "drop"], default="keep", help="Heading handling (default: %(default)s)")
    ap.add_argument("--keep-footnotes", action="store_true", help="Render footnote definitions (default: drop)")
    ap.add_argument("--no-tts-punct", dest="tts_punct", action="store_false", help="Don't append periods to headings")
    ap.add_argument("--numbers", choices=["keep", "words"], default="words",
                    help="spell numbers out as words (default: words), or keep digits as-is")
    opts = ap.parse_args(argv)

    if opts.numbers == "words" and num2words is None:
        print("[error] --numbers words requires the num2words package: pip install num2words", file=sys.stderr)
        return 1

    if opts.math_api:
        ensure_math_api_running(opts.math_api)

    entries = gather(opts.inputs, opts.recursive)
    if not entries:
        print("No .md or .txt files found.", file=sys.stderr)
        return 1

    outdir = Path(opts.output) if opts.output and (len(entries) > 1 or Path(opts.output).is_dir()) else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    stats = {"equations": 0, "converted": 0, "errors": 0}
    for f, subdir in entries:
        raw = f.read_text(encoding="utf-8", errors="replace")
        out = process(raw, opts, stats)
        if opts.stdout:
            sys.stdout.write(out)
        else:
            if outdir:
                target_dir = (outdir / subdir) if (subdir and str(subdir) != ".") else outdir
                target_dir.mkdir(parents=True, exist_ok=True)
                dest = target_dir / (f.stem + ".txt")
            else:
                dest = Path(opts.output) if opts.output else f.with_suffix(".txt")
            dest.write_text(out, encoding="utf-8")
            print(f"{f}  ->  {dest}", file=sys.stderr)

    print(
        f"[done] {len(entries)} file(s) | {stats['equations']} equations "
        f"({stats['converted']} converted, {stats['errors']} errors)",
        file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
