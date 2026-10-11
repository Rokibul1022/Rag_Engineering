"""Normalization: make text from different sources look the same, so equal content gets equal hashes
and embeddings are not distorted by layout noise. Every rule is small, named and logged."""
from __future__ import annotations

import copy
import re
import unicodedata

from .models import Block
from .utils import LIST_RE

_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200e\u200f\u2060\ufeff\u00ad]")
# NOTE: we keep U+200C / U+200D (ZWNJ / ZWJ): Bangla, Persian etc. need them to render words correctly.
_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201c": '"', "\u201d": '"', "\u201e": '"',
           "\u2026": "...", "\u00a0": " ", "\u2009": " ", "\u202f": " "}


def r_unicode(text, b):          # ligatures (ﬁ -> fi), full-width chars, odd compatibility forms
    return unicodedata.normalize("NFKC", text)


def r_dehyphenate(text, b):      # "infra-\nstructure" -> "infrastructure"
    if b.type == "table":
        return text
    return re.sub(r"(\w)[-\u00ad]\s*\n\s*(\w)", lambda m: m.group(1) + m.group(2) if m.group(2).islower() else m.group(0), text)


def r_control(text, b):          # invisible characters that break search and hashing
    return _CONTROL_RE.sub("", text)


def r_punct(text, b):            # curly quotes -> straight quotes, NBSP -> space
    return "".join(_QUOTES.get(c, c) for c in text)


def r_bullets(text, b):          # "• item" / "1. item" -> "item" (marker stored in metadata)
    if b.type == "list_item":
        m = LIST_RE.match(text)
        if m:
            b.meta["marker"] = m.group(0).strip()
            return text[m.end():]
    return text


def r_join_lines(text, b):       # hard line breaks inside a paragraph -> spaces (tables keep their rows)
    return text if b.type in ("table", "code") else re.sub(r"\s*\n\s*", " ", text)


def r_whitespace(text, b):
    if b.type in ("table", "code"):
        return "\n".join(re.sub(r"[ \t]+", " ", ln).rstrip() for ln in text.splitlines()).strip()
    return re.sub(r"\s+", " ", text).strip()


def r_ocr_digits(text, b):       # OCR often reads 0 as O and 1 as l inside numbers: "1O25" -> "1025"
    if not b.meta.get("ocr"):
        return text
    table = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "S": "5"})

    def fix(tok):
        core = tok.strip(".,;:()$%€£")
        digits = sum(c.isdigit() for c in core)
        if core and digits >= 2 and digits >= len(core) - 2:
            return tok.replace(core, core.translate(table))
        return tok
    return " ".join(fix(t) for t in text.split(" "))


RULES = [
    ("unicode_nfkc", "Unicode NFKC: ligatures (ﬁ→fi), full-width and compatibility characters", r_unicode),
    ("dehyphenate", "Join words split by a line-end hyphen (infra-\\nstructure → infrastructure)", r_dehyphenate),
    ("strip_invisible", "Remove control characters, zero-width spaces and soft hyphens (keeps ZWJ/ZWNJ for Bangla)", r_control),
    ("straighten_punctuation", "Curly quotes → straight quotes, NBSP → normal space, … → ...", r_punct),
    ("strip_list_markers", "Remove bullet/number markers from list items (kept in metadata)", r_bullets),
    ("join_wrapped_lines", "Turn hard line wraps inside a paragraph into spaces", r_join_lines),
    ("collapse_whitespace", "Collapse repeated spaces/tabs, trim ends", r_whitespace),
    ("ocr_digit_fix", "OCR only: fix letters that look like digits inside numbers (O→0, l→1)", r_ocr_digits),
]


def normalize_blocks(blocks):
    """Returns (new_blocks, rules_summary). Originals are not modified."""
    out, counts = [], {name: 0 for name, _, _ in RULES}
    for src in blocks:
        b = copy.deepcopy(src)
        if b.removed_reason:
            out.append(b)
            continue
        before = b.text
        applied = []
        for name, _, fn in RULES:
            new = fn(b.text, b)
            if new != b.text:
                counts[name] += 1
                applied.append(name)
                b.text = new
        if applied:
            b.meta["before"] = before
            b.meta["rules"] = applied
        if not b.text.strip():
            b.removed_reason = "empty after normalization"
        elif b.meta.get("ocr") and len(b.text) <= 30:
            alnum = sum(c.isalnum() for c in b.text) / len(b.text)
            if alnum < 0.4:
                b.removed_reason = "OCR noise (mostly symbols)"
        out.append(b)
    summary = [{"rule": n, "description": d, "blocks_changed": counts[n]} for n, d, _ in RULES]
    return out, summary
