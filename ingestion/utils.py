"""Tiny helpers used by several modules."""
from __future__ import annotations

import re

# "- item", "* item", "• item", "1. item", "2) item", "a) item"
LIST_RE = re.compile(r"^\s*(?:[\u2022\u25CF\u25AA\u2023\u2043*\-]|\d{1,3}[.)]|[A-Za-z][.)])\s+")


def table_to_text(rows: list) -> str:
    """Render a table (list of rows) as a Markdown table. LLMs read this format well."""
    rows = [[(c or "").strip() for c in r] for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    esc = lambda c: c.replace("|", "\\|")
    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |",
             "| " + " | ".join("---" for _ in rows[0]) + " |"]
    for r in rows[1:]:
        lines.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(lines)


def decode_bytes(data: bytes):
    """Decode text safely. Returns (text, warnings). Never raises."""
    from .models import warn
    try:
        return data.decode("utf-8-sig"), []
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), [warn("warning", "ENCODING_FALLBACK",
                                            "File is not valid UTF-8; decoded as Windows-1252. Check for garbled characters.")]
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace"), [warn("warning", "ENCODING_FALLBACK",
                                                               "Unknown encoding; decoded as Latin-1 with replacement characters.")]
