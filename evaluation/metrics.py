"""Small, dependency-free metrics."""
from __future__ import annotations

import re


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def words(s: str) -> list:
    return re.findall(r"\w+", (s or "").lower())


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate: 0.0 = perfect. Used for OCR."""
    ref, hyp = norm_ws(ref), norm_ws(hyp)
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    """Word error rate."""
    r, h = norm_ws(ref).split(), norm_ws(hyp).split()
    return edit_distance(r, h) / max(1, len(r))


def prf(expected: set, got: set):
    """precision, recall, F1 of two sets."""
    tp = len(expected & got)
    p = tp / len(got) if got else (1.0 if not expected else 0.0)
    r = tp / len(expected) if expected else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f, 3)
