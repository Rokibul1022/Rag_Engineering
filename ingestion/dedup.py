"""Duplicate detection at three levels:
 1. exact duplicate blocks inside one document (repeated boilerplate paragraphs)
 2. duplicate files (same bytes) -> handled in store.decide()
 3. near-duplicate chunks (same text with small edits), within a document or across documents
"""
from __future__ import annotations

import hashlib
import re


def text_key(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode("utf-8")).hexdigest()


def mark_exact_duplicates(blocks, min_chars: int = 40) -> list:
    """Flag later copies of an identical block. Short blocks (e.g. 'Yes') are never treated as duplicates."""
    seen, found = {}, []
    for i, b in enumerate(blocks):
        if b.removed_reason or b.type == "heading" or len(b.text) < min_chars:
            continue
        k = text_key(b.text)
        if k in seen:
            b.removed_reason = f"exact duplicate of block #{seen[k]}"
            found.append({"block": i, "duplicate_of": seen[k], "preview": b.text[:90]})
        else:
            seen[k] = i
    return found


def shingles(text: str, k: int = 5) -> set:
    words = re.findall(r"\w+", text.lower())
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def find_near_duplicates(chunks, others, threshold: float = 0.8, limit: int = 20) -> list:
    """chunks: this run's Chunk objects. others: dicts from the store (other documents' active chunks)."""
    mine = [c for c in chunks if c.kind != "parent"]
    sh = [shingles(c.text) for c in mine]
    pairs = []
    for i in range(len(mine)):
        for j in range(i + 1, len(mine)):
            s = jaccard(sh[i], sh[j])
            if s >= threshold:
                pairs.append({"scope": "same document", "a": mine[i].chunk_id, "b": mine[j].chunk_id, "similarity": round(s, 2),
                              "preview": mine[i].text[:90]})
    for i, c in enumerate(mine):
        for o in others:
            s = jaccard(sh[i], shingles(o["text"]))
            if s >= threshold:
                pairs.append({"scope": f"other document: {o['source_name']}", "a": c.chunk_id, "b": o["chunk_id"],
                              "similarity": round(s, 2), "preview": c.text[:90]})
    return sorted(pairs, key=lambda p: -p["similarity"])[:limit]
