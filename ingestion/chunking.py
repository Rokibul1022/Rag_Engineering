"""Three chunking strategies you can compare side by side:

  structure : follow headings/tables/lists; parent (section) + child (small chunk) relationship   <- default
  fixed     : cut every N tokens with overlap, ignoring structure (the naive baseline)
  semantic  : cut where the topic changes (lexical similarity between neighbouring sentences)

"Tokens" here are counted with a simple regex (words and punctuation). Real LLM tokenizers differ by
roughly 20-40%, but for learning and for comparing strategies the regex is transparent and dependency-free.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .models import Chunk
from .utils import table_to_text

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENT_RE = re.compile(r"(?<=[.!?।])\s+(?=[\"'(\[]?[A-Z0-9\u0980-\u09FF])")
TERMINAL = (".", "!", "?", "।", '"', "'", ")", ":")
STOP = set("a an the and or but if of to in on at for with by from as is are was were be been it this that these those "
           "we our you your they their he she his her its not no so than then there which who whom will would can could "
           "should may might do does did has have had into over under about also more most such".split())


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def make_chunk_id(doc_id, kind, path, text, used: set) -> str:
    """Content-addressed ID: same document + same section path + same text => same ID in every version.
    That property is what makes incremental re-indexing possible."""
    h = hashlib.sha1(f"{doc_id}|{kind}|{' > '.join(path)}|{_norm(text)}".encode("utf-8")).hexdigest()[:12]
    cid, n = "c_" + h, 1
    while cid in used:           # identical text twice in the same place: disambiguate deterministically
        n += 1
        cid = f"c_{h}-{n}"
    used.add(cid)
    return cid


def split_sentences(text: str) -> list:
    return [s.strip() for s in SENT_RE.split(text.strip()) if s.strip()]


def hard_split(text: str, max_tokens: int) -> list:
    """Last resort for a 'sentence' longer than max_tokens: cut by token windows."""
    spans = [m.span() for m in TOKEN_RE.finditer(text)]
    return [text[spans[i][0]:spans[min(i + max_tokens, len(spans)) - 1][1]] for i in range(0, len(spans), max_tokens)]


def _pages(idx_pages):
    pages = [p for p in idx_pages if p]
    return (min(pages), max(pages)) if pages else (None, None)


def _embed_text(path, text):
    return (" > ".join(path) + "\n" + text) if path else text


# ============================================================ 1) structure-aware
def build_sections(blocks):
    """Group blocks under their heading path, e.g. ['Report', 'Finance', 'Revenue']."""
    sections, stack = [], []
    cur = {"path": [], "blocks": []}
    for idx, b in enumerate(blocks):
        if b.removed_reason:
            continue
        if b.type == "heading":
            if cur["blocks"]:
                sections.append(cur)
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            stack.append((b.level, b.text))
            cur = {"path": [t for _, t in stack], "blocks": []}
        else:
            cur["blocks"].append((idx, b))
    if cur["blocks"]:
        sections.append(cur)
    return sections


def _to_units(body, max_tokens):
    """Smallest pieces we are allowed to split at: whole paragraphs, whole lists, sentences, atomic tables."""
    units, i = [], 0
    while i < len(body):
        idx, b = body[i]
        if b.type == "list_item":
            j = i
            while j < len(body) and body[j][1].type == "list_item":
                j += 1
            items = body[i:j]
            whole = "\n".join("- " + x.text for _, x in items)
            if count_tokens(whole) <= max_tokens:
                units.append({"text": whole, "type": "list", "idxs": [k for k, _ in items], "page": items[0][1].page, "para": idx})
            else:
                for k, x in items:
                    units.append({"text": "- " + x.text, "type": "list", "idxs": [k], "page": x.page, "para": k})
            i = j
            continue
        if b.type == "table":
            units.append({"text": b.text, "type": "table", "idxs": [idx], "page": b.page, "para": idx, "rows": b.meta.get("rows")})
        elif count_tokens(b.text) <= max_tokens:
            units.append({"text": b.text, "type": "paragraph", "idxs": [idx], "page": b.page, "para": idx})
        else:
            for s in split_sentences(b.text):
                for piece in ([s] if count_tokens(s) <= max_tokens else hard_split(s, max_tokens)):
                    units.append({"text": piece, "type": "sentence", "idxs": [idx], "page": b.page, "para": idx})
        i += 1
    return units


def _split_table(u, max_tokens):
    if count_tokens(u["text"]) <= max_tokens or not u.get("rows") or len(u["rows"]) < 3:
        return [u]
    header, rows = u["rows"][0], u["rows"][1:]
    parts, cur = [], []
    for r in rows:
        if cur and count_tokens(table_to_text([header] + cur + [r])) > max_tokens:
            parts.append(cur)
            cur = []
        cur.append(r)
    if cur:
        parts.append(cur)
    return [dict(u, text=table_to_text([header] + p), table_part=k + 1, table_parts=len(parts)) for k, p in enumerate(parts)]


def _join_units(units):
    out = ""
    for k, u in enumerate(units):
        if k:
            out += " " if (units[k - 1]["para"] == u["para"] and u["type"] == "sentence") else "\n\n"
        out += u["text"]
    return out


def _pack(units, max_tokens, overlap, min_tokens):
    """Greedy packing of units into chunks of <= max_tokens, with sentence-level overlap."""
    groups, cur, cur_t, cur_ov = [], [], 0, 0

    def close():
        nonlocal cur, cur_t, cur_ov
        if cur:
            groups.append({"units": cur, "overlap": cur_ov})
        cur, cur_t, cur_ov = [], 0, 0

    for u in units:
        if u["type"] == "table":
            close()
            for piece in _split_table(u, max_tokens):
                groups.append({"units": [piece], "overlap": 0})
            continue
        t = count_tokens(u["text"])
        if cur and cur_t + t > max_tokens:
            tail, tt = [], 0
            for pu in reversed(cur):
                pt = count_tokens(pu["text"])
                if pu["type"] not in ("sentence", "paragraph") or tt + pt > overlap:
                    break
                tail.insert(0, pu)
                tt += pt
            close()
            if tt + t <= max_tokens:
                cur, cur_t, cur_ov = tail, tt, tt
        cur.append(u)
        cur_t += t
    close()
    # a tiny last chunk is merged into its neighbour
    if len(groups) >= 2:
        last, prev = groups[-1], groups[-2]
        if (all(u["type"] != "table" for u in last["units"] + prev["units"])
                and count_tokens(_join_units(last["units"])) < min_tokens):
            extra = [u for u in last["units"] if not any(u is x for x in prev["units"])]
            if count_tokens(_join_units(prev["units"] + extra)) <= max_tokens:
                prev["units"] += extra
                groups.pop()
    return groups


def structure_chunks(blocks, doc_id, max_tokens=200, overlap=30, min_tokens=25, parent_max_tokens=800):
    chunks, used = [], set()
    for sec in build_sections(blocks):
        path = sec["path"]
        body = [(i, b) for i, b in sec["blocks"] if b.type != "footnote"]
        notes = [(i, b) for i, b in sec["blocks"] if b.type == "footnote"]

        # --- parents: the whole section (split if it is very long)
        parents, cur, t = [], [], 0
        for i, b in body + notes:
            bt = count_tokens(b.text)
            if cur and t + bt > parent_max_tokens:
                parents.append(cur)
                cur, t = [], 0
            cur.append((i, b))
            t += bt
        if cur:
            parents.append(cur)
        parent_objs = []
        for pb in parents:
            text = "\n\n".join(b.text for _, b in pb)
            p0, p1 = _pages([b.page for _, b in pb])
            pc = Chunk(make_chunk_id(doc_id, "parent", path, text, used), "parent", text, _embed_text(path, text), count_tokens(text),
                       path, p0, p1, None, [i for i, _ in pb], sorted({b.type for _, b in pb}), {"children": 0})
            chunks.append(pc)
            parent_objs.append((pc, {i for i, _ in pb}))

        def parent_for(idx):
            for pc, idxs in parent_objs:
                if idx in idxs:
                    return pc
            return parent_objs[-1][0]

        # --- children: small chunks that respect paragraph/list/table boundaries
        child_groups = _pack(_to_units(body, max_tokens), max_tokens, overlap, min_tokens)
        if notes:   # footnotes become their own chunk(s), packed note by note so the ceiling holds
            note_groups, cur_n, cur_t = [], [], 0
            for i, b in notes:
                label = (f"[{b.meta['id']}] " if b.meta.get("id") else "") + b.text
                t = count_tokens(label)
                if cur_n and cur_t + t > max_tokens:
                    note_groups.append(cur_n)
                    cur_n, cur_t = [], 0
                cur_n.append((i, b, label))
                cur_t += t
            if cur_n:
                note_groups.append(cur_n)
            for ng in note_groups:
                child_groups.append({"units": [{"text": "\n".join(l for _, _, l in ng), "type": "footnote", "idxs": [i for i, _, _ in ng],
                                                "page": ng[0][1].page, "para": -1}], "overlap": 0})
        for g in child_groups:
            us = g["units"]
            text = _join_units(us)
            idxs = sorted({i for u in us for i in u["idxs"]})
            p0, p1 = _pages([u["page"] for u in us])
            par = parent_for(idxs[0])
            meta = {"overlap_tokens": g["overlap"]}
            if us[0].get("table_part"):
                meta["table_part"] = f"{us[0]['table_part']}/{us[0]['table_parts']}"
            if us[0]["type"] == "footnote":
                meta["footnotes"] = True
            c = Chunk(make_chunk_id(doc_id, "child", path, text, used), "child", text, _embed_text(path, text), count_tokens(text), path, p0, p1,
                      par.chunk_id, idxs, sorted({blocks[i].type for i in idxs}), meta)
            par.meta["children"] += 1
            chunks.append(c)
    return chunks


# ============================================================ 2) fixed token windows
def fixed_chunks(blocks, doc_id, max_tokens=200, overlap=30):
    parts, spans, pos = [], [], 0
    for idx, b in enumerate(blocks):
        if b.removed_reason:
            continue
        parts.append(b.text)
        spans.append((pos, pos + len(b.text), b.page))
        pos += len(b.text) + 2
    full = "\n\n".join(parts)
    toks = [m.span() for m in TOKEN_RE.finditer(full)]
    chunks, used, i, step = [], set(), 0, max(1, max_tokens - overlap)
    while i < len(toks):
        j = min(i + max_tokens, len(toks))
        s, e = toks[i][0], toks[j - 1][1]
        text = full[s:e]
        p0, p1 = _pages([pg for (a, b_, pg) in spans if a < e and b_ > s])
        chunks.append(Chunk(make_chunk_id(doc_id, "flat", [], text, used), "flat", text, text, j - i, [], p0, p1, None, [], [],
                            {"overlap_tokens": 0 if i == 0 else overlap}))
        if j == len(toks):
            break
        i += step
    return chunks


# ============================================================ 3) semantic boundaries
def _tf(text):
    return Counter(w for w in re.findall(r"\w+", text.lower()) if w not in STOP and len(w) > 2)


def _cos(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a if k in b)
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def semantic_chunks(blocks, doc_id, max_tokens=200, min_tokens=25, percentile=25, window=2):
    """Cut where neighbouring sentences share the fewest content words (a cheap stand-in for embeddings).
    To upgrade: replace _tf/_cos with sentence embeddings + cosine similarity. Everything else stays the same."""
    chunks, used = [], set()
    runs, cur = [], []
    for idx, b in enumerate(blocks):
        if b.removed_reason:
            continue
        if b.type == "table":
            if cur:
                runs.append(("text", cur)); cur = []
            runs.append(("table", [(idx, b, b.text)]))
        else:
            for s in (split_sentences(b.text) or [b.text]):
                cur.append((idx, b, s))
    if cur:
        runs.append(("text", cur))

    def emit(items, meta=None):
        text = " ".join(s for _, _, s in items) if all(i[1].type != "table" for i in items) else items[0][2]
        idxs = sorted({i for i, _, _ in items})
        p0, p1 = _pages([b.page for _, b, _ in items])
        chunks.append(Chunk(make_chunk_id(doc_id, "flat", [], text, used), "flat", text, text, count_tokens(text), [], p0, p1, None,
                            idxs, sorted({b.type for _, b, _ in items}), meta or {}))

    for kind, items in runs:
        if kind == "table":
            emit(items)
            continue
        vecs = [_tf(s) for _, _, s in items]
        sims = [0.0] * len(items)
        for i in range(1, len(items)):
            left, right = Counter(), Counter()
            for v in vecs[max(0, i - window):i]: left.update(v)
            for v in vecs[i:i + window]: right.update(v)
            sims[i] = _cos(left, right)
        ordered = sorted(sims[1:])
        thr = ordered[int(len(ordered) * percentile / 100)] if ordered else 0
        group, gt = [], 0
        for i, it in enumerate(items):
            t = count_tokens(it[2])
            if group and (gt + t > max_tokens or (sims[i] <= thr and gt >= min_tokens)):
                emit(group, {"boundary_similarity": round(sims[i], 2)})
                group, gt = [], 0
            group.append(it)
            gt += t
        if group:
            emit(group)
    return chunks


STRATEGIES = {"structure": "Structure-aware (parent-child)", "fixed": "Fixed token windows", "semantic": "Semantic boundaries"}


def chunk_document(blocks, doc_id, strategy="structure", max_tokens=200, overlap=30):
    if strategy == "fixed":
        return fixed_chunks(blocks, doc_id, max_tokens, overlap)
    if strategy == "semantic":
        return semantic_chunks(blocks, doc_id, max_tokens)
    return structure_chunks(blocks, doc_id, max_tokens, overlap)


def chunk_stats(chunks) -> dict:
    leaf = [c for c in chunks if c.kind != "parent"]
    toks = [c.tokens for c in leaf] or [0]
    mid = sum(1 for c in leaf if "table" not in c.block_types and not c.text.rstrip().endswith(TERMINAL) and c.text.rstrip()[-1:].isalnum())
    return {"total": len(chunks), "parents": sum(1 for c in chunks if c.kind == "parent"), "retrievable": len(leaf),
            "min_tokens": min(toks), "avg_tokens": round(sum(toks) / len(toks), 1), "max_tokens": max(toks),
            "cut_mid_sentence": mid, "split_tables": sum(1 for c in leaf if "table_part" in c.meta),
            "with_overlap": sum(1 for c in leaf if c.meta.get("overlap_tokens"))}
