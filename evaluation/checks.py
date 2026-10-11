"""Reference-free integrity checks: they need no answer key, so they run on ANY document you upload.

Each check answers one question about validity and returns pass / warn / fail / n/a with the measured value.
"""
from __future__ import annotations

import re
from collections import Counter

from ingestion.chunking import chunk_document
from ingestion.utils import table_to_text

from .metrics import norm_ws, words

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"


def _c(group, cid, name, status, value, detail):
    return {"group": group, "id": cid, "name": name, "status": status, "value": value, "detail": detail}


def _grade(x, ok, warn):          # higher is better
    return PASS if x >= ok else WARN if x >= warn else FAIL


def run_checks(*, parsed, blocks, chunks, chunk_dicts, strategy, max_tokens, overlap, doc_id, file_type, near_dups) -> dict:
    """blocks = normalized blocks (with removed flags); parsed = original ParseResult."""
    out = []
    active = [b for b in blocks if not b.removed_reason]
    leaf = [c for c in chunks if c.kind != "parent"]
    plain = "\n\n".join(b.text for b in active)

    # ============================================================ PARSING
    raw = re.sub(r"^--- page \d+ ---$", " ", parsed.raw_text, flags=re.M)
    raw = re.sub(r"\(cid:\d+\)", " ", raw)
    raw_words = {w for w in words(raw) if len(w) > 1} - set(words(parsed.dropped_text))
    got_words = set(words(" ".join(b.text for b in parsed.blocks)))
    if raw_words:
        rec = len(raw_words & got_words) / len(raw_words)
        missing = sorted(raw_words - got_words)[:8]
        out.append(_c("Parsing", "parse_recall", "No text lost in parsing", _grade(rec, 0.98, 0.9), f"{rec:.1%} of raw words kept",
                      "Every word of the naive extraction should still exist in the parsed blocks." + (f" Missing e.g.: {', '.join(missing)}" if missing else "")))
    errs = [w for w in parsed.warnings if w["level"] == "error"]
    warns = [w for w in parsed.warnings if w["level"] == "warning"]
    out.append(_c("Parsing", "parse_warnings", "Parser reported no problems", FAIL if errs else WARN if warns else PASS,
                  f"{len(errs)} error(s), {len(warns)} warning(s)", "; ".join(w["code"] for w in errs + warns) or "Clean parse."))
    n_head = sum(b.type == "heading" for b in active)
    out.append(_c("Parsing", "headings", "Document structure detected", PASS if n_head else WARN, f"{n_head} heading(s)",
                  "Structure-aware chunking needs headings; without them the whole document is one section." if not n_head else "Headings found."))

    # ============================================================ NORMALIZATION
    issues = {"control/invisible chars": len(re.findall("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u00ad\ufeff]", plain)),
              "(cid:N) glyphs": len(re.findall(r"\(cid:\d+\)", plain)),
              "curly quotes / NBSP": len(re.findall("[\u2018\u2019\u201c\u201d\u00a0]", plain)),
              "line-end hyphens": len(re.findall(r"\w-\n\w", plain)),
              "repeated spaces": len(re.findall(r"[^\S\n]{2,}", plain))}
    bad = {k: v for k, v in issues.items() if v}
    out.append(_c("Normalization", "clean_text", "No residual text noise", FAIL if bad else PASS, f"{sum(issues.values())} issue(s)",
                  ", ".join(f"{k}: {v}" for k, v in bad.items()) or "No control characters, cid glyphs, curly quotes, hyphen breaks or double spaces left."))
    before = sum(sum(c.isalnum() for c in b.meta.get("before", b.text)) for b in active)
    after = sum(sum(c.isalnum() for c in b.text) for b in active)
    if before:
        r = after / before
        out.append(_c("Normalization", "no_loss", "Normalization removed no content", _grade(r, 0.98, 0.95), f"{r:.1%} of letters/digits kept",
                      "Only formatting (hyphens, spaces, list markers) may disappear, not letters or digits."))

    # ============================================================ OCR
    ocr_blocks = [b for b in active if b.meta.get("ocr")]
    if ocr_blocks:
        conf = parsed.doc_meta.get("ocr_mean_confidence") or 0
        low = sum(len(b.meta.get("low_conf_words", [])) for b in ocr_blocks)
        out.append(_c("OCR", "ocr_conf", "OCR confidence is acceptable", _grade(conf, 85, 70), f"mean {conf}%, {low} low-confidence word(s)",
                      "Below 85% expect visible errors; below 70% re-scan or choose another language pack."))
    else:
        out.append(_c("OCR", "ocr_conf", "OCR confidence is acceptable", NA, "no OCR used", "The file has a text layer."))

    # ============================================================ CHUNKING
    chunk_words = Counter(w for c in leaf for w in words(c.text))
    src_words = Counter(w for b in active if b.type != "heading" for w in words(b.text))
    if src_words:
        kept = sum(min(n, chunk_words[w]) for w, n in src_words.items()) / sum(src_words.values())
        out.append(_c("Chunking", "coverage", "Every word is in some chunk", _grade(kept, 0.995, 0.97), f"{kept:.1%} coverage",
                      "Chunking must not drop content (headings are excluded: they live in the section path)."))

    norm_chunks = [norm_ws(c.text).lower() for c in leaf]
    active_texts = {norm_ws(b.text).lower() for b in active}
    leaks = []
    for b in blocks:
        if b.removed_reason and b.type in ("header", "footer", "boilerplate") and not b.removed_reason.startswith("exact duplicate"):
            t = norm_ws(b.text).lower()
            if len(t) >= 10 and t not in active_texts and any(t in c for c in norm_chunks):
                leaks.append(b.text[:50])
    out.append(_c("Chunking", "no_leak", "Removed noise is absent from chunks", FAIL if leaks else PASS, f"{len(leaks)} leak(s)",
                  ("Still present: " + "; ".join(leaks)) if leaks else "Headers, footers, menus and banners never reach a chunk."))

    over = [c for c in leaf if c.tokens > max_tokens and not (strategy == "semantic" and "table" in c.block_types)]
    out.append(_c("Chunking", "size", f"Chunks respect the {max_tokens}-token limit", FAIL if over else PASS, f"{len(over)} over limit",
                  f"Largest: {max((c.tokens for c in leaf), default=0)} tokens." + (" (A semantic-chunk table is kept atomic on purpose.)" if strategy == "semantic" else "")))

    tiny = [c for c in leaf if c.tokens < 10 and not c.meta.get("footnotes") and "table" not in c.block_types]
    ratio = len(tiny) / len(leaf) if leaf else 0
    out.append(_c("Chunking", "tiny", "Few tiny chunks", PASS if ratio <= 0.1 else WARN if ratio <= 0.25 else FAIL, f"{len(tiny)} of {len(leaf)} under 10 tokens",
                  "Tiny chunks carry little meaning and crowd search results."))

    # chunks without block-type labels (fixed windows) mix everything, so they count as prose too
    prose = [c for c in leaf if not (c.block_types and set(c.block_types) <= {"table", "footnote", "list_item", "heading"})]
    ends_in_list = lambda c: c.text.rstrip().splitlines()[-1].startswith("- ")      # a bullet without a full stop is not a cut
    mid = [c for c in prose if c.text.rstrip()[-1:].isalnum() and not ends_in_list(c)]
    mr = len(mid) / len(prose) if prose else 0
    out.append(_c("Chunking", "mid_sentence", "Chunks end at sentence boundaries", PASS if mr <= 0.1 else WARN if mr <= 0.4 else FAIL, f"{len(mid)} of {len(prose)} cut mid-sentence",
                  "Structure and semantic strategies should rarely cut mid-sentence; fixed windows often do."))

    if strategy == "structure":
        heads = {i for i, b in enumerate(blocks) if b.type == "heading" and not b.removed_reason}
        cross = [c for c in leaf if c.block_indices and any(min(c.block_indices) < h < max(c.block_indices) for h in heads)]
        out.append(_c("Chunking", "heading_integrity", "No chunk crosses a heading", FAIL if cross else PASS, f"{len(cross)} violation(s)", "Chunks must stay inside one section."))
        parents = {c.chunk_id: c for c in chunks if c.kind == "parent"}
        kids = [c for c in chunks if c.kind == "child"]
        orphans = [c for c in kids if c.parent_id not in parents]
        mism = [c for c in kids if c.parent_id in parents and parents[c.parent_id].section_path != c.section_path]
        pw = {k: Counter(words(p.text)) for k, p in parents.items()}
        tot = hit = 0
        for c in kids:
            cw = Counter(words(c.text))
            tot += sum(cw.values())
            hit += sum(min(n, pw.get(c.parent_id, Counter())[w]) for w, n in cw.items())
        rec = hit / tot if tot else 1.0
        status = FAIL if orphans or mism else _grade(rec, 0.97, 0.9)
        out.append(_c("Chunking", "parent_child", "Parent-child links are valid", status, f"{len(orphans)} orphan(s), {len(mism)} mismatch(es), {rec:.1%} of child words inside parent",
                      "Every child must point to an existing parent of the same section whose text contains it."))
        no_ctx = [c for c in chunks if c.section_path and not c.embed_text.startswith(" > ".join(c.section_path))]
        out.append(_c("Chunking", "context", "Embedded text carries the section path", FAIL if no_ctx else PASS, f"{len(no_ctx)} missing", "embed_text = 'Section > Subsection' + text."))
    else:
        for cid, name in (("heading_integrity", "No chunk crosses a heading"), ("parent_child", "Parent-child links are valid")):
            out.append(_c("Chunking", cid, name, NA, "structure strategy only", "Switch the strategy to 'structure' to evaluate this."))

    tables = [(i, b) for i, b in enumerate(blocks) if b.type == "table" and not b.removed_reason and b.meta.get("rows")]
    if tables:
        lines_all = {ln for c in leaf for ln in c.text.splitlines()}
        total = intact = no_header = 0
        for i, b in tables:
            # normalization collapses repeated spaces, so compare against the collapsed form of the expected rows
            t = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in table_to_text(b.meta["rows"]).splitlines()]
            header, data = t[0], t[2:]
            total += len(data)
            intact += sum(1 for ln in data if ln in lines_all)
            for c in leaf:
                ls = c.text.splitlines()
                if any(ln in data for ln in ls) and header not in ls:
                    no_header += 1
        frac = intact / total if total else 1.0
        status = PASS if frac == 1 and not no_header else WARN if frac == 1 else FAIL
        out.append(_c("Chunking", "tables", "Tables stay intact", status, f"{intact}/{total} rows whole, {no_header} part(s) without header",
                      "Every table row must be inside one chunk; split tables must repeat the header row."))
    else:
        out.append(_c("Chunking", "tables", "Tables stay intact", NA, "no tables", "This document has no tables."))

    again = [c.chunk_id for c in chunk_document(blocks, doc_id, strategy, max_tokens, overlap)]
    same = again == [c.chunk_id for c in chunks]
    out.append(_c("Identity", "deterministic", "Same input gives the same chunk IDs", PASS if same else FAIL, "identical" if same else "IDs changed between runs",
                  "Required for incremental re-indexing."))
    ids = [c.chunk_id for c in chunks]
    out.append(_c("Identity", "unique_ids", "Chunk IDs are unique", PASS if len(ids) == len(set(ids)) else FAIL, f"{len(ids) - len(set(ids))} collision(s)", ""))
    dups = [n for n, k in Counter(norm_chunks).items() if k > 1 and len(n) > 20]
    out.append(_c("Identity", "dup_chunks", "No duplicate chunks", PASS if not dups and not near_dups else WARN, f"{len(dups)} exact, {len(near_dups)} near-duplicate",
                  "Duplicates fill the top-k with the same passage."))
    missing = []
    for d in chunk_dicts:
        if not d.get("citation") or d["tokens"] <= 0 or not d["source"].get("doc_id"):
            missing.append(d["chunk_id"])
        elif d["kind"] == "child" and not d.get("parent_id"):
            missing.append(d["chunk_id"])
        elif file_type in ("pdf", "image") and d["page_start"] is None and d["kind"] != "flat":
            missing.append(d["chunk_id"])
    out.append(_c("Identity", "metadata", "Every chunk has complete metadata", PASS if not missing else FAIL, f"{len(missing)} incomplete",
                  "doc_id, citation, tokens, parent link and (for PDFs/images) page number."))

    scored = [c for c in out if c["status"] != NA]
    pts = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}
    score = round(100 * sum(pts[c["status"]] for c in scored) / len(scored)) if scored else None
    return {"score": score, "passed": sum(c["status"] == PASS for c in out), "warned": sum(c["status"] == WARN for c in out),
            "failed": sum(c["status"] == FAIL for c in out), "checks": out}
