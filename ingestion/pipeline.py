"""The pipeline: one function that runs every stage and returns a JSON-friendly report of what happened."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .chunking import STRATEGIES, chunk_document, chunk_stats
from .dedup import find_near_duplicates, mark_exact_duplicates
from .normalize import normalize_blocks
from .parsers import parse_file
from .store import Store, citation


def make_doc_id(source_name: str) -> str:
    """Stable ID = hash of the source name. Same source => same ID forever, whatever its content.
    (In production use something that never changes: a URL, a DMS id, a relative path.)"""
    return "doc_" + hashlib.sha1(source_name.strip().lower().encode("utf-8")).hexdigest()[:10]


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000, 1)


def run_pipeline(path, source_name=None, strategy="structure", max_tokens=200, overlap=30, save=True,
                 ocr_lang="eng", store: Store | None = None, original_filename=None) -> dict:
    path = Path(path)
    timings = {}
    data = path.read_bytes()
    file_hash = hashlib.sha256(data).hexdigest()
    filename = original_filename or path.name
    source_name = (source_name or filename).strip()
    doc_id = make_doc_id(source_name)
    params = {"max_tokens": int(max_tokens), "overlap": int(overlap), "ocr_lang": ocr_lang}

    # 1-2) parse
    t0 = time.perf_counter()
    parsed = parse_file(path, ocr_lang)
    timings["parse"] = _ms(t0)

    # 3) normalize
    t0 = time.perf_counter()
    blocks, rules = normalize_blocks(parsed.blocks)
    timings["normalize"] = _ms(t0)

    # 4) exact duplicate blocks
    t0 = time.perf_counter()
    exact_dups = mark_exact_duplicates(blocks)
    timings["dedup"] = _ms(t0)

    active = [b for b in blocks if not b.removed_reason]
    plain_text = "\n\n".join(b.text for b in active)
    text_hash = hashlib.sha256(plain_text.encode("utf-8")).hexdigest()
    failed = not active                        # nothing usable came out of the parser

    # 5) what should happen to this file? (new / updated / unchanged / duplicate ...)
    decision = {"status": "failed", "version": None} if failed else (
        store.decide(doc_id, file_hash, strategy, params) if store else {"status": "preview", "version": None})

    # 6) chunk (selected strategy + the other two, for the comparison table)
    t0 = time.perf_counter()
    chunks, comparison = [], {}
    if not failed:
        for s in STRATEGIES:
            cs = chunk_document(blocks, doc_id, s, max_tokens, overlap)
            comparison[s] = {"label": STRATEGIES[s], **chunk_stats(cs)}
            if s == strategy:
                chunks = cs
        if not chunks:
            chunks = chunk_document(blocks, doc_id, "structure", max_tokens, overlap)
    timings["chunk"] = _ms(t0)

    near = find_near_duplicates(chunks, store.other_chunks(doc_id) if store else []) if chunks else []

    # 7) store (source tracking + incremental reindex)
    diff, stored_path = None, None
    status = decision["status"]
    if store and save and not failed:
        if status in ("new", "updated", "reprocessed"):
            stored_path = store.save_upload(doc_id, decision["version"], filename, data)
            title = parsed.doc_meta.get("title") or next((b.text for b in active if b.type == "heading"), filename)
            diff = store.commit(doc_id=doc_id, source_name=source_name, file_type=parsed.doc_meta.get("file_type"), title=title,
                                version=decision["version"], status=status, file_hash=file_hash, text_hash=text_hash,
                                size=len(data), stored_path=stored_path, parser=parsed.parser, strategy=strategy, params=params,
                                n_blocks=len(active), chunks=chunks)
        elif status == "duplicate":
            store.add_alias(source_name, decision["duplicate_of"]["doc_id"], file_hash)
    version = decision.get("version") or (decision.get("previous") or {}).get("version")

    # ---- assemble the report
    chunk_dicts = []
    for c in chunks:
        d = c.to_dict()
        d["citation"] = citation(source_name, c.page_start, c.page_end, c.section_path)
        d["source"] = {"doc_id": doc_id, "source_name": source_name, "version": version, "file_hash": file_hash[:12],
                       "parser": parsed.parser}
        chunk_dicts.append(d)

    counts: dict = {}
    for b in blocks:
        counts[b.type] = counts.get(b.type, 0) + 1
    source = {"doc_id": doc_id, "source_name": source_name, "filename": filename, "file_type": parsed.doc_meta.get("file_type"),
              "size_bytes": len(data), "file_hash": file_hash, "text_hash": text_hash, "parser": parsed.parser,
              "version": version, "status": status, "saved": bool(store and save and not failed and status in ("new", "updated", "reprocessed", "duplicate")),
              "duplicate_of": (decision.get("duplicate_of") or {}).get("source_name"),
              "previous_version": (decision.get("previous") or {}).get("version"),
              "previous_file_hash": (decision.get("previous") or {}).get("file_hash"),
              "previous_text_hash": (decision.get("previous") or {}).get("text_hash"),
              "doc_meta": parsed.doc_meta, "strategy": strategy, "params": params}
    result = {
        "source": source,
        "timings_ms": timings,
        "stages": {
            "raw": {"text": parsed.raw_text, "chars": len(parsed.raw_text)},
            "parse": {"blocks": [dict(b.to_dict(), idx=i) for i, b in enumerate(parsed.blocks)], "counts": counts,
                      "warnings": parsed.warnings, "doc_meta": parsed.doc_meta},
            "normalize": {"rules": rules, "plain_text": plain_text,
                          "blocks": [dict(b.to_dict(), idx=i) for i, b in enumerate(blocks)]},
            "dedup": {"exact": exact_dups, "near": near},
            "chunks": {"strategy": strategy, "items": chunk_dicts, "stats": chunk_stats(chunks) if chunks else {}, "comparison": comparison},
            "store": {"status": status, "diff": diff},
        },
    }
    result["evidence"] = build_evidence(result, blocks, chunks, parsed)
    if chunks:
        from evaluation.checks import run_checks      # imported here: evaluation depends on ingestion, not the other way round
        result["evaluation"] = run_checks(parsed=parsed, blocks=blocks, chunks=chunks, chunk_dicts=chunk_dicts, strategy=strategy,
                                          max_tokens=int(max_tokens), overlap=int(overlap), doc_id=doc_id,
                                          file_type=parsed.doc_meta.get("file_type"), near_dups=near)
    else:
        result["evaluation"] = {"score": None, "passed": 0, "warned": 0, "failed": 0, "checks": []}
    return result


def _short(h):
    return (h or "")[:10]


def build_evidence(r, blocks, chunks, parsed) -> dict:
    """Live answers for the 'Document engineering' tab: what happened to THIS document."""
    src, st = r["source"], r["stages"]
    ev = {}
    # parsing failures
    issues = [w for w in parsed.warnings if w["level"] in ("warning", "error")]
    infos = [w for w in parsed.warnings if w["level"] == "info"]
    fmt = lambda w: f"[{w['level']}] {w['code']}" + (f" (page {w['page']})" if "page" in w else "") + f": {w['message']}"
    ev["parsing_failures"] = {"headline": f"{len(issues)} problem(s) detected" if issues else "No parsing problems detected",
                              "items": [fmt(w) for w in issues] + [fmt(w) for w in infos]}
    # OCR
    ocr_blocks = [b for b in blocks if b.meta.get("ocr")]
    if ocr_blocks:
        fixed = sum(1 for b in ocr_blocks if "ocr_digit_fix" in b.meta.get("rules", []))
        low = sum(len(b.meta.get("low_conf_words", [])) for b in ocr_blocks)
        ev["ocr"] = {"headline": f"OCR used on page(s) {parsed.doc_meta.get('ocr_pages')} · mean confidence {parsed.doc_meta.get('ocr_mean_confidence')}%",
                     "items": [f"{len(ocr_blocks)} block(s) came from OCR; {fixed} had digit-lookalike corrections", f"{low} low-confidence word(s) flagged (<60%), e.g. "
                               + ", ".join(sorted({w for b in ocr_blocks for w in b.meta.get('low_conf_words', [])})[:8])]}
    else:
        ev["ocr"] = {"headline": "No OCR needed - the file has a real text layer", "items": ["Upload a scanned PDF or a photo/PNG of text to see OCR in action."]}
    # tables
    tables = [b for b in blocks if b.type == "table"]
    ev["tables"] = {"headline": f"{len(tables)} table(s) detected" if tables else "No tables in this document",
                    "items": [f"Table on page {b.page or '-'}: {b.meta.get('n_rows')} rows x {b.meta.get('n_cols')} cols" + (f", {b.meta['merged_cells']} merged cell(s)" if b.meta.get("merged_cells") else "") for b in tables]
                    + ([f"{sum(1 for c in chunks if 'table_part' in c.meta)} table chunk(s) were split with the header row repeated"] if any("table_part" in c.meta for c in chunks) else [])
                    + (["Naive extraction flattens or drops tables; here they are kept as Markdown and never split mid-row."] if tables else [])}
    # headers / footers
    hf = [b for b in blocks if b.removed_reason and b.type in ("header", "footer", "boilerplate")]
    ev["headers_footers"] = {"headline": f"{len(hf)} header/footer/boilerplate block(s) removed" if hf else "No repeated headers/footers found",
                             "items": [f"{b.type}: \"{b.text[:70]}\" - {b.removed_reason}" for b in hf[:10]]}
    # footnotes
    fn = [b for b in blocks if b.type == "footnote" and not b.removed_reason]
    ev["footnotes"] = {"headline": f"{len(fn)} footnote(s) found" if fn else "No footnotes found",
                       "items": [f"p.{b.page or '-'}: {b.text[:90]}" for b in fn[:6]] + (["Stored as a separate chunk linked to the section's parent."] if fn else [])}
    # duplicates
    ex, near = st["dedup"]["exact"], st["dedup"]["near"]
    items = [f"Block #{d['block']} duplicates #{d['duplicate_of']}: \"{d['preview']}...\"" for d in ex]
    items += [f"Near-duplicate ({p['similarity']}) {p['scope']}: \"{p['preview']}...\"" for p in near[:5]]
    if src["status"] == "duplicate":
        items.append(f"Whole file is a byte-identical copy of '{src['duplicate_of']}' - not indexed again.")
    ev["duplicates"] = {"headline": f"{len(ex)} exact, {len(near)} near-duplicate" + (", FILE DUPLICATE" if src["status"] == "duplicate" else ""),
                        "items": items or ["No duplicate content found."]}
    # identity / versioning / metadata / reindex
    ev["doc_id"] = {"headline": f"{src['doc_id']}  <-  source name '{src['source_name']}'",
                    "items": ["ID = sha1(source name). Upload a changed file under the SAME source name and you get the SAME ID with a new version.",
                              "Chunk ID = sha1(doc_id + section path + text): unchanged text keeps its ID across versions."]}
    prev = src["previous_file_hash"]
    ver_items = [f"file_hash {_short(src['file_hash'])} (bytes) · text_hash {_short(src['text_hash'])} (normalized text)"]
    if prev:
        ver_items.append(f"previous version {src['previous_version']}: file_hash {_short(prev)}, text_hash {_short(src['previous_text_hash'])}")
        if prev != src["file_hash"] and src["previous_text_hash"] == src["text_hash"]:
            ver_items.append("Bytes changed but the normalized text did not -> cosmetic change, nothing to re-embed.")
    ev["versioning"] = {"headline": f"status = {src['status'].upper()}" + (f" · version {src['version']}" if src["version"] else ""), "items": ver_items}
    ev["metadata"] = {"headline": "Metadata attached at 3 levels",
                      "items": ["Document: " + ", ".join(f"{k}={v}" for k, v in {**{"doc_id": src["doc_id"], "version": src["version"], "file_type": src["file_type"], "parser": src["parser"]}, **{k: v for k, v in src["doc_meta"].items() if v not in ("", None, [])}}.items())[:400],
                                "Chunk: chunk_id, kind, parent_id, section_path, page_start/page_end, tokens, block_types, overlap_tokens, citation",
                                "Source: source_name, file_hash, version_added / version_removed (in the library)"]}
    diff = st["store"]["diff"]
    if diff:
        ev["reindex"] = {"headline": f"+{diff['added']} added · -{diff['removed']} removed · {diff['unchanged']} unchanged · {diff['relinked']} re-linked",
                         "items": [f"Embed only {len(diff['embed_these'])} new retrievable chunk(s); delete {len(diff['delete_vectors_for'])} stale vector(s); "
                                   f"skip {diff['unchanged']} unchanged chunk(s)."]}
    else:
        why = {"unchanged": "Same file, same settings: nothing to do.", "duplicate": "Duplicate file: skipped.", "failed": "Parsing failed: nothing indexed, any previous version stays active.",
               "preview": "Preview only.", "new": "Not saved (Save to library was off)."}.get(src["status"], "Not saved (Save to library was off).")
        ev["reindex"] = {"headline": "No index changes", "items": [why]}
    # chunking
    cs = st["chunks"]["stats"]
    comp = st["chunks"]["comparison"]
    ev["chunk_structure"] = {"headline": f"{cs.get('parents', 0)} section parents · {cs.get('retrievable', 0)} retrievable chunks",
                             "items": [f"Strategy used: {src['strategy']}. Chunks never cross a heading; tables and lists are kept whole."] +
                                      [f"Section path example: {' > '.join(c.section_path)}" for c in chunks if c.section_path][:1]}
    ev["chunk_tokens"] = {"headline": f"tokens per chunk: min {cs.get('min_tokens')} · avg {cs.get('avg_tokens')} · max {cs.get('max_tokens')} (limit {src['params']['max_tokens']})",
                          "items": [f"{cs.get('with_overlap', 0)} chunk(s) start with overlap from the previous chunk (repeated sentences keep context across the cut)."]}
    ev["chunk_semantic"] = {"headline": "Strategy comparison on this document",
                            "items": [f"{v['label']}: {v['retrievable']} chunks, avg {v['avg_tokens']} tokens, {v['cut_mid_sentence']} cut mid-sentence, {v['split_tables']} split tables" for v in comp.values()]}
    kids = cs.get("retrievable", 0)
    ev["parent_child"] = {"headline": f"{cs.get('parents', 0)} parent(s) ← {kids} child chunk(s)" if cs.get("parents") else "Parent-child only applies to the 'structure' strategy",
                          "items": ["Search matches small children (precise); the answer prompt can expand to the parent (full section context)."] if cs.get("parents") else ["Switch strategy to 'structure' to see parents."]}
    ev["normalization"] = {"headline": f"{sum(1 for x in st['normalize']['rules'] if x['blocks_changed'])} rule(s) changed text",
                           "items": [f"{x['rule']}: changed {x['blocks_changed']} block(s)" for x in st["normalize"]["rules"] if x["blocks_changed"]] or ["Text was already clean."]}
    return ev
