"""Gold-set benchmark: compares the pipeline's output with known-correct answers.

  parsing        heading precision/recall/F1, table / footnote / header-footer counts, must-contain & must-not-contain text
  OCR            character and word error rate against a reference transcription
  ingestion      status (new/updated/duplicate/failed) and incremental re-index savings
  chunk quality  for each strategy: is the answer kept in a chunk (answer_kept), and does a simple retriever find it (hit@1/3, MRR)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ingestion.chunking import STRATEGIES, chunk_document
from ingestion.normalize import normalize_blocks
from ingestion.dedup import mark_exact_duplicates
from ingestion.parsers import parse_file
from ingestion.pipeline import make_doc_id, run_pipeline
from ingestion.store import Store

from .metrics import cer, norm_ws, prf, wer
from .retrieval import evaluate_retrieval

ROOT = Path(__file__).resolve().parent.parent
GOLD = Path(__file__).with_name("gold.json")


def _norm_heading(t):
    return norm_ws(t).lower()


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def evaluate_sample(entry, result, samples_dir, strategies=tuple(STRATEGIES), max_tokens=200, overlap=30):
    """Returns {"checks":[...], "metrics":{...}, "retrieval":{strategy: {...}}} for one gold entry."""
    exp, checks, metrics = entry.get("expect", {}), [], {}
    status = result["source"]["status"]
    checks.append(_check("ingestion status", status == entry["expect_status"], f"expected {entry['expect_status']}, got {status}"))
    if "expect_reindex" in entry:
        d = result["stages"]["store"]["diff"] or {}
        total = result["stages"]["chunks"]["stats"].get("retrievable", 1) or 1
        frac = len(d.get("embed_these", [])) / total
        r = entry["expect_reindex"]
        checks.append(_check("incremental re-index", d.get("unchanged", 0) >= r["min_unchanged"] and frac <= r["max_embed_fraction"],
                             f"{d.get('unchanged')} unchanged, re-embed {frac:.0%} of chunks (limit {r['max_embed_fraction']:.0%})"))
        metrics["reembed_fraction"] = round(frac, 3)
    if not exp or status in ("failed",):
        return {"checks": checks, "metrics": metrics, "retrieval": {}}

    # parse again here so the comparison does not depend on whether the store skipped the document (duplicate/unchanged)
    parsed = parse_file(Path(samples_dir) / entry["file"])
    blocks, _ = normalize_blocks(parsed.blocks)
    exact = mark_exact_duplicates(blocks)
    active = [b for b in blocks if not b.removed_reason]
    plain = norm_ws(" ".join(b.text for b in active) if True else "")
    plain_lines = norm_ws("\n".join(b.text for b in active)).lower()

    if "headings" in exp:
        got = {_norm_heading(b.text) for b in active if b.type == "heading"}
        p, r, f = prf({_norm_heading(h) for h in exp["headings"]}, got)
        metrics.update(heading_precision=p, heading_recall=r, heading_f1=f)
        checks.append(_check("headings found", f == 1.0, f"precision {p:.0%}, recall {r:.0%}" + ("" if f == 1.0 else f"; got {sorted(got)}")))
    for key, label, count in (("tables", "tables detected", sum(b.type == "table" for b in active)),
                              ("footnotes", "footnotes detected", sum(b.type == "footnote" for b in active)),
                              ("headers_footers_removed", "headers/footers/boilerplate removed",
                               sum(1 for b in blocks if b.removed_reason and b.type in ("header", "footer", "boilerplate"))),
                              ("exact_duplicates", "duplicate blocks removed", len(exact))):
        if key in exp:
            checks.append(_check(label, count == exp[key], f"expected {exp[key]}, got {count}"))
    if "must_contain" in exp:
        missing = [t for t in exp["must_contain"] if norm_ws(t).lower() not in plain_lines]
        metrics["must_contain_recall"] = round(1 - len(missing) / len(exp["must_contain"]), 3)
        checks.append(_check("expected text present", not missing, f"{len(exp['must_contain']) - len(missing)}/{len(exp['must_contain'])} found" + (f"; missing: {missing}" if missing else "")))
    if exp.get("must_not_contain"):
        leaks = [t for t in exp["must_not_contain"] if norm_ws(t).lower() in plain_lines]
        metrics["leaks"] = len(leaks)
        checks.append(_check("unwanted text removed", not leaks, "none leaked" if not leaks else f"still present: {leaks}"))
    if exp.get("ocr_reference"):
        c, w = cer(exp["ocr_reference"], plain), wer(exp["ocr_reference"], plain)
        metrics.update(cer=round(c, 4), wer=round(w, 4))
        checks.append(_check("OCR accuracy", c <= 0.03, f"character error rate {c:.1%}, word error rate {w:.1%} (limit 3% CER)"))

    retrieval = {}
    if entry.get("qa"):
        doc_id = make_doc_id(entry.get("source_name") or entry["file"])
        for s in strategies:
            retrieval[s] = evaluate_retrieval(chunk_document(blocks, doc_id, s, max_tokens, overlap), entry["qa"])
    return {"checks": checks, "metrics": metrics, "retrieval": retrieval}


def run_benchmark(samples_dir=None, max_tokens=200, overlap=30, gold_path=None):
    samples_dir = Path(samples_dir or ROOT / "samples")
    gold = json.loads(Path(gold_path or GOLD).read_text(encoding="utf-8"))
    store = Store(tempfile.mkdtemp(prefix="bench_"))        # throw-away store: the benchmark never touches your library
    rows = []
    # unchanged check: ingest the first sample twice
    for entry in gold["samples"]:
        res = run_pipeline(samples_dir / entry["file"], source_name=entry.get("source_name"), store=store, max_tokens=max_tokens, overlap=overlap)
        ev = evaluate_sample(entry, res, samples_dir, max_tokens=max_tokens, overlap=overlap)
        ev.update(file=entry["file"], integrity=res.get("evaluation", {}))
        rows.append(ev)
    again = run_pipeline(samples_dir / gold["samples"][0]["file"], store=store, max_tokens=max_tokens, overlap=overlap)
    rows.append({"file": "(re-upload of " + gold["samples"][0]["file"] + ")", "metrics": {}, "retrieval": {}, "integrity": {},
                 "checks": [_check("ingestion status", again["source"]["status"] == "unchanged", f"expected unchanged, got {again['source']['status']}")]})

    # aggregate
    all_checks = [c for r in rows for c in r["checks"]]
    agg_ret = {}
    for s in STRATEGIES:
        rs = [(r["retrieval"][s], r["retrieval"][s]["n"]) for r in rows if s in r["retrieval"]]
        n = sum(k for _, k in rs)
        if n:
            agg_ret[s] = {"label": STRATEGIES[s], "questions": n,
                          **{m: round(sum(x[m] * k for x, k in rs) / n, 3) for m in ("answer_kept", "hit1", "hit3", "mrr", "parent_hit3")}}
    ints = [r["integrity"] for r in rows if r.get("integrity", {}).get("score") is not None]
    return {"settings": {"max_tokens": max_tokens, "overlap": overlap},
            "summary": {"gold_checks_passed": sum(c["ok"] for c in all_checks), "gold_checks_total": len(all_checks),
                        "integrity_mean_score": round(sum(i["score"] for i in ints) / len(ints)) if ints else None,
                        "integrity_failed_checks": sum(i["failed"] for i in ints)},
            "retrieval_by_strategy": agg_ret, "samples": rows}
