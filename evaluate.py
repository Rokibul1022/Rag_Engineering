"""Evaluate the pipeline.

  python evaluate.py                    benchmark on all sample files against evaluation/gold.json (writes evaluation/report.md)
  python evaluate.py --file my.pdf      integrity checks on any one document (no answer key needed)
  python evaluate.py --gold my.json     benchmark with your own answer key (same format as evaluation/gold.json)
Exit code is 1 if any check fails, so you can use it in CI.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "n/a": "n/a "}


def print_integrity(ev, title):
    print(f"\n{title}: integrity score {ev['score']}/100  ({ev['passed']} pass, {ev['warned']} warn, {ev['failed']} fail)")
    for c in ev["checks"]:
        print(f"  [{ICON[c['status']]}] {c['name']:<44} {c['value']}")
        if c["status"] in ("warn", "fail"):
            print(f"         -> {c['detail']}")


def markdown(rep):
    s = rep["summary"]
    L = ["# Evaluation report", "", f"Settings: max_tokens={rep['settings']['max_tokens']}, overlap={rep['settings']['overlap']}", "",
         f"- Gold checks passed: **{s['gold_checks_passed']}/{s['gold_checks_total']}**",
         f"- Mean integrity score: **{s['integrity_mean_score']}/100**, failed integrity checks: **{s['integrity_failed_checks']}**", "",
         "## Chunk quality by strategy (retrieval with a simple TF-IDF search)", "",
         "| Strategy | Questions | Answer kept in a chunk | Hit@1 | Hit@3 | MRR | Hit@3 via parent |", "|---|---|---|---|---|---|---|"]
    for r in rep["retrieval_by_strategy"].values():
        L.append(f"| {r['label']} | {r['questions']} | {r['answer_kept']:.0%} | {r['hit1']:.0%} | {r['hit3']:.0%} | {r['mrr']:.2f} | {r['parent_hit3']:.0%} |")
    L += ["", "## Per sample", ""]
    for r in rep["samples"]:
        L.append(f"### {r['file']}")
        for c in r["checks"]:
            L.append(f"- {'PASS' if c['ok'] else 'FAIL'} {c['name']}: {c['detail']}")
        if r["metrics"]:
            L.append("- metrics: " + ", ".join(f"{k}={v}" for k, v in r["metrics"].items()))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file"); ap.add_argument("--gold"); ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=30); ap.add_argument("--strategy", default="structure"); ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    from ingestion.pipeline import run_pipeline
    if a.file:
        r = run_pipeline(a.file, strategy=a.strategy, max_tokens=a.max_tokens, overlap=a.overlap)
        if r["source"]["status"] == "failed":
            print("Document could not be parsed:", *[w["message"] for w in r["stages"]["parse"]["warnings"]], sep="\n  ")
            return 1
        print_integrity(r["evaluation"], f"{a.file} [{a.strategy}]")
        return 1 if r["evaluation"]["failed"] else 0
    from evaluation.benchmark import run_benchmark
    rep = run_benchmark(max_tokens=a.max_tokens, overlap=a.overlap, gold_path=a.gold)
    if a.json:
        print(json.dumps(rep, indent=2)); return 0
    for r in rep["samples"]:
        print(f"\n=== {r['file']}")
        for c in r["checks"]:
            print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['name']:<38} {c['detail']}")
        if r.get("integrity", {}).get("score") is not None:
            ev = r["integrity"]; bad = [c for c in ev["checks"] if c["status"] in ("warn", "fail")]
            print(f"  integrity score {ev['score']}/100" + ("" if not bad else "  <- " + "; ".join(f"{c['name']} ({c['value']})" for c in bad)))
    print("\nChunk quality by strategy (answer kept / hit@1 / hit@3 / MRR / hit@3 via parent):")
    for r in rep["retrieval_by_strategy"].values():
        print(f"  {r['label']:<32} {r['answer_kept']:.0%} / {r['hit1']:.0%} / {r['hit3']:.0%} / {r['mrr']:.2f} / {r['parent_hit3']:.0%}   ({r['questions']} questions)")
    s = rep["summary"]
    print(f"\nGold checks: {s['gold_checks_passed']}/{s['gold_checks_total']} passed · integrity score {s['integrity_mean_score']}/100 · failed integrity checks {s['integrity_failed_checks']}")
    out = Path(__file__).parent / "evaluation" / "report.md"
    out.write_text(markdown(rep), encoding="utf-8"); print("Report written to", out)
    return 0 if s["gold_checks_passed"] == s["gold_checks_total"] and not s["integrity_failed_checks"] else 1


if __name__ == "__main__":
    sys.exit(main())
