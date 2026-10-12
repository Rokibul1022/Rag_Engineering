"""Run the whole evaluation from the terminal and write evaluation/report.md.

  python evaluate.py                      compare every ready retriever on the built-in dataset + ANN benchmark + filter experiment
  python evaluate.py --methods bm25 lsa   only these methods
  python evaluate.py --corpus c.jsonl --queries q.jsonl     your own labeled dataset
  python evaluate.py --no-ann             skip the slow ANN parts
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*"); ap.add_argument("--corpus"); ap.add_argument("--queries")
    ap.add_argument("--no-ann", action="store_true"); ap.add_argument("--ann-n", type=int, default=20000)
    a = ap.parse_args()
    from retrieval import ann, compare, dataset, vectordb

    ds = dataset.from_text(Path(a.corpus).read_text(encoding="utf-8"), Path(a.queries).read_text(encoding="utf-8"), "custom") if a.corpus and a.queries else dataset.load_builtin()
    ids = a.methods or [m["id"] for m in compare.method_catalog() if m["state"] == "ready"]
    print(f"Dataset: {ds['name']}: {len(ds['corpus'])} passages, {len(ds['queries'])} queries {ds['types']}")
    r = compare.run_comparison(ds, ids)
    types = list(ds["types"])
    out = ["# Retrieval evaluation report", "", f"Dataset: **{ds['name']}**, {len(ds['corpus'])} passages, {len(ds['queries'])} labeled queries.", "",
           "## Leaderboard", "", md_table(["Method", "Recall@1", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "search ms p50", "index"],
                                          [[m["label"], *(f"{m['metrics'][k]:.3f}" for k in ("recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10")), m["search_ms_p50"], f"{m['index_bytes'] / 1e3:.0f} KB"] for m in r["methods"]]),
           "", "## MRR@10 by query type", "", md_table(["Method", *types], [[m["label"], *(f"{m['by_type'][t]['mrr@10']:.2f}" for t in types)] for m in r["methods"]])]
    print(f"\n{'method':<46}{'R@1':>6}{'R@5':>6}{'MRR':>6}{'nDCG':>6}")
    for m in r["methods"]:
        x = m["metrics"]
        print(f"{m['label']:<46}{x['recall@1']:>6.2f}{x['recall@5']:>6.2f}{x['mrr@10']:>6.2f}{x['ndcg@10']:>6.2f}")
    print("\nMRR@10 by query type:  " + "  ".join(f"{t:>10}" for t in types))
    for m in r["methods"]:
        print(f"{m['id']:<22}" + "  ".join(f"{m['by_type'][t]['mrr@10']:>10.2f}" for t in types))
    c = r["analysis"]["complementarity"]
    if c:
        out += ["", f"## BM25 vs {c['dense_model']} (answer in top {c['k']})", "", md_table(["both", "only sparse", "only dense", "neither", "oracle"], [[c["counts"]["both"], c["counts"]["sparse_only"], c["counts"]["dense_only"], c["counts"]["neither"], f"{c['oracle_hit_at_k']:.0%}"]])]
        print(f"\nBM25 vs {c['dense_model']}: {c['counts']}  oracle {c['oracle_hit_at_k']:.0%}")
    if r["skipped"]:
        print("skipped:", r["skipped"])
    if not a.no_ann and ann.faiss_status()["available"]:
        print(f"\nANN benchmark ({a.ann_n} vectors)...")
        res = ann.run_benchmark(n=a.ann_n, d=128)
        out += ["", f"## ANN benchmark ({a.ann_n:,} synthetic vectors, 128 dims, exact search {res['flat']['latency_ms']} ms)", "",
                md_table(["Index", "recall>=0.95: setting", "latency ms", "speedup", "RAM/vector"],
                         [[b["label"], *(lambda x: (x["param"], x["latency_ms"], f"{x['speedup']}x") if x and x["recall"] else ("not reached", "-", "-"))(next((q for q in res["best"] if q["index"] == b["label"] and q["target"] == 0.95), None)), f"{b['bytes_per_vector']} B"] for b in res["builds"]])]
        for b in res["best"]:
            if b["target"] == 0.95:
                print(f"  {b['index']:<30} {b['param']:<14} {b['latency_ms']} ms  x{b['speedup']}")
        f = vectordb.filter_experiment()
        out += ["", "## Filtered search (20,000 vectors, k=10)", "", md_table(["filter keeps", "mode", "recall", "results returned", "ms"], [[f"{x['selectivity']:.1%}", x["mode"], x["recall"], x["avg_results"], x["latency_ms"]] for x in f["rows"]])]
    (ROOT / "evaluation").mkdir(exist_ok=True)
    (ROOT / "evaluation" / "report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\nReport written to evaluation/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
