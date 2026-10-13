"""Prepare MS MARCO v1.1 into the vector-search-lab JSONL format (corpus.jsonl + queries.jsonl).

Usage:
  python datasets/ms_marco/prepare_ms_marco.py [--split validation] [--n-queries 500]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Remove local dir from sys.path if it conflicts with HF datasets
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))

from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser(description="Convert MS MARCO to vector-search-lab format")
    parser.add_argument("--split", default="validation", choices=["validation", "train", "test"], help="Dataset split to use")
    parser.add_argument("--n-queries", type=int, default=500, help="Number of queries to include (default: 500)")
    parser.add_argument("--out-dir", default=str(ROOT / "datasets" / "ms_marco"), help="Output directory")
    args = parser.parse_args()

    data_path = ROOT / "datasets" / "ms_marco" / "v1.1"
    if not data_path.exists():
        data_path = ROOT / "ms_marco" / "v1.1"
    
    print(f"Loading {data_path} [{args.split}]...")
    ds = load_from_disk(str(data_path))[args.split]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_map = {}  # text -> doc_id
    corpus_rows = []
    queries_rows = []

    doc_counter = 0
    query_counter = 0

    print(f"Extracting up to {args.n_queries} queries with relevant passages...")
    for row in ds:
        passages = row.get("passages", {})
        texts = passages.get("passage_text", [])
        selected = passages.get("is_selected", [])

        # Only include queries that have at least one marked relevant passage
        relevant_indices = [i for i, s in enumerate(selected) if s == 1]
        if not relevant_indices:
            continue

        relevant_doc_ids = []
        for i, text in enumerate(texts):
            clean_text = text.strip()
            if not clean_text:
                continue
            if clean_text not in corpus_map:
                doc_counter += 1
                doc_id = f"d{doc_counter:06d}"
                corpus_map[clean_text] = doc_id
                corpus_rows.append({
                    "id": doc_id,
                    "text": clean_text,
                    "title": "",
                    "domain": (row.get("query_type") or "general").lower()
                })
            else:
                doc_id = corpus_map[clean_text]

            if i in relevant_indices:
                relevant_doc_ids.append(doc_id)

        if not relevant_doc_ids:
            continue

        query_counter += 1
        q_id = f"q{query_counter:05d}"
        q_type = (row.get("query_type") or "other").lower()

        queries_rows.append({
            "id": q_id,
            "query": row["query"],
            "relevant": list(dict.fromkeys(relevant_doc_ids)),  # unique
            "type": q_type
        })

        if query_counter >= args.n_queries:
            break

    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"

    print(f"Writing {len(corpus_rows)} passages to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in corpus_rows:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Writing {len(queries_rows)} queries to {queries_path}...")
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries_rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\nDone! Successfully created:")
    print(f"  - {corpus_path} ({len(corpus_rows):,} passages)")
    print(f"  - {queries_path} ({len(queries_rows):,} queries)")
    print("\nYou can now test it with:")
    print(f"  python evaluate.py --corpus {corpus_path} --queries {queries_path}")

if __name__ == "__main__":
    main()
