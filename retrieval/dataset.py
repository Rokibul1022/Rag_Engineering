"""Labeled retrieval datasets: a corpus of passages + queries labeled with the passage(s) that answer them."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILTIN = ROOT / "datasets" / "acme_kb"
_STOP = set("a an the and or but if of to in on at for with by from as is are was were be been it this that these those we our you your they their he she his her its not no so than then there which who what when where how why do does did has have had into over under about also can could should may might i my me us".split())


def _read_jsonl(text: str, what: str) -> list:
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"{what}, line {n}: not valid JSON ({e.msg})")
    return rows


def build(corpus_rows: list, query_rows: list, name: str = "custom") -> dict:
    """Validate and normalize a dataset. Raises ValueError with a readable message."""
    if not corpus_rows:
        raise ValueError("The corpus is empty.")
    if not query_rows:
        raise ValueError("There are no queries.")
    corpus, seen = [], set()
    for i, r in enumerate(corpus_rows, 1):
        if not isinstance(r, dict) or "id" not in r or "text" not in r:
            raise ValueError(f"corpus row {i}: needs at least 'id' and 'text'.")
        did = str(r["id"])
        if did in seen:
            raise ValueError(f"corpus row {i}: duplicate id '{did}'.")
        seen.add(did)
        corpus.append({"id": did, "text": str(r["text"]), "title": str(r.get("title", "")), "domain": str(r.get("domain", "")) or "all"})
    ids = {d["id"] for d in corpus}
    queries = []
    for i, r in enumerate(query_rows, 1):
        if not isinstance(r, dict) or "query" not in r or "relevant" not in r:
            raise ValueError(f"query row {i}: needs 'query' and 'relevant' (a list of corpus ids).")
        rel = [str(x) for x in (r["relevant"] if isinstance(r["relevant"], list) else [r["relevant"]])]
        missing = [x for x in rel if x not in ids]
        if not rel or missing:
            raise ValueError(f"query row {i}: relevant id(s) not in the corpus: {missing or 'none given'}.")
        queries.append({"id": str(r.get("id", f"q{i:03d}")), "query": str(r["query"]), "relevant": rel, "type": str(r.get("type", "other")),
                        "lexical_overlap": r.get("lexical_overlap")})
    types, domains = {}, {}
    for q in queries:
        types[q["type"]] = types.get(q["type"], 0) + 1
    for d in corpus:
        domains[d["domain"]] = domains.get(d["domain"], 0) + 1
    return {"name": name, "corpus": corpus, "queries": queries, "types": types, "domains": domains}


def load_builtin() -> dict:
    return build(_read_jsonl((BUILTIN / "corpus.jsonl").read_text(encoding="utf-8"), "corpus.jsonl"),
                 _read_jsonl((BUILTIN / "queries.jsonl").read_text(encoding="utf-8"), "queries.jsonl"), "acme_kb (built-in)")


def from_text(corpus_text: str, queries_text: str, name: str = "uploaded") -> dict:
    return build(_read_jsonl(corpus_text, "corpus"), _read_jsonl(queries_text, "queries"), name)


def summary(ds: dict) -> dict:
    return {"name": ds["name"], "n_docs": len(ds["corpus"]), "n_queries": len(ds["queries"]), "types": ds["types"], "domains": ds["domains"]}
