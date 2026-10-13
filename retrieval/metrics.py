"""Ranking metrics. `ranked` = list of doc ids, best first. `relevant` = set of correct doc ids."""
from __future__ import annotations

import math


def hit_at_k(ranked, relevant, k):
    return float(any(d in relevant for d in ranked[:k]))


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked, relevant, k=10):
    for r, d in enumerate(ranked[:k], 1):
        if d in relevant:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked, relevant, k=10):
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def first_rank(ranked, relevant):
    for r, d in enumerate(ranked, 1):
        if d in relevant:
            return r
    return None


def summarize(rankings: dict, queries: list, ks=(1, 5, 10)) -> dict:
    """rankings: {query_id: [doc ids]}.  Returns mean metrics over `queries`."""
    n = len(queries) or 1
    out = {f"recall@{k}": sum(recall_at_k(rankings[q["id"]], set(q["relevant"]), k) for q in queries) / n for k in ks}
    out["mrr@10"] = sum(mrr_at_k(rankings[q["id"]], set(q["relevant"]), 10) for q in queries) / n
    out["ndcg@10"] = sum(ndcg_at_k(rankings[q["id"]], set(q["relevant"]), 10) for q in queries) / n
    return {k: round(v, 4) for k, v in out.items()}
