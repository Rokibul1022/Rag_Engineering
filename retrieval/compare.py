"""Run sparse, dense and hybrid retrievers on a labeled dataset and analyse where each one fails."""
from __future__ import annotations

import re
import time

import numpy as np

from . import embedders
from .dense import ExactIndex
from .hybrid import linear, ranks_from_scores, rrf
from .metrics import first_rank, summarize
from .sparse import BM25, TfidfCosine
from .text import tokenize

SPARSE = {"bm25": ("BM25", BM25), "tfidf": ("TF-IDF cosine", TfidfCosine)}
TOP = 50


def method_catalog() -> list:
    out = [{"id": k, "label": v[0], "family": "sparse", "state": "ready", "note": "no embeddings: exact word matching"} for k, v in SPARSE.items()]
    for m in embedders.list_models():
        out.append({"id": m["id"], "label": m["label"], "family": "dense", "state": m["state"], "note": m["note"], "kind": m["kind"], "description": m["description"]})
    out += [{"id": "hybrid-rrf", "label": "Hybrid: BM25 + dense (RRF)", "family": "hybrid", "state": "ready", "note": "rank fusion of BM25 and the best dense model"},
            {"id": "hybrid-linear", "label": "Hybrid: BM25 + dense (score mix)", "family": "hybrid", "state": "ready", "note": "min-max scaled scores, alpha = 0.5"}]
    return out


def _pcts(ms):
    return round(float(np.percentile(ms, 50)), 3), round(float(np.percentile(ms, 95)), 3)


def _rank_lists(S, ids, top=TOP):
    order = np.argsort(-S, axis=1, kind="stable")[:, :top]
    return order, [[ids[j] for j in row] for row in order]


def _score_dense(model_id, texts, qtexts, lat_n, report):
    emb = embedders.create(model_id)
    t0 = time.perf_counter()
    emb.fit(texts)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    D = emb.encode(texts, "doc")
    enc_docs_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    Q = emb.encode(qtexts, "query")
    enc_q_s = time.perf_counter() - t0
    S = (Q @ D.T).astype(np.float32)
    idx = ExactIndex(D)
    search_ms, enc_ms = [], []
    for i in range(min(lat_n, len(qtexts))):
        t0 = time.perf_counter(); qv = emb.encode([qtexts[i]], "query"); enc_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); idx.search(qv, 10); search_ms.append((time.perf_counter() - t0) * 1000)
    return S, {"dim": int(D.shape[1]), "fit_s": round(fit_s, 3), "encode_docs_s": round(enc_docs_s, 3), "encode_queries_s": round(enc_q_s, 3),
               "index_bytes": idx.nbytes, "search_ms_p50": _pcts(search_ms)[0], "search_ms_p95": _pcts(search_ms)[1],
               "encode_query_ms_p50": _pcts(enc_ms)[0]}, emb, D, Q


def _score_sparse(key, texts, qtexts, lat_n):
    model = SPARSE[key][1]()
    t0 = time.perf_counter(); model.fit(texts); fit_s = time.perf_counter() - t0
    S, lat = np.zeros((len(qtexts), len(texts)), dtype=np.float32), []
    for i, q in enumerate(qtexts):
        t0 = time.perf_counter(); S[i] = model.scores(q); dt = (time.perf_counter() - t0) * 1000
        if i < lat_n:
            lat.append(dt)
    return S, {"dim": None, "fit_s": round(fit_s, 3), "encode_docs_s": 0.0, "encode_queries_s": 0.0, "index_bytes": int(model.index_bytes()),
               "search_ms_p50": _pcts(lat)[0], "search_ms_p95": _pcts(lat)[1], "encode_query_ms_p50": 0.0}, model


def explain_miss(kind, query, rel_text, vocab):
    """A plain-language guess at WHY a retriever missed. Heuristic, meant for learning."""
    qt = tokenize(query)
    dt = set(tokenize(rel_text))
    if kind == "sparse":
        oov = [t for t in qt if t not in vocab]
        if oov and len(oov) == len([t for t in qt if t not in dt]):
            return f"unknown or misspelled word(s) {oov[:3]}: they match nothing in the index"
        if not (set(qt) & dt):
            return "vocabulary mismatch: no query word appears in the answer passage"
        return "shared words are too common; another passage matched better"
    if any(re.search(r"\d", t) for t in qt):
        return "exact code or number: embeddings blur near-identical passages"
    return "a topically similar passage scored higher (semantic confusion)"


def run_comparison(ds, method_ids, hybrid_dense=None, latency_queries=100, report=None) -> dict:
    report = report or (lambda *a, **k: None)
    ids = [d["id"] for d in ds["corpus"]]
    texts = [d["text"] for d in ds["corpus"]]
    titles = {d["id"]: d["title"] or d["text"][:50] for d in ds["corpus"]}
    text_by = {d["id"]: d["text"] for d in ds["corpus"]}
    queries = ds["queries"]
    qtexts = [q["query"] for q in queries]
    if len(texts) * len(queries) > 20_000_000:
        raise ValueError("dataset too large for the in-memory comparison (limit: docs x queries <= 20 million)")

    scores, info, skipped, raw = {}, {}, {}, {}
    base = [m for m in method_ids if not m.startswith("hybrid")]
    for n, mid in enumerate(base):
        report(0.05 + 0.6 * n / max(1, len(base)), f"running {mid}")
        try:
            if mid in SPARSE:
                scores[mid], info[mid], model = _score_sparse(mid, texts, qtexts, latency_queries)
                if mid == "bm25":
                    raw["bm25_vocab"] = model.vocabulary()
            else:
                scores[mid], info[mid], emb, D, Q = _score_dense(mid, texts, qtexts, latency_queries, report)
                raw[mid] = (emb, D, Q)
        except Exception as e:                     # one broken model must not break the whole comparison
            skipped[mid] = f"{type(e).__name__}: {e}"

    labels = {m["id"]: m["label"] for m in method_catalog()}
    family = {m["id"]: m["family"] for m in method_catalog()}

    def evaluate(mid, S):
        order, lists = _rank_lists(S, ids)
        rankings = {q["id"]: lists[i] for i, q in enumerate(queries)}
        overall = summarize({k: v[:10] for k, v in rankings.items()}, queries)
        by_type = {}
        for t in ds["types"]:
            qs = [q for q in queries if q["type"] == t]
            by_type[t] = {"n": len(qs), **summarize({q["id"]: rankings[q["id"]][:10] for q in qs}, qs)}
        ranks = {q["id"]: first_rank(rankings[q["id"]], set(q["relevant"])) for q in queries}
        return rankings, overall, by_type, ranks

    results = {}
    for mid, S in scores.items():
        rankings, overall, by_type, ranks = evaluate(mid, S)
        results[mid] = {"id": mid, "label": labels[mid], "family": family[mid], **info[mid], "metrics": overall, "by_type": by_type, "first_rank": ranks, "rankings": {k: v[:10] for k, v in rankings.items()}}

    dense_ids = [m for m in results if family[m] == "dense"]
    # default dense model for the hybrid and the failure analysis: prefer a SEMANTIC model (neural > pretrained > offline), then best MRR
    tier = {"neural": 0, "pretrained": 1, "offline": 2}
    kind_of = {m["id"]: m.get("kind", "offline") for m in method_catalog()}
    best_dense = hybrid_dense if hybrid_dense in results else (min(dense_ids, key=lambda m: (tier.get(kind_of[m], 3), -results[m]["metrics"]["mrr@10"])) if dense_ids else None)
    hyb = [m for m in method_ids if m.startswith("hybrid")]
    if hyb and ("bm25" in scores) and best_dense:
        for mid in hyb:
            fused = np.vstack([rrf([scores["bm25"][i], scores[best_dense][i]]) if mid == "hybrid-rrf" else linear(scores["bm25"][i], scores[best_dense][i])
                               for i in range(len(queries))])
            rankings, overall, by_type, ranks = evaluate(mid, fused)
            a, b = results["bm25"], results[best_dense]
            results[mid] = {"id": mid, "label": f"{labels[mid]} [{best_dense}]", "family": "hybrid", "dim": b["dim"], "fit_s": round(a["fit_s"] + b["fit_s"], 3),
                            "encode_docs_s": b["encode_docs_s"], "encode_queries_s": b["encode_queries_s"], "index_bytes": a["index_bytes"] + b["index_bytes"],
                            "search_ms_p50": round(a["search_ms_p50"] + b["search_ms_p50"], 3), "search_ms_p95": round(a["search_ms_p95"] + b["search_ms_p95"], 3),
                            "encode_query_ms_p50": b["encode_query_ms_p50"], "metrics": overall, "by_type": by_type, "first_rank": ranks,
                            "rankings": {k: v[:10] for k, v in rankings.items()}}
            scores[mid] = fused
    elif hyb:
        skipped["hybrid"] = "needs BM25 and at least one dense model in the selection"

    analysis = _analyse(ds, results, best_dense, text_by, titles, raw)
    report(1.0, "done")
    order = sorted(results.values(), key=lambda r: -r["metrics"]["mrr@10"])
    return {"dataset": {"name": ds["name"], "n_docs": len(texts), "n_queries": len(queries), "types": ds["types"]},
            "methods": order, "skipped": skipped, "best_dense": best_dense, "analysis": analysis,
            "queries": [{"id": q["id"], "query": q["query"], "type": q["type"], "relevant": q["relevant"]} for q in queries],
            "titles": titles}


def _analyse(ds, results, best_dense, text_by, titles, raw):
    out = {"complementarity": None, "failures": {"sparse": [], "dense": []}, "metric_demo": None}
    queries = ds["queries"]
    if "bm25" in results and best_dense:
        k = 5
        cats = {"both": 0, "sparse_only": 0, "dense_only": 0, "neither": 0}
        by_cat_type = {}
        sp, de = results["bm25"]["first_rank"], results[best_dense]["first_rank"]
        for q in queries:
            s_ok = sp[q["id"]] is not None and sp[q["id"]] <= k
            d_ok = de[q["id"]] is not None and de[q["id"]] <= k
            c = "both" if s_ok and d_ok else "sparse_only" if s_ok else "dense_only" if d_ok else "neither"
            cats[c] += 1
            by_cat_type.setdefault(q["type"], {"both": 0, "sparse_only": 0, "dense_only": 0, "neither": 0})[c] += 1
            if c in ("dense_only", "neither") and len(out["failures"]["sparse"]) < 40:
                rel = q["relevant"][0]
                out["failures"]["sparse"].append(_example(q, rel, results["bm25"], results[best_dense], text_by, titles, "sparse", raw))
            if c in ("sparse_only", "neither") and len(out["failures"]["dense"]) < 40:
                rel = q["relevant"][0]
                out["failures"]["dense"].append(_example(q, rel, results["bm25"], results[best_dense], text_by, titles, "dense", raw))
        n = len(queries)
        out["complementarity"] = {"k": k, "dense_model": best_dense, "counts": cats, "by_type": by_cat_type,
                                  "oracle_hit_at_k": round((cats["both"] + cats["sparse_only"] + cats["dense_only"]) / n, 4)}
        for kind in ("sparse", "dense"):                       # keep the most instructive examples, diverse by query type
            seen, picked = {}, []
            for ex in out["failures"][kind]:
                if seen.get(ex["type"], 0) < 3:
                    picked.append(ex)
                    seen[ex["type"]] = seen.get(ex["type"], 0) + 1
            out["failures"][kind] = picked[:14]
    if best_dense and best_dense in raw:
        emb, D, Q = raw[best_dense]
        Draw, Qraw = emb.encode([d["text"] for d in ds["corpus"]], "doc", normalize=False), emb.encode([q["query"] for q in queries], "query", normalize=False)
        cos_top = np.argsort(-(Qraw / np.maximum(np.linalg.norm(Qraw, axis=1, keepdims=True), 1e-9)) @ (Draw / np.maximum(np.linalg.norm(Draw, axis=1, keepdims=True), 1e-9)).T, axis=1)[:, :5]
        dot_top = np.argsort(-(Qraw @ Draw.T), axis=1)[:, :5]
        l2_raw = np.argsort(((Qraw[:, None, :] - Draw[None, :, :]) ** 2).sum(-1), axis=1)[:, :5] if Qraw.shape[0] * Draw.shape[0] * Draw.shape[1] < 60_000_000 else None
        l2_unit = np.argsort(((Q[:, None, :] - D[None, :, :]) ** 2).sum(-1), axis=1)[:, :5] if Q.shape[0] * D.shape[0] * D.shape[1] < 60_000_000 else None
        cos_unit = np.argsort(-(Q @ D.T), axis=1)[:, :5]
        ov = lambda a, b: round(float(np.mean([len(set(x) & set(y)) / 5 for x, y in zip(a, b)])), 4)
        out["metric_demo"] = {"model": best_dense, "mean_doc_norm_before_normalizing": round(float(np.linalg.norm(Draw, axis=1).mean()), 3),
                              "norm_spread": [round(float(np.linalg.norm(Draw, axis=1).min()), 3), round(float(np.linalg.norm(Draw, axis=1).max()), 3)],
                              "dot_vs_cosine_raw_overlap@5": ov(dot_top, cos_top),
                              "l2_vs_cosine_raw_overlap@5": ov(l2_raw, cos_top) if l2_raw is not None else None,
                              "l2_vs_cosine_unit_overlap@5": ov(l2_unit, cos_unit) if l2_unit is not None else None}
    return out


def _example(q, rel, sparse, dense, text_by, titles, kind, raw):
    s_rank, d_rank = sparse["first_rank"][q["id"]], dense["first_rank"][q["id"]]
    top_s, top_d = sparse["rankings"][q["id"]][0], dense["rankings"][q["id"]][0]
    miss_top = top_s if kind == "sparse" else top_d
    return {"qid": q["id"], "query": q["query"], "type": q["type"], "answer": titles[rel], "answer_id": rel,
            "bm25_rank": s_rank, "dense_rank": d_rank, "top_returned": titles.get(miss_top, miss_top),
            "why": explain_miss(kind, q["query"], text_by[rel], raw.get("bm25_vocab", set()))}
