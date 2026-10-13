"""Teaching content for the Concepts tab, plus live evidence computed from the runs you did in the other tabs."""
from __future__ import annotations

CONCEPTS = [
    {"group": "Embeddings and similarity", "id": "embeddings", "title": "What an embedding is",
     "text": "An embedding model turns text into a list of numbers (a vector) so that texts with similar meaning end up close together.\n"
             "- Dimension = how many numbers per text (50 for GloVe, 384 for MiniLM, 768+ for larger models). More dimensions can hold more nuance but cost more memory and time.\n"
             "- Different models live in different spaces: vectors from two models can never be compared with each other.\n"
             "- Queries and documents must be embedded by the SAME model (some models want a prefix such as 'query:' or 'passage:').\n"
             "- Here you can compare: LSA (learned from your corpus), character n-gram hashing (spelling, not meaning), GloVe (pretrained word meaning, averaged), and optional neural sentence models.",
     "code": "retrieval/embedders.py"},
    {"group": "Embeddings and similarity", "id": "cosine", "title": "Cosine similarity, dot product and distance",
     "text": "Cosine similarity compares the ANGLE between two vectors: 1 = same direction, 0 = unrelated, -1 = opposite. It ignores vector length.\n"
             "- If every vector is scaled to length 1 (L2-normalized), cosine similarity equals the dot product, and Euclidean distance gives the same ranking. That is why almost every vector index is built on normalized vectors.\n"
             "- If vectors are NOT normalized, dot product favours long vectors (for averaged word vectors, long texts or frequent words) and Euclidean distance is dominated by length. Rankings then differ from cosine.\n"
             "- Rule of thumb: normalize once at indexing time, then use the cheapest metric (inner product).",
     "code": "retrieval/dense.py (cosine_matrix), embedders.l2norm()"},
    {"group": "Dense vs sparse retrieval", "id": "sparse", "title": "Sparse retrieval (BM25, TF-IDF)",
     "text": "Sparse methods score a passage by the words it shares with the query, weighted by how rare each word is (idf).\n"
             "- Strengths: exact terms, rare words, product codes, names, numbers; no training; tiny index; easy to explain.\n"
             "- Failure mode 1, vocabulary mismatch: 'how much paid time off' never matches a passage that says 'annual leave'. No shared word means a score of 0.\n"
             "- Failure mode 2, spelling: a misspelled word matches nothing at all.\n"
             "- Failure mode 3: common words can give a wrong passage a higher score than the right one.",
     "code": "retrieval/sparse.py"},
    {"group": "Dense vs sparse retrieval", "id": "dense", "title": "Dense retrieval",
     "text": "Dense retrieval embeds the query and every passage, then returns the nearest passages by cosine similarity.\n"
             "- Strengths: paraphrases and synonyms, natural-language questions, cross-language matching (with multilingual models).\n"
             "- Failure mode 1, exact tokens: 'E4012' and 'E4013' or 'RT-3340' and 'RT-3345' produce almost identical vectors, so near-identical passages are confused.\n"
             "- Failure mode 2, semantic neighbours: a passage on a related topic can outrank the one with the answer. Negation and numbers are weakly encoded.\n"
             "- Failure mode 3, domain shift: a model trained on general web text can be poor on your specialist vocabulary.\n"
             "- Quality depends heavily on the model. The offline models here are baselines; a neural model (sentence-transformers) usually does far better on paraphrases.",
     "code": "retrieval/embedders.py, retrieval/dense.py"},
    {"group": "Dense vs sparse retrieval", "id": "failures", "title": "Complementary failures",
     "text": "Sparse and dense fail on DIFFERENT queries. The comparison tab counts, for the top 5 results:\n"
             "- both found it, only sparse found it, only dense found it, neither found it.\n"
             "The 'oracle' (a query counted as solved if either method solves it) shows the best possible gain from combining them.",
     "code": "retrieval/compare.py (_analyse)"},
    {"group": "Dense vs sparse retrieval", "id": "hybrid", "title": "Hybrid retrieval",
     "text": "Run both retrievers and merge the lists.\n"
             "- RRF (reciprocal rank fusion): score = sum of 1/(60 + rank) over the lists. Uses only ranks, so no score calibration is needed. A robust default.\n"
             "- Linear fusion: min-max scale each score list to 0-1, then mix with a weight (alpha). Can be tuned, but is sensitive to score distributions.\n"
             "Hybrid usually matches the better retriever on each query type, at the cost of two indexes.",
     "code": "retrieval/hybrid.py"},
    {"group": "ANN indexes", "id": "exact", "title": "Why approximate search?",
     "text": "Exact ('flat') search compares the query with every vector: perfect recall, but the cost grows linearly with the corpus. At millions of vectors this is too slow and too memory-hungry.\n"
             "ANN (approximate nearest neighbour) indexes give up a little recall to search a small fraction of the vectors.\n"
             "- recall@k = the share of the true k nearest neighbours that the index also returns.\n"
             "- Every ANN index has a knob that trades recall for latency. The benchmark sweeps it so you can see the curve.",
     "code": "retrieval/ann.py"},
    {"group": "ANN indexes", "id": "hnsw", "title": "HNSW (graph index)",
     "text": "Hierarchical Navigable Small World: vectors are nodes in a layered proximity graph. A search starts at the top (sparse) layer and greedily walks towards the query, then refines on the bottom layer.\n"
             "- M = links per node. Larger M: better recall and more memory (see the memory table: HNSW is LARGER than flat because it stores the graph).\n"
             "- efConstruction = effort while building. Higher means a better graph and a slower build.\n"
             "- efSearch = size of the candidate list while searching: THE recall/latency knob.\n"
             "- No training step, supports adding vectors incrementally, usually the best recall per millisecond. Deletes are awkward (tombstones).",
     "code": "retrieval/ann.py (build_index 'hnsw')"},
    {"group": "ANN indexes", "id": "ivf", "title": "IVF (inverted file)",
     "text": "IVF clusters all vectors with k-means into nlist cells. At query time only the nprobe cells closest to the query are scanned.\n"
             "- nlist is often about 4 x sqrt(N). nprobe is the recall/latency knob: nprobe = nlist equals exact search.\n"
             "- Needs a training step (k-means) on representative data, so it is sensitive to data drift.\n"
             "- Memory is about the same as flat, because the full vectors are still stored. Fast to build, simple, good with filters on disk.",
     "code": "retrieval/ann.py ('ivf')"},
    {"group": "ANN indexes", "id": "pq", "title": "Product quantization (PQ)",
     "text": "PQ compresses vectors. Each vector is cut into m sub-vectors; each sub-vector is replaced by the id of its nearest centroid from a 256-entry codebook (1 byte). A 128-d float vector (512 bytes) becomes m bytes.\n"
             "- Memory: 10-30x smaller, which is what lets billions of vectors fit in RAM.\n"
             "- Cost: distances are approximate, so recall drops, often a lot.\n"
             "- Usual fix: IVF-PQ finds candidates cheaply, then the top candidates are RE-RANKED with the exact vectors (kept on disk or in a slower tier). The benchmark shows how much recall this recovers.\n"
             "- Note: random synthetic vectors compress worse than real embeddings, so treat the absolute recall numbers as pessimistic.",
     "code": "retrieval/ann.py ('ivfpq', 'ivfpq_refine', 'pq')"},
    {"group": "ANN indexes", "id": "tradeoff", "title": "Recall, latency and memory: choosing an index",
     "text": "There is no best index, only the best one for your constraint.\n"
             "- Up to roughly 100k vectors: flat search is often fast enough, exact, and simplest.\n"
             "- Need high recall at low latency and RAM is available: HNSW.\n"
             "- RAM is the bottleneck (tens of millions of vectors and up): IVF-PQ with re-ranking.\n"
             "- Vectors change constantly: HNSW (incremental add) over IVF (needs retraining).\n"
             "Always measure recall against exact search on YOUR data, at the latency you can afford. Remember that retrieval recall is not answer quality: a 95% recall index only hurts if the missed 5% were the relevant passages.",
     "code": "retrieval/ann.py (best_configs)"},
    {"group": "Vector databases", "id": "vdb", "title": "What a vector database adds",
     "text": "A vector index is only a data structure. A vector database wraps it with the things an application needs:\n"
             "- storage of vectors together with a payload (metadata) and ids; upsert and delete\n"
             "- an ANN index kept up to date, persisted to disk\n"
             "- metadata filtering, a query planner, replication, access control, an API\n"
             "Examples: Chroma, Qdrant, Weaviate, Milvus, pgvector (PostgreSQL), FAISS (index only). The Vector DB tab has a 150-line version so you can see every moving part.",
     "code": "retrieval/vectordb.py"},
    {"group": "Vector databases", "id": "vdb_filter", "title": "Filtered search is hard",
     "text": "'Find similar passages, but only from the HR domain' sounds easy and is not.\n"
             "- Post-filtering (search, then drop non-matching hits) returns fewer than k results, or none, when the filter is selective.\n"
             "- Pre-filtering with exact scan is always correct but slow when many vectors survive.\n"
             "- In-graph filtering (the HNSW walk skips non-matching ids) works well until the filter is very selective, then the graph becomes disconnected and recall collapses.\n"
             "- A query planner picks per query: very selective filter, scan the survivors; otherwise filter inside the index. That is what the mini database does in 'auto' mode.",
     "code": "retrieval/vectordb.py (search, filter_experiment)"},
    {"group": "Evaluation", "id": "metrics", "title": "How retrieval is measured",
     "text": "Each query is labeled with the passage(s) that answer it.\n"
             "- recall@k: share of relevant passages found in the top k. With one relevant passage this equals 'hit rate'.\n"
             "- MRR@10: 1/rank of the first relevant result, averaged. Rewards putting the answer first.\n"
             "- nDCG@10: like MRR but handles several relevant passages and discounts lower ranks.\n"
             "- Report results PER QUERY TYPE. One average hides exactly the failure modes this assignment is about.\n"
             "- Latency (p50 and p95 ms per query) and index size belong next to quality, because they are the other side of the trade-off.",
     "code": "retrieval/metrics.py"},
]


def _best(m_by_type: dict, hi=True):
    items = sorted(m_by_type.items(), key=lambda kv: kv[1]["mrr@10"], reverse=hi)
    return items[0]


def build_evidence(cmp=None, ann=None, flt=None, models=None) -> dict:
    ev = {}
    if models:
        ready = [m for m in models if m["state"] == "ready"]
        ev["embeddings"] = {"headline": f"{len(ready)} of {len(models)} models ready",
                            "items": [f"{m['label']}: {m['state'].replace('_', ' ')}" + (f" ({m['note']})" if m["note"] else "") for m in models]}
    if cmp:
        by = {m["id"]: m for m in cmp["methods"]}
        md = cmp["analysis"].get("metric_demo")
        if md:
            ev["cosine"] = {"headline": f"on {md['model']}: L2-normalized, L2 and cosine agree {md['l2_vs_cosine_unit_overlap@5']:.0%}",
                            "items": [f"Before normalizing, vector lengths range from {md['norm_spread'][0]} to {md['norm_spread'][1]}.",
                                      f"Top-5 overlap with cosine on RAW vectors: dot product {md['dot_vs_cosine_raw_overlap@5']:.0%}, Euclidean {md['l2_vs_cosine_raw_overlap@5']:.0%}" if md["l2_vs_cosine_raw_overlap@5"] is not None else "",
                                      "After normalizing, cosine, dot product and Euclidean distance give the same ranking."]}
        if "bm25" in by:
            b = by["bm25"]
            top, worst = _best(b["by_type"]), _best(b["by_type"], False)
            ev["sparse"] = {"headline": f"BM25 MRR@10 = {b['metrics']['mrr@10']:.2f}: best on '{top[0]}' ({top[1]['mrr@10']:.2f}), worst on '{worst[0]}' ({worst[1]['mrr@10']:.2f})",
                            "items": [f"{t}: MRR {v['mrr@10']:.2f} over {v['n']} queries" for t, v in b["by_type"].items()]}
        bd = cmp.get("best_dense")
        if bd and bd in by:
            d = by[bd]
            top, worst = _best(d["by_type"]), _best(d["by_type"], False)
            ev["dense"] = {"headline": f"{d['label']}: MRR@10 {d['metrics']['mrr@10']:.2f}, best on '{top[0]}' ({top[1]['mrr@10']:.2f}), worst on '{worst[0]}' ({worst[1]['mrr@10']:.2f})",
                           "items": [f"{t}: MRR {v['mrr@10']:.2f}" + (f" (BM25: {by['bm25']['by_type'][t]['mrr@10']:.2f})" if "bm25" in by else "") for t, v in d["by_type"].items()]}
        comp = cmp["analysis"].get("complementarity")
        if comp:
            c = comp["counts"]
            ev["failures"] = {"headline": f"top-{comp['k']}: both {c['both']}, only sparse {c['sparse_only']}, only dense {c['dense_only']}, neither {c['neither']}",
                              "items": [f"Oracle (either method succeeds): {comp['oracle_hit_at_k']:.0%} of queries", f"Dense model used: {comp['dense_model']}"] +
                                       [f"Sparse missed: '{e['query']}' -> {e['why']}" for e in cmp["analysis"]["failures"]["sparse"][:2]] +
                                       [f"Dense missed: '{e['query']}' -> {e['why']}" for e in cmp["analysis"]["failures"]["dense"][:2]]}
        hy = [m for m in cmp["methods"] if m["family"] == "hybrid"]
        if hy and "bm25" in by and bd:
            h = max(hy, key=lambda m: m["metrics"]["mrr@10"])
            ev["hybrid"] = {"headline": f"{h['label']}: MRR@10 {h['metrics']['mrr@10']:.2f} vs BM25 {by['bm25']['metrics']['mrr@10']:.2f} and dense {by[bd]['metrics']['mrr@10']:.2f}",
                            "items": [f"{m['label']}: MRR {m['metrics']['mrr@10']:.2f}, recall@5 {m['metrics']['recall@5']:.2f}" for m in hy]}
        ev["metrics"] = {"headline": f"{cmp['dataset']['n_queries']} labeled queries over {cmp['dataset']['n_docs']} passages",
                         "items": [f"{t}: {n} queries" for t, n in cmp["dataset"]["types"].items()]}
    if ann:
        pts, builds, flat = ann["points"], {b["label"]: b for b in ann["builds"]}, ann["flat"]
        fam = lambda pref: [p for p in pts if p["index"].startswith(pref)]

        def best_for(pref, target):
            ok = [p for p in fam(pref) if p["recall"] >= target]
            return min(ok, key=lambda p: p["latency_ms"]) if ok else None
        ev["exact"] = {"headline": f"exact search over {ann['settings']['n']:,} vectors: {flat['latency_ms']} ms per query, {flat['bytes'] / 1e6:.1f} MB",
                       "items": ["Latency grows linearly with the number of vectors; ANN indexes search only a fraction."]}
        h = best_for("HNSW", 0.95)
        if h:
            hb = next(b for b in ann["builds"] if b["label"] == h["index"])
            ev["hnsw"] = {"headline": f"{h['index']} reaches recall {h['recall']:.0%} at {h['latency_ms']} ms ({h['speedup']}x faster than exact) with efSearch={h['param']}",
                          "items": [f"{b['label']}: build {b['build_s']} s, {b['bytes_per_vector']} bytes/vector ({b['compression_vs_flat']}x vs flat)" for b in ann["builds"] if b["type"] == "hnsw"]
                                   + [f"{p['index']} efSearch={p['param']}: recall {p['recall']:.2f}, {p['latency_ms']} ms" for p in fam("HNSW M=16")]}
        v = best_for("IVF-Flat", 0.95)
        if v:
            ev["ivf"] = {"headline": f"{v['index']} reaches recall {v['recall']:.0%} at {v['latency_ms']} ms ({v['speedup']}x faster) with nprobe={v['param']}",
                         "items": [f"{p['index']} nprobe={p['param']}: recall {p['recall']:.2f}, {p['latency_ms']} ms" for p in fam("IVF-Flat")]}
        pq, rr = [b for b in ann["builds"] if b["type"] == "ivfpq"], [b for b in ann["builds"] if b["type"] == "ivfpq_refine"]
        if pq:
            bestpq = max(fam("IVF-PQ"), key=lambda p: p["recall"] if "re-rank" not in p["index"] else -1)
            bestrr = max([p for p in pts if "re-rank" in p["index"]], key=lambda p: p["recall"], default=None)
            ev["pq"] = {"headline": f"IVF-PQ compresses {pq[0]['compression_vs_flat']}x ({pq[0]['bytes_per_vector']} bytes/vector) but its best recall is {bestpq['recall']:.0%}",
                        "items": [f"With exact re-ranking of the top candidates recall rises to {bestrr['recall']:.0%} at the same RAM (exact vectors on disk)." if bestrr else "",
                                  *[f"{b['label']}: {b['bytes_per_vector']} bytes/vector, {b['compression_vs_flat']}x smaller than flat" for b in ann["builds"] if b["type"] in ("pq", "ivfpq", "ivfpq_refine")]]}
        rows = [b for b in ann["best"] if b["target"] == 0.95 and b["speedup"]]
        if rows:
            w = max(rows, key=lambda b: b["speedup"])
            ev["tradeoff"] = {"headline": f"fastest index at recall >= 95%: {w['index']} ({w['param']}), {w['speedup']}x faster than exact",
                              "items": [f"{b['index']}: {b['param']} -> recall {b['recall']:.2f}, {b['latency_ms']} ms (x{b['speedup']})" if b["recall"] else f"{b['index']}: never reaches 95% recall" for b in ann["best"] if b["target"] == 0.95]}
    if flt:
        rows = flt["rows"]
        sel = sorted({r["selectivity"] for r in rows})
        lo = sel[0] if sel else None
        low = {r["mode"]: r for r in rows if r["selectivity"] == lo}
        mid_s = sel[min(2, len(sel) - 1)] if sel else None
        mid = {r["mode"]: r for r in rows if r["selectivity"] == mid_s}
        if low and "post" in low:
            ev["vdb_filter"] = {"headline": f"at a filter keeping {mid_s:.1%}: post-filter recall {mid['post']['recall']:.2f} (returns {mid['post']['avg_results']} of {flt['settings']['k']}), in-graph {mid['in-graph']['recall']:.2f}",
                                "items": [f"filter keeps {r['selectivity']:.1%} ({r['survivors']} vectors) | {r['mode']}: recall {r['recall']:.2f}, {r['avg_results']} results, {r['latency_ms']} ms" for r in rows]}
    return {k: v for k, v in ev.items() if v}
