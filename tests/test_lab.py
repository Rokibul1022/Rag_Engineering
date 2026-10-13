"""Run with:  pytest -q"""
import sys
import types
import time

import numpy as np
import pytest

from retrieval import ann, compare, dataset, embedders, hybrid, metrics, vectordb
from retrieval.dense import ExactIndex, cosine_matrix
from retrieval.sparse import BM25, TfidfCosine

needs_faiss = pytest.mark.skipif(ann.faiss is None, reason="faiss-cpu not installed")
DOCS = ["The router RT-3345 has 5 Ethernet ports.", "The router RT-3340 has 4 Ethernet ports.", "Annual leave is 25 days per year.", "Sourdough starter needs flour and water."]


# ---------------------------------------------------------------- metrics
def test_ranking_metrics():
    ranked = ["a", "b", "c", "d"]
    assert metrics.hit_at_k(ranked, {"c"}, 2) == 0.0 and metrics.hit_at_k(ranked, {"c"}, 3) == 1.0
    assert metrics.recall_at_k(ranked, {"a", "d"}, 3) == 0.5
    assert metrics.mrr_at_k(ranked, {"c"}) == pytest.approx(1 / 3)
    assert metrics.ndcg_at_k(["a", "b"], {"a"}) == 1.0 and metrics.ndcg_at_k(["b", "a"], {"a"}) == pytest.approx(1 / np.log2(3))
    assert metrics.first_rank(ranked, {"z"}) is None


# ---------------------------------------------------------------- sparse
def test_bm25_finds_exact_identifier_and_misses_paraphrase():
    bm = BM25().fit(DOCS)
    assert int(np.argmax(bm.scores("RT-3345 ports"))) == 0 and int(np.argmax(bm.scores("RT-3340 ports"))) == 1
    assert bm.scores("holiday allowance").max() == 0.0            # no shared word: vocabulary mismatch
    assert bm.scores("sourdogh starter").max() > 0                  # one misspelled word still leaves 'starter'
    assert bm.scores("sourdogh").max() == 0                         # ...but a misspelled word alone matches nothing
    assert TfidfCosine().fit(DOCS).scores("leave days").argmax() == 2


# ---------------------------------------------------------------- embedders
@pytest.mark.parametrize("mid", ["lsa", "hash-ngram"])
def test_offline_embedders_are_unit_vectors_and_deterministic(mid):
    texts = DOCS * 3
    e = embedders.create(mid).fit(texts)
    a, b = e.encode(texts), embedders.create(mid).fit(texts).encode(texts)
    assert a.dtype == np.float32 and np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5) and np.allclose(a, b)
    raw = e.encode(texts, normalize=False)
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0)         # raw vectors have varying lengths


def test_hash_ngram_tolerates_typos_but_not_meaning():
    e = embedders.create("hash-ngram")
    v = e.encode(["sourdough starter", "sourdogh starter", "annual leave policy"])
    assert v[0] @ v[1] > 0.7 > v[0] @ v[2]
    syn = e.encode(["a car", "an automobile"])
    assert syn[0] @ syn[1] < 0.2                                    # no semantic knowledge


def test_cosine_equals_dot_for_unit_vectors_and_l2_ranking_matches():
    rng = np.random.default_rng(0)
    X, q = rng.normal(size=(50, 16)).astype(np.float32), rng.normal(size=(1, 16)).astype(np.float32)
    Xn, qn = embedders.l2norm(X), embedders.l2norm(q)
    assert np.allclose(cosine_matrix(X, q).ravel(), (Xn @ qn.T).ravel(), atol=1e-5)
    assert (np.argsort(-(Xn @ qn.T).ravel()) == np.argsort(((Xn - qn) ** 2).sum(1))).all()


def test_sentence_transformer_adapter_with_stub(monkeypatch):
    calls = {}

    class Fake:
        def __init__(self, name): calls["name"] = name
        def get_sentence_embedding_dimension(self): return 4
        def encode(self, texts, **kw):
            calls["texts"] = texts
            return np.tile(np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=Fake))
    monkeypatch.setattr(embedders, "_ST_CACHE", {})
    e = embedders.create("st:multilingual-e5-small")
    out = e.encode(["hello"], role="query")
    assert calls["name"] == "intfloat/multilingual-e5-small" and calls["texts"] == ["query: hello"]   # e5 needs its prefix
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0) and e.dim == 4
    assert e.encode(["x"], role="doc") is not None and calls["texts"] == ["passage: x"]


def test_missing_sentence_transformers_is_reported_not_raised(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    assert embedders.create("st:all-MiniLM-L6-v2").status()["state"] == "needs_install"


# ---------------------------------------------------------------- dense / hybrid
def test_exact_index_matches_brute_force():
    rng = np.random.default_rng(1)
    X = embedders.l2norm(rng.normal(size=(200, 32)).astype(np.float32))
    Q = embedders.l2norm(rng.normal(size=(5, 32)).astype(np.float32))
    s, i = ExactIndex(X).search(Q, 7)
    assert (i == np.argsort(-(Q @ X.T), axis=1)[:, :7]).all() and (np.diff(s, axis=1) <= 1e-6).all()


def test_fusion():
    a, b = np.array([0.9, 0.5, 0.1], dtype=np.float32), np.array([0.2, 0.8, 0.7], dtype=np.float32)
    assert hybrid.rrf([a, b]).argmax() == 1                          # doc 1 is 2nd and 1st: best combined rank
    assert hybrid.linear(a, b, alpha=0.0).argmax() == 0 and hybrid.linear(a, b, alpha=1.0).argmax() == 1


# ---------------------------------------------------------------- dataset
def test_builtin_dataset_is_labeled_honestly():
    ds = dataset.load_builtin()
    assert len(ds["corpus"]) >= 80 and len(ds["queries"]) >= 150 and set(ds["types"]) >= {"keyword", "paraphrase", "typo", "identifier", "question"}
    ov = {t: np.mean([q["lexical_overlap"] for q in ds["queries"] if q["type"] == t]) for t in ds["types"]}
    assert ov["paraphrase"] < 0.15 and ov["keyword"] > 0.9            # paraphrase queries really share (almost) no words
    ids = {d["id"] for d in ds["corpus"]}
    assert all(set(q["relevant"]) <= ids for q in ds["queries"])


def test_dataset_validation_errors():
    with pytest.raises(ValueError, match="duplicate id"):
        dataset.build([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], [{"query": "q", "relevant": ["a"]}])
    with pytest.raises(ValueError, match="not in the corpus"):
        dataset.build([{"id": "a", "text": "x"}], [{"query": "q", "relevant": ["zzz"]}])
    with pytest.raises(ValueError, match="not valid JSON"):
        dataset.from_text("{bad", "{}")
    ok = dataset.from_text('{"id":"a","text":"hello world"}\n{"id":"b","text":"other"}\n', '{"query":"hello","relevant":"a"}\n')
    assert ok["queries"][0]["relevant"] == ["a"]


# ---------------------------------------------------------------- comparison: the failure modes this assignment is about
@pytest.fixture(scope="module")
def cmp_result():
    return compare.run_comparison(dataset.load_builtin(), ["bm25", "tfidf", "lsa", "hash-ngram", "hybrid-rrf", "hybrid-linear"])


def test_comparison_exposes_sparse_failure_modes(cmp_result):
    by = {m["id"]: m for m in cmp_result["methods"]}
    assert by["bm25"]["by_type"]["keyword"]["mrr@10"] > 0.95 and by["bm25"]["by_type"]["identifier"]["mrr@10"] > 0.95
    assert by["bm25"]["by_type"]["paraphrase"]["mrr@10"] < 0.4         # vocabulary mismatch
    assert by["hash-ngram"]["by_type"]["typo"]["mrr@10"] >= by["bm25"]["by_type"]["typo"]["mrr@10"]
    assert by["bm25"]["metrics"]["recall@10"] >= by["bm25"]["metrics"]["recall@5"] >= by["bm25"]["metrics"]["recall@1"]


def test_comparison_analysis_and_hybrid(cmp_result):
    c = cmp_result["analysis"]["complementarity"]
    assert sum(c["counts"].values()) == cmp_result["dataset"]["n_queries"] and c["oracle_hit_at_k"] >= max(c["counts"]["both"] + c["counts"]["sparse_only"], 0) / cmp_result["dataset"]["n_queries"]
    assert cmp_result["analysis"]["failures"]["sparse"] and all("why" in e for e in cmp_result["analysis"]["failures"]["sparse"])
    md = cmp_result["analysis"]["metric_demo"]
    assert md["l2_vs_cosine_unit_overlap@5"] > 0.98 and md["l2_vs_cosine_raw_overlap@5"] < md["l2_vs_cosine_unit_overlap@5"]
    assert any(m["family"] == "hybrid" for m in cmp_result["methods"])


def test_a_broken_model_is_skipped_not_fatal():
    r = compare.run_comparison(dataset.load_builtin(), ["bm25", "st:all-MiniLM-L6-v2"])
    assert "st:all-MiniLM-L6-v2" in r["skipped"] and r["methods"][0]["id"] == "bm25"


# ---------------------------------------------------------------- ANN
@needs_faiss
def test_ann_indexes_behave():
    xb, xq = ann.make_vectors(4000, 32, clusters=16, nq=50, seed=3)
    flat, _ = ann.build_index({"type": "flat"}, xb)
    _, truth = flat.search(xq, 10)
    hn, _ = ann.build_index({"type": "hnsw", "M": 16, "efC": 100}, xb)
    ann.set_search_params(hn, {"type": "hnsw"}, ef=4)
    low = ann.recall(hn.search(xq, 10)[1], truth, 10)
    ann.set_search_params(hn, {"type": "hnsw"}, ef=256)
    high = ann.recall(hn.search(xq, 10)[1], truth, 10)
    assert high >= 0.97 and high > low                                # efSearch is the recall knob
    spec = {"type": "ivf", "nlist": 32}
    iv, _ = ann.build_index(spec, xb)
    ann.set_search_params(iv, spec, nprobe=1); r1 = ann.recall(iv.search(xq, 10)[1], truth, 10)
    ann.set_search_params(iv, spec, nprobe=32); rall = ann.recall(iv.search(xq, 10)[1], truth, 10)
    assert rall == 1.0 and r1 < rall                                  # nprobe = nlist is exact search
    pq, _ = ann.build_index({"type": "pq", "m": 4, "nbits": 8}, xb)
    assert ann.index_bytes(pq) < ann.index_bytes(flat) / 5            # PQ compresses
    assert ann.recall(pq.search(xq, 10)[1], truth, 10) < 0.95         # ...at a cost in recall
    assert ann.index_bytes(hn) > ann.index_bytes(flat)                # HNSW stores a graph on top of the vectors


@needs_faiss
def test_pq_rerank_recovers_recall():
    xb, xq = ann.make_vectors(6000, 32, clusters=16, nq=50, seed=5)
    flat, _ = ann.build_index({"type": "flat"}, xb)
    truth = flat.search(xq, 10)[1]
    plain_spec = {"type": "ivfpq", "nlist": 64, "m": 4, "nbits": 8}
    rr_spec = {"type": "ivfpq_refine", "nlist": 64, "m": 4, "nbits": 8, "k_factor": 10}
    plain, _ = ann.build_index(plain_spec, xb); rr, _ = ann.build_index(rr_spec, xb)
    ann.set_search_params(plain, plain_spec, nprobe=16); ann.set_search_params(rr, rr_spec, nprobe=16)
    assert ann.recall(rr.search(xq, 10)[1], truth, 10) > ann.recall(plain.search(xq, 10)[1], truth, 10) + 0.1


@needs_faiss
def test_benchmark_runs_small():
    r = ann.run_benchmark(n=3000, d=32, nq=40, latency_queries=10)
    assert r["points"] and r["builds"] and all(0 <= p["recall"] <= 1 for p in r["points"])
    assert any(b["target"] == 0.9 and b["recall"] for b in r["best"])


# ---------------------------------------------------------------- vector database
def make_db(n=400, d=16, thr=100):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(n, d)).astype(np.float32)
    db = vectordb.MiniVectorDB(d, hnsw_threshold=thr)
    db.upsert([{"id": f"v{i}", "vector": X[i], "payload": {"cat": "a" if i % 4 == 0 else "b", "n": i}} for i in range(n)])
    return db, X


@needs_faiss
def test_vector_db_crud_filter_and_planner(tmp_path):
    db, X = make_db()
    assert db.search(X[7], k=3)["hits"][0]["id"] == "v7"
    r = db.search(X[7], k=5, where={"cat": "a"})
    assert all(h["payload"]["cat"] == "a" for h in r["hits"]) and len(r["hits"]) == 5
    exact = db.search(X[0], k=3, where={"cat": "a"}, mode="exact")["hits"]
    graph = db.search(X[0], k=3, where={"cat": "a"}, mode="in-graph")["hits"]
    assert exact[0]["id"] == "v0" and exact[0]["id"] in [h["id"] for h in graph]      # in-graph agrees with exact on the best hit
    few = db.search(X[0], k=3, where={"n": [0, 1, 2]})                  # 3 survivors: planner must scan, not walk the graph
    assert few["mode"] == "exact" and {h["id"] for h in few["hits"]} == {"v0", "v1", "v2"}
    assert db.delete(["v7"]) == 1 and "v7" not in [h["id"] for h in db.search(X[7], k=5)["hits"]] and db.count() == 399
    db.upsert([{"id": "v8", "vector": X[7], "payload": {"cat": "z"}}])     # replace an existing id
    assert db.search(X[7], k=1)["hits"][0]["id"] == "v8" and db.count() == 399
    with pytest.raises(ValueError):
        db.upsert([{"id": "bad", "vector": np.zeros(3)}])
    db.save(tmp_path / "db")
    again = vectordb.MiniVectorDB.load(tmp_path / "db", hnsw_threshold=100)
    assert again.count() == 399 and again.search(X[7], k=1)["hits"][0]["id"] == "v8"


@needs_faiss
def test_post_filtering_underfills_with_selective_filters():
    r = vectordb.filter_experiment(n=6000, d=32, k=10, nq=30, selectivities=(0.5, 0.01))
    get = lambda s, m: next(x for x in r["rows"] if x["selectivity"] == s and x["mode"] == m)
    assert get(0.01, "exact")["recall"] == 1.0 and get(0.01, "post")["avg_results"] < 5     # the failure the assignment asks about
    assert get(0.5, "in-graph")["recall"] > 0.9 > get(0.01, "post")["recall"]


# ---------------------------------------------------------------- web API
def test_api(tmp_path):
    from webapp.app import create_app
    c = create_app().test_client()
    assert c.get("/").status_code == 200 and c.get("/api/status").json["dataset"]["n_queries"] >= 150
    assert c.post("/api/dataset/upload").status_code == 400
    r = c.post("/api/compare", json={"methods": []})
    assert r.status_code == 400

    def wait(job):
        for _ in range(200):
            j = c.get(f"/api/jobs/{job}").json
            if j["state"] != "running":
                return j
            time.sleep(0.1)
    j = wait(c.post("/api/compare", json={"methods": ["bm25", "hash-ngram", "hybrid-rrf"]}).json["job"])
    assert j["state"] == "done" and len(j["result"]["methods"]) == 3
    assert "sparse" in c.get("/api/concepts").json["evidence"]
    sim = c.post("/api/similarity", json={"a": "sourdough starter", "b": "sourdogh starter"}).json["rows"]
    assert {r["model"] for r in sim} >= {"lsa", "hash-ngram"}
    if ann.faiss is not None:
        j = wait(c.post("/api/vdb/build", json={"model": "hash-ngram"}).json["job"])
        assert j["state"] == "done"
        hits = c.post("/api/vdb/search", json={"query": "annual leave days", "domain": "hr", "k": 3}).json["hits"]
        assert hits and all(h["payload"]["domain"] == "hr" for h in hits)
    assert c.get("/api/dataset/download/corpus").status_code == 200 and c.get("/api/dataset/download/nope").status_code == 404
