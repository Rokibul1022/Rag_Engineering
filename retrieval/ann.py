"""Approximate nearest neighbour (ANN) benchmark with FAISS: recall vs latency vs memory for Flat, HNSW, IVF, IVF-PQ and PQ.

recall@k = how many of the TRUE k nearest neighbours (from exact search) the approximate index also returned.
"""
from __future__ import annotations

import math
import time

import numpy as np

try:
    import faiss
except ImportError:                      # pragma: no cover
    faiss = None


_BASE: dict = {}                     # id(refine index) -> its compressed IVF-PQ base index


def faiss_status() -> dict:
    return {"available": faiss is not None, "version": getattr(faiss, "__version__", None),
            "reason": "" if faiss else "faiss-cpu is not installed: pip install faiss-cpu"}


def make_vectors(n: int, d: int, clusters: int = 64, nq: int = 200, seed: int = 0):
    """Synthetic 'embedding-like' vectors: a mixture of clusters (topics) with decaying variance per dimension, L2-normalized.
    Queries come from the same distribution but are NOT in the dataset (as in real search)."""
    rng = np.random.default_rng(seed)
    scale = (1.0 / np.sqrt(1 + np.arange(d) / (d / 8))).astype(np.float32)
    centers = rng.normal(size=(clusters, d)).astype(np.float32) * scale

    def sample(m):
        a = rng.integers(0, clusters, size=m)
        x = centers[a] + 0.55 * rng.normal(size=(m, d)).astype(np.float32) * scale
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return sample(n), sample(nq)


def build_index(spec: dict, xb: np.ndarray):
    """spec: {"type": flat|hnsw|ivf|ivfpq|pq, ...}. Returns (index, build_seconds)."""
    n, d = xb.shape
    ip = faiss.METRIC_INNER_PRODUCT
    t0 = time.perf_counter()
    typ = spec["type"]
    if typ == "flat":
        idx = faiss.IndexFlatIP(d)
    elif typ == "hnsw":
        idx = faiss.IndexHNSWFlat(d, spec["M"], ip)
        idx.hnsw.efConstruction = spec.get("efC", 100)
    elif typ == "ivf":
        idx = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, spec["nlist"], ip)
    elif typ == "ivfpq":
        idx = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, spec["nlist"], spec["m"], spec.get("nbits", 8), ip)
    elif typ == "pq":
        idx = faiss.IndexPQ(d, spec["m"], spec.get("nbits", 8), ip)
    elif typ == "ivfpq_refine":      # IVF-PQ finds k*k_factor candidates cheaply, then exact vectors re-score them
        base = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, spec["nlist"], spec["m"], spec.get("nbits", 8), ip)
        idx = faiss.IndexRefineFlat(base)
        idx.k_factor = spec.get("k_factor", 10)
        _BASE[id(idx)] = base         # FAISS objects reject new attributes, so remember the compressed base index here
    else:
        raise ValueError(f"unknown index type {typ}")
    if not idx.is_trained:
        idx.train(xb[: min(n, 100_000)])
    idx.add(xb)
    return idx, time.perf_counter() - t0


def set_search_params(idx, spec: dict, **p):
    if spec["type"] == "hnsw":
        idx.hnsw.efSearch = p["ef"]
    elif spec["type"] in ("ivf", "ivfpq"):
        idx.nprobe = p["nprobe"]
    elif spec["type"] == "ivfpq_refine":
        faiss.downcast_index(idx.base_index).nprobe = p["nprobe"]


def index_bytes(idx) -> int:
    """Bytes held in RAM. For the re-rank index only the compressed PQ part counts: its exact vectors can live on disk."""
    if id(idx) in _BASE:
        return int(len(faiss.serialize_index(_BASE[id(idx)])))
    return int(len(faiss.serialize_index(idx)))


def recall(found: np.ndarray, truth: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(f[:k].tolist()) & set(t[:k].tolist())) / k for f, t in zip(found, truth)]))


def _time_single(idx, xq, k, n_lat, repeats=3):
    """Latency of ONE query at a time (what a live search box does). Median of a few repeats per query removes timer noise."""
    ms = []
    for i in range(min(n_lat, len(xq))):
        q = xq[i:i + 1]
        t = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            idx.search(q, k)
            t.append((time.perf_counter() - t0) * 1000)
        ms.append(float(np.median(t)))
    return float(np.mean(ms)), float(np.percentile(ms, 95))


def pq_m(d: int) -> int:
    """Number of PQ sub-vectors: aim for 8 dimensions per sub-vector; it must divide d."""
    m = max(1, d // 8)
    while d % m:
        m -= 1
    return m


def default_specs(n: int, d: int) -> list:
    """Each entry: (family label, build spec, name of the search parameter, values to sweep)."""
    nlist = int(min(4096, max(16, round(4 * math.sqrt(n)))))
    nprobes = [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= nlist]
    efs = [4, 8, 16, 32, 64, 128, 256]
    m = pq_m(d)
    specs = [(f"HNSW M={M}", {"type": "hnsw", "M": M, "efC": 100}, "ef", efs) for M in (8, 16, 32)]
    specs += [(f"IVF-Flat nlist={nlist}", {"type": "ivf", "nlist": nlist}, "nprobe", nprobes),
              (f"IVF-PQ m={m}", {"type": "ivfpq", "nlist": nlist, "m": m, "nbits": 8}, "nprobe", nprobes),
              (f"IVF-PQ m={m} + exact re-rank", {"type": "ivfpq_refine", "nlist": nlist, "m": m, "nbits": 8, "k_factor": 10}, "nprobe", nprobes),
              (f"PQ m={m} (no IVF)", {"type": "pq", "m": m, "nbits": 8}, None, [None])]
    return specs


def run_benchmark(n=30000, d=128, clusters=64, nq=200, k=10, latency_queries=100, seed=0, report=None) -> dict:
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed: pip install faiss-cpu")
    report = report or (lambda *a, **kw: None)
    xb, xq = make_vectors(n, d, clusters, nq, seed)
    threads = faiss.omp_get_max_threads()

    report(0.02, "exact search (ground truth)")
    flat, build_s = build_index({"type": "flat"}, xb)
    _, truth = flat.search(xq, k)
    faiss.omp_set_num_threads(1)                               # fair single-thread latency
    lat_mean, lat_p95 = _time_single(flat, xq, k, latency_queries)
    flat_info = {"label": "Flat (exact)", "recall": 1.0, "latency_ms": round(lat_mean, 3), "p95_ms": round(lat_p95, 3),
                 "build_s": round(build_s, 3), "bytes": index_bytes(flat), "param": None}
    faiss.omp_set_num_threads(threads)

    points, builds = [], []
    specs = default_specs(n, d)
    for si, (label, spec, pname, values) in enumerate(specs):
        report(0.08 + 0.9 * si / len(specs), f"building {label}")
        idx, bsec = build_index(spec, xb)
        size = index_bytes(idx)
        builds.append({"label": label, "type": spec["type"], "build_s": round(bsec, 3), "bytes": size, "bytes_per_vector": round(size / n, 1),
                       "compression_vs_flat": round(flat_info["bytes"] / size, 2),
                       "note": "exact vectors for re-ranking kept on disk (not counted)" if spec["type"] == "ivfpq_refine" else ""})
        for v in values:
            if pname:
                set_search_params(idx, spec, **{pname: v})
            _, found = idx.search(xq, k)
            faiss.omp_set_num_threads(1)
            m, p95 = _time_single(idx, xq, k, latency_queries)
            faiss.omp_set_num_threads(threads)
            points.append({"index": label, "type": spec["type"], "param_name": pname, "param": v, "recall": round(recall(found, truth, k), 4),
                           "latency_ms": round(m, 4), "p95_ms": round(p95, 4), "speedup": round(flat_info["latency_ms"] / m, 2) if m else None})
    faiss.omp_set_num_threads(threads)
    report(1.0, "done")
    return {"settings": {"n": n, "d": d, "clusters": clusters, "nq": nq, "k": k, "threads_for_latency": 1}, "flat": flat_info,
            "builds": builds, "points": points, "best": best_configs(points, flat_info)}


def best_configs(points: list, flat_info: dict, targets=(0.90, 0.95, 0.99)) -> list:
    """For every index family and recall target: the FASTEST setting that reaches it."""
    out = []
    for t in targets:
        for fam in dict.fromkeys(p["index"] for p in points):
            ok = [p for p in points if p["index"] == fam and p["recall"] >= t]
            if ok:
                b = min(ok, key=lambda p: p["latency_ms"])
                out.append({"target": t, "index": fam, "param": f"{b['param_name']}={b['param']}" if b["param_name"] else "-", "recall": b["recall"],
                            "latency_ms": b["latency_ms"], "speedup": b["speedup"]})
            else:
                out.append({"target": t, "index": fam, "param": "not reached", "recall": None, "latency_ms": None, "speedup": None})
    return out
