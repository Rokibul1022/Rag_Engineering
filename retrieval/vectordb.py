"""MiniVectorDB: what a vector database does, in ~150 lines you can read.

  store     vectors + a payload (metadata) per id          upsert / delete / count
  index     HNSW graph (FAISS) once there are enough vectors, exact scan below that
  filter    metadata filters, with a small QUERY PLANNER that picks the strategy:
              exact     filter first, then brute-force the survivors          (always correct, slow if many survive)
              post      ANN first, drop non-matching hits afterwards          (fast, but can return < k results)
              in-graph  the HNSW search itself skips non-matching ids         (fast and correct)
  persist   save / load to a folder
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:                      # pragma: no cover
    faiss = None


class MiniVectorDB:
    def __init__(self, dim: int, hnsw_threshold: int = 2000, M: int = 16, ef_construction: int = 100, ef_search: int = 64):
        self.dim, self.thr, self.M, self.efc, self.ef = dim, hnsw_threshold, M, ef_construction, ef_search
        self.ids, self.payloads = [], []
        self.vecs = np.zeros((0, dim), dtype=np.float32)
        self.alive = np.zeros(0, dtype=bool)
        self.row = {}                        # id -> row
        self._index, self._dirty = None, True

    # ---------------------------------------------------------------- write path
    def upsert(self, items: list):
        """items: [{"id", "vector", "payload"}]. Existing ids are replaced."""
        new_v, new_i, new_p = [], [], []
        for it in items:
            v = np.asarray(it["vector"], dtype=np.float32)
            if v.shape != (self.dim,):
                raise ValueError(f"vector for '{it['id']}' has shape {v.shape}, expected ({self.dim},)")
            v = v / (np.linalg.norm(v) or 1.0)                 # store unit vectors: cosine = dot product
            if it["id"] in self.row:
                r = self.row[it["id"]]
                self.vecs[r], self.payloads[r], self.alive[r] = v, dict(it.get("payload", {})), True
            else:
                new_v.append(v); new_i.append(it["id"]); new_p.append(dict(it.get("payload", {})))
        if new_v:
            start = len(self.ids)
            self.vecs = np.vstack([self.vecs, np.vstack(new_v)])
            self.alive = np.concatenate([self.alive, np.ones(len(new_v), dtype=bool)])
            self.ids += new_i
            self.payloads += new_p
            for k, i in enumerate(new_i):
                self.row[i] = start + k
        self._dirty = True

    def delete(self, ids: list) -> int:
        n = 0
        for i in ids:
            r = self.row.get(i)
            if r is not None and self.alive[r]:
                self.alive[r] = False                          # tombstone; the graph is rebuilt lazily
                n += 1
        if n:
            self._dirty = True
        return n

    def count(self) -> int:
        return int(self.alive.sum())

    # ---------------------------------------------------------------- index
    def _ensure_index(self):
        if not self._dirty:
            return
        self._index = None
        if faiss is not None and self.count() >= self.thr:
            rows = np.flatnonzero(self.alive)
            idx = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_INNER_PRODUCT)
            idx.hnsw.efConstruction = self.efc
            idx.add(self.vecs[rows])
            self._index, self._index_rows = idx, rows.astype(np.int64)    # position in index -> row in storage
        self._dirty = False

    def _mask(self, where):
        m = self.alive.copy()
        for field, want in (where or {}).items():
            wants = set(want) if isinstance(want, (list, tuple, set)) else {want}
            m &= np.array([p.get(field) in wants for p in self.payloads], dtype=bool)
        return m

    # ---------------------------------------------------------------- search
    def search(self, vector, k: int = 5, where: dict | None = None, mode: str = "auto", overfetch: int = 3) -> dict:
        """Returns {"hits": [{"id","score","payload"}], "mode": used, "ms": float, "candidates": int}."""
        t0 = time.perf_counter()
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) or 1.0)
        self._ensure_index()
        mask = self._mask(where)
        allowed = int(mask.sum())
        if mode == "auto":                                     # the query planner
            if self._index is None or allowed <= max(k * 20, 0.02 * self.count()):
                mode = "exact"                                 # few survivors: scanning them is cheapest AND exact
            elif where:
                mode = "in-graph"
            else:
                mode = "ann"
        if self._index is None and mode in ("post", "in-graph", "ann"):
            mode = "exact"                                     # not enough vectors for a graph yet
        if mode == "exact":
            rows = np.flatnonzero(mask)
            s = self.vecs[rows] @ q[0]
            top = np.argsort(-s)[:k]
            found = [(int(rows[i]), float(s[i])) for i in top]
        elif mode == "ann":
            D, I = self._index.search(q, k)
            found = [(int(self._index_rows[i]), float(d)) for d, i in zip(D[0], I[0]) if i >= 0]
        elif mode == "post":
            D, I = self._index.search(q, k * overfetch)
            found = [(int(self._index_rows[i]), float(d)) for d, i in zip(D[0], I[0]) if i >= 0 and mask[self._index_rows[i]]][:k]
        elif mode == "in-graph":
            ok_pos = np.flatnonzero(mask[self._index_rows]).astype(np.int64)
            params = faiss.SearchParametersHNSW()
            params.efSearch = max(self.ef, k)
            params.sel = faiss.IDSelectorBatch(ok_pos)
            D, I = self._index.search(q, k, params=params)
            found = [(int(self._index_rows[i]), float(d)) for d, i in zip(D[0], I[0]) if i >= 0]
        else:
            raise ValueError(f"unknown mode '{mode}'")
        if self._index is not None and mode in ("ann", "in-graph", "post"):
            pass
        return {"hits": [{"id": self.ids[r], "score": round(s, 4), "payload": self.payloads[r]} for r, s in found],
                "mode": mode, "ms": round((time.perf_counter() - t0) * 1000, 3), "candidates_allowed": allowed}

    # ---------------------------------------------------------------- persistence
    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vecs)
        (folder / "meta.json").write_text(json.dumps({"dim": self.dim, "ids": self.ids, "payloads": self.payloads, "alive": self.alive.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, folder, **kw):
        folder = Path(folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        db = cls(meta["dim"], **kw)
        db.vecs, db.ids, db.payloads = np.load(folder / "vectors.npy"), meta["ids"], meta["payloads"]
        db.alive = np.array(meta["alive"], dtype=bool)
        db.row = {i: r for r, i in enumerate(db.ids)}
        return db


def filter_experiment(n=20000, d=64, k=10, nq=100, selectivities=(0.5, 0.2, 0.05, 0.01, 0.002), seed=0, report=None) -> dict:
    """How much do filters hurt ANN search? Compares the three strategies against exact filtered search at several selectivities."""
    from .ann import make_vectors
    if faiss is None:
        raise RuntimeError("faiss-cpu is not installed: pip install faiss-cpu")
    report = report or (lambda *a, **kw: None)
    xb, xq = make_vectors(n, d, clusters=32, nq=nq, seed=seed)
    rng = np.random.default_rng(seed + 1)
    attr = rng.random(n)                                       # a random metadata value in [0,1): filter attr < s keeps ~s of the data
    db = MiniVectorDB(d, hnsw_threshold=1)
    report(0.1, "indexing")
    db.upsert([{"id": str(i), "vector": xb[i], "payload": {"attr": float(attr[i])}} for i in range(n)])
    db._ensure_index()
    rows = []
    for si, s in enumerate(selectivities):
        report(0.2 + 0.75 * si / len(selectivities), f"filter keeps {s:.1%}")
        allowed = np.flatnonzero(attr < s)
        if len(allowed) == 0:
            continue
        truth = []
        for q in xq:
            sc = xb[allowed] @ q
            truth.append(set(allowed[np.argsort(-sc)[:k]].tolist()))
        for mode in ("exact", "post", "in-graph"):
            rec, got, ms = [], [], []
            for q, tr in zip(xq, truth):
                t0 = time.perf_counter()
                res = _filtered(db, q, k, attr, s, mode)
                ms.append((time.perf_counter() - t0) * 1000)
                ids = {int(h["id"]) for h in res["hits"]}
                rec.append(len(ids & tr) / min(k, len(tr)))
                got.append(len(res["hits"]))
            rows.append({"selectivity": s, "survivors": int(len(allowed)), "mode": mode, "recall": round(float(np.mean(rec)), 3),
                         "avg_results": round(float(np.mean(got)), 2), "latency_ms": round(float(np.mean(ms)), 3)})
    report(1.0, "done")
    return {"settings": {"n": n, "d": d, "k": k, "overfetch": 3}, "rows": rows}


def _filtered(db, q, k, attr, s, mode):
    # a numeric range filter implemented directly on the metadata array (the payload dict equality filter is for category values)
    db_mask = attr < s
    db_alive_backup = db.alive
    try:
        db.alive = db_alive_backup & db_mask
        db._dirty = False                                      # keep the already-built graph
        return db.search(q, k=k, where=None, mode=mode)
    finally:
        db.alive = db_alive_backup
