"""Flask server: one HTML page + JSON endpoints around the retrieval package."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from retrieval import ann as annmod
from retrieval import compare as cmpmod
from retrieval import dataset as dsmod
from retrieval import embedders
from retrieval import vectordb as vdbmod
from retrieval.concepts import CONCEPTS, build_evidence
from retrieval.jobs import Jobs

ROOT = Path(__file__).resolve().parent.parent


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
    jobs = Jobs()
    state = {"ds": dsmod.load_builtin(), "cmp": None, "ann": None, "flt": None, "vdb": None, "map_cache": {}}

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    def body():
        return request.get_json(silent=True) or {}

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/status")
    def status():
        return jsonify({"faiss": annmod.faiss_status(), "dataset": dsmod.summary(state["ds"])})

    # ---------------------------------------------------------------- dataset
    @app.get("/api/dataset")
    def dataset():
        ds = state["ds"]
        return jsonify({**dsmod.summary(ds), "corpus": ds["corpus"], "queries": ds["queries"]})

    @app.post("/api/dataset/upload")
    def dataset_upload():
        c, q = request.files.get("corpus"), request.files.get("queries")
        if not c or not q:
            return err("Upload both files: corpus.jsonl and queries.jsonl.")
        try:
            ds = dsmod.from_text(c.read().decode("utf-8-sig"), q.read().decode("utf-8-sig"), name=f"uploaded ({c.filename})")
        except (ValueError, UnicodeDecodeError) as e:
            return err(str(e))
        state.update(ds=ds, cmp=None, vdb=None, map_cache={})
        return jsonify(dsmod.summary(ds))

    @app.post("/api/dataset/reset")
    def dataset_reset():
        state.update(ds=dsmod.load_builtin(), cmp=None, vdb=None, map_cache={})
        return jsonify(dsmod.summary(state["ds"]))

    @app.get("/api/dataset/download/<which>")
    def dataset_download(which):
        if which not in ("corpus", "queries"):
            return err("unknown file", 404)
        return send_from_directory(dsmod.BUILTIN, f"{which}.jsonl", as_attachment=True)

    # ---------------------------------------------------------------- models / jobs
    @app.get("/api/models")
    def models():
        return jsonify(embedders.list_models())

    @app.post("/api/models/<model_id>/prepare")
    def prepare(model_id):
        try:
            emb = embedders.create(model_id)
        except KeyError:
            return err("unknown model", 404)
        return jsonify({"job": jobs.start(lambda report: emb.prepare(report) or {"ok": True})})

    @app.get("/api/jobs/<jid>")
    def job(jid):
        j = jobs.get(jid)
        return (jsonify(j), 200) if j else err("unknown job", 404)

    def ready_models(ids=None):
        out = []
        for m in embedders.list_models():
            if m["state"] == "ready" and (ids is None or m["id"] in ids):
                out.append(m["id"])
        return out

    def fitted(model_id):
        emb = embedders.create(model_id)
        emb.fit([d["text"] for d in state["ds"]["corpus"]])
        return emb

    # ---------------------------------------------------------------- embeddings tab
    @app.post("/api/similarity")
    def similarity():
        b = body()
        a, t = (b.get("a") or "").strip(), (b.get("b") or "").strip()
        if not a or not t:
            return err("Enter two texts.")
        rows = []
        for mid in ready_models(b.get("models")):
            try:
                emb = fitted(mid)
                raw = emb.encode([a, t], "doc", normalize=False)
                unit = embedders.l2norm(raw)
                rows.append({"model": mid, "label=": None, "dim": int(raw.shape[1]), "cosine": round(float(unit[0] @ unit[1]), 4),
                             "dot_raw": round(float(raw[0] @ raw[1]), 4), "l2_raw": round(float(np.linalg.norm(raw[0] - raw[1])), 4),
                             "norms": [round(float(np.linalg.norm(raw[0])), 3), round(float(np.linalg.norm(raw[1])), 3)],
                             "preview": [round(float(x), 3) for x in unit[0][:8]]})
            except Exception as e:
                rows.append({"model": mid, "error": f"{type(e).__name__}: {e}"})
        labels = {m["id"]: m["label"] for m in embedders.list_models()}
        for r in rows:
            r["label"] = labels.get(r["model"], r["model"])
            r.pop("label=", None)
        return jsonify({"rows": rows})

    @app.post("/api/embedding-map")
    def embedding_map():
        mid = body().get("model", "lsa")
        if mid not in ready_models():
            return err("That model is not ready. Prepare it first.")
        try:
            emb = fitted(mid)
            X = emb.encode([d["text"] for d in state["ds"]["corpus"]], "doc")
        except Exception as e:
            return err(f"{type(e).__name__}: {e}")
        Xc = X - X.mean(axis=0)
        _, S, vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ vt[:2].T
        var = (S[:2] ** 2).sum() / (S ** 2).sum()
        pts = [{"id": d["id"], "title": d["title"] or d["text"][:40], "domain": d["domain"], "x": round(float(P[i, 0]), 4), "y": round(float(P[i, 1]), 4)}
               for i, d in enumerate(state["ds"]["corpus"])]
        qs = []
        for q in state["ds"]["queries"][:60]:
            qv = emb.encode([q["query"]], "query")
            qp = (qv - X.mean(axis=0)) @ vt[:2].T
            qs.append({"id": q["id"], "query": q["query"], "type": q["type"], "x": round(float(qp[0, 0]), 4), "y": round(float(qp[0, 1]), 4)})
        return jsonify({"points": pts, "queries": qs, "explained_variance": round(float(var), 3), "model": mid})

    # ---------------------------------------------------------------- compare
    @app.get("/api/methods")
    def methods():
        return jsonify(cmpmod.method_catalog())

    @app.post("/api/compare")
    def compare():
        b = body()
        ids = b.get("methods") or []
        if not ids:
            return err("Select at least one method.")
        ds = state["ds"]

        def work(report):
            res = cmpmod.run_comparison(ds, ids, hybrid_dense=b.get("hybrid_dense"), report=report)
            state["cmp"] = res
            return res
        return jsonify({"job": jobs.start(work)})

    # ---------------------------------------------------------------- ANN benchmark
    @app.post("/api/ann/run")
    def ann_run():
        if not annmod.faiss_status()["available"]:
            return err(annmod.faiss_status()["reason"], 503)
        b = body()
        n = int(max(2000, min(200_000, b.get("n", 30000))))
        d = int(max(16, min(768, b.get("d", 128))))
        k = int(max(1, min(100, b.get("k", 10))))

        def work(report):
            res = annmod.run_benchmark(n=n, d=d, k=k, clusters=int(b.get("clusters", 64)), report=report)
            state["ann"] = res
            return res
        return jsonify({"job": jobs.start(work)})

    @app.post("/api/vdb/filter-experiment")
    def vdb_filter():
        if not annmod.faiss_status()["available"]:
            return err(annmod.faiss_status()["reason"], 503)
        b = body()

        def work(report):
            res = vdbmod.filter_experiment(n=int(max(2000, min(100_000, b.get("n", 20000)))), d=int(b.get("d", 64)), k=int(b.get("k", 10)), report=report)
            state["flt"] = res
            return res
        return jsonify({"job": jobs.start(work)})

    # ---------------------------------------------------------------- mini vector database
    @app.post("/api/vdb/build")
    def vdb_build():
        mid = body().get("model", "lsa")
        if mid not in ready_models():
            return err("That model is not ready. Prepare it first.")
        ds = state["ds"]

        def work(report):
            emb = fitted(mid)
            report(0.3, "embedding the corpus")
            X = emb.encode([d["text"] for d in ds["corpus"]], "doc")
            db = vdbmod.MiniVectorDB(X.shape[1], hnsw_threshold=50)
            db.upsert([{"id": d["id"], "vector": X[i], "payload": {"domain": d["domain"], "title": d["title"], "text": d["text"]}} for i, d in enumerate(ds["corpus"])])
            state["vdb"] = {"db": db, "emb": emb, "model": mid}
            return {"count": db.count(), "dim": int(X.shape[1]), "model": mid, "domains": sorted({d["domain"] for d in ds["corpus"]})}
        return jsonify({"job": jobs.start(work)})

    def need_vdb():
        return state["vdb"]

    @app.get("/api/vdb/info")
    def vdb_info():
        v = need_vdb()
        if not v:
            return jsonify({"built": False})
        db = v["db"]
        return jsonify({"built": True, "count": db.count(), "dim": db.dim, "model": v["model"], "indexed_with_hnsw": db._index is not None,
                        "domains": sorted({p.get("domain", "") for p, a in zip(db.payloads, db.alive) if a})})

    @app.post("/api/vdb/search")
    def vdb_search():
        v = need_vdb()
        if not v:
            return err("Build the database first.")
        b = body()
        q = (b.get("query") or "").strip()
        if not q:
            return err("Enter a query.")
        where = {"domain": b["domain"]} if b.get("domain") else None
        qv = v["emb"].encode([q], "query")[0]
        res = v["db"].search(qv, k=int(b.get("k", 5)), where=where, mode=b.get("mode", "auto"))
        for h in res["hits"]:
            h["payload"] = {"title": h["payload"].get("title"), "domain": h["payload"].get("domain"), "text": h["payload"].get("text", "")[:160]}
        return jsonify(res)

    @app.post("/api/vdb/upsert")
    def vdb_upsert():
        v = need_vdb()
        b = body()
        if not v or not (b.get("text") or "").strip():
            return err("Build the database first and enter some text.")
        did = (b.get("id") or "").strip() or f"new{v['db'].count() + 1}"
        vec = v["emb"].encode([b["text"]], "doc")[0]
        v["db"].upsert([{"id": did, "vector": vec, "payload": {"domain": b.get("domain") or "new", "title": b.get("title") or did, "text": b["text"]}}])
        return jsonify({"id": did, "count": v["db"].count()})

    @app.post("/api/vdb/delete")
    def vdb_delete():
        v = need_vdb()
        if not v:
            return err("Build the database first.")
        n = v["db"].delete([str(body().get("id", ""))])
        return jsonify({"deleted": n, "count": v["db"].count()})

    # ---------------------------------------------------------------- concepts
    @app.get("/api/concepts")
    def concepts():
        return jsonify({"concepts": CONCEPTS, "evidence": build_evidence(state["cmp"], state["ann"], state["flt"], embedders.list_models())})

    return app
