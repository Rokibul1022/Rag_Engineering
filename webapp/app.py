"""Tiny Flask server: one HTML page + a few JSON endpoints around run_pipeline()."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from ingestion.concepts import CONCEPTS
from ingestion.ocr import ocr_status
from ingestion.pipeline import run_pipeline
from ingestion.store import Store

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
SAMPLES = [
    {"file": "messy_report.pdf", "label": "Messy PDF: header/footer, table, footnotes, hyphenation, duplicate paragraph"},
    {"file": "scanned_report.pdf", "label": "Scanned PDF (no text layer → OCR)"},
    {"file": "scan_note.png", "label": "Image of text (PNG → OCR)"},
    {"file": "handbook_v1.docx", "label": "Word: Handbook v1 (source name handbook.docx)", "source_name": "handbook.docx"},
    {"file": "handbook_v2.docx", "label": "Word: Handbook v2 (same source name → new version)", "source_name": "handbook.docx"},
    {"file": "faq_page.html", "label": "HTML page: menu, cookie banner, footnotes"},
    {"file": "copy_of_messy_report.pdf", "label": "Identical copy of the messy PDF → duplicate file"},
    {"file": "broken.pdf", "label": "Truncated PDF → parsing failure"},
]


def create_app(data_dir=None) -> Flask:
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
    app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
    store = Store(data_dir or ROOT / "data")
    tmp_dir = Path(tempfile.gettempdir()) / "doc_ingestion_uploads"
    tmp_dir.mkdir(exist_ok=True)

    def options(form):
        def num(key, default, lo, hi):
            try:
                return max(lo, min(hi, int(form.get(key, default))))
            except (TypeError, ValueError):
                return default
        max_tokens = num("max_tokens", 200, 20, 2000)
        return dict(strategy=form.get("strategy", "structure") if form.get("strategy") in ("structure", "fixed", "semantic") else "structure",
                    max_tokens=max_tokens, overlap=min(num("overlap", 30, 0, 500), max_tokens // 2),
                    save=form.get("save", "1") not in ("0", "false", "off"), ocr_lang=form.get("ocr_lang", "eng") or "eng",
                    source_name=(form.get("source_name") or "").strip() or None)

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.post("/api/ingest")
    def ingest():
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "No file uploaded."}), 400
        original = Path(f.filename).name
        tmp = tmp_dir / f"{uuid.uuid4().hex}{Path(original).suffix.lower()}"
        f.save(tmp)
        try:
            return jsonify(run_pipeline(tmp, store=store, original_filename=original, **options(request.form)))
        except Exception as e:  # the pipeline reports parse problems itself; this is for true bugs
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
        finally:
            tmp.unlink(missing_ok=True)

    @app.post("/api/ingest-sample")
    def ingest_sample():
        name = request.form.get("sample", "")
        sample = next((s for s in SAMPLES if s["file"] == name), None)
        if not sample:
            return jsonify({"error": "Unknown sample."}), 400
        opts = options(request.form)
        opts["source_name"] = opts["source_name"] or sample.get("source_name")
        try:
            return jsonify(run_pipeline(SAMPLES_DIR / sample["file"], store=store, original_filename=sample["file"], **opts))
        except Exception as e:
            return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    @app.get("/api/samples")
    def samples():
        return jsonify(SAMPLES)

    @app.get("/api/concepts")
    def concepts():
        return jsonify(CONCEPTS)

    @app.get("/api/ocr")
    def ocr_info():
        return jsonify(ocr_status())

    @app.get("/api/library")
    def library():
        return jsonify(store.list_documents())

    @app.get("/api/library/<doc_id>")
    def library_doc(doc_id):
        d = store.document_detail(doc_id)
        return (jsonify(d), 200) if d else (jsonify({"error": "Not found"}), 404)

    @app.post("/api/reset")
    def reset():
        store.reset()
        return jsonify({"ok": True})

    return app
