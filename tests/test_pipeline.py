"""Run with:  pytest -q"""
from pathlib import Path

import pytest

from ingestion.chunking import chunk_document, count_tokens
from ingestion.models import Block
from ingestion.normalize import normalize_blocks
from ingestion.ocr import ocr_status
from ingestion.pipeline import make_doc_id, run_pipeline
from ingestion.store import Store

S = Path(__file__).resolve().parent.parent / "samples"
needs_ocr = pytest.mark.skipif(not ocr_status()["available"], reason="Tesseract not installed")


@pytest.fixture()
def store(tmp_path):
    return Store(tmp_path / "data")


def run(name, store=None, **kw):
    return run_pipeline(S / name, store=store, **kw)


def active_text(r):
    return r["stages"]["normalize"]["plain_text"]


# ---------------------------------------------------------------- parsing
def test_pdf_headers_footers_tables_footnotes_duplicates():
    r = run("messy_report.pdf")
    blocks = r["stages"]["normalize"]["blocks"]
    removed = [b for b in blocks if b["removed_reason"]]
    assert sum(b["type"] == "header" for b in removed) == 3 and sum(b["type"] == "footer" for b in removed) == 3
    assert sum(b["type"] == "table" for b in blocks) == 1
    assert sum(b["type"] == "footnote" for b in blocks) == 3
    assert len(r["stages"]["dedup"]["exact"]) == 1                      # the repeated disclaimer
    text = active_text(r)
    assert "infrastructure" in text and "infra-" not in text            # de-hyphenated
    assert "risk for the regions that bill" in text                     # paragraph merged across the page break
    assert "Confidential" not in text and "ACME Corp - Quarterly" not in text
    assert "(cid:" not in text and "• Revenue growth" not in text       # bullets repaired, then markers stripped
    assert "(cid:127)" in r["stages"]["raw"]["text"]                    # ...but the naive text shows the failure


def test_docx_header_table_merged_cells_footnotes():
    r = run("handbook_v2.docx")
    blocks = r["stages"]["parse"]["blocks"]
    assert any(b["type"] == "header" and b["removed_reason"] for b in blocks)
    table = next(b for b in blocks if b["type"] == "table")
    assert table["meta"]["merged_cells"] >= 1 and "| Seniority |" in table["text"]
    assert [b["text"] for b in blocks if b["type"] == "footnote"][0].startswith("Leave year")
    assert "Seniority" not in r["stages"]["raw"]["text"]                # naive python-docx loop drops tables


def test_html_boilerplate_and_footnotes():
    r = run("faq_page.html")
    blocks = r["stages"]["parse"]["blocks"]
    gone = [b["type"] for b in blocks if b["removed_reason"]]
    assert {"header", "boilerplate", "footer"} <= set(gone)
    assert "tracking" not in active_text(r)                             # <script> content is gone
    assert "two business days.[1] Express" in active_text(r)            # <sup>1</sup> did not glue onto the word
    assert sum(b["type"] == "footnote" for b in blocks) == 2


@needs_ocr
def test_ocr_image_and_scanned_pdf():
    for name in ("scan_note.png", "scanned_report.pdf"):
        r = run(name)
        text = active_text(r)
        assert "10254" in text and "1,250.00" in text, name
        assert r["source"]["doc_meta"]["ocr_mean_confidence"] > 80
        assert any(w["code"].startswith("OCR") for w in r["stages"]["parse"]["warnings"])


def test_failures_do_not_crash(tmp_path):
    r = run("broken.pdf")
    assert r["source"]["status"] == "failed"
    assert r["stages"]["parse"]["warnings"][0]["code"] == "PDF_CORRUPT"
    fake = tmp_path / "report.pdf"
    fake.write_bytes((S / "handbook_v1.docx").read_bytes())             # a .docx pretending to be a .pdf
    r = run_pipeline(fake)
    assert any(w["code"] == "EXTENSION_MISMATCH" for w in r["stages"]["parse"]["warnings"])
    assert r["source"]["file_type"] == "docx" and r["stages"]["chunks"]["items"]
    odd = tmp_path / "x.xyz"
    odd.write_bytes(b"hello")
    assert run_pipeline(odd)["stages"]["parse"]["warnings"][0]["code"] == "UNSUPPORTED_TYPE"
    latin = tmp_path / "l.txt"
    latin.write_bytes("Caf\xe9 au lait is served daily at the main caf\xe9 downtown.".encode("cp1252"))
    r = run_pipeline(latin)
    assert "Café" in active_text(r) and r["stages"]["parse"]["warnings"][0]["code"] == "ENCODING_FALLBACK"


# ---------------------------------------------------------------- normalization
def norm(text, type="paragraph", **meta):
    out, _ = normalize_blocks([Block(type, text, meta=meta)])
    return out[0].text


def test_normalization_rules():
    assert norm("infra-\nstructure rollout") == "infrastructure rollout"
    assert norm("state-\nOf") == "state- Of"                             # next part is uppercase: keep the hyphen
    assert norm("of\ufb01ce \u201cquoted\u201d  text\u00a0here") == 'office "quoted" text here'
    assert norm("zero\u200bwidth soft\u00adhyphen") == "zerowidth softhyphen"
    assert norm("\u09ae\u09bf\u200c\u09b2") == "\u09ae\u09bf\u200c\u09b2"  # ZWNJ must survive (Bangla / Persian)
    assert norm("Total 1O25 and l25 items", ocr=True) == "Total 1025 and 125 items"
    assert norm("code l5O stays", ocr=True) == "code l5O stays"          # ambiguous (one digit): left alone on purpose
    assert norm("Total 1O25", ocr=False) == "Total 1O25"                 # digit fix is OCR-only
    assert norm("\u2022 first item", type="list_item") == "first item"


# ---------------------------------------------------------------- chunking
@pytest.mark.parametrize("strategy", ["structure", "fixed", "semantic"])
def test_chunk_size_limits_and_determinism(strategy):
    r = run("messy_report.pdf", strategy=strategy, max_tokens=60, overlap=10)
    leaf = [c for c in r["stages"]["chunks"]["items"] if c["kind"] != "parent"]
    assert leaf
    for c in leaf:
        if strategy == "semantic" and "table" in c["block_types"]:
            continue                                                       # semantic keeps a table atomic by design
        assert c["tokens"] <= 60, (strategy, c["tokens"], c["text"][:60])
    again = run("messy_report.pdf", strategy=strategy, max_tokens=60, overlap=10)
    assert [c["chunk_id"] for c in leaf] == [c["chunk_id"] for c in again["stages"]["chunks"]["items"] if c["kind"] != "parent"]


def test_structure_chunks_respect_headings_and_parents():
    r = run("messy_report.pdf")
    items = r["stages"]["chunks"]["items"]
    parents = {c["chunk_id"]: c for c in items if c["kind"] == "parent"}
    for c in items:
        if c["kind"] == "child":
            assert c["parent_id"] in parents and parents[c["parent_id"]]["section_path"] == c["section_path"]
    table_chunks = [c for c in items if c["kind"] == "child" and "table" in c["block_types"]]
    assert table_chunks and all(c["text"].startswith("| Region") for c in table_chunks)
    assert all(c["embed_text"].startswith(" > ".join(c["section_path"])) for c in items if c["section_path"])


def test_big_table_splits_by_rows_and_repeats_header():
    rows = [["id", "value"]] + [[str(i), "x" * 5] for i in range(60)]
    from ingestion.utils import table_to_text
    b = Block("table", table_to_text(rows), meta={"rows": rows})
    chunks = chunk_document([Block("heading", "T", level=1), b], "doc_x", "structure", max_tokens=80, overlap=0)
    kids = [c for c in chunks if c.kind == "child"]
    assert len(kids) > 1 and all(c.text.startswith("| id | value |") for c in kids) and all(c.tokens <= 80 for c in kids)


def test_fixed_chunks_cut_mid_sentence_semantic_less_so():
    comp = run("handbook_v1.docx", max_tokens=40, overlap=5)["stages"]["chunks"]["comparison"]
    assert comp["fixed"]["cut_mid_sentence"] >= comp["structure"]["cut_mid_sentence"]


# ---------------------------------------------------------------- identity, versioning, incremental reindex
def test_doc_id_is_stable_and_case_insensitive():
    assert make_doc_id("Handbook.docx") == make_doc_id(" handbook.DOCX ") != make_doc_id("other.docx")


def test_versioning_and_incremental_reindex(store):
    v1 = run("handbook_v1.docx", store, source_name="handbook.docx")
    assert (v1["source"]["status"], v1["source"]["version"]) == ("new", 1)
    again = run("handbook_v1.docx", store, source_name="handbook.docx")
    assert again["source"]["status"] == "unchanged" and again["stages"]["store"]["diff"] is None

    v2 = run("handbook_v2.docx", store, source_name="handbook.docx")
    assert (v2["source"]["status"], v2["source"]["version"]) == ("updated", 2)
    assert v2["source"]["doc_id"] == v1["source"]["doc_id"]
    d = v2["stages"]["store"]["diff"]
    assert d["added"] and d["removed"] and d["unchanged"]
    assert len(d["embed_these"]) < len(v2["stages"]["chunks"]["items"])    # not everything is re-embedded

    detail = store.document_detail(v1["source"]["doc_id"])
    assert len(detail["versions"]) == 2
    retired = [c for c in detail["chunks"] if c["version_removed"] == 2]
    assert retired and all(c["version_added"] == 1 for c in retired)
    active = [c for c in detail["chunks"] if c["version_removed"] is None]
    assert len(active) == v2["stages"]["chunks"]["stats"]["total"]
    assert any("25 days" in c["text"] for c in active) and not any("20 days" in c["text"] for c in active)

    again2 = run("handbook_v2.docx", store, source_name="handbook.docx", max_tokens=100)
    assert again2["source"]["status"] == "reprocessed" and again2["source"]["version"] == 3


def test_duplicate_file_and_failed_file_do_not_touch_index(store):
    run("messy_report.pdf", store)
    dup = run("copy_of_messy_report.pdf", store)
    assert dup["source"]["status"] == "duplicate" and dup["source"]["duplicate_of"] == "messy_report.pdf"
    assert len(store.list_documents()["documents"]) == 1 and len(store.list_documents()["aliases"]) == 1
    assert run("broken.pdf", store)["source"]["status"] == "failed"
    assert len(store.list_documents()["documents"]) == 1


def test_preview_mode_saves_nothing(store):
    r = run("faq_page.html", store, save=False)
    assert r["source"]["status"] == "new" and not r["source"]["saved"]
    assert store.list_documents()["documents"] == []


def test_near_duplicate_across_documents(store, tmp_path):
    run("handbook_v1.docx", store, source_name="a.docx")
    r = run("handbook_v2.docx", store, source_name="b.docx")             # nearly the same content, different source
    assert any(p["scope"].startswith("other document") for p in r["stages"]["dedup"]["near"])


# ---------------------------------------------------------------- web API
def test_api_end_to_end(tmp_path):
    from webapp.app import create_app
    c = create_app(tmp_path / "d").test_client()
    assert c.get("/").status_code == 200
    r = c.post("/api/ingest-sample", data={"sample": "handbook_v1.docx"})
    assert r.status_code == 200 and r.json["source"]["status"] == "new"
    with open(S / "faq_page.html", "rb") as f:
        r = c.post("/api/ingest", data={"file": (f, "faq.html"), "strategy": "semantic"})
    assert r.status_code == 200 and r.json["stages"]["chunks"]["strategy"] == "semantic"
    lib = c.get("/api/library").json
    assert len(lib["documents"]) == 2
    assert c.get(f"/api/library/{lib['documents'][0]['doc_id']}").status_code == 200
    assert c.post("/api/ingest").status_code == 400
    assert c.post("/api/reset").json["ok"] and c.get("/api/library").json["documents"] == []
