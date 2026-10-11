from pathlib import Path

from evaluation.benchmark import run_benchmark
from evaluation.metrics import cer, edit_distance, prf, wer
from ingestion.pipeline import run_pipeline

S = Path(__file__).resolve().parent.parent / "samples"


def test_metrics():
    assert edit_distance("kitten", "sitting") == 3
    assert cer("abcd", "abcd") == 0 and cer("abcd", "abxd") == 0.25
    assert wer("the cat sat", "the cat sat down") == 1 / 3
    assert prf({"a", "b"}, {"a", "c"}) == (0.5, 0.5, 0.5)


def test_integrity_checks_on_clean_document():
    ev = run_pipeline(S / "messy_report.pdf")["evaluation"]
    assert ev["failed"] == 0 and ev["score"] >= 90
    assert {c["id"] for c in ev["checks"]} >= {"parse_recall", "clean_text", "coverage", "no_leak", "size", "parent_child", "tables", "deterministic", "metadata"}


def test_checks_catch_real_problems(tmp_path):
    # a chunk limit smaller than a table row forces an oversize chunk -> the size check must notice
    ev = run_pipeline(S / "messy_report.pdf", strategy="fixed", max_tokens=60, overlap=10)["evaluation"]
    by_id = {c["id"]: c for c in ev["checks"]}
    assert by_id["mid_sentence"]["status"] in ("warn", "fail")        # fixed windows cut sentences: the check must say so
    assert by_id["tables"]["status"] in ("warn", "fail")              # ...and split tables
    assert by_id["parent_child"]["status"] == "n/a"


def test_benchmark_runs_and_passes():
    rep = run_benchmark(S)
    s = rep["summary"]
    assert s["gold_checks_passed"] == s["gold_checks_total"] and s["integrity_failed_checks"] == 0
    for strat in rep["retrieval_by_strategy"].values():
        assert strat["answer_kept"] == 1.0 and strat["hit3"] >= 0.9


def test_benchmark_endpoint(tmp_path):
    from webapp.app import create_app
    r = create_app(tmp_path / "d").test_client().get("/api/benchmark?max_tokens=120&overlap=20")
    assert r.status_code == 200 and r.json["settings"]["max_tokens"] == 120
