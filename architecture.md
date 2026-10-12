# Architecture

## 1. Big picture

```
 labeled dataset (corpus.jsonl + queries.jsonl)          one query type per query:
        │                                                keyword · paraphrase · question · typo · identifier
        ▼
 ┌────────────────────────── retrieval/compare.py ──────────────────────────┐
 │  SPARSE            DENSE (one score matrix per model)           HYBRID    │
 │  BM25, TF-IDF      embedders.py: LSA · char-hash · GloVe ·       RRF /    │
 │  (sparse.py)       sentence-transformers (optional)              linear   │
 │                    cosine = dot product on unit vectors          (hybrid.py)
 │                              │                                            │
 │                              ▼                                            │
 │        rankings → metrics.py (recall@k, MRR, nDCG) overall AND per type   │
 │        failure analysis: who found what, heuristic "why" for each miss    │
 └───────────────────────────────────────────────────────────────────────────┘
        │
        ▼ the dense retriever at scale
 ┌──────────────── retrieval/ann.py (FAISS) ───────────────┐   ┌──── retrieval/vectordb.py ────┐
 │ Flat (exact)  HNSW (M, efSearch)  IVF-Flat (nlist, nprobe) │   │ MiniVectorDB: upsert / delete  │
 │ IVF-PQ  IVF-PQ + exact re-rank  PQ                         │   │ HNSW index, payload filters,   │
 │ recall@k vs exact · latency · RAM · build time             │   │ query planner: exact / post /  │
 └────────────────────────────────────────────────────────────┘   │ in-graph; filter experiment    │
                                                                  └────────────────────────────────┘
        ▼
 Flask API (webapp/app.py) ── background jobs (jobs.py) ──► single-page UI (HTML + CSS + JS, SVG charts)
```

## 2. The assignment, requirement by requirement

| Requirement | Where it is addressed |
|---|---|
| Embeddings | `embedders.py`: four families behind one interface (`fit`, `encode(texts, role, normalize)`), plus the playground and 2D map in tab 2 |
| Cosine similarity | `dense.py`; the *metric demo* in tab 3 measures how often dot product and Euclidean distance disagree with cosine on raw vectors, and shows they agree after L2 normalization |
| Compare dense retrieval approaches on a labeled dataset | `compare.py` + `datasets/acme_kb`; leaderboard, per-query-type heatmap, per-query inspector; upload your own dataset |
| Dense vs sparse and their failure modes | Query types are built to trigger specific failures: **paraphrase** (vocabulary mismatch breaks BM25), **typo** (unknown token breaks BM25), **identifier** (near-identical passages break dense), plus the complementarity analysis and the heuristic "why" per miss |
| HNSW, IVF, PQ | `ann.py`: each family with its quality knob swept (`efSearch`, `nprobe`), PQ with and without exact re-ranking |
| Recall / latency trade-offs | Tab 4: recall-vs-latency curves, "fastest setting reaching recall 90/95/99%", memory per vector, build time |
| Vector databases | `vectordb.py` and tab 5: storage, payloads, upsert/delete, index maintenance, persistence, filtered search and its failure modes |

## 3. Design decisions

- **Unit vectors everywhere.** Every embedder returns L2-normalized vectors by default, so cosine similarity is a dot product and Euclidean distance gives the same ranking. `normalize=False` exists to demonstrate what goes wrong without it.
- **Fit on the corpus, never on the queries.** LSA and the IDF weights in GloVe averaging are fitted on corpus text only (no label leakage).
- **Score matrices.** Every method produces a `queries × documents` score matrix; rankings, hybrids and metrics are derived from it. This keeps the comparison uniform. Latency is measured separately, one query at a time, through the real search call.
- **Honest labels.** The dataset builder measures the word overlap between every query and its answer: paraphrase queries share about 6% of their words, keyword queries about 96%. A test fails if the labels drift.
- **Fair latency.** The ANN benchmark measures single-query latency on one CPU thread (median of three repeats per query), and recall against exact search on the same vectors.
- **Synthetic vectors for ANN.** ANN methods only pay off at scale, and an 85-passage corpus cannot show that. The benchmark therefore uses embedding-like clustered vectors (decaying variance per dimension, L2-normalized). Real embeddings compress better under PQ, so absolute PQ recall here is pessimistic.
- **Failures never crash a run.** A model that cannot load is reported in `skipped`; the rest of the comparison completes.
- **Long work runs as jobs** (`jobs.py`): the UI polls `/api/jobs/<id>` and shows progress.

## 4. Embedding models

| id | Kind | Needs | What it knows |
|---|---|---|---|
| `lsa` | offline | nothing | word co-occurrence in YOUR corpus only; unknown words give a zero vector |
| `hash-ngram` | offline | nothing | spelling (character 3-5-grams hashed into 512-d); typo tolerant, no meaning |
| `glove-50` | pretrained | 69 MB download from GitHub, cached in `data/models` | word meaning (Wikipedia + Gigaword), averaged and IDF-weighted; ignores order; cannot read codes |
| `st:*` | neural | `pip install sentence-transformers`, weights from Hugging Face | sentence-level meaning; per-model query/passage prefixes are applied (`query: ` for e5, the BGE instruction) |

## 5. Mini vector database

`MiniVectorDB` stores unit vectors with a payload dict, keeps a tombstone mask for deletes, and lazily (re)builds an HNSW index when the vector count passes a threshold (below it, exact scan). Search strategies:

- **exact**: apply the filter, scan the survivors. Always correct; cost grows with survivors.
- **post**: ANN for `k × overfetch` candidates, then drop non-matching. Fast but returns fewer than `k` (or nothing) for selective filters.
- **in-graph**: HNSW search with an id selector, so non-matching nodes are skipped during the walk. Good until the filter is very selective, then the graph is too sparse to navigate.
- **auto** (the planner): scan when few vectors survive (≤ max(20k, 2% of the data)), in-graph otherwise, plain ANN with no filter.

## 6. Evaluation methodology

- Queries are labeled with the passage that answers them; metrics are recall@1/5/10, MRR@10 and nDCG@10, reported overall and per query type.
- *Complementarity*: for the top 5, each query is counted as found by both / only BM25 / only dense / neither. The "oracle" (either succeeds) bounds what any fusion can achieve.
- Failure explanations are heuristics (shared words, unknown tokens, digits in the query), meant for learning rather than as ground truth.
- `evaluate.py` reproduces everything headlessly and writes `evaluation/report.md`.
- 21 tests cover metrics, BM25 failure modes, embedder properties (unit norm, determinism, typo tolerance, no semantics for hashing), the sentence-transformers adapter (with a stub), dataset validation and label honesty, comparison findings, ANN behaviour (efSearch and nprobe move recall, PQ compresses and loses recall, re-ranking recovers it, HNSW costs memory), the vector DB (CRUD, filters, planner, persistence, post-filter under-filling) and the API.

## 7. Limitations

- The dataset is small (85 passages, 206 queries) and written for the assignment. Conclusions about which model "wins" belong to this dataset, not to retrieval in general.
- The offline and GloVe models are baselines. Neural models are expected to do much better on paraphrases; they are supported but were not benchmarked here (no access to Hugging Face in the development environment, so only the adapter logic is tested, with a stub).
- ANN timings depend on your CPU; compare the shapes of the curves, not the absolute milliseconds.
- `MiniVectorDB` is for learning: single process, in-memory, no concurrency control, rebuilds the graph after deletes.
- Retrieval quality is not answer quality. Week 6 will feed the best retriever into generation and evaluate answers.
