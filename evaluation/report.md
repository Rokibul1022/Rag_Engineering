# Retrieval evaluation report

Dataset: **acme_kb (built-in)**, 85 passages, 206 labeled queries.

## Leaderboard

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | search ms p50 | index |
|---|---|---|---|---|---|---|---|
| Hybrid: BM25 + dense (score mix) [glove-50] | 0.738 | 0.840 | 0.883 | 0.781 | 0.806 | 0.032 | 33 KB |
| BM25 | 0.728 | 0.777 | 0.806 | 0.751 | 0.764 | 0.012 | 16 KB |
| TF-IDF cosine | 0.718 | 0.777 | 0.806 | 0.745 | 0.760 | 0.011 | 16 KB |
| Char n-gram hashing | 0.714 | 0.786 | 0.816 | 0.743 | 0.760 | 0.024 | 174 KB |
| LSA (TF-IDF + SVD) | 0.680 | 0.762 | 0.786 | 0.716 | 0.733 | 0.021 | 22 KB |
| Hybrid: BM25 + dense (RRF) [glove-50] | 0.631 | 0.796 | 0.845 | 0.699 | 0.734 | 0.032 | 33 KB |
| GloVe 50d (averaged word vectors) | 0.558 | 0.733 | 0.820 | 0.630 | 0.675 | 0.02 | 17 KB |

## MRR@10 by query type

| Method | keyword | paraphrase | question | typo | identifier |
|---|---|---|---|---|---|
| Hybrid: BM25 + dense (score mix) [glove-50] | 1.00 | 0.29 | 0.93 | 0.95 | 1.00 |
| BM25 | 1.00 | 0.17 | 0.94 | 0.95 | 1.00 |
| TF-IDF cosine | 1.00 | 0.16 | 0.93 | 0.95 | 1.00 |
| Char n-gram hashing | 1.00 | 0.18 | 0.87 | 1.00 | 1.00 |
| LSA (TF-IDF + SVD) | 1.00 | 0.16 | 0.92 | 0.97 | 0.73 |
| Hybrid: BM25 + dense (RRF) [glove-50] | 0.99 | 0.19 | 0.87 | 0.88 | 0.74 |
| GloVe 50d (averaged word vectors) | 0.91 | 0.33 | 0.74 | 0.72 | 0.40 |

## BM25 vs glove-50 (answer in top 5)

| both | only sparse | only dense | neither | oracle |
|---|---|---|---|---|
| 132 | 28 | 19 | 27 | 87% |

## ANN benchmark (20,000 synthetic vectors, 128 dims, exact search 0.393 ms)

| Index | recall>=0.95: setting | latency ms | speedup | RAM/vector |
|---|---|---|---|---|
| HNSW M=8 | ef=32 | 0.0147 | 26.8x | 592.6 B |
| HNSW M=16 | ef=32 | 0.0164 | 23.94x | 656.2 B |
| HNSW M=32 | ef=16 | 0.0148 | 26.58x | 784.1 B |
| IVF-Flat nlist=566 | nprobe=8 | 0.0162 | 24.3x | 534.7 B |
| IVF-PQ m=16 | not reached | - | - | 45.3 B |
| IVF-PQ m=16 + exact re-rank | not reached | - | - | 45.3 B |
| PQ m=16 (no IVF) | not reached | - | - | 22.6 B |

## Filtered search (20,000 vectors, k=10)

| filter keeps | mode | recall | results returned | ms |
|---|---|---|---|---|
| 50.0% | exact | 1.0 | 10.0 | 0.657 |
| 50.0% | post | 0.91 | 9.97 | 0.085 |
| 50.0% | in-graph | 0.999 | 10.0 | 0.905 |
| 20.0% | exact | 1.0 | 10.0 | 0.301 |
| 20.0% | post | 0.575 | 5.78 | 0.08 |
| 20.0% | in-graph | 0.995 | 10.0 | 0.502 |
| 5.0% | exact | 1.0 | 10.0 | 0.093 |
| 5.0% | post | 0.159 | 1.59 | 0.073 |
| 5.0% | in-graph | 0.968 | 10.0 | 0.25 |
| 1.0% | exact | 1.0 | 10.0 | 0.047 |
| 1.0% | post | 0.019 | 0.19 | 0.062 |
| 1.0% | in-graph | 0.439 | 4.55 | 0.143 |
| 0.2% | exact | 1.0 | 10.0 | 0.041 |
| 0.2% | post | 0.008 | 0.08 | 0.056 |
| 0.2% | in-graph | 0.076 | 0.76 | 0.118 |
