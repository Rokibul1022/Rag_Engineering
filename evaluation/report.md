# Retrieval evaluation report

Dataset: **custom**, 4132 passages, 500 labeled queries.

## Leaderboard

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | search ms p50 | index |
|---|---|---|---|---|---|---|---|
| BM25 | 0.220 | 0.730 | 0.919 | 0.447 | 0.559 | 0.174 | 1314 KB |
| TF-IDF cosine | 0.209 | 0.696 | 0.911 | 0.429 | 0.543 | 0.055 | 1314 KB |

## MRR@10 by query type

| Method | numeric | description | location | entity | person |
|---|---|---|---|---|---|
| BM25 | 0.42 | 0.47 | 0.35 | 0.38 | 0.59 |
| TF-IDF cosine | 0.41 | 0.44 | 0.37 | 0.39 | 0.55 |
