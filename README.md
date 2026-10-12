# Vector search lab (Week 5: embeddings and vector search)

Compare **sparse, dense and hybrid retrieval** on a labeled dataset, benchmark **ANN indexes** (HNSW, IVF, PQ) on recall, latency and memory, and explore a **mini vector database**. Everything runs in the browser.
Builds toward the *Evaluated document Q&A system*. See [`architecture.md`](architecture.md) for how it works.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```
Open **http://localhost:8000** (it also opens automatically).

### Optional: real neural embedding models
```bash
pip install sentence-transformers      # large (PyTorch); weights download from Hugging Face on first use
```
Restart, then tab 2 shows MiniLM, BGE-small and multilingual-e5-small (Bangla supported) as ready to load. Without it you still get BM25, TF-IDF, LSA, character n-gram hashing and GloVe (69 MB, downloaded from GitHub once).

## Tour (about 10 minutes)
1. **Dataset**: 85 passages and 206 queries, each labeled with its answer and a *query type* (keyword, paraphrase, question, typo, identifier). Upload your own JSONL files to evaluate other data.
2. **Embeddings**: prepare GloVe, try the *similarity playground* (synonyms, negation, near-identical codes), draw the 2D embedding map.
3. **Compare retrievers**: run everything, then read the per-query-type table, the "where they differ" bar and the failure examples with reasons.
4. **ANN indexes**: run the benchmark and read the recall-vs-latency chart and the memory table.
5. **Vector DB**: build the mini database, search with a domain filter, try the strategies, run the filter experiment.
6. **Concepts**: every topic explained next to the numbers from YOUR runs.

## From the terminal
```bash
python evaluate.py                        # full comparison + ANN + filter experiment, writes evaluation/report.md
python evaluate.py --methods bm25 glove-50
python evaluate.py --corpus c.jsonl --queries q.jsonl
pytest -q                                 # 21 tests
python datasets/acme_kb/build_dataset.py  # regenerate the built-in dataset
```

## Tools (beginner friendly)
Flask (server) · NumPy (vectors, written-from-scratch BM25, LSA, cosine) · FAISS (HNSW, IVF, PQ; `faiss-cpu`) · plain HTML/CSS/JavaScript with hand-written SVG charts. No GPU, no API keys, no database to install.

## Layout
```
run.py  evaluate.py  architecture.md
retrieval/   sparse.py (BM25, TF-IDF)  embedders.py  dense.py  hybrid.py  metrics.py  compare.py
             ann.py (FAISS benchmark)  vectordb.py (mini vector DB)  dataset.py  concepts.py  jobs.py
webapp/      app.py  static/index.html
datasets/acme_kb/   corpus.jsonl  queries.jsonl  build_dataset.py
tests/       test_lab.py
```
