"""Embedding models: text -> vector.  All return float32 arrays; by default L2-normalized, so cosine similarity = dot product.

  lsa        TF-IDF + SVD (Latent Semantic Analysis). No download. Learns word co-occurrence from YOUR corpus only.
  hash-ngram character n-gram hashing (fastText-style subwords). No download. Tolerates typos, knows no meaning.
  glove-50   average of pretrained GloVe word vectors (downloads 69 MB from GitHub once). A real pretrained semantic space.
  st:*       neural sentence-transformers models (pip install sentence-transformers; weights download from Hugging Face).
"""
from __future__ import annotations

import gzip
import math
import os
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from .text import tokenize

MODELS_DIR = Path(os.environ.get("LAB_MODELS_DIR", Path(__file__).resolve().parent.parent / "data" / "models"))


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class Embedder:
    id = label = kind = description = ""
    dim = 0
    needs_fit = False

    def status(self) -> dict:
        return {"state": "ready", "note": ""}

    def prepare(self, report=None):
        pass

    def fit(self, texts):
        return self

    def encode(self, texts, role="doc", normalize=True) -> np.ndarray:
        raise NotImplementedError


# ------------------------------------------------------------------ LSA
class LSAEmbedder(Embedder):
    id, label, kind = "lsa", "LSA (TF-IDF + SVD)", "offline"
    description = "Latent Semantic Analysis: compresses TF-IDF with SVD so words that co-occur share dimensions. Fitted on the corpus, no download."
    needs_fit = True

    def __init__(self, dim=64):
        self.dim = dim

    def fit(self, texts):
        docs = [tokenize(t) for t in texts]
        df = Counter(w for d in docs for w in set(d))
        self.vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items()))}
        n, v = len(docs), len(self.vocab)
        if n * v > 60_000_000:
            raise ValueError("corpus too large for the dense numpy LSA demo (use another model)")
        self.idf = np.zeros(v, dtype=np.float32)
        for w, i in self.vocab.items():
            self.idf[i] = math.log((n + 1) / (df[w] + 1)) + 1
        X = np.vstack([self._tfidf(t) for t in texts]).astype(np.float32)
        k = max(2, min(self.dim, n - 1, v - 1))
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        self.V = vt[:k].T.copy()                     # vocabulary x k: the learned "concept" directions
        self.dim = k
        return self

    def _tfidf(self, text):
        row = np.zeros(len(self.vocab), dtype=np.float32)
        for w, c in Counter(tokenize(text)).items():
            if w in self.vocab:
                row[self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        return row

    def encode(self, texts, role="doc", normalize=True):
        x = np.vstack([self._tfidf(t) for t in texts]) @ self.V
        return l2norm(x) if normalize else x.astype(np.float32)


# ------------------------------------------------------------------ char n-gram hashing
class HashNgramEmbedder(Embedder):
    id, label, kind = "hash-ngram", "Char n-gram hashing", "offline"
    description = "Hashes character 3-5-grams of every word into a 512-d vector (the subword idea of fastText). Robust to typos; has no notion of meaning."

    def __init__(self, dim=512, nmin=3, nmax=5):
        self.dim, self.nmin, self.nmax = dim, nmin, nmax

    def _feats(self, tok):
        p = f"<{tok}>"
        yield tok
        for n in range(self.nmin, self.nmax + 1):
            for i in range(len(p) - n + 1):
                yield p[i:i + n]

    def encode(self, texts, role="doc", normalize=True):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for tok in tokenize(t):
                for f in self._feats(tok):
                    h = zlib.crc32(f.encode("utf-8"))
                    out[r, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return l2norm(out) if normalize else out


# ------------------------------------------------------------------ GloVe (pretrained static word vectors, averaged)
GLOVE_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-{d}/glove-wiki-gigaword-{d}.gz"
_GLOVE_CACHE: dict = {}


def _glove_paths(dim):
    return MODELS_DIR / f"glove-{dim}.npy", MODELS_DIR / f"glove-{dim}.vocab.txt", MODELS_DIR / f"glove-{dim}.gz"


def glove_cached(dim) -> bool:
    npy, vocab, _ = _glove_paths(dim)
    return npy.exists() and vocab.exists()


def glove_prepare(dim=50, report=None, max_words=200_000):
    """Download (once) and convert to a fast .npy cache."""
    report = report or (lambda *a, **k: None)
    if glove_cached(dim):
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    npy, vocab, gz = _glove_paths(dim)
    if not gz.exists():
        req = urllib.request.Request(GLOVE_URL.format(d=dim), headers={"User-Agent": "vector-search-lab"})
        with urllib.request.urlopen(req, timeout=60) as r, open(str(gz) + ".part", "wb") as f:
            total, got = int(r.headers.get("Content-Length", 0)), 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                report(0.6 * got / total if total else 0.3, f"downloading GloVe {dim}d: {got / 1e6:.0f} MB")
        os.replace(str(gz) + ".part", gz)
    report(0.62, "converting to a fast cache (one time)")
    words, rows = [], []
    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as f:
        next(f)                                              # header: "<n_words> <dim>"
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            words.append(parts[0])
            rows.append(np.asarray(parts[1:], dtype=np.float32))
            if i % 20000 == 0:
                report(0.62 + 0.36 * i / max_words, f"converting: {i} words")
    np.save(npy, np.vstack(rows))
    vocab.write_text("\n".join(words), encoding="utf-8")
    gz.unlink(missing_ok=True)                               # keep only the small cache


def _glove_load(dim):
    if dim not in _GLOVE_CACHE:
        npy, vocab, _ = _glove_paths(dim)
        words = vocab.read_text(encoding="utf-8").split("\n")
        _GLOVE_CACHE[dim] = ({w: i for i, w in enumerate(words)}, np.load(npy))
    return _GLOVE_CACHE[dim]


class GloveEmbedder(Embedder):
    kind, needs_fit = "pretrained", True

    def __init__(self, dim=50):
        self.dim = dim
        self.id, self.label = f"glove-{dim}", f"GloVe {dim}d (averaged word vectors)"
        self.description = ("Pretrained GloVe word vectors (Wikipedia + Gigaword), averaged per text and weighted by IDF. "
                            "Knows that 'car' ~ 'automobile', but ignores word order and cannot read codes like E4012. "
                            "Downloads 69 MB from GitHub on first use.")
        self.idf = {}

    def status(self):
        if glove_cached(self.dim):
            return {"state": "ready", "note": "cached"}
        return {"state": "needs_download", "note": "69 MB download from GitHub (one time)" if self.dim == 50 else "download from GitHub (one time)"}

    def prepare(self, report=None):
        glove_prepare(self.dim, report)

    def fit(self, texts):
        glove_prepare(self.dim)
        docs = [set(tokenize(t)) for t in texts]
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        return self

    def encode(self, texts, role="doc", normalize=True):
        words, vecs = _glove_load(self.dim)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            tot = 0.0
            for w in tokenize(t):
                i = words.get(w)
                if i is not None:
                    wt = self.idf.get(w, 1.0)
                    out[r] += wt * vecs[i]
                    tot += wt
            if tot:
                out[r] /= tot
        return l2norm(out) if normalize else out


# ------------------------------------------------------------------ neural sentence-transformers (optional)
_ST_CACHE: dict = {}
ST_MODELS = [
    # id, label, hub name, query prefix, doc prefix, approx size, note
    ("st:all-MiniLM-L6-v2", "MiniLM-L6 (sentence-transformers)", "sentence-transformers/all-MiniLM-L6-v2", "", "", "90 MB", "small, fast English bi-encoder"),
    ("st:bge-small-en", "BGE-small-en v1.5", "BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", "", "130 MB", "strong small English retriever"),
    ("st:multilingual-e5-small", "multilingual-e5-small", "intfloat/multilingual-e5-small", "query: ", "passage: ", "470 MB", "100 languages including Bangla"),
]


def st_cached(hub: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(hub, "modules.json")
        return isinstance(res, str)
    except Exception:
        return False


class SentenceTransformerEmbedder(Embedder):
    kind = "neural"

    def __init__(self, spec):
        self.id, self.label, self.hub, self.qp, self.dp, size, note = spec
        self.description = f"Neural bi-encoder ({note}). Needs 'pip install sentence-transformers'; the {size} of weights download from Hugging Face on first use."
        self.dim = 0

    def status(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return {"state": "needs_install", "note": "pip install sentence-transformers"}
        if self.hub in _ST_CACHE:
            return {"state": "ready", "note": "loaded"}
        if st_cached(self.hub):
            return {"state": "ready", "note": "cached"}
        return {"state": "needs_download", "note": "downloads weights on first use"}

    def prepare(self, report=None):
        if self.hub not in _ST_CACHE:
            from sentence_transformers import SentenceTransformer
            (report or (lambda *a, **k: None))(0.2, f"loading {self.hub} (downloads on first use)")
            _ST_CACHE[self.hub] = SentenceTransformer(self.hub)
        self.dim = _ST_CACHE[self.hub].get_sentence_embedding_dimension()

    def encode(self, texts, role="doc", normalize=True):
        self.prepare()
        prefix = self.qp if role == "query" else self.dp
        x = _ST_CACHE[self.hub].encode([prefix + t for t in texts], normalize_embeddings=False, show_progress_bar=False, batch_size=32)
        x = np.asarray(x, dtype=np.float32)
        return l2norm(x) if normalize else x


# ------------------------------------------------------------------ registry
def _all():
    return [LSAEmbedder(), HashNgramEmbedder(), GloveEmbedder(50)] + [SentenceTransformerEmbedder(s) for s in ST_MODELS]


def create(model_id: str) -> Embedder:
    for m in _all():
        if m.id == model_id:
            return m
    raise KeyError(f"unknown model '{model_id}'")


def list_models() -> list:
    out = []
    for m in _all():
        st = m.status()
        out.append({"id": m.id, "label": m.label, "kind": m.kind, "description": m.description, "dim": m.dim or None,
                    "state": st["state"], "note": st["note"]})
    return out
