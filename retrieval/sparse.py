"""Sparse (lexical) retrieval, written from scratch so you can read every line.

BM25 and TF-IDF both score a document by the words it SHARES with the query. If no word is shared, the score is 0:
that is the 'vocabulary mismatch' failure mode that dense embeddings are meant to fix.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .text import tokenize


class _Postings:
    def __init__(self, docs):
        self.n = len(docs)
        self.doc_tokens = [tokenize(d) for d in docs]
        self.lengths = np.array([len(t) for t in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0
        self.post = defaultdict(list)                       # term -> [(doc_index, term_frequency)]
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.post[term].append((i, tf))
        self.df = {t: len(p) for t, p in self.post.items()}

    def idf(self, term):                                    # BM25's non-negative idf
        return math.log(1 + (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5))


class BM25:
    """Okapi BM25.  score = sum over query terms of idf * tf*(k1+1) / (tf + k1*(1 - b + b*len/avglen))"""
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.p = _Postings(docs)
        return self

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.p.n, dtype=np.float32)
        for term in set(tokenize(query)):
            if term not in self.p.post:
                continue                                    # an unknown (or misspelled) word contributes nothing
            idf = self.p.idf(term)
            for i, tf in self.p.post[term]:
                norm = self.k1 * (1 - self.b + self.b * self.p.lengths[i] / self.p.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def vocabulary(self):
        return set(self.p.post)

    def index_bytes(self):
        return sum(len(t) + 8 * len(v) for t, v in self.p.post.items())


class TfidfCosine:
    """TF-IDF vectors compared with cosine similarity (sublinear tf, smoothed idf)."""
    name = "tfidf"

    def fit(self, docs):
        self.p = _Postings(docs)
        self.idf = {t: math.log((self.p.n + 1) / (df + 1)) + 1 for t, df in self.p.df.items()}
        norms = np.zeros(self.p.n, dtype=np.float64)
        self.w = defaultdict(list)                          # term -> [(doc_index, weight)]
        for term, plist in self.p.post.items():
            for i, tf in plist:
                wt = (1 + math.log(tf)) * self.idf[term]
                self.w[term].append((i, wt))
                norms[i] += wt * wt
        self.norms = np.sqrt(norms)
        self.norms[self.norms == 0] = 1.0
        return self

    def scores(self, query: str) -> np.ndarray:
        q = {t: (1 + math.log(c)) * self.idf[t] for t, c in Counter(tokenize(query)).items() if t in self.idf}
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0
        out = np.zeros(self.p.n, dtype=np.float32)
        for term, qw in q.items():
            for i, dw in self.w[term]:
                out[i] += qw * dw
        return out / (self.norms.astype(np.float32) * qn)

    def index_bytes(self):
        return sum(len(t) + 8 * len(v) for t, v in self.w.items())
