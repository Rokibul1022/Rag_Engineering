"""Exact (brute-force) dense search. For unit vectors, cosine similarity = dot product, so one matrix product scores everything."""
from __future__ import annotations

import numpy as np


class ExactIndex:
    """The 'flat' index: compares the query with EVERY vector. 100% recall, cost grows linearly with the corpus."""

    def __init__(self, vectors: np.ndarray):
        self.x = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, q: np.ndarray, k: int = 10):
        q = np.atleast_2d(q).astype(np.float32)
        s = q @ self.x.T
        k = min(k, s.shape[1])
        top = np.argpartition(-s, k - 1, axis=1)[:, :k]
        order = np.take_along_axis(s, top, axis=1).argsort(axis=1)[:, ::-1]
        ids = np.take_along_axis(top, order, axis=1)
        return np.take_along_axis(s, ids, axis=1), ids

    @property
    def nbytes(self):
        return int(self.x.nbytes)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of a with every row of b (does its own normalization)."""
    an = a / np.where(np.linalg.norm(a, axis=1, keepdims=True) == 0, 1, np.linalg.norm(a, axis=1, keepdims=True))
    bn = b / np.where(np.linalg.norm(b, axis=1, keepdims=True) == 0, 1, np.linalg.norm(b, axis=1, keepdims=True))
    return an @ bn.T
