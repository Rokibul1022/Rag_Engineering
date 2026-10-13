"""Hybrid retrieval: combine a sparse and a dense ranking.

RRF (reciprocal rank fusion) only uses RANKS, so it needs no score calibration:  score(d) = sum 1/(60 + rank_i(d)).
Linear fusion min-max scales each score list to [0,1] and mixes them with a weight alpha.
"""
from __future__ import annotations

import numpy as np


def ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """rank[i] = position (1 = best) of document i."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def rrf(score_lists, k: int = 60, weights=None) -> np.ndarray:
    weights = weights or [1.0] * len(score_lists)
    return sum(w / (k + ranks_from_scores(s)) for w, s in zip(weights, score_lists)).astype(np.float32)


def minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def linear(sparse_scores, dense_scores, alpha: float = 0.5) -> np.ndarray:
    return (alpha * minmax(dense_scores) + (1 - alpha) * minmax(sparse_scores)).astype(np.float32)
