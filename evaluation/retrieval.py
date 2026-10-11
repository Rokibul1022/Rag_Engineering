"""A tiny TF-IDF retriever, only to answer one question: 'if someone asks X, does the right chunk come back?'
It is a stand-in for your real embedding search, so you can judge chunking before you build the vector store."""
from __future__ import annotations

import math
import re
from collections import Counter

from ingestion.chunking import STOP

from .metrics import norm_ws


def _tok(t):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP]


class TfidfIndex:
    def __init__(self, texts):
        docs = [Counter(_tok(t)) for t in texts]
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts):
        v = {w: (1 + math.log(c)) * self.idf.get(w, 1.0) for w, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def search(self, query, k=3):
        q = self._vec(Counter(_tok(query)))
        scores = [(sum(x * v.get(w, 0) for w, x in q.items()), i) for i, v in enumerate(self.vecs)]
        return [i for s, i in sorted(scores, key=lambda x: (-x[0], x[1])) if s > 0][:k]


def contains(text, answers):
    t = norm_ws(text).lower()
    return any(norm_ws(a).lower() in t for a in answers)


def evaluate_retrieval(chunks, qa, k=3):
    """chunks: Chunk objects. qa: [{"q":..., "answers":[...]}]. Index = what you would embed (embed_text) of leaf chunks."""
    leaf = [c for c in chunks if c.kind != "parent"]
    parents = {c.chunk_id: c for c in chunks if c.kind == "parent"}
    if not leaf or not qa:
        return {"n": len(qa), "hit1": 0.0, "hit3": 0.0, "mrr": 0.0, "parent_hit3": 0.0, "answer_kept": 0.0, "details": []}
    idx = TfidfIndex([c.embed_text for c in leaf])
    h1 = h3 = ph3 = kept = 0
    mrr = 0.0
    details = []
    for item in qa:
        ans = item["answers"]
        top = idx.search(item["q"], k)
        rank = next((r for r, i in enumerate(top, 1) if contains(leaf[i].text, ans)), None)
        p_hit = any(contains(parents[leaf[i].parent_id].text, ans) if leaf[i].parent_id in parents else contains(leaf[i].text, ans) for i in top)
        preserved = any(contains(c.text, ans) for c in leaf)
        h1 += rank == 1
        h3 += rank is not None
        mrr += 1 / rank if rank else 0
        ph3 += p_hit
        kept += preserved
        details.append({"q": item["q"], "answer": ans[0], "rank": rank, "answer_in_any_chunk": preserved, "top_chunk": leaf[top[0]].text[:80] if top else ""})
    n = len(qa)
    return {"n": n, "hit1": round(h1 / n, 3), "hit3": round(h3 / n, 3), "mrr": round(mrr / n, 3), "parent_hit3": round(ph3 / n, 3),
            "answer_kept": round(kept / n, 3), "details": details}
