"""Tokenization shared by the sparse retrievers."""
from __future__ import annotations

import re

STOP = set("a an the and or but if of to in on at for with by from as is are was were be been it this that these those we our you "
           "your they their he she his her its not no so than then there which who whom what when where how why do does did has have "
           "had into over under about also can could should may might i my me us will would".split())


def tokenize(text: str, remove_stop: bool = True) -> list:
    """Lower-case word tokens. Codes like RT-3345 become 'rt' and '3345', and '3345' is what makes them findable."""
    toks = re.findall(r"\w+", (text or "").lower())
    return [t for t in toks if t not in STOP] if remove_stop else toks
