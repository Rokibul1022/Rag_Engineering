"""Small data classes shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Block:
    """One piece of document structure produced by a parser.

    type: heading | paragraph | list_item | table | footnote | header | footer | boilerplate | code
    A block with `removed_reason` set is kept for display/auditing but is NOT used downstream.
    """
    type: str
    text: str
    page: int | None = None
    level: int = 0                      # heading level (0 = document title, 1..n = sections)
    meta: dict = field(default_factory=dict)
    removed_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParseResult:
    blocks: list
    raw_text: str                       # "naive" extraction, i.e. what a one-liner would give you
    warnings: list = field(default_factory=list)
    doc_meta: dict = field(default_factory=dict)
    parser: str = ""


def warn(level: str, code: str, message: str, page: int | None = None) -> dict:
    """level: info | warning | error"""
    d = {"level": level, "code": code, "message": message}
    if page is not None:
        d["page"] = page
    return d


@dataclass
class Chunk:
    chunk_id: str
    kind: str                           # parent | child | flat
    text: str                           # what the user/LLM reads
    embed_text: str                     # what you would embed (section path + text)
    tokens: int
    section_path: list
    page_start: int | None
    page_end: int | None
    parent_id: str | None = None
    block_indices: list = field(default_factory=list)
    block_types: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
