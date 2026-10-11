"""SQLite storage = the "source tracking" layer.

documents : one row per logical document (stable doc_id, current version)
versions  : one row per ingested version (file hash, text hash, parser, settings, timestamp)
chunks    : every chunk ever produced, with the version it appeared in and (if any) the version it disappeared in
aliases   : other file names that turned out to be byte-identical copies of an existing document
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(doc_id TEXT PRIMARY KEY, source_name TEXT UNIQUE NOT NULL, file_type TEXT, title TEXT,
  current_version INTEGER, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS versions(doc_id TEXT, version INTEGER, file_hash TEXT, text_hash TEXT, size_bytes INTEGER, stored_path TEXT,
  ingested_at TEXT, parser TEXT, strategy TEXT, params_json TEXT, n_blocks INTEGER, n_chunks INTEGER, status TEXT,
  PRIMARY KEY(doc_id, version));
CREATE TABLE IF NOT EXISTS chunks(doc_id TEXT, chunk_id TEXT, kind TEXT, parent_id TEXT, text TEXT, embed_text TEXT, tokens INTEGER,
  section_path TEXT, page_start INTEGER, page_end INTEGER, meta_json TEXT, version_added INTEGER, version_removed INTEGER,
  PRIMARY KEY(doc_id, chunk_id));
CREATE TABLE IF NOT EXISTS aliases(source_name TEXT PRIMARY KEY, canonical_doc_id TEXT, file_hash TEXT, first_seen TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def citation(source_name, page_start, page_end, path) -> str:
    pg = "" if page_start is None else (f", p.{page_start}" if page_start == page_end else f", pp.{page_start}-{page_end}")
    sec = f", §{' > '.join(path)}" if path else ""
    return f"{source_name}{pg}{sec}"


class Store:
    def __init__(self, data_dir="data"):
        self.dir = Path(data_dir)
        self.upload_dir = self.dir / "uploads"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "ingest.db"
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    # ------------------------------------------------------------ lookups
    def get_document(self, doc_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
            return dict(r) if r else None

    def get_version(self, doc_id, version):
        with self._conn() as c:
            r = c.execute("SELECT * FROM versions WHERE doc_id=? AND version=?", (doc_id, version)).fetchone()
            return dict(r) if r else None

    def find_duplicate_file(self, file_hash, exclude_doc_id):
        with self._conn() as c:
            r = c.execute("""SELECT d.* FROM documents d JOIN versions v ON v.doc_id=d.doc_id AND v.version=d.current_version
                             WHERE v.file_hash=? AND d.doc_id<>?""", (file_hash, exclude_doc_id)).fetchone()
            return dict(r) if r else None

    def other_chunks(self, doc_id):
        with self._conn() as c:
            rows = c.execute("""SELECT c.chunk_id, c.text, d.source_name FROM chunks c JOIN documents d ON d.doc_id=c.doc_id
                                WHERE c.doc_id<>? AND c.version_removed IS NULL AND c.kind<>'parent'""", (doc_id,)).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------ the key decision
    def decide(self, doc_id, file_hash, strategy, params):
        """What should happen to this upload?
        new | unchanged | updated (file bytes changed) | reprocessed (same file, new chunk settings) | duplicate (copy of another doc)"""
        existing = self.get_document(doc_id)
        if not existing:
            dup = self.find_duplicate_file(file_hash, doc_id)
            if dup:
                return {"status": "duplicate", "version": None, "duplicate_of": dup}
            return {"status": "new", "version": 1, "previous": None}
        cur = self.get_version(doc_id, existing["current_version"])
        if cur["file_hash"] == file_hash:
            if cur["strategy"] == strategy and json.loads(cur["params_json"]) == params:
                return {"status": "unchanged", "version": cur["version"], "previous": cur}
            return {"status": "reprocessed", "version": cur["version"] + 1, "previous": cur}
        return {"status": "updated", "version": cur["version"] + 1, "previous": cur}

    def save_upload(self, doc_id, version, filename, data: bytes) -> str:
        safe = re.sub(r"[^\w.\- ]", "_", Path(filename).name)
        d = self.upload_dir / doc_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"v{version}_{safe}"
        p.write_bytes(data)
        return str(p)

    # ------------------------------------------------------------ writes
    def add_alias(self, source_name, canonical_doc_id, file_hash):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO aliases VALUES(?,?,?,?)", (source_name, canonical_doc_id, file_hash, now()))

    def commit(self, *, doc_id, source_name, file_type, title, version, status, file_hash, text_hash, size, stored_path,
               parser, strategy, params, n_blocks, chunks):
        """Write document + version + chunk diff in ONE transaction. Returns the reindex diff."""
        with self._conn() as c:
            ts = now()
            if self.get_document(doc_id):
                c.execute("UPDATE documents SET current_version=?, updated_at=?, title=?, file_type=? WHERE doc_id=?",
                          (version, ts, title, file_type, doc_id))
            else:
                c.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?)", (doc_id, source_name, file_type, title, version, ts, ts))
            diff = self._apply_chunks(c, doc_id, version, chunks)
            n_active = c.execute("SELECT COUNT(*) FROM chunks WHERE doc_id=? AND version_removed IS NULL", (doc_id,)).fetchone()[0]
            c.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (doc_id, version, file_hash, text_hash, size, stored_path, ts, parser, strategy, json.dumps(params),
                       n_blocks, n_active, status))
        return diff

    @staticmethod
    def _apply_chunks(c, doc_id, version, chunks):
        """Incremental re-indexing: compare chunk IDs (content hashes) of the new version with the active ones."""
        active = {r["chunk_id"]: dict(r) for r in c.execute(
            "SELECT chunk_id, kind, parent_id FROM chunks WHERE doc_id=? AND version_removed IS NULL", (doc_id,))}
        new = {ch.chunk_id: ch for ch in chunks}
        added = [i for i in new if i not in active]
        removed = [i for i in active if i not in new]
        kept = [i for i in new if i in active]
        relinked = [i for i in kept if (active[i]["parent_id"] or None) != (new[i].parent_id or None)]
        for i in added:
            ch = new[i]
            c.execute("INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                      (doc_id, i, ch.kind, ch.parent_id, ch.text, ch.embed_text, ch.tokens, json.dumps(ch.section_path),
                       ch.page_start, ch.page_end, json.dumps({**ch.meta, "block_types": ch.block_types}), version))
        for i in removed:
            c.execute("UPDATE chunks SET version_removed=? WHERE doc_id=? AND chunk_id=?", (version, doc_id, i))
        for i in relinked:   # same text, different parent: update the link only, no re-embedding needed
            c.execute("UPDATE chunks SET parent_id=? WHERE doc_id=? AND chunk_id=?", (new[i].parent_id, doc_id, i))
        # Parents are stored for context expansion but are NOT embedded, so they never need (re-)embedding.
        embed_these = [i for i in added if new[i].kind != "parent"]
        delete_vectors = [i for i in removed if active[i]["kind"] != "parent"]
        return {"added": len(added), "removed": len(removed), "unchanged": len(kept), "relinked": len(relinked),
                "embed_these": embed_these, "delete_vectors_for": delete_vectors}

    # ------------------------------------------------------------ reads for the UI
    def list_documents(self):
        with self._conn() as c:
            rows = c.execute("""SELECT d.*, v.file_hash, v.n_chunks, v.parser, v.strategy,
                  (SELECT COUNT(*) FROM versions WHERE doc_id=d.doc_id) AS n_versions
                  FROM documents d JOIN versions v ON v.doc_id=d.doc_id AND v.version=d.current_version
                  ORDER BY d.updated_at DESC""").fetchall()
            aliases = [dict(r) for r in c.execute("SELECT * FROM aliases").fetchall()]
            return {"documents": [dict(r) for r in rows], "aliases": aliases}

    def document_detail(self, doc_id):
        doc = self.get_document(doc_id)
        if not doc:
            return None
        with self._conn() as c:
            versions = [dict(r) for r in c.execute("SELECT * FROM versions WHERE doc_id=? ORDER BY version DESC", (doc_id,))]
            rows = [dict(r) for r in c.execute("SELECT * FROM chunks WHERE doc_id=? ORDER BY rowid", (doc_id,))]
        for r in rows:
            r["section_path"] = json.loads(r["section_path"])
            r["meta"] = json.loads(r.pop("meta_json"))
            r["citation"] = citation(doc["source_name"], r["page_start"], r["page_end"], r["section_path"])
        return {"document": doc, "versions": versions, "chunks": rows}

    def reset(self):
        for p in self.dir.glob("ingest.db*"):
            p.unlink()
        shutil.rmtree(self.upload_dir, ignore_errors=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
