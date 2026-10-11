# Architecture

A document ingestion pipeline with **source tracking**. A file goes in; clean, traceable, versioned chunks come out, ready to be embedded for a document Q&A (RAG) system.

## 1. Big picture

```
 upload (PDF / DOCX / HTML / TXT / image)
        │
        ▼
┌───────────────┐   file bytes ──► file_hash (SHA-256)
│ 0. Identify   │   source name ─► doc_id   (stable)
└──────┬────────┘
       ▼
┌───────────────┐  PDF  → pdfplumber (text layer, tables, font sizes) ── no text? → OCR fallback
│ 1. Parse      │  DOCX → python-docx (+ footnotes.xml)      Image → Tesseract OCR
│  parsers.py   │  HTML → BeautifulSoup                      TXT   → paragraphs
│  ocr.py       │  Output: typed Blocks + warnings + "naive" raw text (for comparison)
└──────┬────────┘
       ▼
┌───────────────┐  Unicode NFKC · de-hyphenate · strip invisible chars · straighten quotes
│ 2. Normalize  │  strip list markers · join wrapped lines · collapse whitespace · OCR digit fix
│  normalize.py │  Every rule is logged per block (before/after kept for auditing)
└──────┬────────┘
       ▼
┌───────────────┐  exact duplicate blocks removed; near-duplicate chunks reported
│ 3. Deduplicate│  text_hash (SHA-256 of normalized text)
│  dedup.py     │
└──────┬────────┘
       ▼
┌───────────────┐  structure (parent-child) │ fixed token windows │ semantic boundaries
│ 4. Chunk      │  content-addressed chunk IDs, section path, page range, token counts
│  chunking.py  │
└──────┬────────┘
       ▼
┌───────────────┐  decide: new / unchanged / updated / reprocessed / duplicate / failed
│ 5. Store      │  commit document + version + chunk diff in ONE SQLite transaction
│  store.py     │  → tells you exactly which chunks to embed and which vectors to delete
└──────┬────────┘
       ▼
 report (JSON) ──► web UI (webapp/) shows every stage
```

`ingestion/pipeline.py → run_pipeline()` wires the stages together and returns one JSON report. The web app is a thin layer over it (`webapp/app.py`, one HTML page with plain JavaScript, no build step).

## 2. Files

| File | Responsibility |
|---|---|
| `run.py` | Starts the server on port 8000 (checks dependencies and OCR first) |
| `ingestion/models.py` | `Block`, `ParseResult`, `Chunk`, `warn()` |
| `ingestion/parsers.py` | File-type detection (magic bytes), one parser per format, never raises |
| `ingestion/ocr.py` | Tesseract wrapper: preprocessing, per-block confidence, graceful fallback |
| `ingestion/normalize.py` | Ordered, named, logged text rules |
| `ingestion/dedup.py` | Exact-block and near-duplicate detection |
| `ingestion/chunking.py` | Three chunking strategies + statistics |
| `ingestion/store.py` | SQLite: documents, versions, chunks, aliases; incremental diff |
| `ingestion/pipeline.py` | Orchestration + the "live evidence" for the Engineering tab |
| `ingestion/concepts.py` | Teaching text shown in the UI |
| `webapp/` | Flask API + `static/index.html` |
| `samples/` | Demo files (messy PDF, scans, DOCX v1/v2, HTML, duplicate, broken) |
| `tests/` | 18 tests (`pytest -q`) |

## 3. The data model

**Block**: what a parser produces. `type` (heading, paragraph, list_item, table, footnote, header, footer, boilerplate, code), `text`, `page`, `level`, `meta`, `removed_reason`. A block with `removed_reason` stays visible for auditing but is excluded downstream.

**Chunk**: what you embed. `chunk_id`, `kind` (parent | child | flat), `text`, `embed_text` (section path + text), `tokens`, `section_path`, `page_start/end`, `parent_id`, `block_types`, `meta`.

**SQLite tables**

```
documents(doc_id PK, source_name UNIQUE, file_type, title, current_version, created_at, updated_at)
versions (doc_id, version, file_hash, text_hash, size_bytes, stored_path, ingested_at, parser,
          strategy, params_json, n_blocks, n_chunks, status)
chunks   (doc_id, chunk_id, kind, parent_id, text, embed_text, tokens, section_path, page_start,
          page_end, meta_json, version_added, version_removed)       -- history is kept
aliases  (source_name, canonical_doc_id, file_hash, first_seen)      -- byte-identical copies
```

## 4. Document engineering: how each problem is handled

### Parsing failures
Every parser returns `ParseResult(blocks, raw_text, warnings, doc_meta)` and **never raises**. Handled cases: corrupt/truncated PDF, password-protected PDF, wrong file extension (type detected from magic bytes), bad text encoding (UTF-8 → Windows-1252 → Latin-1 with a warning), unsupported formats (`.doc`), PDF fonts without a Unicode map (`(cid:127)` repaired for bullets, others dropped with a warning), per-page failures. If no usable text comes out, status is `failed`: nothing is indexed and any earlier good version stays active.

### OCR errors
OCR only runs when there is no text layer (scanned PDF pages, images). Steps: grayscale → upscale small images → autocontrast → Tesseract with the selected language. Confidence is kept per block (`meta.confidence`, `low_conf_words`) and surfaced as warnings. Safe corrections only: letters that look like digits inside numbers (`1O25 → 1025`, needs ≥ 2 digits). Blocks that are mostly symbols are dropped as OCR noise. No blind spell-checking.

### Tables
Detected separately (`pdfplumber.find_tables`, DOCX `Table`, HTML `<table>`), cut out of the text flow, stored as Markdown plus structured rows. DOCX merged cells are detected across rows and columns. In chunking a table is atomic; if it exceeds `max_tokens` it is split **by rows with the header row repeated**.

### Headers, footers, page numbers
For each PDF page only the top and bottom 9% margins are examined. Digits are masked (`Page 3 of 9 → Page # of #`); a line repeated on at least half the pages is a running header/footer. Standalone page numbers are removed. DOCX section headers/footers and HTML `<nav>`, `<footer>`, cookie banners are treated the same way. Removed blocks remain visible, struck through, in the UI.

### Footnotes
PDF: lines in the lower part of the page with a font size clearly below the body size. DOCX: `word/footnotes.xml`. HTML: elements marked footnote/endnote; `<sup>1</sup>` becomes `[1]`. Footnotes become their own child chunk under the same parent section, so they are searchable without breaking the main text.

### Duplicate content
1. **File level**: same SHA-256 under a different name → recorded in `aliases`, not indexed again.
2. **Block level**: identical blocks (≥ 40 characters) inside one document → later copies removed.
3. **Near-duplicate chunks**: 5-word shingles + Jaccard ≥ 0.8, within the document and against other documents → reported.

### Stable document IDs
`doc_id = "doc_" + sha1(lower(source_name))[:10]`. The ID identifies the *source*, not the content, so edits keep the ID. Use a URL or DMS id as the source name in production. `chunk_id = "c_" + sha1(doc_id | kind | section_path | normalized text)[:12]`: unchanged text, same ID, in every version.

### Versioning
Two hashes: `file_hash` (bytes) and `text_hash` (normalized text). Decision table, from `Store.decide()`:

| Situation | Status | Version |
|---|---|---|
| source never seen, bytes never seen | `new` | 1 |
| source never seen, bytes match another document | `duplicate` | — |
| same source, same bytes, same settings | `unchanged` | current |
| same source, same bytes, different chunk settings | `reprocessed` | +1 |
| same source, different bytes | `updated` | +1 |
| nothing usable parsed | `failed` | — |

Old versions are never deleted: chunks carry `version_added` / `version_removed`.

### Metadata
Document: `doc_id`, source name, file type, parser, version, hashes, title/author/pages when present. Chunk: `chunk_id`, `parent_id`, `section_path`, page range, tokens, block types, overlap, plus a ready citation like `handbook.docx, p.2, §Employee Handbook > Leave Policy`.

### Incremental re-indexing
`Store._apply_chunks()` compares the new chunk IDs with the active ones:
- **added** → embed + insert
- **removed** → delete vectors; the row is kept as history (`version_removed`)
- **unchanged** → nothing
- **re-linked** → same text, new parent → update `parent_id` only

Parents are stored for context but never embedded, so they never appear in `embed_these`.

### Chunking
| Strategy | How it cuts | Strength | Weakness |
|---|---|---|---|
| **structure** (default) | Sections from heading hierarchy; paragraphs, lists, tables kept whole; oversize paragraphs split at sentences; sentence overlap | Keeps ideas together; gives parent-child | Needs headings |
| **fixed** | Every N tokens with overlap | Simple, predictable size | Cuts mid-sentence and mid-table |
| **semantic** | New chunk at the weakest-similarity gap between sentence windows (word-overlap cosine) | Follows topic changes | Cheap similarity is crude; upgrade to embeddings |

**Parent-child**: the parent is the whole section (split into several parents if over 800 tokens); children are ≤ `max_tokens`. Search the children, hand the LLM the parent. `embed_text` prefixes the heading path so a chunk like "It grew 12%" keeps its topic.

Token counting is a regex (`\w+|[^\w\s]`), deliberately dependency-free. Real tokenizers count roughly 20–40% more; swap `count_tokens()` for `tiktoken` or your embedding model's tokenizer when you need exact budgets.

## 5. Extending toward the evaluated Q&A system

1. After `Store.commit()`, embed `diff["embed_these"]` and delete `diff["delete_vectors_for"]` in your vector DB, using `chunk_id` as the vector ID and the chunk metadata as filters.
2. At query time: retrieve children → expand to `parent_id` text → answer with the `citation` string.
3. Evaluate retrieval with a small gold set (question → expected chunk/page). The `source`, `page` and `section_path` metadata make hit-rate and citation checks easy.
4. Replace `semantic_chunks`' word-overlap similarity with embeddings.

## 6. Known limitations

- De-hyphenation joins when the next part is lowercase: `well-⏎known` becomes `wellknown`.
- Header/footer detection needs ≥ 2 pages; single-page PDFs only lose standalone page numbers.
- PDF heading detection relies on font size and boldness; PDFs with uniform fonts yield few headings.
- Multi-column PDFs are read in `pdfplumber` line order and may interleave columns.
- OCR quality depends on the scan and on the installed Tesseract language packs.
- Near-duplicate search is O(n²); fine for hundreds of chunks, use MinHash/LSH at scale.
