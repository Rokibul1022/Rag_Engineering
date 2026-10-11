"""Teaching content for the 'Document engineering' tab. `id` matches a key in the pipeline's live `evidence` dict,
so each explanation is shown next to what actually happened to the document you just uploaded."""

CONCEPTS = [
    # ------------------------------------------------------------------ parsing
    {"group": "Parsing & extraction", "id": "parsing_failures", "title": "Parsing failures",
     "text": "Real-world files break in predictable ways. A production pipeline must never crash on them - it must detect, report and move on.\n"
             "- Corrupt / truncated file → error warning, document is quarantined (not indexed). A previous good version stays active.\n"
             "- Password-protected PDF → clear error asking for an unlocked copy.\n"
             "- Wrong extension (a .docx renamed .pdf) → we detect the type from the first bytes, not the name.\n"
             "- Bad encoding in text/HTML → fall back to Windows-1252 / Latin-1 and warn.\n"
             "- PDF fonts without a Unicode map (shows as (cid:127)) → repair known bullets, drop the rest, warn.\n"
             "- Unsupported formats (.doc, .xlsx...) → explicit message instead of silent garbage.\n"
             "Rule of thumb: every parser returns warnings; the pipeline decides what to do with them.",
     "code": "ingestion/parsers.py → parse_file(), detect_type(); status 'failed' in pipeline.py"},
    {"group": "Parsing & extraction", "id": "ocr", "title": "OCR errors",
     "text": "OCR is only used when there is no text layer (scanned PDF pages, photos). It is the noisiest input you will ingest.\n"
             "- Preprocess the image first (grayscale, upscale small images, boost contrast) - this alone often fixes many errors.\n"
             "- Keep the engine's confidence per word/block in metadata. Low confidence is a signal you can filter or down-weight at retrieval time.\n"
             "- Fix only what is safe: letters that look like digits inside numbers (1O25 → 1025). Never 'spell-correct' blindly - it corrupts names and codes.\n"
             "- Use the right language pack (e.g. 'ben' for Bangla) - a wrong language pack is the #1 cause of garbage output.\n"
             "- Drop blocks that are mostly symbols (OCR noise from smudges and borders).",
     "code": "ingestion/ocr.py; OCR fallback in parsers.parse_pdf(); r_ocr_digits in normalize.py"},
    {"group": "Parsing & extraction", "id": "tables", "title": "Tables",
     "text": "Naive text extraction reads a table row by row, cell by cell, and the meaning (which number belongs to which column) is lost - python-docx's paragraph loop even skips tables entirely.\n"
             "- Detect tables separately from body text and remove their area from the text flow.\n"
             "- Convert to Markdown, which LLMs understand well.\n"
             "- Treat a table as one atomic unit when chunking: never cut in the middle of a row.\n"
             "- If a table is bigger than the chunk limit, split by rows and repeat the header row in every part.\n"
             "- Merged cells need special handling (python-docx repeats them once per grid column).",
     "code": "parsers.py (find_tables / docx Table / HTML <table>); chunking._split_table()"},
    {"group": "Parsing & extraction", "id": "headers_footers", "title": "Headers, footers & page numbers",
     "text": "Running headers and footers ('Confidential - Page 3 of 9') repeat on every page. If left in, they pollute chunks, dilute embeddings, and make unrelated chunks look similar.\n"
             "- Look only in the top/bottom margin of each page.\n"
             "- Mask digits (Page 3 of 9 → Page # of #) and count how many pages share the same line. Repeated on ≥50% of pages → header/footer.\n"
             "- Standalone page numbers are removed too.\n"
             "- Removed blocks are kept (greyed out in the Parsed structure tab) so you can audit what was dropped.\n"
             "- HTML equivalent: <nav>, <footer>, cookie banners, sidebars.",
     "code": "parsers.parse_pdf() (repeated set), parse_html() (is_boiler)"},
    {"group": "Parsing & extraction", "id": "footnotes", "title": "Footnotes",
     "text": "A footnote is small text at the bottom of a page that belongs to a sentence somewhere above. Mixed into body text it breaks sentences mid-way.\n"
             "- PDF: lines in the lower part of the page whose font size is clearly smaller than the body text.\n"
             "- DOCX: footnotes live in a separate XML part (word/footnotes.xml), read directly.\n"
             "- HTML: elements marked footnote/endnote, and <sup>1</sup> markers are rewritten to [1] so they do not glue onto words.\n"
             "- Footnotes are stored as their own chunk, linked to the same parent section, so they are searchable without breaking the main text.",
     "code": "parsers.py (classify 'footnote'), chunking.structure_chunks()"},
    {"group": "Parsing & extraction", "id": "duplicates", "title": "Duplicate content",
     "text": "Duplicates waste storage, skew retrieval (the same passage fills the top-k) and make citations confusing. We check three levels:\n"
             "- Duplicate FILE: identical bytes (file hash) under a different name → registered as an alias, not indexed twice.\n"
             "- Duplicate BLOCK: identical paragraph repeated inside one document (boilerplate, disclaimers) → later copies removed.\n"
             "- NEAR-duplicate chunk: 5-word shingles + Jaccard similarity ≥ 0.8, within the document and across other documents → reported (you decide whether to drop).\n"
             "Short blocks (<40 chars) are never treated as duplicates - 'Yes' or 'N/A' are legitimately repeated.",
     "code": "ingestion/dedup.py; store.decide() for file level"},
    # ------------------------------------------------------------------ identity
    {"group": "Identity, versioning & metadata", "id": "doc_id", "title": "Stable document IDs",
     "text": "An ID must identify the document, not its content, otherwise every edit looks like a brand-new document.\n"
             "- doc_id = hash of the source name (here: the file name or the name you type in 'Source name'). Same source → same ID forever.\n"
             "- Not a random UUID (re-uploading would create duplicates) and not a content hash (changes with every edit).\n"
             "- In production use something that never changes: a URL, a document-management ID, a relative path.\n"
             "- Chunk ID = hash(doc_id + section path + text). Unchanged text → unchanged ID across versions. This is what makes incremental re-indexing cheap.",
     "code": "pipeline.make_doc_id(), chunking.make_chunk_id()"},
    {"group": "Identity, versioning & metadata", "id": "versioning", "title": "Versioning",
     "text": "Two hashes tell you what changed:\n"
             "- file_hash (SHA-256 of the bytes): did the file change at all?\n"
             "- text_hash (SHA-256 of the normalized text): did the content change? Different bytes but same text = cosmetic change (re-saved, new metadata).\n"
             "Statuses: new → first time seen · unchanged → same file and settings, skip · updated → new bytes, version+1 · reprocessed → same file but different chunk settings, version+1 · duplicate → copy of another document · failed → parser error, nothing indexed.\n"
             "Old versions are never deleted: chunks get version_added / version_removed so you can answer 'what did the document say in v1?'.",
     "code": "store.decide(), store.commit()"},
    {"group": "Identity, versioning & metadata", "id": "metadata", "title": "Metadata & source tracking",
     "text": "Every chunk must be traceable back to its origin, otherwise you cannot cite sources or debug bad answers.\n"
             "- Document level: doc_id, source name, file type, parser used, version, file hash, title/author when present.\n"
             "- Chunk level: chunk_id, parent_id, section path (heading breadcrumb), page range, token count, block types, overlap.\n"
             "- A ready-made citation string such as 'handbook.docx, p.2, §Leave Policy > Annual Leave'.\n"
             "- Metadata is also a filter at query time (e.g. only version 2, only PDFs, only this section).",
     "code": "models.Chunk, store.py (documents/versions/chunks tables), store.citation()"},
    {"group": "Identity, versioning & metadata", "id": "reindex", "title": "Incremental re-indexing",
     "text": "Re-embedding a whole corpus on every edit is slow and expensive. Instead, compare chunk IDs of the new version with the active ones:\n"
             "- added → embed and insert\n"
             "- removed → delete their vectors (the rows stay in SQLite as history)\n"
             "- unchanged → do nothing\n"
             "- re-linked → same text but its parent changed: update the link only, no re-embedding\n"
             "Upload Handbook v1, then Handbook v2 (both use source name 'handbook.docx') to watch this happen. Parents are stored for context but never embedded.",
     "code": "store._apply_chunks()"},
    # ------------------------------------------------------------------ chunking
    {"group": "Chunking", "id": "chunk_structure", "title": "Chunking by structure",
     "text": "Documents have natural boundaries - headings, paragraphs, lists, tables. Respecting them keeps ideas together.\n"
             "- Group blocks under their heading path (breadcrumb).\n"
             "- Never let a chunk cross a heading.\n"
             "- Keep lists and tables whole; split oversized paragraphs at sentence boundaries.\n"
             "- Prefix what you embed with the breadcrumb ('Report > Finance > Revenue') so a chunk like 'It grew 12%' still carries its topic.\n"
             "Weakness: documents without headings (scans, plain text) give one big section - fall back to token or semantic chunking.",
     "code": "chunking.structure_chunks(), build_sections()"},
    {"group": "Chunking", "id": "chunk_tokens", "title": "Chunking by token count",
     "text": "Embedding models and LLM prompts have size limits, and very large chunks blur many topics into one vector.\n"
             "- max_tokens is a hard ceiling; typical values are 200-500 tokens.\n"
             "- overlap repeats the last sentence(s) of the previous chunk so context is not lost at the cut.\n"
             "- Pure fixed-size windows are the simplest baseline but they cut mid-sentence and mid-table (see the comparison table).\n"
             "- Too-small trailing chunks are merged into their neighbour.\n"
             "Here 'tokens' = words + punctuation counted by a regex. A real tokenizer (tiktoken, your embedding model's tokenizer) counts ~20-40% higher.",
     "code": "chunking.count_tokens(), fixed_chunks(), _pack()"},
    {"group": "Chunking", "id": "chunk_semantic", "title": "Chunking by semantic boundaries",
     "text": "Cut where the topic changes, not where a counter runs out.\n"
             "- Split text into sentences, compare each sentence window with the next (word-overlap cosine here).\n"
             "- Start a new chunk at the lowest-similarity gaps (bottom 25%), still respecting max_tokens.\n"
             "- Production upgrade: replace the word-count vectors with sentence embeddings. Nothing else in the code changes.\n"
             "Compare all three strategies in the table on the Chunks tab: the same document produces different chunk counts, sizes and mid-sentence cuts.",
     "code": "chunking.semantic_chunks()"},
    {"group": "Chunking", "id": "parent_child", "title": "Parent-child relationships",
     "text": "Small chunks retrieve precisely; large chunks answer well. Parent-child gives you both.\n"
             "- Child = small chunk, embedded and searched.\n"
             "- Parent = the whole section the child came from, stored but not embedded.\n"
             "- At query time: search children → collect their parent_ids → give the LLM the parent text (or child + neighbours).\n"
             "- Very long sections are split into several parents so the parent stays prompt-sized.",
     "code": "chunking.structure_chunks() (parent_id), store.chunks.parent_id"},
    # ------------------------------------------------------------------ normalization
    {"group": "Normalization", "id": "normalization", "title": "How normalization works",
     "text": "Normalization makes text from different sources look the same, so equal content gets equal hashes and embeddings are not distorted by layout noise. Rules run in a fixed order and each one is logged:\n"
             "- unicode_nfkc: ligatures ﬁ→fi, full-width characters, compatibility forms\n"
             "- dehyphenate: infra-⏎structure → infrastructure\n"
             "- strip_invisible: control characters, zero-width spaces, soft hyphens (ZWJ/ZWNJ are kept - Bangla needs them)\n"
             "- straighten_punctuation: curly quotes → straight, NBSP → space\n"
             "- strip_list_markers, join_wrapped_lines, collapse_whitespace\n"
             "- ocr_digit_fix (OCR text only)\n"
             "We do NOT lowercase or strip accents - that would destroy meaning (names, codes, Bangla). Lowercasing is only used internally for duplicate detection.\n"
             "Limitation: dehyphenation cannot tell 'well-⏎known' from 'exam-⏎ple'; it joins when the next part is lowercase.",
     "code": "ingestion/normalize.py → RULES list"},
]
