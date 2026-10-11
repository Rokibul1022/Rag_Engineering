# Evaluation report

Settings: max_tokens=200, overlap=30

- Gold checks passed: **47/47**
- Mean integrity score: **98/100**, failed integrity checks: **0**

## Chunk quality by strategy (retrieval with a simple TF-IDF search)

| Strategy | Questions | Answer kept in a chunk | Hit@1 | Hit@3 | MRR | Hit@3 via parent |
|---|---|---|---|---|---|---|
| Structure-aware (parent-child) | 16 | 100% | 94% | 100% | 0.97 | 100% |
| Fixed token windows | 16 | 100% | 94% | 100% | 0.97 | 100% |
| Semantic boundaries | 16 | 100% | 88% | 100% | 0.94 | 100% |

## Per sample

### messy_report.pdf
- PASS ingestion status: expected new, got new
- PASS headings found: precision 100%, recall 100%
- PASS tables detected: expected 1, got 1
- PASS footnotes detected: expected 3, got 3
- PASS headers/footers/boilerplate removed: expected 6, got 6
- PASS duplicate blocks removed: expected 1, got 1
- PASS expected text present: 5/5 found
- PASS unwanted text removed: none leaked
- metrics: heading_precision=1.0, heading_recall=1.0, heading_f1=1.0, must_contain_recall=1.0, leaks=0

### scan_note.png
- PASS ingestion status: expected new, got new
- PASS headings found: precision 100%, recall 100%
- PASS tables detected: expected 0, got 0
- PASS footnotes detected: expected 0, got 0
- PASS headers/footers/boilerplate removed: expected 0, got 0
- PASS expected text present: 2/2 found
- PASS OCR accuracy: character error rate 0.0%, word error rate 0.0% (limit 3% CER)
- metrics: heading_precision=1.0, heading_recall=1.0, heading_f1=1.0, must_contain_recall=1.0, cer=0.0, wer=0.0

### scanned_report.pdf
- PASS ingestion status: expected new, got new
- PASS headings found: precision 100%, recall 100%
- PASS tables detected: expected 0, got 0
- PASS footnotes detected: expected 0, got 0
- PASS headers/footers/boilerplate removed: expected 0, got 0
- PASS expected text present: 2/2 found
- PASS OCR accuracy: character error rate 0.0%, word error rate 0.0% (limit 3% CER)
- metrics: heading_precision=1.0, heading_recall=1.0, heading_f1=1.0, must_contain_recall=1.0, cer=0.0, wer=0.0

### faq_page.html
- PASS ingestion status: expected new, got new
- PASS headings found: precision 100%, recall 100%
- PASS tables detected: expected 1, got 1
- PASS footnotes detected: expected 2, got 2
- PASS headers/footers/boilerplate removed: expected 3, got 3
- PASS expected text present: 3/3 found
- PASS unwanted text removed: none leaked
- metrics: heading_precision=1.0, heading_recall=1.0, heading_f1=1.0, must_contain_recall=1.0, leaks=0

### handbook_v1.docx
- PASS ingestion status: expected new, got new
- PASS headings found: precision 100%, recall 100%
- PASS tables detected: expected 1, got 1
- PASS footnotes detected: expected 2, got 2
- PASS headers/footers/boilerplate removed: expected 2, got 2
- PASS expected text present: 3/3 found
- PASS unwanted text removed: none leaked
- metrics: heading_precision=1.0, heading_recall=1.0, heading_f1=1.0, must_contain_recall=1.0, leaks=0

### handbook_v2.docx
- PASS ingestion status: expected updated, got updated
- PASS incremental re-index: 3 unchanged, re-embed 67% of chunks (limit 80%)
- PASS headings found: precision 100%, recall 100%
- PASS tables detected: expected 1, got 1
- PASS footnotes detected: expected 2, got 2
- PASS headers/footers/boilerplate removed: expected 2, got 2
- PASS expected text present: 2/2 found
- PASS unwanted text removed: none leaked
- metrics: reembed_fraction=0.667, heading_precision=1.0, heading_recall=1.0, heading_f1=1.0, must_contain_recall=1.0, leaks=0

### copy_of_messy_report.pdf
- PASS ingestion status: expected duplicate, got duplicate

### broken.pdf
- PASS ingestion status: expected failed, got failed

### (re-upload of messy_report.pdf)
- PASS ingestion status: expected unchanged, got unchanged
