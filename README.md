# Document ingestion pipeline (with source tracking)

Upload a PDF, Word file, web page, text file or an image of text, and watch it go through:
**raw text → parsing → normalization → duplicate removal → chunking → versioned storage.**
Part of the *Evaluated document Q&A system* (RAG) course path. See [`architecture.md`](architecture.md) for how it works.

## Run it

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Then open **http://localhost:8000** (it also opens automatically).

### OCR (scanned PDFs and images)
OCR needs the free **Tesseract** program in addition to the Python package:
- Windows: install from https://github.com/UB-Mannheim/tesseract/wiki (tick the *Bengali* language data if you need Bangla)
- Ubuntu/Debian: `sudo apt install tesseract-ocr tesseract-ocr-ben`
- macOS: `brew install tesseract tesseract-lang`

Without Tesseract everything else works; scanned files just report a clear `OCR_UNAVAILABLE` message.

## Try it (5 minutes)
1. Pick **Messy PDF** in *Try a sample* → open every numbered tab. Compare *Raw text* with *Normalized text*.
2. Run **Handbook v1**, then **Handbook v2** → *Source & versioning* shows the incremental re-index plan. Open *Library* to see retired chunks.
3. Run **Identical copy** → detected as a duplicate. Run **Truncated PDF** → a parsing failure that does not crash anything.
4. Run **Scanned PDF** or **Image of text** → OCR with confidence scores.
5. Change *Chunking strategy* and *Max tokens*, process the same file again, compare the table on the *Chunks* tab.
6. Read the *Document engineering* tab: every concept is explained next to what happened to your document.

## Evaluate
- **In the browser:** open the **6 Evaluate** tab. Part A shows integrity checks for the document you just processed; part B runs a benchmark on the samples against `evaluation/gold.json`.
- **In the terminal:**
  ```bash
  python evaluate.py                 # benchmark on the samples, writes evaluation/report.md
  python evaluate.py --file my.pdf   # integrity checks on any document (no answer key needed)
  python evaluate.py --gold my.json  # benchmark with your own answer key
  ```

## Tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Tools used (all beginner friendly)
Flask (web server) · pdfplumber (PDF) · python-docx (Word) · BeautifulSoup (HTML) · Pillow + pytesseract (images/OCR) · SQLite (built into Python). No databases to install, no API keys, no GPU.

## Layout
```
run.py                  start here
architecture.md         how the pipeline is designed
ingestion/              the pipeline (parsers, normalize, dedup, chunking, store, pipeline)
webapp/                 Flask API + single-page UI
samples/                demo files (regenerate with samples/make_samples.py)
tests/                  pytest suite
data/                   created at runtime (SQLite database + saved uploads), git-ignored
```
