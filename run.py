"""Start the demo:   python run.py      then open   http://localhost:8000"""
import argparse
import importlib
import sys
import threading
import webbrowser

REQUIRED = {"flask": "flask", "pdfplumber": "pdfplumber", "docx": "python-docx", "bs4": "beautifulsoup4",
            "PIL": "pillow", "pytesseract": "pytesseract"}


def check_dependencies():
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing Python packages:", ", ".join(missing))
        print("Install everything with:\n    pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Document ingestion pipeline demo")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()
    check_dependencies()

    from ingestion.ocr import ocr_status
    from webapp.app import create_app

    ocr = ocr_status()
    print("OCR :", "ready, languages: " + ", ".join(ocr["languages"]) if ocr["available"] else "NOT available - " + ocr["reason"])
    print(f"Open: http://localhost:{args.port}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    create_app().run(host="127.0.0.1", port=args.port, debug=False)
