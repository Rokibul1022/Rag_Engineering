"""Start the lab:   python run.py      then open   http://localhost:8000"""
import argparse
import importlib
import sys
import threading
import webbrowser

REQUIRED = {"flask": "flask", "numpy": "numpy", "faiss": "faiss-cpu"}


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
    ap = argparse.ArgumentParser(description="Embeddings and vector search lab")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()
    check_dependencies()
    from retrieval import embedders
    from webapp.app import create_app

    for m in embedders.list_models():
        print(f"  model {m['id']:<26} {m['state']:<15} {m['note']}")
    print(f"Open: http://localhost:{args.port}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    create_app().run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
