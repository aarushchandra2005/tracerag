"""Start the TraceRAG web app.

    python run_app.py                 # http://127.0.0.1:8000, opens your browser
    python run_app.py --port 8080     # another port
    python run_app.py --no-browser    # do not open a browser tab
    python run_app.py --no-nli        # skip the 570 MB NLI model (cosine verifier only)
    python run_app.py --host 0.0.0.0  # reachable from other devices on your network

The first start downloads SciFact and builds the indexes (about a minute), then
the dense embeddings are computed in the background while the app is already
usable with tf-idf and BM25. Later starts take a few seconds.
"""
from __future__ import annotations

import argparse
import importlib.util
import socket
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from src.utils import setup_console  # noqa: E402


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host if host != "0.0.0.0" else "", port))
            return True
        except OSError:
            return False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-nli", action="store_true", help="do not load the NLI model")
    args = ap.parse_args(argv)
    setup_console()
    config.ensure_dirs()

    if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("uvicorn") is None:
        sys.exit("The web app needs FastAPI and uvicorn: pip install -r requirements.txt")
    import uvicorn
    if not port_free(args.host, args.port):
        sys.exit(f"Port {args.port} is already in use. Start with another one: python run_app.py --port {args.port + 1}")

    from src.api.engine import Engine
    from src.api.server import create_app

    print("TraceRAG: preparing data and indexes")
    engine = Engine(nli_enabled=not args.no_nli)
    try:
        engine.prepare()
    except Exception as e:  # noqa: BLE001 - show the reason, not a traceback
        sys.exit(f"Could not prepare SciFact: {e}")
    engine.start_background()
    app = create_app(engine)

    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{shown_host}:{args.port}"
    print(f"\nTraceRAG is running at {url}   (API docs at {url}/docs; press Ctrl+C to stop)")
    print("Dense retrieval and the NLI verifier finish loading in the background; the page shows their status.\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
