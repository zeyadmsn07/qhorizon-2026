"""Rebuild the dashboard data and open it in your browser.

    python dashboard/serve.py            # http://localhost:8000
    python dashboard/serve.py --port 9000 --no-browser
"""
import argparse
import functools
import http.server
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    subprocess.run([sys.executable, str(HERE / "build_data.py")], check=True)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(HERE))
    url = f"http://localhost:{a.port}/index.html"
    print(f"Serving {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    with http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()