#!/usr/bin/env python3
"""Local demo server for the Jev demos.  Stdlib only.

    GET /                    homepage listing the demos
    GET /<slug>/             a demo's page
    GET /<slug>/static/*     that demo's assets
    GET /<slug>/api/*        that demo's own endpoints
    POST /<slug>/api/*       ditto, for demos that take one

Each demo lives in its own package with its own static/ and its own logic; the
only thing shared is the HTTP plumbing below.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from chat import routes as chat
from dim import routes as dim
from minecraft import routes as minecraft
from music import routes as music
from readout import routes as readout
from wiki import routes as wiki

DEMOS = [music, chat, dim, minecraft, readout, wiki]


class Request:
    """What a demo is handed: the query, and the ways it may reply."""

    def __init__(self, handler: BaseHTTPRequestHandler, query: dict) -> None:
        self._h = handler
        self.query = query
        self.server = handler.server

    def float_param(self, key: str, default: float) -> float:
        try:
            return max(0.0, float(self.query[key][0]))
        except (KeyError, IndexError, ValueError):
            return default

    def json_body(self) -> dict:
        length = int(self._h.headers.get("Content-Length", 0))
        return json.loads(self._h.rfile.read(length))

    def send(self, body: bytes, content_type: str, status: int = 200) -> None:
        h = self._h
        h.send_response(status)
        h.send_header("Content-Type", content_type)
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        h.wfile.write(body)

    def send_json(self, payload: object, status: int = 200) -> None:
        self.send(json.dumps(payload).encode(), "application/json; charset=utf-8", status)

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_json({"error": "not found"}, status=404)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self.send(path.read_bytes(), ctype)

    def redirect(self, location: str) -> None:
        h = self._h
        h.send_response(308)
        h.send_header("Location", location)
        h.send_header("Content-Length", "0")
        h.end_headers()

    def begin_sse(self) -> None:
        h = self._h
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream; charset=utf-8")
        h.send_header("Cache-Control", "no-store")
        h.send_header("Connection", "close")
        h.end_headers()
        h.close_connection = True

    def event(self, name: str, payload: object) -> None:
        self._h.wfile.write(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode())
        self._h.wfile.flush()


def _serve_under(req: Request, root: Path, name: str) -> None:
    target = (root / name).resolve()
    if not str(target).startswith(str(root.resolve())):
        req.send_json({"error": "not found"}, status=404)
        return
    req.send_file(target)


HOME_CSS = """
:root { color-scheme: dark; }
body {
  margin: 0; min-height: 100vh; background: #0c0e11; color: #e8ebf0;
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", sans-serif;
  -webkit-font-smoothing: antialiased;
  display: grid; place-items: center; padding: 48px 24px;
}
main { width: 100%; max-width: 420px; }
h1 {
  margin: 0 0 28px; font-size: 13px; font-weight: 500; letter-spacing: 0.18em;
  text-transform: uppercase; color: #5d6573;
}
ul { margin: 0; padding: 0; list-style: none; }
li { border-top: 1px solid #1b1f25; }
li:last-child { border-bottom: 1px solid #1b1f25; }
a {
  display: block; padding: 18px 2px; color: #e8ebf0; text-decoration: none;
  transition: color 200ms ease;
}
a:hover { color: #b69cff; }
a:focus-visible { outline: 1px solid #3a4250; outline-offset: 2px; }
"""


def _home(req: Request) -> None:
    items = "\n".join(
        f'    <li><a href="/{d.SLUG}/">{d.TITLE}</a></li>' for d in DEMOS
    )
    html = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Jev demos</title>\n<style>" + HOME_CSS + "</style>\n</head>\n"
        "<body>\n<main>\n  <h1>Jev demos</h1>\n  <ul>\n" + items +
        "\n  </ul>\n</main>\n</body>\n</html>\n"
    )
    req.send(html.encode(), "text/html; charset=utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "JevDemos/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:          # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        route = parsed.path
        req = Request(self, parse_qs(parsed.query))
        try:
            self._route(req, route)
        except BrokenPipeError:
            pass                        # browser navigated away mid-stream
        except ConnectionResetError:
            pass

    def do_POST(self) -> None:         # noqa: N802 (stdlib naming)
        parsed = urlparse(self.path)
        req = Request(self, parse_qs(parsed.query))
        try:
            self._route_post(req, parsed.path)
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass

    def _route_post(self, req: Request, route: str) -> None:
        # A demo opts in by defining handle_post, the way it does handle_api.
        for demo in DEMOS:
            prefix = f"/{demo.SLUG}/api/"
            if not route.startswith(prefix):
                continue
            handle = getattr(demo, "handle_post", None)
            if handle is not None and handle(req, route[len(prefix):]):
                return
            break
        req.send_json({"error": "not found", "path": route}, status=404)

    def _route(self, req: Request, route: str) -> None:
        if route in ("/", "/index.html"):
            _home(req)
            return
        for demo in DEMOS:
            prefix = f"/{demo.SLUG}"
            if route == prefix:
                # Assets are referenced relative to the page, so it needs the
                # trailing slash to sit inside its own directory.
                req.redirect(prefix + "/")
                return
            if not route.startswith(prefix + "/"):
                continue
            rest = route[len(prefix) + 1:]
            if rest in ("", "index.html"):
                req.send_file(demo.STATIC / "index.html")
            elif rest.startswith("static/"):
                _serve_under(req, demo.STATIC, rest[len("static/"):])
            elif rest.startswith("api/") and demo.handle_api(req, rest[len("api/"):]):
                pass
            else:
                req.send_json({"error": "not found", "path": route}, status=404)
            return
        req.send_json({"error": "not found", "path": route}, status=404)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--tempo", type=int, default=92)
    ap.add_argument(
        "--lookahead-bars", type=int, default=music.LOOKAHEAD_BARS,
        help="how many bars a stream stays ahead of playback",
    )
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.tempo = args.tempo
    httpd.lookahead_bars = args.lookahead_bars
    print(f"Jev demos on http://{args.host}:{args.port}/")
    for demo in DEMOS:
        print(f"  /{demo.SLUG}/  {demo.TITLE}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
