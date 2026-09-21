"""The readout demo's endpoints.

    GET /readout/api/pages    the pages, in carousel order
    GET /readout/api/read     Server-Sent Events: one judgment per question
"""

from __future__ import annotations

import hashlib
import html
import inspect
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from .questions import read

load_dotenv()

SLUG = "readout"
TITLE = "Judging landing pages"
STATIC = Path(__file__).resolve().parent / "static"

MODEL = "jev-latest"

_HERE = Path(__file__).resolve().parent
_CACHE_PATH = _HERE / "cache.json"

QUESTIONS = [
    "usable_today", "has_price", "names_customer", "says_what_it_does",
    "readiness", "shipped_vs_roadmap", "concreteness", "category",
    "omission", "coyness",
]


def _load_pages() -> List[Dict[str, str]]:
    pages = []
    for path in sorted((_HERE / "pages").glob("*.txt")):
        name, _, body = path.read_text(encoding="utf-8").partition("\n\n")
        pages.append({
            "id": path.stem,
            "name": name.strip(),
            "text": body.strip(),
        })
    return pages


PAGES = _load_pages()
BY_ID = {p["id"]: p for p in PAGES}

RENDER = STATIC / "render"
LIVE_SITE = {"09": "https://typesafe.ai/"}

_DROP = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def _flatten(markup: str) -> str:
    text = html.unescape(_TAG.sub(" ", _DROP.sub(" ", markup)))
    return " ".join(text.split())


def _verify_renders() -> None:
    """Every judged line must appear verbatim in what is displayed."""
    for page in PAGES:
        if page["id"] in LIVE_SITE:
            continue
        path = RENDER / f"{page['id']}.html"
        assert path.is_file(), path
        flat = _flatten(path.read_text(encoding="utf-8"))
        for line in page["text"].splitlines():
            line = " ".join(line.split())
            if line and line not in flat:
                raise AssertionError(
                    f"{path.name} does not show: {line!r}")


_verify_renders()

_CACHE_LOCK = threading.Lock()
_CACHE: Dict[str, Any] = (
    json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    if _CACHE_PATH.exists() else {}
)


def handle_api(req, rest: str) -> bool:
    if rest == "pages":
        req.send_json({
            "model": {"name": MODEL, "live": True},
            "pages": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "text": p["text"],
                    "site": LIVE_SITE.get(
                        p["id"], f"static/render/{p['id']}.html"),
                    "live": p["id"] in LIVE_SITE,
                }
                for p in PAGES
            ],
        })
    elif rest == "read":
        _read(req)
    else:
        return False
    return True


def _cache_key(page: Dict[str, str]) -> str:
    payload = json.dumps(
        [page["id"], page["text"], inspect.getsource(read)]).encode()
    return hashlib.sha256(payload).hexdigest()


def judge(client: TypeSafeClient, page: Dict[str, str],
          fresh: bool) -> Dict[str, Any]:
    key = _cache_key(page)
    if not fresh:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit is not None:
            return {**hit, "cached": True}

    response = read(client, page["text"])
    assert set(response.answers) == set(QUESTIONS), list(response.answers)

    answers: Dict[str, Any] = {}
    for name in QUESTIONS:
        answer = response.answers[name]
        result: Dict[str, Any] = {"kind": answer.type}
        if answer.type == "noul":
            result["p"] = answer.noul
        elif answer.type == "score":
            result["score"] = answer.score
            result["confidence"] = answer.confidence
            result["probabilities"] = answer.probabilities
        else:
            assert answer.type == "choice", answer.type
            result["choice"] = answer.choice
            result["confidence"] = answer.confidence
            result["probabilities"] = answer.probabilities
        answers[name] = result

    result = {
        "answers": answers,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    with _CACHE_LOCK:
        _CACHE[key] = result
        _CACHE_PATH.write_text(
            json.dumps(_CACHE, indent=1, sort_keys=True), encoding="utf-8")
    return {**result, "cached": False}


def _read(req) -> None:
    """Every question about one page in one request; ``?fresh=1`` re-bills."""
    page = BY_ID[req.query["page"][0]]
    fresh = req.query.get("fresh", ["0"])[0] not in ("0", "")

    _require("TYPESAFE_API_KEY")

    req.begin_sse()
    req.event("meta", {"page": page["id"], "count": len(QUESTIONS)})

    started = time.monotonic()
    with TypeSafeClient() as client:
        try:
            result = judge(client, page, fresh)
        except BaseException as exc:      # reported on the wire, then raised
            req.event("error", {"error": f"{type(exc).__name__}: {exc}"})
            raise
    for name in QUESTIONS:
        req.event("judged", {**result["answers"][name], "key": name,
                             "cached": result["cached"]})
    tokens = ((0, 0) if result["cached"] else
              (result["input_tokens"], result["output_tokens"]))
    req.event("done", {
        "seconds": round(time.monotonic() - started, 2),
        "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
    })


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set; put it in .env (see .env.example)")
    return value
