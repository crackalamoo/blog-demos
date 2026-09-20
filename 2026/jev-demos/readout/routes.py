"""The readout demo's endpoints.

    GET /readout/api/pages    the pages, in carousel order
    GET /readout/api/read     Server-Sent Events: one judgment per question
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

from .questions import QUESTIONS

load_dotenv()

SLUG = "readout"
TITLE = "Judging landing pages"
STATIC = Path(__file__).resolve().parent / "static"

MODEL = "jev-latest"

_HERE = Path(__file__).resolve().parent
_CACHE_PATH = _HERE / "cache.json"


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
            "questions": [
                {k: v for k, v in q.items() if k != "question"}
                for q in QUESTIONS
            ],
        })
    elif rest == "read":
        _read(req)
    else:
        return False
    return True


def _cache_key(page: Dict[str, str], spec: Dict[str, Any]) -> str:
    payload = json.dumps(
        [page["id"], page["text"], spec], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _spec(question) -> Dict[str, Any]:
    return json.loads(question.model_dump_json(exclude_none=True))


def judge(client: TypeSafeClient, page: Dict[str, str],
          entry: Dict[str, Any], fresh: bool) -> Dict[str, Any]:
    """One page, one question, one request."""
    question = entry["question"]()
    key = _cache_key(page, _spec(question))
    if not fresh:
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit is not None:
            return {**hit, "key": entry["key"], "cached": True}

    response = client.system_one(
        state={"landing_page": page["text"]},
        questions={entry["key"]: question},
        model=MODEL,
    )
    answer = response.answers[entry["key"]]
    assert answer.type == entry["kind"], (answer.type, entry["kind"])

    result: Dict[str, Any] = {
        "kind": entry["kind"],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    if answer.type == "noul":
        result["p"] = answer.noul
    elif answer.type == "score":
        result["score"] = answer.score
        result["confidence"] = answer.confidence
        result["probabilities"] = answer.probabilities
    else:
        result["choice"] = answer.choice
        result["confidence"] = answer.confidence
        result["probabilities"] = answer.probabilities

    with _CACHE_LOCK:
        _CACHE[key] = result
        _CACHE_PATH.write_text(
            json.dumps(_CACHE, indent=1, sort_keys=True), encoding="utf-8")
    return {**result, "key": entry["key"], "cached": False}


def _read(req) -> None:
    """Every question about one page at once; ``?fresh=1`` re-bills."""
    page = BY_ID[req.query["page"][0]]
    fresh = req.query.get("fresh", ["0"])[0] not in ("0", "")

    _require("TYPESAFE_API_KEY")
    landed: queue.Queue = queue.Queue()

    req.begin_sse()
    req.event("meta", {"page": page["id"], "count": len(QUESTIONS)})

    started = time.monotonic()
    client = TypeSafeClient()
    pool = ThreadPoolExecutor(max_workers=len(QUESTIONS))

    def run(entry: Dict[str, Any]) -> None:
        try:
            landed.put(judge(client, page, entry, fresh))
        except BaseException as exc:      # reported on the wire, then raised
            landed.put({"key": entry["key"],
                        "error": f"{type(exc).__name__}: {exc}"})

    for entry in QUESTIONS:
        pool.submit(run, entry)

    tokens = [0, 0]
    try:
        for _ in range(len(QUESTIONS)):
            result = landed.get()
            if "error" in result:
                req.event("error", result)
                raise RuntimeError(result["error"])
            if not result["cached"]:
                tokens[0] += result["input_tokens"]
                tokens[1] += result["output_tokens"]
            req.event("judged", result)
        req.event("done", {
            "seconds": round(time.monotonic() - started, 2),
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
        })
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        client.close()


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set; put it in .env (see .env.example)")
    return value
