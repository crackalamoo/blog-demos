"""The dim demo's endpoints.

    GET /dim/api/story    the story, split into paragraphs
    GET /dim/api/sweep    Server-Sent Events: one Noul per paragraph
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient

load_dotenv()

SLUG = "dim"
TITLE = "Ctrl-F by meaning"
STATIC = Path(__file__).resolve().parent / "static"

MODEL = "jev-latest"

INSTRUCTIONS = (
    "A reader of this Sherlock Holmes story is looking for: {query}\n\n"
    "Is this paragraph one of the passages they are looking for?"
)

TRUE_CRITERIA = "This paragraph is part of what the reader asked for."

FALSE_CRITERIA = "This paragraph is not what the reader asked for."

_TEXT = (Path(__file__).resolve().parent / "silver_blaze.txt").read_text(
    encoding="utf-8")

PARAGRAPHS = [p for p in _TEXT.split("\n\n") if p.strip()]


def handle_api(req, rest: str) -> bool:
    if rest == "story":
        req.send_json({
            "model": {"name": MODEL, "live": True},
            "paragraphs": PARAGRAPHS,
        })
    elif rest == "sweep":
        _sweep(req)
    else:
        return False
    return True


def judge(client: TypeSafeClient, index: int, query: str) -> dict:
    """One paragraph, one question, one ``Noul``.

    A Noul answer carries only the probability; there is no confidence on it.
    """
    response = client.system_one(
        state={"paragraph": PARAGRAPHS[index]},
        questions={"relevant": Noul(
            instructions=INSTRUCTIONS.format(query=query),
            criteria={"true": TRUE_CRITERIA, "false": FALSE_CRITERIA},
        )},
        model=MODEL,
    )
    answer = response.answers["relevant"]
    assert answer.type == "noul", answer.type
    return {
        "i": index,
        "p": answer.noul,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _sweep(req) -> None:
    """Judge every paragraph against ``?q=``, emitting each as it lands.

    Every paragraph goes out at once; the whole sweep is one burst.
    """
    query = req.query["q"][0].strip()
    assert query, "empty query"

    _require("TYPESAFE_API_KEY")
    stop = threading.Event()
    landed: queue.Queue = queue.Queue()

    req.begin_sse()
    req.event("meta", {"query": query, "count": len(PARAGRAPHS)})

    started = time.monotonic()
    client = TypeSafeClient()
    pool = ThreadPoolExecutor(max_workers=len(PARAGRAPHS))

    def run(index: int) -> None:
        if stop.is_set():
            landed.put(None)
            return
        try:
            landed.put(judge(client, index, query))
        except BaseException as exc:      # reported on the wire, then raised
            landed.put({"i": index, "error": f"{type(exc).__name__}: {exc}"})

    for i in range(len(PARAGRAPHS)):
        pool.submit(run, i)

    tokens = [0, 0]
    try:
        for _ in range(len(PARAGRAPHS)):
            result = landed.get()
            if result is None:
                continue
            if "error" in result:
                req.event("error", result)
                raise RuntimeError(result["error"])
            tokens[0] += result["input_tokens"]
            tokens[1] += result["output_tokens"]
            req.event("judged", {"i": result["i"], "p": result["p"]})
        req.event("done", {
            "seconds": round(time.monotonic() - started, 2),
            "usage": {"input_tokens": tokens[0], "output_tokens": tokens[1]},
        })
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        client.close()


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set; put it in .env (see .env.example)")
    return value
