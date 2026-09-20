"""The dim demo's endpoints.

    GET /dim/api/story    the story, split into paragraphs
    GET /dim/api/sweep    Server-Sent Events: one Noul per paragraph,
                          sent in chunks of consecutive paragraphs
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
    "Is paragraph_{n} one of the passages they are looking for?"
)

TRUE_CRITERIA = "This paragraph is part of what the reader asked for."

FALSE_CRITERIA = "This paragraph is not what the reader asked for."

_TEXT = (Path(__file__).resolve().parent / "silver_blaze.txt").read_text(
    encoding="utf-8")

PARAGRAPHS = [p for p in _TEXT.split("\n\n") if p.strip()]

# The API caps a request at 32 questions; a chunk is one request.
MAX_QUESTIONS_PER_REQUEST = 32


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


def judge(client: TypeSafeClient, chunk: range, query: str) -> tuple:
    """One request, one Noul per paragraph of ``chunk``.

    Neighbors share the request: a line too short to judge alone, like
    "That was the curious incident,", is plain beside the ones around it.
    A Noul answer carries only the probability; there is no confidence.
    """
    response = client.system_one(
        state={f"paragraph_{n}": PARAGRAPHS[i]
               for n, i in enumerate(chunk)},
        questions={f"q{n}": Noul(
            instructions=INSTRUCTIONS.format(query=query, n=n),
            criteria={"true": TRUE_CRITERIA, "false": FALSE_CRITERIA},
        ) for n, i in enumerate(chunk)},
        model=MODEL,
    )
    assert len(response.answers) == len(chunk), len(response.answers)
    results = []
    for n, i in enumerate(chunk):
        answer = response.answers[f"q{n}"]
        assert answer.type == "noul", answer.type
        results.append({"i": i, "p": answer.noul})
    return results, response.usage


def _sweep(req) -> None:
    """Judge every paragraph against ``?q=``, emitting each as it lands.

    Every chunk goes out at once; the whole sweep is one burst.
    """
    query = req.query["q"][0].strip()
    assert query, "empty query"

    _require("TYPESAFE_API_KEY")
    stop = threading.Event()
    landed: queue.Queue = queue.Queue()

    req.begin_sse()
    req.event("meta", {"query": query, "count": len(PARAGRAPHS)})

    chunks = [range(start, min(start + MAX_QUESTIONS_PER_REQUEST,
                               len(PARAGRAPHS)))
              for start in range(0, len(PARAGRAPHS),
                                 MAX_QUESTIONS_PER_REQUEST)]

    started = time.monotonic()
    client = TypeSafeClient()
    pool = ThreadPoolExecutor(max_workers=len(chunks))

    def run(chunk: range) -> None:
        if stop.is_set():
            landed.put(None)
            return
        try:
            results, usage = judge(client, chunk, query)
            landed.put({"results": results, "usage": usage})
        except BaseException as exc:      # reported on the wire, then raised
            landed.put({"i": chunk.start,
                        "error": f"{type(exc).__name__}: {exc}"})

    for chunk in chunks:
        pool.submit(run, chunk)

    tokens = [0, 0]
    try:
        for _ in range(len(chunks)):
            result = landed.get()
            if result is None:
                continue
            if "error" in result:
                req.event("error", result)
                raise RuntimeError(result["error"])
            tokens[0] += result["usage"].input_tokens
            tokens[1] += result["usage"].output_tokens
            for judged in result["results"]:
                req.event("judged", judged)
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
