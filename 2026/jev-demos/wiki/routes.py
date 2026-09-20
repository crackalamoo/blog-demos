"""The wiki demo's endpoints.

    GET /wiki/api/stream   Server-Sent Events: one judgment per live edit
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from .questions import (
    ACTION_CRITERIA,
    ACTION_INSTRUCTIONS,
    ACTION_LABELS,
    MISLEADING_CRITERIA,
    MISLEADING_INSTRUCTIONS,
    MISLEADING_RUNGS,
    SUMMARY_CRITERIA,
    SUMMARY_INSTRUCTIONS,
)

SLUG = "wiki"
TITLE = "Reading every Wikipedia edit, live"
STATIC = Path(__file__).resolve().parent / "static"

MODEL = "jev-latest"

FIREHOSE = "https://stream.wikimedia.org/v2/stream/recentchange"
COMPARE = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "jev-demos/0.1 (local demo)"

load_dotenv()

#: Optional.  Without it Wikipedia throttles compare to a few requests
#: per minute, and the demo judges a fraction of the edits it sees.
TOKEN = os.environ.get("WIKIMEDIA_TOKEN", "").strip()

HEADERS = {"User-Agent": USER_AGENT}
if TOKEN:
    HEADERS["Authorization"] = f"Bearer {TOKEN}"

WORKERS = 8 if TOKEN else 4
DIFF_CHARS = 1400

#: Measured: authenticated compare sustains ~5/s with no 429, against an
#: in-scope arrival rate of ~1.1/s.  Anonymous dies after ~10 requests.
COMPARE_MIN_INTERVAL = 0.2 if TOKEN else 1.5
COMPARE_MAX_INTERVAL = 12.0
RATE_WINDOW = 6.0
RATE_INTERVAL = 0.5

PRICE_IN_PER_MTOK = 0.042
PRICE_OUT_PER_MTOK = 0.0


#: One entry per open stream, keyed by the page's ``sid``.  The page
#: flips ``paused``; the reader thread below obeys it.
SESSIONS: dict[str, dict] = {}
SESSIONS_LOCK = threading.Lock()


def handle_api(req, rest: str) -> bool:
    if rest == "stream":
        _stream(req)
    elif rest == "pause":
        _set_pause(req)
    else:
        return False
    return True


def _set_pause(req) -> None:
    sid = req.query.get("sid", [""])[0]
    on = req.query.get("on", ["0"])[0] == "1"
    with SESSIONS_LOCK:
        session = SESSIONS.get(sid)
    if session is None:
        req.send_json({"error": "no such stream", "sid": sid}, 404)
        return
    if on:
        session["skipped_now"] = 0
        session["paused"].set()
    else:
        session["paused"].clear()
    req.send_json({"paused": on})


def in_scope(event: dict) -> bool:
    return (
        event.get("wiki") == "enwiki"
        and event.get("type") == "edit"
        and event.get("namespace") == 0
        and event.get("bot") is False
    )


class _DiffParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.added: list[str] = []
        self.removed: list[str] = []
        self._sink: list[str] | None = None

    def handle_starttag(self, tag, attrs) -> None:
        if tag != "td":
            return
        classes = dict(attrs).get("class", "").split()
        if "diff-addedline" in classes:
            self._sink = self.added
        elif "diff-deletedline" in classes:
            self._sink = self.removed
        else:
            self._sink = None

    def handle_endtag(self, tag) -> None:
        if tag == "td":
            self._sink = None

    def handle_data(self, data) -> None:
        if self._sink is not None and data.strip():
            self._sink.append(data)


class Pacer:
    """en.wikipedia.org throttles anonymous callers; this finds the line."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0
        self.interval = COMPARE_MIN_INTERVAL

    def wait(self) -> None:
        with self._lock:
            when = max(time.monotonic(), self._next)
            self._next = when + self.interval
        time.sleep(max(0.0, when - time.monotonic()))

    def throttled(self, retry_after: float) -> None:
        with self._lock:
            self.interval = min(COMPARE_MAX_INTERVAL, self.interval * 1.5)
            self._next = time.monotonic() + retry_after

    def eased(self) -> None:
        with self._lock:
            self.interval = max(COMPARE_MIN_INTERVAL, self.interval * 0.93)


PACER = Pacer()


def fetch_diff(old_rev: int, new_rev: int) -> dict | None:
    """The diff, or None when Wikipedia throttled this request."""
    assert old_rev and new_rev, (old_rev, new_rev)
    PACER.wait()
    query = urllib.parse.urlencode({
        "action": "compare",
        "fromrev": old_rev,
        "torev": new_rev,
        "prop": "diff",
        "format": "json",
    })
    request = urllib.request.Request(
        f"{COMPARE}?{query}", headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code != 429:
            raise
        PACER.throttled(float(exc.headers.get("Retry-After", 5)))
        return None
    PACER.eased()
    parser = _DiffParser()
    parser.feed(body["compare"]["*"])
    return {
        "added": " ".join(parser.added)[:DIFF_CHARS],
        "removed": " ".join(parser.removed)[:DIFF_CHARS],
    }


def state_of(event: dict, diff: dict) -> dict:
    return {
        "title": event["title"],
        "user": event["user"],
        "summary": event["comment"],
        "byte_delta": event["length"]["new"] - event["length"]["old"],
        "added": diff["added"],
        "removed": diff["removed"],
    }


def judge_action(client: TypeSafeClient, state: dict) -> tuple:
    """One edit, one question, one ``Choice`` over the seven labels."""
    response = client.system_one(
        state=state,
        questions={"action": Choice(
            instructions=ACTION_INSTRUCTIONS,
            criteria=ACTION_CRITERIA,
        )},
        model=MODEL,
    )
    answer = response.answers["action"]
    assert answer.type == "choice", answer.type
    return {
        "label": answer.choice,
        "confidence": answer.confidence,
        "probabilities": dict(answer.probabilities),
    }, response.usage


def judge_misleading(client: TypeSafeClient, state: dict) -> tuple:
    """One edit, one question, one ``Score`` over the five rungs.

    ``score`` is a probability-weighted mean and is passed through raw.
    """
    response = client.system_one(
        state=state,
        questions={"misleading": Score(
            instructions=MISLEADING_INSTRUCTIONS,
            criteria=MISLEADING_CRITERIA,
        )},
        model=MODEL,
    )
    answer = response.answers["misleading"]
    assert answer.type == "score", answer.type
    return {
        "score": answer.score,
        "confidence": answer.confidence,
        "probabilities": {str(k): v
                          for k, v in answer.probabilities.items()},
    }, response.usage


def judge_summary_honest(client: TypeSafeClient, state: dict) -> tuple:
    """One edit, one question, one ``Noul``; it carries no confidence."""
    response = client.system_one(
        state=state,
        questions={"summary_honest": Noul(
            instructions=SUMMARY_INSTRUCTIONS,
            criteria=SUMMARY_CRITERIA,
        )},
        model=MODEL,
    )
    answer = response.answers["summary_honest"]
    assert answer.type == "noul", answer.type
    return {"p": answer.noul}, response.usage


def judge(client: TypeSafeClient, event: dict) -> dict | None:
    """Three questions, three requests, one per question, in parallel."""
    diff = fetch_diff(event["revision"]["old"], event["revision"]["new"])
    if diff is None:
        return None
    state = state_of(event, diff)
    with ThreadPoolExecutor(max_workers=3) as asking:
        futures = [asking.submit(fn, client, state) for fn in
                   (judge_action, judge_misleading, judge_summary_honest)]
        (action, u_a), (misleading, u_m), (summary, u_s) = [
            f.result() for f in futures]
    return {
        "id": event["id"],
        "title": state["title"],
        "user": state["user"],
        "summary": state["summary"],
        "byte_delta": state["byte_delta"],
        "added": state["added"],
        "removed": state["removed"],
        "url": event["notify_url"],
        "action": action,
        "misleading": misleading,
        "summary_honest": summary,
        "input_tokens": (u_a.input_tokens + u_m.input_tokens
                         + u_s.input_tokens),
        "output_tokens": (u_a.output_tokens + u_m.output_tokens
                          + u_s.output_tokens),
    }


def _read_firehose(client, response, stop, pool, landed, seen, scope,
                   slots, session) -> None:
    while not stop.is_set():
        try:
            line = response.readline()
        except (OSError, ValueError):
            if stop.is_set():
                return              # the browser left; we closed the socket
            raise
        if not line:
            break
        if not line.startswith(b"data: "):
            continue
        event = json.loads(line[6:])
        if session["paused"].is_set():
            if in_scope(event):       # dropped outright: never judged
                session["skipped_now"] += 1
                session["skipped_total"] += 1
            continue
        seen.append(time.monotonic())
        if not in_scope(event):
            continue
        scope.append(time.monotonic())
        if slots.acquire(blocking=False):
            pool.submit(_run, client, event, stop, landed, slots)


def _run(client, event, stop, landed, slots) -> None:
    try:
        if stop.is_set():
            return
        result = judge(client, event)
        landed.put({"throttled": True} if result is None else result)
    except BaseException as exc:      # reported on the wire, then raised
        landed.put({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        slots.release()


def _prune(stamps: deque, now: float) -> float:
    while stamps and now - stamps[0] > RATE_WINDOW:
        stamps.popleft()
    return len(stamps) / RATE_WINDOW


def _emit(req, totals, judged, result, stamp: bool) -> None:
    """Send one judged edit.  ``stamp`` feeds the live rate window."""
    totals["judged"] += 1
    totals["input_tokens"] += result.pop("input_tokens")
    totals["output_tokens"] += result.pop("output_tokens")
    if stamp:
        judged.append(time.monotonic())
    req.event("edit", result)


def _send_pause(req, session, paused: bool) -> None:
    req.event("paused", {
        "paused": paused,
        "skipped_now": session["skipped_now"],
        "skipped_total": session["skipped_total"],
    })


def _stream(req) -> None:
    """Judge the live enwiki firehose for as long as the page is open."""
    stop = threading.Event()
    landed: queue.Queue = queue.Queue()
    seen: deque = deque()
    scope: deque = deque()
    judged: deque = deque()
    held: list = []
    slots = threading.Semaphore(WORKERS)
    sid = req.query.get("sid", [""])[0]
    session = {"paused": threading.Event(), "skipped_now": 0,
               "skipped_total": 0}
    with SESSIONS_LOCK:
        SESSIONS[sid] = session

    _require("TYPESAFE_API_KEY")
    client = TypeSafeClient()
    request = urllib.request.Request(
        FIREHOSE, headers={"User-Agent": USER_AGENT})
    response = urllib.request.urlopen(request, timeout=30)
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    reader = threading.Thread(
        target=_read_firehose,
        args=(client, response, stop, pool, landed, seen, scope, slots,
              session),
        daemon=True,
    )

    req.begin_sse()
    req.event("meta", {
        "model": {"name": MODEL, "live": True},
        "action_labels": ACTION_LABELS,
        "action_criteria": ACTION_CRITERIA,
        "misleading_rungs": MISLEADING_RUNGS,
        "misleading_criteria": MISLEADING_CRITERIA,
    })
    reader.start()

    totals = {"judged": 0, "throttled": 0,
              "input_tokens": 0, "output_tokens": 0}
    was_paused = False
    next_rates = time.monotonic()
    try:
        while True:
            try:
                result = landed.get(timeout=RATE_INTERVAL)
            except queue.Empty:
                result = None
            if result is not None:
                if "error" in result:
                    req.event("error", result)
                    raise RuntimeError(result["error"])
                if "throttled" in result:
                    totals["throttled"] += 1
                    result = None
            paused = session["paused"].is_set()
            if result is not None and paused:
                # In flight when the pause began: paid for already, so
                # it is shown on resume rather than thrown away.
                held.append(result)
                result = None
            if result is not None:
                _emit(req, totals, judged, result, True)
            now = time.monotonic()
            if paused != was_paused:
                was_paused = paused
                if not paused:
                    seen.clear()
                    scope.clear()
                    judged.clear()
                    for item in held:
                        _emit(req, totals, judged, item, False)
                    held.clear()
                _send_pause(req, session, paused)
                next_rates = now + RATE_INTERVAL
            if now >= next_rates:
                next_rates = now + RATE_INTERVAL
                if paused:
                    _send_pause(req, session, True)
                    continue
                req.event("rates", {
                    "seen_per_sec": round(_prune(seen, now), 2),
                    "in_scope_per_sec": round(_prune(scope, now), 2),
                    "judged_per_sec": round(_prune(judged, now), 2),
                    "total_judged": totals["judged"],
                    "throttled": totals["throttled"],
                    "skipped_while_paused": session["skipped_total"],
                    "compare_interval": round(PACER.interval, 2),
                    "input_tokens": totals["input_tokens"],
                    "output_tokens": totals["output_tokens"],
                    "spend_usd": round(
                        totals["input_tokens"] / 1e6 * PRICE_IN_PER_MTOK
                        + totals["output_tokens"] / 1e6
                        * PRICE_OUT_PER_MTOK, 6),
                })
    finally:
        stop.set()
        with SESSIONS_LOCK:
            SESSIONS.pop(sid, None)
        response.close()
        pool.shutdown(wait=False, cancel_futures=True)
        client.close()


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set; put it in .env (see .env.example)")
    return value
