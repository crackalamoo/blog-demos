"""The Minecraft demo's endpoints.

    GET  /minecraft/api/meta      the model names and the running session
    GET  /minecraft/api/stream    Server-Sent Events: one decision at a time
    POST /minecraft/api/session   the bot announcing itself on connect
    POST /minecraft/api/decide    world state + legal actions in, one action
                                  out

The bot is a Node process (bot/bot.js) that speaks the game protocol; every
decision it makes is made here.  It plays a world generated from a fresh
random seed each run, and nothing on either side knows anything about that
world beyond what the bot can currently see.

Milestones are the demo's result: the game events that mark progress, with
the decision and the world clock at which each first happened.  They are
shown on the page.  They are not goals -- nothing is offered to, scored
for, or steered by them.

Nothing here is authoritative.  The bot holds the session and the whole
history of what has happened to it and sends both with every decision, and
what is below is a mirror of that, rebuilt from each POST.  Restarting this
server therefore costs a running bot nothing: the next decision restores
the seed, the viewer and every milestone already reached.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .decider import (
    CHAT_MODEL,
    DECISIONS_PER_OBJECTIVE,
    JEV_MODEL,
    OBJECTIVE_KEY,
    ActionLog,
    choose_action,
    choose_objective,
)

SLUG = "minecraft"
TITLE = "Playing Minecraft"
STATIC = Path(__file__).resolve().parent / "static"

#: 20 ticks to the second, 24000 to the Minecraft day.
TICKS_PER_DAY = 24000

#: Dollars per million input tokens.  Output is free.
PRICE_IN_PER_MTOK = 0.042

_LOCK = threading.Lock()
_LOG = ActionLog()
_SUBSCRIBERS: List["queue.Queue[Dict[str, Any]]"] = []

_SESSION: Optional[Dict[str, Any]] = None
_LATEST: Optional[Dict[str, Any]] = None
_MILESTONES: List[Dict[str, Any]] = []
_objective: Optional[str] = None
_since_objective = 0
_decisions = 0
_input_tokens = 0


def handle_api(req, rest: str) -> bool:
    if rest == "meta":
        req.send_json(meta())
    elif rest == "stream":
        _stream(req)
    else:
        return False
    return True


def handle_post(req, rest: str) -> bool:
    if rest == "session":
        req.send_json(_session(req.json_body()))
    elif rest == "decide":
        req.send_json(_decide(req.json_body()))
    else:
        return False
    return True


def meta() -> dict:
    with _LOCK:
        return {
            "model": {"name": JEV_MODEL, "live": True},
            "objective_model": {"name": CHAT_MODEL, "live": True},
            "objective": _objective,
            "decisions_per_objective": DECISIONS_PER_OBJECTIVE,
            "session": _SESSION,
            "milestones": list(_MILESTONES),
        }


def _require(body: Dict[str, Any], keys: tuple) -> None:
    for key in keys:
        if key not in body:
            raise KeyError(f"payload has no {key!r}: {sorted(body)}")


def _publish(name: str, payload: Dict[str, Any]) -> None:
    for sub in list(_SUBSCRIBERS):
        sub.put({"name": name, "payload": payload})


def _session(body: Dict[str, Any]) -> Dict[str, Any]:
    with _LOCK:
        _adopt(body)
    print(f"minecraft: seed {body['seed']} on {body['minecraft']}")
    return {"ok": True}


def _adopt(session: Dict[str, Any]) -> None:
    """Take the bot's session, clearing the mirror if it is a new one.

    Called under _LOCK, from the connect POST and from every decision.  A
    session this process has never seen is a new run only when it differs
    from the one held; an identical one arriving after a restart is the run
    already in progress, and its milestones are about to be replayed.
    """
    global _SESSION, _LATEST, _objective, _since_objective, _decisions
    global _input_tokens
    _require(session, ("seed", "viewer", "minecraft", "version"))
    if _SESSION == session:
        return
    _SESSION = dict(session)
    _LATEST = None
    _MILESTONES.clear()
    _objective = None
    _since_objective = 0
    _decisions = 0
    _input_tokens = 0
    _publish("session", _SESSION)


def _clock(ticks: int) -> str:
    """The world clock as days and time of day, from the tick count."""
    return f"day {ticks // TICKS_PER_DAY + 1}, {ticks % TICKS_PER_DAY} ticks"


def _decide(body: Dict[str, Any]) -> Dict[str, Any]:
    """One decision: the bot's readout in, the chosen action's label out.

    A death, and every ``DECISIONS_PER_OBJECTIVE`` decisions, re-sets the
    objective before the state is assembled, so the bot is told what it is
    working towards on the same decision it learns it died.
    """
    global _LATEST, _objective, _since_objective, _decisions, _input_tokens
    _require(body, ("state", "actions", "history", "ticks", "decision",
                    "session"))
    state: Dict[str, Any] = body["state"]
    actions: List[Dict[str, Any]] = body["actions"]
    history: List[Dict[str, Any]] = body["history"]
    if OBJECTIVE_KEY in state:
        raise ValueError("the objective is set here, not by the bot")

    with _LOCK:
        _adopt(body["session"])
        _decisions = body["decision"]
        if len(history) < len(_MILESTONES):
            raise ValueError(
                f"the bot sent {len(history)} milestones, fewer than the "
                f"{len(_MILESTONES)} already mirrored from it"
            )
        died = False
        for event in history[len(_MILESTONES):]:
            _require(event, ("name", "detail", "decision", "ticks"))
            _MILESTONES.append({
                "name": event["name"],
                "detail": event["detail"],
                "decision": event["decision"],
                "clock": _clock(event["ticks"]),
            })
            _publish("milestone", _MILESTONES[-1])
            if event["name"] == "death":
                died = True
                _LOG.record({"did": "died", "result": event["detail"]})

        last = state.get("last_action")
        if last is not None:
            _LOG.record(dict(last))

        stale = _since_objective >= DECISIONS_PER_OBJECTIVE
        if _objective is None or died or stale:
            _objective = choose_objective(_LOG, state)
            _since_objective = 0

        payload = {OBJECTIVE_KEY: _objective, **state}
        decision = choose_action(payload, actions)
        chosen = decision.label
        labels = [a["label"] for a in actions]
        if chosen not in labels:
            raise ValueError(f"chose {chosen!r}, not one of {labels}")

        _since_objective += 1
        _input_tokens += decision.input_tokens
        record = {
            "decision": _decisions,
            "clock": _clock(body["ticks"]),
            "state": payload,
            "actions": actions,
            "chosen": chosen,
            "confidence": decision.confidence,
            "probabilities": decision.probabilities,
            "objective": _objective,
            "since_objective": _since_objective,
            "usage": {
                "input_tokens": decision.input_tokens,
                "session_input_tokens": _input_tokens,
                "session_cost_usd": _input_tokens / 1e6 * PRICE_IN_PER_MTOK,
            },
        }
        _LATEST = record
        _publish("decision", record)

    return {"chosen": chosen, "objective": _objective}


def _stream(req) -> None:
    sub: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    with _LOCK:
        _SUBSCRIBERS.append(sub)
        session, latest = _SESSION, _LATEST
        milestones = list(_MILESTONES)
    try:
        req.begin_sse()
        req.event("meta", meta())
        if session is not None:
            req.event("session", session)
        for milestone in milestones:
            req.event("milestone", milestone)
        if latest is not None:
            req.event("decision", latest)
        while True:
            try:
                message = sub.get(timeout=15.0)
            except queue.Empty:
                req.event("ping", {})
                continue
            req.event(message["name"], message["payload"])
    finally:
        with _LOCK:
            if sub in _SUBSCRIBERS:
                _SUBSCRIBERS.remove(sub)
