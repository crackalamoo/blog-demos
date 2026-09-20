"""The chat demo's endpoints.

    GET /chat/api/meta      the two model names
    GET /chat/api/stream    Server-Sent Events: one round -- ?round=N says which
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List

from .speaker import (
    CHAT_MODEL,
    GPT,
    JEV,
    JEV_MODEL,
    MAX_CANDIDATES,
    MIN_CANDIDATES,
    LiveChat,
    LiveJev,
    Message,
    chat_client,
    jev_client,
)

SLUG = "chat"
TITLE = "Talking to ChatGPT"
STATIC = Path(__file__).resolve().parent / "static"

#: Beat between one speaker finishing and the next starting.
TURN_PAUSE_MS = 900

#: Below this confidence a choice is marked unsure for the front end.
LOW_CONFIDENCE = 0.55

#: A round is one ChatGPT turn and Jev's reply to it; a click runs one round.
TURNS_PER_ROUND = 2

#: How far a single visit may run, so a page cannot spend without limit.
ROUNDS = 8

#: The conversation both models are shown.  A request carries only the round
#: index, so the turns live here between requests; one run at a time.
_LOCK = threading.Lock()
_HISTORY: List[Message] = []


def handle_api(req, rest: str) -> bool:
    if rest == "meta":
        req.send_json(meta())
    elif rest == "stream":
        _stream(req)
    else:
        return False
    return True


def meta() -> dict:
    return {
        "jev": {"name": JEV_MODEL, "live": True},
        "chat": {"name": CHAT_MODEL, "live": True},
        "candidates": [MIN_CANDIDATES, MAX_CANDIDATES],
        "low_confidence": LOW_CONFIDENCE,
        "rounds": ROUNDS,
    }


def _stream(req) -> None:
    """Run one round: a ChatGPT turn and Jev's choice of reply to it.

    Each click costs one ChatGPT call and one classification, so the client asks
    for one round at a time and carries the resume index.  ``?round=0`` starts a
    fresh conversation; any later round must be the one that follows the turns
    already spoken.  ``?turn_pause`` overrides the beat between speakers, in
    milliseconds.  The browser leaving mid-stream is the one expected failure;
    anything else is reported on the wire and then raised.
    """
    round_index = int(req.float_param("round", 0))
    if not 0 <= round_index < ROUNDS:
        req.send_json(
            {"error": f"round {round_index} is outside 0..{ROUNDS - 1}"}, status=404)
        return

    turn_pause = req.float_param("turn_pause", TURN_PAUSE_MS) / 1000.0

    with _LOCK:
        if round_index == 0:
            _HISTORY.clear()
        elif round_index * TURNS_PER_ROUND != len(_HISTORY):
            req.send_json({"error": (
                f"round {round_index} does not follow the {len(_HISTORY)} turns "
                "spoken so far"
            )}, status=409)
            return

        turn_index = round_index * TURNS_PER_ROUND

        req.begin_sse()
        req.event("meta", meta())

        try:
            with jev_client() as typesafe, chat_client() as openai_client:
                jev = LiveJev(typesafe)
                chat = LiveChat(openai_client)

                req.event("turn_start", {"turn": turn_index, "speaker": GPT})
                proposal = chat.propose(_HISTORY, lambda text: req.event(
                    "chunk", {"turn": turn_index, "text": text}))
                _HISTORY.append(Message(GPT, proposal.turn))
                req.event("turn_done", {
                    "turn": turn_index, "speaker": GPT, "text": proposal.turn})

                turn_index += 1
                time.sleep(turn_pause)
                req.event("turn_start", {"turn": turn_index, "speaker": JEV})
                req.event("candidates", {
                    "turn": turn_index, "candidates": proposal.candidates})

                choice = jev.choose(_HISTORY, proposal.candidates)
                _HISTORY.append(
                    Message(JEV, choice.chosen, choice.measurement()))
                req.event("choice", {
                    "turn": turn_index,
                    "chosen": choice.chosen,
                    "confidence": choice.confidence,
                    "low": choice.confidence < LOW_CONFIDENCE,
                    "ranked": choice.ranked(),
                })
                req.event("turn_done", {
                    "turn": turn_index, "speaker": JEV, "text": choice.chosen})

            next_round = round_index + 1
            req.event("done", {
                "round": round_index,
                "next_round": next_round if next_round < ROUNDS else None,
            })
        except (BrokenPipeError, ConnectionResetError):
            raise                       # the browser left; nothing to report to
        except BaseException as exc:
            # Say why the stream stopped before unwinding, so the page fails visibly.
            req.event(
                "error", {"turn": turn_index, "error": f"{type(exc).__name__}: {exc}"})
            raise
