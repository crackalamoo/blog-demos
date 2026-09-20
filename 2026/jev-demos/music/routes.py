"""The music demo's endpoints.

    GET /music/api/meta      grid + vocabulary + voice layout (no decisions)
    GET /music/api/stream    Server-Sent Events: one bar at a time, endless
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from .decider import configured_model, make_decider
from .grid import DECISIONS_PER_BAR, NoteResolver, Piece

SLUG = "music"
TITLE = "Endless music"
STATIC = Path(__file__).resolve().parent / "static"

#: How far ahead of the playhead the stream is allowed to get.  Generation
#: outrunning playback is the point of the demo; racing to the end is not.
LOOKAHEAD_BARS = 2

#: Bars of context handed to the decider.  A run is endless, so the whole piece
#: cannot be kept; decisions carry their own bar number regardless.
CONTEXT_BARS = 8


def handle_api(req, rest: str) -> bool:
    if rest == "meta":
        req.send_json(meta())
    elif rest == "stream":
        _stream(req)
    else:
        return False
    return True


def meta() -> dict:
    skeleton = Piece().to_json()
    for key in ("bars", "notes"):
        skeleton.pop(key)
    skeleton["decisions_per_bar"] = DECISIONS_PER_BAR
    skeleton["model"] = {"name": configured_model(), "live": True}
    return skeleton


def _stream(req) -> None:
    """Emit bars forever, paced to stay ``lookahead`` bars ahead of playback.

    ``?bars=N`` bounds the run instead, and only a bounded run ends with a
    ``done`` event carrying the whole piece.  A failed bar ends the stream with
    an ``error`` event and then raises: the demo fails visibly, in both places.
    """
    lookahead = int(req.float_param("lookahead_bars", req.server.lookahead_bars))
    limit = int(req.float_param("bars", 0))
    tempo = req.server.tempo
    bar_seconds = 4 * 60.0 / tempo          # one 4/4 bar

    req.begin_sse()

    decider = make_decider()
    resolver = NoteResolver()
    recent = deque(maxlen=CONTEXT_BARS)
    archive = [] if limit else None          # only a bounded run keeps it all

    bar_index = 0
    try:
        tonal_center = decider.tonal_center()
        head = meta() | {
            "lookahead_bars": lookahead,
            "tonal_center": tonal_center.to_json(),
        }
        if limit:
            head["bar_count"] = limit
        req.event("meta", head)

        started = time.monotonic()
        while not limit or bar_index < limit:
            # The client starts playing once `lookahead` bars have landed, so bar
            # N is due one bar-length after bar N-1 from that moment on.
            due = (bar_index - lookahead + 1) * bar_seconds
            ahead = due - (time.monotonic() - started)
            if ahead > 0:
                time.sleep(ahead)

            req.event("bar_start", {"bar": bar_index})
            decisions = decider.decide_bar(
                Piece(tempo_bpm=tempo, bars=list(recent)), bar_index
            )
            recent.append(list(decisions))
            if archive is not None:
                archive.append(list(decisions))
            notes = resolver.feed(decisions)
            for decision in decisions:
                req.event("decision", decision.to_json())
            req.event(
                "bar_done",
                {"bar": bar_index, "notes": notes, "usage": decider.last_usage},
            )
            bar_index += 1

        if archive is not None:
            req.event(
                "done",
                {
                    "piece": Piece(
                        tempo_bpm=tempo, bars=archive, tonal_center=tonal_center
                    ).to_json()
                },
            )
    except (BrokenPipeError, ConnectionResetError):
        raise                           # the browser left; nothing to report to
    except BaseException as exc:
        # Report on the wire before unwinding, so the page says why it stopped
        # rather than seeing the socket close on it.
        req.event(
            "error",
            {"bar": bar_index, "error": f"{type(exc).__name__}: {exc}"},
        )
        raise
    finally:
        decider.close()
