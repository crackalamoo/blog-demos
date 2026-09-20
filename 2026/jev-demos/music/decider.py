"""The decider interface and its Jev implementation.

Before the first bar, one call fixes the tonal center for the whole run.  One
bar is decided by mask-predict refinement: a rough pass asks all 24 slots
blind and samples each from the model's own distribution, then each refinement
round re-asks the still-unfrozen slots with the bar so far written out and a
``?`` at the slot being asked.  Rounds are not Gibbs sampling: there is no
joint distribution here and nothing converges.  Bars stay sequential, so bar
N's state carries the bars before it, and one bar is still one reveal.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, List, Protocol, Tuple

from .grid import (
    DECISIONS_PER_BAR,
    HOLD,
    OPTIONS,
    PITCH_COUNT,
    REST,
    PITCH_CLASS_NAMES,
    SLOTS_PER_BAR,
    VOICES,
    Decision,
    Piece,
    TonalCenter,
)

#: Per-request question ceiling on the API; a bar must fit in one request.
MAX_QUESTIONS = 32
assert DECISIONS_PER_BAR <= MAX_QUESTIONS

DEFAULT_MODEL = "jev-latest"

#: Total rounds per bar: one rough pass plus REFINEMENT_ROUNDS - 1 refinements.
REFINEMENT_ROUNDS = 2

#: A rough-pass slot whose top option carries at least this much mass is taken
#: at argmax and frozen; the rest are sampled and re-asked.
FREEZE_CONFIDENCE = 0.6

REST_CRITERIA = "Silence: any note sounding in this voice stops here."

HOLD_CRITERIA = "No new note: the note already sounding in this voice continues."

TONAL_CENTER_QUESTION = "tonal_center"

BEATS_PER_BAR = 4

_RNG = random.Random()


class Decider(Protocol):
    def tonal_center(self) -> TonalCenter:
        """The run's tonal center, decided on first call and kept after."""
        ...

    def decide_bar(self, piece: Piece, bar_index: int) -> List[Decision]:
        """Decide bar ``bar_index``, for any ``bar_index >= 0``.

        A run is unbounded, so ``piece`` carries only the most recent bars as
        context, not necessarily bars 0..bar_index-1.  Do not mutate it.
        """
        ...

    def close(self) -> None:
        ...


def configured_model() -> str:
    return os.environ.get("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL


def _question_name(voice_name: str, slot: int) -> str:
    return f"{voice_name}_{slot:02d}"


def _blank_key(voice_name: str, slot: int) -> str:
    return f"bar_with_blank_{voice_name}_{slot:02d}"


def _beat_label(voice, slot: int) -> str:
    step = slot * voice.steps_per_slot
    beat = step // (SLOTS_PER_BAR // BEATS_PER_BAR)
    sixteenth = step % (SLOTS_PER_BAR // BEATS_PER_BAR)
    return f"beat {beat + 1}" + (f" + {sixteenth}/4" if sixteenth else "")


def _criteria(voice) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for option in OPTIONS:
        if option == REST:
            out[option] = REST_CRITERIA
        elif option == HOLD:
            out[option] = HOLD_CRITERIA
        else:
            out[option] = f"{voice.label_for_option(option)} sounds here."
    return out


def _voice_lines(piece: Piece) -> Dict[str, List[str]]:
    """The recent bars as one whitespace-joined token line per voice."""
    lines: Dict[str, List[str]] = {v.name: [] for v in VOICES}
    for bar in piece.bars:
        by_voice: Dict[str, Dict[int, Decision]] = {v.name: {} for v in VOICES}
        for d in bar:
            by_voice[d.voice][d.slot] = d
        for voice in VOICES:
            slots = by_voice[voice.name]
            tokens = [
                voice.label_for_option(slots[i].chosen)
                for i in range(voice.slots_per_bar)
            ]
            lines[voice.name].append(" ".join(tokens))
    return lines


def _sounding(piece: Piece) -> Dict[str, str]:
    """What each voice is carrying into this bar, so ``hold`` at slot 0 means
    something."""
    out: Dict[str, str] = {}
    for voice in VOICES:
        label = "nothing"
        for bar in piece.bars:
            for d in sorted(
                (d for d in bar if d.voice == voice.name), key=lambda d: d.slot
            ):
                if d.chosen == REST:
                    label = "nothing"
                elif d.chosen != HOLD:
                    label = voice.label_for_option(d.chosen)
        out[voice.name] = label
    return out


def _masked_bar(
    chosen: Dict[Tuple[str, int], str], voice_name: str, slot: int
) -> str:
    """The bar so far as one token line per voice, ``?`` at the asked slot."""
    lines = []
    for voice in VOICES:
        tokens = [
            "?"
            if voice.name == voice_name and i == slot
            else voice.label_for_option(chosen[(voice.name, i)])
            for i in range(voice.slots_per_bar)
        ]
        lines.append(f"{voice.name}: " + " ".join(tokens))
    return "\n".join(lines)


def _tonal_center_criteria() -> Dict[str, str]:
    return {name: f"{name} is the tonal center." for name in PITCH_CLASS_NAMES}


class JevDecider:
    """Decides each bar by mask-predict refinement over ``system_one`` calls."""

    def __init__(self, model: str | None = None, timeout: float = 120.0) -> None:
        self._model = model
        self._timeout = timeout
        self._client: Any | None = None
        self._choice: Any | None = None
        self._tonal_center: TonalCenter | None = None
        self.last_usage: Dict[str, Any] | None = None

    # The SDK is imported here so the demo package stays importable without it.
    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from dotenv import load_dotenv
        from typesafe_sdk import Choice, TypeSafeClient

        load_dotenv()
        self._client = TypeSafeClient(model=self._model, timeout=self._timeout)
        self._choice = Choice

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def tonal_center(self) -> TonalCenter:
        if self._tonal_center is not None:
            return self._tonal_center
        self._ensure_client()
        assert self._client is not None and self._choice is not None

        started = time.monotonic()
        response = self._client.system_one(
            state={"task": "Choose the tonal center of a piece of music about "
                           "to be written."},
            questions={
                TONAL_CENTER_QUESTION: self._choice(
                    instructions="Which of the twelve pitch classes is the "
                                 "tonal center?",
                    criteria=_tonal_center_criteria(),
                )
            },
        )
        seconds = time.monotonic() - started
        answer = response.answers[TONAL_CENTER_QUESTION]
        assert answer.type == "choice", answer.type
        distribution = _renormalise(answer.probabilities, PITCH_CLASS_NAMES)
        center = TonalCenter(
            pitch_class=PITCH_CLASS_NAMES.index(
                _sample(distribution, PITCH_CLASS_NAMES)
            ),
            confidence=float(answer.confidence),
            distribution=distribution,
        )
        self._tonal_center = center
        print(
            f"tonal center: {center.name} p={distribution[center.name]:.3f} "
            f"conf={center.confidence:.3f}, alternatives "
            f"{[(t['name'], t['p']) for t in center.top()][1:]}, "
            f"{seconds:.1f}s, {_tokens(response)}"
        )
        return center

    def state(self, piece: Piece, bar_index: int) -> Dict[str, Any]:
        lines = _voice_lines(piece)
        sounding = _sounding(piece)
        first_recent = max(0, bar_index - len(piece.bars))
        return {
            "task": "Write the next bar of music, one sixteenth at a time.",
            "tonal_center": self.tonal_center().name,
            "tempo_bpm": piece.tempo_bpm,
            "voices": [
                {
                    "name": v.name,
                    "range": (
                        f"{v.label_for_option(OPTIONS[0])} to "
                        f"{v.label_for_option(OPTIONS[PITCH_COUNT - 1])}"
                    ),
                    "slots_this_bar": v.slots_per_bar,
                    "sounding_at_bar_start": sounding[v.name],
                }
                for v in VOICES
            ],
            "bar_index": bar_index,
            "recent_bars": [
                {
                    "bar_index": first_recent + offset,
                    **{v.name: lines[v.name][offset] for v in VOICES},
                }
                for offset in range(len(piece.bars))
            ]
            or "This is the opening bar; nothing has sounded yet.",
        }

    def questions(self) -> Dict[str, Any]:
        self._ensure_client()
        assert self._choice is not None
        out: Dict[str, Any] = {}
        for voice in VOICES:
            criteria = _criteria(voice)
            for slot in range(voice.slots_per_bar):
                out[_question_name(voice.name, slot)] = self._choice(
                    instructions=(
                        f"Which option sounds in the {voice.name} at step "
                        f"{slot + 1} of {voice.slots_per_bar} in this bar "
                        f"({_beat_label(voice, slot)})?"
                    ),
                    criteria=criteria,
                )
        assert len(out) == DECISIONS_PER_BAR
        return out

    def _refinement_questions(self, slots: List[Tuple[str, int]]) -> Dict[str, Any]:
        self._ensure_client()
        assert self._choice is not None
        criteria_by_voice = {v.name: _criteria(v) for v in VOICES}
        out: Dict[str, Any] = {}
        for voice_name, slot in slots:
            key = _blank_key(voice_name, slot)
            out[_question_name(voice_name, slot)] = self._choice(
                instructions=(
                    f"{key} writes out this bar, one line per voice, in these "
                    f"same options. Exactly one position is hidden and marked "
                    f"?, in the {voice_name} line. Which option belongs there?"
                ),
                criteria=criteria_by_voice[voice_name],
            )
        assert len(out) <= MAX_QUESTIONS
        return out

    def decide_bar(self, piece: Piece, bar_index: int) -> List[Decision]:
        if bar_index < 0:
            raise IndexError(f"bar index must be non-negative, got {bar_index}")
        self._ensure_client()
        assert self._client is not None

        base_state = self.state(piece, bar_index)
        usage = {"input_tokens": 0, "output_tokens": 0}

        started = time.monotonic()
        response = self._client.system_one(
            state=base_state, questions=self.questions()
        )
        rough_seconds = time.monotonic() - started
        _add_usage(usage, response)

        chosen: Dict[Tuple[str, int], str] = {}
        answered: Dict[Tuple[str, int], Tuple[float, Dict[str, float]]] = {}
        open_slots: List[Tuple[str, int]] = []
        for voice in VOICES:
            for slot in range(voice.slots_per_bar):
                answer = response.answers[_question_name(voice.name, slot)]
                assert answer.type == "choice", (voice.name, slot, answer.type)
                distribution = _distribution(answer.probabilities)
                key = (voice.name, slot)
                answered[key] = (float(answer.confidence), distribution)
                peak = max(distribution, key=lambda o: distribution[o])
                if distribution[peak] >= FREEZE_CONFIDENCE:
                    chosen[key] = peak
                else:
                    chosen[key] = _sample(distribution)
                    open_slots.append(key)

        print(
            f"bar {bar_index} round 0 (rough): {len(open_slots)}/"
            f"{DECISIONS_PER_BAR} slots open, {_census(chosen)}, "
            f"{rough_seconds:.1f}s, {_tokens(response)}"
        )

        for round_index in range(1, REFINEMENT_ROUNDS):
            if not open_slots:
                print(f"bar {bar_index} round {round_index}: nothing open, stopping")
                break
            state = dict(base_state)
            for voice_name, slot in open_slots:
                state[_blank_key(voice_name, slot)] = _masked_bar(
                    chosen, voice_name, slot
                )
            started = time.monotonic()
            response = self._client.system_one(
                state=state, questions=self._refinement_questions(open_slots)
            )
            seconds = time.monotonic() - started
            _add_usage(usage, response)

            changed = 0
            for voice_name, slot in open_slots:
                answer = response.answers[_question_name(voice_name, slot)]
                assert answer.type == "choice", (voice_name, slot, answer.type)
                distribution = _distribution(answer.probabilities)
                key = (voice_name, slot)
                answered[key] = (float(answer.confidence), distribution)
                pick = _sample(distribution)
                changed += pick != chosen[key]
                chosen[key] = pick
            print(
                f"bar {bar_index} round {round_index}: asked {len(open_slots)}, "
                f"changed {changed}, {_census(chosen)}, {seconds:.1f}s, "
                f"{_tokens(response)}"
            )

        self.last_usage = {
            "model": response.model,
            "request_id": response.request_id,
            "rounds": REFINEMENT_ROUNDS,
            **usage,
        }

        decisions: List[Decision] = []
        for voice in VOICES:
            for slot in range(voice.slots_per_bar):
                key = (voice.name, slot)
                confidence, distribution = answered[key]
                decisions.append(
                    Decision(
                        bar=bar_index,
                        voice=voice.name,
                        slot=slot,
                        chosen=chosen[key],
                        confidence=confidence,
                        distribution=distribution,
                    )
                )
        return decisions


def _census(chosen: Dict[Tuple[str, int], str]) -> str:
    pitches = sum(1 for o in chosen.values() if o not in (REST, HOLD))
    rests = sum(1 for o in chosen.values() if o == REST)
    holds = sum(1 for o in chosen.values() if o == HOLD)
    distinct = len({o for o in chosen.values() if o not in (REST, HOLD)})
    return (
        f"pitch {pitches} ({distinct} distinct) rest {rests} hold {holds}"
    )


def _tokens(response: Any) -> str:
    return (
        f"in {response.usage.input_tokens} out {response.usage.output_tokens}"
    )


def _add_usage(usage: Dict[str, int], response: Any) -> None:
    usage["input_tokens"] += response.usage.input_tokens or 0
    usage["output_tokens"] += response.usage.output_tokens or 0


def _sample(distribution: Dict[str, float], keys: List[str] = OPTIONS) -> str:
    """Draw one option from the model's own distribution.

    Argmax collapses onto ``rest``: it holds the largest single share of the
    mass in most slots even where the pitches together hold more. Drawing
    honours the calibration instead of keeping only the winner.
    """
    threshold = _RNG.random()
    cumulative = 0.0
    for option in keys:
        cumulative += distribution[option]
        if cumulative >= threshold:
            return option
    return keys[-1]


def _renormalise(
    probabilities: Dict[str, float], keys: List[str]
) -> Dict[str, float]:
    """The answer's probabilities, renormalised.

    The API returns them summing to approximately 1; the dataclasses want
    exactly 1.
    """
    total = sum(probabilities[key] for key in keys)
    assert total > 0.0, "answer carried no probability mass"
    dist = {key: probabilities[key] / total for key in keys}
    # Absorb the float residual into the largest mass so the sum is exact.
    peak = max(dist, key=lambda k: dist[k])
    dist[peak] += 1.0 - sum(dist.values())
    return dist


def _distribution(probabilities: Dict[str, float]) -> Dict[str, float]:
    return _renormalise(probabilities, OPTIONS)


def make_decider(kind: str = "jev") -> Decider:
    if kind == "jev":
        return JevDecider()
    raise ValueError(f"unknown decider {kind!r}")
