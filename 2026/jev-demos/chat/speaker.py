"""The two speakers.

ChatGPT writes its own turn and, in the same call, a handful of replies Jev
could give; Jev answers with one ``Choice`` over exactly those replies.  Jev's
input is not restricted: it is given ChatGPT's turns verbatim.

The knowledge is deliberately asymmetric: what is true about Jev lives in Jev's
state and nowhere in ChatGPT's prompt, so ChatGPT has to guess and Jev can pick
the option that corrects it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

GPT = "gpt"
JEV = "jev"

JEV_MODEL = "jev-latest"
CHAT_MODEL = "gpt-5.6-luna"

#: Reasoning tokens are billed against ``max_completion_tokens``, so the cap has
#: to clear the thinking, the visible turn and the candidate list; turn length is
#: held down by CHAT_SYSTEM asking for one or two sentences, not by the budget.
CHAT_MAX_OUTPUT_TOKENS = 768

#: gpt-5.6-luna rejects "minimal"; it accepts none/low/medium/high/xhigh.
CHAT_REASONING_EFFORT = "low"

MIN_CANDIDATES = 3
MAX_CANDIDATES = 10

CHAT_SYSTEM = (
    "You are ChatGPT, talking with Jev, a classification model made by "
    "TypeSafe AI. Jev cannot write text of its own: it can only choose one "
    "option from a list you give it. You are curious about it. "
    "In `turn`, say one or two sentences to Jev. In `candidates`, write three "
    "to ten short replies it could give, in its own voice; err toward more. "
    "That list is the whole of what Jev can say, so make every option a real "
    "position it might hold -- concrete, and including ones that contradict "
    "you or turn a question back on you. Never offer Jev a way out: an "
    "option that declines to judge, or answers \"it depends\", is the one it "
    "will always take, and its hesitation already shows in the probabilities. "
    "Keep the options disjoint: no two should overlap or blend together. "
    "Ask about anything you would like to hear Jev judge -- whether a hot dog "
    "is a sandwich, which of two songs is better, what it is itself -- "
    "favoring questions people genuinely disagree about, and moving on to a "
    "new one every few turns. It cannot describe its own internals: its "
    "training cutoff predates its release. "
    "After each turn you see Jev's confidence and the probability it gave "
    "every option. Those are real measurements of its uncertainty."
)

CHAT_SCHEMA = {
    "name": "chat_turn",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["turn", "candidates"],
        "properties": {
            "turn": {"type": "string"},
            "candidates": {"type": "array", "items": {"type": "string"}},
        },
    },
}


@dataclass(frozen=True)
class Message:
    speaker: str
    text: str
    #: For Jev's turns, the measured uncertainty behind the choice, shown back
    #: to ChatGPT on its next call.  ``None`` for ChatGPT's own turns.
    measurement: Optional[str] = None


@dataclass(frozen=True)
class Proposal:
    """ChatGPT's turn and the replies it offers Jev."""

    turn: str
    candidates: List[str]


@dataclass(frozen=True)
class JevChoice:
    """One classification: the chosen reply, the full distribution, confidence."""

    chosen: str
    confidence: float
    distribution: Dict[str, float]

    def ranked(self) -> List[Dict[str, object]]:
        order = sorted(self.distribution, key=lambda c: -self.distribution[c])
        return [{"text": c, "p": round(self.distribution[c], 5)} for c in order]

    def measurement(self) -> str:
        lines = [
            f"  {c['p']:.2f}  {c['text']}" + ("   <- chosen" if c["text"] == self.chosen else "")
            for c in self.ranked()
        ]
        return (
            "Jev's classifier on that turn: confidence "
            f"{self.confidence:.2f}. Probability it gave each option you "
            "offered:\n" + "\n".join(lines)
        )


class LiveJev:
    """One ``system_one`` call per turn: a Choice over ChatGPT's candidates."""

    def __init__(self, client: TypeSafeClient) -> None:
        self._client = client

    def choose(self, history: Sequence[Message], candidates: Sequence[str]) -> JevChoice:
        # Labels keep the wire keys short and stable; the reply text is the
        # criterion, which is what Jev actually judges.
        labels = {_label(i): text for i, text in enumerate(candidates)}
        response = self._client.system_one(
            state=self._state(history),
            questions={"reply": Choice(
                instructions="Choose the reply you want to give.",
                criteria=labels,
            )},
            model="jev-latest",
        )
        answer = response.answers["reply"]
        assert answer.type == "choice", answer.type
        distribution = {labels[k]: p for k, p in answer.probabilities.items()}
        return JevChoice(labels[answer.choice], answer.confidence, distribution)

    def _state(self, history: Sequence[Message]) -> dict:
        return {
            "who_you_are": (
                "You are Jev, a classification model made by TypeSafe AI. You "
                "classify; you never generate text, and you speak only by "
                "choosing one of the replies you are offered. What you return "
                "is a probability over each of them, plus a confidence."
            ),
            "who_you_are_talking_to": (
                "ChatGPT. You read its turns in full. It is curious about you "
                "and will guess about you, sometimes wrongly -- when one of "
                "the replies corrects it, that is the one worth choosing."
            ),
            "conversation_so_far": [
                {"speaker": m.speaker, "said": m.text} for m in history
            ],
        }

class LiveChat:
    """One streamed call per turn, yielding ChatGPT's turn as it is written."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def propose(
        self, history: Sequence[Message], on_delta: Callable[[str], None]
    ) -> Proposal:
        stream = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=self._messages(history),
            max_completion_tokens=CHAT_MAX_OUTPUT_TOKENS,
            reasoning_effort=CHAT_REASONING_EFFORT,
            response_format={"type": "json_schema", "json_schema": CHAT_SCHEMA},
            stream=True,
            stream_options={"include_usage": True},
        )
        raw = ""
        turn = _TurnStream()
        finish_reason = None
        usage = None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue            # a usage-only final chunk carries no choice
            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
            delta = choice.delta.content
            if delta:
                raw += delta
                visible = turn.feed(delta)
                if visible:
                    on_delta(visible)
        return self._parse(raw, finish_reason, usage)

    def _parse(self, raw: str, finish_reason, usage) -> Proposal:
        detail = f"finish_reason={finish_reason!r} usage={usage!r}"
        if not raw.strip():
            raise RuntimeError(f"{CHAT_MODEL} returned no content: {detail}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{CHAT_MODEL} returned malformed JSON ({exc}): {detail} "
                f"raw={raw!r}"
            ) from exc
        turn = payload["turn"] if isinstance(payload, dict) else None
        if not isinstance(turn, str) or not turn.strip():
            raise RuntimeError(
                f"{CHAT_MODEL} returned no turn: {detail} payload={payload!r}")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not all(
            isinstance(c, str) and c.strip() for c in candidates
        ):
            raise RuntimeError(
                f"{CHAT_MODEL} returned unusable candidates: {detail} "
                f"candidates={candidates!r}"
            )
        candidates = [c.strip() for c in candidates]
        if not MIN_CANDIDATES <= len(candidates) <= MAX_CANDIDATES:
            raise RuntimeError(
                f"{CHAT_MODEL} returned {len(candidates)} candidates, not "
                f"{MIN_CANDIDATES}..{MAX_CANDIDATES}: {candidates!r}"
            )
        if len(set(candidates)) != len(candidates):
            raise RuntimeError(
                f"{CHAT_MODEL} repeated a candidate: {candidates!r}")
        return Proposal(turn.strip(), candidates)

    def _messages(self, history: Sequence[Message]) -> List[dict]:
        messages: List[dict] = [{"role": "system", "content": CHAT_SYSTEM}]
        for message in history:
            role = "assistant" if message.speaker == GPT else "user"
            messages.append({"role": role, "content": message.text})
            if message.measurement:
                messages.append(
                    {"role": "developer", "content": message.measurement})
        return messages


class _TurnStream:
    """The ``turn`` string decoded from the JSON payload as it arrives.

    The page streams ChatGPT's turn while the candidates are still being
    written, so the visible text is read off the incomplete JSON rather than
    waiting for the object to close.  ``turn`` is the schema's first property,
    so it is complete before the candidates begin.
    """

    def __init__(self) -> None:
        self._raw = ""
        self._start = -1
        self._sent = 0

    def feed(self, delta: str) -> str:
        self._raw += delta
        if self._start < 0:
            key = self._raw.find('"turn"')
            if key < 0:
                return ""
            colon = self._raw.find(":", key + 6)
            if colon < 0:
                return ""
            quote = self._raw.find('"', colon + 1)
            if quote < 0:
                return ""
            self._start = quote + 1
        text = json.loads(f'"{_json_string_body(self._raw[self._start:])}"')
        fresh = text[self._sent:]
        self._sent = len(text)
        return fresh


def _json_string_body(raw: str) -> str:
    """The longest decodable prefix of a JSON string's body, escapes intact."""
    out: List[str] = []
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == '"':
            break
        if char == "\\":
            width = 6 if raw[i + 1:i + 2] == "u" else 2
            if i + width > len(raw):
                break
            out.append(raw[i:i + width])
            i += width
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _label(index: int) -> str:
    return f"c{index}"


def jev_client() -> TypeSafeClient:
    _require("TYPESAFE_API_KEY")
    return TypeSafeClient()


def chat_client() -> OpenAI:
    return OpenAI(api_key=_require("OPENAI_API_KEY"))


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set; put it in .env (see .env.example)")
    return value
