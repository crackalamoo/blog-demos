"""Where the two models plug in.

Each function below is the whole of its own plug point.  ``choose_action``
is one Jev ``Choice`` over the enumerated actions: the state as it is sent,
and the legal actions with their descriptions as the criteria.
``choose_objective`` is the ChatGPT call over the recent-action log.

Neither of them is shown a milestone, a route, a plan or a coordinate.  The
bot is playing a world generated from a fresh random seed and can only see
what it can see.
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from typesafe_sdk import Choice, TypeSafeClient

load_dotenv()

JEV_MODEL = "jev-latest"
CHAT_MODEL = "gpt-5.6-luna"

#: One decision is one question.  Batching collapses the accuracy.
ACTION_INSTRUCTIONS = (
    "You are playing this Minecraft character. The state is everything it "
    "can see right now. Each option is an action the game will let it take "
    "this instant. Choose the one to take now."
)

#: The API's ceiling on options in a single Choice.
MAX_CRITERIA = 255

#: gpt-5.6-luna rejects "minimal"; it accepts none/low/medium/high/xhigh.
CHAT_REASONING_EFFORT = "low"

#: Reasoning tokens are billed against ``max_completion_tokens``, so the cap
#: has to clear the thinking as well as the one-line objective.
OBJECTIVE_MAX_OUTPUT_TOKENS = 512

#: Decisions between objectives.  A death re-sets the objective immediately and
#: restarts the count.
DECISIONS_PER_OBJECTIVE = 32

#: Recent actions, for the objective call only.  It is deliberately not part of
#: the per-decision payload: that payload is a snapshot plus one last_action,
#: and nothing in it accumulates.
LOG_LENGTH = 64

OBJECTIVE_SYSTEM = (
    "You are setting the current objective for a bot playing Minecraft "
    "survival. Its world was generated from a random seed that neither of "
    "you has seen. "
    "You will be given the bot's situation right now, and its recent "
    "actions and their outcomes, oldest first. Reply with one objective "
    "in the imperative, under ten words, "
    "saying what the bot should try to accomplish next. Base it on what the "
    "log shows actually happened, including what failed. If its recent "
    "actions show it making a mistake, aim the objective at correcting "
    "that. An empty log means the bot has just spawned."
)

OBJECTIVE_SCHEMA = {
    "name": "objective",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["objective"],
        "properties": {"objective": {"type": "string"}},
    },
}

#: The key the objective travels under in the per-decision payload.  Jev is
#: otherwise stateless across decisions; this is the one thread of continuity
#: it is given.  Who wrote the objective is for the page to say, not for the
#: state to carry, so the key names the thing and not its author.
OBJECTIVE_KEY = "current_objective"

_CLIENT: Any = None
_JEV: Any = None


class ActionLog:
    def __init__(self, length: int = LOG_LENGTH) -> None:
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=length)

    def record(self, entry: Dict[str, Any]) -> None:
        self._entries.append(entry)

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)


def choose_objective(log: ActionLog, state: Dict[str, Any]) -> str:
    """One ChatGPT call over the recent-action log.

    The static prompt is the whole prefix and the log is the last message,
    so the cached prefix is identical on every call.
    """
    messages = [
        {"role": "system", "content": OBJECTIVE_SYSTEM},
        {"role": "user", "content": json.dumps(
            {"situation": state, "recent_actions": log.entries()},
            indent=2)},
    ]
    response = _client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        max_completion_tokens=OBJECTIVE_MAX_OUTPUT_TOKENS,
        reasoning_effort=CHAT_REASONING_EFFORT,
        response_format={
            "type": "json_schema", "json_schema": OBJECTIVE_SCHEMA},
    )
    choice = response.choices[0]
    raw = choice.message.content or ""
    detail = (f"finish_reason={choice.finish_reason!r} "
              f"usage={response.usage!r}")
    if not raw.strip():
        raise RuntimeError(f"{CHAT_MODEL} returned no content: {detail}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{CHAT_MODEL} returned malformed JSON ({exc}): {detail} "
            f"raw={raw!r}"
        ) from exc
    objective = payload.get("objective") if isinstance(payload, dict) else None
    if not isinstance(objective, str) or not objective.strip():
        raise RuntimeError(
            f"{CHAT_MODEL} returned no objective: {detail} "
            f"payload={payload!r}"
        )
    return objective.strip()


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(api_key=_require("OPENAI_API_KEY"))
    return _CLIENT


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set; put it in .env (see .env.example)")
    return value


@dataclass(frozen=True)
class Decision:
    label: str
    confidence: float
    #: Keyed by action label, summing to about 1.
    probabilities: Dict[str, float]
    input_tokens: int


def choose_action(
    state: Dict[str, Any], actions: List[Dict[str, Any]]
) -> Decision:
    if not actions:
        raise ValueError("the bot enumerated no actions; wait is always legal")
    if len(actions) > MAX_CRITERIA:
        raise ValueError(
            f"{len(actions)} actions exceeds the {MAX_CRITERIA} a single "
            f"Choice takes")
    keys = {f"a{i}": a["label"] for i, a in enumerate(actions)}
    criteria = {f"a{i}": a["description"] for i, a in enumerate(actions)}
    response = _jev().system_one(
        state=state,
        questions={"action": Choice(
            instructions=ACTION_INSTRUCTIONS,
            criteria=criteria,
        )},
        model=JEV_MODEL,
    )
    answer = response.answers["action"]
    assert answer.type == "choice", answer.type
    return Decision(
        label=keys[answer.choice],
        confidence=answer.confidence,
        probabilities={keys[k]: p for k, p in answer.probabilities.items()},
        input_tokens=response.usage.input_tokens,
    )


def _jev() -> TypeSafeClient:
    global _JEV
    if _JEV is None:
        _require("TYPESAFE_API_KEY")
        _JEV = TypeSafeClient()
    return _JEV
