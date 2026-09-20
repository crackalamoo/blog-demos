"""Core music representation.

Fixed 16th-note grid; a note has no duration field, it lasts as long as the
run of ``hold`` decisions after it.  One grid slot = one decision over a
26-option vocabulary: ``n0``..``n23`` (chromatic, relative to the voice's
``base_midi``), ``rest``, ``hold``.  Options being voice-relative is what
lets one 26-way head serve both voices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

SCHEMA_VERSION = "jev.piece/v1"

PITCH_COUNT = 24
REST = "rest"
HOLD = "hold"

#: Canonical order; the front end relies on it for stable indices.
OPTIONS: List[str] = [f"n{i}" for i in range(PITCH_COUNT)] + [REST, HOLD]
assert len(OPTIONS) == 26

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
assert len(PITCH_CLASS_NAMES) == 12

TICKS_PER_BEAT = 480          # MIDI resolution
SLOTS_PER_BAR = 16            # 16th notes in a 4/4 bar
TICKS_PER_SLOT = TICKS_PER_BEAT // 4


@dataclass(frozen=True)
class Voice:
    name: str
    slots_per_bar: int
    base_midi: int
    midi_program: int
    midi_channel: int

    @property
    def steps_per_slot(self) -> int:
        """Grid steps (16ths) covered by one slot of this voice."""
        return SLOTS_PER_BAR // self.slots_per_bar

    def midi_for_option(self, option: str) -> int | None:
        if option in (REST, HOLD):
            return None
        return self.base_midi + int(option[1:])

    def label_for_option(self, option: str) -> str:
        if option == REST:
            return "rest"
        if option == HOLD:
            return "hold"
        midi = self.midi_for_option(option)
        assert midi is not None
        return f"{PITCH_CLASS_NAMES[midi % 12]}{midi // 12 - 1}"


#: Bases put each voice's two-octave window where it sings: A4, A3.
VOICES: List[Voice] = [
    Voice("melody", slots_per_bar=16, base_midi=69, midi_program=0, midi_channel=0),
    Voice("harmony", slots_per_bar=8, base_midi=57, midi_program=48, midi_channel=1),
]
VOICES_BY_NAME: Dict[str, Voice] = {v.name: v for v in VOICES}

DECISIONS_PER_BAR = sum(v.slots_per_bar for v in VOICES)
assert DECISIONS_PER_BAR == 24


@dataclass
class TonalCenter:
    """Decided once per run and then carried, unchanged, in every bar's state."""

    pitch_class: int
    confidence: float
    distribution: Dict[str, float]

    def __post_init__(self) -> None:
        if not 0 <= self.pitch_class < 12:
            raise ValueError(f"pitch class {self.pitch_class!r} out of range")
        missing = set(PITCH_CLASS_NAMES) - set(self.distribution)
        if missing:
            raise ValueError(f"distribution missing {sorted(missing)}")
        total = sum(self.distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"distribution sums to {total!r}, not 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence!r} out of range")

    @property
    def name(self) -> str:
        return PITCH_CLASS_NAMES[self.pitch_class]

    def top(self, n: int = 4) -> List[Dict[str, object]]:
        ranked = sorted(self.distribution.items(), key=lambda kv: -kv[1])[:n]
        return [{"name": name, "p": round(p, 5)} for name, p in ranked]

    def to_json(self) -> Dict[str, object]:
        return {
            "pitch_class": self.pitch_class,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "distribution": {k: round(v, 5) for k, v in self.distribution.items()},
            "top": self.top(),
        }


@dataclass
class Decision:
    bar: int
    voice: str
    slot: int
    chosen: str
    confidence: float
    distribution: Dict[str, float]

    def __post_init__(self) -> None:
        if self.chosen not in OPTIONS:
            raise ValueError(f"unknown option {self.chosen!r}")
        missing = set(OPTIONS) - set(self.distribution)
        if missing:
            raise ValueError(f"distribution missing {len(missing)} options")
        total = sum(self.distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"distribution sums to {total!r}, not 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence!r} out of range")

    @property
    def step(self) -> int:
        """Offset in 16th-note steps within the bar (bar-relative)."""
        return self.slot * VOICES_BY_NAME[self.voice].steps_per_slot

    @property
    def abs_step(self) -> int:
        """16th-note steps from the start of the piece.

        Anything laid out along a timeline wants this, not ``step``, which is
        bar-relative.
        """
        return self.bar * SLOTS_PER_BAR + self.step

    def top(self, n: int = 4) -> List[Dict[str, object]]:
        voice = VOICES_BY_NAME[self.voice]
        ranked = sorted(self.distribution.items(), key=lambda kv: -kv[1])[:n]
        return [
            {"option": opt, "label": voice.label_for_option(opt), "p": round(p, 5)}
            for opt, p in ranked
        ]

    def to_json(self) -> Dict[str, object]:
        voice = VOICES_BY_NAME[self.voice]
        return {
            "bar": self.bar,
            "voice": self.voice,
            "slot": self.slot,
            "step": self.step,
            "abs_step": self.abs_step,
            "chosen": self.chosen,
            "label": voice.label_for_option(self.chosen),
            "midi": voice.midi_for_option(self.chosen),
            "confidence": round(self.confidence, 4),
            "distribution": {k: round(v, 5) for k, v in self.distribution.items()},
            "top": self.top(),
        }


class NoteResolver:
    """Turns a stream of bars of decisions into notes, incrementally.

    A note lasts as long as the run of ``hold`` decisions after it, and that run
    can cross bar lines, so its duration is not final when its own bar closes.
    ``feed`` therefore returns a note as soon as it starts, with the duration
    known so far, and returns it again -- same ``id``, larger ``steps`` --
    from every later bar that extends it.  A repeat is a correction, not a new
    note; consumers key on ``id``.
    """

    def __init__(self) -> None:
        self._open: Dict[str, Dict[str, object] | None] = {v.name: None for v in VOICES}
        self._next_id = 0

    def feed(self, decisions: Sequence[Decision]) -> List[Dict[str, object]]:
        by_voice: Dict[str, List[Decision]] = {v.name: [] for v in VOICES}
        for d in decisions:
            by_voice[d.voice].append(d)

        touched: Dict[int, Dict[str, object]] = {}
        for voice in VOICES:
            for d in sorted(by_voice[voice.name], key=lambda d: d.slot):
                start = d.abs_step
                if d.chosen == HOLD:
                    open_note = self._open[voice.name]
                    if open_note is not None:
                        open_note["steps"] = (
                            start + voice.steps_per_slot - int(open_note["start"])
                        )
                        touched[int(open_note["id"])] = open_note
                    continue
                self._open[voice.name] = None
                if d.chosen == REST:
                    continue
                note = {
                    "id": self._next_id,
                    "voice": voice.name,
                    "midi": voice.midi_for_option(d.chosen),
                    "label": voice.label_for_option(d.chosen),
                    "start": start,
                    "steps": voice.steps_per_slot,
                    "confidence": round(d.confidence, 4),
                }
                self._next_id += 1
                self._open[voice.name] = note
                touched[self._next_id - 1] = note

        # Copies: the internal notes keep being mutated by later bars.
        out = [dict(note) for note in touched.values()]
        out.sort(key=lambda n: (n["start"], n["voice"]))
        return out


@dataclass
class Piece:
    tempo_bpm: int = 92
    bars: List[List[Decision]] = field(default_factory=list)
    tonal_center: TonalCenter | None = None

    def add_bar(self, decisions: Sequence[Decision]) -> None:
        if len(decisions) != DECISIONS_PER_BAR:
            raise ValueError(
                f"a bar needs {DECISIONS_PER_BAR} decisions, got {len(decisions)}"
            )
        self.bars.append(list(decisions))

    def decision(self, bar: int, voice: str, slot: int) -> Decision:
        for d in self.bars[bar]:
            if d.voice == voice and d.slot == slot:
                return d
        raise KeyError((bar, voice, slot))

    def notes(self) -> List[Dict[str, object]]:
        """Resolve the grid into notes; times are in 16th-note steps."""
        resolver = NoteResolver()
        latest: Dict[int, Dict[str, object]] = {}
        for bar in self.bars:
            for note in resolver.feed(bar):
                latest[int(note["id"])] = note
        out = [{k: v for k, v in n.items() if k != "id"} for n in latest.values()]
        out.sort(key=lambda n: (n["start"], n["voice"]))
        return out

    def to_json(self) -> Dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "tempo_bpm": self.tempo_bpm,
            "time_signature": "4/4",
            "ticks_per_beat": TICKS_PER_BEAT,
            "slots_per_bar": SLOTS_PER_BAR,
            "decisions_per_bar": DECISIONS_PER_BAR,
            "vocabulary": {
                "options": OPTIONS,
                "pitch_options": OPTIONS[:PITCH_COUNT],
                "special_options": [REST, HOLD],
            },
            "voices": [
                {
                    "name": v.name,
                    "slots_per_bar": v.slots_per_bar,
                    "steps_per_slot": v.steps_per_slot,
                    "base_midi": v.base_midi,
                    "midi_program": v.midi_program,
                    "midi_channel": v.midi_channel,
                }
                for v in VOICES
            ],
            "bars": [
                {"index": i, "decisions": [d.to_json() for d in bar]}
                for i, bar in enumerate(self.bars)
            ],
            "tonal_center": self.tonal_center.to_json() if self.tonal_center else None,
            "notes": self.notes(),
        }
