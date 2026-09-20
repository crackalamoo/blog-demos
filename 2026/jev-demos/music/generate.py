#!/usr/bin/env python3
"""Generate the piece offline: writes out/piece.json as a record of a run.

A live run is endless; this takes a bar count so the record is finite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run directly (`python3 generate.py`) there is no package context, so give
# ourselves one; the demo's modules import each other relatively.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "music"

from .decider import Decider, make_decider
from .grid import Piece

OUT_DIR = Path(__file__).resolve().parent / "out"

DEFAULT_BARS = 8        # two four-bar phrases


def build_piece(
    decider: Decider | None = None,
    tempo_bpm: int = 92,
    bars: int = DEFAULT_BARS,
) -> Piece:
    """Bar N sees the bars before it as context."""
    own = decider is None
    decider = decider or make_decider()
    piece = Piece(tempo_bpm=tempo_bpm, tonal_center=decider.tonal_center())
    try:
        for bar_index in range(bars):
            piece.add_bar(decider.decide_bar(piece, bar_index))
    finally:
        if own:
            decider.close()
    return piece


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=int, default=DEFAULT_BARS)
    ap.add_argument("--tempo", type=int, default=92)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    piece = build_piece(tempo_bpm=args.tempo, bars=args.bars)
    args.out.mkdir(parents=True, exist_ok=True)

    json_path = args.out / "piece.json"
    json_path.write_text(json.dumps(piece.to_json(), indent=2) + "\n")

    assert piece.tonal_center is not None
    print(f"tonal_center={piece.tonal_center.name}")
    print(f"bars={len(piece.bars)} decisions={sum(len(b) for b in piece.bars)} "
          f"notes={len(piece.notes())}")
    print(f"wrote {json_path} ({json_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
