"""CLI for the rotation stage: `python -m src.rotation <dataset> [options]`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config.loader import load_config
from src.courtmodel.model import FRONT_ROW_SLOTS
from src.dataset.io import read_dataset
from src.rotation.classifier import RotationClassifier, RotationError


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(
        prog="python -m src.rotation",
        description="Assign rotation slots to the analysed side of one clip.",
    )
    parser.add_argument("dataset", type=Path, help="dataset JSON file to read")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="config file to read thresholds from",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON file to write the result to",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Classify one dataset file and report. Exit 0 resolved, 1 not."""
    args = _parse_args(argv)
    try:
        dataset = read_dataset(args.dataset)
        classifier = RotationClassifier(load_config(args.config))
        result = classifier.classify(dataset)
    except (RotationError, OSError, ValueError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1

    side = "none" if result.side is None else result.side.value
    print(f"clip {dataset.clip.id}: analysed side {side}")
    if result.is_resolved:
        print(f"confidence {result.confidence:.3f}")
        for track_id in result.assigned_tracks():
            slot = result.slots[track_id]
            row = "front" if slot in FRONT_ROW_SLOTS else "back"
            print(f"  slot {slot}: track {track_id}  {row}")
    else:
        print("unresolved")
    for note in result.notes:
        print(f"  note: {note}")

    if args.output is not None:
        payload = {
            "clipId": dataset.clip.id,
            "side": None if result.side is None else result.side.value,
            "slots": {str(key): value for key, value in sorted(result.slots.items())},
            "confidence": result.confidence,
            "isResolved": result.is_resolved,
            "notes": list(result.notes),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}")

    return 0 if result.is_resolved else 1


if __name__ == "__main__":
    raise SystemExit(main())