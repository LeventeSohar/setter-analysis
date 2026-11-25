"""The command line for clip intake.

``main`` lives here rather than in ``__main__.py`` so that the tests can call
it with an argv list instead of spawning a subprocess, and rather than in
``clip.py`` because it is the composition root: this is the module that imports
the config loader and chooses the OpenCV source. ``clip.py`` itself imports
neither, which is what keeps NFR-R6 and NFR-R8 intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.clip.clip import ClipError, RallyClip
from src.config.loader import DEFAULT_CONFIG_PATH, ConfigError, load_config

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_UNUSABLE = 2


def describe(clip: RallyClip, *, decoded_frames: int | None = None) -> dict[str, Any]:
    """Return a summary of a clip: its metadata, and what validation found."""
    metadata = clip.metadata()
    properties = clip.properties()
    validation = clip.validate()
    summary: dict[str, Any] = {
        "clipId": metadata.id,
        "sourceFile": metadata.source_file,
        "duration": round(metadata.duration, 3),
        "frameRate": metadata.frame_rate,
        "resolution": {
            "width": metadata.resolution.width,
            "height": metadata.resolution.height,
        },
        "headerFrameCount": properties.frame_count,
        "isUsable": validation.is_usable,
        "notes": list(validation.notes),
    }
    if decoded_frames is not None:
        summary["decodedFrameCount"] = decoded_frames
    return summary


def main(argv: list[str] | None = None) -> int:
    """Open a clip, print its metadata and validation, and report by exit code.

    Exit 0 when the clip opened and is plausibly a rally, 1 when it could not
    be opened or the config could not be read, and 2 when it opened but failed
    validation. The third code exists because an unusable clip is a finding,
    not a crash, and a batch script needs to tell the two apart.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.clip",
        description="Read a rally clip's metadata and check it is plausibly one rally.",
    )
    parser.add_argument("clip", type=Path, help="path to the rally clip to read")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"config file to read thresholds from (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the summary here as JSON instead of to stdout",
    )
    parser.add_argument(
        "--count-frames",
        action="store_true",
        help="decode the whole clip and report how many frames actually arrived",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    try:
        clip = RallyClip(args.clip, config)
        decoded = sum(1 for _ in clip.frames()) if args.count_frames else None
        summary = describe(clip, decoded_frames=decoded)
    except ClipError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    text = json.dumps(summary, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")

    if not summary["isUsable"]:
        for note in summary["notes"]:
            print(f"warning: {note}", file=sys.stderr)
        return EXIT_UNUSABLE
    return EXIT_OK