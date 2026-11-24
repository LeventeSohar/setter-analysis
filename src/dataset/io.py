"""reading and writing dataset files
the dataset file is the interface between clip level and contact elvel pipelines.
JSOn is used because the data is nested and ragged, and because a documented text format can be opened and checked manually
every file has the schema version
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .schema import SUPPORTED_SCHEMA_VERSIONS, Dataset

class DatasetError(Exception):
    """Base class for every error raised while reading or writing a dataset.

    Deliberately plain: it formats no message of its own. A base class that
    builds one subclass's message makes every other subclass lie.
    """

class UnsupportedSchemaVersionError(DatasetError):
    """Raised when a file declares a schema version this code cannot read."""

    def __init__(self, found: str) -> None:
        """Record the offending version and build a message naming it."""
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        super().__init__(
            f"dataset file declares schema version {found!r}; "
            f"this code supports: {supported}"
        )
        self.found = found

class MalformedDatasetError(DatasetError):
    """raised when i file is not a valid JSON, or its missing required fileds"""

def write_dataset(dataset: Dataset, path: Path | str, *, indent: int = 2) -> Path:
    """Write a dataset to ``path`` as JSON and return the path written.

    Parent directories are created if they do not exist. ``indent`` controls
    readability only; it has no effect on what is read back.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dataset.to_dict()
    destination.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination

def read_dataset(path: Path | str) -> Dataset:
    """Read a dataset from ``path``.

    Raises ``UnsupportedSchemaVersionError`` if the file declares a schema
    version this code does not support, and ``MalformedDatasetError`` if the
    file is not valid JSON or is missing required fields. The version is
    checked before any other field is touched.
    """
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MalformedDatasetError(f"{source} is not valid JSON: {error}") from error

    if not isinstance(raw, dict):
        raise MalformedDatasetError(f"{source} does not contain a JSON object")

    version = raw.get("schemaVersion")
    if version is None:
        raise MalformedDatasetError(f"{source} has no schemaVersion field")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise UnsupportedSchemaVersionError(str(version))

    try:
        return Dataset.from_dict(raw)
    except (KeyError, TypeError, ValueError) as error:
        raise MalformedDatasetError(
            f"{source} does not match schema {version}: {error}"
        ) from error

def describe(dataset: Dataset) -> dict[str, Any]:
    """Return a small summary of a dataset, for CLI output and logging."""
    analysed = dataset.analysed_team()
    return {
        "schemaVersion": dataset.schema_version,
        "clipId": dataset.clip.id,
        "frames": len(dataset.frames),
        "observations": len(dataset.observations),
        "players": len(dataset.players),
        "teams": len(dataset.teams),
        "analysedSide": None if analysed is None else analysed.side.value,
        "framesWithBall": sum(
            1 for frame in dataset.frames if frame.ball_position is not None
        ),
    }

def main(argv: list[str] | None = None) -> int:
    """Validate a dataset file from the command line.

    Reads the file, applies the version check and prints a summary. Returns 0
    when the file is readable and 1 when it is not.
    """
    parser = argparse.ArgumentParser(
        description="Validate and summarise a dataset file."
    )
    parser.add_argument("input", type=Path, help="path to the dataset file to read")
    arguments = parser.parse_args(argv)

    try:
        dataset = read_dataset(arguments.input)
    except DatasetError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(json.dumps(describe(dataset), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())