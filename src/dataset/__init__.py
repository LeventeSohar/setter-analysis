"""the dataset file: schema definition, reader and writer"""

from src.dataset.io import (
    DatasetError,
    MalformedDatasetError,
    UnsupportedSchemaVersionError,
    describe,
    read_dataset,
    write_dataset,
)
from src.dataset.schema import (
    SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    Calibration,
    ClipMetadata,
    CourtSide,
    Dataset,
    Frame,
    Player,
    PlayerObservation,
    PlayerRole,
    Point,
    Rect,
    Size,
    Team,
)

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "Calibration",
    "ClipMetadata",
    "CourtSide",
    "Dataset",
    "DatasetError",
    "Frame",
    "MalformedDatasetError",
    "Player",
    "PlayerObservation",
    "PlayerRole",
    "Point",
    "Rect",
    "Size",
    "Team",
    "UnsupportedSchemaVersionError",
    "describe",
    "read_dataset",
    "write_dataset",
]