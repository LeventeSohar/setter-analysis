"""Tests for the dataset schema, reader and writer.
 
These run with no video, no model weights and no other pipeline stage
(NFR-R8). Every fixture is built by hand in this file.
"""
 
from __future__ import annotations
 
import json
from pathlib import Path
 
import pytest
 
from src.dataset.io import (
    MalformedDatasetError,
    UnsupportedSchemaVersionError,
    describe,
    read_dataset,
    write_dataset,
)
from src.dataset.schema import (
    SCHEMA_VERSION,
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
 
 
def build_dataset() -> Dataset:
    """Build a small but complete dataset covering every record type."""
    return Dataset(
        clip=ClipMetadata(
            id="match01_rally0007",
            source_file="match01.mp4",
            duration=12.0,
            frame_rate=60.0,
            resolution=Size(width=2898, height=1870),
        ),
        calibration=Calibration(
            transform=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            court_lines=[(Point(0.0, 0.0), Point(100.0, 0.0))],
            confidence=0.82,
        ),
        frames=[
            Frame(index=0, timestamp=0.0, ball_position=Point(12.5, 40.0)),
            Frame(index=1, timestamp=1 / 60, ball_position=None),
        ],
        observations=[
            PlayerObservation(
                frame_index=0,
                track_id=3,
                bounding_box=Rect(x=100.0, y=200.0, width=40.0, height=90.0),
                court_position=Point(4.5, 6.0),
            ),
            PlayerObservation(
                frame_index=1,
                track_id=3,
                bounding_box=Rect(x=104.0, y=202.0, width=40.0, height=90.0),
                court_position=Point(4.6, 6.1),
            ),
            PlayerObservation(
                frame_index=0,
                track_id=7,
                bounding_box=Rect(x=800.0, y=210.0, width=38.0, height=88.0),
                court_position=Point(3.0, 12.5),
            ),
        ],
        players=[
            Player(track_id=3, team=CourtSide.NEAR, role=PlayerRole.SETTER),
            Player(track_id=7, team=CourtSide.FAR, role=PlayerRole.OTHER),
            Player(track_id=9, team=None, role=None),
        ],
        teams=[
            Team(side=CourtSide.NEAR, is_analysed=True),
            Team(side=CourtSide.FAR, is_analysed=False),
        ],
    )
 
 
def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    """Writing a dataset and reading it back returns an equal dataset."""
    original = build_dataset()
    path = write_dataset(original, tmp_path / "clip.json")
 
    restored = read_dataset(path)
 
    assert restored == original
 
 
def test_written_file_declares_current_schema_version(tmp_path: Path) -> None:
    """The version field is present on disk and matches the module constant."""
    path = write_dataset(build_dataset(), tmp_path / "clip.json")
 
    raw = json.loads(path.read_text(encoding="utf-8"))
 
    assert raw["schemaVersion"] == SCHEMA_VERSION
 
 
def test_nullable_fields_survive_the_round_trip(tmp_path: Path) -> None:
    """Absent ball positions, teams and roles come back as None, not as defaults."""
    path = write_dataset(build_dataset(), tmp_path / "clip.json")
 
    restored = read_dataset(path)
 
    assert restored.frames[1].ball_position is None
    unassigned = next(
        player for player in restored.players if player.track_id == 9
    )
    assert unassigned.team is None
    assert unassigned.role is None
 
 
def test_reader_refuses_an_unknown_schema_version(tmp_path: Path) -> None:
    """A file from a future schema is refused rather than misread."""
    path = tmp_path / "future.json"
    payload = build_dataset().to_dict()
    payload["schemaVersion"] = "9.9.9"
    path.write_text(json.dumps(payload), encoding="utf-8")
 
    with pytest.raises(UnsupportedSchemaVersionError) as error:
        read_dataset(path)
 
    assert "9.9.9" in str(error.value)
 
 
def test_reader_refuses_a_file_with_no_version(tmp_path: Path) -> None:
    """A file without a version field is malformed, not assumed to be current."""
    path = tmp_path / "unversioned.json"
    payload = build_dataset().to_dict()
    del payload["schemaVersion"]
    path.write_text(json.dumps(payload), encoding="utf-8")
 
    with pytest.raises(MalformedDatasetError):
        read_dataset(path)
 
 
def test_reader_rejects_invalid_json(tmp_path: Path) -> None:
    """A file that is not JSON raises a dataset error, not a JSON error."""
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
 
    with pytest.raises(MalformedDatasetError):
        read_dataset(path)
 
 
def test_reader_rejects_a_file_missing_a_required_field(tmp_path: Path) -> None:
    """A file with the right version but the wrong shape is refused."""
    path = tmp_path / "incomplete.json"
    payload = build_dataset().to_dict()
    del payload["clip"]["frameRate"]
    path.write_text(json.dumps(payload), encoding="utf-8")
 
    with pytest.raises(MalformedDatasetError):
        read_dataset(path)
 
 
def test_writer_creates_missing_directories(tmp_path: Path) -> None:
    """Writing into a path whose parent does not exist creates the parent."""
    path = write_dataset(build_dataset(), tmp_path / "nested" / "deeper" / "clip.json")
 
    assert path.exists()
 
 
def test_analysed_team_returns_the_flagged_team() -> None:
    """Exactly one team is flagged and it is the one returned."""
    dataset = build_dataset()
 
    analysed = dataset.analysed_team()
 
    assert analysed is not None
    assert analysed.side is CourtSide.NEAR
 
 
def test_analysed_team_returns_none_when_nothing_is_flagged() -> None:
    """A dataset with no analysed team reports None rather than guessing."""
    dataset = build_dataset()
    dataset.teams = [
        Team(side=CourtSide.NEAR, is_analysed=False),
        Team(side=CourtSide.FAR, is_analysed=False),
    ]
 
    assert dataset.analysed_team() is None
 
 
def test_observations_for_track_filters_and_orders() -> None:
    """Observations come back for one track only, in frame order."""
    dataset = build_dataset()
 
    observations = dataset.observations_for_track(3)
 
    assert [item.frame_index for item in observations] == [0, 1]
    assert all(item.track_id == 3 for item in observations)
 
 
def test_describe_summarises_without_reading_the_file_again() -> None:
    """The summary counts records and reports the analysed side."""
    summary = describe(build_dataset())
 
    assert summary["frames"] == 2
    assert summary["observations"] == 3
    assert summary["players"] == 3
    assert summary["analysedSide"] == "NEAR"
    assert summary["framesWithBall"] == 1
 
