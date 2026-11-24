from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

SCHEMA_VERSION = "0.1.0"
"""Version written into every dataset file produced by this code"""

SUPPORTED_SCHEMA_VERSIONS = frozenset({"0.1.0"})


class PlayerRole(str, Enum):
    SETTER = "SETTER"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

class CourtSide(str, Enum):
    NEAR = "NEAR"
    FAR = "FAR"


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Size":
        return cls(width=int(data["width"]), height=int(data["height"]))

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Point":
        return cls(x=float(data["x"]), y=float(data["y"]))

@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> dict[str, Any]:
        """Build the record as a plain dictionary."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rect":
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            width=float(data["width"]),
            height=float(data["height"]),
        )

@dataclass(frozen=True)
class ClipMetadata:
    """identity and source properties of any rally clip"""
    id: str
    source_file: str
    duration: float
    frame_rate: float
    resolution: Size

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {
            "id": self.id,
            "sourceFile": self.source_file,
            "duration": self.duration,
            "frameRate": self.frame_rate,
            "resolution": self.resolution.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClipMetadata":
        """Build a record from a plain dictionary."""
        return cls(
            id=str(data["id"]),
            source_file=str(data["sourceFile"]),
            duration=float(data["duration"]),
            frame_rate=float(data["frameRate"]),
            resolution=Size.from_dict(data["resolution"]),
        )
    
@dataclass(frozen=True)
class Calibration:
    """The image-to-court transform computed once for a clip.
 
    ``transform`` is a 3x3 matrix as a list of three rows of three floats.
    ``court_lines`` holds the detected lines as point pairs, kept so that a
    calibration can be inspected and re-measured without the source video.
    """
 
    transform: list[list[float]]
    court_lines: list[tuple[Point, Point]]
    confidence: float
 
    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {
            "transform": [list(row) for row in self.transform],
            "courtLines": [
                {"start": start.to_dict(), "end": end.to_dict()}
                for start, end in self.court_lines
            ],
            "confidence": self.confidence,
        }
 
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        """Build a record from a plain dictionary."""
        return cls(
            transform=[[float(value) for value in row] for row in data["transform"]],
            court_lines=[
                (Point.from_dict(line["start"]), Point.from_dict(line["end"]))
                for line in data["courtLines"]
            ],
            confidence=float(data["confidence"]),
        )


@dataclass(frozen=True)
class Frame:
    """One sampled frame of the clip.
 
    ``ball_position`` is ``None`` when no ball was detected in this frame,
    which is expected on a meaningful share of frames and is not an error.
    """
 
    index: int
    timestamp: float
    ball_position: Optional[Point] = None
 
    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "ballPosition": (
                None if self.ball_position is None else self.ball_position.to_dict()
            ),
        }
 
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Frame":
        """Build a record from a plain dictionary."""
        raw_ball = data.get("ballPosition")
        return cls(
            index=int(data["index"]),
            timestamp=float(data["timestamp"]),
            ball_position=None if raw_ball is None else Point.from_dict(raw_ball),
        )

@dataclass(frozen=True)
class PlayerObservation:
    """One player seen in one frame.
 
    ``court_position`` is stored rather than derived on read so that the file
    is self-describing: the contact-level pipeline can work entirely in court
    coordinates without re-implementing the projection.
    """
 
    frame_index: int
    track_id: int
    bounding_box: Rect
    court_position: Point
 
    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {
            "frameIndex": self.frame_index,
            "trackId": self.track_id,
            "boundingBox": self.bounding_box.to_dict(),
            "courtPosition": self.court_position.to_dict(),
        }
 
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerObservation":
        """Build a record from a plain dictionary."""
        return cls(
            frame_index=int(data["frameIndex"]),
            track_id=int(data["trackId"]),
            bounding_box=Rect.from_dict(data["boundingBox"]),
            court_position=Point.from_dict(data["courtPosition"]),
        )

@dataclass(frozen=True)
class Player:
    """A track identity held for the duration of one clip.
 
    ``team`` is ``None`` when team separation could not place the track, and
    ``role`` is ``None`` until the setter identifier has run.
    """
 
    track_id: int
    team: Optional[CourtSide] = None
    role: Optional[PlayerRole] = None
 
    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {
            "trackId": self.track_id,
            "team": None if self.team is None else self.team.value,
            "role": None if self.role is None else self.role.value,
        }
 
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Player":
        """Build a record from a plain dictionary."""
        raw_team = data.get("team")
        raw_role = data.get("role")
        return cls(
            track_id=int(data["trackId"]),
            team=None if raw_team is None else CourtSide(raw_team),
            role=None if raw_role is None else PlayerRole(raw_role),
        )

@dataclass(frozen=True)
class Team:
    """One of the two teams on court."""
 
    side: CourtSide
    is_analysed: bool
 
    def to_dict(self) -> dict[str, Any]:
        """Return the record as a plain dictionary."""
        return {"side": self.side.value, "isAnalysed": self.is_analysed}
 
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Team":
        """Build a record from a plain dictionary."""
        return cls(side=CourtSide(data["side"]), is_analysed=bool(data["isAnalysed"]))

@dataclass
class Dataset:
    """The complete contents of one dataset file.
 
    One dataset describes exactly one rally clip. It is written once by the
    clip-level pipeline and read once by the contact-level pipeline.
    """
 
    clip: ClipMetadata
    calibration: Calibration
    frames: list[Frame] = field(default_factory=list)
    observations: list[PlayerObservation] = field(default_factory=list)
    players: list[Player] = field(default_factory=list)
    teams: list[Team] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the whole dataset as a plain dictionary."""
        return {
            "schemaVersion": self.schema_version,
            "clip": self.clip.to_dict(),
            "calibration": self.calibration.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
            "players": [player.to_dict() for player in self.players],
            "teams": [team.to_dict() for team in self.teams],
        }
 
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Dataset":
        """Build a dataset from a plain dictionary.
 
        This does not check the schema version; that is the reader's job, and
        it must happen before this method is reached.
        """
        return cls(
            schema_version=str(data["schemaVersion"]),
            clip=ClipMetadata.from_dict(data["clip"]),
            calibration=Calibration.from_dict(data["calibration"]),
            frames=[Frame.from_dict(item) for item in data["frames"]],
            observations=[
                PlayerObservation.from_dict(item) for item in data["observations"]
            ],
            players=[Player.from_dict(item) for item in data["players"]],
            teams=[Team.from_dict(item) for item in data["teams"]],
        )
    def analysed_team(self) -> Optional[Team]:
        """Return the team marked as analysed, or None if none is marked."""
        for team in self.teams:
            if team.is_analysed:
                return team
        return None

    def observations_for_track(self, track_id: int) -> list[PlayerObservation]:
        """Return every observation belonging to one track, in frame order."""
        matching = [
            observation
            for observation in self.observations
            if observation.track_id == track_id
        ]
        return sorted(matching, key=lambda observation: observation.frame_index)