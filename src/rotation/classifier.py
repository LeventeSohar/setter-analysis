"""
Rotation classification (FR-9).

Assigns each track on the analysed side to a rotation slot 1-6. It does not
name the setter: that is FR-10. The rotation's identity in 5-1 is the setter's
slot, so a stage doing both would be circular — this stage recovers the
arrangement, SetterIdentifier picks one of the six.

Every court dimension, slot layout and rotation rule comes from CourtModel
(NFR-R5). Every threshold comes from the config object (NFR-R4). The config
object is duck-typed on ``get(key)`` so that this module imports no other
stage and the tests need no config loader (NFR-R6, NFR-R8).

Method
------
1. One reference position per track: the per-axis median of its observations
   over an opening window of frames. The median absorbs a bad frame without
   letting it move a player. The window stands in for serve contact until
   contact detection lands.
2. Rows split by rank, not by the attack line. A rotation always has three
   players per row, so ranking by distance to the net enforces the structure
   the rules guarantee. ``CourtModel.is_front_row`` returns four front-row
   players on real footage, which is not a nameable rotation.
3. Within each row, order left to right from the attacking team's own
   perspective and read the slots off ``CourtModel.slot_grid``.
4. Confidence is the smaller of two normalised margins — the row split and
   the in-row ordering are the only two ways the assignment can be wrong, and
   one ambiguous pair should cap the result rather than be averaged away.

Six players or nothing: with five tracks there is no way to know which slot is
vacant, so a partial assignment would hand FR-10 a lineup that cannot exist.
That reports ``is_resolved=False`` with a note rather than raising. A missing
analysed team does raise — that is pipeline wiring, not bad footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Optional, Protocol

from src.courtmodel.model import CourtModel
from src.dataset.schema import CourtSide, Dataset, Point

CONFIG_KEY_REFERENCE_WINDOW = "rotation.reference_window_frames"
CONFIG_KEY_MIN_TRACK_OBSERVATIONS = "rotation.min_track_observations"
CONFIG_KEY_MIN_ROW_MARGIN = "rotation.min_row_margin"
CONFIG_KEY_MIN_SLOT_MARGIN = "rotation.min_slot_margin"


class RotationError(Exception):
    """Raised when the stage cannot run at all, not when it cannot decide."""


class ConfigLike(Protocol):
    """The only part of the config object this stage uses."""

    def get(self, key: str) -> Any:
        """Return the value for a dotted config key."""
        ...


@dataclass(frozen=True)
class RotationResult:
    """The slot assignment for one clip.

    ``slots`` maps track id to rotation slot 1-6. It is empty when the
    assignment is unresolved. ``confidence`` is 0.0 when unresolved, and
    otherwise the smaller of the normalised row and slot margins.
    ``notes`` explains any unresolved or low-confidence outcome.
    """

    side: Optional[CourtSide]
    slots: dict[int, int]
    confidence: float
    is_resolved: bool
    notes: tuple[str, ...] = ()

    def slot_of(self, track_id: int) -> Optional[int]:
        """Return the slot held by one track, or None if it has none."""
        return self.slots.get(track_id)

    def track_in_slot(self, slot: int) -> Optional[int]:
        """Return the track holding one slot, or None if the slot is vacant."""
        for track_id, held in self.slots.items():
            if held == slot:
                return track_id
        return None

    def assigned_tracks(self) -> tuple[int, ...]:
        """Return the assigned track ids ordered by the slot they hold."""
        return tuple(
            track_id
            for track_id, _ in sorted(self.slots.items(), key=lambda item: item[1])
        )


def _unresolved(side: Optional[CourtSide], note: str) -> RotationResult:
    """Return a result reporting that the arrangement could not be recovered."""
    return RotationResult(
        side=side, slots={}, confidence=0.0, is_resolved=False, notes=(note,)
    )


class RotationClassifier:
    """Assigns rotation slots to the tracks on the analysed side (FR-9)."""

    def __init__(self, config: ConfigLike, court: Optional[CourtModel] = None) -> None:
        """Read every threshold up front so an unset key fails here, not mid-clip."""
        self._court = CourtModel() if court is None else court
        self._reference_window = int(config.get(CONFIG_KEY_REFERENCE_WINDOW))
        self._min_track_observations = int(
            config.get(CONFIG_KEY_MIN_TRACK_OBSERVATIONS)
        )
        self._min_row_margin = float(config.get(CONFIG_KEY_MIN_ROW_MARGIN))
        self._min_slot_margin = float(config.get(CONFIG_KEY_MIN_SLOT_MARGIN))
        if self._reference_window <= 0:
            raise RotationError(
                f"{CONFIG_KEY_REFERENCE_WINDOW} must be positive, "
                f"got {self._reference_window}"
            )
        if self._min_track_observations <= 0:
            raise RotationError(
                f"{CONFIG_KEY_MIN_TRACK_OBSERVATIONS} must be positive, "
                f"got {self._min_track_observations}"
            )
        if self._min_row_margin <= 0.0 or self._min_slot_margin <= 0.0:
            raise RotationError("rotation margins must be positive")

    @property
    def court(self) -> CourtModel:
        """Return the court model this classifier reads geometry from."""
        return self._court

    # -- Public API -----------------------------------------------------------

    def classify(self, dataset: Dataset) -> RotationResult:
        """Return the slot assignment for the analysed side of one clip."""
        team = dataset.analysed_team()
        if team is None:
            raise RotationError(
                "dataset has no team marked as analysed; team separation "
                "must run before rotation classification"
            )
        side = team.side

        track_ids = [
            player.track_id for player in dataset.players if player.team is side
        ]
        rows = self._court.slot_grid()
        required = sum(len(row) for row in rows)

        if len(track_ids) != required:
            return _unresolved(
                side,
                f"{len(track_ids)} tracks on the {side.value} side, "
                f"expected exactly {required}",
            )

        references = self.reference_positions(dataset, track_ids)
        if len(references) != required:
            missing = sorted(set(track_ids) - set(references))
            return _unresolved(
                side,
                f"only {len(references)} of {required} tracks have at least "
                f"{self._min_track_observations} observation(s) in the first "
                f"{self._reference_window} frames; missing tracks {missing}",
            )

        return self._assign(side, references, rows)

    def reference_positions(
        self, dataset: Dataset, track_ids: list[int]
    ) -> dict[int, Point]:
        """Return the per-axis median position of each track over the window.

        Tracks with fewer than ``rotation.min_track_observations`` observations
        inside the window are left out. Dropping detector noise is FR-8's job
        and has its own key; sampling is sparse mid-rally, so a high value
        here would fight the stride rather than the noise.
        """
        if not dataset.frames:
            return {}
        first_index = min(frame.index for frame in dataset.frames)
        last_index = first_index + self._reference_window

        references: dict[int, Point] = {}
        for track_id in track_ids:
            inside = [
                observation.court_position
                for observation in dataset.observations_for_track(track_id)
                if first_index <= observation.frame_index <= last_index
            ]
            if len(inside) < self._min_track_observations:
                continue
            references[track_id] = Point(
                x=float(median(position.x for position in inside)),
                y=float(median(position.y for position in inside)),
            )
        return references

    # -- Internals ------------------------------------------------------------

    def _assign(
        self,
        side: CourtSide,
        references: dict[int, Point],
        rows: tuple[tuple[int, ...], ...],
    ) -> RotationResult:
        """Split the rows by rank, order each row, and read off the slots."""
        by_net_distance = sorted(
            references.items(),
            key=lambda item: self._court.distance_to_net(item[1]),
        )
        distances = [
            self._court.distance_to_net(position) for _, position in by_net_distance
        ]

        notes: list[str] = []
        assigned: dict[int, int] = {}
        slot_gaps: list[float] = []
        cursor = 0

        for row_slots in rows:
            size = len(row_slots)
            members = by_net_distance[cursor : cursor + size]
            cursor += size
            ordered = sorted(
                members, key=lambda item: self._team_left_key(item[1], side)
            )
            keys = [self._team_left_key(position, side) for _, position in ordered]
            slot_gaps.extend(
                keys[index + 1] - keys[index] for index in range(len(keys) - 1)
            )
            for slot, (track_id, _) in zip(row_slots, ordered):
                assigned[track_id] = slot

        row_boundary = len(rows[0])
        row_gap = distances[row_boundary] - distances[row_boundary - 1]
        slot_gap = min(slot_gaps) if slot_gaps else 0.0

        row_score = self._normalise(row_gap, self._min_row_margin)
        slot_score = self._normalise(slot_gap, self._min_slot_margin)

        if row_gap < self._min_row_margin:
            notes.append(
                f"row split margin {row_gap:.2f} m is below "
                f"{self._min_row_margin:.2f} m"
            )
        if slot_gap < self._min_slot_margin:
            notes.append(
                f"closest in-row pair is {slot_gap:.2f} m apart, below "
                f"{self._min_slot_margin:.2f} m"
            )

        return RotationResult(
            side=side,
            slots=assigned,
            confidence=min(row_score, slot_score),
            is_resolved=True,
            notes=tuple(notes),
        )

    def _team_left_key(self, position: Point, side: CourtSide) -> float:
        """Return a sort key increasing from the team's own left to its right.

        The handedness is read out of CourtModel rather than restated here:
        the camera sits behind the near half, so the far half is mirrored, and
        only CourtModel knows which way (NFR-R5).
        """
        front_row = self._court.slot_grid()[0]
        leftmost = self._court.nominal_position(front_row[0], side).x
        rightmost = self._court.nominal_position(front_row[-1], side).x
        sign = 1.0 if leftmost <= rightmost else -1.0
        return sign * position.x

    @staticmethod
    def _normalise(gap: float, threshold: float) -> float:
        """Return ``gap`` scaled by ``threshold`` and clipped to [0.0, 1.0]."""
        if gap <= 0.0:
            return 0.0
        return min(1.0, gap / threshold)