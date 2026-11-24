"""
this file defines the geometry of the court sizes in volleyball

every court dimension, zone boundary and rotation rule is only in this file. no other module defiens them

Coordinate system
-----------------
All court positions are defined in meters as produced by CourtCalibration
 
 Origin is the court corner on the left of the camera
 x runs across the court from 0 to width increasing to the cameras right side
 y runs along the court from 0 to lenght increasing away from the camrea
 the net is at y == net_y. The near half is at y < net_y. The far half is y > net_y

 Sider are NEAR and FAR to match with the dataset schema


LEFT and RIGHT side
-------------------
AttackZone and rotation slots are expressed from the perspective of that teams side facing the net
The camera sits in the near half so what is on the left for the NEAR team is mirrored for the FAR team resulting in as right side. Every method that needs to resolve this takes a side argument

Rotation slots
--------------
Slots are numbered as according to official volleyball rules, 1 to 6 facing the net 

FAR team

1  6  5   FAR back row
2  3  4   FAR front row
-------   net
4  3  2   NEAR front row
5  6  1   NEAR back row

NEAR team

Rotation is clockwise: 1 -> 6 -> 5 -> 4 ...
"""


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.dataset.schema import CourtSide, Point

SLOT_COUNT = 6
FRONT_ROW_SLOTS = frozenset({2, 3, 4})
BACK_ROW_SLOTS = frozenset({1, 5, 6})

# Slot numbers indexed by band from the team's left, per row.
_FRONT_ROW_BY_BAND = (4, 3, 2)
_BACK_ROW_BY_BAND = (5, 6, 1)


class AttackZone(Enum):
    LEFT = "LEFT"
    MIDDLE = "MIDDLE"
    RIGHT = "RIGHT"


def _clamp(value: float, low: float, high: float) -> float:
    """Return ``value`` restricted to the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


@dataclass(frozen=True)
class CourtModel:
    width: float = 9.0
    half_length: float = 9.0
    attack_line_distance: float = 3.0
    zone_count: int = 3

    @property
    def length(self) -> float:
        return self.half_length * 2.0

    @property
    def net_y(self) -> float:
        return self.half_length

    @property
    def zone_width(self) -> float:
        return self.width / self.zone_count

    #Containment and sides

    def is_in_bounds(self, position: Point) -> bool:
        return 0.0 <= position.x <= self.width and 0.0 <= position.y <= self.length

    def side_of(self, position: Point) -> Optional[CourtSide]:
        if position.y < self.net_y:
            return CourtSide.NEAR
        if position.y > self.net_y:
            return CourtSide.FAR
        return None

    def distance_to_net(self, position: Point) -> float:
        return abs(position.y - self.net_y)

    def is_front_row(self, position: Point) -> bool:
        return self.distance_to_net(position) <= self.attack_line_distance

    #Zones and slots 

    def zone_of(self, position: Point, side: CourtSide) -> AttackZone:
        band = self._band_from_team_left(position.x, side)
        return (AttackZone.LEFT, AttackZone.MIDDLE, AttackZone.RIGHT)[band]

    def slot_at(self, position: Point, side: CourtSide) -> int:
        band = self._band_from_team_left(position.x, side)
        row = _FRONT_ROW_BY_BAND if self.is_front_row(position) else _BACK_ROW_BY_BAND
        return row[band]
    
    def slot_grid(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """return the slot layout as the front row then the back row.
        each row runs left to right from the teams perspective facing the net"""
        return (_FRONT_ROW_BY_BAND, _BACK_ROW_BY_BAND)

    def nominal_position(self, slot: int, side: CourtSide) -> Point:
        band = self._band_of_slot(slot)
        camera_band = band if side is CourtSide.NEAR else self.zone_count - 1 - band
        x = (camera_band + 0.5) * self.zone_width

        if slot in FRONT_ROW_SLOTS:
            offset = self.attack_line_distance / 2.0
        else:
            offset = (self.half_length + self.attack_line_distance) / 2.0
        y = self.net_y - offset if side is CourtSide.NEAR else self.net_y + offset

        return Point(x=x, y=y)

    # -- Rotation rules -------------------------------------------------------

    def rotate(self, slot: int, steps: int = 1) -> int:
        self._check_slot(slot)
        return (slot - 1 - steps) % SLOT_COUNT + 1

    def setter_slot(self, rotation: int) -> int:
        self._check_slot(rotation)
        return rotation

    def is_front_row_setter(self, rotation: int) -> bool:
        return self.setter_slot(rotation) in FRONT_ROW_SLOTS

    # -- Internals ------------------------------------------------------------

    def _band_from_team_left(self, x: float, side: CourtSide) -> int:
        """Return the across-court band index, 0 at the team's left."""
        clamped = _clamp(x, 0.0, self.width)
        camera_band = min(int(clamped / self.zone_width), self.zone_count - 1)
        if side is CourtSide.NEAR:
            return camera_band
        return self.zone_count - 1 - camera_band

    def _band_of_slot(self, slot: int) -> int:
        """Return the band index, 0 at the team's left, holding ``slot``."""
        self._check_slot(slot)
        row = _FRONT_ROW_BY_BAND if slot in FRONT_ROW_SLOTS else _BACK_ROW_BY_BAND
        return row.index(slot)

    @staticmethod
    def _check_slot(slot: int) -> None:
        """Raise ``ValueError`` unless ``slot`` is a rotation slot 1-6."""
        if not isinstance(slot, int) or not 1 <= slot <= SLOT_COUNT:
            raise ValueError(f"rotation slot must be an integer 1-{SLOT_COUNT}, got {slot!r}")


def main() -> int:
    """Print the court geometry this module defines. Entry point for the CLI."""
    court = CourtModel()
    print(f"court: {court.width:.1f} m wide x {court.length:.1f} m long")
    print(f"net at y = {court.net_y:.1f} m, attack lines at "
          f"y = {court.net_y - court.attack_line_distance:.1f} m and "
          f"y = {court.net_y + court.attack_line_distance:.1f} m")
    print(f"attack zones: {court.zone_count} bands of {court.zone_width:.1f} m")
    for side in (CourtSide.NEAR, CourtSide.FAR):
        print(f"\n{side.value} side - nominal slot centres (x, y in metres):")
        for slot in range(1, SLOT_COUNT + 1):
            point = court.nominal_position(slot, side)
            row = "front" if slot in FRONT_ROW_SLOTS else "back"
            zone = court.zone_of(point, side).value
            print(f"  slot {slot}: ({point.x:.2f}, {point.y:.2f})  {row:5s}  {zone}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())