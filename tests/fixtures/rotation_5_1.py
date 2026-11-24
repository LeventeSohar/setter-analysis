"""
Hand-built 5-1 rotation fixtures

Six datasets, for the 6 different rotation, one for each used to develop RotationClassifier and SetterIdentifier without waiting for the computer vision stages
Court positions come from CourtModel.nominal_position and slot arithmetic from CourtModel.rotate, so no rotation rule or court dimension is restated in this file

the six states are named by the setter's slot, matching the official rules that the setter's slot is the rotation's identity

"""

from __future__ import annotations

from pathlib import Path

from src.courtmodel.model import SLOT_COUNT, CourtModel
from src.dataset.io import write_dataset
from src.dataset.schema import (
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

FIXTURE_DIR = Path(__file__).resolve().parent / "data"
FILENAME_TEMPLATE = "rotation_5_1_setter_slot_{rotation}.json"

#track identities in serve order, as a lineup sheet is written
#Index into LINEUP_TRACK_IDS of the single setter of a 5-1 lineup
#Zero by convention: a 5-1 lineup is normally staring with the setter in slot 1, so track 1 is the setter and tracks1-6 fill slots 1-6 in the starting rotation
#build_rotation_dataset takes this as a parameter so a test can build an off-convention lineup and confirm nothing depends on the default

LINEUP_TRACK_IDS = (1,2,3,4,5,6)
SETTER_LINEUP_INDEX = 0

#clip metadata follows the reference clip
#data/clips/Volleyball USA - Brazil Amazing FULL Match 2026_720p_rally_0001.mp4 (1280x720, 30fps)
#the source file is deliberately not that clip: these positions are nominal rather than measured from footage, and six rotational states cannot all describe one rally

SOURCE_FILE = "synthetic:5-1-rotation-fixture"
RESOLUTION = Size(width=1280, height=720)
FRAME_RATE = 30.0
FRAME_INDICES = (0, 15, 30, 45, 60)


#synthetic image mapping. this is an affinie fit, not a real homography
PIXELS_PER_METRE_X = 64.0
PIXELS_PER_METRE_Y = 32.0
IMAGE_X_AT_COURT_ORIGIN = 352.0
IMAGE_Y_AT_COURT_ORIGIN = 650.0

#bounding box size shrinking away from the camera

BOX_HEIGHT_AT_NEAR_BASELINE = 150.0
BOX_HEIGHT_SHRINK_PER_METRE = 3.0
BOX_ASPECT = 0.4

CALIBRATION_CONFIDENCE = 1.0

def image_to_court_transform() -> list[list[float]]:
    """Return the fixture's image-to-court transform as a 3x3 matrix."""
    return [
        [1.0 / PIXELS_PER_METRE_X, 0.0, -IMAGE_X_AT_COURT_ORIGIN / PIXELS_PER_METRE_X],
        [0.0, -1.0 / PIXELS_PER_METRE_Y, IMAGE_Y_AT_COURT_ORIGIN / PIXELS_PER_METRE_Y],
        [0.0, 0.0, 1.0],
    ]

def court_to_image(position: Point) -> Point:
    """Return the image point a court position projects to in this fixture."""
    return Point(
        x=IMAGE_X_AT_COURT_ORIGIN + PIXELS_PER_METRE_X * position.x,
        y=IMAGE_Y_AT_COURT_ORIGIN - PIXELS_PER_METRE_Y * position.y,
    )

def bounding_box_for(position: Point) -> Rect:
    """return a boundign box which bottom centre projects to position
    bottom centre is used as the boxs reference point"""

    foot = court_to_image(position)
    height = BOX_HEIGHT_AT_NEAR_BASELINE - BOX_HEIGHT_SHRINK_PER_METRE * position.y
    width = height * BOX_ASPECT
    return Rect(x=foot.x - width / 2.0, y=foot.y - height, width=width, height=height)

def slot_of_lineup_index(court: CourtModel, rotation: int, lineup_index: int, setter_lineup_index: int = SETTER_LINEUP_INDEX) -> int:
    """return the slot held by one lineup member in a rotation. 
    the rotation is the slot the setter is at. each player rotates together, so each of them are the same fix number away from the setter"""
    return court.rotate(rotation, setter_lineup_index - lineup_index)

def build_rotation_dataset(rotation: int, court: CourtModel | None = None, setter_lineup_index: int = SETTER_LINEUP_INDEX) -> Dataset:
    """build the dataset for the state with the setter in slot rotation
    setter_lineup_index says which lineup member is the setter. the default is the conventional lineup, with the setter starting in slot 1. the 6 committed fixture files all use the default"""
    court = CourtModel() if court is None else court
    court_model_slot = court.setter_slot(rotation)

    frames = [
        Frame(index=index, timestamp=index / FRAME_RATE, ball_position=None)
        for index in FRAME_INDICES
    ]

    players: list[Player] = []
    observations: list[PlayerObservation] = []

    for lineup_index, track_id in enumerate(LINEUP_TRACK_IDS):
        slot = slot_of_lineup_index(
            court, court_model_slot, lineup_index, setter_lineup_index
        )
        position = court.nominal_position(slot, CourtSide.NEAR)
        role = (
            PlayerRole.SETTER
            if lineup_index == setter_lineup_index
            else PlayerRole.OTHER
        )
        players.append(Player(track_id=track_id, team=CourtSide.NEAR, role=role))
        for frame in frames:
            observations.append(
                PlayerObservation(
                    frame_index=frame.index,
                    track_id=track_id,
                    bounding_box=bounding_box_for(position),
                    court_position=position,
                )
            )

    clip = ClipMetadata(
        id=f"fixture-5-1-setter-slot-{court_model_slot}",
        source_file=SOURCE_FILE,
        duration=(FRAME_INDICES[-1] + 1) / FRAME_RATE,
        frame_rate=FRAME_RATE,
        resolution=RESOLUTION,
    )
    calibration = Calibration(
        transform=image_to_court_transform(),
        court_lines=[],
        confidence=CALIBRATION_CONFIDENCE,
    )
    teams = [
        Team(side=CourtSide.NEAR, is_analysed=True),
        Team(side=CourtSide.FAR, is_analysed=False),
    ]

    return Dataset(
        clip=clip,
        calibration=calibration,
        frames=frames,
        observations=observations,
        players=players,
        teams=teams,
    )

def fixture_path(rotation: int) -> Path:
    """return the on disk path of one rotational states dataset file"""
    return FIXTURE_DIR / FILENAME_TEMPLATE.format(rotation=rotation)

def write_all(directory: Path | None = None) -> list[Path]:
    target = FIXTURE_DIR if directory is None else Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rotation in range(1, SLOT_COUNT + 1):
        path = target / FILENAME_TEMPLATE.format(rotation=rotation)
        write_dataset(build_rotation_dataset(rotation), path)
        written.append(path)
    return written

def main() -> int:
    """regenerate the 6 fixture files. entry point for cli"""
    court = CourtModel()
    for path in write_all():
        rotation = int(path.stem.rsplit("_", 1)[1])
        row = "front" if court.is_front_row_setter(rotation) else "back"
        print(f"wrote {path} setter slot {rotation} ({row} row)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())