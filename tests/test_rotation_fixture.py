from __future__ import annotations

import pytest

from src.courtmodel.model import BACK_ROW_SLOTS, FRONT_ROW_SLOTS, SLOT_COUNT, CourtModel
from src.courtmodel.notation import parse_rotation, render_rotation
from src.dataset.io import read_dataset
from src.dataset.schema import CourtSide, PlayerRole
from tests.fixtures.rotation_5_1 import (
    LINEUP_TRACK_IDS,
    SETTER_LINEUP_INDEX,
    FRAME_INDICES,
    build_rotation_dataset,
    court_to_image,
    fixture_path,
    slot_of_lineup_index,
)

ROTATIONS = tuple(range(1, SLOT_COUNT + 1))
POSITION_TOLERANCE = 1e-9


@pytest.fixture(name="court")
def court_fixture() -> CourtModel:
    return CourtModel()


def project_to_court(transform, point):
    """Apply a 3x3 image-to-court transform to an image point."""
    x = transform[0][0] * point.x + transform[0][1] * point.y + transform[0][2]
    y = transform[1][0] * point.x + transform[1][1] * point.y + transform[1][2]
    w = transform[2][0] * point.x + transform[2][1] * point.y + transform[2][2]
    return x / w, y / w


def slots_by_track(dataset, court: CourtModel) -> dict[int, int]:
    """Return each track's slot, measured from its stored court position."""
    slots = {}
    for player in dataset.players:
        position = dataset.observations_for_track(player.track_id)[0].court_position
        slots[player.track_id] = court.slot_at(position, CourtSide.NEAR)
    return slots


#The files on disk 

@pytest.mark.parametrize("rotation", ROTATIONS)
def test_fixture_file_exists(rotation: int) -> None:
    assert fixture_path(rotation).is_file()


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_fixture_file_is_readable(rotation: int) -> None:
    assert read_dataset(fixture_path(rotation)) is not None


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_file_matches_builder(rotation: int) -> None:
    """The committed file is what the builder produces today."""
    assert read_dataset(fixture_path(rotation)) == build_rotation_dataset(rotation)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_clip_id_names_the_rotation(rotation: int) -> None:
    assert read_dataset(fixture_path(rotation)).clip.id.endswith(str(rotation))


#Teams and players 


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_near_team_is_the_analysed_team(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    analysed = dataset.analysed_team()
    assert analysed is not None
    assert analysed.side is CourtSide.NEAR


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_far_team_is_present_but_empty(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    assert {team.side for team in dataset.teams} == {CourtSide.NEAR, CourtSide.FAR}
    assert not [player for player in dataset.players if player.team is CourtSide.FAR]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_six_near_players_with_expected_tracks(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    assert len(dataset.players) == SLOT_COUNT
    assert {player.track_id for player in dataset.players} == set(LINEUP_TRACK_IDS)
    assert all(player.team is CourtSide.NEAR for player in dataset.players)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_exactly_one_setter(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    setters = [p for p in dataset.players if p.role is PlayerRole.SETTER]
    others = [p for p in dataset.players if p.role is PlayerRole.OTHER]
    assert len(setters) == 1
    assert len(others) == SLOT_COUNT - 1
    assert setters[0].track_id == LINEUP_TRACK_IDS[SETTER_LINEUP_INDEX]


#Rotation


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_slots_are_a_permutation(rotation: int, court: CourtModel) -> None:
    dataset = read_dataset(fixture_path(rotation))
    assert sorted(slots_by_track(dataset, court).values()) == list(ROTATIONS)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_setter_occupies_the_named_slot(rotation: int, court: CourtModel) -> None:
    dataset = read_dataset(fixture_path(rotation))
    setter = next(p for p in dataset.players if p.role is PlayerRole.SETTER)
    assert slots_by_track(dataset, court)[setter.track_id] == rotation


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_setter_row_matches_court_model(rotation: int, court: CourtModel) -> None:
    dataset = read_dataset(fixture_path(rotation))
    setter = next(p for p in dataset.players if p.role is PlayerRole.SETTER)
    position = dataset.observations_for_track(setter.track_id)[0].court_position
    assert court.is_front_row(position) == court.is_front_row_setter(rotation)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_three_players_in_each_row(rotation: int, court: CourtModel) -> None:
    dataset = read_dataset(fixture_path(rotation))
    slots = set(slots_by_track(dataset, court).values())
    assert len(slots & FRONT_ROW_SLOTS) == 3
    assert len(slots & BACK_ROW_SLOTS) == 3


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_lineup_order_is_preserved(rotation: int, court: CourtModel) -> None:
    """Each track sits where the shared lineup says it should."""
    dataset = read_dataset(fixture_path(rotation))
    slots = slots_by_track(dataset, court)
    for lineup_index, track_id in enumerate(LINEUP_TRACK_IDS):
        assert slots[track_id] == slot_of_lineup_index(court, rotation, lineup_index)


#Positions


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_positions_are_nominal(rotation: int, court: CourtModel) -> None:
    dataset = read_dataset(fixture_path(rotation))
    for track_id, slot in slots_by_track(dataset, court).items():
        expected = court.nominal_position(slot, CourtSide.NEAR)
        for observation in dataset.observations_for_track(track_id):
            assert observation.court_position == expected


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_every_player_is_in_the_near_half_and_in_bounds(
    rotation: int, court: CourtModel
) -> None:
    dataset = read_dataset(fixture_path(rotation))
    for observation in dataset.observations:
        assert court.is_in_bounds(observation.court_position)
        assert court.side_of(observation.court_position) is CourtSide.NEAR


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_boxes_project_to_their_court_position(rotation: int) -> None:
    """The bounding box and the court position describe the same player."""
    dataset = read_dataset(fixture_path(rotation))
    transform = dataset.calibration.transform
    for observation in dataset.observations:
        box = observation.bounding_box
        foot = court_to_image(observation.court_position)
        assert abs((box.x + box.width / 2.0) - foot.x) < POSITION_TOLERANCE
        assert abs((box.y + box.height) - foot.y) < POSITION_TOLERANCE
        x, y = project_to_court(transform, foot)
        assert abs(x - observation.court_position.x) < POSITION_TOLERANCE
        assert abs(y - observation.court_position.y) < POSITION_TOLERANCE


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_boxes_are_inside_the_image(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    resolution = dataset.clip.resolution
    for observation in dataset.observations:
        box = observation.bounding_box
        assert 0.0 <= box.x and box.x + box.width <= resolution.width
        assert 0.0 <= box.y and box.y + box.height <= resolution.height


#Frames


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_every_player_is_seen_in_every_frame(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    assert [frame.index for frame in dataset.frames] == list(FRAME_INDICES)
    assert len(dataset.observations) == SLOT_COUNT * len(FRAME_INDICES)
    for track_id in LINEUP_TRACK_IDS:
        seen = [o.frame_index for o in dataset.observations_for_track(track_id)]
        assert seen == list(FRAME_INDICES)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_timestamps_follow_the_frame_rate(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    for frame in dataset.frames:
        assert frame.timestamp == pytest.approx(frame.index / dataset.clip.frame_rate)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_no_ball_is_claimed(rotation: int) -> None:
    """The fixture carries no ball data; FR-9 and FR-10 do not need it."""
    dataset = read_dataset(fixture_path(rotation))
    assert all(frame.ball_position is None for frame in dataset.frames)


#Across the six states


def test_the_six_states_are_one_rotating_lineup(court: CourtModel) -> None:
    """Rotating each state by one step gives the next state's slots."""
    by_rotation = {
        rotation: slots_by_track(read_dataset(fixture_path(rotation)), court)
        for rotation in ROTATIONS
    }
    for rotation in ROTATIONS:
        rotated = {
            track_id: court.rotate(slot)
            for track_id, slot in by_rotation[rotation].items()
        }
        assert rotated == by_rotation[court.rotate(rotation)]


def test_the_states_are_distinct(court: CourtModel) -> None:
    arrangements = {
        tuple(sorted(slots_by_track(read_dataset(fixture_path(r)), court).items()))
        for r in ROTATIONS
    }
    assert len(arrangements) == SLOT_COUNT


def test_setter_covers_every_slot_once(court: CourtModel) -> None:
    occupied = []
    for rotation in ROTATIONS:
        dataset = read_dataset(fixture_path(rotation))
        setter = next(p for p in dataset.players if p.role is PlayerRole.SETTER)
        occupied.append(slots_by_track(dataset, court)[setter.track_id])
    assert sorted(occupied) == list(ROTATIONS)


#The lineup convention


def test_starting_rotation_is_the_conventional_lineup(court: CourtModel) -> None:
    """With the setter in slot 1, tracks 1-6 fill slots 1-6 in serve order."""
    dataset = read_dataset(fixture_path(1))
    slots = slots_by_track(dataset, court)
    assert [slots[track_id] for track_id in LINEUP_TRACK_IDS] == list(ROTATIONS)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_setter_is_track_one_by_convention(rotation: int) -> None:
    dataset = read_dataset(fixture_path(rotation))
    setter = next(p for p in dataset.players if p.role is PlayerRole.SETTER)
    assert setter.track_id == LINEUP_TRACK_IDS[SETTER_LINEUP_INDEX]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_invariants_hold_for_an_off_convention_lineup(
    rotation: int, court: CourtModel
) -> None:
    """The fixture's claims do not depend on the setter being track 1.

    The committed files all use the conventional lineup, which means a stage
    that simply returns the lowest track id would pass every setter test. This
    builds the same rotational state with a different lineup member as the
    setter and asserts the same invariants, so that blind spot is covered
    without a seventh file.
    """
    off_convention_index = 3
    dataset = build_rotation_dataset(
        rotation, setter_lineup_index=off_convention_index
    )
    setters = [p for p in dataset.players if p.role is PlayerRole.SETTER]
    assert len(setters) == 1
    assert setters[0].track_id == LINEUP_TRACK_IDS[off_convention_index]
    assert setters[0].track_id != LINEUP_TRACK_IDS[SETTER_LINEUP_INDEX]

    slots = slots_by_track(dataset, court)
    assert sorted(slots.values()) == list(ROTATIONS)
    assert slots[setters[0].track_id] == rotation


#Readable check against the drawn rotations

#: The six rotational states as drawn by hand, front row first, each row left to
#: right from the team's perspective. Written out rather than generated, so a
#: reader can check the fixture against a diagram instead of against code.
DRAWN_ROTATIONS = {
    1: "O - O - O\nO - O - S",
    2: "O - O - S\nO - O - O",
    3: "O - S - O\nO - O - O",
    4: "S - O - O\nO - O - O",
    5: "O - O - O\nS - O - O",
    6: "O - O - O\nO - S - O",
}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_fixture_matches_the_drawn_rotation(rotation: int, court: CourtModel) -> None:
    """The players in the file stand where the hand-drawn diagram says."""
    dataset = read_dataset(fixture_path(rotation))
    setter = next(p for p in dataset.players if p.role is PlayerRole.SETTER)
    setter_slot = slots_by_track(dataset, court)[setter.track_id]
    assert render_rotation(setter_slot, court) == DRAWN_ROTATIONS[rotation]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_drawn_rotation_parses_to_its_slot(rotation: int, court: CourtModel) -> None:
    assert parse_rotation(DRAWN_ROTATIONS[rotation], court) == rotation