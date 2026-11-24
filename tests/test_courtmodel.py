import pytest

from src.courtmodel.model import(
    BACK_ROW_SLOTS,
    FRONT_ROW_SLOTS,
    SLOT_COUNT,
    AttackZone,
    CourtModel,
)
from src.dataset.schema import CourtSide, Point

NEAR = CourtSide.NEAR
FAR = CourtSide.FAR

@pytest.fixture
def court() -> CourtModel:
    return CourtModel()

#dimensions

def test_default_dimensions_are_official(court):
    assert court.width == pytest.approx(9.0)
    assert court.length == pytest.approx(18.0)
    assert court.net_y == pytest.approx(9.0)
    assert court.attack_line_distance == pytest.approx(3.0)
    assert court.zone_width == pytest.approx(3.0)

def test_rows_partition_the_slots():
    assert FRONT_ROW_SLOTS.isdisjoint(BACK_ROW_SLOTS)
    assert FRONT_ROW_SLOTS | BACK_ROW_SLOTS == set(range(1, SLOT_COUNT + 1))


#containment and sides

@pytest.mark.parametrize(
    "point, expected",
    [
        (Point(4.5, 9.0), True),
        (Point(0.0, 0.0), True),
        (Point(9.0, 18.0), True),
        (Point(-0.1, 4.0), False),
        (Point(4.5, 18.5), False),
    ],
)
def test_is_in_bounds(court, point, expected):
    assert court.is_in_bounds(point) is expected

def test_side_of_each_half(court):
    assert court.side_of(Point(4.5, 3.0)) is NEAR
    assert court.side_of(Point(4.5, 15.0)) is FAR


def test_side_of_is_undecided_on_the_net_line(court):
    assert court.side_of(Point(4.5, 9.0)) is None


def test_side_of_works_behind_the_baseline(court):
    assert court.side_of(Point(4.5, -1.0)) is NEAR
    assert court.side_of(Point(4.5, 19.0)) is FAR


def test_distance_to_net(court):
    assert court.distance_to_net(Point(4.5, 6.0)) == pytest.approx(3.0)
    assert court.distance_to_net(Point(4.5, 12.0)) == pytest.approx(3.0)


def test_is_front_row_uses_the_attack_line(court):
    assert court.is_front_row(Point(4.5, 7.0)) is True
    assert court.is_front_row(Point(4.5, 6.0)) is True
    assert court.is_front_row(Point(4.5, 5.9)) is False
    assert court.is_front_row(Point(4.5, 11.0)) is True


#attack zones
@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, AttackZone.LEFT),
        (2.9, AttackZone.LEFT),
        (3.0, AttackZone.MIDDLE),
        (5.9, AttackZone.MIDDLE),
        (6.0, AttackZone.RIGHT),
        (9.0, AttackZone.RIGHT),
    ],
)
def test_zone_of_near_side(court, x, expected):
    assert court.zone_of(Point(x, 7.0), NEAR) is expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (0.0, AttackZone.RIGHT),
        (3.0, AttackZone.MIDDLE),
        (9.0, AttackZone.LEFT),
    ],
)
def test_zone_of_far_side_is_mirrored(court, x, expected):
    assert court.zone_of(Point(x, 11.0), FAR) is expected


def test_zone_of_clamps_positions_outside_the_side_lines(court):
    assert court.zone_of(Point(-2.0, 7.0), NEAR) is AttackZone.LEFT
    assert court.zone_of(Point(11.0, 7.0), NEAR) is AttackZone.RIGHT


#Rotation slots 


@pytest.mark.parametrize(
    "point, expected",
    [
        (Point(1.5, 7.5), 4),
        (Point(4.5, 7.5), 3),
        (Point(7.5, 7.5), 2),
        (Point(1.5, 3.0), 5),
        (Point(4.5, 3.0), 6),
        (Point(7.5, 3.0), 1),
    ],
)
def test_slot_at_near_side(court, point, expected):
    assert court.slot_at(point, NEAR) == expected


@pytest.mark.parametrize(
    "point, expected",
    [
        (Point(7.5, 10.5), 4),
        (Point(4.5, 10.5), 3),
        (Point(1.5, 10.5), 2),
        (Point(7.5, 15.0), 5),
        (Point(4.5, 15.0), 6),
        (Point(1.5, 15.0), 1),
    ],
)
def test_slot_at_far_side(court, point, expected):
    assert court.slot_at(point, FAR) == expected


@pytest.mark.parametrize("side", [NEAR, FAR])
@pytest.mark.parametrize("slot", range(1, SLOT_COUNT + 1))
def test_nominal_position_round_trips_through_slot_at(court, slot, side):
    assert court.slot_at(court.nominal_position(slot, side), side) == slot


@pytest.mark.parametrize("side", [NEAR, FAR])
@pytest.mark.parametrize("slot", range(1, SLOT_COUNT + 1))
def test_nominal_positions_are_inside_the_court(court, slot, side):
    assert court.is_in_bounds(court.nominal_position(slot, side))


@pytest.mark.parametrize("side", [NEAR, FAR])
def test_nominal_positions_sit_on_the_correct_side(court, side):
    for slot in range(1, SLOT_COUNT + 1):
        assert court.side_of(court.nominal_position(slot, side)) is side


def test_nominal_position_near_slot_four(court):
    point = court.nominal_position(4, NEAR)
    assert point.x == pytest.approx(1.5)
    assert point.y == pytest.approx(7.5)


def test_nominal_position_far_slot_four_is_mirrored(court):
    point = court.nominal_position(4, FAR)
    assert point.x == pytest.approx(7.5)
    assert point.y == pytest.approx(10.5)


@pytest.mark.parametrize("side", [NEAR, FAR])
@pytest.mark.parametrize("slot", range(1, SLOT_COUNT + 1))
def test_nominal_position_row_matches_the_slot(court, slot, side):
    point = court.nominal_position(slot, side)
    assert court.is_front_row(point) is (slot in FRONT_ROW_SLOTS)


@pytest.mark.parametrize("slot", [0, 7, -1, 1.5, "1", None])
def test_nominal_position_rejects_bad_slots(court, slot):
    with pytest.raises(ValueError):
        court.nominal_position(slot, NEAR)

@pytest.mark.parametrize(
    "slot, expected",
    [(1, 6), (6, 5), (5, 4), (4, 3), (3, 2), (2, 1)],
)
def test_rotate_is_clockwise(court, slot, expected):
    assert court.rotate(slot) == expected


@pytest.mark.parametrize("slot", range(1, SLOT_COUNT + 1))
def test_rotate_full_cycle_is_identity(court, slot):
    assert court.rotate(slot, SLOT_COUNT) == slot


def test_rotate_multiple_and_negative_steps(court):
    assert court.rotate(1, 2) == 5
    assert court.rotate(6, -1) == 1
    assert court.rotate(3, 7) == court.rotate(3, 1)


def test_rotate_rejects_bad_slots(court):
    with pytest.raises(ValueError):
        court.rotate(0)


@pytest.mark.parametrize("rotation", range(1, SLOT_COUNT + 1))
def test_setter_slot_is_the_rotation_index(court, rotation):
    assert court.setter_slot(rotation) == rotation


def test_front_row_setter_rotations(court):
    front = {r for r in range(1, SLOT_COUNT + 1) if court.is_front_row_setter(r)}
    assert front == set(FRONT_ROW_SLOTS)


def test_setter_slot_rejects_bad_rotations(court):
    with pytest.raises(ValueError):
        court.setter_slot(7)