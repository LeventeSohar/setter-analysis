"""Tests for the rotation text notation.
No video, no pipeline stage and no fixture file: the notation depends only on CourtModel.
"""

from __future__ import annotations

import pytest

from src.courtmodel.model import SLOT_COUNT, CourtModel
from src.courtmodel.notation import (
    OTHER_MARK,
    SETTER_MARK,
    parse_rotation,
    render_rotation,
)

ROTATIONS = tuple(range(1, SLOT_COUNT + 1))


@pytest.fixture(name="court")
def court_fixture() -> CourtModel:
    return CourtModel()


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_round_trip(rotation: int, court: CourtModel) -> None:
    assert parse_rotation(render_rotation(rotation, court), court) == rotation


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_shape_is_two_rows_of_three(rotation: int, court: CourtModel) -> None:
    rows = render_rotation(rotation, court).splitlines()
    assert len(rows) == len(court.slot_grid())
    for row, slots in zip(rows, court.slot_grid()):
        assert len(row.replace("-", " ").split()) == len(slots)


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_exactly_one_setter_is_drawn(rotation: int, court: CourtModel) -> None:
    rendered = render_rotation(rotation, court)
    assert rendered.count(SETTER_MARK) == 1
    assert rendered.count(OTHER_MARK) == SLOT_COUNT - 1


def test_setter_is_drawn_in_the_right_row(court: CourtModel) -> None:
    """A front-row setter is drawn on the top row, a back-row setter below."""
    for rotation in ROTATIONS:
        front, back = render_rotation(rotation, court).splitlines()
        drawn_in_front = SETTER_MARK in front
        assert SETTER_MARK in back or drawn_in_front
        assert drawn_in_front == court.is_front_row_setter(rotation)


def test_case_and_separators_are_ignored(court: CourtModel) -> None:
    assert parse_rotation("o - o - o\no - o - s", court) == 1
    assert parse_rotation("O O O\nO O S", court) == 1
    assert parse_rotation("  o-o-o \n\n  o-o-s  ", court) == 1


@pytest.mark.parametrize(
    "text",
    [
        "O - O - O",
        "O - O - O\nO - O - O\nO - O - O",
        "O - O\nO - O - O",
        "O - O - O\nO - O - O",
        "S - S - O\nO - O - O",
        "O - X - O\nO - O - O",
        "",
    ],
)
def test_malformed_input_is_rejected(text: str, court: CourtModel) -> None:
    with pytest.raises(ValueError):
        parse_rotation(text, court)


def test_every_slot_is_drawn_exactly_once_across_the_six(court: CourtModel) -> None:
    assert sorted(parse_rotation(render_rotation(r, court), court) for r in ROTATIONS) \
        == list(ROTATIONS)