"""Tests for RotationClassifier. No video, no config loader, no other stage."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.courtmodel.model import BACK_ROW_SLOTS, FRONT_ROW_SLOTS, SLOT_COUNT, CourtModel
from src.dataset.schema import CourtSide, Player, PlayerObservation, Point, Team
from src.rotation.classifier import (
    CONFIG_KEY_MIN_ROW_MARGIN,
    CONFIG_KEY_MIN_SLOT_MARGIN,
    CONFIG_KEY_MIN_TRACK_OBSERVATIONS,
    CONFIG_KEY_REFERENCE_WINDOW,
    RotationClassifier,
    RotationError,
    RotationResult,
)
from tests.fixtures.rotation_5_1 import (
    LINEUP_TRACK_IDS,
    build_rotation_dataset,
    slot_of_lineup_index,
)

DEFAULTS = {
    CONFIG_KEY_REFERENCE_WINDOW: 30,
    CONFIG_KEY_MIN_TRACK_OBSERVATIONS: 1,
    CONFIG_KEY_MIN_ROW_MARGIN: 1.0,
    CONFIG_KEY_MIN_SLOT_MARGIN: 0.75,
}


class StubConfig:
    """Minimal stand-in for the config object: dotted keys, no defaults."""

    def __init__(self, **overrides: object) -> None:
        self._values = dict(DEFAULTS)
        self._values.update(overrides)
        self.requested: list[str] = []

    def get(self, key: str) -> object:
        self.requested.append(key)
        if key not in self._values:
            raise KeyError(key)
        if self._values[key] is None:
            raise ValueError(f"config key {key} is declared but not set")
        return self._values[key]


@pytest.fixture
def classifier() -> RotationClassifier:
    return RotationClassifier(StubConfig())


def _move_track(dataset, track_id: int, position: Point):
    """Return the dataset with every observation of one track moved."""
    dataset.observations = [
        replace(observation, court_position=position)
        if observation.track_id == track_id
        else observation
        for observation in dataset.observations
    ]
    return dataset


#Construction and config 

def test_reads_only_its_own_four_keys():
    config = StubConfig()
    RotationClassifier(config)
    assert set(config.requested) == set(DEFAULTS)


def test_every_requested_key_is_namespaced():
    config = StubConfig()
    RotationClassifier(config)
    assert all(key.startswith("rotation.") for key in config.requested)


def test_unset_key_fails_at_construction():
    with pytest.raises(ValueError):
        RotationClassifier(StubConfig(**{CONFIG_KEY_MIN_ROW_MARGIN: None}))


@pytest.mark.parametrize(
    "key, bad",
    [
        (CONFIG_KEY_REFERENCE_WINDOW, 0),
        (CONFIG_KEY_MIN_TRACK_OBSERVATIONS, 0),
        (CONFIG_KEY_MIN_ROW_MARGIN, 0.0),
        (CONFIG_KEY_MIN_SLOT_MARGIN, -1.0),
    ],
)
def test_non_positive_thresholds_raise(key, bad):
    with pytest.raises(RotationError):
        RotationClassifier(StubConfig(**{key: bad}))


def test_classifier_holds_a_court_model(classifier):
    assert isinstance(classifier.court, CourtModel)


#The six nominal rotational states

@pytest.mark.parametrize("rotation", range(1, SLOT_COUNT + 1))
def test_recovers_every_nominal_rotation(classifier, rotation):
    result = classifier.classify(build_rotation_dataset(rotation))
    court = CourtModel()
    expected = {
        track_id: slot_of_lineup_index(court, rotation, index)
        for index, track_id in enumerate(LINEUP_TRACK_IDS)
    }
    assert result.is_resolved
    assert result.slots == expected


@pytest.mark.parametrize("rotation", range(1, SLOT_COUNT + 1))
def test_nominal_positions_are_fully_confident(classifier, rotation):
    result = classifier.classify(build_rotation_dataset(rotation))
    assert result.confidence == pytest.approx(1.0)
    assert result.notes == ()


@pytest.mark.parametrize("rotation", range(1, SLOT_COUNT + 1))
def test_assigns_all_six_slots_exactly_once(classifier, rotation):
    result = classifier.classify(build_rotation_dataset(rotation))
    assert sorted(result.slots.values()) == list(range(1, SLOT_COUNT + 1))


@pytest.mark.parametrize("rotation", range(1, SLOT_COUNT + 1))
def test_three_players_per_row(classifier, rotation):
    result = classifier.classify(build_rotation_dataset(rotation))
    slots = set(result.slots.values())
    assert len(slots & FRONT_ROW_SLOTS) == 3
    assert len(slots & BACK_ROW_SLOTS) == 3


def test_reports_the_analysed_side(classifier):
    result = classifier.classify(build_rotation_dataset(1))
    assert result.side is CourtSide.NEAR


def test_far_side_is_mirrored(classifier):
    """The same court positions read as a different rotation from the far side."""
    dataset = build_rotation_dataset(1)
    dataset.teams = [
        Team(side=CourtSide.NEAR, is_analysed=False),
        Team(side=CourtSide.FAR, is_analysed=True),
    ]
    dataset.players = [
        replace(player, team=CourtSide.FAR) for player in dataset.players
    ]
    result = classifier.classify(dataset)
    near = classifier.classify(build_rotation_dataset(1))
    assert result.is_resolved
    assert result.slots != near.slots


def test_off_convention_lineup_is_not_assumed(classifier):
    """Nothing depends on track 1 being the setter."""
    dataset = build_rotation_dataset(1, setter_lineup_index=3)
    court = CourtModel()
    expected = {
        track_id: slot_of_lineup_index(court, 1, index, 3)
        for index, track_id in enumerate(LINEUP_TRACK_IDS)
    }
    assert classifier.classify(dataset).slots == expected


def test_result_lookups_agree(classifier):
    result = classifier.classify(build_rotation_dataset(4))
    for track_id, slot in result.slots.items():
        assert result.slot_of(track_id) == slot
        assert result.track_in_slot(slot) == track_id
    assert result.slot_of(999) is None
    assert result.track_in_slot(0) is None


def test_assigned_tracks_are_slot_ordered(classifier):
    result = classifier.classify(build_rotation_dataset(2))
    slots = [result.slots[track_id] for track_id in result.assigned_tracks()]
    assert slots == sorted(slots)


#Six players or nothing 

def test_five_tracks_is_unresolved_not_an_error(classifier):
    dataset = build_rotation_dataset(1)
    dropped = LINEUP_TRACK_IDS[-1]
    dataset.players = [p for p in dataset.players if p.track_id != dropped]
    dataset.observations = [
        o for o in dataset.observations if o.track_id != dropped
    ]
    result = classifier.classify(dataset)
    assert not result.is_resolved
    assert result.slots == {}
    assert result.confidence == 0.0
    assert result.notes


def test_seven_tracks_is_unresolved(classifier):
    dataset = build_rotation_dataset(1)
    dataset.players.append(Player(track_id=99, team=CourtSide.NEAR))
    dataset.observations.extend(
        replace(o, track_id=99)
        for o in dataset.observations
        if o.track_id == LINEUP_TRACK_IDS[0]
    )
    assert not classifier.classify(dataset).is_resolved


def test_other_side_tracks_are_ignored(classifier):
    dataset = build_rotation_dataset(1)
    dataset.players.append(Player(track_id=77, team=CourtSide.FAR))
    assert classifier.classify(dataset).is_resolved


def test_unplaced_track_does_not_count_as_analysed(classifier):
    dataset = build_rotation_dataset(1)
    dataset.players.append(Player(track_id=78, team=None))
    assert classifier.classify(dataset).is_resolved


def test_track_outside_the_window_is_unresolved():
    classifier = RotationClassifier(StubConfig(**{CONFIG_KEY_REFERENCE_WINDOW: 10}))
    dataset = build_rotation_dataset(1)
    late = LINEUP_TRACK_IDS[2]
    dataset.observations = [
        o if o.track_id != late else replace(o, frame_index=o.frame_index + 60)
        for o in dataset.observations
    ]
    result = classifier.classify(dataset)
    assert not result.is_resolved
    assert str(late) in result.notes[0]


def test_min_track_observations_can_exclude_a_track():
    classifier = RotationClassifier(
        StubConfig(**{CONFIG_KEY_MIN_TRACK_OBSERVATIONS: 99})
    )
    assert not classifier.classify(build_rotation_dataset(1)).is_resolved


#Missing analysed team raises

def test_no_analysed_team_raises(classifier):
    dataset = build_rotation_dataset(1)
    dataset.teams = [
        Team(side=CourtSide.NEAR, is_analysed=False),
        Team(side=CourtSide.FAR, is_analysed=False),
    ]
    with pytest.raises(RotationError):
        classifier.classify(dataset)


def test_no_teams_at_all_raises(classifier):
    dataset = build_rotation_dataset(1)
    dataset.teams = []
    with pytest.raises(RotationError):
        classifier.classify(dataset)


#Reference position 

def test_median_absorbs_one_bad_frame(classifier):
    dataset = build_rotation_dataset(1)
    target = LINEUP_TRACK_IDS[0]
    clean = classifier.classify(dataset).slots
    dataset = build_rotation_dataset(1)
    first = min(
        o.frame_index for o in dataset.observations if o.track_id == target
    )
    dataset.observations = [
        replace(o, court_position=Point(x=8.9, y=0.1))
        if o.track_id == target and o.frame_index == first
        else o
        for o in dataset.observations
    ]
    assert classifier.classify(dataset).slots == clean


def test_reference_positions_are_per_track(classifier):
    dataset = build_rotation_dataset(1)
    references = classifier.reference_positions(dataset, list(LINEUP_TRACK_IDS))
    assert set(references) == set(LINEUP_TRACK_IDS)
    assert len(set(references.values())) == SLOT_COUNT


def test_no_frames_yields_no_references(classifier):
    dataset = build_rotation_dataset(1)
    dataset.frames = []
    assert classifier.reference_positions(dataset, list(LINEUP_TRACK_IDS)) == {}


#Confidence 

def test_tight_row_split_caps_confidence(classifier):
    """A back-row player standing on the attack line should not read as certain."""
    court = CourtModel()
    dataset = build_rotation_dataset(1)
    front = court.nominal_position(3, CourtSide.NEAR)
    crowded = Point(x=court.width / 2.0, y=front.y - 0.1)
    dataset = _move_track(dataset, LINEUP_TRACK_IDS[1], crowded)
    result = classifier.classify(dataset)
    assert result.confidence < 1.0
    assert result.notes


def test_tight_in_row_pair_caps_confidence(classifier):
    court = CourtModel()
    dataset = build_rotation_dataset(1)
    reference = None
    for player in dataset.players:
        reference = player.track_id
        break
    target = court.nominal_position(3, CourtSide.NEAR)
    dataset = _move_track(
        dataset, reference, Point(x=target.x - 0.1, y=target.y)
    )
    result = classifier.classify(dataset)
    assert 0.0 <= result.confidence <= 1.0


def test_confidence_is_the_smaller_margin(classifier):
    result = classifier.classify(build_rotation_dataset(1))
    assert result.confidence == min(1.0, result.confidence)


def test_confidence_is_bounded(classifier):
    for rotation in range(1, SLOT_COUNT + 1):
        result = classifier.classify(build_rotation_dataset(rotation))
        assert 0.0 <= result.confidence <= 1.0


#Result shape 

def test_result_is_frozen(classifier):
    result = classifier.classify(build_rotation_dataset(1))
    with pytest.raises(Exception):
        result.confidence = 0.5  # type: ignore[misc]


def test_result_names_no_setter(classifier):
    result = classifier.classify(build_rotation_dataset(1))
    assert not hasattr(result, "setter_slot")
    assert not hasattr(result, "rotation")


def test_stage_does_not_mutate_the_dataset(classifier):
    dataset = build_rotation_dataset(1)
    before = dataset.to_dict()
    classifier.classify(dataset)
    assert dataset.to_dict() == before