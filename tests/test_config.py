"""Tests for the config loader. No video, no other stage (NFR-R8)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    ConfigFileError,
    MissingConfigKeyError,
    UnknownConfigKeyError,
    describe,
    flatten,
    load_config,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "config" / "default.yaml"

# Every key section 5 of CONTEXT.md declares, with the values it records.
EXPECTED_SET_VALUES = {
    "rotation.reference_window_frames": 30,
    "rotation.min_track_observations": 1,
    "rotation.min_row_margin": 1.0,
    "rotation.min_slot_margin": 0.75,
}
EXPECTED_UNSET_KEYS = (
    "calibration.min_confidence",
    "detector.confidence",
    "detector.crop_margin",
    "detector.min_box_height",
    "sampling.serve_window",
    "sampling.set_window",
    "sampling.stride",
    "setter.ambiguity_margin",
    "teams.min_track_observations",
)


def write_yaml(path: Path, text: str) -> Path:
    """Write a YAML file for one test and return its path."""
    path.write_text(text, encoding="utf-8")
    return path


# -- flatten ------------------------------------------------------------------


def test_flatten_returns_dotted_keys():
    assert flatten({"a": {"b": 1}}) == {"a.b": 1}


def test_flatten_handles_several_levels():
    assert flatten({"a": {"b": {"c": 2}}}) == {"a.b.c": 2}


def test_flatten_keeps_top_level_scalars():
    assert flatten({"a": 1, "b": {"c": 2}}) == {"a": 1, "b.c": 2}


def test_flatten_preserves_none():
    assert flatten({"a": {"b": None}}) == {"a.b": None}


def test_flatten_treats_a_list_as_a_leaf():
    assert flatten({"a": {"b": [1, 2]}}) == {"a.b": [1, 2]}


def test_flatten_treats_an_empty_mapping_as_a_leaf():
    """An empty section must not vanish, or a typo'd section looks undeclared."""
    assert flatten({"a": {}}) == {"a": {}}


def test_flatten_of_an_empty_mapping_is_empty():
    assert flatten({}) == {}


def test_flatten_accepts_a_prefix():
    assert flatten({"b": 1}, prefix="a.") == {"a.b": 1}


# -- the shipped default file -------------------------------------------------


def test_shipped_config_loads():
    assert isinstance(load_config(SHIPPED_CONFIG), Config)


def test_shipped_config_declares_thirteen_keys():
    assert len(load_config(SHIPPED_CONFIG)) == 13


@pytest.mark.parametrize("key, value", sorted(EXPECTED_SET_VALUES.items()))
def test_shipped_config_sets_the_rotation_keys(key: str, value: object):
    assert load_config(SHIPPED_CONFIG).get(key) == value


@pytest.mark.parametrize("key", EXPECTED_UNSET_KEYS)
def test_shipped_config_declares_the_unwritten_stage_keys(key: str):
    """Declared but unset: the file is the register of what is still to choose."""
    config = load_config(SHIPPED_CONFIG)
    assert key in config
    assert not config.is_set(key)


@pytest.mark.parametrize("key", EXPECTED_UNSET_KEYS)
def test_reading_an_unset_shipped_key_raises(key: str):
    with pytest.raises(MissingConfigKeyError):
        load_config(SHIPPED_CONFIG).get(key)


def test_shipped_config_unset_keys_are_exactly_the_expected_nine():
    assert load_config(SHIPPED_CONFIG).unset_keys() == EXPECTED_UNSET_KEYS


def test_default_config_path_points_at_the_shipped_file():
    assert (REPO_ROOT / DEFAULT_CONFIG_PATH).resolve() == SHIPPED_CONFIG.resolve()


def test_shipped_config_declares_no_court_dimensions():
    """Court geometry belongs to CourtModel (NFR-R5), not to config."""
    keys = load_config(SHIPPED_CONFIG).keys()
    assert not [key for key in keys if "court" in key or "width" in key]


# -- get ----------------------------------------------------------------------


def test_get_returns_a_set_value(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: 3\n"))
    assert config.get("a.b") == 3


def test_get_raises_on_an_undeclared_key(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: 3\n"))
    with pytest.raises(UnknownConfigKeyError):
        config.get("a.missing")


def test_get_raises_on_a_null_value(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: null\n"))
    with pytest.raises(MissingConfigKeyError):
        config.get("a.b")


def test_get_takes_no_default_argument(tmp_path: Path):
    """A stage able to fall back to a literal would carry its own threshold."""
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: null\n"))
    with pytest.raises(TypeError):
        config.get("a.b", 0)  # type: ignore[call-arg]


def test_get_returns_zero_rather_than_treating_it_as_unset(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: 0\n"))
    assert config.get("a.b") == 0


def test_get_returns_false_rather_than_treating_it_as_unset(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: false\n"))
    assert config.get("a.b") is False


def test_missing_key_error_names_the_key(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: null\n"))
    with pytest.raises(MissingConfigKeyError) as caught:
        config.get("a.b")
    assert "a.b" in str(caught.value)
    assert caught.value.key == "a.b"


def test_unknown_key_error_names_the_key(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: 1\n"))
    with pytest.raises(UnknownConfigKeyError) as caught:
        config.get("nope")
    assert "nope" in str(caught.value)
    assert caught.value.key == "nope"


def test_the_two_key_errors_do_not_share_a_message(tmp_path: Path):
    """A base class formatting one subclass's message makes the others lie."""
    assert str(MissingConfigKeyError("k")) != str(UnknownConfigKeyError("k"))


def test_both_key_errors_are_config_errors():
    assert issubclass(MissingConfigKeyError, ConfigError)
    assert issubclass(UnknownConfigKeyError, ConfigError)


def test_config_error_base_takes_a_plain_message():
    assert str(ConfigError("plain")) == "plain"


# -- is_set, keys, container protocol -----------------------------------------


def test_is_set_is_true_for_a_value(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: 1\n"))
    assert config.is_set("a.b")


def test_is_set_is_false_for_null(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: null\n"))
    assert not config.is_set("a.b")


def test_is_set_raises_on_an_undeclared_key(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a:\n  b: 1\n"))
    with pytest.raises(UnknownConfigKeyError):
        config.is_set("nope")


def test_keys_are_sorted(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "z: 1\na: 2\n"))
    assert config.keys() == ("a", "z")


def test_contains_is_true_for_an_unset_key(tmp_path: Path):
    """Declared-but-unset is still declared."""
    config = load_config(write_yaml(tmp_path / "c.yaml", "a: null\n"))
    assert "a" in config


def test_contains_is_false_for_an_undeclared_key(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a: 1\n"))
    assert "b" not in config


def test_iterating_yields_sorted_keys(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "z: 1\na: 2\n"))
    assert list(config) == ["a", "z"]


def test_len_counts_declared_keys(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a: 1\nb: null\n"))
    assert len(config) == 2


def test_repr_reports_counts(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a: 1\nb: null\n"))
    assert repr(config) == "Config(2 keys, 1 unset)"


# -- overlay ------------------------------------------------------------------


def test_overlay_changes_a_declared_value(tmp_path: Path):
    base = write_yaml(tmp_path / "d.yaml", "a:\n  b: 1\n")
    over = write_yaml(tmp_path / "o.yaml", "a:\n  b: 2\n")
    assert load_config(base, overlay=over).get("a.b") == 2


def test_overlay_can_set_a_null_key(tmp_path: Path):
    base = write_yaml(tmp_path / "d.yaml", "a:\n  b: null\n")
    over = write_yaml(tmp_path / "o.yaml", "a:\n  b: 7\n")
    assert load_config(base, overlay=over).get("a.b") == 7


def test_overlay_leaves_untouched_keys_alone(tmp_path: Path):
    base = write_yaml(tmp_path / "d.yaml", "a: 1\nb: 2\n")
    over = write_yaml(tmp_path / "o.yaml", "a: 9\n")
    config = load_config(base, overlay=over)
    assert (config.get("a"), config.get("b")) == (9, 2)


def test_overlay_rejects_an_undeclared_key(tmp_path: Path):
    """A mistyped override must not leave the stage reading the default."""
    base = write_yaml(tmp_path / "d.yaml", "a:\n  b: 1\n")
    over = write_yaml(tmp_path / "o.yaml", "a:\n  typo: 2\n")
    with pytest.raises(UnknownConfigKeyError):
        load_config(base, overlay=over)


def test_overlay_rejection_names_the_overlay_file(tmp_path: Path):
    base = write_yaml(tmp_path / "d.yaml", "a: 1\n")
    over = write_yaml(tmp_path / "o.yaml", "typo: 2\n")
    with pytest.raises(UnknownConfigKeyError) as caught:
        load_config(base, overlay=over)
    assert "o.yaml" in str(caught.value)
    assert caught.value.source == over


def test_overlay_rejects_before_applying_any_override(tmp_path: Path):
    """All-or-nothing: a bad overlay must not half-apply."""
    base = write_yaml(tmp_path / "d.yaml", "a: 1\nb: 2\n")
    over = write_yaml(tmp_path / "o.yaml", "a: 9\ntypo: 3\n")
    with pytest.raises(UnknownConfigKeyError):
        load_config(base, overlay=over)
    assert load_config(base).get("a") == 1


def test_an_empty_overlay_changes_nothing(tmp_path: Path):
    base = write_yaml(tmp_path / "d.yaml", "a: 1\n")
    over = write_yaml(tmp_path / "o.yaml", "")
    assert load_config(base, overlay=over).get("a") == 1


def test_overlay_can_set_a_value_to_null(tmp_path: Path):
    """Unsetting is legal: it makes the key fail loudly again."""
    base = write_yaml(tmp_path / "d.yaml", "a: 1\n")
    over = write_yaml(tmp_path / "o.yaml", "a: null\n")
    with pytest.raises(MissingConfigKeyError):
        load_config(base, overlay=over).get("a")


# -- file errors --------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ConfigFileError):
        load_config(tmp_path / "absent.yaml")


def test_missing_file_error_names_the_path(tmp_path: Path):
    absent = tmp_path / "absent.yaml"
    with pytest.raises(ConfigFileError) as caught:
        load_config(absent)
    assert "absent.yaml" in str(caught.value)
    assert caught.value.path == absent


def test_invalid_yaml_raises(tmp_path: Path):
    with pytest.raises(ConfigFileError):
        load_config(write_yaml(tmp_path / "c.yaml", "a: [1,\n"))


def test_a_non_mapping_file_raises(tmp_path: Path):
    with pytest.raises(ConfigFileError):
        load_config(write_yaml(tmp_path / "c.yaml", "- 1\n- 2\n"))


def test_an_empty_file_declares_no_keys(tmp_path: Path):
    assert len(load_config(write_yaml(tmp_path / "c.yaml", ""))) == 0


def test_a_missing_overlay_raises(tmp_path: Path):
    base = write_yaml(tmp_path / "d.yaml", "a: 1\n")
    with pytest.raises(ConfigFileError):
        load_config(base, overlay=tmp_path / "absent.yaml")


def test_file_error_is_a_config_error():
    assert issubclass(ConfigFileError, ConfigError)


def test_load_config_accepts_a_string_path(tmp_path: Path):
    path = write_yaml(tmp_path / "c.yaml", "a: 1\n")
    assert load_config(str(path)).get("a") == 1


# -- describe and CLI ---------------------------------------------------------


def test_describe_reports_key_count_and_unset(tmp_path: Path):
    config = load_config(write_yaml(tmp_path / "c.yaml", "a: 1\nb: null\n"))
    summary = describe(config)
    assert summary["keyCount"] == 2
    assert summary["unsetKeys"] == ["b"]


def test_describe_uses_camel_case_keys(tmp_path: Path):
    """Summaries follow the dataset file's convention."""
    config = load_config(write_yaml(tmp_path / "c.yaml", "a: 1\n"))
    assert set(describe(config)) == {"keyCount", "unsetKeys", "values"}


def test_cli_exits_zero_on_the_shipped_config(capsys):
    assert main([str(SHIPPED_CONFIG)]) == 0
    assert "13 keys declared, 9 not set" in capsys.readouterr().out


def test_cli_marks_unset_keys(capsys):
    main([str(SHIPPED_CONFIG)])
    assert "<not set>" in capsys.readouterr().out


def test_cli_exits_one_on_a_missing_file(tmp_path: Path, capsys):
    assert main([str(tmp_path / "absent.yaml")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_accepts_an_overlay(tmp_path: Path, capsys):
    base = write_yaml(tmp_path / "d.yaml", "a: 1\n")
    over = write_yaml(tmp_path / "o.yaml", "a: 42\n")
    assert main([str(base), "--overlay", str(over)]) == 0
    assert "42" in capsys.readouterr().out