"""Tests for clip intake.

Every test but the integration section runs with no video, no codec and no
OpenCV: ``RallyClip`` talks to a ``FrameSource`` Protocol and the tests pass a
fake one (NFR-R8). The integration section generates its own clip and is
skipped where OpenCV or a working writer is absent, so the suite stays green on
a machine that has neither.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from src.clip.clip import (
    CONFIG_KEY_MAX_DURATION,
    CONFIG_KEY_MIN_DURATION,
    ClipError,
    ClipOpenError,
    ClipValidation,
    RallyClip,
    VideoFrame,
    VideoProperties,
)

DEFAULTS = {
    CONFIG_KEY_MIN_DURATION: 2.0,
    CONFIG_KEY_MAX_DURATION: 60.0,
}

CLIP_PATH = Path("data/clips/rally_0001.mp4")


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


class FakeFrameSource:
    """A frame source that invents frames, so no file and no codec is needed."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        frame_rate: float = 30.0,
        frame_count: int = 450,
        yield_count: int | None = None,
    ) -> None:
        self._properties = VideoProperties(
            width=width,
            height=height,
            frame_rate=frame_rate,
            frame_count=frame_count,
        )
        # Containers lie about frame_count, so the number yielded is separate.
        self._yield_count = frame_count if yield_count is None else yield_count
        self.properties_calls = 0
        self.read_calls = 0

    def properties(self) -> VideoProperties:
        self.properties_calls += 1
        return self._properties

    def read_frames(self) -> Iterator[VideoFrame]:
        self.read_calls += 1
        rate = self._properties.frame_rate
        for index in range(self._yield_count):
            yield VideoFrame(
                index=index,
                timestamp=index / rate if rate > 0 else 0.0,
                image=f"frame-{index}",
            )


def make_clip(source: FakeFrameSource | None = None, **config: object) -> RallyClip:
    """Build a RallyClip over a fake source."""
    return RallyClip(
        CLIP_PATH, StubConfig(**config), source=source or FakeFrameSource()
    )


#construction and config 


def test_reads_only_its_own_two_keys():
    config = StubConfig()
    RallyClip(CLIP_PATH, config, source=FakeFrameSource())
    assert set(config.requested) == {CONFIG_KEY_MIN_DURATION, CONFIG_KEY_MAX_DURATION}


def test_reads_both_thresholds_in_init():
    """An unset key must fail at construction, not halfway through a clip."""
    with pytest.raises(ValueError):
        RallyClip(CLIP_PATH, StubConfig(**{CONFIG_KEY_MIN_DURATION: None}))


def test_rejects_a_non_positive_min_duration():
    with pytest.raises(ClipError):
        make_clip(**{CONFIG_KEY_MIN_DURATION: 0.0})


def test_rejects_a_negative_min_duration():
    with pytest.raises(ClipError):
        make_clip(**{CONFIG_KEY_MIN_DURATION: -1.0})


def test_rejects_a_max_duration_below_the_min():
    with pytest.raises(ClipError):
        make_clip(**{CONFIG_KEY_MIN_DURATION: 10.0, CONFIG_KEY_MAX_DURATION: 5.0})


def test_rejects_equal_bounds():
    with pytest.raises(ClipError):
        make_clip(**{CONFIG_KEY_MIN_DURATION: 5.0, CONFIG_KEY_MAX_DURATION: 5.0})


def test_bound_error_names_both_keys():
    with pytest.raises(ClipError) as caught:
        make_clip(**{CONFIG_KEY_MIN_DURATION: 10.0, CONFIG_KEY_MAX_DURATION: 5.0})
    assert CONFIG_KEY_MIN_DURATION in str(caught.value)
    assert CONFIG_KEY_MAX_DURATION in str(caught.value)


def test_accepts_integer_thresholds():
    """YAML gives an int for a whole number; the stage must not care."""
    clip = make_clip(**{CONFIG_KEY_MIN_DURATION: 2, CONFIG_KEY_MAX_DURATION: 60})
    assert clip.validate().is_usable


def test_construction_does_not_touch_the_source():
    """Opening a file in __init__ would make the stage unusable for a dry run."""
    source = FakeFrameSource()
    RallyClip(CLIP_PATH, StubConfig(), source=source)
    assert source.properties_calls == 0
    assert source.read_calls == 0


def test_path_is_exposed():
    assert make_clip().path == CLIP_PATH


def test_path_accepts_a_string():
    clip = RallyClip("data/clips/x.mp4", StubConfig(), source=FakeFrameSource())
    assert clip.path == Path("data/clips/x.mp4")


def test_repr_names_the_clip_without_opening_it():
    source = FakeFrameSource()
    clip = RallyClip(CLIP_PATH, StubConfig(), source=source)
    assert "rally_0001.mp4" in repr(clip)
    assert source.properties_calls == 0


#properties 


def test_properties_come_from_the_source():
    clip = make_clip(FakeFrameSource(width=1920, height=1080))
    assert (clip.properties().width, clip.properties().height) == (1920, 1080)


def test_properties_are_read_once_and_cached():
    source = FakeFrameSource()
    clip = make_clip(source)
    clip.properties()
    clip.properties()
    clip.metadata()
    assert source.properties_calls == 1


def test_duration_is_frames_over_frame_rate():
    properties = VideoProperties(
        width=1280, height=720, frame_rate=30.0, frame_count=450
    )
    assert properties.duration == pytest.approx(15.0)


def test_duration_is_zero_when_the_frame_rate_is_zero():
    """A zero frame rate must not raise here; validate reports it instead."""
    properties = VideoProperties(
        width=1280, height=720, frame_rate=0.0, frame_count=450
    )
    assert properties.duration == 0.0


def test_duration_is_zero_when_the_frame_rate_is_negative():
    properties = VideoProperties(
        width=1280, height=720, frame_rate=-1.0, frame_count=450
    )
    assert properties.duration == 0.0


def test_resolution_is_the_schema_size_type():
    from src.dataset.schema import Size

    assert make_clip().properties().resolution == Size(width=1280, height=720)


#metadata 


def test_metadata_id_is_the_file_stem():
    assert make_clip().metadata().id == "rally_0001"


def test_metadata_source_file_is_the_path():
    assert make_clip().metadata().source_file == str(CLIP_PATH)


def test_metadata_carries_duration_and_frame_rate():
    metadata = make_clip().metadata()
    assert metadata.duration == pytest.approx(15.0)
    assert metadata.frame_rate == pytest.approx(30.0)


def test_metadata_resolution_matches_the_source():
    metadata = make_clip(FakeFrameSource(width=640, height=360)).metadata()
    assert (metadata.resolution.width, metadata.resolution.height) == (640, 360)


def test_metadata_is_the_schema_type():
    from src.dataset.schema import ClipMetadata

    assert isinstance(make_clip().metadata(), ClipMetadata)


def test_metadata_round_trips_through_the_schema():
    """What this stage produces must survive the dataset writer unchanged."""
    from src.dataset.schema import ClipMetadata

    metadata = make_clip().metadata()
    assert ClipMetadata.from_dict(metadata.to_dict()) == metadata


#validation 


def test_a_fifteen_second_clip_is_usable():
    assert make_clip().validate().is_usable


def test_a_usable_clip_carries_no_notes():
    assert make_clip().validate().notes == ()


def test_a_clip_below_the_minimum_is_unusable():
    source = FakeFrameSource(frame_count=30)  # 1.0s at 30fps
    assert not make_clip(source).validate().is_usable


def test_a_short_clip_note_names_the_key_and_the_duration():
    source = FakeFrameSource(frame_count=30)
    notes = make_clip(source).validate().notes
    assert len(notes) == 1
    assert CONFIG_KEY_MIN_DURATION in notes[0]
    assert "1.00s" in notes[0]


def test_a_clip_above_the_maximum_is_unusable():
    source = FakeFrameSource(frame_count=30 * 300)  # 300s
    assert not make_clip(source).validate().is_usable


def test_a_long_clip_note_suggests_an_uncut_match():
    source = FakeFrameSource(frame_count=30 * 300)
    notes = make_clip(source).validate().notes
    assert CONFIG_KEY_MAX_DURATION in notes[0]
    assert "uncut match" in notes[0]


def test_a_clip_exactly_at_the_minimum_is_usable():
    source = FakeFrameSource(frame_count=60)  # 2.0s
    assert make_clip(source).validate().is_usable


def test_a_clip_exactly_at_the_maximum_is_usable():
    source = FakeFrameSource(frame_count=1800)  # 60.0s
    assert make_clip(source).validate().is_usable


def test_a_zero_frame_rate_is_reported():
    source = FakeFrameSource(frame_rate=0.0)
    validation = make_clip(source).validate()
    assert not validation.is_usable
    assert "frame rate" in validation.notes[0]


def test_a_zero_frame_count_is_reported():
    source = FakeFrameSource(frame_count=0)
    validation = make_clip(source).validate()
    assert not validation.is_usable
    assert "0 frames" in " ".join(validation.notes)


def test_a_broken_header_does_not_also_report_a_duration():
    """A duration derived from a zero frame rate is noise, not a second finding."""
    source = FakeFrameSource(frame_rate=0.0)
    notes = make_clip(source).validate().notes
    assert not [note for note in notes if "below" in note or "above" in note]


def test_validation_does_not_raise_on_bad_footage():
    """Bad footage reports; only wiring raises."""
    assert isinstance(make_clip(FakeFrameSource(frame_count=1)).validate(), ClipValidation)


def test_validation_reads_properties_not_frames():
    source = FakeFrameSource()
    make_clip(source).validate()
    assert source.read_calls == 0


#frames 


def test_frames_yields_every_frame():
    source = FakeFrameSource(frame_count=10)
    assert len(list(make_clip(source).frames())) == 10


def test_frames_are_indexed_from_zero():
    source = FakeFrameSource(frame_count=5)
    assert [frame.index for frame in make_clip(source).frames()] == [0, 1, 2, 3, 4]


def test_frame_timestamps_follow_the_frame_rate():
    source = FakeFrameSource(frame_count=3, frame_rate=30.0)
    timestamps = [frame.timestamp for frame in make_clip(source).frames()]
    assert timestamps == pytest.approx([0.0, 1 / 30, 2 / 30])


def test_frames_is_lazy():
    """A rally is short, but nothing here may assume a clip fits in memory."""
    source = FakeFrameSource(frame_count=1000)
    stream = make_clip(source).frames()
    assert next(stream).index == 0


def test_frames_are_yielded_even_when_validation_failed():
    """The runner decides what to do with a bad clip, not this stage."""
    source = FakeFrameSource(frame_count=30)
    clip = make_clip(source)
    assert not clip.validate().is_usable
    assert len(list(clip.frames())) == 30


def test_a_container_lying_about_its_frame_count_is_tolerated():
    """Headers are routinely off by a frame or two; the decoder is the truth."""
    source = FakeFrameSource(frame_count=450, yield_count=448)
    assert len(list(make_clip(source).frames())) == 448


#errors 


def test_open_error_is_a_clip_error():
    assert issubclass(ClipOpenError, ClipError)


def test_open_error_names_the_path_and_the_reason():
    error = ClipOpenError(CLIP_PATH, "does not exist")
    assert "rally_0001.mp4" in str(error)
    assert "does not exist" in str(error)
    assert error.path == CLIP_PATH

def test_a_missing_opencv_becomes_a_clip_error(monkeypatch):
    """Without this the user gets a ModuleNotFoundError from three frames down."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "src.clip.source":
            raise ModuleNotFoundError("No module named 'cv2'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    clip = RallyClip(CLIP_PATH, StubConfig())
    with pytest.raises(ClipError) as caught:
        clip.source
    assert "opencv-python" in str(caught.value)

def test_clip_error_base_takes_a_plain_message():
    """A base that formats a subclass's message makes every other subclass lie."""
    assert str(ClipError("plain")) == "plain"


def test_the_two_clip_errors_do_not_share_a_message():
    assert str(ClipError("x")) != str(ClipOpenError(Path("x"), "does not exist"))


#integration: a real decoder against a generated clip 
@pytest.fixture
def cv2():
    """OpenCV, or skip just the tests that need a real decoder."""
    return pytest.importorskip("cv2", reason="OpenCV is not installed")


@pytest.fixture
def generated_clip(cv2, tmp_path: Path) -> Path:
    """Write a short synthetic clip, so the decoder is tested without footage."""
    np = pytest.importorskip("numpy", reason="numpy is not installed")

    path = tmp_path / "generated_rally.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180)
    )
    if not writer.isOpened():
        pytest.skip("no usable mp4v encoder on this machine")
    for index in range(90):  # 3.0s at 30fps
        frame = np.full((180, 320, 3), index % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("the encoder produced no output on this machine")
    return path


def test_opencv_source_reports_the_generated_properties(generated_clip: Path):
    from src.clip.source import OpenCvFrameSource

    properties = OpenCvFrameSource(generated_clip).properties()
    assert (properties.width, properties.height) == (320, 180)
    assert properties.frame_rate == pytest.approx(30.0, abs=0.5)


def test_opencv_source_decodes_every_frame(generated_clip: Path):
    from src.clip.source import OpenCvFrameSource

    assert len(list(OpenCvFrameSource(generated_clip).read_frames())) == 90


def test_opencv_source_frames_carry_an_image(generated_clip: Path):
    from src.clip.source import OpenCvFrameSource

    first = next(OpenCvFrameSource(generated_clip).read_frames())
    assert first.image.shape == (180, 320, 3)


def test_rally_clip_validates_a_generated_clip(generated_clip: Path):
    clip = RallyClip(generated_clip, StubConfig())
    assert clip.validate().is_usable
    assert clip.metadata().duration == pytest.approx(3.0, abs=0.1)


def test_rally_clip_builds_its_own_source_when_none_is_given(generated_clip: Path):
    from src.clip.source import OpenCvFrameSource

    clip = RallyClip(generated_clip, StubConfig())
    assert isinstance(clip.source, OpenCvFrameSource)


def test_a_missing_file_raises_open_error(cv2, tmp_path: Path):
    clip = RallyClip(tmp_path / "absent.mp4", StubConfig())
    with pytest.raises(ClipOpenError):
        clip.properties()


def test_a_file_that_is_not_video_raises_open_error(cv2, tmp_path: Path):
    path = tmp_path / "not_video.mp4"
    path.write_text("this is not an mp4", encoding="utf-8")
    clip = RallyClip(path, StubConfig())
    with pytest.raises(ClipOpenError):
        clip.properties()


#CLI 


def test_cli_exits_zero_on_a_usable_clip(generated_clip: Path, capsys):
    from src.clip.cli import main

    assert main([str(generated_clip)]) == 0
    assert '"isUsable": true' in capsys.readouterr().out


def test_cli_exits_two_on_an_unusable_clip(tmp_path: Path, generated_clip: Path, capsys):
    """Unusable is a finding, not a crash, so it gets its own exit code."""
    from src.clip.cli import main

    strict = tmp_path / "strict.yaml"
    strict.write_text(
        "clip:\n  min_duration: 10.0\n  max_duration: 60.0\n", encoding="utf-8"
    )
    assert main([str(generated_clip), "--config", str(strict)]) == 2
    assert "too short to be a rally" in capsys.readouterr().err


def test_cli_exits_one_when_the_clip_cannot_be_opened(tmp_path: Path, capsys):
    from src.clip.cli import main

    assert main([str(tmp_path / "absent.mp4")]) == 1
    assert "error:" in capsys.readouterr().err


def test_cli_writes_the_summary_to_a_file(generated_clip: Path, tmp_path: Path):
    import json

    from src.clip.cli import main

    out = tmp_path / "summary.json"
    assert main([str(generated_clip), "--output", str(out)]) == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["clipId"] == "generated_rally"
    assert summary["isUsable"] is True


def test_cli_counts_decoded_frames(generated_clip: Path, capsys):
    import json

    from src.clip.cli import main

    assert main([str(generated_clip), "--count-frames"]) == 0
    assert json.loads(capsys.readouterr().out)["decodedFrameCount"] == 90


def test_cli_reports_a_missing_config(generated_clip: Path, tmp_path: Path, capsys):
    from src.clip.cli import main

    assert main([str(generated_clip), "--config", str(tmp_path / "absent.yaml")]) == 1
    assert "error:" in capsys.readouterr().err