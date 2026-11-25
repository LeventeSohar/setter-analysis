"""clip intake: metadata, validation and frame streaming
RallyClip is the only stage that turns a file on disk into frames.
it is given an already cut video. I have cut the full match separately because its not part of the project or scope
it looks at 3 things:
- what are the clips properties?
- is it a plausible rally at all?
- what are the clips frames?

Decoding is not done here. RallyClip interacts with FrameSource, a teo-method protocol and src.clip.source holds the one OpenCV implementation.
before unit tests pass a fake source and no need video and no OpenCV, while one integration test exercises the real decoder against a clip in generates.

Validation report, not raises errors. a clip which duration is not plausible is bad footage, which is the point of this stage to notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol

from src.dataset.schema import ClipMetadata, Size

CONFIG_KEY_MIN_DURATION = "clip.min_duration"
CONFIG_KEY_MAX_DURATION = "clip.max_duration"

class ClipError(Exception):
    """base class for every error raised by clip intake"""

class ClipOpenError(ClipError):
    """Raised when a clip file cannot be opened or carries no usable header."""

    def __init__(self, path: Path, reason: str) -> None:
        """Record the offending path and build a message naming it."""
        super().__init__(f"clip {str(path)!r} {reason}")
        self.path = path
        self.reason = reason

class ConfigLike(Protocol):
    """The only part of the config object this stage uses."""

    def get(self, key: str) -> Any:
        """Return the value for a dotted config key."""
        ...

@dataclass(frozen=True)
class VideoProperties:
    """What a decoder reports about a clip before any frame is read.
    """

    width: int
    height: int
    frame_rate: float
    frame_count: int

    @property
    def duration(self) -> float:
        """Return the clip's duration in seconds, from the header values."""
        if self.frame_rate <= 0.0:
            return 0.0
        return self.frame_count / self.frame_rate

    @property
    def resolution(self) -> Size:
        """Return the frame size as the schema's ``Size`` value type."""
        return Size(width=self.width, height=self.height)

@dataclass(frozen=True)
class VideoFrame:
    """One decoded frame."""

    index: int
    timestamp: float
    image: Any

@dataclass(frozen=True)
class ClipValidation:
    """Whether a clip is plausibly a single rally.

    ``is_usable`` false does not stop frames being read; it is a report for the
    runner to act on.
    """

    is_usable: bool
    notes: tuple[str, ...] = ()

class FrameSource(Protocol):
    """A decoder, as this stage needs it."""

    def properties(self) -> VideoProperties:
        """Return the clip's header properties."""
        ...

    def read_frames(self) -> Iterator[VideoFrame]:
        """Yield every frame in order, starting at index 0."""
        ...
class RallyClip:
    """Opens one rally clip, reports its metadata and streams its frames (FR-1)."""

    def __init__(
        self,
        path: Path | str,
        config: ConfigLike,
        source: Optional[FrameSource] = None,
    ) -> None:
        """Read both thresholds up front so an unset key fails here, not mid-clip.

        ``source`` is injected by the tests and by the runner. Left out, the
        OpenCV source is built lazily, so importing this module never imports
        OpenCV.
        """
        self._path = Path(path)
        self._min_duration = float(config.get(CONFIG_KEY_MIN_DURATION))
        self._max_duration = float(config.get(CONFIG_KEY_MAX_DURATION))
        if self._min_duration <= 0.0:
            raise ClipError(
                f"{CONFIG_KEY_MIN_DURATION} must be positive, "
                f"got {self._min_duration}"
            )
        if self._max_duration <= self._min_duration:
            raise ClipError(
                f"{CONFIG_KEY_MAX_DURATION} must exceed {CONFIG_KEY_MIN_DURATION}, "
                f"got {self._max_duration} and {self._min_duration}"
            )
        self._source = source
        self._properties: Optional[VideoProperties] = None

    @property
    def path(self) -> Path:
        """Return the path this clip was opened from."""
        return self._path

    @property
    def source(self) -> FrameSource:
        """Return the frame source, building the OpenCV one on first use."""
        if self._source is None:
            try:
                from src.clip.source import OpenCvFrameSource
            except ModuleNotFoundError as error:
                raise ClipError(
                    "decoding a clip needs OpenCV, which is not installed; "
                    "install opencv-python, or construct RallyClip with an "
                    "explicit source"
                ) from error

            self._source = OpenCvFrameSource(self._path)
        return self._source

    def properties(self) -> VideoProperties:
        """Return the clip's header properties, reading them once."""
        if self._properties is None:
            self._properties = self.source.properties()
        return self._properties

    def metadata(self) -> ClipMetadata:
        """Return the schema's ``ClipMetadata`` for this clip.

        The id is the file's stem, which is what the extraction tool's
        ``{source}_rally_{NNNN}`` naming already makes unique per match.
        """
        properties = self.properties()
        return ClipMetadata(
            id=self._path.stem,
            source_file=str(self._path),
            duration=properties.duration,
            frame_rate=properties.frame_rate,
            resolution=properties.resolution,
        )

    def validate(self) -> ClipValidation:
        """Report whether the clip is plausibly a single rally.

        Checks the duration against the configured bounds, and reports a frame
        rate or frame count that would make the duration meaningless. Does not
        raise: a clip failing here is the bad footage this stage exists to
        notice, and the runner decides what to do about it.
        """
        properties = self.properties()
        notes: list[str] = []

        if properties.frame_rate <= 0.0:
            notes.append(
                f"frame rate is {properties.frame_rate}, so duration cannot be "
                f"derived from the header"
            )
        if properties.frame_count <= 0:
            notes.append(
                f"header reports {properties.frame_count} frames"
            )

        if not notes:
            duration = properties.duration
            if duration < self._min_duration:
                notes.append(
                    f"duration {duration:.2f}s is below "
                    f"{CONFIG_KEY_MIN_DURATION} ({self._min_duration}s); "
                    f"too short to be a rally"
                )
            elif duration > self._max_duration:
                notes.append(
                    f"duration {duration:.2f}s is above "
                    f"{CONFIG_KEY_MAX_DURATION} ({self._max_duration}s); "
                    f"this may be an uncut match rather than one rally"
                )

        return ClipValidation(is_usable=not notes, notes=tuple(notes))

    def frames(self) -> Iterator[VideoFrame]:
        """Yield every frame in order.

        Streams rather than collecting: a rally is short, but nothing here
        should assume a clip fits in memory. Frames are yielded even when
        ``validate`` reported the clip unusable.
        """
        yield from self.source.read_frames()

    def __repr__(self) -> str:
        """Return a summary naming the clip, without opening it."""
        return f"RallyClip({str(self._path)!r})"