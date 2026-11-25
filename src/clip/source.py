"""The OpenCV frame source — the only module in this component that imports cv2.

Keeping the decoder here, behind the ``FrameSource`` Protocol in ``clip.py``,
is what confines OpenCV to one file (NFR-R2) and lets the rest of the stage be
tested without video or a codec (NFR-R8). Swapping to PyAV or an ffmpeg pipe
would mean writing a second class in this module and nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2

from src.clip.clip import ClipOpenError, VideoFrame, VideoProperties


class OpenCvFrameSource:
    """Decodes a clip with ``cv2.VideoCapture``."""

    def __init__(self, path: Path | str) -> None:
        """Record the path. Nothing is opened until it is needed."""
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """Return the path this source decodes."""
        return self._path

    def _capture(self) -> "cv2.VideoCapture":
        """Open a capture, raising if the file is missing or undecodable."""
        if not self._path.exists():
            raise ClipOpenError(self._path, "does not exist")
        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise ClipOpenError(
                self._path, "could not be opened; the codec or container may be unsupported"
            )
        return capture

    def properties(self) -> VideoProperties:
        """Return the container's header values.

        The capture is opened and released again, so asking for properties
        does not leave a decoder running.
        """
        capture = self._capture()
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

        if width <= 0 or height <= 0:
            raise ClipOpenError(
                self._path, f"reports a {width}x{height} frame size"
            )
        return VideoProperties(
            width=width,
            height=height,
            frame_rate=frame_rate,
            frame_count=frame_count,
        )

    def read_frames(self) -> Iterator[VideoFrame]:
        """Yield frames in order until the decoder stops returning them.

        The loop ends on the decoder's own signal rather than on the header's
        frame count, because containers routinely report a count that is off
        by a frame or two. Timestamps are derived from the index and the frame
        rate rather than read per frame: the source clips are constant frame
        rate, and CAP_PROP_POS_MSEC is unreliable across backends.
        """
        capture = self._capture()
        try:
            frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
            index = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    return
                timestamp = index / frame_rate if frame_rate > 0.0 else 0.0
                yield VideoFrame(index=index, timestamp=timestamp, image=image)
                index += 1
        finally:
            capture.release()

    def __repr__(self) -> str:
        """Return a summary naming the file this source decodes."""
        return f"OpenCvFrameSource({str(self._path)!r})"