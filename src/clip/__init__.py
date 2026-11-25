"""clip intake: metadata, validation and frame streaming (FR-1)"""

from src.clip.clip import (
    CONFIG_KEY_MAX_DURATION,
    CONFIG_KEY_MIN_DURATION,
    ClipError,
    ClipOpenError,
    ClipValidation,
    ConfigLike,
    FrameSource,
    RallyClip,
    VideoFrame,
    VideoProperties,
)

__all__ = [
    "CONFIG_KEY_MAX_DURATION",
    "CONFIG_KEY_MIN_DURATION",
    "ClipError",
    "ClipOpenError",
    "ClipValidation",
    "ConfigLike",
    "FrameSource",
    "RallyClip",
    "VideoFrame",
    "VideoProperties",
]