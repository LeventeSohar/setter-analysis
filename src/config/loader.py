"""Loading and reading the configuration file.

Every tunable value in the pipeline resolves through this module, so
no stage carries a numeric literal of its own. ``config/default.yaml`` declares
the full set of keys; an overlay file may change the value of a declared key
but may not introduce one.

Three rules shape the API, and each exists because of a failure it prevents:

* A key set to ``null`` is declared but not set, and reading it raises
  ``MissingConfigKeyError``. A threshold nobody has chosen fails loudly at the
  point of use instead of silently behaving as zero.
* An overlay key the default file does not declare is an error. A mistyped
  override would otherwise leave the stage reading the default while the
  author believed the override took effect.
* ``Config.get`` takes no default argument. A stage able to fall back to a
  literal would be carrying a threshold of its own, which is the thing NFR-R4
  forbids.

Keys are addressed as dotted strings — ``rotation.min_row_margin`` — matching
the constants each stage exports. PyYAML is confined to this module (NFR-R2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


class ConfigError(Exception):
    """Base class for every error raised while loading or reading config.

    Deliberately plain: it formats no message of its own. A base class that
    builds one subclass's message makes every other subclass lie about what
    went wrong.
    """


class ConfigFileError(ConfigError):
    """Raised when a config file is missing, unreadable or not a mapping."""

    def __init__(self, path: Path, reason: str) -> None:
        """Record the offending path and build a message naming it."""
        super().__init__(f"config file {str(path)!r} {reason}")
        self.path = path
        self.reason = reason


class UnknownConfigKeyError(ConfigError):
    """Raised when a key is read, or overridden, without being declared."""

    def __init__(self, key: str, *, source: Path | None = None) -> None:
        """Name the undeclared key and, for an overlay, the file it came from."""
        where = "" if source is None else f" in overlay {str(source)!r}"
        super().__init__(
            f"config key {key!r} is not declared{where}; "
            f"add it to {DEFAULT_CONFIG_PATH} before using it"
        )
        self.key = key
        self.source = source


class MissingConfigKeyError(ConfigError):
    """Raised when a declared key is read while still set to ``null``."""

    def __init__(self, key: str) -> None:
        """Name the unset key and say what has to happen before it is read."""
        super().__init__(
            f"config key {key!r} is declared but not set; "
            f"choose a value in {DEFAULT_CONFIG_PATH} before the stage reads it"
        )
        self.key = key


def flatten(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return a nested mapping as one dict keyed by dotted paths.

    Only mappings are descended into; a list is a leaf value. An empty mapping
    is a leaf too, so a section with no keys does not vanish silently.
    """
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = value
    return flat


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read one YAML file and return its top-level mapping."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigFileError(path, "does not exist") from error
    except OSError as error:
        raise ConfigFileError(path, f"could not be read: {error}") from error

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigFileError(path, f"is not valid YAML: {error}") from error

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigFileError(
            path, f"must contain a mapping at the top level, found {type(loaded).__name__}"
        )
    return dict(loaded)


class Config:
    """A flat, read-only view of the resolved configuration.

    Construct it through ``load_config``. ``get`` is the whole interface a
    stage needs, which is why stages depend on a one-method Protocol rather
    than importing this class (NFR-R6).
    """

    def __init__(self, values: Mapping[str, Any]) -> None:
        """Store a copy of the resolved dotted-key values."""
        self._values = dict(values)

    def get(self, key: str) -> Any:
        """Return the value for a dotted config key.

        Raises ``UnknownConfigKeyError`` if the key was never declared and
        ``MissingConfigKeyError`` if it is declared but still ``null``. There
        is no default argument on purpose; see the module docstring.
        """
        if key not in self._values:
            raise UnknownConfigKeyError(key)
        value = self._values[key]
        if value is None:
            raise MissingConfigKeyError(key)
        return value

    def is_set(self, key: str) -> bool:
        """Return whether a declared key has a value, without raising.

        the CLI below and a runner deciding which stages can
        run. A stage reads ``get`` and lets an unset key fail.
        """
        if key not in self._values:
            raise UnknownConfigKeyError(key)
        return self._values[key] is not None

    def keys(self) -> tuple[str, ...]:
        """Return every declared key in sorted order."""
        return tuple(sorted(self._values))

    def unset_keys(self) -> tuple[str, ...]:
        """Return every declared key still awaiting a value, sorted."""
        return tuple(key for key in self.keys() if self._values[key] is None)

    def __contains__(self, key: object) -> bool:
        """Return whether a key is declared, set or not."""
        return key in self._values

    def __iter__(self) -> Iterator[str]:
        """Iterate over declared keys in sorted order."""
        return iter(self.keys())

    def __len__(self) -> int:
        """Return the number of declared keys."""
        return len(self._values)

    def __repr__(self) -> str:
        """Return a summary naming how many keys are declared and unset."""
        return (
            f"Config({len(self._values)} keys, "
            f"{len(self.unset_keys())} unset)"
        )


def load_config(
    path: Path | str = DEFAULT_CONFIG_PATH,
    overlay: Path | str | None = None,
) -> Config:
    """Load the default config file, optionally overridden by an overlay.

    The default file defines which keys exist. The overlay may change the
    value of any declared key, including setting one that was ``null``, but a
    key it introduces raises ``UnknownConfigKeyError`` rather than being
    accepted quietly.
    """
    default_path = Path(path)
    values = flatten(_read_yaml_mapping(default_path))

    if overlay is not None:
        overlay_path = Path(overlay)
        overrides = flatten(_read_yaml_mapping(overlay_path))
        for key in overrides:
            if key not in values:
                raise UnknownConfigKeyError(key, source=overlay_path)
        values.update(overrides)

    return Config(values)


def describe(config: Config) -> dict[str, Any]:
    """Return a summary of a config: its keys, values and what is unset."""
    return {
        "keyCount": len(config),
        "unsetKeys": list(config.unset_keys()),
        "values": {key: config._values[key] for key in config.keys()},
    }


def main(argv: list[str] | None = None) -> int:
    """Print every config key and whether it is set. Exit 0 always.

    Unset keys are expected while stages are unwritten, so they are reported
    rather than treated as failure. A file that cannot be read is a failure
    and exits 1.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.config",
        description="Show the resolved configuration and which keys are unset.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"config file to read (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="optional overlay file; may change declared keys, not add them",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, overlay=args.overlay)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    width = max((len(key) for key in config.keys()), default=0)
    for key in config:
        if config.is_set(key):
            print(f"{key:<{width}}  {config.get(key)!r}")
        else:
            print(f"{key:<{width}}  <not set>")

    unset = config.unset_keys()
    print(f"\n{len(config)} keys declared, {len(unset)} not set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())