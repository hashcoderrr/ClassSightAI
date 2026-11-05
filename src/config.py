"""Configuration loading for SmartAttend.

The YAML file is wrapped in :class:`Config`, which gives attribute access
(``cfg.recognition.similarity_threshold``) as well as dict access, and resolves
all entries under ``paths`` to absolute paths.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config(Mapping[str, Any]):
    """Read-only nested view over a configuration dictionary."""

    def __init__(self, data: Mapping[str, Any], base_dir: Path | None = None) -> None:
        self._data = dict(data)
        self.base_dir = base_dir or PROJECT_ROOT

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{item}'") from exc

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        if isinstance(value, Mapping):
            return Config(value, self.base_dir)
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self[key] if key in self._data else default

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the underlying data."""
        return copy.deepcopy(self._data)

    def path(self, key: str) -> Path:
        """Absolute path for ``paths.<key>``."""
        p = Path(self._data["paths"][key])
        return p if p.is_absolute() else (self.base_dir / p).resolve()


def _deep_update(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | os.PathLike[str] | None = None,
                overrides: Mapping[str, Any] | None = None) -> Config:
    """Load ``config.yaml`` (or ``$SMARTATTEND_CONFIG``) and apply overrides.

    Args:
        path: Path to a YAML file. Defaults to ``$SMARTATTEND_CONFIG`` or the
            project's ``config.yaml``.
        overrides: Nested dict merged on top of the file contents.
    """
    cfg_path = Path(path or os.environ.get("SMARTATTEND_CONFIG") or DEFAULT_CONFIG_PATH).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if overrides:
        _deep_update(data, overrides)
    if os.environ.get("SMARTATTEND_INSIGHTFACE_ROOT"):  # e.g. models baked into a Docker image
        data.setdefault("paths", {})["insightface_root"] = os.environ["SMARTATTEND_INSIGHTFACE_ROOT"]
    cfg = Config(data, cfg_path.parent)
    for key in ("data_dir", "outputs_dir"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
    return cfg


def setup_logging(cfg: Config) -> None:
    """Configure root logging to console and the configured log file."""
    level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    log_file = cfg.path("log_file")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if getattr(root, "_smartattend_configured", False):
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    root._smartattend_configured = True  # type: ignore[attr-defined]
