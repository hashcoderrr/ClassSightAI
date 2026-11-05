"""Board / teacher zone storage, one polygon per camera.

Polygons are stored in normalised (0-1) image coordinates in the YAML file at
``paths.zones_file`` (written by the UI's zone editor); inline defaults can be
given under ``behaviour.board.zones`` in config.yaml. The zones file wins.
"""
from __future__ import annotations

from typing import Any, Optional

import yaml


def load_zones(cfg: Any) -> dict[str, list[list[float]]]:
    zones: dict[str, list[list[float]]] = {}
    inline = cfg.behaviour.board.get("zones") or {}
    zones.update({str(k): [list(map(float, p)) for p in v] for k, v in dict(inline).items() if v})
    path = cfg.path("zones_file")
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        zones.update({str(k): [list(map(float, p)) for p in v] for k, v in data.items() if v})
    return zones


def get_zone(cfg: Any, camera_id: str) -> Optional[list[list[float]]]:
    return load_zones(cfg).get(str(camera_id))


def save_zone(cfg: Any, camera_id: str, polygon: Optional[list[list[float]]]) -> None:
    """Save (or delete, with ``polygon=None``) a camera's zone. Points must be normalised 0-1."""
    path = cfg.path("zones_file")
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if polygon:
        pts = [[round(min(1.0, max(0.0, float(x))), 4), round(min(1.0, max(0.0, float(y))), 4)] for x, y in polygon]
        if len(pts) < 3:
            raise ValueError("A zone needs at least 3 points.")
        data[str(camera_id)] = pts
    else:
        data.pop(str(camera_id), None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
