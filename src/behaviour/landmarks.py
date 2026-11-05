"""Facial landmarks (InsightFace 1k3d68, part of buffalo_l) -> EAR / MAR / head pose.

The 68-point output follows the iBUG-300W layout, so the classic Eye Aspect
Ratio (Soukupova & Cech, 2016) and Mouth Aspect Ratio formulas apply directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

LEFT_EYE = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
INNER_MOUTH = [60, 61, 62, 63, 64, 65, 66, 67]


def _d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(lmk: np.ndarray) -> float:
    """Mean EAR of both eyes (~0.25-0.35 open, < ~0.18 closed)."""
    def ear(idx: list[int]) -> float:
        p = lmk[idx, :2]
        return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-6)
    return (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2.0


def mouth_aspect_ratio(lmk: np.ndarray) -> float:
    """Inner-mouth MAR (~0.0-0.1 closed, > 0.3 wide open)."""
    p = lmk[INNER_MOUTH, :2]
    return (_d(p[1], p[7]) + _d(p[2], p[6]) + _d(p[3], p[5])) / (2.0 * _d(p[0], p[4]) + 1e-6)


class FaceLandmarker:
    """Runs the 68-point 3D landmark model on a face box."""

    def __init__(self, model_path: str | Path, providers: list[str], ctx_id: int = -1) -> None:
        from insightface.app.common import Face
        from insightface.model_zoo import get_model
        self._Face = Face
        self.model = get_model(str(model_path), providers=providers)
        if self.model is None:
            raise RuntimeError(f"Could not load landmark model {model_path}")
        self.model.prepare(ctx_id)
        log.info("Facial landmark model loaded: %s", Path(model_path).name)

    @classmethod
    def from_config(cls, cfg: Any) -> Optional["FaceLandmarker"]:
        from ..detector import select_providers
        path = cfg.path("insightface_root") / "models" / cfg.models.name / "1k3d68.onnx"
        if not path.exists():
            log.warning("Landmark model %s not found - EAR/MAR disabled, head pose from 5 keypoints.", path)
            return None
        providers, ctx = select_providers(bool(cfg.models.use_gpu))
        return cls(path, providers, ctx)

    def landmarks(self, img: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """(68, 2) landmarks in image coordinates, or None if the face is degenerate."""
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        face = self._Face(bbox=np.array([x1, y1, x2, y2], dtype=np.float32))
        try:
            pred = self.model.get(img, face)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("landmark failure: %s", exc)
            return None
        return np.asarray(pred, dtype=np.float32)[:, :2]
