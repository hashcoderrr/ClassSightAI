"""Basic face anti-spoofing (liveness) check.

Two modes:
* **ONNX model** - if ``liveness.model_path`` points to a MiniFASNet-style model
  (input 1x3x80x80 BGR, softmax output where index 1 = real face), it is used.
* **Heuristic fallback** - flags crops that look like printed photos or phone/
  monitor screens using (a) moire / periodic high-frequency energy in the FFT
  spectrum, (b) unnaturally low colour saturation, and (c) specular glare.

The heuristic is deliberately conservative and is NOT a security guarantee; see
the README "Known limitations" section.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class LivenessChecker:
    """Scores a face crop in [0, 1]; higher = more likely a live face."""

    def __init__(self, model_path: str = "", min_score: float = 0.5, use_gpu: bool = False) -> None:
        self.min_score = min_score
        self._session = None
        if model_path and Path(model_path).exists():
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
            if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            self._session = ort.InferenceSession(model_path, providers=providers)
            self._input = self._session.get_inputs()[0].name
            log.info("Liveness ONNX model loaded: %s", model_path)
        elif model_path:
            log.warning("Liveness model %s not found - using heuristic check.", model_path)

    def score(self, crop: np.ndarray) -> float:
        if crop is None or crop.size == 0:
            return 0.0
        if self._session is not None:
            return self._model_score(crop)
        return self._heuristic_score(crop)

    def is_live(self, crop: np.ndarray) -> bool:
        return self.score(crop) >= self.min_score

    def _model_score(self, crop: np.ndarray) -> float:
        x = cv2.resize(crop, (80, 80)).astype(np.float32).transpose(2, 0, 1)[None]
        out = np.asarray(self._session.run(None, {self._input: x})[0]).reshape(-1)
        e = np.exp(out - out.max())
        probs = e / e.sum()
        return float(probs[1]) if probs.size > 1 else float(probs[0])

    @staticmethod
    def _heuristic_score(crop: np.ndarray) -> float:
        face = cv2.resize(crop, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # (a) moire: share of spectral energy in isolated high-frequency peaks
        spec = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
        spec = np.log1p(spec)
        h, w = spec.shape
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
        high = spec[r > min(h, w) * 0.3]
        peak_ratio = float((high > high.mean() + 4 * high.std()).mean()) if high.size else 0.0
        moire_s = float(np.clip(1.0 - peak_ratio * 50.0, 0.0, 1.0))
        # (b) saturation: printed / screen-replayed faces are often washed out
        sat = float(cv2.cvtColor(face, cv2.COLOR_BGR2HSV)[..., 1].mean()) / 255.0
        sat_s = float(np.clip(sat / 0.15, 0.0, 1.0))
        # (c) glare: large blown-out regions from glossy screens
        glare = float((gray > 250).mean())
        glare_s = float(np.clip(1.0 - glare * 10.0, 0.0, 1.0))
        return 0.5 * moire_s + 0.3 * sat_s + 0.2 * glare_s


def build_liveness(cfg) -> Optional[LivenessChecker]:
    """Create a checker from config, or ``None`` when disabled."""
    lcfg = cfg.liveness
    if not lcfg.enabled:
        return None
    return LivenessChecker(lcfg.get("model_path", ""), float(lcfg.min_score), bool(cfg.models.use_gpu))
