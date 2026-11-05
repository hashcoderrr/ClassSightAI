"""Face-crop quality scoring and image enhancement helpers."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def laplacian_variance(img: np.ndarray) -> float:
    """Sharpness measure: variance of the Laplacian of the gray image."""
    if img is None or img.size == 0:
        return 0.0
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frontal_score(kps: Optional[np.ndarray]) -> float:
    """Estimate how frontal a face is from its 5 landmarks (0..1, 1 = frontal).

    Landmarks order (InsightFace): left eye, right eye, nose, left mouth, right mouth.
    Yaw is approximated by how far the nose sits from the eye midpoint relative to
    the inter-ocular distance; roll by the eye-line angle.
    """
    if kps is None or len(kps) < 5:
        return 0.5
    kps = np.asarray(kps, dtype=np.float32)
    le, re, nose = kps[0], kps[1], kps[2]
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < 1e-3:
        return 0.0
    mid = (le + re) / 2.0
    yaw = abs(float(nose[0] - mid[0])) / eye_dist          # ~0 frontal, ~0.5 strong profile
    roll = abs(float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0]))))
    yaw_s = max(0.0, 1.0 - yaw / 0.5)
    roll_s = max(0.0, 1.0 - roll / 45.0)
    return float(yaw_s * 0.75 + roll_s * 0.25)


def quality_score(crop: np.ndarray, bbox: np.ndarray, kps: Optional[np.ndarray], det_score: float,
                  ref_size: float = 112.0, ref_sharpness: float = 150.0) -> float:
    """Combine sharpness, face size, frontal pose and detector confidence into 0..1."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    size = min(x2 - x1, y2 - y1)
    size_s = float(np.clip(size / ref_size, 0.0, 1.0))
    sharp_s = float(np.clip(laplacian_variance(crop) / ref_sharpness, 0.0, 1.0))
    pose_s = frontal_score(kps)
    det_s = float(np.clip(det_score, 0.0, 1.0))
    return 0.30 * size_s + 0.25 * sharp_s + 0.30 * pose_s + 0.15 * det_s


def safe_crop(img: np.ndarray, bbox: np.ndarray, pad: float = 0.15) -> np.ndarray:
    """Crop a padded box, clipped to the image; returns an empty array if invalid."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - bw * pad)), int(max(0, y1 - bh * pad))
    x2, y2 = int(min(w, x2 + bw * pad)), int(min(h, y2 + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return img[y1:y2, x1:x2].copy()


def enhance_low_light(frame: np.ndarray, threshold: float = 90.0, clip_limit: float = 2.0,
                      always: bool = False) -> np.ndarray:
    """Apply CLAHE on the L channel (+ gamma) when the frame is dark."""
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if not always and gray_mean >= threshold:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    if gray_mean < threshold * 0.6:
        gamma = 1.5
        table = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
        out = cv2.LUT(out, table)
    return out
