"""Head pose (yaw / pitch / roll) and 'looking at the board' geometry.

Convention used everywhere in SmartAttend (camera coordinates, OpenCV axes):
* yaw   > 0  -> the face points towards the image's right side
* pitch > 0  -> the face points downwards (looking at the desk)
* yaw = pitch = 0 -> looking straight into the camera

The forward (face-normal) unit vector is
``f = (sin(yaw)cos(pitch), sin(pitch), -cos(yaw)cos(pitch))``; its (x, y)
part is the head-direction arrow drawn on the video.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np

# generic 3D face model (mm), y up, z towards the viewer; nose tip at the origin
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # nose tip            (68-pt idx 30)
    (0.0, -330.0, -65.0),     # chin                 (8)
    (-225.0, 170.0, -135.0),  # image-left eye corner (36)
    (225.0, 170.0, -135.0),   # image-right eye corner (45)
    (-150.0, -150.0, -125.0),  # image-left mouth corner (48)
    (150.0, -150.0, -125.0),   # image-right mouth corner (54)
], dtype=np.float64)
LMK68_IDX = [30, 8, 36, 45, 48, 54]


def forward_vector(yaw: float, pitch: float) -> np.ndarray:
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)])


def angles_from_forward(f: np.ndarray) -> tuple[float, float]:
    f = f / (np.linalg.norm(f) + 1e-12)
    yaw = math.degrees(math.atan2(f[0], -f[2]))
    pitch = math.degrees(math.asin(float(np.clip(f[1], -1.0, 1.0))))
    return yaw, pitch


def camera_matrix(w: int, h: int) -> np.ndarray:
    focal = float(max(w, h))
    return np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)


def pose_from_landmarks68(lmk: np.ndarray, frame_w: int, frame_h: int) -> Optional[tuple[float, float, float]]:
    """solvePnP on six 68-point landmarks -> (yaw, pitch, roll) in degrees."""
    try:
        pts = np.asarray(lmk, dtype=np.float64)[LMK68_IDX, :2]
    except (IndexError, TypeError):
        return None
    ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts, camera_matrix(frame_w, frame_h), np.zeros(4),
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    yaw, pitch = angles_from_forward(rot @ np.array([0.0, 0.0, 1.0]))
    le, re = pts[2], pts[3]
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    return yaw, pitch, roll


def pose_from_kps5(kps: np.ndarray) -> Optional[tuple[float, float, float]]:
    """Coarse fallback from the detector's 5 landmarks (eyes, nose, mouth corners)."""
    if kps is None or len(kps) < 5:
        return None
    k = np.asarray(kps, dtype=np.float64)
    le, re, nose, lm, rm = k[0], k[1], k[2], k[3], k[4]
    eye_mid, mouth_mid = (le + re) / 2, (lm + rm) / 2
    eye_d = np.linalg.norm(re - le)
    if eye_d < 1e-3:
        return None
    yaw = float(np.clip((nose[0] - eye_mid[0]) / eye_d, -1, 1)) * 90.0
    span = mouth_mid[1] - eye_mid[1]
    ratio = (nose[1] - eye_mid[1]) / span if abs(span) > 1e-3 else 0.55
    pitch = float(np.clip((ratio - 0.55) * 150.0, -60, 70))
    roll = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
    return yaw, pitch, roll


def pose_from_body(kps: np.ndarray, min_conf: float = 0.3) -> Optional[tuple[float, float]]:
    """Very coarse (yaw, pitch) from COCO keypoints when the face is not visible."""
    if kps is None:
        return None
    c = kps[:, 2] >= min_conf
    nose, l_ear, r_ear, l_sho, r_sho = 0, 3, 4, 5, 6
    if not (c[l_sho] and c[r_sho]):
        return None
    sho_y = (kps[l_sho, 1] + kps[r_sho, 1]) / 2
    sw = abs(kps[l_sho, 0] - kps[r_sho, 0]) + 1e-6
    pitch = 0.0
    if not c[nose]:
        pitch = 50.0                                  # face hidden: head down / turned away
    elif kps[nose, 1] > sho_y - 0.15 * sw:
        pitch = 45.0
    yaw = 0.0
    if c[r_ear] and not c[l_ear]:
        yaw = 50.0
    elif c[l_ear] and not c[r_ear]:
        yaw = -50.0
    return yaw, pitch


def head_arrow(center: Sequence[float], yaw: float, pitch: float, length: float) -> tuple[int, int]:
    f = forward_vector(yaw, pitch)
    return int(center[0] + f[0] * length), int(center[1] + f[1] * length)


def _ray_hits_polygon(origin: np.ndarray, direction: np.ndarray, poly: np.ndarray) -> bool:
    if cv2.pointPolygonTest(poly.astype(np.float32), (float(origin[0]), float(origin[1])), False) >= 0:
        return True
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        denom = direction[0] * e[1] - direction[1] * e[0]
        if abs(denom) < 1e-9:
            continue
        diff = a - origin
        t = (diff[0] * e[1] - diff[1] * e[0]) / denom
        u = (diff[0] * direction[1] - diff[1] * direction[0]) / denom
        if t >= 0 and 0 <= u <= 1:
            return True
    return False


class BoardZone:
    """Decides whether a head direction points at the board / teacher zone.

    * ``facing_camera`` mode (no polygon drawn; board next to the camera):
      looking at the board == yaw/pitch within configured limits.
    * polygon mode: the 2D head-direction ray from the face must hit the polygon
      (normalised 0-1 image coordinates). If the head points almost straight at the
      camera, the result is ``camera_is_board``.
    """

    def __init__(self, polygon: Optional[Sequence[Sequence[float]]] = None, yaw_limit: float = 25.0,
                 pitch_min: float = -25.0, pitch_max: float = 15.0, camera_is_board: bool = True,
                 min_ray: float = 0.15) -> None:
        self.polygon = np.asarray(polygon, dtype=np.float64) if polygon is not None and len(polygon) >= 3 else None
        self.yaw_limit, self.pitch_min, self.pitch_max = yaw_limit, pitch_min, pitch_max
        self.camera_is_board = camera_is_board
        self.min_ray = min_ray

    @classmethod
    def from_config(cls, bcfg, polygon=None) -> "BoardZone":
        z = bcfg.board
        return cls(polygon, float(z.yaw_limit), float(z.pitch_min), float(z.pitch_max),
                   bool(z.camera_is_board), float(z.get("min_ray", 0.15)))

    def facing_camera(self, yaw: float, pitch: float) -> bool:
        return abs(yaw) <= self.yaw_limit and self.pitch_min <= pitch <= self.pitch_max

    def looking_at(self, center: Sequence[float], yaw: Optional[float], pitch: Optional[float],
                   frame_w: int, frame_h: int) -> Optional[bool]:
        if yaw is None or pitch is None:
            return None
        if self.polygon is None:
            return self.facing_camera(yaw, pitch)
        f = forward_vector(yaw, pitch)
        d = np.array([f[0], f[1]])
        if np.linalg.norm(d) < self.min_ray:
            return self.camera_is_board and self.facing_camera(yaw, pitch)
        poly = self.polygon * np.array([frame_w, frame_h])
        return _ray_hits_polygon(np.asarray(center, dtype=np.float64), d / np.linalg.norm(d), poly)
