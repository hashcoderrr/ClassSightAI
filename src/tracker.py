"""ByteTrack multi-object tracker (pure NumPy / SciPy implementation).

Follows Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
Detection Box" (ECCV 2022): a constant-velocity Kalman filter per track and a
two-stage IoU association - first with high-score detections, then the
remaining tracks with low-score detections - so that briefly occluded or blurry
faces keep their track ID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """Kalman filter on (cx, cy, aspect, height) + velocities, as in SORT/ByteTrack."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._motion = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion[i, ndim + i] = dt
        self._update = np.eye(ndim, 2 * ndim)
        self._std_pos = 1.0 / 20
        self._std_vel = 1.0 / 160

    def initiate(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        std = [self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
               self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]
        mean = self._motion @ mean
        cov = self._motion @ cov @ self._motion.T + np.diag(np.square(std))
        return mean, cov

    def update(self, mean: np.ndarray, cov: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = mean[3]
        r = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]))
        proj_mean = self._update @ mean
        proj_cov = self._update @ cov @ self._update.T + r
        gain = np.linalg.solve(proj_cov, (cov @ self._update.T).T).T
        mean = mean + gain @ (z - proj_mean)
        cov = cov - gain @ proj_cov @ gain.T
        return mean, cov


def xyxy_to_xyah(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def xyah_to_xyxy(m: np.ndarray) -> np.ndarray:
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    ix1, iy1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    ix2, iy2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return (inter / (area_a + area_b - inter + 1e-9)).astype(np.float32)


def linear_assignment(cost: np.ndarray, thresh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching keeping only pairs with cost <= thresh."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= thresh]
    mr = {m[0] for m in matches}
    mc = {m[1] for m in matches}
    return (matches, [i for i in range(cost.shape[0]) if i not in mr],
            [j for j in range(cost.shape[1]) if j not in mc])


@dataclass
class Track:
    track_id: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    start_frame: int
    last_frame: int
    hits: int = 1
    state: str = "tracked"      # tracked | lost | removed
    activated: bool = False

    @property
    def bbox(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)


@dataclass
class TrackOutput:
    """Association result for the current frame."""
    track_id: int
    bbox: np.ndarray
    det_index: int
    score: float


@dataclass
class ByteTracker:
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer: int = 30
    kf: KalmanFilter = field(default_factory=KalmanFilter)
    tracks: list[Track] = field(default_factory=list)
    frame_id: int = 0
    _next_id: int = 1

    @classmethod
    def from_config(cls, tcfg) -> "ByteTracker":
        return cls(track_high_thresh=float(tcfg.track_high_thresh), track_low_thresh=float(tcfg.track_low_thresh),
                   new_track_thresh=float(tcfg.new_track_thresh), match_thresh=float(tcfg.match_thresh),
                   track_buffer=int(tcfg.track_buffer))

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> tuple[list[TrackOutput], list[int]]:
        """Advance one (processed) frame.

        Args:
            boxes: (N, 4) detections xyxy.
            scores: (N,) detection confidences.

        Returns:
            (outputs for active tracks matched to a detection this frame,
             IDs of tracks removed in this step).
        """
        self.frame_id += 1
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        for t in self.tracks:
            if t.state != "removed":
                t.mean, t.cov = self.kf.predict(t.mean, t.cov)

        high = np.where(scores >= self.track_high_thresh)[0]
        low = np.where((scores >= self.track_low_thresh) & (scores < self.track_high_thresh))[0]
        pool = [t for t in self.tracks if t.state in ("tracked", "lost")]
        confirmed = [t for t in pool if t.activated]
        unconfirmed = [t for t in pool if not t.activated]
        outputs: list[TrackOutput] = []

        def apply(track: Track, di: int) -> None:
            track.mean, track.cov = self.kf.update(track.mean, track.cov, xyxy_to_xyah(boxes[di]))
            track.score = float(scores[di])
            track.last_frame = self.frame_id
            track.hits += 1
            track.state = "tracked"
            track.activated = True
            outputs.append(TrackOutput(track.track_id, boxes[di].copy(), int(di), float(scores[di])))

        # stage 1: confirmed tracks vs high-score detections (IoU fused with score)
        tb = np.array([t.bbox for t in confirmed]).reshape(-1, 4)
        cost = 1.0 - iou_matrix(tb, boxes[high]) * scores[high][None, :] if len(high) else \
            np.zeros((len(confirmed), 0))
        matches, u_tracks, u_high = linear_assignment(cost, self.match_thresh)
        for ti, dj in matches:
            apply(confirmed[ti], int(high[dj]))

        # stage 2: leftover *tracked* tracks vs low-score detections
        remain = [confirmed[i] for i in u_tracks if confirmed[i].state == "tracked"]
        lost_left = [confirmed[i] for i in u_tracks if confirmed[i].state != "tracked"]
        tb2 = np.array([t.bbox for t in remain]).reshape(-1, 4)
        cost2 = 1.0 - iou_matrix(tb2, boxes[low])
        m2, u2, _ = linear_assignment(cost2, 0.5)
        for ti, dj in m2:
            apply(remain[ti], int(low[dj]))
        for i in u2:
            remain[i].state = "lost"

        # stage 3: unconfirmed tracks vs remaining high detections
        left_high = [high[j] for j in u_high]
        tb3 = np.array([t.bbox for t in unconfirmed]).reshape(-1, 4)
        cost3 = 1.0 - iou_matrix(tb3, boxes[left_high]) if left_high else np.zeros((len(unconfirmed), 0))
        m3, u3, u_new = linear_assignment(cost3, 0.7)
        for ti, dj in m3:
            apply(unconfirmed[ti], int(left_high[dj]))
        removed: list[int] = []
        for i in u3:
            unconfirmed[i].state = "removed"
            removed.append(unconfirmed[i].track_id)

        # new tracks
        for j in u_new:
            di = int(left_high[j])
            if scores[di] < self.new_track_thresh:
                continue
            mean, cov = self.kf.initiate(xyxy_to_xyah(boxes[di]))
            t = Track(self._new_id(), mean, cov, float(scores[di]), self.frame_id, self.frame_id)
            # first frame of a video activates immediately (ByteTrack convention)
            t.activated = self.frame_id == 1
            self.tracks.append(t)
            outputs.append(TrackOutput(t.track_id, boxes[di].copy(), di, float(scores[di])))

        # expire lost tracks
        for t in lost_left + [tt for tt in pool if tt.state == "lost"]:
            if t.state == "lost" and self.frame_id - t.last_frame > self.track_buffer:
                t.state = "removed"
                if t.track_id not in removed:
                    removed.append(t.track_id)
        self.tracks = [t for t in self.tracks if t.state != "removed"]
        return outputs, removed

    def active_ids(self) -> set[int]:
        return {t.track_id for t in self.tracks}

    def get(self, track_id: int) -> Optional[Track]:
        for t in self.tracks:
            if t.track_id == track_id:
                return t
        return None
