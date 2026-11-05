"""Body pose estimation (17 COCO keypoints) with Ultralytics YOLO-pose.

Optional dependency: ``pip install ultralytics``. Without it the analyser runs
in face-only mode (hand-raise and posture rules are then disabled).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .signals import PersonPose

log = logging.getLogger(__name__)


def torch_device(use_gpu: bool) -> str:
    try:
        import torch
        if use_gpu and torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def load_yolo(model_path: str | Path):
    """Load an Ultralytics model; known asset names (yolo11n-pose.pt, ...) are downloaded once."""
    from ultralytics import YOLO
    p = Path(model_path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        from ultralytics.utils.downloads import attempt_download_asset
        attempt_download_asset(str(p))
    return YOLO(str(p))


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(self, frame: np.ndarray) -> list[PersonPose]:
        """Return all persons in the frame."""


class NullPoseEstimator(PoseEstimator):
    def estimate(self, frame: np.ndarray) -> list[PersonPose]:
        return []


class YoloPoseEstimator(PoseEstimator):
    def __init__(self, model_path: str | Path, conf: float = 0.3, imgsz: int = 960, use_gpu: bool = True) -> None:
        self.model = load_yolo(model_path)
        self.conf, self.imgsz = conf, imgsz
        self.device = torch_device(use_gpu)
        log.info("YOLO pose model %s on %s", Path(model_path).name, self.device)

    def estimate(self, frame: np.ndarray) -> list[PersonPose]:
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
        return [PersonPose(boxes[i].astype(np.float32),
                           np.concatenate([kxy[i], kc[i][:, None]], axis=1).astype(np.float32), float(scores[i]))
                for i in range(len(boxes))]


def build_pose_estimator(cfg: Any) -> PoseEstimator:
    pcfg = cfg.behaviour.pose
    if not pcfg.enabled:
        return NullPoseEstimator()
    try:
        path = Path(pcfg.model)
        if not path.is_absolute():
            path = cfg.path("data_dir") / "models" / path.name
        return YoloPoseEstimator(path, float(pcfg.conf), int(pcfg.imgsz), bool(cfg.models.use_gpu))
    except ImportError:
        log.warning("ultralytics not installed - body pose disabled (pip install ultralytics).")
    except Exception as exc:
        log.warning("Pose model unavailable (%s) - body pose disabled.", exc)
    return NullPoseEstimator()


def associate_persons(face_boxes: list[np.ndarray], persons: list[PersonPose],
                      min_kp_conf: float = 0.3) -> dict[int, int]:
    """Greedy face -> person assignment. Returns {face_index: person_index}.

    A person matches a face when its head keypoints (nose/eyes/ears) fall inside
    the face box enlarged 1.6x, or - if no head keypoint is visible - when the
    face centre lies in the top 45% of the person box.
    """
    cands: list[tuple[float, int, int]] = []
    for fi, fb in enumerate(face_boxes):
        cx, cy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
        fw, fh = fb[2] - fb[0], fb[3] - fb[1]
        size = max(fw, fh, 1.0)
        for pi, p in enumerate(persons):
            head = p.keypoints[:5]
            vis = head[head[:, 2] >= min_kp_conf]
            if len(vis):
                hx, hy = vis[:, 0].mean(), vis[:, 1].mean()
                if abs(hx - cx) <= 0.8 * fw * 1.6 and abs(hy - cy) <= 0.8 * fh * 1.6:
                    cands.append((float(np.hypot(hx - cx, hy - cy)) / size, fi, pi))
            else:
                x1, y1, x2, y2 = p.bbox
                if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.45 * (y2 - y1):
                    cands.append((1.0 + abs(cx - (x1 + x2) / 2) / size, fi, pi))
    out: dict[int, int] = {}
    used: set[int] = set()
    for _, fi, pi in sorted(cands):
        if fi not in out and pi not in used:
            out[fi] = pi
            used.add(pi)
    return out


def head_box_from_pose(p: PersonPose, min_conf: float = 0.3) -> Optional[np.ndarray]:
    """Approximate head box from body keypoints (face not detected)."""
    k = p.keypoints
    head = k[:5][k[:5, 2] >= min_conf]
    sho = k[5:7][k[5:7, 2] >= min_conf]
    if len(sho) == 2:
        sw = abs(sho[0, 0] - sho[1, 0])
        if len(head):
            cx, cy = head[:, 0].mean(), head[:, 1].mean()
        else:
            cx, cy = sho[:, 0].mean(), sho[:, 1].mean() - 0.6 * sw
        half = max(8.0, 0.35 * sw)
        return np.array([cx - half, cy - half * 1.2, cx + half, cy + half * 1.2], dtype=np.float32)
    if len(head) >= 2:
        cx, cy = head[:, 0].mean(), head[:, 1].mean()
        span = max(8.0, float(np.ptp(head[:, 0])))
        return np.array([cx - span, cy - span * 1.2, cx + span, cy + span * 1.2], dtype=np.float32)
    return None
