"""Shared data types and behaviour vocabulary."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

LOOKING_BOARD = "looking_board"
WRITING = "writing"
LOOKING_AWAY = "looking_away"
SLEEPING = "sleeping"
TALKING = "talking"
HAND_RAISED = "hand_raised"
PHONE = "phone"
STANDING = "standing"
OUT_OF_FRAME = "out_of_frame"
OTHER = "other"

BEHAVIOURS = [LOOKING_BOARD, WRITING, HAND_RAISED, LOOKING_AWAY, TALKING, PHONE, SLEEPING, STANDING,
              OTHER, OUT_OF_FRAME]
DISPLAY = {
    LOOKING_BOARD: "Looking at board",
    WRITING: "Looking down / writing",
    LOOKING_AWAY: "Looking away / side",
    SLEEPING: "Head down / sleeping",
    TALKING: "Talking",
    HAND_RAISED: "Hand raised",
    PHONE: "Using phone",
    STANDING: "Standing / moving",
    OUT_OF_FRAME: "Out of frame / absent",
    OTHER: "Other / neutral",
}
# event pop-up tags (OpenCV Hershey fonts cannot draw emoji)
POPUP = {HAND_RAISED: "[HAND] Hand raised", PHONE: "[PHONE] Phone detected", SLEEPING: "[ZZZ] Head down",
         STANDING: "[UP] Standing", TALKING: "[TALK] Talking", LOOKING_AWAY: "[<>] Looking away"}

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, L_HIP, R_HIP = range(13)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4),
            (11, 13), (13, 15), (12, 14), (14, 16)]


@dataclass
class PersonPose:
    """One person from the body-pose model."""
    bbox: np.ndarray            # (4,) xyxy
    keypoints: np.ndarray       # (17, 3) x, y, conf
    score: float = 1.0


@dataclass
class DetectedObject:
    label: str                  # canonical: phone | book | notebook | laptop | pen
    bbox: np.ndarray
    score: float


@dataclass
class FrameSignals:
    """Everything observed for one track in one processed frame."""
    track_id: int
    ts: float
    frame_idx: int
    bbox: np.ndarray                         # face box (or head box estimated from body)
    face_visible: bool = True
    det_score: float = 1.0
    yaw: Optional[float] = None              # deg, + = facing image right
    pitch: Optional[float] = None            # deg, + = looking down
    roll: Optional[float] = None
    pose_source: str = "none"                # landmarks68 | kps5 | body | none
    ear: Optional[float] = None
    mar: Optional[float] = None
    keypoints: Optional[np.ndarray] = None   # (17, 3)
    person_bbox: Optional[np.ndarray] = None
    objects: list[str] = field(default_factory=list)
    on_board: Optional[bool] = None
    frame_w: int = 0
    frame_h: int = 0

    @property
    def center(self) -> tuple[float, float]:
        b = self.bbox
        return float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)

    @property
    def face_h(self) -> float:
        return float(max(1.0, self.bbox[3] - self.bbox[1]))

    def kp(self, i: int, min_conf: float = 0.3) -> Optional[np.ndarray]:
        if self.keypoints is None or self.keypoints[i, 2] < min_conf:
            return None
        return self.keypoints[i, :2]

    def to_row(self) -> dict[str, Any]:
        """Flat row for the Parquet cache."""
        return {
            "track_id": self.track_id, "ts": self.ts, "frame_idx": self.frame_idx,
            "x1": float(self.bbox[0]), "y1": float(self.bbox[1]), "x2": float(self.bbox[2]),
            "y2": float(self.bbox[3]), "face_visible": self.face_visible, "det_score": self.det_score,
            "yaw": self.yaw, "pitch": self.pitch, "roll": self.roll, "pose_source": self.pose_source,
            "ear": self.ear, "mar": self.mar,
            "keypoints": None if self.keypoints is None else ",".join(f"{v:.2f}" for v in self.keypoints.ravel()),
            "person_bbox": None if self.person_bbox is None else ",".join(f"{v:.1f}" for v in self.person_bbox),
            "objects": ",".join(self.objects), "frame_w": self.frame_w, "frame_h": self.frame_h,
        }

    @classmethod
    def from_row(cls, r: dict[str, Any]) -> "FrameSignals":
        def opt(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return None if np.isnan(f) else f

        kp = r.get("keypoints")
        pb = r.get("person_bbox")
        objs = r.get("objects")
        return cls(
            track_id=int(r["track_id"]), ts=float(r["ts"]), frame_idx=int(r["frame_idx"]),
            bbox=np.array([r["x1"], r["y1"], r["x2"], r["y2"]], dtype=np.float32),
            face_visible=bool(r["face_visible"]), det_score=float(r["det_score"]),
            yaw=opt(r.get("yaw")), pitch=opt(r.get("pitch")), roll=opt(r.get("roll")),
            pose_source=str(r.get("pose_source") or "none"), ear=opt(r.get("ear")), mar=opt(r.get("mar")),
            keypoints=np.array([float(v) for v in kp.split(",")], dtype=np.float32).reshape(17, 3)
            if isinstance(kp, str) and kp else None,
            person_bbox=np.array([float(v) for v in pb.split(",")], dtype=np.float32)
            if isinstance(pb, str) and pb else None,
            objects=[o for o in objs.split(",") if o] if isinstance(objs, str) else [],
            frame_w=int(r.get("frame_w") or 0), frame_h=int(r.get("frame_h") or 0),
        )


@dataclass
class Classification:
    state: str
    confidence: float
