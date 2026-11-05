"""Object detection (phone, book/notebook, laptop, pen) and student association.

The default Ultralytics COCO model knows ``cell phone``, ``book`` and
``laptop``. COCO has no ``pen`` class; point ``behaviour.objects.model`` to a
custom-trained model to get pens/notebooks - the class mapping lives in config.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .pose_estimator import load_yolo, torch_device
from .signals import L_WRI, R_WRI, DetectedObject, PersonPose

log = logging.getLogger(__name__)


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        """Return objects with canonical labels."""


class NullObjectDetector(ObjectDetector):
    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        return []


class YoloObjectDetector(ObjectDetector):
    def __init__(self, model_path: str | Path, class_map: dict[str, list[str]], conf: float = 0.3,
                 imgsz: int = 960, use_gpu: bool = True) -> None:
        self.model = load_yolo(model_path)
        names = self.model.names
        lookup = {str(v).lower(): int(k) for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
        self.id_to_label: dict[int, str] = {}
        for canon, aliases in class_map.items():
            for a in aliases:
                if str(a).lower() in lookup:
                    self.id_to_label.setdefault(lookup[str(a).lower()], canon)
        self.conf, self.imgsz = conf, imgsz
        self.device = torch_device(use_gpu)
        log.info("YOLO object model %s: tracking %s", Path(model_path).name, sorted(set(self.id_to_label.values())))

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        if not self.id_to_label:
            return []
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
                                 classes=sorted(self.id_to_label))[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        sc = res.boxes.conf.cpu().numpy()
        return [DetectedObject(self.id_to_label[c], b.astype(np.float32), float(s))
                for b, c, s in zip(boxes, cls, sc) if c in self.id_to_label]


def build_object_detector(cfg: Any) -> ObjectDetector:
    ocfg = cfg.behaviour.objects
    if not ocfg.enabled:
        return NullObjectDetector()
    try:
        path = Path(ocfg.model)
        if not path.is_absolute():
            path = cfg.path("data_dir") / "models" / path.name
        cmap = {k: list(v) for k, v in ocfg.classes.items()}
        return YoloObjectDetector(path, cmap, float(ocfg.conf), int(ocfg.imgsz), bool(cfg.models.use_gpu))
    except ImportError:
        log.warning("ultralytics not installed - object detection disabled.")
    except Exception as exc:
        log.warning("Object model unavailable (%s) - object detection disabled.", exc)
    return NullObjectDetector()


def associate_objects(objects: list[DetectedObject], face_boxes: list[np.ndarray],
                      persons: list[Optional[PersonPose]], hand_dist: float = 1.5,
                      min_kp_conf: float = 0.3) -> dict[int, list[str]]:
    """Assign each object to the nearest student. Returns {face_index: [labels]}.

    Distance is measured from the object centre to the student's wrists (in face
    heights) when body keypoints exist; otherwise to the desk region below the face
    (or the face itself, for a phone held to the ear).
    """
    out: dict[int, list[str]] = {}
    for obj in objects:
        ox, oy = (obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2
        best: Optional[tuple[float, int]] = None
        for fi, fb in enumerate(face_boxes):
            fh = max(1.0, float(fb[3] - fb[1]))
            fcx, fcy = (fb[0] + fb[2]) / 2, (fb[1] + fb[3]) / 2
            person = persons[fi] if fi < len(persons) else None
            dists: list[float] = []
            if person is not None:
                for w in (L_WRI, R_WRI):
                    if person.keypoints[w, 2] >= min_kp_conf:
                        dists.append(float(np.hypot(person.keypoints[w, 0] - ox, person.keypoints[w, 1] - oy)) / fh)
            if not dists:
                # desk region: below the face, within +-1.5 face widths, up to 3.5 face heights down
                dx = abs(ox - fcx) / fh
                dy = (oy - fcy) / fh
                if dx <= 1.5 and -0.8 <= dy <= 3.5:
                    dists.append(np.hypot(dx, max(0.0, dy - 1.0)) * 0.9)
            if dists:
                d = min(dists)
                if d <= hand_dist and (best is None or d < best[0]):
                    best = (d, fi)
        if best is not None:
            out.setdefault(best[1], []).append(obj.label)
    return out
