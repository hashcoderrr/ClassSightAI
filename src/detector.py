"""Face detection (SCRFD) and embedding extraction (ArcFace) via InsightFace.

Only the detection and recognition models of the ``buffalo_l`` pack are loaded.
Detection supports frame upscaling and tiled detection so that small faces on
the back benches of wide-angle CCTV footage are still found. Embeddings are
computed on demand (only for the best crops of each track), not for every face
in every frame.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)
# insightface's face_align uses a scikit-image API that is deprecated (harmless)
warnings.filterwarnings("ignore", message=".*estimate.* is deprecated", category=FutureWarning)


@dataclass
class Detection:
    """A detected face in frame coordinates."""
    bbox: np.ndarray          # (4,) x1, y1, x2, y2
    score: float
    kps: Optional[np.ndarray]  # (5, 2) landmarks

    @property
    def size(self) -> float:
        return float(min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))


def select_providers(use_gpu: bool) -> tuple[list[str], int]:
    """Return ONNX Runtime providers and InsightFace ctx_id, falling back to CPU."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:  # pragma: no cover - onnxruntime missing
        available = []
    if use_gpu and "CUDAExecutionProvider" in available:
        log.info("Using CUDAExecutionProvider (GPU).")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], 0
    if use_gpu:
        log.info("CUDA not available - falling back to CPU.")
    return ["CPUExecutionProvider"], -1


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.4) -> list[int]:
    """Plain NMS; returns kept indices sorted by score."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tiles (x1, y1, x2, y2) covering the full image."""
    step = max(1, int(tile * (1.0 - overlap)))

    def starts(total: int) -> list[int]:
        if total <= tile:
            return [0]
        s = list(range(0, total - tile, step))
        s.append(total - tile)
        return sorted(set(s))

    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height) for x in starts(width)]


class FaceEngine:
    """Wraps InsightFace SCRFD detector + ArcFace recogniser."""

    def __init__(self, cfg: Any) -> None:
        try:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except ImportError as exc:  # pragma: no cover
            raise ImportError("insightface is not installed. Run: pip install -r requirements.txt") from exc
        self._face_align = face_align
        mcfg = cfg.models
        providers, ctx_id = select_providers(bool(mcfg.use_gpu))
        if ctx_id == 0:
            ctx_id = int(mcfg.get("gpu_id", 0))
        root = cfg.path("insightface_root")
        root.mkdir(parents=True, exist_ok=True)
        self.app = FaceAnalysis(name=mcfg.name, root=str(root), providers=providers,
                                allowed_modules=["detection", "recognition"])
        self.det_size = tuple(int(v) for v in mcfg.det_size)
        self.app.prepare(ctx_id=ctx_id, det_thresh=float(mcfg.det_thresh), det_size=self.det_size)
        self.det_model = self.app.det_model
        self.rec_model = self.app.models["recognition"]
        vcfg = cfg.video
        self.upscale = float(vcfg.get("upscale_factor", 1.0))
        self.tiled = bool(vcfg.get("tiled_detection", False))
        self.tile_size = int(vcfg.get("tile_size", 640))
        self.tile_overlap = float(vcfg.get("tile_overlap", 0.2))
        self.min_face = float(vcfg.get("min_face_size", 16))
        log.info("FaceEngine ready (model=%s, providers=%s, tiled=%s, upscale=%.2f)",
                 mcfg.name, providers, self.tiled, self.upscale)

    # ---------------------------------------------------------------- detection
    def _detect_raw(self, img: np.ndarray, input_size: Optional[tuple[int, int]] = None
                    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        bboxes, kpss = self.det_model.detect(img, input_size=input_size, max_num=0, metric="default")
        return bboxes, kpss

    def detect(self, frame: np.ndarray, tiled: Optional[bool] = None,
               upscale: Optional[float] = None) -> list[Detection]:
        """Detect faces; returns detections in original-frame coordinates."""
        if frame is None or frame.size == 0:
            return []
        tiled = self.tiled if tiled is None else tiled
        scale = self.upscale if upscale is None else upscale
        img = frame
        if scale and abs(scale - 1.0) > 1e-3:
            img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        all_boxes: list[np.ndarray] = []
        all_kps: list[np.ndarray] = []
        if tiled and (w > self.tile_size or h > self.tile_size):
            # full-frame pass catches large/near faces, tiles catch small/far ones
            regions = [(0, 0, w, h)] + tile_grid(w, h, self.tile_size, self.tile_overlap)
        else:
            regions = [(0, 0, w, h)]
        for (x1, y1, x2, y2) in regions:
            sub = img[y1:y2, x1:x2]
            input_size = None
            if (x2 - x1, y2 - y1) != (w, h):
                input_size = (self.tile_size, self.tile_size)
            b, k = self._detect_raw(sub, input_size)
            if b is None or len(b) == 0:
                continue
            b = b.copy()
            b[:, [0, 2]] += x1
            b[:, [1, 3]] += y1
            all_boxes.append(b)
            if k is not None:
                k = k.copy()
                k[:, :, 0] += x1
                k[:, :, 1] += y1
                all_kps.append(k)
            else:
                all_kps.append(np.zeros((len(b), 5, 2), dtype=np.float32))
        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes)
        kps = np.concatenate(all_kps)
        keep = nms(boxes[:, :4], boxes[:, 4]) if len(regions) > 1 else list(range(len(boxes)))
        out: list[Detection] = []
        for i in keep:
            bbox = boxes[i, :4] / scale
            det = Detection(bbox=bbox.astype(np.float32), score=float(boxes[i, 4]),
                            kps=(kps[i] / scale).astype(np.float32))
            if det.size >= self.min_face:
                out.append(det)
        return out

    # ---------------------------------------------------------------- embedding
    def align(self, frame: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """112x112 ArcFace-aligned crop."""
        return self._face_align.norm_crop(frame, landmark=np.asarray(kps, dtype=np.float32), image_size=112)

    def embed_aligned(self, aligned: Sequence[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings (N, 512) for aligned 112x112 crops."""
        if len(aligned) == 0:
            return np.zeros((0, 512), dtype=np.float32)
        feats = np.asarray(self.rec_model.get_feat(list(aligned)), dtype=np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
        return feats

    def embed(self, frame: np.ndarray, det: Detection) -> np.ndarray:
        return self.embed_aligned([self.align(frame, det.kps)])[0]
