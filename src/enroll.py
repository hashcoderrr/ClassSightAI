"""Student enrollment: validate photos, extract ArcFace embeddings, store the mean.

Each photo is checked for: readable image, exactly one face (largest face is used
and a warning raised if several are present), minimum face size and sharpness.
Embeddings far from the others (likely a wrong photo) are rejected as outliers.
Only the averaged, L2-normalised embedding is persisted - never the photos.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import cv2
import numpy as np

from .db import Database, Student
from .quality import laplacian_variance, safe_crop

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ImageInput = Union[str, Path, np.ndarray, bytes]


def read_image(src: ImageInput) -> Optional[np.ndarray]:
    """Read an image from a path (unicode-safe), raw bytes or an array."""
    try:
        if isinstance(src, np.ndarray):
            return src
        if isinstance(src, (bytes, bytearray)):
            data = np.frombuffer(src, dtype=np.uint8)
        else:
            data = np.fromfile(str(src), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        log.warning("Cannot read image %s: %s", src if not isinstance(src, (bytes, np.ndarray)) else "<bytes>", exc)
        return None


def limit_size(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    f = max_side / s
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def sample_video_frames(path: str | Path, n: int) -> list[np.ndarray]:
    """Evenly sample ``n`` frames from a (selfie) video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames: list[np.ndarray] = []
    if total > 0:
        for idx in np.linspace(0, max(0, total - 1), num=min(n * 2, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
    else:  # unknown length (some webm): read sequentially
        while len(frames) < n * 2:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    cap.release()
    return frames


@dataclass
class EnrollResult:
    roll_no: str
    accepted: int = 0
    rejected: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    success: bool = False
    message: str = ""


class Enroller:
    def __init__(self, cfg: Any, db: Database, engine: Any) -> None:
        self.cfg = cfg
        self.ecfg = cfg.enrollment
        self.db = db
        self.engine = engine

    def embed_image(self, img: np.ndarray, label: str, result: EnrollResult) -> Optional[np.ndarray]:
        """Validate one image and return its embedding (or None if rejected)."""
        if img is None or img.size == 0:
            result.rejected.append(f"{label}: unreadable / corrupted image")
            return None
        img = limit_size(img)
        dets = self.engine.detect(img, tiled=False, upscale=1.0)
        if not dets:
            result.rejected.append(f"{label}: no face found")
            return None
        if len(dets) > 1:
            result.warnings.append(f"{label}: {len(dets)} faces found - using the largest one")
        det = max(dets, key=lambda d: d.size)
        if det.size < float(self.ecfg.min_face_size):
            result.rejected.append(f"{label}: face too small ({det.size:.0f}px < {self.ecfg.min_face_size}px)")
            return None
        crop = safe_crop(img, det.bbox, pad=0.0)
        blur = laplacian_variance(cv2.resize(crop, (112, 112)))
        if blur < float(self.ecfg.min_blur_var):
            result.rejected.append(f"{label}: too blurry (sharpness {blur:.0f} < {self.ecfg.min_blur_var})")
            return None
        return self.engine.embed(img, det)

    def enroll(self, roll_no: str, name: str, class_name: str, images: Sequence[ImageInput] = (),
               video: Optional[str | Path] = None, consent: bool = False, guardian: str = "",
               labels: Optional[Sequence[str]] = None, behaviour_consent: Optional[bool] = None) -> EnrollResult:
        """Enroll (or re-enroll) one student.

        Args:
            roll_no, name, class_name: Student identity.
            images: Paths / bytes / arrays of face photos.
            video: Optional selfie video to sample frames from.
            consent: Explicit consent obtained (mandatory, DPDP Act 2023).
            guardian: Optional guardian e-mail / phone for absence alerts.
            labels: Display names for ``images`` (e.g. uploaded file names).
            behaviour_consent: Guardian consent for behaviour analytics; None keeps the stored value.
        """
        roll_no, name, class_name = str(roll_no).strip(), str(name).strip(), str(class_name).strip()
        res = EnrollResult(roll_no)
        if not roll_no or not name or not class_name:
            res.message = "Roll No, Name and Class are required."
            return res
        if not consent:
            res.message = "Consent is required before storing biometric data (DPDP Act)."
            return res
        inputs: list[tuple[str, Optional[np.ndarray]]] = []
        for i, src in enumerate(images):
            lab = labels[i] if labels and i < len(labels) else (
                Path(src).name if isinstance(src, (str, Path)) else f"image_{i + 1}")
            inputs.append((lab, read_image(src)))
        if video:
            try:
                for j, f in enumerate(sample_video_frames(video, int(self.ecfg.video_sample_frames))):
                    inputs.append((f"video_frame_{j + 1}", f))
            except ValueError as exc:
                res.rejected.append(str(exc))
        max_imgs = int(self.ecfg.max_images)
        embs: list[np.ndarray] = []
        for lab, img in inputs:
            if len(embs) >= max_imgs:
                res.warnings.append(f"{lab}: skipped (max {max_imgs} images)")
                continue
            e = self.embed_image(img, lab, res)
            if e is not None:
                embs.append(e)
        # outlier rejection: drop embeddings that disagree with the consensus
        if len(embs) >= 3:
            m = np.stack(embs)
            mean = m.mean(axis=0)
            mean /= np.linalg.norm(mean) + 1e-12
            sims = m @ mean
            keep = sims >= 0.35
            if (~keep).any():
                res.warnings.append(f"{int((~keep).sum())} photo(s) rejected as outliers (different person?)")
            embs = [e for e, k in zip(embs, keep) if k]
        res.accepted = len(embs)
        if len(embs) < int(self.ecfg.min_images):
            res.message = (f"Only {len(embs)} usable face image(s); at least {self.ecfg.min_images} required.")
            return res
        mean = np.mean(np.stack(embs), axis=0)
        mean /= np.linalg.norm(mean) + 1e-12
        if behaviour_consent is None:
            existing = self.db.get_student(roll_no)
            behaviour_consent = bool(existing.behaviour_consent) if existing else False
        self.db.upsert_student(Student(roll_no=roll_no, name=name, class_name=class_name, embedding=mean,
                                       num_images=len(embs), consent=True,
                                       consent_date=datetime.now().isoformat(timespec="seconds"),
                                       guardian=guardian, behaviour_consent=bool(behaviour_consent)))
        res.success = True
        res.message = f"Enrolled {roll_no} - {name} ({class_name}) from {len(embs)} images."
        log.info(res.message)
        return res

    def enroll_folder(self, dataset_dir: str | Path, class_name: Optional[str] = None,
                      consent: bool = False, behaviour_consent: Optional[bool] = None) -> list[EnrollResult]:
        """Bulk-enroll from ``dataset/<RollNo>_<Name>/*.jpg`` (selfie videos also accepted)."""
        root = Path(dataset_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"Dataset folder not found: {root}")
        class_name = class_name or str(self.ecfg.default_class)
        results: list[EnrollResult] = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            roll, name = parse_folder_name(sub.name)
            if roll is None:
                log.warning("Skipping folder %s (expected <RollNo>_<Name>)", sub.name)
                continue
            imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            vids = sorted(p for p in sub.iterdir() if p.suffix.lower() in VIDEO_EXTS)
            r = self.enroll(roll, name, class_name, images=imgs, video=vids[0] if vids else None, consent=consent,
                            behaviour_consent=behaviour_consent)
            results.append(r)
        return results


def parse_folder_name(folder: str) -> tuple[Optional[str], str]:
    """``'23_Aditya_Kumar' -> ('23', 'Aditya Kumar')``."""
    m = re.match(r"^\s*([^_]+)_(.+)$", folder)
    if not m:
        return None, ""
    return m.group(1).strip(), m.group(2).replace("_", " ").strip()


def iter_dataset(root: str | Path) -> Iterable[tuple[str, str, list[Path]]]:
    for sub in sorted(Path(root).iterdir()):
        if sub.is_dir():
            roll, name = parse_folder_name(sub.name)
            if roll:
                yield roll, name, sorted(p for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTS)
