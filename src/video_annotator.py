"""Annotated output video: boxes, labels and a live 'Present: X / Y' counter."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

GREEN = (40, 200, 40)
RED = (40, 40, 230)
ORANGE = (0, 165, 255)


@dataclass
class BoxLabel:
    bbox: np.ndarray
    track_id: int
    label: Optional[str]      # "Name (Roll)" when recognised
    flagged: bool = False


def draw_annotations(frame: np.ndarray, boxes: Sequence[BoxLabel], present: int, total: int,
                     ts: Optional[float] = None) -> np.ndarray:
    """Draw boxes and the counter overlay on a copy of ``frame``."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(1.0, w / 1600))
    thick = max(1, int(round(2 * scale)))
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.bbox[:4]]
        color = ORANGE if b.flagged else (GREEN if b.label else RED)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = b.label if b.label else f"Unknown #{b.track_id}"
        if b.flagged:
            text += " [DUP?]"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
        ty = max(th + 4, y1)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty), color, -1)
        cv2.putText(out, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    banner = f"Present: {present} / {total}"
    if ts is not None:
        s = int(ts)
        banner += f"   t={s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    fs = 0.9 * max(scale, 0.6)
    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    overlay = out.copy()
    cv2.rectangle(overlay, (10, 10), (30 + tw, 30 + th), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    cv2.putText(out, banner, (20, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class VideoAnnotator:
    """Writes annotated frames to an MP4, trying browser-friendly codecs first."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int],
                 codecs: Sequence[str] = ("avc1", "mp4v")) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.writer: Optional[cv2.VideoWriter] = None
        fps = fps if fps and fps > 0 else 25.0
        for codec in codecs:
            wr = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*codec), fps, size)
            if wr.isOpened():
                self.writer = wr
                self.codec = codec
                log.info("Annotated video -> %s (codec %s, %.1f fps)", self.path, codec, fps)
                break
            wr.release()
        if self.writer is None:
            log.error("Could not open a VideoWriter for %s; annotated video disabled.", self.path)

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            return
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def close(self) -> Optional[Path]:
        if self.writer is None:
            return None
        self.writer.release()
        self.writer = None
        return self.path
