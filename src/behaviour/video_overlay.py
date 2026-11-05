"""Annotated behaviour video: state-coloured boxes, labels, skeleton, head arrows,
event pop-ups, HUD with class-attention sparkline and a live distraction panel.

Two files are written in one pass: full resolution and a lightweight preview
(default 720p). If the preview was not encoded as H.264 and ``ffmpeg`` is
available, it is transcoded so it plays in browsers.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional

import cv2
import numpy as np

from ..video_annotator import VideoAnnotator
from .head_pose import head_arrow
from .signals import DISPLAY, POPUP, SKELETON

log = logging.getLogger(__name__)

GREEN, YELLOW, RED, GREY = (60, 190, 60), (0, 200, 230), (50, 50, 230), (160, 160, 160)


def band_color(score: Optional[float], attentive: float = 50.0, partial: float = 30.0) -> tuple[int, int, int]:
    if score is None:
        return GREY
    return GREEN if score >= attentive else (YELLOW if score >= partial else RED)


def find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def transcode_h264(src: Path) -> bool:
    """Re-encode to browser-friendly H.264 (+faststart). Returns True on success."""
    exe = find_ffmpeg()
    if not exe:
        return False
    tmp = src.with_name(src.stem + "_h264.mp4")
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    try:
        subprocess.run(cmd, check=True, timeout=3600)
        tmp.replace(src)
        return True
    except Exception as exc:
        log.warning("ffmpeg transcode failed: %s", exc)
        tmp.unlink(missing_ok=True)
        return False


@dataclass
class LiveTrack:
    label: str
    state: str
    score: Optional[float]
    bbox: np.ndarray
    ts: float
    keypoints: Optional[np.ndarray] = None
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    popup: str = ""
    popup_until: float = -1.0
    is_student: bool = False


@dataclass
class BehaviourOverlay:
    out_dir: Path
    fps: float
    size: tuple[int, int]
    codecs: list[str] = field(default_factory=lambda: ["avc1", "mp4v"])
    preview_height: int = 720
    skeleton: bool = True
    head_arrow: bool = True
    side_panel: bool = True
    hold_sec: float = 2.0
    popup_sec: float = 2.5
    attentive: float = 50.0
    partial: float = 30.0
    tracks: dict[int, LiveTrack] = field(default_factory=dict)
    class_hist: Deque[float] = field(default_factory=lambda: deque(maxlen=240))

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        w, h = self.size
        self.full_path = self.out_dir / "behaviour_full.mp4"
        self.full = VideoAnnotator(self.full_path, self.fps, (w, h), self.codecs)
        self.preview = None
        self.preview_path = self.full_path
        if h > self.preview_height:
            pw = int(round(w * self.preview_height / h / 2) * 2)
            self.preview_path = self.out_dir / "behaviour_preview.mp4"
            self.preview = VideoAnnotator(self.preview_path, self.fps, (pw, self.preview_height), self.codecs)
        self._last: Optional[np.ndarray] = None

    def update(self, tid: int, label: str, state: str, score: Optional[float], bbox: np.ndarray, ts: float,
               keypoints: Optional[np.ndarray], yaw: Optional[float], pitch: Optional[float],
               is_student: bool) -> None:
        prev = self.tracks.get(tid)
        lt = LiveTrack(label, state, score, bbox, ts, keypoints, yaw, pitch, is_student=is_student)
        if prev is not None:
            lt.popup, lt.popup_until = prev.popup, prev.popup_until
        if (prev is None or prev.state != state) and state in POPUP:
            lt.popup, lt.popup_until = POPUP[state], ts + self.popup_sec
        self.tracks[tid] = lt

    def class_attention(self, ts: float) -> Optional[float]:
        vals = [t.score for t in self.tracks.values()
                if t.is_student and t.score is not None and ts - t.ts <= self.hold_sec]
        return float(np.mean(vals)) if vals else None

    def push_class_point(self, ts: float) -> None:
        v = self.class_attention(ts)
        if v is not None:
            self.class_hist.append(v)

    # ------------------------------------------------------------------ drawing
    def draw(self, frame: np.ndarray, ts: float, present: int, total: int) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        sc = max(0.45, min(1.2, w / 1500))
        th = max(1, int(round(2 * sc)))
        live = {k: t for k, t in self.tracks.items() if ts - t.ts <= self.hold_sec}
        for t in live.values():
            col = band_color(t.score, self.attentive, self.partial)
            x1, y1, x2, y2 = [int(v) for v in t.bbox[:4]]
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
            if self.skeleton and t.keypoints is not None:
                k = t.keypoints
                for a, b in SKELETON:
                    if k[a, 2] >= 0.3 and k[b, 2] >= 0.3:
                        cv2.line(out, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])), col, 1,
                                 cv2.LINE_AA)
            if self.head_arrow and t.yaw is not None and t.pitch is not None:
                c = ((x1 + x2) / 2, (y1 + y2) / 2)
                tip = head_arrow(c, t.yaw, t.pitch, 1.6 * max(10, y2 - y1))
                cv2.arrowedLine(out, (int(c[0]), int(c[1])), tip, (255, 200, 0), th, cv2.LINE_AA, tipLength=0.25)
            score_txt = "--" if t.score is None else f"{t.score:.0f}"
            text = f"{t.label} | {DISPLAY.get(t.state, t.state)} | {score_txt}"
            fs = 0.45 * sc
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = max(tht + 4, y1)
            cv2.rectangle(out, (x1, ty - tht - 4), (x1 + tw + 4, ty), col, -1)
            cv2.putText(out, text, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 1, cv2.LINE_AA)
            if t.popup and ts <= t.popup_until:
                (pw, ph), _ = cv2.getTextSize(t.popup, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, 2)
                py = min(h - 4, y2 + ph + 10)
                cv2.rectangle(out, (x1, py - ph - 6), (x1 + pw + 8, py + 4), (30, 30, 30), -1)
                cv2.putText(out, t.popup, (x1 + 4, py), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * sc, (0, 255, 255), 2,
                            cv2.LINE_AA)
        out = self._hud(out, ts, present, total, live, sc)
        self._last = out
        self.full.write(out)
        if self.preview is not None:
            self.preview.write(out)
        return out

    def _hud(self, out: np.ndarray, ts: float, present: int, total: int, live: dict[int, LiveTrack],
             sc: float) -> np.ndarray:
        h, w = out.shape[:2]
        bar_h = int(46 * sc)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        att = self.class_attention(ts)
        s = int(ts)
        text = (f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}   Present {present}/{total}   "
                f"Class attention: {'--' if att is None else f'{att:.0f}%'}")
        panel_lines: list[str] = []
        if self.side_panel:
            cnt = Counter(t.state for t in live.values() if t.score is not None and t.score < self.attentive)
            panel_lines = [f"{DISPLAY.get(k, k)}: {v}" for k, v in cnt.most_common(5)]
            if panel_lines:
                pw = int(250 * sc)
                ph = int((len(panel_lines) + 1) * 22 * sc + 10)
                cv2.rectangle(overlay, (w - pw - 10, bar_h + 10), (w - 10, bar_h + 10 + ph), (20, 20, 20), -1)
        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
        cv2.putText(out, text, (12, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc, (255, 255, 255), 2,
                    cv2.LINE_AA)
        # sparkline of the class attention curve
        sw, sh = int(260 * sc), int(bar_h * 0.7)
        x0, y0 = w - sw - 14, int(bar_h * 0.15)
        cv2.rectangle(out, (x0, y0), (x0 + sw, y0 + sh), (90, 90, 90), 1)
        if len(self.class_hist) >= 2:
            vals = np.array(self.class_hist, dtype=np.float32)
            xs = np.linspace(x0, x0 + sw, len(vals))
            ys = y0 + sh - vals / 100.0 * sh
            cv2.polylines(out, [np.stack([xs, ys], 1).astype(np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
        if panel_lines:
            pw = int(250 * sc)
            yy = bar_h + 10 + int(22 * sc)
            cv2.putText(out, "Live distractions", (w - pw, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (0, 255, 255),
                        1, cv2.LINE_AA)
            for line in panel_lines:
                yy += int(22 * sc)
                cv2.putText(out, line, (w - pw, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * sc, (255, 255, 255), 1,
                            cv2.LINE_AA)
        return out

    def hold(self, frame: np.ndarray, ts: float, present: int, total: int) -> np.ndarray:
        """Skipped frame: redraw the latest overlay state on the new frame."""
        return self.draw(frame, ts, present, total)

    def close(self) -> tuple[Optional[Path], Optional[Path]]:
        full = self.full.close()
        prev = self.preview.close() if self.preview is not None else full
        if prev is not None and getattr(self.preview or self.full, "codec", "") != "avc1":
            if transcode_h264(prev):
                log.info("Preview transcoded to H.264 for the web player.")
        return full, prev


def build_overlay(cfg: Any, out_dir: Path, fps: float, size: tuple[int, int]) -> BehaviourOverlay:
    v = cfg.behaviour.video
    labels = dict(cfg.behaviour.scoring.labels)
    return BehaviourOverlay(out_dir, fps, size, list(cfg.annotate.get("codec_preference", ["avc1", "mp4v"])),
                            int(v.preview_height), bool(v.skeleton), bool(v.head_arrow), bool(v.side_panel),
                            float(v.get("hold_sec", 2.0)), float(v.get("popup_sec", 2.5)),
                            float(labels.get("Attentive", 50)), float(labels.get("Partially Attentive", 30)))
