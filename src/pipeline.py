"""End-to-end processing: video / CCTV -> detection -> tracking -> voting -> reports."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np

from .attendance import AttendanceSession, Crop, build_rows
from .db import Database
from .quality import enhance_low_light, quality_score, safe_crop
from .recognizer import FaceGallery
from .report_excel import generate_report, safe_name
from .tracker import ByteTracker
from .video_annotator import BoxLabel, VideoAnnotator, draw_annotations

log = logging.getLogger(__name__)

Source = Union[str, int, Path]
ProgressCB = Callable[[dict[str, Any]], None]
FrameCB = Callable[[np.ndarray], None]


def is_stream(source: Source) -> bool:
    if isinstance(source, int):
        return True
    s = str(source).strip().lower()
    return s.isdigit() or s.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://"))


def open_capture(source: Source) -> cv2.VideoCapture:
    src: Union[str, int] = int(source) if (isinstance(source, int) or str(source).isdigit()) else str(source)
    if isinstance(src, str) and not is_stream(src) and not Path(src).exists():
        raise FileNotFoundError(f"Video not found: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source} (corrupted file or unreachable stream)")
    return cap


def thumb_of(img: np.ndarray, max_side: int = 160) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side or s == 0:
        return img
    f = max_side / s
    return cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)


def write_image(path: Path, img: np.ndarray) -> bool:
    """Unicode-safe imwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return bool(ok)


class PrefetchCapture:
    """Decodes frames in a background thread (OpenCV releases the GIL while decoding)."""

    def __init__(self, cap: cv2.VideoCapture, size: int = 64) -> None:
        self.cap = cap
        self._q: "queue.Queue[tuple[bool, Optional[np.ndarray]]]" = queue.Queue(maxsize=size)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        misses = 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            item = (ok, frame if ok else None)
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue
            misses = misses + 1 if not ok else 0
            if misses > 40:   # end of file (plus tolerance for corrupted frames)
                break
        try:
            self._q.put((False, None), timeout=1.0)
        except queue.Full:
            pass

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._t.is_alive() and self._q.empty():
            return False, None
        try:
            return self._q.get(timeout=30.0)
        except queue.Empty:
            return False, None

    def get(self, prop: int) -> float:
        return self.cap.get(prop)

    def release(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)
        self.cap.release()


@dataclass
class ProcessResult:
    session_id: int
    report_path: Optional[Path]
    video_path: Optional[Path]
    output_dir: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    unknown_count: int = 0
    conflicts: int = 0
    duration_sec: float = 0.0
    status: str = "completed"
    behaviour: Optional[dict[str, Any]] = None


class AttendanceProcessor:
    """Runs one attendance session over a video file or live stream."""

    def __init__(self, cfg: Any, db: Database, engine: Any = None, liveness: Any = None) -> None:
        self.cfg = cfg
        self.db = db
        if engine is None:
            from .detector import FaceEngine
            engine = FaceEngine(cfg)
        self.engine = engine
        if liveness is None:
            from .liveness import build_liveness
            liveness = build_liveness(cfg)
        self.liveness = liveness

    def run(self, source: Source, class_name: str, subject: str, *, progress_cb: Optional[ProgressCB] = None,
            frame_cb: Optional[FrameCB] = None, should_stop: Optional[Callable[[], bool]] = None,
            source_label: Optional[str] = None, delete_source_after: bool = False,
            max_seconds: Optional[float] = None, behaviour: bool = False, camera_id: str = "default",
            analyzer: Any = None) -> ProcessResult:
        """Process one session.

        Args:
            behaviour: Also run Class Attentiveness & Behaviour Analytics.
            camera_id: Camera / room id used to look up the board zone.
            analyzer: Pre-built BehaviourAnalyzer (tests / custom models).
        """
        cfg = self.cfg
        rcfg, vcfg = cfg.recognition, cfg.video
        students = self.db.list_students(class_name)
        if not students:
            raise ValueError(f"No students enrolled for class '{class_name}'.")
        names = {s.roll_no: s.name for s in students}
        gallery = FaceGallery.from_db(self.db, class_name, float(rcfg.similarity_threshold),
                                      float(rcfg.get("margin", 0.0)), bool(cfg.models.get("use_faiss", False)))
        stream = is_stream(source)
        cap = open_capture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and 1.0 <= fps <= 240.0 else 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not stream else 0
        if not stream and bool(vcfg.get("prefetch", True)):
            cap = PrefetchCapture(cap)
        if behaviour and analyzer is None:
            from .behaviour.analyzer import BehaviourAnalyzer
            analyzer = BehaviourAnalyzer(cfg, self.db, camera_id)
        replace_video = analyzer is not None and bool(cfg.behaviour.video.replace_attendance_video)
        skip = max(1, int(vcfg.frame_skip))
        started = datetime.now()
        out_dir = cfg.path("outputs_dir") / (f"{started.strftime('%Y-%m-%d_%H-%M-%S')}_"
                                             f"{safe_name(class_name)}_{safe_name(subject)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        label = source_label or (f"camera:{source}" if stream else Path(str(source)).name)
        session_id = self.db.create_session(class_name, subject, label, started.isoformat(timespec="seconds"),
                                            str(out_dir), {"behaviour": analyzer is not None,
                                                           "camera_id": camera_id})
        log.info("Session %d started: class=%s subject=%s source=%s fps=%.1f frames=%d",
                 session_id, class_name, subject, label, fps, total_frames)

        occ = cfg.occlusion
        thr = float(rcfg.similarity_threshold) - (float(occ.threshold_relaxation) if occ.enabled else 0.0)
        extra_votes = int(occ.extra_votes) if occ.enabled else 0
        tracker = ByteTracker.from_config(cfg.tracker)
        att = AttendanceSession(min_votes=int(rcfg.min_votes), min_seconds=float(rcfg.min_seconds),
                                max_crops=int(rcfg.max_crops_per_track), frame_gap=3.0 * skip / fps,
                                unknown_min_frames=int(cfg.unknown.min_track_frames),
                                unknown_cluster_threshold=float(cfg.unknown.cluster_threshold))
        annotator: Optional[VideoAnnotator] = None
        hc_cfg = cfg.headcount
        hc_interval = float(hc_cfg.interval_minutes) * 60.0
        next_hc = hc_interval
        enh = cfg.enhance
        max_w = int(vcfg.get("max_width", 1920))
        min_q = float(rcfg.min_quality)
        embed_interval = max(1, int(rcfg.embed_interval))
        stream_limit = float(max_seconds or vcfg.get("max_stream_seconds", 3600))

        frame_idx = processed = 0
        fails = 0
        ts = 0.0
        t0 = time.monotonic()
        last_boxes: list[BoxLabel] = []
        status = "completed"
        try:
            while True:
                if should_stop and should_stop():
                    status = "stopped"
                    break
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    fails += 1
                    if stream:
                        if fails > int(vcfg.get("reconnect_attempts", 5)):
                            log.error("Stream lost after %d reconnect attempts.", fails - 1)
                            break
                        log.warning("Stream read failed - reconnecting (%d)...", fails)
                        cap.release()
                        time.sleep(2.0)
                        try:
                            cap = open_capture(source)
                        except ValueError:
                            pass
                        continue
                    # file: tolerate a few corrupted frames, then stop at EOF
                    if total_frames and frame_idx < total_frames - 1 and fails <= 30:
                        frame_idx += 1
                        continue
                    break
                fails = 0
                ts = (time.monotonic() - t0) if stream else frame_idx / fps
                if stream and ts > stream_limit:
                    status = "stopped"
                    break
                if frame.shape[1] > max_w:
                    f = max_w / frame.shape[1]
                    frame = cv2.resize(frame, (max_w, int(frame.shape[0] * f)), interpolation=cv2.INTER_AREA)
                if analyzer is not None and analyzer.out_dir is None:
                    analyzer.start(out_dir, fps, (frame.shape[1], frame.shape[0]), skip, students)
                if annotator is None and cfg.annotate.enabled and not replace_video:
                    annotator = VideoAnnotator(out_dir / "annotated.mp4", fps if not stream else fps,
                                               (frame.shape[1], frame.shape[0]),
                                               list(cfg.annotate.get("codec_preference", ["avc1", "mp4v"])))
                if frame_idx % skip == 0:
                    processed += 1
                    last_boxes = self._process_frame(frame, ts, processed, tracker, att, gallery, names, thr,
                                                     extra_votes, enh, min_q, embed_interval)
                    if hc_cfg.enabled and ts >= next_hc:
                        att.record_headcount(ts, hc_interval)
                        next_hc += hc_interval
                    annotated = draw_annotations(frame, last_boxes, att.present_count(), len(students), ts)
                    if annotator:
                        annotator.write(annotated)
                    if analyzer is not None:
                        work, outputs, dets = self._last
                        bframe = analyzer.on_frame(work, frame, ts, frame_idx, outputs, dets, att,
                                                   att.present_count(), len(students))
                        if bframe is not None:
                            annotated = bframe
                    if frame_cb:
                        frame_cb(annotated)
                    if progress_cb:
                        elapsed = time.monotonic() - t0
                        frac = min(1.0, (frame_idx + 1) / total_frames) if total_frames else 0.0
                        eta = elapsed * (1 - frac) / frac if frac > 0 else None
                        progress_cb({"fraction": frac, "eta_sec": eta, "elapsed_sec": elapsed,
                                     "present": att.present_count(), "total": len(students), "ts": ts,
                                     "frame": frame_idx, "total_frames": total_frames})
                else:
                    if annotator and cfg.annotate.write_all_frames:
                        annotator.write(draw_annotations(frame, last_boxes, att.present_count(), len(students), ts))
                    if analyzer is not None:
                        analyzer.on_skipped_frame(frame, ts, att.present_count(), len(students))
                frame_idx += 1
        except KeyboardInterrupt:
            status = "interrupted"
            log.warning("Processing interrupted - saving partial results.")
        except BaseException:
            status = "interrupted"
            raise
        finally:
            cap.release()
            video_path = annotator.close() if annotator else None
            if analyzer is not None and analyzer.out_dir is not None:
                full, _ = analyzer.close_video()
                if replace_video and full is not None:
                    video_path = full
            result = self._finalize(session_id, att, students, out_dir, ts, video_path, status, extra_votes,
                                    analyzer)
            if delete_source_after and not stream:
                try:
                    Path(str(source)).unlink(missing_ok=True)
                    log.info("Raw video deleted (privacy.retain_raw_video=false).")
                except OSError as exc:
                    log.warning("Could not delete raw video: %s", exc)
        return result

    # ------------------------------------------------------------------ per frame
    def _process_frame(self, frame: np.ndarray, ts: float, processed: int, tracker: ByteTracker,
                       att: AttendanceSession, gallery: FaceGallery, names: dict[str, str], thr: float,
                       extra_votes: int, enh: Any, min_q: float, embed_interval: int) -> list[BoxLabel]:
        work = frame
        if enh.clahe:
            work = enhance_low_light(frame, float(enh.low_light_threshold), float(enh.clip_limit),
                                     bool(enh.clahe_always))
        dets = self.engine.detect(work)
        boxes = np.array([d.bbox for d in dets], dtype=np.float32).reshape(-1, 4)
        scores = np.array([d.score for d in dets], dtype=np.float32)
        outputs, removed = tracker.update(boxes, scores)
        self._last = (work, outputs, dets)
        att.begin_frame(ts)
        to_embed: list[tuple[int, np.ndarray, Crop]] = []
        for o in outputs:
            t = att.observe(o.track_id, ts)
            det = dets[o.det_index]
            if det.kps is None or not np.isfinite(det.kps).all():
                continue
            crop_img = safe_crop(work, det.bbox)
            if crop_img.size == 0:
                continue
            q = quality_score(crop_img, det.bbox, det.kps, det.score)
            if q < min_q:
                continue
            if self.liveness is not None and not self.liveness.is_live(crop_img):
                log.debug("Track %d crop rejected by liveness check", o.track_id)
                continue
            crop = Crop(q, ts, thumb_of(crop_img))
            kept = t.add_crop(crop, att.max_crops)
            due = processed - t.last_embed_frame >= embed_interval
            refresh = processed - t.last_embed_frame >= embed_interval * 5
            if due and (kept or t.identity is None or refresh):
                to_embed.append((o.track_id, self.engine.align(work, det.kps), crop))
                t.last_embed_frame = processed
        if to_embed:
            embs = self.engine.embed_aligned([a for _, a, _ in to_embed])
            for (tid, _, crop), emb, m in zip(to_embed, embs, gallery.match_many(embs, thr)):
                crop.embedding = emb
                att.add_vote(tid, m.roll_no, m.score, ts, emb)
        att.end_frame(extra_votes)
        for tid in removed:
            att.finish_track(tid, extra_votes)
        labels: list[BoxLabel] = []
        for o in outputs:
            t = att.tracks[o.track_id]
            lab = f"{names.get(t.identity, t.identity)} ({t.identity})" if t.identity else None
            labels.append(BoxLabel(o.bbox, o.track_id, lab, t.flagged and not t.identity))
        return labels

    # ------------------------------------------------------------------ finalize
    def _finalize(self, session_id: int, att: AttendanceSession, students: list[Any], out_dir: Path,
                  duration: float, video_path: Optional[Path], status: str, extra_votes: int,
                  analyzer: Any = None) -> ProcessResult:
        cfg = self.cfg
        att.finalize(extra_votes)
        late_after = None
        if cfg.late.enabled:
            late_after = float(cfg.late.class_start_offset_sec) + float(cfg.late.grace_minutes) * 60.0
        left_before = None
        le = float(cfg.headcount.left_early_minutes) * 60.0
        if cfg.headcount.enabled and duration > 2 * le:
            left_before = duration - le
        if cfg.headcount.enabled:
            att.record_headcount(duration, float(cfg.headcount.interval_minutes) * 60.0)
        rows = build_rows(att, students, late_after, left_before)
        snap_dir = out_dir / "snapshots"
        for r in rows:
            snap = r.pop("snapshot", None)
            r["snapshot_path"] = None
            if snap is not None:
                p = snap_dir / f"{safe_name(r['roll_no'])}.jpg"
                if write_image(p, snap):
                    r["snapshot_path"] = str(p)
        self.db.save_attendance(session_id, rows)
        unknown_count = 0
        if cfg.unknown.enabled:
            unk_dir = out_dir / "Unknown"
            for cid, t in att.cluster_unknowns():
                best = t.best_crop()
                if best is None or best.thumb is None:
                    continue
                p = unk_dir / f"cluster_{cid:03d}_track_{t.track_id}.jpg"
                write_image(p, best.thumb)
                self.db.save_unknown(session_id, cid, t.track_id, round(t.first_seen, 2), round(t.last_seen, 2),
                                     str(p), t.mean_embedding())
                unknown_count += 1
        conflicts = [c.__dict__ for c in att.conflicts]
        meta = {"conflicts": conflicts, "headcounts": att.headcounts, "fps_tracks": len(att.tracks),
                "behaviour": analyzer is not None, "camera_id": getattr(analyzer, "camera_id", None)}
        self.db.update_session(session_id, duration_sec=round(duration, 2), status=status,
                               video_path=str(video_path) if video_path else None, meta_json=json.dumps(meta))
        report = None
        try:
            report = generate_report(self.db, session_id, cfg)
        except Exception as exc:  # report errors must not lose the attendance data
            log.exception("Report generation failed: %s", exc)
        behaviour = None
        if analyzer is not None and analyzer.out_dir is not None:
            try:
                behaviour = analyzer.finalize(session_id, att, duration)
            except Exception as exc:  # behaviour errors must not lose attendance
                log.exception("Behaviour analysis failed: %s", exc)
        present = sum(1 for r in rows if r["status"] in ("Present", "Late"))
        log.info("Session %d %s: %d/%d present, %d unknown tracks, %d conflicts", session_id, status, present,
                 len(rows), unknown_count, len(conflicts))
        return ProcessResult(session_id, report, video_path, out_dir, rows, unknown_count, len(conflicts),
                             duration, status, behaviour)


def assign_unknown(db: Database, cfg: Any, session_id: int, cluster_id: int, roll_no: str, username: str,
                   update_gallery: bool = False) -> Path:
    """Assign an unknown-face cluster to a student, mark them present and rebuild the report."""
    student = db.get_student(roll_no)
    if student is None:
        raise ValueError(f"Student {roll_no} not enrolled")
    unknowns = [u for u in db.get_unknowns(session_id, include_embeddings=True) if u["cluster_id"] == cluster_id]
    if not unknowns:
        raise ValueError(f"Cluster {cluster_id} not found in session {session_id}")
    db.assign_unknown_cluster(session_id, cluster_id, roll_no)
    att = {a["roll_no"]: a for a in db.get_attendance(session_id)}
    cur = att.get(roll_no, {})
    first = min(u["first_seen"] for u in unknowns)
    last = max(u["last_seen"] for u in unknowns)
    if cur.get("status") in ("Present", "Late"):
        first, last = min(first, cur.get("first_seen") or first), max(last, cur.get("last_seen") or last)
    extra = {"first_seen": first, "last_seen": last,
             "time_visible": max(float(cur.get("time_visible") or 0), last - first),
             "remarks": f"Assigned from unknown cluster {cluster_id}"}
    if not cur.get("snapshot_path"):
        extra["snapshot_path"] = unknowns[0]["thumb_path"]
    new_status = cur.get("status") if cur.get("status") in ("Present", "Late") else "Present"
    if cfg.late.enabled and new_status == "Present":
        if first > float(cfg.late.class_start_offset_sec) + float(cfg.late.grace_minutes) * 60:
            new_status = "Late"
    db.set_status(session_id, roll_no, new_status, username, f"assigned unknown cluster {cluster_id}", **extra)
    if update_gallery:
        embs = [u["embedding"] for u in unknowns if u.get("embedding") is not None]
        if embs:
            n = max(1, student.num_images)
            new = student.embedding * n + np.sum(np.stack(embs), axis=0)
            student.embedding = new / (np.linalg.norm(new) + 1e-12)
            student.num_images = n + len(embs)
            db.upsert_student(student)
            db.log_audit(username, "gallery_update", session_id, roll_no, reason=f"cluster {cluster_id}")
    return generate_report(db, session_id, cfg)
