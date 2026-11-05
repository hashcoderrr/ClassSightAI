"""Behaviour analysis orchestration.

Live pass (inside the attendance pipeline): for every Nth processed frame run body
pose + object detection on the whole frame, landmarks / head pose per face track,
store :class:`FrameSignals`, and drive the annotated overlay with causal smoothing.

Final pass (:func:`compute_results`): link every track to its student via the
attendance identities, classify offline, smooth with a centred window, apply
teacher corrections, then build events, scores, timelines and movement metrics.
Signals are cached (encrypted Parquet) so :func:`rescore_session` can recompute
everything with new weights/thresholds without re-processing the video.
"""
from __future__ import annotations

import io
import json
import logging
from collections import Counter, deque
from pathlib import Path
from typing import Any, Deque, Optional

import numpy as np

from . import DISCLAIMER
from .behaviour_classifier import BehaviourClassifier, build_classifier
from .head_pose import BoardZone, pose_from_body, pose_from_kps5, pose_from_landmarks68
from .landmarks import FaceLandmarker, eye_aspect_ratio, mouth_aspect_ratio
from .object_detector import NullObjectDetector, ObjectDetector, associate_objects, build_object_detector
from .pose_estimator import (NullPoseEstimator, PoseEstimator, associate_persons, build_pose_estimator,
                             head_box_from_pose)
from .scoring import ScoringConfig, binned_scores, class_timeline, drop_points, mean_confidence, weighted_score
from .signals import (BEHAVIOURS, DISPLAY, HAND_RAISED, OUT_OF_FRAME, PHONE, SLEEPING, STANDING, TALKING,
                      FrameSignals, PersonPose)
from .timeline import (Sample, apply_corrections, behaviour_seconds, build_segments, count_head_turns, displacement,
                       merge_segments, per_second_rows, smooth_states)
from .video_overlay import BehaviourOverlay, build_overlay
from .zones import get_zone

log = logging.getLogger(__name__)
CACHE_DIR = "behaviour_cache"


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class BehaviourAnalyzer:
    """Collects behaviour signals during processing and renders the behaviour video."""

    def __init__(self, cfg: Any, db: Any, camera_id: str = "default", pose: Optional[PoseEstimator] = None,
                 objects: Optional[ObjectDetector] = None, landmarker: Optional[FaceLandmarker] = None,
                 load_models: bool = True) -> None:
        self.cfg, self.db, self.camera_id = cfg, db, str(camera_id or "default")
        self.bcfg = cfg.behaviour
        self.zone = BoardZone.from_config(self.bcfg, get_zone(cfg, self.camera_id))
        self.pose = pose if pose is not None else (build_pose_estimator(cfg) if load_models else NullPoseEstimator())
        self.objects = objects if objects is not None else (
            build_object_detector(cfg) if load_models else NullObjectDetector())
        self.landmarker = landmarker if landmarker is not None else (
            FaceLandmarker.from_config(cfg) if load_models else None)
        self.classifier: BehaviourClassifier = build_classifier(cfg)
        self.scoring = ScoringConfig.from_config(self.bcfg.scoring)
        self.every = max(1, int(self.bcfg.every_n_processed))
        self.min_lmk_face = float(self.bcfg.get("min_landmark_face", 24))
        self.signals: list[FrameSignals] = []
        self.overlay: Optional[BehaviourOverlay] = None
        self._n = 0
        self._recent: dict[int, dict[str, Any]] = {}     # track -> last person bbox / face ts
        self._live: dict[int, Deque[tuple[float, str]]] = {}
        self._ema: dict[int, float] = {}
        self.names: dict[str, str] = {}
        self.consent: dict[str, bool] = {}
        self.class_only = bool(self.bcfg.privacy.class_level_only)
        self.fps = 25.0
        self.step = 0.2
        self.frame_size = (0, 0)
        self.out_dir: Optional[Path] = None
        self.video_paths: tuple[Optional[Path], Optional[Path]] = (None, None)
        log.info("Behaviour analyzer ready (camera=%s, zone=%s, pose=%s, objects=%s, landmarks=%s)",
                 self.camera_id, "polygon" if self.zone.polygon is not None else "facing-camera",
                 type(self.pose).__name__, type(self.objects).__name__, self.landmarker is not None)

    # ------------------------------------------------------------------ live
    def start(self, out_dir: Path, fps: float, frame_size: tuple[int, int], frame_skip: int,
              students: list[Any]) -> None:
        self.out_dir, self.fps, self.frame_size = out_dir, fps, frame_size
        self.step = frame_skip * self.every / fps
        self.names = {s.roll_no: s.name for s in students}
        self.consent = {s.roll_no: bool(getattr(s, "behaviour_consent", False)) for s in students}
        if self.bcfg.video.enabled:
            self.overlay = build_overlay(self.cfg, out_dir, fps, frame_size)

    def _label(self, roll: Optional[str], tid: int) -> tuple[str, bool]:
        if roll is None:
            return f"Unknown_{tid}", False
        if self.class_only or not self.consent.get(roll, False):
            return "Student", True
        return f"{roll} | {self.names.get(roll, roll)}", True

    def _signal_for_face(self, frame: np.ndarray, tid: int, ts: float, frame_idx: int, det: Any,
                         person: Optional[PersonPose], objs: list[str]) -> FrameSignals:
        h, w = frame.shape[:2]
        bbox = np.asarray(det.bbox, dtype=np.float32)
        yaw = pitch = roll = ear = mar = None
        source = "none"
        lmk = None
        if self.landmarker is not None and min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= self.min_lmk_face:
            lmk = self.landmarker.landmarks(frame, bbox)
        if lmk is not None:
            p = pose_from_landmarks68(lmk, w, h)
            if p is not None:
                yaw, pitch, roll = p
                source = "landmarks68"
            ear, mar = eye_aspect_ratio(lmk), mouth_aspect_ratio(lmk)
        if source == "none" and det.kps is not None:
            p = pose_from_kps5(det.kps)
            if p is not None:
                yaw, pitch, roll = p
                source = "kps5"
        s = FrameSignals(tid, ts, frame_idx, bbox, True, float(det.score), yaw, pitch, roll, source, ear, mar,
                         None if person is None else person.keypoints,
                         None if person is None else person.bbox, objs, None, w, h)
        s.on_board = self.zone.looking_at(s.center, yaw, pitch, w, h)
        return s

    def on_frame(self, work: np.ndarray, frame: np.ndarray, ts: float, frame_idx: int, outputs: list[Any],
                 dets: list[Any], att: Any, present: int, total: int) -> Optional[np.ndarray]:
        """Called for every processed frame. Returns the annotated frame (or None if video disabled)."""
        heavy = self._n % self.every == 0
        self._n += 1
        if heavy:
            persons = self.pose.estimate(work)
            objects = self.objects.detect(work)
            face_boxes = [np.asarray(dets[o.det_index].bbox) for o in outputs]
            pmap = associate_persons(face_boxes, persons, float(self.bcfg.rules.kp_conf))
            per_face = [persons[pmap[i]] if i in pmap else None for i in range(len(outputs))]
            omap = associate_objects(objects, face_boxes, per_face, float(self.bcfg.objects.hand_dist))
            current: set[int] = set()
            new: list[FrameSignals] = []
            for i, o in enumerate(outputs):
                s = self._signal_for_face(work, o.track_id, ts, frame_idx, dets[o.det_index], per_face[i],
                                          omap.get(i, []))
                new.append(s)
                current.add(o.track_id)
                self._recent[o.track_id] = {"person": None if per_face[i] is None else per_face[i].bbox,
                                            "face_ts": ts}
            # body-only fallback for students whose face is temporarily not visible
            used = set(pmap.values())
            fb_sec = float(self.bcfg.body_fallback_sec)
            for pi, p in enumerate(persons):
                if pi in used:
                    continue
                best, best_iou = None, 0.3
                for tid, info in self._recent.items():
                    if tid in current or info["person"] is None or ts - info["face_ts"] > fb_sec:
                        continue
                    v = _iou(info["person"], p.bbox)
                    if v > best_iou:
                        best, best_iou = tid, v
                if best is None:
                    continue
                hb = head_box_from_pose(p, float(self.bcfg.rules.kp_conf))
                if hb is None:
                    continue
                yp = pose_from_body(p.keypoints, float(self.bcfg.rules.kp_conf))
                h, w = work.shape[:2]
                s = FrameSignals(best, ts, frame_idx, hb, False, p.score, None if yp is None else yp[0],
                                 None if yp is None else yp[1], None, "body", None, None, p.keypoints, p.bbox,
                                 [], None, w, h)
                s.on_board = self.zone.looking_at(s.center, s.yaw, s.pitch, w, h)
                new.append(s)
                current.add(best)
                self._recent[best]["person"] = p.bbox
            self.signals.extend(new)
            self._update_live(new, att, ts)
        if self.overlay is None:
            return None
        self.overlay.push_class_point(ts)
        return self.overlay.draw(frame, ts, present, total)

    def _update_live(self, new: list[FrameSignals], att: Any, ts: float) -> None:
        if self.overlay is None:
            return
        window = float(self.bcfg.smoothing_window_sec)
        for s in new:
            c = self.classifier.classify(s)
            dq = self._live.setdefault(s.track_id, deque())
            dq.append((s.ts, c.state))
            while dq and s.ts - dq[0][0] > window:
                dq.popleft()
            state = Counter(st for _, st in dq).most_common(1)[0][0]
            fs = self.scoring.frame_score(state)
            if fs is not None:
                prev = self._ema.get(s.track_id)
                self._ema[s.track_id] = fs if prev is None else 0.7 * prev + 0.3 * fs
            roll = att.label_for(s.track_id) if att is not None else None
            label, is_student = self._label(roll, s.track_id)
            self.overlay.update(s.track_id, label, state, self._ema.get(s.track_id), s.bbox, ts, s.keypoints,
                                s.yaw, s.pitch, is_student)

    def on_skipped_frame(self, frame: np.ndarray, ts: float, present: int, total: int) -> None:
        if self.overlay is not None and self.cfg.annotate.get("write_all_frames", True):
            self.overlay.hold(frame, ts, present, total)

    def close_video(self) -> tuple[Optional[Path], Optional[Path]]:
        if self.overlay is not None and self.video_paths == (None, None):
            self.video_paths = self.overlay.close()
        return self.video_paths

    # ------------------------------------------------------------------ final
    def finalize(self, session_id: int, att: Any, duration: float) -> dict[str, Any]:
        """Save the signal cache, compute results, write reports. Returns the stored result."""
        assert self.out_dir is not None, "start() was not called"
        identities = {tid: t.identity for tid, t in att.tracks.items()}
        meta = {"session_id": session_id, "fps": self.fps, "step": self.step, "duration": duration,
                "frame_w": self.frame_size[0], "frame_h": self.frame_size[1], "camera_id": self.camera_id,
                "identities": {str(k): v for k, v in identities.items()}}
        save_cache(self.db, self.out_dir, self.signals, meta)
        full, prev = self.close_video()
        files = {"video_full": str(full) if full else None, "video_preview": str(prev) if prev else None}
        return run_scoring(self.db, self.cfg, session_id, self.signals, meta, files)


# ---------------------------------------------------------------------- cache
def save_cache(db: Any, out_dir: Path, signals: list[FrameSignals], meta: dict[str, Any]) -> Path:
    import pandas as pd
    cdir = out_dir / CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([s.to_row() for s in signals])
    buf = io.BytesIO()
    try:
        df.to_parquet(buf, index=False)
        fmt = "parquet"
    except (ImportError, ValueError):
        buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        fmt = "csv"
    meta = {**meta, "format": fmt}
    (cdir / f"signals.{fmt}.enc").write_bytes(db.crypto.encrypt(buf.getvalue()))
    (cdir / "meta.json.enc").write_bytes(db.crypto.encrypt(json.dumps(meta).encode("utf-8")))
    return cdir


def load_cache(db: Any, out_dir: Path) -> tuple[list[FrameSignals], dict[str, Any]]:
    import pandas as pd
    cdir = Path(out_dir) / CACHE_DIR
    meta_p = cdir / "meta.json.enc"
    if not meta_p.exists():
        raise FileNotFoundError(f"No behaviour cache in {cdir} (deleted by retention?)")
    meta = json.loads(db.crypto.decrypt(meta_p.read_bytes()))
    raw = db.crypto.decrypt((cdir / f"signals.{meta['format']}.enc").read_bytes())
    if not raw.strip():
        return [], meta
    df = pd.read_parquet(io.BytesIO(raw)) if meta["format"] == "parquet" else pd.read_csv(io.BytesIO(raw))
    df = df.astype(object).where(df.notna(), None)
    return [FrameSignals.from_row(r) for r in df.to_dict("records")], meta


# ---------------------------------------------------------------------- scoring
def compute_results(signals: list[FrameSignals], meta: dict[str, Any], cfg: Any, students: dict[str, Any],
                    attendance: dict[str, str], corrections: list[dict[str, Any]],
                    class_only: Optional[bool] = None) -> dict[str, Any]:
    """Pure computation of the behaviour result (no I/O)."""
    bcfg = cfg.behaviour
    sc = ScoringConfig.from_config(bcfg.scoring)
    classifier = build_classifier(cfg)
    class_only = bool(bcfg.privacy.class_level_only) if class_only is None else class_only
    identities = {int(k): v for k, v in meta.get("identities", {}).items()}
    duration = float(meta.get("duration", 0.0))
    step = float(meta.get("step", 0.2))
    fw, fh = int(meta.get("frame_w") or 1), int(meta.get("frame_h") or 1)
    window = float(bcfg.smoothing_window_sec)
    out_gap = float(bcfg.out_of_frame_sec)
    bin_sec = float(bcfg.timeline_bin_sec)
    min_unknown = int(cfg.unknown.min_track_frames)
    # board decision is recomputed from the *current* zone so redrawn zones apply on rescore
    zone = BoardZone.from_config(bcfg, get_zone(cfg, str(meta.get("camera_id") or "default")))
    for s in signals:
        s.on_board = zone.looking_at(s.center, s.yaw, s.pitch, s.frame_w or fw, s.frame_h or fh)

    by_track: dict[int, list[FrameSignals]] = {}
    for s in signals:
        by_track.setdefault(s.track_id, []).append(s)
    groups: dict[str, list[tuple[FrameSignals, Any]]] = {}
    for tid, sigs in by_track.items():
        sigs.sort(key=lambda x: (x.ts, x.face_visible is False))
        roll = identities.get(tid)
        if roll is None and len(sigs) < min_unknown:
            continue
        key = roll if roll is not None else f"Unknown_{tid}"
        for s, c in zip(sigs, classifier.classify_sequence(sigs)):
            groups.setdefault(key, []).append((s, c))

    corr_by_key: dict[str, list[dict[str, Any]]] = {}
    for c in corrections:
        corr_by_key.setdefault(c["subject_key"], []).append(c)

    subjects: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    per_second: list[dict[str, Any]] = []
    state_scores = sc.state_scores()
    for key, items in groups.items():
        items.sort(key=lambda sc_: sc_[0].ts)
        dedup: list[tuple[FrameSignals, Any]] = []
        for s, c in items:  # one sample per timestamp (face beats body-only)
            if dedup and abs(dedup[-1][0].ts - s.ts) < 1e-6:
                continue
            dedup.append((s, c))
        samples = [Sample(s.ts, s.frame_idx, c.state, c.confidence, s.yaw, s.pitch, s.center[0], s.center[1],
                          s.face_h, c.state) for s, c in dedup]
        smoothed = smooth_states([x.ts for x in samples], [x.state for x in samples], window)
        for x, st in zip(samples, smoothed):
            x.state = st
        corrected = apply_corrections(samples, corr_by_key.get(key, []))
        segs = build_segments(samples, step, out_gap, duration)
        for sg in segs:
            sg.corrected = any(c["start"] - 1e-6 <= sg.start <= c["end"] + 1e-6 for c in corr_by_key.get(key, []))
        events = merge_segments(segs)
        is_student = not key.startswith("Unknown_")
        stu = students.get(key)
        name = stu.name if stu is not None else key
        secs = behaviour_seconds(segs)
        observed = sum(v for k, v in secs.items() if k != OUT_OF_FRAME)
        total_t = sum(secs.values())
        score = weighted_score(segs, sc)
        ev_states = [e.state for e in events]
        transitions = sum(1 for a, b in zip(ev_states, ev_states[1:])
                          if a != b and OUT_OF_FRAME not in (a, b))
        disp_px, disp_norm = displacement(samples)
        xs, ys = np.array([x.x for x in samples]), np.array([x.y for x in samples])
        trail = []
        last_t = -1.0
        for x in samples:
            if x.ts - last_t >= 1.0:
                trail.append([round(x.ts, 2), round(x.x / fw, 4), round(x.y / fh, 4)])
                last_t = x.ts
        subj = {
            "key": key, "roll_no": key if is_student else "", "name": name if is_student else key,
            "kind": "student" if is_student else "unknown",
            "attendance": attendance.get(key, "Present") if is_student else "Unknown",
            "consent": bool(getattr(stu, "behaviour_consent", False)) if is_student else True,
            "score": score, "label": sc.label(score), "confidence": mean_confidence(segs),
            "observed_sec": round(observed, 2),
            "behaviour_seconds": {b: round(secs.get(b, 0.0), 2) for b in BEHAVIOURS},
            "behaviour_pct": {b: round(100 * secs.get(b, 0.0) / total_t, 1) if total_t else 0.0 for b in BEHAVIOURS},
            "counts": {"hand_raises": ev_states.count(HAND_RAISED), "phone_uses": ev_states.count(PHONE),
                       "talking_episodes": ev_states.count(TALKING), "sleep_episodes": ev_states.count(SLEEPING),
                       "stand_ups": ev_states.count(STANDING)},
            "movement": {"displacement_px": round(disp_px, 1), "displacement_face_heights": round(disp_norm, 2),
                         "posture_changes": transitions,
                         "head_turns": count_head_turns(samples, float(bcfg.rules.away_yaw)),
                         "stand_ups": ev_states.count(STANDING)},
            "position": [round(float(np.median(xs)) / fw, 4), round(float(np.median(ys)) / fh, 4)]
            if len(xs) else None,
            "trail": trail,
            "bins": binned_scores(segs, sc, bin_sec, duration),
            "corrected_samples": corrected,
            "first_seen": round(samples[0].ts, 2) if samples else None,
        }
        subjects.append(subj)
        for e in events:
            all_events.append({"key": key, "roll_no": subj["roll_no"], "name": subj["name"], "behaviour": e.state,
                               "display": DISPLAY.get(e.state, e.state), "start": round(e.start, 2),
                               "end": round(e.end, 2), "duration": round(e.duration, 2),
                               "confidence": round(e.conf, 3), "frame_start": e.frame_start,
                               "frame_end": e.frame_end, "corrected": e.corrected})
        if bcfg.report.include_movement_sheet:
            for r in per_second_rows(samples, segs, state_scores, fw, fh):
                per_second.append({"key": key, "roll_no": subj["roll_no"], "name": subj["name"], **r})

    # enrolled students with no behaviour data (absent / never analysed)
    have = {s["key"] for s in subjects}
    for roll, stu in students.items():
        if roll not in have:
            subjects.append({"key": roll, "roll_no": roll, "name": stu.name, "kind": "student",
                             "attendance": attendance.get(roll, "Absent"),
                             "consent": bool(getattr(stu, "behaviour_consent", False)), "score": None,
                             "label": "No data", "confidence": 0.0, "observed_sec": 0.0,
                             "behaviour_seconds": {b: 0.0 for b in BEHAVIOURS},
                             "behaviour_pct": {b: 0.0 for b in BEHAVIOURS},
                             "counts": {"hand_raises": 0, "phone_uses": 0, "talking_episodes": 0,
                                        "sleep_episodes": 0, "stand_ups": 0},
                             "movement": {"displacement_px": 0.0, "displacement_face_heights": 0.0,
                                          "posture_changes": 0, "head_turns": 0, "stand_ups": 0},
                             "position": None, "trail": [], "bins": [], "corrected_samples": 0,
                             "first_seen": None})

    include_unknown = bool(bcfg.include_unknown_in_class)
    in_class = [s for s in subjects if s["score"] is not None and (s["kind"] == "student" or include_unknown)]
    timeline = class_timeline([s["bins"] for s in in_class], bin_sec)
    drops = drop_points(timeline, float(bcfg.drop_threshold), int(bcfg.drop_lookback))
    totals = {b: round(sum(s["behaviour_seconds"][b] for s in in_class), 2) for b in BEHAVIOURS}
    affected = {b: sum(1 for s in in_class if s["behaviour_seconds"][b] > 0) for b in BEHAVIOURS}
    class_scores = [s["score"] for s in in_class]
    weights = sc.weights
    distractions = sorted(
        [{"behaviour": b, "display": DISPLAY[b], "minutes": round(totals[b] / 60, 2), "students": affected[b]}
         for b in BEHAVIOURS if weights.get(b) is not None and weights[b] < 0 and totals[b] > 0],
        key=lambda d: -d["minutes"])
    label_counts = Counter(sc.label(s["score"]) for s in in_class)
    class_score = round(float(np.mean(class_scores)), 2) if class_scores else None
    attentive_thr = next((t for t, n in sc.labels if n == "Attentive"), 50.0)

    result = {
        "session_id": meta.get("session_id"), "camera_id": meta.get("camera_id"), "duration": duration,
        "disclaimer": DISCLAIMER, "class_only": class_only, "bin_sec": bin_sec,
        "class_score": class_score, "class_label": sc.label(class_score),
        "attentive_count": sum(1 for v in class_scores if v >= attentive_thr),
        "inattentive_count": sum(1 for v in class_scores if v < attentive_thr),
        "label_counts": {n: label_counts.get(n, 0) for _, n in sc.labels},
        "students_analysed": len(in_class), "class_timeline": timeline, "drop_points": drops,
        "behaviour_totals": totals, "distractions": distractions,
        "weights": {k: v for k, v in weights.items()}, "state_scores": state_scores,
        "subjects": subjects, "events": sorted(all_events, key=lambda e: (e["start"], e["key"])),
        "per_second": per_second, "corrections": len(corrections),
        "frame_w": fw, "frame_h": fh,
    }
    return privacy_filter(result, class_only)


def privacy_filter(result: dict[str, Any], class_only: bool) -> dict[str, Any]:
    """Drop per-student data for students without behaviour consent, or entirely in class-only mode."""
    r = dict(result)
    if class_only:
        r["subjects"], r["events"], r["per_second"] = [], [], []
        r["class_only"] = True
        return r
    allowed = {s["key"] for s in r["subjects"] if s["kind"] == "unknown" or s["consent"]}
    r["withheld_no_consent"] = sum(1 for s in r["subjects"] if s["key"] not in allowed)
    r["subjects"] = [s for s in r["subjects"] if s["key"] in allowed]
    r["events"] = [e for e in r["events"] if e["key"] in allowed]
    r["per_second"] = [p for p in r["per_second"] if p["key"] in allowed]
    return r


def run_scoring(db: Any, cfg: Any, session_id: int, signals: list[FrameSignals], meta: dict[str, Any],
                files: dict[str, Any]) -> dict[str, Any]:
    """Compute, store (encrypted) and export the behaviour result of a session."""
    from .report_behaviour_excel import export_behaviour
    sess = db.get_session(session_id)
    students = {s.roll_no: s for s in db.list_students(sess["class_name"])}
    attendance = {a["roll_no"]: a["status"] for a in db.get_attendance(session_id)}
    corrections = db.get_behaviour_corrections(session_id)
    result = compute_results(signals, meta, cfg, students, attendance, corrections)
    result.update({"class_name": sess["class_name"], "subject": sess["subject"], "started_at": sess["started_at"]})
    out_dir = Path(sess.get("output_dir") or cfg.path("outputs_dir"))
    result["files"] = {**files, **export_behaviour(result, out_dir, cfg)}
    db.save_behaviour_result(session_id, result)
    log.info("Behaviour analysis for session %d: class score %s (%s), %d events, %d subjects",
             session_id, result["class_score"], result["class_label"], len(result["events"]),
             len(result["subjects"]))
    return result


def rescore_session(db: Any, cfg: Any, session_id: int, username: str = "system") -> dict[str, Any]:
    """Recompute behaviour scores from the cached signals with the current config and corrections."""
    sess = db.get_session(session_id)
    if sess is None:
        raise ValueError(f"Session {session_id} not found")
    signals, meta = load_cache(db, Path(sess["output_dir"]))
    prev = db.get_behaviour_result(session_id) or {}
    files = {k: v for k, v in (prev.get("files") or {}).items() if k in ("video_full", "video_preview")}
    result = run_scoring(db, cfg, session_id, signals, meta, files)
    db.log_audit(username, "behaviour_rescore", session_id, reason=f"{len(signals)} cached signals")
    return result
