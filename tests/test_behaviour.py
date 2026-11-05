"""Tests for the attentiveness & behaviour module."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from openpyxl import load_workbook

from src.behaviour.analyzer import BehaviourAnalyzer, compute_results, load_cache, privacy_filter, rescore_session
from src.behaviour.behaviour_classifier import RuleBasedClassifier, Rules
from src.behaviour.evaluation import evaluate_behaviours, load_intervals
from src.behaviour.head_pose import (LMK68_IDX, MODEL_POINTS, BoardZone, camera_matrix, pose_from_kps5,
                                     pose_from_landmarks68)
from src.behaviour.landmarks import eye_aspect_ratio, mouth_aspect_ratio
from src.behaviour.object_detector import associate_objects
from src.behaviour.pose_estimator import PoseEstimator, associate_persons
from src.behaviour.report_behaviour_excel import write_behaviour_workbook
from src.behaviour.scoring import ScoringConfig, binned_scores, class_timeline, drop_points, weighted_score
from src.behaviour.signals import DetectedObject, FrameSignals, PersonPose
from src.behaviour.timeline import Sample, Segment, apply_corrections, build_segments, merge_segments, smooth_states
from src.behaviour.zones import get_zone, save_zone
from src.pipeline import AttendanceProcessor
from tests.conftest import FakeEngine


# ------------------------------------------------------------------ smoothing / timeline
def test_smoothing_removes_flicker():
    ts = [i * 0.2 for i in range(20)]
    states = ["board"] * 20
    states[7] = "phone"            # single-frame flicker
    out = smooth_states(ts, states, 2.0)
    assert out == ["board"] * 20


def test_smoothing_keeps_sustained_change_and_ties():
    ts = [i * 0.5 for i in range(12)]
    states = ["a"] * 6 + ["b"] * 6
    out = smooth_states(ts, states, 2.0)
    assert out[:5] == ["a"] * 5 and out[-5:] == ["b"] * 5
    assert smooth_states([], [], 2.0) == []
    assert smooth_states([0, 1], ["x", "y"], 0) == ["x", "y"]


def test_segments_out_of_frame_and_merge():
    samples = [Sample(t, int(t * 10), "looking_board", 1.0) for t in (0, 1, 2)] + \
              [Sample(t, int(t * 10), "looking_board", 1.0) for t in (10, 11)]
    segs = build_segments(samples, step=1.0, out_gap=4.0, video_end=20.0)
    ev = merge_segments(segs)
    assert [(e.state, e.start, e.end) for e in ev] == [("looking_board", 0, 3), ("out_of_frame", 3, 10),
                                                        ("looking_board", 10, 12), ("out_of_frame", 12, 20)]
    # short gaps are interpolated (state held), not out of frame
    segs2 = build_segments(samples[:3], 1.0, 4.0, 3.5)
    assert {s.state for s in segs2} == {"looking_board"}


def test_corrections_replace_states():
    samples = [Sample(t, t, "phone", 0.6) for t in range(5)]
    n = apply_corrections(samples, [{"start": 1, "end": 3, "behaviour": "phone", "corrected_to": "writing"}])
    assert n == 3 and [s.state for s in samples] == ["phone", "writing", "writing", "writing", "phone"]
    assert samples[2].conf == 1.0


# ------------------------------------------------------------------ scoring
def test_frame_scores_and_labels():
    sc = ScoringConfig()
    assert sc.frame_score("looking_board") == 100 and sc.frame_score("other") == 50
    assert sc.frame_score("phone") == 0 and sc.frame_score("talking") == 30
    assert sc.frame_score("out_of_frame") is None
    assert sc.label(80) == "Highly Attentive" and sc.label(75) == "Highly Attentive"
    assert sc.label(60) == "Attentive" and sc.label(30) == "Partially Attentive" and sc.label(10) == "Inattentive"
    assert sc.label(None) == "No data"


def test_weighted_score_time_and_confidence():
    sc = ScoringConfig()
    segs = [Segment("looking_board", 0, 30, 1.0, 0, 0), Segment("phone", 30, 40, 1.0, 0, 0),
            Segment("out_of_frame", 40, 100, 1.0, 0, 0)]
    assert weighted_score(segs, sc) == pytest.approx(75.0)          # out of frame excluded
    segs[1].conf = 0.5                                               # less confident phone detection
    assert weighted_score(segs, sc) == pytest.approx(100 * 30 / 35, abs=0.01)
    sc.use_confidence = False
    assert weighted_score(segs, sc) == pytest.approx(75.0)
    bins = binned_scores([Segment("looking_board", 0, 60, 1, 0, 0), Segment("sleeping", 60, 120, 1, 0, 0)], sc,
                         60, 120)
    assert bins == [100.0, 0.0]


def test_class_timeline_and_drop_points():
    tl = class_timeline([[100, 90, 40, None], [80, 90, 20, 60]], 60)
    assert [t["score"] for t in tl] == [90, 90, 30, 60] and tl[3]["students"] == 1
    drops = drop_points(tl, threshold=15, lookback=2)
    assert [d["bin"] for d in drops] == [2] and drops[0]["drop"] == 60


# ------------------------------------------------------------------ rules
def sig(tid=1, ts=0.0, yaw=0.0, pitch=0.0, ear=0.3, mar=0.05, kps=None, objects=(), on_board=None, cy=100.0):
    return FrameSignals(tid, ts, int(ts * 25), np.array([80, cy - 20, 120, cy + 20], np.float32), True, 0.9, yaw,
                        pitch, 0.0, "landmarks68", ear, mar, kps, None, list(objects), on_board, 640, 480)


def body(nose_y=100, wrist_y=200, sho_y=150):
    k = np.zeros((17, 3), np.float32)
    k[0] = [100, nose_y, 0.9]
    k[5], k[6] = [80, sho_y, 0.9], [120, sho_y, 0.9]
    k[9], k[10] = [70, wrist_y, 0.9], [130, wrist_y, 0.9]
    return k


def run(clf, sigs):
    return [clf.classify(s).state for s in sigs]


def test_rules_board_phone_hand():
    clf = RuleBasedClassifier(Rules())
    assert clf.classify(sig(on_board=True)).state == "looking_board"
    assert clf.classify(sig(ts=0.2, on_board=True, objects=["phone"])).state == "phone"
    assert clf.classify(sig(ts=0.4, kps=body(wrist_y=60))).state == "hand_raised"
    assert clf.classify(sig(ts=0.6, kps=body(wrist_y=200), on_board=True)).state == "looking_board"


def test_rules_sustained_looking_away_and_sleep():
    clf = RuleBasedClassifier(Rules(away_min_sec=2.0, sleep_min_sec=3.0))
    states = run(clf, [sig(ts=t * 0.5, yaw=50) for t in range(6)])
    assert states[:4] == ["other"] * 4 and states[4] == "looking_away"
    clf.reset()
    states = run(clf, [sig(ts=t * 0.5, pitch=45, ear=0.1) for t in range(8)])
    assert "sleeping" not in states[:6] and states[6] == "sleeping"
    clf.reset()  # head down but eyes open -> looking down (writing, no pose to verify hands)
    assert clf.classify(sig(pitch=40, ear=0.3)).state == "writing"


def test_rules_talking_and_writing_and_standing():
    clf = RuleBasedClassifier(Rules())
    mars = [0.05, 0.35, 0.05, 0.4, 0.05, 0.38]
    states = run(clf, [sig(ts=i * 0.3, yaw=25, mar=m) for i, m in enumerate(mars)])
    assert states[-1] == "talking"
    clf.reset()
    assert clf.classify(sig(pitch=30, kps=body(wrist_y=200))).state == "writing"   # hands on desk
    clf.reset()
    assert clf.classify(sig(pitch=30, kps=body(wrist_y=150))).state == "other"     # hands not on desk
    clf.reset()
    assert clf.classify(sig(pitch=30, objects=["book"])).state == "writing"
    clf.reset()
    seated = [sig(ts=t * 0.5, cy=300, on_board=True) for t in range(8)]
    run(clf, seated)
    assert clf.classify(sig(ts=4.0, cy=200, on_board=True)).state == "standing"


def test_body_only_lowers_confidence():
    clf = RuleBasedClassifier(Rules())
    s = sig(on_board=True)
    full = clf.classify(s).confidence
    s2 = sig(tid=2, on_board=True)
    s2.face_visible = False
    assert clf.classify(s2).confidence == pytest.approx(full * 0.5, abs=1e-3)


# ------------------------------------------------------------------ geometry / landmarks
def _project(yaw, pitch, w=1280, h=720):
    base = np.diag([1.0, -1.0, -1.0])
    ry = cv2.Rodrigues(np.array([0, math.radians(-yaw), 0.0]))[0]
    rx = cv2.Rodrigues(np.array([math.radians(pitch), 0, 0.0]))[0]
    r = ry @ rx @ base
    pts, _ = cv2.projectPoints(MODEL_POINTS, cv2.Rodrigues(r)[0], np.array([0, 0, 1500.0]), camera_matrix(w, h),
                               np.zeros(4))
    lmk = np.zeros((68, 2))
    lmk[LMK68_IDX] = pts.reshape(-1, 2)
    return lmk, r


@pytest.mark.parametrize("yaw,pitch", [(0, 0), (30, 0), (-40, 0), (0, 20), (25, -15)])
def test_head_pose_solvepnp_roundtrip(yaw, pitch):
    from src.behaviour.head_pose import angles_from_forward
    lmk, r = _project(yaw, pitch)
    true_yaw, true_pitch = angles_from_forward(r @ np.array([0, 0, 1.0]))
    est = pose_from_landmarks68(lmk, 1280, 720)
    assert est[0] == pytest.approx(true_yaw, abs=1.0) and est[1] == pytest.approx(true_pitch, abs=1.0)


def test_head_pose_sign_convention_matches_kps5():
    lmk, _ = _project(30, 0)
    # nose to the image right of the eye midpoint -> positive yaw in both estimators
    kps = np.array([lmk[36], lmk[45], lmk[30], lmk[48], lmk[54]])
    assert pose_from_landmarks68(lmk, 1280, 720)[0] > 20 and pose_from_kps5(kps)[0] > 10


def test_board_zone_modes():
    z = BoardZone()
    assert z.looking_at((100, 100), 5, 0, 640, 480) and not z.looking_at((100, 100), 60, 0, 640, 480)
    assert z.looking_at((100, 100), None, 0, 640, 480) is None
    poly = [[0.8, 0.0], [1.0, 0.0], [1.0, 0.3], [0.8, 0.3]]          # board at the top-right
    zp = BoardZone(poly)
    assert zp.looking_at((320, 240), 60, -30, 640, 480)               # turned right and up
    assert not zp.looking_at((320, 240), -60, -30, 640, 480)          # turned left


def test_ear_mar():
    lmk = np.zeros((68, 2), np.float32)
    for base, idx in ((0, range(36, 42)), (100, range(42, 48))):
        pts = [(0, 0), (10, -3), (20, -3), (30, 0), (20, 3), (10, 3)]
        for i, (x, y) in zip(idx, pts):
            lmk[i] = (x + base, y + 50)
    assert eye_aspect_ratio(lmk) == pytest.approx(0.2, abs=1e-3)
    mouth = [(0, 0), (10, -6), (20, -6), (30, -6), (40, 0), (30, 6), (20, 6), (10, 6)]
    for i, (x, y) in zip(range(60, 68), mouth):
        lmk[i] = (x, y + 120)
    assert mouth_aspect_ratio(lmk) == pytest.approx(36 / 80, abs=1e-3)


def test_association():
    faces = [np.array([90, 40, 130, 90], np.float32), np.array([390, 40, 430, 90], np.float32)]
    k1 = np.zeros((17, 3), np.float32)
    k1[0] = [110, 65, 0.9]
    k2 = np.zeros((17, 3), np.float32)
    k2[0] = [410, 66, 0.9]
    k2[9] = [450, 200, 0.9]
    persons = [PersonPose(np.array([380, 30, 470, 400], np.float32), k2), PersonPose(np.array([70, 30, 150, 400],
                                                                                              np.float32), k1)]
    m = associate_persons(faces, persons)
    assert m == {0: 1, 1: 0}
    objs = [DetectedObject("phone", np.array([440, 190, 460, 210], np.float32), 0.8),
            DetectedObject("book", np.array([95, 140, 135, 170], np.float32), 0.7)]
    om = associate_objects(objs, faces, [persons[1], persons[0]])
    assert om[1] == ["phone"] and om[0] == ["book"]


def test_zone_storage(cfg):
    assert get_zone(cfg, "room1") is None
    save_zone(cfg, "room1", [[0.1, 0.1], [0.5, 0.1], [0.5, 0.4]])
    assert get_zone(cfg, "room1") == [[0.1, 0.1], [0.5, 0.1], [0.5, 0.4]]
    with pytest.raises(ValueError):
        save_zone(cfg, "room2", [[0.1, 0.1]])
    save_zone(cfg, "room1", None)
    assert get_zone(cfg, "room1") is None


# ------------------------------------------------------------------ results / privacy / excel
class _Stu:
    def __init__(self, roll, name, consent):
        self.roll_no, self.name, self.behaviour_consent = roll, name, consent


def _signals():
    out = []
    for i in range(50):                       # track 1: board for 6 s, then phone for 4 s
        t = i * 0.2
        out.append(sig(tid=1, ts=t, on_board=None, objects=["phone"] if t >= 6 else []))
    for i in range(50):                       # track 2: turned away the whole time
        out.append(sig(tid=2, ts=i * 0.2, yaw=60))
    for i in range(10):                       # track 9: unknown person
        out.append(sig(tid=9, ts=i * 0.2))
    return out


def _results(cfg, corrections=(), class_only=None):
    meta = {"session_id": 1, "duration": 10.0, "step": 0.2, "frame_w": 640, "frame_h": 480,
            "identities": {"1": "1", "2": "2", "9": None}, "camera_id": "default"}
    students = {"1": _Stu("1", "Asha", True), "2": _Stu("2", "Bilal", False), "3": _Stu("3", "Chitra", True)}
    return compute_results(_signals(), meta, cfg, students, {"1": "Present", "2": "Present", "3": "Absent"},
                           list(corrections), class_only)


def test_compute_results_links_students_and_privacy(cfg):
    r = _results(cfg)
    keys = {s["key"] for s in r["subjects"]}
    assert keys == {"1", "3", "Unknown_9"}            # 2 has no behaviour consent -> withheld
    assert r["withheld_no_consent"] == 1
    s1 = next(s for s in r["subjects"] if s["key"] == "1")
    assert s1["name"] == "Asha" and s1["counts"]["phone_uses"] == 1
    assert s1["behaviour_seconds"]["phone"] == pytest.approx(4.0, abs=0.5)
    assert s1["score"] == pytest.approx(60, abs=6)    # 6 s at 100, 4 s at 0
    s3 = next(s for s in r["subjects"] if s["key"] == "3")
    assert s3["score"] is None and s3["attendance"] == "Absent"
    ev = [e for e in r["events"] if e["key"] == "1"]
    assert [e["behaviour"] for e in ev] == ["looking_board", "phone"]
    assert all({"start", "end", "duration", "confidence", "frame_start", "frame_end"} <= set(e) for e in ev)
    # non-consented student still counts in the class aggregate (2 students, unknown excluded)
    assert r["students_analysed"] == 2 and r["class_score"] < s1["score"]
    co = _results(cfg, class_only=True)
    assert co["subjects"] == [] and co["events"] == [] and co["class_score"] == r["class_score"]
    assert privacy_filter(r, True)["subjects"] == []


def test_teacher_correction_updates_score(cfg):
    base = _results(cfg)
    fixed = _results(cfg, [{"subject_key": "1", "behaviour": "phone", "start": 6.0, "end": 10.0,
                            "corrected_to": "writing"}])
    s_before = next(s for s in base["subjects"] if s["key"] == "1")["score"]
    s_after = next(s for s in fixed["subjects"] if s["key"] == "1")
    assert s_after["score"] > s_before + 20 and s_after["counts"]["phone_uses"] == 0
    assert any(e["corrected"] for e in fixed["events"] if e["key"] == "1")


def test_behaviour_excel(cfg, tmp_path):
    r = _results(cfg)
    r.update({"class_name": "10A", "subject": "Maths", "started_at": "2026-11-05T09:00:00"})
    p = write_behaviour_workbook(tmp_path / "b.xlsx", r)
    wb = load_workbook(p)
    assert wb.sheetnames == ["Class Summary", "Student Scores", "Behaviour Log", "Class Timeline",
                             "Student Timelines", "Movement Data"]
    assert "not to be used for punishment" in wb["Class Summary"]["A1"].value.lower()
    assert len(wb["Class Summary"]._charts) == 2 and len(wb["Class Timeline"]._charts) == 1
    ss = wb["Student Scores"]
    assert ss["D1"].value == "Attentiveness Score"
    rules = [rr for rng in ss.conditional_formatting for rr in rng.rules]
    assert rules and rules[0].type == "colorScale"
    assert any(str(rng.sqref).startswith("D2") for rng in ss.conditional_formatting)
    lg = wb["Behaviour Log"]
    assert lg.auto_filter.ref.startswith("A1") and lg.max_row == 1 + len(r["events"])
    assert wb["Student Timelines"].conditional_formatting
    # class-level only workbook
    co = _results(cfg, class_only=True)
    wb2 = load_workbook(write_behaviour_workbook(tmp_path / "c.xlsx", co))
    assert wb2.sheetnames == ["Class Summary", "Class Timeline"]
    # without movement sheet
    wb3 = load_workbook(write_behaviour_workbook(tmp_path / "d.xlsx", r, include_movement=False))
    assert "Movement Data" not in wb3.sheetnames


def test_behaviour_evaluation(tmp_path):
    gt = tmp_path / "gt.csv"
    gt.write_text("roll_no,start,end,behaviour\n1,0,6,looking_board\n1,6,10,phone\n2,0,4,talking\n")
    pr = tmp_path / "pr.csv"
    pr.write_text("key,roll_no,name,behaviour,start,end\n1,1,A,looking_board,0,7\n1,1,A,phone,7,10\n"
                  "2,2,B,looking_away,0,4\n")
    r = evaluate_behaviours(load_intervals(gt), load_intervals(pr))
    assert r.seconds == 14
    assert r.per_class["looking_board"]["recall"] == 1.0
    assert r.per_class["looking_board"]["precision"] == pytest.approx(6 / 7, abs=1e-3)
    assert r.per_class["phone"]["recall"] == pytest.approx(0.75)
    assert r.per_class["talking"]["recall"] == 0.0
    i, j = r.labels.index("talking"), r.labels.index("looking_away")
    assert r.confusion[i][j] == 4


# ------------------------------------------------------------------ integration: sample video
class FakePose(PoseEstimator):
    """Person under the face at x=100; raises a hand after 2 seconds of video (calls)."""

    def __init__(self):
        self.calls = 0

    def estimate(self, frame):
        self.calls += 1
        k = np.zeros((17, 3), np.float32)
        k[0] = [140, 145, 0.9]
        k[5], k[6] = [110, 200, 0.9], [170, 200, 0.9]
        k[9] = [110, 60, 0.9] if self.calls > 10 else [110, 300, 0.9]
        return [PersonPose(np.array([90, 90, 190, 340], np.float32), k, 0.9)]


class FakeObjects:
    def detect(self, frame):
        return []


def _video(path, n=150):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    rng = np.random.default_rng(1)
    for _ in range(n):
        wr.write(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
    wr.release()


def test_sample_video_integration(cfg, enrolled, tmp_path):
    db = enrolled
    db.set_behaviour_consent("1", True, "test")
    video = tmp_path / "class.mp4"
    _video(video)
    analyzer = BehaviourAnalyzer(cfg, db, "default", pose=FakePose(), objects=FakeObjects(), landmarker=None,
                                 load_models=False)
    res = AttendanceProcessor(cfg, db, engine=FakeEngine(), liveness=None).run(
        str(video), "10A", "Maths", behaviour=True, analyzer=analyzer)
    assert res.behaviour is not None
    b = res.behaviour
    files = b["files"]
    for k in ("excel", "csv_students", "csv_events", "csv_timeline", "json", "video_full"):
        assert files[k] and __import__("pathlib").Path(files[k]).exists(), k
    cap = cv2.VideoCapture(files["video_full"])
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 140
    cap.release()
    assert res.video_path is not None and str(res.video_path) == files["video_full"]
    s1 = next(s for s in b["subjects"] if s["key"] == "1")
    assert s1["counts"]["hand_raises"] == 1 and s1["score"] > 90
    assert any(e["behaviour"] == "hand_raised" for e in b["events"])
    assert any(s["key"].startswith("Unknown_") for s in b["subjects"])
    # stored encrypted and identical after rescore from the Parquet cache
    stored = db.get_behaviour_result(res.session_id)
    assert stored["class_score"] == b["class_score"]
    signals, meta = load_cache(db, res.output_dir)
    assert len(signals) == len(analyzer.signals) and meta["identities"]
    again = rescore_session(db, cfg, res.session_id, "test")
    assert again["class_score"] == b["class_score"] and again["events"] == b["events"]
    # teacher marks the hand raise as wrong -> rescored + audited
    ev = next(e for e in b["events"] if e["behaviour"] == "hand_raised")
    db.add_behaviour_correction(res.session_id, "1", "hand_raised", ev["start"], ev["end"], "other", "teacher", "no")
    fixed = rescore_session(db, cfg, res.session_id, "teacher")
    f1 = next(s for s in fixed["subjects"] if s["key"] == "1")
    assert f1["counts"]["hand_raises"] == 0 and f1["score"] < s1["score"]
    actions = [a["action"] for a in db.get_audit_log(res.session_id)]
    assert "behaviour_correction" in actions and "behaviour_rescore" in actions
    # attendance is unaffected by behaviour mode
    assert {r["roll_no"]: r["status"] for r in res.rows}["1"] == "Present"
