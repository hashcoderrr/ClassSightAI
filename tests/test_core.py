"""Tests for DB encryption, tracker, recognizer, evaluation, CLI parsing and the full pipeline."""
from __future__ import annotations

import sqlite3

import cv2
import numpy as np
import pytest
from openpyxl import load_workbook

from src.db import Student
from src.detector import nms, tile_grid
from src.evaluation import evaluate, load_labels
from src.pipeline import AttendanceProcessor, assign_unknown
from src.quality import frontal_score
from src.recognizer import FaceGallery
from src.tracker import ByteTracker, iou_matrix
from tests.conftest import FakeEngine, unit


# ------------------------------------------------------------------ database
def test_db_encrypts_names_and_embeddings(cfg, enrolled):
    raw = sqlite3.connect(cfg.path("db_path")).execute("SELECT name_enc, embedding_enc FROM students").fetchall()
    for name_enc, emb_enc in raw:
        assert b"Asha" not in name_enc and b"Bilal" not in name_enc
        assert len(emb_enc) > 2048  # Fernet token, not raw float32
    s = enrolled.get_student("1")
    assert s.name == "Asha Patil" and np.allclose(s.embedding, unit(100), atol=1e-6)


def test_consent_required(db):
    with pytest.raises(PermissionError):
        db.upsert_student(Student("9", "X", "10A", unit(1), consent=False))


def test_users_and_audit(db):
    db.create_user("admin", "s3cretpass")
    assert db.verify_user("admin", "s3cretpass") and not db.verify_user("admin", "wrong")
    assert not db.verify_user("nobody", "s3cretpass")
    with pytest.raises(ValueError):
        db.create_user("x", "short")


def test_history_percentage(db, enrolled):
    for i, st in enumerate(["Present", "Absent", "Late", "Excused"]):
        sid = db.create_session("10A", "Maths", f"v{i}")
        db.save_attendance(sid, [{"roll_no": "1", "status": st}])
        db.update_session(sid, status="completed")
    h = {r["roll_no"]: r for r in db.history("10A")}
    assert h["1"]["sessions"] == 4 and h["1"]["attended"] == 2 and h["1"]["percentage"] == round(200 / 3, 1)


# ------------------------------------------------------------------ tracker / detector utils
def test_bytetrack_keeps_ids_and_handles_gaps():
    tr = ByteTracker(track_buffer=5)
    ids = []
    for f in range(10):
        boxes = np.array([[10 + f, 10, 60 + f, 60], [200 - f, 50, 250 - f, 100]], dtype=np.float32)
        out, _ = tr.update(boxes, np.array([0.9, 0.9]))
        ids.append(sorted(o.track_id for o in out))
    assert ids[0] == ids[-1] == [1, 2]
    # low-score detection still associated (ByteTrack second stage)
    out, _ = tr.update(np.array([[20, 10, 70, 60]], dtype=np.float32), np.array([0.3]))
    assert [o.track_id for o in out] == [1]
    removed_all = []
    for _ in range(8):
        _, removed = tr.update(np.zeros((0, 4)), np.zeros(0))
        removed_all += removed
    assert set(removed_all) == {1, 2}


def test_iou_and_nms_and_tiles():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]
    tiles = tile_grid(1920, 1080, 640, 0.2)
    assert tiles[0] == (0, 0, 640, 640) and max(t[2] for t in tiles) == 1920 and max(t[3] for t in tiles) == 1080


def test_frontal_score():
    frontal = np.array([[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]], dtype=np.float32)
    profile = frontal.copy()
    profile[2, 0] = 75
    assert frontal_score(frontal) > 0.95 > frontal_score(profile)


# ------------------------------------------------------------------ recognizer
def test_gallery_threshold_and_margin():
    g = FaceGallery(["1", "2"], np.stack([unit(1), unit(2)]), threshold=0.45, margin=0.05)
    m = g.match(unit(1))
    assert m.roll_no == "1" and m.score == pytest.approx(1.0, abs=1e-5)
    assert g.match(unit(3)).roll_no is None
    mix = unit(1) + unit(2)
    assert g.match(mix).roll_no is None  # ambiguous: within margin
    empty = FaceGallery([], np.zeros((0, 512)), 0.45)
    assert empty.match(unit(1)).roll_no is None


# ------------------------------------------------------------------ evaluation
def test_evaluation_metrics(tmp_path):
    p = tmp_path / "gt.csv"
    p.write_text("roll_no,present\n1,1\n2,0\n3,1\n4,0\n", encoding="utf-8")
    truth = load_labels(p)
    r = evaluate(truth, {"1": True, "2": True, "3": False, "4": False})
    assert (r.tp, r.fp, r.fn, r.tn) == (1, 1, 1, 1)
    assert r.precision == 0.5 and r.recall == 0.5 and r.accuracy == 0.5
    pred_csv = tmp_path / "pred.csv"
    pred_csv.write_text("roll_no,name,status\n1,A,Present\n2,B,Absent\n3,C,Late\n", encoding="utf-8")
    r2 = evaluate(truth, load_labels(pred_csv))
    assert r2.accuracy == 1.0 and r2.missing_in_predictions == ["4"]


# ------------------------------------------------------------------ end-to-end with a fake engine
def _make_video(path, n=150, fps=25):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    assert wr.isOpened()
    rng = np.random.default_rng(0)
    for _ in range(n):
        wr.write(rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
    wr.release()


def test_pipeline_end_to_end(cfg, enrolled, tmp_path):
    video = tmp_path / "class.mp4"
    _make_video(video)
    proc = AttendanceProcessor(cfg, enrolled, engine=FakeEngine(), liveness=None)
    progress = []
    res = proc.run(str(video), "10A", "Maths", progress_cb=progress.append)
    status = {r["roll_no"]: r["status"] for r in res.rows}
    assert status == {"1": "Present", "2": "Absent", "3": "Absent"}
    assert res.unknown_count == 1 and res.conflicts == 0
    assert res.report_path and res.report_path.exists()
    assert res.video_path and res.video_path.exists() and res.video_path.stat().st_size > 0
    assert progress and progress[-1]["fraction"] > 0.95
    row1 = next(r for r in res.rows if r["roll_no"] == "1")
    assert row1["first_seen"] == 0.0 and row1["time_visible"] > 5 and row1["snapshot_path"]
    wb = load_workbook(res.report_path)
    assert wb["Summary"]["B13"].value == 1  # unknown tracks
    # attendance persisted exactly once per student
    assert len(enrolled.get_attendance(res.session_id)) == 3

    # admin assigns the unknown cluster to student 3 -> report regenerated
    path = assign_unknown(enrolled, cfg, res.session_id, 1, "3", "teacher", update_gallery=True)
    att = {a["roll_no"]: a["status"] for a in enrolled.get_attendance(res.session_id)}
    assert att["3"] == "Present" and path.exists()
    assert enrolled.get_unknowns(res.session_id)[0]["assigned_roll"] == "3"
    assert any(a["action"] == "gallery_update" for a in enrolled.get_audit_log(res.session_id))


def test_pipeline_duplicate_student_in_two_places(cfg, enrolled, tmp_path):
    """Same student's face appears twice in one frame (e.g. a photo): only one is kept, flag raised."""
    video = tmp_path / "dup.mp4"
    _make_video(video, n=100)
    eng = FakeEngine(faces=((100, 100), (180, 200)))  # both x<250 -> both embed as student "1"
    res = AttendanceProcessor(cfg, enrolled, engine=eng, liveness=None).run(str(video), "10A", "Maths")
    assert [r["roll_no"] for r in res.rows if r["status"] != "Absent"] == ["1"]
    assert res.conflicts >= 1
    assert "Duplicate/spoof flagged" in next(r for r in res.rows if r["roll_no"] == "1")["remarks"]


def test_pipeline_rejects_missing_or_corrupt_video(cfg, enrolled, tmp_path):
    proc = AttendanceProcessor(cfg, enrolled, engine=FakeEngine(), liveness=None)
    with pytest.raises(FileNotFoundError):
        proc.run(str(tmp_path / "nope.mp4"), "10A", "Maths")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(ValueError):
        proc.run(str(bad), "10A", "Maths")
    with pytest.raises(ValueError):
        proc.run(str(bad), "NOCLASS", "Maths")


def test_cli_short_form_parsing(monkeypatch, cfg):
    import main
    monkeypatch.setenv("SMARTATTEND_CONFIG", str(cfg.base_dir / "config.yaml"))
    captured = {}

    def fake_dispatch(args, cfg, db):
        captured.update(vars(args))
        return 0
    monkeypatch.setattr(main, "dispatch", fake_dispatch)
    assert main.main(["--video", "class.mp4", "--class", "10A", "--subject", "Maths"]) == 0
    assert captured["cmd"] == "process" and captured["video"] == "class.mp4" and captured["class_name"] == "10A"


def test_default_admin_created_and_restored(cfg, db):
    assert db.ensure_default_admin(cfg) == "admin"
    assert db.verify_user("admin", "admin123") and db.get_role("admin") == "admin"
    db.create_user("admin", "changedpass", "staff")      # someone changed it
    db.ensure_default_admin(cfg)
    assert db.verify_user("admin", "admin123") and db.get_role("admin") == "admin"
