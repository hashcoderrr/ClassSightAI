"""Unit tests for voting and de-duplication rules (src/attendance.py)."""
from __future__ import annotations

import numpy as np

from src.attendance import AttendanceSession, Crop, build_rows, merge_intervals, total_duration
from src.db import Student


def feed(sess: AttendanceSession, track_id: int, roll, ts_list, score=0.7):
    for ts in ts_list:
        sess.begin_frame(ts)
        sess.observe(track_id, ts)
        sess.add_vote(track_id, roll, score, ts)
        sess.end_frame()


def students(*rolls):
    return [Student(r, f"S{r}", "10A", np.zeros(512, dtype=np.float32), consent=True) for r in rolls]


def test_not_marked_before_min_votes():
    s = AttendanceSession(min_votes=5, min_seconds=100)
    feed(s, 1, "A", [0.0, 0.2, 0.4, 0.6])
    assert "A" not in s.marked
    feed(s, 1, "A", [0.8])
    assert "A" in s.marked


def test_single_frame_false_positive_ignored():
    s = AttendanceSession(min_votes=5, min_seconds=3)
    feed(s, 1, "A", [0.0])
    s.finalize()
    assert s.marked == {}


def test_seconds_rule_confirms_with_two_votes():
    s = AttendanceSession(min_votes=5, min_seconds=3)
    feed(s, 1, "A", [0.0, 3.5])
    assert "A" in s.marked


def test_marked_once_across_reentry_tracks():
    s = AttendanceSession(min_votes=3, min_seconds=100, frame_gap=1.0)
    feed(s, 1, "A", [0, 1, 2])
    s.finish_track(1)                       # student leaves
    feed(s, 2, "A", [50, 51, 52, 53])       # re-enters with a new track ID
    s.finalize()
    assert list(s.marked) == ["A"]
    rec = s.marked["A"]
    assert rec.track_ids == {1, 2}
    assert rec.first_seen == 0 and rec.last_seen == 53
    assert abs(s.visible_time("A") - (2 + 3)) < 1e-6
    assert s.conflicts == []


def test_simultaneous_tracks_same_student_flagged_keep_higher_confidence():
    s = AttendanceSession(min_votes=3, min_seconds=100)
    for ts in [0, 1, 2, 3]:
        s.begin_frame(ts)
        s.observe(1, ts)
        s.observe(2, ts)
        s.add_vote(1, "A", 0.80, ts)
        s.add_vote(2, "A", 0.55, ts)        # e.g. a photo of A held up by someone
        s.end_frame()
    s.finalize()
    assert len(s.marked) == 1 and s.marked["A"].track_ids == {1}
    assert s.tracks[2].identity is None and s.tracks[2].flagged
    assert "A" in s.tracks[2].blocked
    assert len(s.conflicts) >= 1
    c = s.conflicts[0]
    assert c.kept_track == 1 and c.dropped_track == 2


def test_higher_confidence_later_track_takes_over_without_double_marking():
    s = AttendanceSession(min_votes=2, min_seconds=100)
    feed(s, 1, "A", [0, 1], score=0.5)         # track 1 confirmed first, lower confidence
    for ts in [2, 3]:
        s.begin_frame(ts)
        s.observe(1, ts)
        s.observe(2, ts)
        s.add_vote(1, "A", 0.5, ts)
        s.add_vote(2, "A", 0.9, ts)
        s.end_frame()
    assert len(s.marked) == 1
    assert s.marked["A"].track_ids == {2}
    assert s.tracks[1].flagged


def test_blocked_track_cannot_reclaim_after_conflict():
    s = AttendanceSession(min_votes=2, min_seconds=100)
    for ts in [0, 1]:
        s.begin_frame(ts)
        s.observe(1, ts)
        s.observe(2, ts)
        s.add_vote(1, "A", 0.9, ts)
        s.add_vote(2, "A", 0.6, ts)
        s.end_frame()
    feed(s, 2, "A", [2, 3, 4, 5], score=0.95)  # dropped track keeps matching A
    assert s.tracks[2].identity is None
    assert s.marked["A"].track_ids == {1}


def test_late_confirmation_on_finish_respects_overlap():
    s = AttendanceSession(min_votes=3, min_seconds=100)
    feed(s, 1, "A", [0, 1, 2, 3])
    # track 2 overlapped in time but only reaches the vote count at finish
    s.observe(2, 1.5)
    s.add_vote(2, "A", 0.6, 1.5)
    s.observe(2, 2.5)
    s.add_vote(2, "A", 0.6, 2.5)
    s.observe(2, 2.9)
    s.add_vote(2, "A", 0.6, 2.9)
    s.finish_track(2)
    assert s.marked["A"].track_ids == {1}
    assert s.tracks[2].flagged


def test_two_different_students():
    s = AttendanceSession(min_votes=2, min_seconds=100)
    for ts in [0, 1, 2]:
        s.begin_frame(ts)
        s.observe(1, ts)
        s.observe(2, ts)
        s.add_vote(1, "A", 0.8, ts)
        s.add_vote(2, "B", 0.7, ts)
        s.end_frame()
    assert set(s.marked) == {"A", "B"}
    assert s.conflicts == []


def test_votes_for_none_do_not_count():
    s = AttendanceSession(min_votes=2, min_seconds=1)
    feed(s, 1, None, [0, 1, 2, 3])
    s.finalize()
    assert s.marked == {}


def test_majority_identity_wins():
    s = AttendanceSession(min_votes=3, min_seconds=100)
    feed(s, 1, "A", [0, 1, 2, 3])
    feed(s, 1, "B", [4])
    assert s.tracks[1].identity == "A"


def test_crop_heap_keeps_best():
    s = AttendanceSession(max_crops=3)
    t = s.observe(1, 0)
    for q in [0.1, 0.9, 0.5, 0.3, 0.7]:
        t.add_crop(Crop(q, 0.0), 3)
    assert sorted(c.quality for c in t.crops) == [0.5, 0.7, 0.9]
    assert t.best_crop().quality == 0.9


def test_build_rows_present_late_absent_left_early():
    s = AttendanceSession(min_votes=2, min_seconds=100)
    feed(s, 1, "1", [10, 11])          # on time, leaves early
    feed(s, 2, "2", [400, 401, 3000])  # late
    s.finalize()
    rows = {r["roll_no"]: r for r in build_rows(s, students("1", "2", "3"), late_after=300,
                                                 left_early_before=2400)}
    assert rows["1"]["status"] == "Present" and "Left early" in rows["1"]["remarks"]
    assert rows["2"]["status"] == "Late" and rows["2"]["remarks"] == ""
    assert rows["3"]["status"] == "Absent" and rows["3"]["first_seen"] is None


def test_interval_union():
    assert merge_intervals([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert total_duration([(0, 2), (1, 3), (5, 6)]) == 4


def test_unknown_clustering_groups_same_person():
    rng = np.random.default_rng(0)
    base = rng.normal(size=512)
    other = rng.normal(size=512)
    s = AttendanceSession(unknown_min_frames=2, unknown_cluster_threshold=0.5)
    for tid, vec in [(1, base), (2, base + rng.normal(scale=0.05, size=512)), (3, other)]:
        for ts in [tid * 10, tid * 10 + 1]:
            t = s.observe(tid, ts)
            t.add_crop(Crop(0.5, ts, np.zeros((4, 4, 3), np.uint8)), 3)
            s.add_vote(tid, None, 0.1, ts, (vec / np.linalg.norm(vec)).astype(np.float32))
    clusters = {t.track_id: cid for cid, t in s.cluster_unknowns()}
    assert clusters[1] == clusters[2] != clusters[3]
