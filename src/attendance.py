"""Session-level attendance logic: voting, de-duplication and unknown clustering.

This module is model-free (it only consumes per-track observations and match
results), which keeps the critical de-duplication rules unit-testable.

Rules
-----
1. A track votes for a student each time one of its quality crops matches that
   student (cosine >= threshold). A track is *confirmed* as a student when it
   has ``min_votes`` votes, or >= 2 votes spanning ``min_seconds``.
2. ``marked`` is a session-level dict keyed by roll number. A student enters it
   exactly once; later tracks of the same student (re-entry, track breaks) only
   extend first/last-seen and visible time.
3. If two tracks that are visible in the *same frame* are confirmed as the same
   student, that is physically impossible: the conflict is logged as a possible
   spoof/duplicate, the higher-confidence track keeps the identity and the other
   track is blocked from claiming that student again.
"""
from __future__ import annotations

import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of closed intervals."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def total_duration(intervals: list[tuple[float, float]]) -> float:
    return float(sum(e - s for s, e in merge_intervals(intervals)))


@dataclass
class Vote:
    count: int = 0
    score_sum: float = 0.0
    best: float = 0.0
    first_ts: float = float("inf")
    last_ts: float = float("-inf")

    @property
    def mean(self) -> float:
        return self.score_sum / self.count if self.count else 0.0

    @property
    def span(self) -> float:
        return max(0.0, self.last_ts - self.first_ts) if self.count else 0.0


@dataclass(order=True)
class Crop:
    """A face crop kept in a track's best-K heap (ordered by quality)."""
    quality: float
    ts: float = field(compare=False)
    thumb: Any = field(compare=False, default=None)        # BGR image for reports
    embedding: Optional[np.ndarray] = field(compare=False, default=None)


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    segments: list[list[float]] = field(default_factory=list)
    frames: int = 0
    votes: dict[str, Vote] = field(default_factory=lambda: defaultdict(Vote))
    crops: list[Crop] = field(default_factory=list)       # min-heap on quality
    identity: Optional[str] = None
    confidence: float = 0.0
    blocked: set[str] = field(default_factory=set)
    flagged: bool = False
    last_embed_frame: int = -10**9
    embeddings: list[np.ndarray] = field(default_factory=list)
    finished: bool = False

    def observe(self, ts: float, gap: float) -> None:
        if self.segments and ts - self.segments[-1][1] <= gap:
            self.segments[-1][1] = ts
        else:
            self.segments.append([ts, ts])
        self.last_seen = max(self.last_seen, ts)
        self.frames += 1

    def add_crop(self, crop: Crop, max_crops: int) -> bool:
        """Keep the best ``max_crops`` crops. Returns True if the crop was kept."""
        if len(self.crops) < max_crops:
            heapq.heappush(self.crops, crop)
            return True
        if crop.quality > self.crops[0].quality:
            heapq.heapreplace(self.crops, crop)
            return True
        return False

    def best_crop(self) -> Optional[Crop]:
        return max(self.crops, key=lambda c: c.quality) if self.crops else None

    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.embeddings:
            return None
        m = np.mean(np.stack(self.embeddings), axis=0)
        return m / (np.linalg.norm(m) + 1e-12)

    def leading(self) -> tuple[Optional[str], Optional[Vote]]:
        """Student with the most votes (ties broken by score), excluding blocked ones."""
        cands = [(r, v) for r, v in self.votes.items() if r not in self.blocked and v.count > 0]
        if not cands:
            return None, None
        r, v = max(cands, key=lambda rv: (rv[1].count, rv[1].score_sum))
        return r, v


@dataclass
class StudentRecord:
    roll_no: str
    first_seen: float
    last_seen: float
    confidence: float
    track_ids: set[int] = field(default_factory=set)
    snapshot: Any = None
    snapshot_quality: float = -1.0


@dataclass
class Conflict:
    roll_no: str
    ts: float
    kept_track: int
    dropped_track: int
    kept_score: float
    dropped_score: float


class AttendanceSession:
    """Accumulates per-track evidence and produces de-duplicated attendance."""

    def __init__(self, min_votes: int = 5, min_seconds: float = 3.0, max_crops: int = 10,
                 frame_gap: float = 1.0, unknown_min_frames: int = 5,
                 unknown_cluster_threshold: float = 0.5) -> None:
        self.min_votes = int(min_votes)
        self.min_seconds = float(min_seconds)
        self.max_crops = int(max_crops)
        self.frame_gap = float(frame_gap)
        self.unknown_min_frames = int(unknown_min_frames)
        self.unknown_cluster_threshold = float(unknown_cluster_threshold)
        self.tracks: dict[int, TrackState] = {}
        self.marked: dict[str, StudentRecord] = {}
        self.conflicts: list[Conflict] = []
        self.headcounts: list[dict[str, Any]] = []
        self._visible_now: set[int] = set()
        self.last_ts: float = 0.0

    # ------------------------------------------------------------- observation
    def begin_frame(self, ts: float) -> None:
        self._visible_now = set()
        self.last_ts = max(self.last_ts, ts)

    def observe(self, track_id: int, ts: float) -> TrackState:
        t = self.tracks.get(track_id)
        if t is None:
            t = TrackState(track_id, ts, ts)
            self.tracks[track_id] = t
        t.observe(ts, self.frame_gap)
        self._visible_now.add(track_id)
        return t

    def add_vote(self, track_id: int, roll_no: Optional[str], score: float, ts: float,
                 embedding: Optional[np.ndarray] = None) -> None:
        """Record one recognition result for a track (roll_no None = no match)."""
        t = self.tracks[track_id]
        if embedding is not None:
            t.embeddings.append(np.asarray(embedding, dtype=np.float32))
        if roll_no is None or roll_no in t.blocked:
            return
        v = t.votes[roll_no]
        v.count += 1
        v.score_sum += float(score)
        v.best = max(v.best, float(score))
        v.first_ts = min(v.first_ts, ts)
        v.last_ts = max(v.last_ts, ts)

    def is_confirmed(self, vote: Optional[Vote], extra_votes: int = 0) -> bool:
        if vote is None:
            return False
        need = self.min_votes + extra_votes
        return vote.count >= need or (vote.count >= 2 + extra_votes and vote.span >= self.min_seconds)

    def end_frame(self, extra_votes: int = 0) -> list[Conflict]:
        """Re-evaluate identities after all observations of a frame.

        Returns conflicts raised in this frame.
        """
        for t in self.tracks.values():
            if t.finished:
                continue
            roll, vote = t.leading()
            if roll is not None and self.is_confirmed(vote, extra_votes):
                if t.identity is not None and t.identity != roll:
                    self._unlink(t.identity, t.track_id)   # evidence switched to another student
                t.identity, t.confidence = roll, vote.mean  # type: ignore[union-attr]
            elif t.identity is not None and t.identity in t.blocked:
                t.identity, t.confidence = None, 0.0
        return self._resolve_conflicts()

    def _resolve_conflicts(self) -> list[Conflict]:
        new: list[Conflict] = []
        by_roll: dict[str, list[TrackState]] = defaultdict(list)
        for tid in self._visible_now:
            t = self.tracks[tid]
            if t.identity:
                by_roll[t.identity].append(t)
        for roll, ts_list in by_roll.items():
            if len(ts_list) < 2:
                continue
            ts_list.sort(key=lambda t: (t.confidence, t.votes[roll].count), reverse=True)
            keep = ts_list[0]
            for drop in ts_list[1:]:
                c = Conflict(roll, self.last_ts, keep.track_id, drop.track_id, keep.confidence, drop.confidence)
                log.warning("Possible spoof/duplicate: tracks %d and %d both match %s at %.1fs; keeping %d",
                            keep.track_id, drop.track_id, roll, self.last_ts, keep.track_id)
                drop.blocked.add(roll)
                drop.identity, drop.confidence, drop.flagged = None, 0.0, True
                self._unlink(roll, drop.track_id)
                self.conflicts.append(c)
                new.append(c)
        # only now (conflict-free) register confirmed identities
        for tid in self._visible_now:
            t = self.tracks[tid]
            if t.identity:
                self._mark(t)
        return new

    def _mark(self, t: TrackState) -> None:
        """Mark a student present exactly once; later tracks only extend the record."""
        rec = self.marked.get(t.identity)  # type: ignore[arg-type]
        best = t.best_crop()
        if rec is None:
            rec = StudentRecord(t.identity, t.first_seen, t.last_seen, t.confidence)  # type: ignore[arg-type]
            self.marked[t.identity] = rec  # type: ignore[index]
            log.info("Marked PRESENT: %s (track %d, conf %.3f, t=%.1fs)", t.identity, t.track_id,
                     t.confidence, t.first_seen)
        rec.track_ids.add(t.track_id)
        rec.first_seen = min(rec.first_seen, t.first_seen)
        rec.last_seen = max(rec.last_seen, t.last_seen)
        rec.confidence = max(rec.confidence, t.confidence)
        if best is not None and best.quality > rec.snapshot_quality and best.thumb is not None:
            rec.snapshot, rec.snapshot_quality = best.thumb, best.quality

    def _unlink(self, roll: str, track_id: int) -> None:
        """Detach a (dropped) track from a student's record and recompute its times."""
        rec = self.marked.get(roll)
        if rec is None or track_id not in rec.track_ids:
            return
        rec.track_ids.discard(track_id)
        if not rec.track_ids:
            # will be re-created immediately by the kept track in _resolve_conflicts
            del self.marked[roll]
            return
        owners = [self.tracks[i] for i in rec.track_ids]
        rec.first_seen = min(o.first_seen for o in owners)
        rec.last_seen = max(o.last_seen for o in owners)
        rec.confidence = max(o.confidence for o in owners)

    def finish_track(self, track_id: int, extra_votes: int = 0) -> None:
        """Called when the tracker drops a track; final confirmation pass."""
        t = self.tracks.get(track_id)
        if t is None or t.finished:
            return
        roll, vote = t.leading()
        if t.identity is None and roll is not None and self.is_confirmed(vote, extra_votes):
            # confirm late only if no simultaneous track already owns this student
            if not self._overlaps_owner(t, roll):
                t.identity, t.confidence = roll, vote.mean  # type: ignore[union-attr]
                self._mark(t)
        elif t.identity is not None:
            self._mark(t)
        t.finished = True

    def _overlaps_owner(self, t: TrackState, roll: str) -> bool:
        rec = self.marked.get(roll)
        if rec is None:
            return False
        for oid in rec.track_ids:
            o = self.tracks.get(oid)
            if o is None or o.track_id == t.track_id:
                continue
            for s1, e1 in t.segments:
                for s2, e2 in o.segments:
                    if s1 <= e2 and s2 <= e1:
                        self.conflicts.append(Conflict(roll, s1, o.track_id, t.track_id, o.confidence,
                                                       vote_mean(t, roll)))
                        t.blocked.add(roll)
                        t.flagged = True
                        return True
        return False

    def finalize(self, extra_votes: int = 0) -> None:
        for tid in list(self.tracks):
            self.finish_track(tid, extra_votes)

    # ------------------------------------------------------------------ queries
    def visible_time(self, roll_no: str) -> float:
        rec = self.marked.get(roll_no)
        if rec is None:
            return 0.0
        intervals = [(s, e) for tid in rec.track_ids for s, e in self.tracks[tid].segments]
        return total_duration(intervals)

    def present_count(self) -> int:
        return len(self.marked)

    def label_for(self, track_id: int) -> Optional[str]:
        t = self.tracks.get(track_id)
        return t.identity if t else None

    def record_headcount(self, ts: float, window: float) -> dict[str, Any]:
        rolls = sorted({t.identity for t in self.tracks.values()
                        if t.identity and t.last_seen >= ts - window})
        hc = {"ts": ts, "count": len(rolls), "rolls": rolls}
        self.headcounts.append(hc)
        return hc

    def unknown_tracks(self) -> list[TrackState]:
        return [t for t in self.tracks.values()
                if t.identity is None and t.frames >= self.unknown_min_frames
                and t.mean_embedding() is not None and t.crops]

    def cluster_unknowns(self) -> list[tuple[int, TrackState]]:
        """Greedy online clustering of unknown tracks by mean embedding.

        Returns (cluster_id, track) pairs; cluster ids start at 1.
        """
        centroids: list[np.ndarray] = []
        members: list[int] = []
        out: list[tuple[int, TrackState]] = []
        for t in sorted(self.unknown_tracks(), key=lambda x: x.first_seen):
            e = t.mean_embedding()
            assert e is not None
            if centroids:
                sims = np.stack(centroids) @ e
                j = int(np.argmax(sims))
                if sims[j] >= self.unknown_cluster_threshold:
                    n = members[j]
                    c = centroids[j] * n + e
                    centroids[j] = c / (np.linalg.norm(c) + 1e-12)
                    members[j] += 1
                    out.append((j + 1, t))
                    continue
            centroids.append(e)
            members.append(1)
            out.append((len(centroids), t))
        return out


def vote_mean(t: TrackState, roll: str) -> float:
    v = t.votes.get(roll)
    return v.mean if v else 0.0


def build_rows(session: AttendanceSession, students: list[Any], late_after: Optional[float],
               left_early_before: Optional[float]) -> list[dict[str, Any]]:
    """Attendance rows for every enrolled student (Present / Late / Absent)."""
    rows = []
    for s in students:
        rec = session.marked.get(s.roll_no)
        if rec is None:
            rows.append({"roll_no": s.roll_no, "name": s.name, "class_name": s.class_name, "status": "Absent",
                         "first_seen": None, "last_seen": None, "time_visible": 0.0, "confidence": 0.0,
                         "remarks": "", "snapshot": None})
            continue
        status, remarks = "Present", []
        if late_after is not None and rec.first_seen > late_after:
            status = "Late"
        if left_early_before is not None and rec.last_seen < left_early_before:
            remarks.append("Left early")
        if any(c.roll_no == s.roll_no for c in session.conflicts):
            remarks.append("Duplicate/spoof flagged")
        rows.append({"roll_no": s.roll_no, "name": s.name, "class_name": s.class_name, "status": status,
                     "first_seen": round(rec.first_seen, 2), "last_seen": round(rec.last_seen, 2),
                     "time_visible": round(session.visible_time(s.roll_no), 2),
                     "confidence": round(rec.confidence, 4), "remarks": "; ".join(remarks),
                     "snapshot": rec.snapshot})
    return rows
