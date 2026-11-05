"""Temporal processing: smoothing, gap filling (out-of-frame), events and bins."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .signals import OUT_OF_FRAME


@dataclass
class Sample:
    """One processed frame of one subject (student or unknown) after classification."""
    ts: float
    frame_idx: int
    state: str
    conf: float
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    x: float = 0.0            # face centre, pixels
    y: float = 0.0
    face_h: float = 1.0
    raw_state: str = ""


@dataclass
class Segment:
    state: str
    start: float
    end: float
    conf: float
    frame_start: int
    frame_end: int
    corrected: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def smooth_states(ts: Sequence[float], states: Sequence[str], window_sec: float) -> list[str]:
    """Centred sliding-window majority vote over ``window_sec`` seconds.

    Ties keep the frame's own state when it is among the winners, otherwise the
    winner that appears first in the window. O(n) two-pointer implementation.
    """
    n = len(states)
    if n == 0 or window_sec <= 0:
        return list(states)
    half = window_sec / 2.0
    out: list[str] = []
    counts: Counter[str] = Counter()
    lo = hi = 0
    for i in range(n):
        while hi < n and ts[hi] <= ts[i] + half:
            counts[states[hi]] += 1
            hi += 1
        while ts[lo] < ts[i] - half:
            counts[states[lo]] -= 1
            lo += 1
        best = max(counts.values())
        winners = {s for s, c in counts.items() if c == best}
        if states[i] in winners:
            out.append(states[i])
        else:
            out.append(next(states[j] for j in range(lo, hi) if states[j] in winners))
    return out


def build_segments(samples: Sequence[Sample], step: float, out_gap: float, video_end: float) -> list[Segment]:
    """Turn samples into contiguous per-sample segments.

    A sample holds its state until the next sample (state interpolation between
    processed frames). Gaps longer than ``out_gap`` seconds - including after the
    last sample until ``video_end`` - become ``out_of_frame``.
    """
    segs: list[Segment] = []
    for i, s in enumerate(samples):
        nxt = samples[i + 1].ts if i + 1 < len(samples) else None
        if nxt is not None and nxt - s.ts <= out_gap:
            segs.append(Segment(s.state, s.ts, nxt, s.conf, s.frame_idx, samples[i + 1].frame_idx))
            continue
        end = s.ts + step
        segs.append(Segment(s.state, s.ts, end, s.conf, s.frame_idx, s.frame_idx))
        gap_end = nxt if nxt is not None else max(video_end, end)
        if gap_end - end > out_gap or (nxt is not None and gap_end > end):
            segs.append(Segment(OUT_OF_FRAME, end, gap_end, 1.0, s.frame_idx,
                                samples[i + 1].frame_idx if nxt is not None else s.frame_idx))
    return segs


def merge_segments(segs: Sequence[Segment], eps: float = 1e-6) -> list[Segment]:
    """Merge consecutive segments of the same state into events (confidence = duration-weighted mean)."""
    out: list[Segment] = []
    for sg in segs:
        if out and out[-1].state == sg.state and abs(out[-1].end - sg.start) <= eps \
                and out[-1].corrected == sg.corrected:
            last = out[-1]
            total = last.duration + sg.duration
            if total > 0:
                last.conf = (last.conf * last.duration + sg.conf * sg.duration) / total
            last.end = sg.end
            last.frame_end = sg.frame_end
        else:
            out.append(Segment(sg.state, sg.start, sg.end, sg.conf, sg.frame_start, sg.frame_end, sg.corrected))
    return out


def apply_corrections(samples: list[Sample], corrections: Iterable[dict[str, Any]]) -> int:
    """Teacher corrections: replace ``behaviour`` by ``corrected_to`` within [start, end].

    Returns how many samples changed. The confidence of corrected samples becomes 1.0
    (a human verified them).
    """
    changed = 0
    for c in corrections:
        for s in samples:
            if c["start"] - 1e-6 <= s.ts <= c["end"] + 1e-6 and (not c.get("behaviour") or s.state == c["behaviour"]):
                s.state = c["corrected_to"]
                s.conf = 1.0
                s.raw_state = s.raw_state or "corrected"
                changed += 1
    return changed


def behaviour_seconds(segs: Iterable[Segment]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sg in segs:
        out[sg.state] = out.get(sg.state, 0.0) + sg.duration
    return out


def count_head_turns(samples: Sequence[Sample], yaw_thr: float, hysteresis: float = 10.0) -> int:
    """Times |yaw| rises above ``yaw_thr`` (re-armed once it drops below thr - hysteresis)."""
    turns, armed = 0, True
    for s in samples:
        if s.yaw is None:
            continue
        a = abs(s.yaw)
        if armed and a > yaw_thr:
            turns += 1
            armed = False
        elif not armed and a < yaw_thr - hysteresis:
            armed = True
    return turns


def displacement(samples: Sequence[Sample]) -> tuple[float, float]:
    """Total path length of the face centre in pixels and in face heights."""
    px = norm = 0.0
    for a, b in zip(samples, samples[1:]):
        d = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5
        px += d
        norm += d / max(1.0, (a.face_h + b.face_h) / 2)
    return px, norm


def per_second_rows(samples: Sequence[Sample], segs: Sequence[Segment], scores: dict[str, Optional[float]],
                    frame_w: int, frame_h: int) -> list[dict[str, Any]]:
    """Per-second resampling (last sample at or before each second) for the Movement Data sheet."""
    rows: list[dict[str, Any]] = []
    if not samples:
        return rows
    j = 0
    t = int(samples[0].ts)
    end = samples[-1].ts
    seg_i = 0
    while t <= end:
        while j + 1 < len(samples) and samples[j + 1].ts <= t:
            j += 1
        while seg_i + 1 < len(segs) and segs[seg_i].end <= t:
            seg_i += 1
        s = samples[j]
        state = segs[seg_i].state if segs and segs[seg_i].start <= t < segs[seg_i].end + 1e-6 else s.state
        rows.append({"t": t, "yaw": None if s.yaw is None else round(s.yaw, 1),
                     "pitch": None if s.pitch is None else round(s.pitch, 1),
                     "x": round(s.x / max(1, frame_w), 4), "y": round(s.y / max(1, frame_h), 4),
                     "state": state, "score": scores.get(state)})
        t += 1
    return rows
