"""Attentiveness scoring.

Per-frame score (0-100) for a smoothed state with weight ``w`` in [-1, 1]::

    score = clip(50 + 50 * w, 0, 100)

so +1.0 -> 100, 0 (neutral) -> 50, -1.0 -> 0. States whose weight is ``null``
(e.g. out of frame) are excluded from scoring.

Session score = confidence- and time-weighted mean of per-frame scores::

    S = sum(score_i * dt_i * conf_i) / sum(dt_i * conf_i)

Class score per time bin = mean of the bin scores of all present students.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .timeline import Segment

DEFAULT_WEIGHTS: dict[str, Optional[float]] = {
    "looking_board": 1.0, "writing": 0.9, "hand_raised": 1.0, "looking_away": -0.5, "talking": -0.4,
    "phone": -1.0, "sleeping": -1.0, "standing": 0.0, "other": 0.0, "out_of_frame": None,
}
DEFAULT_LABELS = [(75.0, "Highly Attentive"), (50.0, "Attentive"), (30.0, "Partially Attentive"),
                  (0.0, "Inattentive")]


@dataclass
class ScoringConfig:
    weights: dict[str, Optional[float]] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    labels: list[tuple[float, str]] = field(default_factory=lambda: list(DEFAULT_LABELS))
    use_confidence: bool = True

    @classmethod
    def from_config(cls, scfg: Any) -> "ScoringConfig":
        w = dict(DEFAULT_WEIGHTS)
        w.update({k: (None if v is None else float(v)) for k, v in dict(scfg.weights).items()})
        labels = sorted(((float(v), str(k)) for k, v in dict(scfg.labels).items()), reverse=True)
        return cls(w, labels, bool(scfg.get("use_confidence", True)))

    def frame_score(self, state: str) -> Optional[float]:
        w = self.weights.get(state, 0.0)
        if w is None:
            return None
        return max(0.0, min(100.0, 50.0 + 50.0 * float(w)))

    def label(self, score: Optional[float]) -> str:
        if score is None:
            return "No data"
        for thr, name in self.labels:
            if score >= thr:
                return name
        return self.labels[-1][1]

    def state_scores(self) -> dict[str, Optional[float]]:
        return {k: self.frame_score(k) for k in self.weights}


def weighted_score(segs: Sequence[Segment], sc: ScoringConfig, t0: Optional[float] = None,
                   t1: Optional[float] = None) -> Optional[float]:
    """Time- (and confidence-) weighted mean score of segments, optionally clipped to [t0, t1)."""
    num = den = 0.0
    for sg in segs:
        s = sc.frame_score(sg.state)
        if s is None:
            continue
        a = sg.start if t0 is None else max(sg.start, t0)
        b = sg.end if t1 is None else min(sg.end, t1)
        dt = b - a
        if dt <= 0:
            continue
        w = dt * (max(sg.conf, 1e-3) if sc.use_confidence else 1.0)
        num += s * w
        den += w
    return round(num / den, 2) if den > 0 else None


def mean_confidence(segs: Sequence[Segment]) -> float:
    tot = sum(sg.duration for sg in segs)
    return round(sum(sg.conf * sg.duration for sg in segs) / tot, 3) if tot > 0 else 0.0


def binned_scores(segs: Sequence[Segment], sc: ScoringConfig, bin_sec: float, duration: float
                  ) -> list[Optional[float]]:
    n = max(1, int(-(-duration // bin_sec))) if duration > 0 else 1
    return [weighted_score(segs, sc, i * bin_sec, (i + 1) * bin_sec) for i in range(n)]


def class_timeline(student_bins: Sequence[Sequence[Optional[float]]], bin_sec: float) -> list[dict[str, Any]]:
    """Mean of present students per bin."""
    if not student_bins:
        return []
    n = max(len(b) for b in student_bins)
    out = []
    for i in range(n):
        vals = [b[i] for b in student_bins if i < len(b) and b[i] is not None]
        out.append({"bin": i, "start": i * bin_sec, "end": (i + 1) * bin_sec,
                    "score": round(sum(vals) / len(vals), 2) if vals else None, "students": len(vals)})
    return out


def drop_points(timeline: Sequence[dict[str, Any]], threshold: float, lookback: int = 3) -> list[dict[str, Any]]:
    """Bins where the class score fell >= ``threshold`` below the mean of the previous ``lookback`` bins."""
    out = []
    for i, row in enumerate(timeline):
        if row["score"] is None or i == 0:
            continue
        prev = [r["score"] for r in timeline[max(0, i - lookback):i] if r["score"] is not None]
        if not prev:
            continue
        ref = sum(prev) / len(prev)
        if ref - row["score"] >= threshold:
            out.append({"bin": row["bin"], "start": row["start"], "end": row["end"], "score": row["score"],
                        "previous_mean": round(ref, 2), "drop": round(ref - row["score"], 2)})
    return out
