"""Per-frame behaviour classification.

:class:`BehaviourClassifier` is the interface; :class:`RuleBasedClassifier` is
the default. A trained model (e.g. an LSTM / Transformer over keypoint
sequences, or SlowFast / VideoMAE clips) can replace it by implementing
``classify`` and ``reset`` and being registered in :func:`build_classifier`.

Exactly one *primary* state is produced per frame, chosen by a configurable
priority order; temporal smoothing happens later in :mod:`timeline`.
"""
from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Optional

import numpy as np

from .signals import (HAND_RAISED, L_SHO, L_WRI, LOOKING_AWAY, LOOKING_BOARD, NOSE, OTHER, PHONE, R_SHO, R_WRI,
                      SLEEPING, STANDING, TALKING, WRITING, Classification, FrameSignals)


class BehaviourClassifier(ABC):
    """Maps a stream of :class:`FrameSignals` (per track, in time order) to states."""

    @abstractmethod
    def classify(self, s: FrameSignals) -> Classification:
        ...

    @abstractmethod
    def reset(self, track_id: Optional[int] = None) -> None:
        ...

    def classify_sequence(self, signals: list[FrameSignals]) -> list[Classification]:
        """Classify one track's signals (sorted by time) from a clean state."""
        self.reset(signals[0].track_id if signals else None)
        return [self.classify(s) for s in signals]


@dataclass
class Rules:
    kp_conf: float = 0.3
    away_yaw: float = 35.0
    away_min_sec: float = 2.0
    down_pitch: float = 20.0
    sleep_pitch: float = 35.0
    sleep_pitch_no_eyes: float = 50.0
    ear_closed: float = 0.18
    sleep_min_sec: float = 3.0
    talk_window_sec: float = 2.0
    talk_mar_std: float = 0.04
    talk_yaw: float = 15.0
    talk_min_samples: int = 4
    hand_raise_margin: float = 0.0
    desk_hand_offset: float = 0.5
    stand_rise_shoulder: float = 0.6
    stand_rise_face: float = 1.0
    move_speed: float = 1.5
    baseline_sec: float = 20.0
    body_only_conf: float = 0.5
    kps5_conf: float = 0.8
    no_pose_writing_conf: float = 0.5
    priority: list[str] = field(default_factory=lambda: [PHONE, SLEEPING, HAND_RAISED, STANDING, TALKING,
                                                         LOOKING_AWAY, WRITING, LOOKING_BOARD, OTHER])

    @classmethod
    def from_config(cls, rcfg: Any) -> "Rules":
        r = cls()
        for k, v in dict(rcfg).items():
            if hasattr(r, k):
                setattr(r, k, list(v) if k == "priority" else type(getattr(r, k))(v))
        return r


@dataclass
class _TrackState:
    hist: Deque[tuple[float, Optional[float], float, float, float, Optional[float]]] = field(default_factory=deque)
    # (ts, mar, face_cx, face_cy, face_h, shoulder_y)
    away_since: Optional[float] = None
    sleep_since: Optional[float] = None
    last_ts: Optional[float] = None


class RuleBasedClassifier(BehaviourClassifier):
    def __init__(self, rules: Optional[Rules] = None) -> None:
        self.r = rules or Rules()
        self._tracks: dict[int, _TrackState] = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    # ------------------------------------------------------------ helpers
    def _shoulders(self, s: FrameSignals) -> Optional[tuple[float, float]]:
        ls, rs = s.kp(L_SHO, self.r.kp_conf), s.kp(R_SHO, self.r.kp_conf)
        if ls is None or rs is None:
            return None
        return float((ls[1] + rs[1]) / 2), float(abs(ls[0] - rs[0]) + 1e-6)

    def _hand_raised(self, s: FrameSignals) -> bool:
        sh = self._shoulders(s)
        nose = s.kp(NOSE, self.r.kp_conf)
        for w in (L_WRI, R_WRI):
            wp = s.kp(w, self.r.kp_conf)
            if wp is None:
                continue
            if nose is not None and wp[1] < nose[1] - self.r.hand_raise_margin * s.face_h:
                return True
            if nose is None and sh is not None and wp[1] < sh[0] - (0.5 + self.r.hand_raise_margin) * sh[1]:
                return True
        return False

    def _hands_on_desk(self, s: FrameSignals) -> Optional[bool]:
        sh = self._shoulders(s)
        if sh is None:
            return None
        wrists = [s.kp(w, self.r.kp_conf) for w in (L_WRI, R_WRI)]
        wrists = [w for w in wrists if w is not None]
        if not wrists:
            return None
        return any(w[1] > sh[0] + self.r.desk_hand_offset * sh[1] for w in wrists)

    # ------------------------------------------------------------ main rule set
    def classify(self, s: FrameSignals) -> Classification:
        r = self.r
        st = self._tracks.setdefault(s.track_id, _TrackState())
        sh = self._shoulders(s)
        cx, cy = s.center
        st.hist.append((s.ts, s.mar, cx, cy, s.face_h, sh[0] if sh else None))
        while st.hist and s.ts - st.hist[0][0] > max(r.baseline_sec, r.talk_window_sec):
            st.hist.popleft()

        conf = float(np.clip(s.det_score, 0.3, 1.0))
        if not s.face_visible:
            conf *= r.body_only_conf
        elif s.pose_source == "kps5":
            conf *= r.kps5_conf
        yaw, pitch = s.yaw, s.pitch
        cands: dict[str, float] = {}

        if PHONE_OBJ in s.objects:
            cands[PHONE] = conf
        if s.keypoints is not None and self._hand_raised(s):
            cands[HAND_RAISED] = conf

        # sustained head-down + eyes closed
        sleepy = False
        if pitch is not None:
            if s.ear is not None:
                sleepy = pitch >= r.sleep_pitch and s.ear < r.ear_closed
            else:
                sleepy = pitch >= r.sleep_pitch_no_eyes
        if sleepy:
            st.sleep_since = s.ts if st.sleep_since is None else st.sleep_since
            if s.ts - st.sleep_since >= r.sleep_min_sec:
                cands[SLEEPING] = conf * (1.0 if s.ear is not None else 0.7)
        else:
            st.sleep_since = None

        # standing / moving: rise vs seated baseline, or fast displacement
        older = [h for h in st.hist if s.ts - h[0] >= 1.0]
        shoulder_hist = [h[5] for h in older if h[5] is not None]
        face_hist = [h[3] for h in older]
        rose = False
        if sh is not None and len(shoulder_hist) >= 3:
            rose = statistics.median(shoulder_hist) - sh[0] > r.stand_rise_shoulder * sh[1]
        elif len(face_hist) >= 3:
            rose = statistics.median(face_hist) - cy > r.stand_rise_face * s.face_h
        recent = [h for h in st.hist if 0.5 <= s.ts - h[0] <= 1.5]
        moving = False
        if recent:
            h0 = recent[0]
            moving = float(np.hypot(cx - h0[2], cy - h0[3])) / max(1.0, s.face_h) / max(1e-3, s.ts - h0[0]) > r.move_speed
        if rose or moving:
            cands[STANDING] = conf * 0.9

        # talking: mouth movement + head turned towards a neighbour
        mars = [h[1] for h in st.hist if h[1] is not None and s.ts - h[0] <= r.talk_window_sec]
        if (yaw is not None and abs(yaw) >= r.talk_yaw and len(mars) >= r.talk_min_samples
                and float(np.std(mars)) >= r.talk_mar_std):
            cands[TALKING] = conf * 0.8

        # looking away: |yaw| beyond limit for a sustained time
        if yaw is not None and abs(yaw) > r.away_yaw:
            st.away_since = s.ts if st.away_since is None else st.away_since
            if s.ts - st.away_since >= r.away_min_sec:
                cands[LOOKING_AWAY] = conf
        else:
            st.away_since = None

        # looking down / writing
        if pitch is not None and pitch >= r.down_pitch:
            desk = _WRITING_OBJS & set(s.objects)
            hands = self._hands_on_desk(s)
            if desk or hands:
                cands[WRITING] = conf
            elif hands is None:  # no body keypoints: can't verify the hands
                cands[WRITING] = conf * r.no_pose_writing_conf

        if s.on_board:
            cands[LOOKING_BOARD] = conf

        st.last_ts = s.ts
        for state in r.priority:
            if state in cands:
                return Classification(state, round(float(cands[state]), 3))
        return Classification(OTHER, round(conf * 0.6, 3))


PHONE_OBJ = "phone"
_WRITING_OBJS = {"book", "notebook", "pen", "laptop"}


def build_classifier(cfg: Any) -> BehaviourClassifier:
    kind = str(cfg.behaviour.get("classifier", "rules"))
    if kind == "rules":
        return RuleBasedClassifier(Rules.from_config(cfg.behaviour.rules))
    raise ValueError(f"Unknown behaviour classifier '{kind}'. Implement BehaviourClassifier and register it here.")
