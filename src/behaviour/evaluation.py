"""Per-behaviour evaluation against manually labelled intervals.

Ground-truth CSV (header required)::

    roll_no,start,end,behaviour
    1,0,12.5,looking_board
    1,12.5,20,writing

``start``/``end`` are seconds (or HH:MM:SS). Predictions use the same format -
SmartAttend's ``Behaviour_..._events.csv`` export works directly (``key`` or
``roll_no`` column). Both are rasterised to 1-second steps; only seconds that
are labelled in the ground truth are scored.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .signals import BEHAVIOURS, DISPLAY

ALIASES = {v.lower(): k for k, v in DISPLAY.items()}
ALIASES.update({k: k for k in BEHAVIOURS})
ALIASES.update({"board": "looking_board", "looking at board": "looking_board", "writing": "writing",
                "away": "looking_away", "sleep": "sleeping", "hand": "hand_raised", "phone use": "phone",
                "using phone": "phone", "standing": "standing", "absent": "out_of_frame"})


def parse_time(v: Any) -> float:
    s = str(v).strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(s)


def norm_behaviour(v: Any) -> str:
    key = str(v).strip().lower()
    if key not in ALIASES:
        raise ValueError(f"Unknown behaviour label {v!r}; use one of {BEHAVIOURS}")
    return ALIASES[key]


def load_intervals(path: str | Path) -> dict[str, list[tuple[float, float, str]]]:
    out: dict[str, list[tuple[float, float, str]]] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        rc = cols.get("roll_no") or cols.get("key") or cols.get("roll")
        if rc is None or not {"start", "end", "behaviour"} <= set(cols):
            raise ValueError(f"{path}: needs columns roll_no (or key), start, end, behaviour")
        for row in reader:
            key = str(row[rc]).strip()
            if not key:
                continue
            out.setdefault(key, []).append((parse_time(row[cols["start"]]), parse_time(row[cols["end"]]),
                                            norm_behaviour(row[cols["behaviour"]])))
    return out


def rasterise(intervals: list[tuple[float, float, str]], step: float = 1.0) -> dict[int, str]:
    sec: dict[int, str] = {}
    for start, end, b in sorted(intervals):
        t = int(round(start / step))
        while t * step < end - 1e-9:
            sec[t] = b
            t += 1
    return sec


@dataclass
class BehaviourEval:
    labels: list[str]
    confusion: list[list[int]]          # rows = truth, cols = predicted
    per_class: dict[str, dict[str, float]]
    accuracy: float
    macro_f1: float
    seconds: int


def evaluate_behaviours(truth: dict[str, list[tuple[float, float, str]]],
                        pred: dict[str, list[tuple[float, float, str]]], step: float = 1.0) -> BehaviourEval:
    pairs: list[tuple[str, str]] = []
    for key, iv in truth.items():
        t_sec = rasterise(iv, step)
        p_sec = rasterise(pred.get(key, []), step)
        for t, b in t_sec.items():
            pairs.append((b, p_sec.get(t, "out_of_frame")))
    labels = [b for b in BEHAVIOURS if any(b in p for p in pairs)]
    idx = {b: i for i, b in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in pairs:
        cm[idx[t]][idx[p]] += 1
    per: dict[str, dict[str, float]] = {}
    for b, i in idx.items():
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        fn = sum(cm[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[b] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": tp + fn}
    total = len(pairs)
    acc = sum(cm[i][i] for i in range(len(labels))) / total if total else 0.0
    present = [v["f1"] for v in per.values() if v["support"] > 0]
    return BehaviourEval(labels, cm, per, round(acc, 4), round(sum(present) / len(present), 4) if present else 0.0,
                         total)
